"""Multi-temporal NISAR GCOV products: speckle-filtered frames, season mean, change composite, water frequency.

Per track, all dual-pol dates share one fixed grid, so we can stack them:
  * Quegan multi-temporal speckle filter (Quegan & Yu 2001): J_k = E_k * mean_i(I_i / E_i), E = local 5x5 mean.
    Each date keeps its own changes but speckle drops by ~sqrt(N).
  * Bilinear resampling to the page grids (no more nearest-neighbour blocks).
Outputs (site/img):
  nisarmt_<view>_<date>_t<track>.jpg      filtered RGB per date (R=HH, G=HV, B=HH, anchored on lake cores)
  nisarmt_<view>_mean_t<track>.jpg        season-mean RGB (cleanest image)
  nisarmt_<view>_change_t<track>.jpg      HV change composite: R=June, G=August, B=October
  nisarmt_<view>_waterfreq.jpg            % of passes (both tracks) a pixel is open water, over a grey base
  nisarmt_<view>_newwater.jpg             water early (Jun–Jul) vs late (Sep–Oct): stable / new / lost
and site/data/nisar_mt.json (water areas from the filtered stack).

Usage: uv run --with numpy --with scipy --with pillow --with pyproj python -I gcov_mt.py <npz_dir> <site_dir> [<gunw_dir>]
"""
import glob, json, math, os, re, sys

import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy.ndimage import map_coordinates, uniform_filter, binary_dilation, binary_erosion

src, site = sys.argv[1:3]
src_gunw = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(src), "nisar_gunw")
COH_WATER = 0.3  # InSAR coherence below this = water candidate (pit only)
IMG, DATA = os.path.join(site, "img"), os.path.join(site, "data")
VIEWS = {"region": "layers.json", "lakes": "lakes_layers.json", "pit": "pit_layers.json"}
W_HH, W_HV = 16.0, 12.0  # dB above the lake-core anchor that maps to white


def view_grid(view):
    m = json.load(open(os.path.join(IMG, VIEWS[view])))
    L0, A0, L1, A1 = m["bbox"]; W, H = m["size"]
    lon = L0 + (np.arange(W) + 0.5) * (L1 - L0) / W
    lat = A1 - (np.arange(H) + 0.5) * (A1 - A0) / H
    LON, LAT = np.meshgrid(lon, lat)
    px_ha = ((L1 - L0) * 111320 * math.cos(math.radians((A0 + A1) / 2)) / W) * ((A1 - A0) * 110574 / H) / 1e4
    return m, LON, LAT, px_ha


def occ(prefix):
    return np.asarray(Image.open(os.path.join(IMG, f"{prefix}jrc_occurrence.png")).convert("RGBA"))[..., 3]


grids = {v: view_grid(v) for v in VIEWS}
o = occ("lakes_")
lake_mask = binary_dilation(o > 0, iterations=2)
core = binary_erosion(o >= 200, iterations=3)
land = ~binary_dilation(o > 0, iterations=15)
# pit footprint: excavation (bright sand) or JRC water, same as gcov_views.py
PIT_BOX = (18.12, 52.40, 18.22, 52.445)
_, PLON, PLAT, _ = grids["pit"]
sand = np.zeros(PLON.shape, bool)
for yr in ("2016", "2024"):
    sand |= np.asarray(Image.open(os.path.join(IMG, f"pit_s2_{yr}.jpg")).convert("RGB")).astype(float).mean(2) > 140
pit_fp = binary_dilation(sand | (occ("pit_") > 0), iterations=4)
pit_fp &= (PLON >= PIT_BOX[0]) & (PLON <= PIT_BOX[2]) & (PLAT >= PIT_BOX[1]) & (PLAT <= PIT_BOX[3])

# per-pass thresholds and quality from the single-date run
single = {(r["date"], r["track"]): r for r in json.load(open(os.path.join(DATA, "nisar_water.json")))["series"]}

files = {}
for p in sorted(glob.glob(os.path.join(src, "*.npz"))):
    m = re.search(r"_(\d{3})_([AD])_\d{3}_.*?_(\d{8})T", os.path.basename(p))
    files.setdefault(m.group(1), []).append((f"{m.group(3)[:4]}-{m.group(3)[4:6]}-{m.group(3)[6:]}", p))

stretch = lambda a, lo, hi: np.clip((a - lo) / (hi - lo), 0, 1)
to_db = lambda a: 10 * np.log10(np.maximum(a, 1e-6))
save = lambda rgb, name: Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(os.path.join(IMG, name), quality=88)

water_votes = {v: [] for v in ("lakes", "pit")}  # list of (date, track, bool map)
out_series = []

for track, items in sorted(files.items()):
    items.sort()
    dates = [d for d, _ in items]
    z0 = np.load(items[0][1])
    x, y, epsg = z0["x"], z0["y"], int(z0["epsg"])
    stack = {"hh": [], "hv": []}
    for _, p in items:
        z = np.load(p)
        assert z["hh_db"].shape == z0["hh_db"].shape, "dates of one track must share a grid"
        for k in ("hh", "hv"):
            a = 10 ** (z[f"{k}_db"].astype(np.float32) / 10)
            stack[k].append(np.where(np.isfinite(a), a, np.nan))
    filt = {}
    for k in ("hh", "hv"):
        I = np.stack(stack[k])
        valid = np.isfinite(I)
        I0 = np.where(valid, I, 0)
        E = np.stack([uniform_filter(a, 5) / np.maximum(uniform_filter(v.astype(np.float32), 5), 1e-3)
                      for a, v in zip(I0, valid)])
        ratio = np.nanmean(np.where(valid, I / np.maximum(E, 1e-9), np.nan), axis=0)
        filt[k] = np.where(valid, E * ratio[None], np.nan)  # (N, rows, cols) linear
        del I, I0, E
    tr = Transformer.from_crs(4326, epsg, always_xy=True)

    for view in ("lakes", "region", "pit"):  # lakes first: it sets the colour anchors
        meta, LON, LAT, px_ha = grids[view]
        X, Y = tr.transform(LON, LAT)
        cols, rows = (X - x[0]) / (x[1] - x[0]), (Y - y[0]) / (y[1] - y[0])
        samp = lambda a: map_coordinates(a, [rows, cols], order=1, mode="constant", cval=np.nan)
        HH = np.stack([to_db(samp(a)) for a in filt["hh"]])
        HV = np.stack([to_db(samp(a)) for a in filt["hv"]])
        ok = np.isfinite(HH).all(0) & np.isfinite(HV).all(0)
        if ok.mean() < 0.8:
            continue
        if view == "lakes":
            anchors = [(np.nanmedian(h[core]), np.nanmedian(v[core])) for h, v in zip(HH, HV)]
            anchor_mean = tuple(np.mean(anchors, axis=0))
        # per-date filtered RGB
        for i, d in enumerate(dates):
            ah, av = anchors[i]
            rgb = np.dstack([stretch(HH[i], ah, ah + W_HH), stretch(HV[i], av, av + W_HV), stretch(HH[i], ah, ah + W_HH)])
            rgb[~ok] = 0
            save(rgb, f"nisarmt_{view}_{d}_t{track}.jpg")
        # season mean
        mh, mv = to_db(np.nanmean(10 ** (HH / 10), 0)), to_db(np.nanmean(10 ** (HV / 10), 0))
        ah, av = anchor_mean
        rgb = np.dstack([stretch(mh, ah, ah + W_HH), stretch(mv, av, av + W_HV), stretch(mh, ah, ah + W_HH)])
        rgb[~ok] = 0
        save(rgb, f"nisarmt_{view}_mean_t{track}.jpg")
        if view == "lakes":  # single-channel season means for the HH vs HV slider
            g = lambda a, lo, w: np.where(ok, stretch(a, lo, lo + w), 0)
            Image.fromarray((g(mh, ah, W_HH) * 255).astype(np.uint8)).save(os.path.join(IMG, f"nisarmt_lakes_meanhh_t{track}.jpg"), quality=88)
            Image.fromarray((g(mv, av, W_HV) * 255).astype(np.uint8)).save(os.path.join(IMG, f"nisarmt_lakes_meanhv_t{track}.jpg"), quality=88)
        # HV change composite (June / August / October means): grey = no change, colour = when it was brightest
        months = np.array([int(d[5:7]) for d in dates])
        groups = [months <= 7, (months == 8), months >= 9]
        if all(g.any() for g in groups):
            chans = [to_db(np.nanmean(10 ** (HV[g] / 10), 0)) for g in groups]
            rgb = np.dstack([stretch(c, av, av + W_HV) for c in chans])
            rgb[~ok] = 0
            save(rgb, f"nisarmt_{view}_change_t{track}.jpg")
        # water per pass on the filtered stack, using that pass's calibrated thresholds
        if view in ("lakes", "pit"):
            # optional InSAR coherence (from gunw_views.py): water decorrelates, dry sand stays coherent
            cohf = os.path.join(src_gunw, f"coh_{view}_{track}.npz")
            coh = np.load(cohf) if os.path.exists(cohf) else None
            for i, d in enumerate(dates):
                r = single.get((d, track), {})
                if "thr_hh" not in r or r.get("contrast_db", 0) < 3:
                    continue
                w = ok & (HH[i] < r["thr_hh"]) & (HV[i] < r["thr_hv"])
                good = r.get("good", False)
                if view == "pit" and coh is not None and d in coh.files:
                    w &= coh[d].astype(float) < COH_WATER
                    good = True  # coherence removes the dry-sand / bare-field confusion
                water_votes[view].append((d, track, w, good))
    print("track", track, "dates", len(dates), flush=True)

# water frequency + early/late change maps, and areas from the filtered stack
rec = {}
for view, votes in water_votes.items():
    meta, LON, LAT, px_ha = grids[view]
    base = np.asarray(Image.open(os.path.join(IMG, f"{view}_s2_2024.jpg")).convert("L")).astype(float) / 255
    base = np.dstack([base] * 3) * 0.6
    use = votes if view == "lakes" else [v for v in votes if v[3]]  # pit: only passes without dry-field confusion
    if not use:
        continue
    freq = np.mean([w for _, _, w, _ in use], axis=0)
    zone = lake_mask if view == "lakes" else pit_fp
    blue = np.array([0.15, 0.45, 0.95])
    a = np.where(zone, freq, 0)[..., None]
    save(base * (1 - a) + blue * a, f"nisarmt_{view}_waterfreq.jpg")
    early = np.mean([w for d, _, w, _ in use if d[5:7] in ("06", "07")], axis=0) >= 0.5
    late = np.mean([w for d, _, w, _ in use if d[5:7] in ("09", "10", "11")], axis=0) >= 0.5
    img = base.copy()
    img[zone & early & late] = (0.15, 0.45, 0.95)   # water all season
    img[zone & ~early & late] = (0.10, 0.85, 0.55)  # new water by autumn
    img[zone & early & ~late] = (0.95, 0.55, 0.15)  # water lost by autumn
    save(img, f"nisarmt_{view}_newwater.jpg")
    rec[view] = {"passes": len(use),
                 "stable_ha": round(float((zone & early & late).sum() * px_ha), 1),
                 "new_ha": round(float((zone & ~early & late).sum() * px_ha), 1),
                 "lost_ha": round(float((zone & early & ~late).sum() * px_ha), 1),
                 "series": [{"date": d, "track": t, "water_ha": round(float((w & zone).sum() * px_ha), 1), "good": g}
                            for d, t, w, g in sorted(votes, key=lambda v: v[0])]}
    print(view, {k: v for k, v in rec[view].items() if k != "series"}, flush=True)

json.dump({"method": "Quegan multi-temporal filter (5x5) per track, bilinear resampling, per-pass thresholds from nisar_water.json",
           **rec}, open(os.path.join(DATA, "nisar_mt.json"), "w"), indent=1)
