"""Full-resolution NISAR GCOV (20 m) windows -> page images + open-water area time series.

Inputs are npz files from fetch_gcov_window.py (UTM grid, HH/HV in dB).
For each date and each view grid (layers.json files) writes:
  img/nisar20_<view>_<date>_t<track>.jpg   RGB = HH, HV, HH-HV (fixed stretch, same for all dates)
and writes data/nisar_water.json with open-water area (ha) per date for the lake chain and the Jóźwin pit.

Usage: uv run --with numpy --with pillow --with pyproj python -I gcov_views.py <npz_dir> <site_dir>
"""
import glob, json, math, os, re, sys

import numpy as np
from PIL import Image
from pyproj import Transformer

src, site = sys.argv[1:3]
IMG, DATA = os.path.join(site, "img"), os.path.join(site, "data")
VIEWS = {"region": "layers.json", "lakes": "lakes_layers.json", "pit": "pit_layers.json"}
HH_R, HV_R = (-20.0, -4.0), (-23.0, -11.0)  # widths used above the per-pass water anchor (16 dB HH, 12 dB HV)  # RGB = HH, HV, HH (ASF-style: water dark, forest green-white, fields magenta)
FVEG_HH, FVEG_RATIO = -7.0, 9.0  # dB; bright HH with weak HV = double bounce (water + stems)
PIT_BOX = (18.12, 52.40, 18.22, 52.445)  # same box as the Sentinel-1 Jóźwin series


def boxcar(a, k):
    """k x k mean filter (speckle reduction) on a 2-D array, NaN-aware."""
    v = np.isfinite(a)
    z = np.where(v, a, 0.0)
    def s(x):
        c = np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        r = k // 2
        c = np.pad(c, ((r, r), (r, r)), mode="edge")
        h, w = x.shape
        return c[k:k + h, k:k + w] - c[:h, k:k + w] - c[k:k + h, :w] + c[:h, :w]
    n = s(v.astype(float))
    return np.where(n > 0, s(z) / np.maximum(n, 1), np.nan)


def grid(view):
    m = json.load(open(os.path.join(IMG, VIEWS[view])))
    L0, A0, L1, A1 = m["bbox"]; W, H = m["size"]
    lon = L0 + (np.arange(W) + 0.5) * (L1 - L0) / W
    lat = A1 - (np.arange(H) + 0.5) * (A1 - A0) / H
    return m, np.meshgrid(lon, lat)


def stretch(a, lo, hi):
    return np.clip((a - lo) / (hi - lo), 0, 1)


grids = {v: grid(v) for v in VIEWS}


def shift_or(a, r):
    d = a.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d |= np.roll(np.roll(a, dy, 0), dx, 1)
    return d


def occurrence(prefix):
    return np.asarray(Image.open(os.path.join(IMG, f"{prefix}jrc_occurrence.png")).convert("RGBA"))[..., 3]


# Lake-chain masks (lakes view): where to measure, where to calibrate.
occ = occurrence("lakes_")
lake_mask = shift_or(occ > 0, 2)               # same mask as the Landsat series
core = ~shift_or(~(occ >= 200), 3)             # permanent-water core, eroded 3 px: "surely water"
land = ~shift_or(occ > 0, 15)                  # >15 px from any water: "surely land"

# Jóźwin pit footprint (pit view): excavation (bright sand in 2016/2024) or any JRC water, inside PIT_BOX.
pm = json.load(open(os.path.join(IMG, "pit_layers.json")))
_, (PLON, PLAT) = grids["pit"]
sand = np.zeros(PLON.shape, bool)
for y in ("2016", "2024"):
    rgb = np.asarray(Image.open(os.path.join(IMG, f"pit_s2_{y}.jpg")).convert("RGB")).astype(float)
    sand |= rgb.mean(axis=2) > 140
pit_fp = shift_or(sand | (occurrence("pit_") > 0), 4)
pit_fp &= (PLON >= PIT_BOX[0]) & (PLON <= PIT_BOX[2]) & (PLAT >= PIT_BOX[1]) & (PLAT <= PIT_BOX[3])

THR_F = float(os.environ.get("THR_F", 0.25))  # threshold position between water (0) and land (1) medians
MAX_FALSE_HA = 500.0  # ha of "water" >15 px from any JRC water in the lakes view
MIN_CONTRAST = 3.0  # dB between land and water medians (HH); below this the pass is too noisy/windy

series = []
for p in sorted(glob.glob(os.path.join(src, "*.npz"))):
    name = os.path.basename(p)
    m = re.search(r"_(\d{3})_([AD])_\d{3}_.*?_(\d{8})T", name)
    track, direction, ds = m.group(1), m.group(2), m.group(3)
    date = f"{ds[:4]}-{ds[4:6]}-{ds[6:]}"
    z = np.load(p)
    x, y, epsg = z["x"], z["y"], int(z["epsg"])
    lin = lambda db: 10 ** (db.astype(np.float32) / 10)
    with np.errstate(divide="ignore", invalid="ignore"):
        hh = 10 * np.log10(boxcar(lin(z["hh_db"]), 5))
        hv = 10 * np.log10(boxcar(lin(z["hv_db"]), 5))
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    dx, dy = x[1] - x[0], y[1] - y[0]
    rec = {"date": date, "track": track, "dir": direction}
    th = None
    for view in ("lakes", "region", "pit"):  # lakes first: it calibrates the water thresholds
        meta, (LON, LAT) = grids[view]
        X, Y = tr.transform(LON, LAT)
        ci = np.round((X - x[0]) / dx).astype(int)
        ri = np.round((Y - y[0]) / dy).astype(int)
        ok = (ci >= 0) & (ci < len(x)) & (ri >= 0) & (ri < len(y))
        ci, ri = np.clip(ci, 0, len(x) - 1), np.clip(ri, 0, len(y) - 1)
        HHv, HVv = np.where(ok, hh[ri, ci], np.nan), np.where(ok, hv[ri, ci], np.nan)
        valid = np.isfinite(HHv) & np.isfinite(HVv)
        if valid.mean() < 0.8:
            continue
        if view == "lakes":
            # per-pass colour anchor: the permanent-lake core is black, so water looks the same on every date/track
            anchor = (float(np.nanmedian(HHv[core & valid])), float(np.nanmedian(HVv[core & valid])))
        hr = (anchor[0], anchor[0] + (HH_R[1] - HH_R[0]))
        vr = (anchor[1], anchor[1] + (HV_R[1] - HV_R[0]))
        rgb = np.dstack([stretch(HHv, *hr), stretch(HVv, *vr), stretch(HHv, *hr)])
        rgb[~valid] = 0
        Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(IMG, f"nisar20_{view}_{date}_t{track}.jpg"), quality=86)
        rec.setdefault("views", []).append(view)
        L0, A0, L1, A1 = meta["bbox"]; W, H = meta["size"]
        px_ha = ((L1 - L0) * 111320 * math.cos(math.radians((A0 + A1) / 2)) / W) * ((A1 - A0) * 110574 / H) / 1e4
        if view == "lakes":
            w_hh, l_hh = np.nanmedian(HHv[core & valid]), np.nanmedian(HHv[land & valid])
            w_hv, l_hv = np.nanmedian(HVv[core & valid]), np.nanmedian(HVv[land & valid])
            th = (w_hh + THR_F * (l_hh - w_hh), w_hv + THR_F * (l_hv - w_hv))
            rec.update({"contrast_db": round(float(l_hh - w_hh), 2), "thr_hh": round(float(th[0]), 2), "thr_hv": round(float(th[1]), 2)})
            for nm, arr, rng in (("hh", HHv, hr), ("hv", HVv, vr)):
                g = (stretch(arr, *rng) * 255).astype(np.uint8)
                g[~valid] = 0
                Image.fromarray(g).save(os.path.join(IMG, f"nisar20_lakes{nm}_{date}_t{track}.jpg"), quality=84)
            if rec["contrast_db"] < MIN_CONTRAST:
                th = None
                continue
            water = valid & (HHv < th[0]) & (HVv < th[1])
            # L-band special: emergent/flooded vegetation (reeds in water) gives a strong HH double-bounce
            fveg = valid & lake_mask & ~water & (HHv > FVEG_HH) & ((HHv - HVv) > FVEG_RATIO)
            rec["lakes_water_ha"] = round(float((water & lake_mask).sum() * px_ha), 1)
            rec["lakes_flooded_veg_ha"] = round(float(fveg.sum() * px_ha), 1)
            rec["false_water_outside_ha"] = round(float((water & land).sum() * px_ha), 1)
            # quality gate: dry smooth fields can look like water; flag passes with lots of "water" far from any lake
            rec["good"] = bool(rec["false_water_outside_ha"] < MAX_FALSE_HA)
            base = np.asarray(Image.open(os.path.join(IMG, "lakes_s2_2024.jpg")).convert("RGB")).astype(float) * 0.55
            base[water & lake_mask] = (40, 110, 230)
            base[fveg] = (255, 150, 30)
            Image.fromarray(base.astype(np.uint8)).save(os.path.join(IMG, f"nisar20_lakeswater_{date}_t{track}.jpg"), quality=86)
        if view == "pit" and th is not None:
            water = valid & (HHv < th[0]) & (HVv < th[1])
            rec["pit_water_ha"] = round(float((water & pit_fp).sum() * px_ha), 1)
    series.append(rec)
    print(rec, flush=True)

series.sort(key=lambda r: (r["date"], r["track"]))
os.makedirs(DATA, exist_ok=True)
json.dump({"product": "NISAR L2 GCOV (PROVISIONAL/V1), frequency A, 20 m, 5x5 boxcar",
           "water_rule": "per pass: HH and HV below the midpoint between permanent-lake core and far-from-water land medians; "
                         f"passes with HH contrast < {MIN_CONTRAST} dB skipped",
           "flooded_veg_rule": f"in lake mask, HH > {FVEG_HH} dB and HH-HV > {FVEG_RATIO} dB",
           "lake_mask": "JRC occurrence>0 dilated 2px (same as Landsat series)",
           "pit_mask": f"excavation footprint (bright sand 2016/2024 or JRC water) inside {PIT_BOX}",
           "rgb": "R=HH, G=HV, B=HH; per pass, black = permanent-lake core median, white = +16 dB (HH) / +12 dB (HV)",
           "series": series}, open(os.path.join(DATA, "nisar_water.json"), "w"), indent=1)
