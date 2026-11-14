"""NISAR SME2 (200 m soil moisture) -> coloured maps on the page grid + a regional time series.

Usage: uv run --with h5py --with numpy --with pillow python -I sme2_maps.py <sme2_dir> <layers.json> <img_out> <json_out>
"""
import glob, json, os, re, sys

import h5py, numpy as np
from PIL import Image

src, layers, img_out, json_out = sys.argv[1:5]
meta = json.load(open(layers))
L0, A0, L1, A1 = meta["bbox"]
W, H = meta["size"]
# Sample at 1/4 of the page grid (~190 m, close to the native 200 m), then upscale smoothly (bicubic) to avoid blocky pixels.
SUB = 4
w, h = W // SUB, H // SUB
lons = L0 + (np.arange(w) + 0.5) * (L1 - L0) / w
lats = A1 - (np.arange(h) + 0.5) * (A1 - A0) / h

VMIN, VMAX = 0.05, 0.35  # m3/m3
# dry (sand) -> wet (teal/blue)
STOPS = np.array([[166, 97, 26], [223, 194, 125], [245, 245, 220], [128, 205, 193], [1, 133, 113], [0, 70, 110]], float)


def colorize(sm, valid):
    t = np.clip((sm - VMIN) / (VMAX - VMIN), 0, 1) * (len(STOPS) - 1)
    i = np.minimum(t.astype(int), len(STOPS) - 2)
    f = (t - i)[..., None]
    rgb = STOPS[i] * (1 - f) + STOPS[i + 1] * f
    alpha = valid.astype(float) if valid.dtype == bool else valid
    rgba = np.dstack([rgb, alpha * 255]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


series = []
for p in sorted(glob.glob(os.path.join(src, "*SME2*.h5"))):
    name = os.path.basename(p)
    if "_SHSH_" in name or "_N_P_" in name:  # skip single-pol (HH only) and partial-frame passes for consistency
        print("skip (single-pol)", name); continue
    m = re.search(r"_(\d{3})_([AD])_\d{3}_.*?_(\d{8})T", name)
    track, direction, date = m.group(1), m.group(2), m.group(3)
    date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    try:
        g = h5py.File(p, "r")["science/LSAR/SME2/grids"]
        lat, lon = g["latitude"][:], g["longitude"][:]
    except Exception as e:
        print("skip (unreadable)", name, e); continue
    # nearest EASE-Grid cell for every output pixel (1-D lat/lon axes)
    ci = np.clip(np.searchsorted(lon, lons), 1, len(lon) - 1)
    ci -= (lons - lon[ci - 1]) < (lon[ci] - lons)
    order = np.argsort(lat)
    ri_sorted = np.clip(np.searchsorted(lat[order], lats), 1, len(lat) - 1)
    ri_sorted -= (lats - lat[order][ri_sorted - 1]) < (lat[order][ri_sorted] - lats)
    ri = order[ri_sorted]
    r0, r1, c0, c1 = ri.min(), ri.max() + 1, ci.min(), ci.max() + 1
    sm = g["soilMoisture"][r0:r1, c0:c1][np.ix_(ri - r0, ci - c0)]
    water = g["ancillaryData/waterbodyFraction"][r0:r1, c0:c1][np.ix_(ri - r0, ci - c0)]
    valid = (sm > -9000) & (sm >= 0) & (sm <= 1) & ~(water > 0.3)
    cover = float(valid.mean())
    if cover < 0.5:
        print("skip (coverage)", date, track, round(cover, 2)); continue
    fn = f"sme2_{date}_t{track}.png"
    # fill invalid cells with nearby values before smoothing so lake edges don't bleed dark
    filled = np.where(valid, sm, np.nan)
    for _ in range(3):
        nb = np.nanmean(np.stack([np.roll(filled, s, a) for a in (0, 1) for s in (-1, 1)]), axis=0)
        filled = np.where(np.isnan(filled), nb, filled)
    # light 3x3 smoothing (display only; statistics below use the raw values)
    f0 = np.nan_to_num(filled)
    sm_disp = sum(np.roll(np.roll(f0, a, 0), b, 1) * wgt for a, b, wgt in
                  [(0, 0, 4), (1, 0, 2), (-1, 0, 2), (0, 1, 2), (0, -1, 2), (1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1)]) / 16
    k = [(0, 0, 4), (1, 0, 2), (-1, 0, 2), (0, 1, 2), (0, -1, 2), (1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1)]
    soft = sum(np.roll(np.roll(valid.astype(float), a, 0), b, 1) * wgt for a, b, wgt in k) / 16
    img = colorize(sm_disp, np.clip((soft - 0.25) / 0.5, 0, 1)).resize((W, H), Image.LANCZOS)
    img.save(os.path.join(img_out, fn), optimize=True)
    v = sm[valid]
    series.append({"date": date, "track": track, "dir": direction, "file": fn, "coverage": round(cover, 3),
                   "median": round(float(np.median(v)), 4), "p25": round(float(np.percentile(v, 25)), 4),
                   "p75": round(float(np.percentile(v, 75)), 4)})
    print(date, track, direction, "median", round(float(np.median(v)), 3), "cover", round(cover, 2))

series.sort(key=lambda s: (s["date"], s["track"]))
json.dump({"units": "m3/m3", "vmin": VMIN, "vmax": VMAX, "bbox": meta["bbox"], "product": "NISAR L3 SME2 PROVISIONAL",
           "series": series}, open(json_out, "w"), indent=1)
