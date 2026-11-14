"""NISAR GUNW (80 m unwrapped interferograms) -> ground-movement and coherence products for the page.

Per track, consecutive 12–36-day pairs are chained (date1->date2->...) and summed into a cumulative
line-of-sight (LOS) change. Per pair: ionosphere screen removed, low-coherence / unconnected pixels masked,
a plane fitted and removed (long-wavelength atmosphere/orbit), and referenced to the median of stable,
coherent pixels. What remains is *relative* movement within the ~65 km window.

Outputs:
  img/insar_cum_<view>_t<track>.png        cumulative LOS change (mm), diverging colours, transparent where masked
  img/insar_coh_<view>_t<track>.jpg        season-mean coherence (black = water/changing, white = stable)
  data/insar.json                          per-pair stats + time series at a few places
  data/nisar_gunw/coh_<view>_<track>.npz   per-date coherence on the view grid (for water/sand separation)

Usage: uv run --with numpy --with scipy --with pillow --with pyproj python -I gunw_views.py <gunw_dir> <site_dir>
"""
import glob, json, math, os, sys

import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy.ndimage import map_coordinates

src, site = sys.argv[1:3]
IMG, DATA = os.path.join(site, "img"), os.path.join(site, "data")
WAVELENGTH = 0.2384  # m, NISAR L-band (1.257 GHz)
COH_MIN = 0.35
VIEWS = {"region": "layers.json", "pit": "pit_layers.json"}
PLACES = {"Jóźwin IIB pit rim": (52.405, 18.20), "Kleczew": (52.371, 18.173), "Lake Wilczyńskie shore": (52.47, 18.10),
          "Konin": (52.229, 18.245), "Tomisławice pit": (52.462, 18.524)}
DIV = np.array([[33, 102, 172], [146, 197, 222], [247, 247, 247], [244, 165, 130], [178, 24, 43]], float)


def grid(view):
    m = json.load(open(os.path.join(IMG, VIEWS[view])))
    L0, A0, L1, A1 = m["bbox"]; W, H = m["size"]
    LON, LAT = np.meshgrid(L0 + (np.arange(W) + 0.5) * (L1 - L0) / W, A1 - (np.arange(H) + 0.5) * (A1 - A0) / H)
    return m, LON, LAT


def plane_detrend(a, ok):
    r, c = np.nonzero(ok)
    if len(r) < 100:
        return a
    A = np.c_[r, c, np.ones_like(r)].astype(float)
    coef, *_ = np.linalg.lstsq(A, a[ok], rcond=None)
    R, Cc = np.indices(a.shape)
    return a - (coef[0] * R + coef[1] * Cc + coef[2])


pairs = {}
for p in sorted(glob.glob(os.path.join(src, "*.npz"))):
    z = np.load(p)
    track = os.path.basename(p).split("_")[5]
    pairs.setdefault(track, []).append(z)

out = {"method": f"ionosphere-corrected, coherence>{COH_MIN}, plane removed, referenced to median of coherent pixels; "
                  f"LOS mm = phase * {WAVELENGTH}/(4π) * 1000", "tracks": {}}
for track, zs in sorted(pairs.items()):
    zs.sort(key=lambda z: str(z["ref_date"]))
    shape = zs[0]["unw"].shape
    if any(z["unw"].shape != shape for z in zs):
        print("grid mismatch in track", track); continue
    cum = np.zeros(shape); okall = np.ones(shape, bool); cohs = []
    steps, chain_ok = [], True
    for i, z in enumerate(zs):
        if i and str(z["ref_date"]) != str(zs[i - 1]["sec_date"]):
            chain_ok = False
        coh = z["coh"].astype(float)
        ok = np.isfinite(z["unw"]) & (coh > COH_MIN) & (z["cc"] > 0)
        ph = z["unw"].astype(float) - np.nan_to_num(z["iono"].astype(float))
        ph = plane_detrend(np.where(ok, ph, 0), ok)
        ph -= np.median(ph[ok]) if ok.any() else 0
        mm = ph * WAVELENGTH / (4 * math.pi) * 1000
        cum += np.where(ok, mm, 0); okall &= ok; cohs.append(coh)
        steps.append({"ref": str(z["ref_date"]), "sec": str(z["sec_date"]), "valid_frac": round(float(ok.mean()), 3),
                      "std_mm": round(float(np.std(mm[ok])), 2) if ok.any() else None})
    x, y, epsg = zs[0]["x"], zs[0]["y"], int(zs[0]["epsg"])
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    rc = lambda lon, lat: ((np.asarray(tr.transform(lon, lat)[1]) - y[0]) / (y[1] - y[0]),
                           (np.asarray(tr.transform(lon, lat)[0]) - x[0]) / (x[1] - x[0]))
    cum_m = np.where(okall, cum, np.nan)
    series = {}
    for name, (la, lo) in PLACES.items():
        r, c = rc(lo, la); r, c = int(round(float(r))), int(round(float(c)))
        if 2 <= r < shape[0] - 2 and 2 <= c < shape[1] - 2:
            v = cum_m[r - 2:r + 3, c - 2:c + 3]
            series[name] = round(float(np.nanmedian(v)), 1) if np.isfinite(v).any() else None
    mcoh = np.mean(cohs, 0)
    for view in VIEWS:
        meta, LON, LAT = grid(view)
        R, Cc = rc(LON, LAT)
        s = lambda a, o=1: map_coordinates(a, [R, Cc], order=o, mode="constant", cval=np.nan)
        cv = s(cum_m)
        t = (np.clip(np.nan_to_num(cv) / 20.0, -1, 1) + 1) / 2 * (len(DIV) - 1)
        i = np.minimum(t.astype(int), len(DIV) - 2); f = (t - i)[..., None]
        rgb = DIV[i] * (1 - f) + DIV[i + 1] * f
        alpha = np.where(np.isfinite(cv), 235, 0)
        Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA").save(os.path.join(IMG, f"insar_cum_{view}_t{track}.png"), optimize=True)
        cg = np.clip(np.nan_to_num(s(mcoh)), 0, 1)
        Image.fromarray((cg * 255).astype(np.uint8)).save(os.path.join(IMG, f"insar_coh_{view}_t{track}.jpg"), quality=88)
        # per-date coherence (mean of the pairs touching each date) for water/sand separation
        dates = sorted({st["ref"] for st in steps} | {st["sec"] for st in steps})
        per = {}
        for d in dates:
            touching = [cohs[k] for k, st in enumerate(steps) if d in (st["ref"], st["sec"])]
            per[f"{d[:4]}-{d[4:6]}-{d[6:]}"] = np.clip(np.nan_to_num(s(np.mean(touching, 0))), 0, 1).astype(np.float16)
        np.savez_compressed(os.path.join(src, f"coh_{view}_{track}.npz"), **per)
    v = cum_m[np.isfinite(cum_m)]
    out["tracks"][track] = {"first": steps[0]["ref"], "last": steps[-1]["sec"], "pairs": steps, "chain_continuous": chain_ok,
                            "valid_frac": round(float(okall.mean()), 3), "p5_p95_mm": [round(float(np.percentile(v, 5)), 1), round(float(np.percentile(v, 95)), 1)] if v.size else None,
                            "places_mm": series}
    print(track, out["tracks"][track]["first"], "->", out["tracks"][track]["last"], "valid", out["tracks"][track]["valid_frac"],
          "p5/p95", out["tracks"][track]["p5_p95_mm"], series, "chain", chain_ok, flush=True)
json.dump(out, open(os.path.join(DATA, "insar.json"), "w"), indent=1, ensure_ascii=False)
