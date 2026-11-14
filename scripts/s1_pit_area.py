"""Monthly open-water area of the Jóźwin IIB pit from Sentinel-1 RTC (Planetary Computer, anonymous).

Water = VV gamma0 < THR dB (smooth water is radar-dark), median-filtered 3x3, inside the pit box.
One scene per month from a single relative orbit, so the viewing geometry is constant.

Usage: uv run --with numpy python -I s1_pit_area.py <out.json> [rel_orbit] [thr_db]
"""
import json, math, os, sys, urllib.parse, urllib.request

import numpy as np

PC = "https://planetarycomputer.microsoft.com/api"
BBOX = [18.12, 52.40, 18.22, 52.445]  # Jóźwin IIB pit (Kazimierz pit lake to the south is excluded)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "s1")
os.makedirs(CACHE, exist_ok=True)
out_json = sys.argv[1]
ORBIT = int(sys.argv[2]) if len(sys.argv) > 2 else 175
THR = float(sys.argv[3]) if len(sys.argv) > 3 else -18.0
lat_c = (BBOX[1] + BBOX[3]) / 2
W = round((BBOX[2] - BBOX[0]) * 111320 * math.cos(math.radians(lat_c)) / 20)
H = round((BBOX[3] - BBOX[1]) * 110574 / 20)
PX_HA = ((BBOX[2] - BBOX[0]) * 111320 * math.cos(math.radians(lat_c)) / W) * ((BBOX[3] - BBOX[1]) * 110574 / H) / 1e4


def search():
    body = {"collections": ["sentinel-1-rtc"], "bbox": BBOX, "datetime": "2024-09-01/2026-12-31", "limit": 200,
            "query": {"sat:relative_orbit": {"eq": ORBIT}}}
    url, out = f"{PC}/stac/v1/search", []
    while url:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": "spaceapps-prototype"})
        d = json.load(urllib.request.urlopen(req, timeout=60))
        out += d["features"]
        nxt = [l for l in d.get("links", []) if l.get("rel") == "next"]
        url, body = (nxt[0]["href"], nxt[0].get("body", body)) if nxt else (None, None)
    return out


def fetch_vv(item):
    p = os.path.join(CACHE, f"{item}_{W}x{H}_vv.npy")
    if not os.path.exists(p):
        q = urllib.parse.urlencode([("collection", "sentinel-1-rtc"), ("item", item), ("assets", "vv")])
        url = f"{PC}/data/v1/item/bbox/{','.join(map(str, BBOX))}/{W}x{H}.npy?{q}"
        open(p, "wb").write(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "x"}), timeout=120).read())
    a = np.load(p)
    return a[0].astype(float), a[-1] > 0


def median3(a):
    p = np.pad(a, 1, mode="edge")
    st = np.stack([p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)])
    return np.median(st, axis=0)


items = sorted(search(), key=lambda f: f["properties"]["datetime"])
by_month = {}
for f in items:
    by_month.setdefault(f["properties"]["datetime"][:7], []).append(f)
res = []
for month, fs in sorted(by_month.items()):
    scenes = []
    for f in fs:
        vv, valid = fetch_vv(f["id"])
        if valid.mean() < 0.98:
            continue
        db = 10 * np.log10(np.clip(vv, 1e-5, None))
        water = (median3(db) < THR) & valid
        scenes.append({"date": f["properties"]["datetime"][:10], "item": f["id"], "water_ha": round(float(water.sum() * PX_HA), 1)})
    if not scenes:
        continue
    a = [x["water_ha"] for x in scenes]
    # wind-roughened water turns radar-bright and is missed, so the monthly max is the most robust estimate
    res.append({"month": month, "water_ha_max": max(a), "water_ha_median": round(float(np.median(a)), 1), "n": len(a), "scenes": scenes})
    print(month, len(a), "max", max(a), "median", res[-1]["water_ha_median"], flush=True)
json.dump({"bbox": BBOX, "relative_orbit": ORBIT, "method": f"Sentinel-1 RTC VV gamma0 < {THR} dB, 3x3 median, all scenes of one relative orbit; monthly max (robust to wind) and median",
           "months": res}, open(out_json, "w"), indent=1)
