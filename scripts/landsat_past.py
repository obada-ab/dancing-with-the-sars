"""Landsat (1984–2025) "past" layers for the Konin lakes story, via Microsoft Planetary Computer (anonymous).

Subcommands:
  area   <name> <lon0,lat0,lon1,lat1> <mask.png> <mask_bbox> <out.json>
         Annual summer lake-water area (MNDWI) inside a JRC-occurrence mask.
  rgb    <lon0,lat0,lon1,lat1> <W> <H> <item_id> <out.jpg>
         True-colour image on an exact lon/lat grid (pixel edges at bbox), fixed reflectance stretch.
  search <lon0,lat0,lon1,lat1> <date range> [max_cloud]
         List candidate scenes.

Run: uv run --with pillow --with numpy python -I landsat_past.py ...
"""
import io, json, math, os, sys, urllib.parse, urllib.request

import numpy as np
from PIL import Image

PC = "https://planetarycomputer.microsoft.com/api"
CACHE = os.environ.get("LANDSAT_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "landsat"))
os.makedirs(CACHE, exist_ok=True)
UA = {"User-Agent": "spaceapps-prototype", "Content-Type": "application/json"}


def search(bbox, dt, max_cloud=30, platforms=("landsat-5", "landsat-7", "landsat-8", "landsat-9")):
    body = {"collections": ["landsat-c2-l2"], "bbox": bbox, "datetime": dt, "limit": 200,
            "query": {"eo:cloud_cover": {"lt": max_cloud}, "platform": {"in": list(platforms)}}}
    out, url = [], f"{PC}/stac/v1/search"
    while url:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=UA, method="POST")
        d = json.load(urllib.request.urlopen(req, timeout=60))
        out += d["features"]
        nxt = [l for l in d.get("links", []) if l.get("rel") == "next"]
        if not nxt:
            break
        url, body = nxt[0]["href"], nxt[0].get("body", body)
    return out


def fetch(item, bbox, w, h, assets, resampling="nearest"):
    """Return (bands[len(assets)], valid_mask) arrays on an exact EPSG:4326 grid."""
    key = f"{item}_{'_'.join(f'{v:.4f}' for v in bbox)}_{w}x{h}_{'-'.join(assets)}_{resampling}.npy"
    p = os.path.join(CACHE, key)
    if not os.path.exists(p):
        q = [("collection", "landsat-c2-l2"), ("item", item), ("resampling", resampling)] + [("assets", a) for a in assets]
        url = f"{PC}/data/v1/item/bbox/{','.join(map(str, bbox))}/{w}x{h}.npy?" + urllib.parse.urlencode(q)
        data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA["User-Agent"]}), timeout=120).read()
        open(p, "wb").write(data)
    a = np.load(p)
    return a[:-1], a[-1] > 0


def _inside(pt, poly):
    x, y = pt; ins = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            ins = not ins
    return ins


def covers(feature, bbox):
    """True if the scene footprint contains all four bbox corners."""
    g = feature["geometry"]
    polys = [g["coordinates"][0]] if g["type"] == "Polygon" else [c[0] for c in g["coordinates"]]
    corners = [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[1]), (bbox[2], bbox[3])]
    return any(all(_inside(c, [tuple(p) for p in poly]) for c in corners) for poly in polys)


def sr(dn):  # Collection 2 L2 surface reflectance scaling
    return dn.astype(float) * 0.0000275 - 0.2


def load_mask(mask_png, mask_bbox, bbox, w, h, dilate_px=2):
    """JRC occurrence>0 resampled from its own grid to (bbox, w, h), then dilated."""
    m = np.asarray(Image.open(mask_png).convert("RGBA"))[..., 3] > 0
    L0, A0, L1, A1 = mask_bbox
    MH, MW = m.shape
    lons = bbox[0] + (np.arange(w) + 0.5) * (bbox[2] - bbox[0]) / w
    lats = bbox[3] - (np.arange(h) + 0.5) * (bbox[3] - bbox[1]) / h
    ix = np.clip(((lons - L0) / (L1 - L0) * MW).astype(int), 0, MW - 1)
    iy = np.clip(((A1 - lats) / (A1 - A0) * MH).astype(int), 0, MH - 1)
    out = m[iy][:, ix]
    for _ in range(dilate_px):
        o = out.copy()
        o[1:] |= out[:-1]; o[:-1] |= out[1:]; o[:, 1:] |= out[:, :-1]; o[:, :-1] |= out[:, 1:]
        out = o
    return out


def area(name, bbox, mask_png, mask_bbox, out_json, thr=0.0, min_clear=0.9):
    # analysis grid ~30 m
    lat_c = (bbox[1] + bbox[3]) / 2
    w = round((bbox[2] - bbox[0]) * 111320 * math.cos(math.radians(lat_c)) / 30)
    h = round((bbox[3] - bbox[1]) * 110574 / 30)
    px_ha = ((bbox[2] - bbox[0]) * 111320 * math.cos(math.radians(lat_c)) / w) * ((bbox[3] - bbox[1]) * 110574 / h) / 1e4
    mask = load_mask(mask_png, mask_bbox, bbox, w, h)
    res = []
    for year in range(1984, 2026):
        items = search(bbox, f"{year}-05-01/{year}-09-30", max_cloud=40)
        items = [f for f in items if covers(f, bbox)]
        items.sort(key=lambda f: f["properties"]["eo:cloud_cover"])
        scenes = []
        for f in items[:10]:
            if len(scenes) >= 5:
                break
            iid = f["id"]
            try:
                (g, s, qa), valid = fetch(iid, bbox, w, h, ["green", "swir16", "qa_pixel"])
            except Exception as e:
                print("  fail", iid, e, file=sys.stderr)
                continue
            qa = qa.astype(np.uint16)
            clear = ((qa & 1) == 0) & ((qa & (1 << 1)) == 0) & ((qa & (1 << 3)) == 0) & ((qa & (1 << 4)) == 0) & ((qa & (1 << 5)) == 0)
            frac = (clear & mask).sum() / mask.sum()
            if frac < min_clear:
                continue
            G, S = sr(g), sr(s)
            mndwi = (G - S) / (G + S + 1e-6)
            water = (mndwi > thr) & clear & mask
            # fill the small cloudy remainder proportionally (clear fraction >= 90%)
            a_ha = water.sum() / frac * px_ha
            scenes.append({"id": iid, "date": f["properties"]["datetime"][:10], "area_ha": round(float(a_ha), 1),
                           "clear_frac": round(float(frac), 3)})
        if scenes:
            res.append({"year": year, "area_ha": round(float(np.median([s["area_ha"] for s in scenes])), 1),
                        "n_scenes": len(scenes), "scenes": scenes})
        print(name, year, len(items), "candidates", len(scenes), "used", res[-1]["area_ha"] if scenes else "-", flush=True)
    json.dump({"name": name, "bbox": bbox, "method": f"Landsat C2 L2 SR, May-Sep, MNDWI>{thr}, QA clear, mask=JRC occurrence>0 dilated 2px, clear>={min_clear}, per-year median",
               "mask_ha": round(float(mask.sum() * px_ha), 1), "years": res}, open(out_json, "w"), indent=1)


def rgb(bbox, w, h, items, out, lo=0.0, hi=0.22, gamma=0.8, bands=("red", "green", "blue")):
    """items: one id, or several same-day ids joined by '+' (e.g. consecutive WRS rows) to mosaic.
    bands: true colour by default; ("swir16", "nir08", "red") gives a false colour where water is black."""
    img = np.zeros((h, w, 3)); have = np.zeros((h, w), bool)
    layers = []
    for item in items.split("+"):
        x, _ = fetch(item, bbox, w, h, list(bands), resampling="bilinear")
        (qa,), _ = fetch(item, bbox, w, h, ["qa_pixel"])
        nf = (qa.astype(np.uint16) & 1) == 0
        core = nf.copy()  # eroded by 1 px: avoid bilinear-blended edge pixels where possible
        core[1:] &= nf[:-1]; core[:-1] &= nf[1:]; core[:, 1:] &= nf[:, :-1]; core[:, :-1] &= nf[:, 1:]
        layers.append((np.dstack([sr(c) for c in x]), nf, core))
    for pass_ in ("core", "edge"):
        for v, nf, core in layers:
            ok = (core if pass_ == "core" else nf) & ~have
            img[ok] = v[ok]
            have |= ok
    img = np.clip((img - lo) / (hi - lo), 0, 1) ** gamma
    img[~have] = 0
    Image.fromarray((img * 255).astype(np.uint8)).save(out, quality=88)
    print(out, "covered", round(float(have.mean()), 4))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "area":
        name, bb, mpng, mbb, out = sys.argv[2:7]
        area(name, list(map(float, bb.split(","))), mpng, list(map(float, mbb.split(","))), out)
    elif cmd == "rgb":
        bb, w, h, items, out = sys.argv[2:7]
        kw = {}
        if len(sys.argv) > 7 and sys.argv[7] == "false":
            kw = dict(bands=("swir16", "nir08", "red"), hi=0.40, gamma=0.9)
        rgb(list(map(float, bb.split(","))), int(w), int(h), items, out, **kw)
    elif cmd == "search":
        bb, dt = sys.argv[2:4]
        mc = float(sys.argv[4]) if len(sys.argv) > 4 else 20
        for f in sorted(search(list(map(float, bb.split(","))), dt, mc), key=lambda f: f["properties"]["datetime"]):
            print(f["id"], f["properties"]["datetime"][:10], f["properties"]["eo:cloud_cover"])
