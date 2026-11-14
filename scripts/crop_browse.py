"""Crop NISAR LATLON browse PNGs to a lon/lat box, assuming the PNG spans the footprint bbox.

Usage: uv run --with pillow python -I crop_browse.py <index.json> <outdir> <lon0> <lat0> <lon1> <lat1> [scale]
"""
import json, os, sys
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
idx, out = sys.argv[1], sys.argv[2]
lon0, lat0, lon1, lat1 = map(float, sys.argv[3:7])
scale = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
os.makedirs(out, exist_ok=True)
for it in json.load(open(idx)):
    im = Image.open(it["png"]).convert("RGB")
    xs = [c[0] for c in it["footprint"]]; ys = [c[1] for c in it["footprint"]]
    W, H = im.size
    px = lambda lon: (lon - min(xs)) / (max(xs) - min(xs)) * W
    py = lambda lat: (max(ys) - lat) / (max(ys) - min(ys)) * H
    box = tuple(round(v) for v in (px(lon0), py(lat1), px(lon1), py(lat0)))
    c = im.crop(box)
    if scale != 1.0:
        c = c.resize((round(c.width * scale), round(c.height * scale)), Image.LANCZOS)
    dst = os.path.join(out, f"{it['scene'].split('_')[3]}_{it['date']}_{it['scene'].split('_')[5]}.jpg")
    c.save(dst, quality=88)
    print(dst, c.size)
