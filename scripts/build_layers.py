"""Build co-registered image layers for the Konin story page.

- Downloads public XYZ tiles (JRC Global Surface Water, EOX Sentinel-2 cloudless, Esri imagery)
  and resamples them onto one lon/lat grid (equirectangular) covering BBOX.
- Crops NISAR GCOV browse PNGs (see fetch_nisar_browse.py) to the same grid and fixes their
  geolocation (browse quick-looks can be off by km) by registering the radar "dark water" mask
  against the JRC water-occurrence layer.

Usage: uv run --with pillow --with numpy python -I build_layers.py <nisar_browse_dir> <tile_cache> <out_dir>
"""
import glob, io, json, math, os, sys, urllib.request

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
BROWSE, CACHE, OUT = sys.argv[1:4]
# Optional overrides: BBOX="lon0,lat0,lon1,lat1" ZOOM=13 WIDTH=1400 PREFIX=lakes_ RADAR=0
LON0, LAT0, LON1, LAT1 = map(float, os.environ.get("BBOX", "17.70,52.15,18.65,52.70").split(","))
Z = int(os.environ.get("ZOOM", 11))
W = int(os.environ.get("WIDTH", 1400))  # output width in px; height keeps ground aspect
PREFIX = os.environ.get("PREFIX", "")
RADAR = os.environ.get("RADAR", "1") == "1"
H = round(W * (LAT1 - LAT0) / ((LON1 - LON0) * math.cos(math.radians((LAT0 + LAT1) / 2))))

LAYERS = {
    "jrc_occurrence": "https://storage.googleapis.com/global-surface-water/tiles2021/occurrence/{z}/{x}/{y}.png",
    "jrc_transitions": "https://storage.googleapis.com/global-surface-water/tiles2021/transitions/{z}/{x}/{y}.png",
    "s2_2016": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/g/{z}/{y}/{x}.jpg",
    "s2_2024": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg",
}
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

lons = np.linspace(LON0, LON1, W, endpoint=False) + (LON1 - LON0) / W / 2
lats = np.linspace(LAT1, LAT0, H, endpoint=False) - (LAT1 - LAT0) / H / 2
LON, LAT = np.meshgrid(lons, lats)


def merc_px(lon, lat, z):
    n = 256 * 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / math.pi) / 2 * n
    return x, y


def tile(name, url, z, x, y):
    p = os.path.join(CACHE, f"{name}_{z}_{x}_{y}.png")
    if not os.path.exists(p):
        req = urllib.request.Request(url.format(z=z, x=x, y=y), headers={"User-Agent": "spaceapps-prototype"})
        try:
            data = urllib.request.urlopen(req, timeout=30).read()
            Image.open(io.BytesIO(data)).convert("RGBA").save(p)
        except Exception as e:  # missing tile == transparent
            print("tile miss", name, z, x, y, e)
            Image.new("RGBA", (256, 256), (0, 0, 0, 0)).save(p)
    return np.asarray(Image.open(p).convert("RGBA"))


def build_xyz(name, url):
    px, py = merc_px(LON, LAT, Z)
    tx0, ty0 = int(px.min() // 256), int(py.min() // 256)
    tx1, ty1 = int(px.max() // 256), int(py.max() // 256)
    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 4), np.uint8)
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = tile(name, url, Z, tx, ty)
    ix = np.clip((px - tx0 * 256).astype(int), 0, mosaic.shape[1] - 1)
    iy = np.clip((py - ty0 * 256).astype(int), 0, mosaic.shape[0] - 1)
    return mosaic[iy, ix]


def kml_bounds(path):
    """Corner lon/lat lists from the browse image's LATLON.kml (LatLonQuad)."""
    import re
    txt = open(path).read()
    nums = [float(v) for v in re.search(r"<coordinates>(.*?)</coordinates>", txt, re.S).group(1).replace(",", " ").split()]
    return nums[0::2], nums[1::2]


def crop_browse(item, margin):
    """Crop a LATLON browse PNG to BBOX (+margin degrees), resampled to the output grid scale."""
    im = np.asarray(Image.open(item["png"]).convert("RGB"))
    xs, ys = kml_bounds(item["kml"])
    h, w = im.shape[:2]
    dlon, dlat = (LON1 - LON0) / W, (LAT1 - LAT0) / H
    mx, my = round(margin / dlon), round(margin / dlat)
    glon = LON0 - mx * dlon + (np.arange(W + 2 * mx) + 0.5) * dlon
    glat = LAT1 + my * dlat - (np.arange(H + 2 * my) + 0.5) * dlat
    ix = ((glon - min(xs)) / (max(xs) - min(xs)) * w).astype(int)
    iy = ((max(ys) - glat) / (max(ys) - min(ys)) * h).astype(int)
    valid = (ix[None, :] >= 0) & (ix[None, :] < w) & (iy[:, None] >= 0) & (iy[:, None] < h)
    out = im[np.clip(iy, 0, h - 1)][:, np.clip(ix, 0, w - 1)]
    out[~valid] = 0
    return out, mx, my


def blur(a, r=3):
    k = 2 * r + 1
    ii = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    p = np.pad(ii, ((r, r), (r, r)), mode="edge")
    h, w = a.shape
    return (p[k:k + h, k:k + w] - p[:h, k:k + w] - p[k:k + h, :w] + p[:h, :w]) / (k * k)


def best_shift(radar_big, water_ref, mx, my):
    """Find the (dy, dx) shift that best matches radar dark pixels to JRC permanent water.

    Zero-mean cross-correlation (via FFT) of blurred binary masks; dark = darkest 3% of valid pixels.
    """
    g = radar_big.astype(float).mean(axis=2)
    valid = g > 3
    thr = np.percentile(g[valid], 3) if valid.any() else 0
    dark = blur(((g <= thr) & valid).astype(float))
    dark[valid] -= dark[valid].mean()
    dark[~valid] = 0
    ref = blur(water_ref.astype(float))
    ref -= ref.mean()
    pad = np.zeros_like(dark); pad[:H, :W] = ref
    corr = np.real(np.fft.ifft2(np.fft.fft2(dark) * np.conj(np.fft.fft2(pad))))[:2 * my + 1, :2 * mx + 1]
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    peak = corr[dy, dx] / (np.sqrt((dark ** 2).sum() * (ref ** 2).sum()) + 1e-9)
    return (int(dy), int(dx)), float(peak)


meta = {"bbox": [LON0, LAT0, LON1, LAT1], "size": [W, H], "layers": {}, "radar": []}
xyz = {}
for name, url in LAYERS.items():
    arr = build_xyz(name, url)
    xyz[name] = arr
    rgb = Image.fromarray(arr)
    if name.startswith("s2"):
        rgb.convert("RGB").save(os.path.join(OUT, f"{PREFIX}{name}.jpg"), quality=85)
    else:
        rgb.save(os.path.join(OUT, f"{PREFIX}{name}.png"))
    meta["layers"][name] = url
    print("layer", name, arr.shape)

water = xyz["jrc_occurrence"][..., 3] >= 128  # water present >~50% of the time

for idx in sorted(glob.glob(os.path.join(BROWSE, "index_GCOV_*.json"))) if RADAR else []:
    items = json.load(open(idx))
    # Older PROVISIONAL KMLs need an Earthdata login; NISAR frames use a fixed grid, so borrow the
    # public KML of the same track+frame and let the registration absorb any small residual.
    frame_kml = {}
    for i in items:
        if i.get("kml"):
            frame_kml[tuple(i["scene"].split("_")[5:8])] = i["kml"]
    for item in items:
        if item["scene"].split("_")[2] == "PR" and "BETA" in item["png"]:
            continue
        item["kml"] = item.get("kml") or frame_kml.get(tuple(item["scene"].split("_")[5:8]))
        if not item["kml"]:
            continue
        big, mx, my = crop_browse(item, margin=0.03)
        (dy, dx), score = best_shift(big, water, mx, my)
        out = big[dy:dy + H, dx:dx + W]
        coverage = float((out.mean(axis=2) > 3).mean())
        track = item["scene"].split("_")[5]
        name = f"nisar_gcov_{item['date']}_t{track}"
        if coverage < 0.05:
            continue
        Image.fromarray(out).save(os.path.join(OUT, name + ".jpg"), quality=85)
        meta["radar"].append({"file": name + ".jpg", "date": item["date"], "track": track,
                              "scene": item["scene"], "shift_px": [dy - my, dx - mx],
                              "match": round(score, 3), "coverage": round(coverage, 3)})
        print(name, "shift", dy - my, dx - mx, "match", round(score, 3), "cov", round(coverage, 2))

json.dump(meta, open(os.path.join(OUT, f"{PREFIX}layers.json"), "w"), indent=1)
