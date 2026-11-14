"""Read a lon/lat window of NISAR GCOV (HH, HV backscatter) directly from ASF over HTTP range requests.

Needs an Earthdata login in ~/.netrc. Saves <out>/<scene>.npz with hh_db, hv_db (float16, dB), x, y (UTM), epsg.

Usage: uv run --with h5py --with fsspec --with aiohttp --with numpy --with pyproj python -I fetch_gcov_window.py \
         <out_dir> <lon0> <lat0> <lon1> <lat1> <url> [<url> ...]
"""
import os, sys, time

import fsspec, h5py, numpy as np, requests
from pyproj import Transformer

out = sys.argv[1]
lon0, lat0, lon1, lat1 = map(float, sys.argv[2:6])
urls = sys.argv[6:]
os.makedirs(out, exist_ok=True)
sess = requests.Session()  # uses ~/.netrc for the Earthdata OAuth redirect dance
fs = fsspec.filesystem("https", client_kwargs={"trust_env": True})


def signed_url(u):
    """Follow ASF -> Earthdata login -> signed S3/CloudFront URL (valid ~1 h)."""
    r = sess.get(u, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
    r.raise_for_status()
    r.close()
    return r.url

for u in urls:
    scene = os.path.basename(u).removesuffix(".h5")
    dst = os.path.join(out, scene + ".npz")
    if os.path.exists(dst):
        continue
    t = time.time()
    try:
        with fs.open(signed_url(u), block_size=2 * 2**20) as fo:
            g = h5py.File(fo, "r")["science/LSAR/GCOV/grids/frequencyA"]
            epsg = int(g["projection"].attrs["epsg_code"])
            x, y = g["xCoordinates"][:], g["yCoordinates"][:]
            tr = Transformer.from_crs(4326, epsg, always_xy=True)
            cx, cy = tr.transform([lon0, lon1, lon0, lon1], [lat0, lat0, lat1, lat1])
            c0, c1 = np.searchsorted(x, min(cx)), np.searchsorted(x, max(cx))
            ys = y[::-1] if y[0] > y[-1] else y  # y usually decreasing
            r0, r1 = np.searchsorted(ys, min(cy)), np.searchsorted(ys, max(cy))
            if y[0] > y[-1]:
                r0, r1 = len(y) - r1, len(y) - r0
            if c1 <= c0 or r1 <= r0:
                print("no overlap", scene); continue
            to_db = lambda a: (10 * np.log10(np.where(a > 0, a, np.nan))).astype(np.float16)
            hh = to_db(g["HHHH"][r0:r1, c0:c1])
            hv = to_db(g["HVHV"][r0:r1, c0:c1])
            np.savez_compressed(dst, hh_db=hh, hv_db=hv, x=x[c0:c1], y=y[r0:r1], epsg=epsg)
            print(scene, hh.shape, f"{time.time() - t:.0f}s", flush=True)
    except Exception as e:
        print("FAILED", scene, repr(e)[:200], flush=True)
