"""Read a lon/lat window of NISAR GUNW (unwrapped interferogram, 80 m) over HTTP range requests.

Needs an Earthdata login in ~/.netrc. Saves <out>/<scene>.npz with unw (rad), coh, cc (connected components),
iono (rad), x, y (UTM), epsg, ref_date, sec_date.

Usage: uv run --with h5py --with fsspec --with aiohttp --with requests --with numpy --with pyproj python -I \
         fetch_gunw_window.py <out_dir> <lon0> <lat0> <lon1> <lat1> <url> [<url> ...]
"""
import os, re, sys, time

import fsspec, h5py, numpy as np, requests
from pyproj import Transformer

out = sys.argv[1]
lon0, lat0, lon1, lat1 = map(float, sys.argv[2:6])
os.makedirs(out, exist_ok=True)
sess = requests.Session()
fs = fsspec.filesystem("https", client_kwargs={"trust_env": True})


def signed_url(u):
    r = sess.get(u, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
    r.raise_for_status(); r.close()
    return r.url


for u in sys.argv[6:]:
    scene = os.path.basename(u).removesuffix(".h5")
    dst = os.path.join(out, scene + ".npz")
    if os.path.exists(dst):
        continue
    d = re.findall(r"_(\d{8})T\d{6}", scene)
    t = time.time()
    try:
        with fs.open(signed_url(u), block_size=2 * 2**20) as fo:
            g = h5py.File(fo, "r")["science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"]
            epsg = int(g["projection"].attrs["epsg_code"])
            x, y = g["xCoordinates"][:], g["yCoordinates"][:]
            tr = Transformer.from_crs(4326, epsg, always_xy=True)
            cx, cy = tr.transform([lon0, lon1, lon0, lon1], [lat0, lat0, lat1, lat1])
            c0, c1 = np.searchsorted(x, min(cx)), np.searchsorted(x, max(cx))
            if y[0] > y[-1]:
                r0, r1 = np.searchsorted(-y, -max(cy)), np.searchsorted(-y, -min(cy))
            else:
                r0, r1 = np.searchsorted(y, min(cy)), np.searchsorted(y, max(cy))
            pol = "HH" if "HH" in g else list(k for k in g.keys() if len(k) == 2)[0]
            h = g[pol]
            sl = (slice(r0, r1), slice(c0, c1))
            np.savez_compressed(dst, unw=h["unwrappedPhase"][sl], coh=h["coherenceMagnitude"][sl],
                                cc=h["connectedComponents"][sl], iono=h["ionospherePhaseScreen"][sl],
                                x=x[c0:c1], y=y[r0:r1], epsg=epsg, ref_date=d[0], sec_date=d[2] if len(d) > 2 else d[1])
            print(scene[:60], (r1 - r0, c1 - c0), f"{time.time() - t:.0f}s", flush=True)
    except Exception as e:
        print("FAILED", scene[:60], repr(e)[:200], flush=True)
