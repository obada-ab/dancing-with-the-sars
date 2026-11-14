"""Download NISAR browse PNGs + KML footprints for an area from ASF (no login needed).

Usage: python3 -I fetch_nisar_browse.py <outdir> <lon> <lat> <level> <track> [max]
"""
import json, os, sys, urllib.request

out, lon, lat, level, track = sys.argv[1:6]
mx = int(sys.argv[6]) if len(sys.argv) > 6 else 50
q = (f"https://api.daac.asf.alaska.edu/services/search/param?platform=NISAR"
     f"&processingLevel={level}&relativeOrbit={track}"
     f"&intersectsWith=POINT({lon}%20{lat})&output=geojson&maxResults={mx}")
feats = json.load(urllib.request.urlopen(q))["features"]
os.makedirs(out, exist_ok=True)
index = []
for f in sorted(feats, key=lambda f: f["properties"]["startTime"]):
    p = f["properties"]
    name = p["sceneName"]
    allb = [b for b in (p.get("browse") or []) if b.endswith(".png") and "thumbnail" not in b]
    br = [b for b in allb if b.endswith("_LATLON.png")] or allb[:1]
    if not br:
        continue
    dst = os.path.join(out, name + ".png")
    if not os.path.exists(dst):
        urllib.request.urlretrieve(br[0], dst)
    kml = next((u for u in (p.get("additionalUrls") or []) if u.endswith("_LATLON.kml")), br[0].replace(".png", ".kml"))
    kdst = os.path.join(out, name + ".kml")
    try:
        if not os.path.exists(kdst):
            urllib.request.urlretrieve(kml, kdst)
    except Exception:
        kdst = None
    index.append({"scene": name, "date": p["startTime"][:10], "png": dst, "kml": kdst,
                  "footprint": f["geometry"]["coordinates"][0]})
    print(p["startTime"][:10], name)
json.dump(index, open(os.path.join(out, f"index_{level}_{track}.json"), "w"), indent=1)
