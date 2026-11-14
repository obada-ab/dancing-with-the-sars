# Dancing with the SARs

**Team:** `holy@hack:~$`

Our project for the [NASA Space Apps Challenge 2026 – Dancing with the SARs](https://www.spaceappschallenge.org/2026/challenges/dancing-with-the-sars/).

## Prototype: The Thirsty Lakes of Konin

A StoryMap-style page about shrinking lakes, lignite mining and drought around Konin (Wielkopolska, Poland), in three chapters:

1. **The past (1984–2024)**: Landsat 1991 vs 2024 slider, lake area per year (Landsat MNDWI), JRC water-change map.
2. **The turning point**: mines close and pits flood. Landsat region slider, Sentinel-2 pit slider, Sentinel-1 + NISAR pit water series.
3. **Now: NISAR (2026 →)**: full-resolution 20 m GCOV (HH/HV), per-pass water maps and areas, a season time-lapse, and SME2 soil-moisture maps and series.

```bash
cd site && python3 -m http.server 8765   # open http://127.0.0.1:8765
```

### Rebuilding the images

```bash
# 1. NISAR quick-look browse images from ASF (no login needed)
for t in 57 80 129; do python3 -I scripts/fetch_nisar_browse.py data/nisar_browse 18.2 52.4 GCOV $t 50; done
# 2. Co-registered layers (regional + close-ups)
uv run --with pillow --with numpy python -I scripts/build_layers.py data/nisar_browse data/tiles site/img
BBOX="17.98,52.43,18.22,52.54" ZOOM=13 PREFIX=lakes_ RADAR=0 uv run --with pillow --with numpy python -I scripts/build_layers.py data/nisar_browse data/tiles site/img
uv run --with pillow python -I -c "from PIL import Image as I; I.alpha_composite(I.open('site/img/lakes_s2_2024.jpg').convert('RGBA'), I.open('site/img/lakes_jrc_transitions.png').convert('RGBA')).convert('RGB').save('site/img/lakes_jrc_on_s2_2024.jpg')"
BBOX="18.08,52.36,18.26,52.45" ZOOM=13 PREFIX=pit_ RADAR=0 uv run --with pillow --with numpy python -I scripts/build_layers.py data/nisar_browse data/tiles site/img
# 3. Landsat past (Planetary Computer, no login): see scripts/landsat_past.py (area / rgb / search)
# 4. Sentinel-1 Jóźwin pit series (Planetary Computer, no login)
uv run --with numpy python -I scripts/s1_pit_area.py site/data/jozwin_s1_area.json
# 5. NISAR full data (needs Earthdata login in ~/.netrc)
#    GCOV: read only the Konin window over HTTP (each file is ~2 GB; this pulls ~35 MB)
uv run --with h5py --with fsspec --with aiohttp --with requests --with numpy --with pyproj python -I scripts/fetch_gcov_window.py data/nisar_gcov 17.70 52.15 18.65 52.70 <GCOV .h5 URLs from ASF search>
uv run --with numpy --with pillow --with pyproj python -I scripts/gcov_views.py data/nisar_gcov site
#    SME2: download the .h5 files (~130 MB each) into data/nisar_full, then
uv run --with h5py --with numpy --with pillow python -I scripts/sme2_maps.py data/nisar_full site/img/layers.json site/img site/data/sme2.json
```

### Data notes
- ASF has **216 NISAR products** over Konin (Nov 2025 – Nov 2026): GCOV backscatter, GUNW interferograms, SME2 soil moisture, GSLC and RSLC. There's a gap from Feb to May 2026.
- Browse PNGs are public. Full HDF5 products, and the KMLs for provisional products, need a NASA Earthdata login.
- Each NISAR frame is on a fixed grid, so one public `_LATLON.kml` per track/frame places every date correctly. `build_layers.py` checks this against the JRC water map, and every date needed zero shift.
- SME2 soil moisture is still provisional (200 m). Single 12-day GUNW pairs are dominated by atmospheric noise.
- In these provisional GCOV products, open water is close to the noise floor, and the floor differs by track and date. `gcov_views.py` therefore calibrates the water threshold on every pass, using permanent lake cores and dry land, and flags passes where dry fields look like water (`good: false`).
