"""Two-date radar change composite: red = brighter on date A, cyan = brighter on date B.

Usage: uv run --with pillow --with numpy python -I make_composite.py <a.jpg> <b.jpg> <out.jpg>
"""
import sys
import numpy as np
from PIL import Image

a, b = (np.asarray(Image.open(p).convert("L")).astype(float) for p in sys.argv[1:3])
stretch = lambda x: np.clip((x - np.percentile(x, 2)) / (np.percentile(x, 98) - np.percentile(x, 2)), 0, 1)
a, b = stretch(a), stretch(b)
rgb = np.dstack([a, b, b])
Image.fromarray((rgb * 255).astype(np.uint8)).save(sys.argv[3], quality=88)
