"""Zoom on an hww3-grid mesh: HEALPix cell boundaries (blue) and WW3
triangles (red), from a subset written by extract_zoom.py.

usage: python plot_cells_tris.py <zoom.npz> <out.png>
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection


def main():
    z = np.load(sys.argv[1])
    out = sys.argv[2]
    lon, lat, level, tri = z["lon"], z["lat"], z["level"], z["tri"]
    x0, x1, y0, y1 = z["bbox"]
    s = json.loads(str(z["summary"]))
    cells = np.stack([z["cell_lon"], z["cell_lat"]], axis=-1)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.add_collection(PolyCollection(cells, facecolors="none", edgecolors="tab:blue", linewidths=0.6))
    ax.add_collection(PolyCollection(np.stack([lon[tri], lat[tri]], axis=-1), facecolors="none",
                                     edgecolors="tab:red", linewidths=0.4))
    ax.plot(lon, lat, "k.", ms=1.5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(0.5 * (y0 + y1))))
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    inside = (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
    lv = np.unique(level[inside])
    ax.set_title(f"{s['name']}: HEALPix cells (blue), WW3 triangles (red), nodes (black)\n"
                 f"nside {', '.join(str(2**int(l)) for l in lv)} in view")
    fig.tight_layout()
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
