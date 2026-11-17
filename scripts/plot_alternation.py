"""Side-by-side zooms of two hww3-grid meshes on the same nodes: HEALPix cell
boundaries (blue) and WW3 triangles (red), from subsets written by
extract_zoom.py.

usage: python plot_alternation.py <a.npz> <label a> <b.npz> <label b> <out.png>
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection


def panel(ax, path, label):
    z = np.load(path)
    lon, lat, tri = z["lon"], z["lat"], z["tri"]
    x0, x1, y0, y1 = z["bbox"]
    cells = np.stack([z["cell_lon"], z["cell_lat"]], axis=-1)
    ax.add_collection(PolyCollection(cells, facecolors="none", edgecolors="tab:blue", linewidths=0.5))
    ax.add_collection(PolyCollection(np.stack([lon[tri], lat[tri]], axis=-1), facecolors="none",
                                     edgecolors="tab:red", linewidths=0.5))
    ax.plot(lon, lat, "k.", ms=2)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1.0 / np.cos(np.radians(0.5 * (y0 + y1))))
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("longitude (°)")


def main():
    a, la, b, lb, out = sys.argv[1:6]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    panel(axs[0], a, la)
    panel(axs[1], b, lb)
    axs[0].set_ylabel("latitude (°)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main()
