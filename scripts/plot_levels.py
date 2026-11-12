"""Map of the nodes of an hww3-grid mesh coloured by HEALPix nside, from
`extract_zoom.py <grid_dir> nodes.npz --nodes`.

usage: python plot_levels.py <nodes.npz> <out.png>
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    z = np.load(sys.argv[1])
    out = sys.argv[2]
    lon, lat, level, obc = z["lon"], z["lat"], z["level"], z["obc"]
    s = json.loads(str(z["summary"]))
    levels = np.unique(level)
    cmap = plt.get_cmap("viridis", len(levels))
    fig, ax = plt.subplots(figsize=(11, 7))
    # coarse levels first so fine nodes stay visible
    for i, l in enumerate(levels):
        m = level == l
        ax.scatter(lon[m], lat[m], s=0.05 * 4.0 ** (levels[-1] - l), c=[cmap(i)], marker="s",
                   linewidths=0, label=f"nside {2**int(l)}: {m.sum():,} nodes", rasterized=True)
    ax.scatter(lon[obc], lat[obc], s=1, c="r", label=f"open boundary: {obc.size:,}", rasterized=True)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(40)))
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title(f"{s['name']}: {s['n_nodes']:,} nodes, {s['mesh']['n_triangles']:,} triangles")
    ax.legend(loc="lower left", fontsize=8, markerscale=6)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main()
