"""Plot an hww3-grid mesh (raw arrays) coloured by HEALPix level.

usage: python plot_mesh.py <grid_output_dir> <out.png> [lon0 lon1 lat0 lat1]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection


def load(d):
    raw = Path(d) / "raw"
    lon = np.fromfile(raw / "lon.f64", dtype="<f8")
    lat = np.fromfile(raw / "lat.f64", dtype="<f8")
    level = np.fromfile(raw / "level.u8", dtype="u1")
    tri = np.fromfile(raw / "tri.u32", dtype="<u4").reshape(-1, 3)
    obc = np.fromfile(raw / "obc.u32", dtype="<u4")
    meta = json.loads((Path(d) / "meta.json").read_text())
    return lon, lat, level, tri, obc, meta


def main():
    d, out = sys.argv[1], sys.argv[2]
    lon, lat, level, tri, obc, meta = load(d)
    if len(sys.argv) == 7:
        x0, x1, y0, y1 = (float(v) for v in sys.argv[3:7])
        keep = ((lon[tri] >= x0 - 1) & (lon[tri] <= x1 + 1) & (lat[tri] >= y0 - 1) & (lat[tri] <= y1 + 1)).all(axis=1)
        tri = tri[keep]
        obc = obc[(lon[obc] >= x0) & (lon[obc] <= x1) & (lat[obc] >= y0) & (lat[obc] <= y1)]
    tl = level[tri].max(axis=1)
    levels = np.unique(level)
    cmap = plt.get_cmap("viridis", len(levels))
    colors = cmap(np.searchsorted(levels, tl))
    polys = np.stack([lon[tri], lat[tri]], axis=-1)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors="k", linewidths=0.15))
    ax.plot(lon[obc], lat[obc], "r.", ms=2, label="open boundary")
    if len(sys.argv) == 7:
        ax.set_xlim(float(sys.argv[3]), float(sys.argv[4]))
        ax.set_ylim(float(sys.argv[5]), float(sys.argv[6]))
    else:
        ax.autoscale_view()
    ax.set_aspect(1.0 / np.cos(np.deg2rad(lat.mean())))
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    s = meta["summary"]
    ax.set_title(f"{s['name']}: {s['n_nodes']} nodes, {s['mesh']['n_triangles']} triangles")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(levels))]
    ax.legend(handles + [ax.lines[0]], [f"nside {2**int(l)}" for l in levels] + ["open boundary"],
              loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main()
