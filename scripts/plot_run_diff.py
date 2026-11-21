"""Maps of the difference between two WW3 runs on the same nodes: mean Hs of
run B, mean Hs B - A and RMS of the hourly Hs difference, after a spin-up.

usage: python plot_run_diff.py <a/ww3.202601.nc> <label a> <b/ww3.202601.nc> <label b> <grid/raw/tri.u32> <out.png> [spinup_days]

The fields are drawn on the triangles of the grid (linear interpolation).
When the two runs are on different grids, ``DIFF_TRI_A=<a grid/raw/tri.u32>``
interpolates run A linearly on its own triangles to the nodes of run B (nearest
node of A for the few B nodes outside them) and the maps are on the grid of B.
"""

import os

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from netCDF4 import Dataset


def interpolator(lon_a, lat_a, ftri_a, lon_b, lat_b):
    """Linear interpolation of fields on the nodes of A (last axis) to the nodes of B."""
    from scipy.spatial import cKDTree
    tri = np.fromfile(ftri_a, "<u4").reshape(-1, 3).astype(np.int64)
    if tri.max() >= len(lon_a):
        sys.exit("DIFF_TRI_A does not match the nodes of run A")
    it = Triangulation(lon_a, lat_a, tri).get_trifinder()(lon_b, lat_b)
    inside = it >= 0
    v = tri[np.where(inside, it, 0)]
    x, y = lon_a[v], lat_a[v]
    det = (y[:, 1] - y[:, 2]) * (x[:, 0] - x[:, 2]) + (x[:, 2] - x[:, 1]) * (y[:, 0] - y[:, 2])
    w0 = ((y[:, 1] - y[:, 2]) * (lon_b - x[:, 2]) + (x[:, 2] - x[:, 1]) * (lat_b - y[:, 2])) / det
    w1 = ((y[:, 2] - y[:, 0]) * (lon_b - x[:, 2]) + (x[:, 0] - x[:, 2]) * (lat_b - y[:, 2])) / det
    w = np.stack([w0, w1, 1 - w0 - w1], 1)
    near = cKDTree(np.c_[lon_a, lat_a]).query(np.c_[lon_b[~inside], lat_b[~inside]])[1]
    v[~inside] = near[:, None]
    w[~inside] = [1.0, 0.0, 0.0]
    print("run A interpolated to %d nodes of run B (%d outside its triangles, nearest node)"
          % (len(lon_b), (~inside).sum()), flush=True)
    return lambda f: (f[:, v] * w).sum(-1)


def main():
    fa, la, fb, lb, ftri, out = sys.argv[1:7]
    spin = float(sys.argv[7]) if len(sys.argv) > 7 else 3.0
    a, b = Dataset(fa), Dataset(fb)
    lon = b["longitude"][:].astype(float)
    lat = b["latitude"][:].astype(float)
    to_b = None
    if not (np.array_equal(lon, a["longitude"][:]) and np.array_equal(lat, a["latitude"][:])):
        if not os.environ.get("DIFF_TRI_A"):
            sys.exit("the two runs are not on the same nodes (set DIFF_TRI_A to interpolate run A)")
        to_b = interpolator(a["longitude"][:].astype(float), a["latitude"][:].astype(float),
                            os.environ["DIFF_TRI_A"], lon, lat)
    t = a["time"][:]
    t = t - t[0]
    i0 = int(np.searchsorted(t, spin))
    kmax = min(len(t), len(b["time"]))  # common hours when run B is shorter
    nt = kmax - i0
    n = len(lon)
    sb = np.zeros(n)
    sd = np.zeros(n)
    sd2 = np.zeros(n)
    cnt = np.zeros(n)
    for k in range(i0, kmax, 24):
        k1 = min(k + 24, kmax)
        ha = a["hs"][k:k1].astype(float).filled(np.nan)
        if to_b is not None:
            ha = to_b(ha)
        hb = b["hs"][k:k1].astype(float).filled(np.nan)
        ok = np.isfinite(ha) & np.isfinite(hb)
        d = np.where(ok, hb - ha, 0.0)
        sb += np.where(ok, hb, 0.0).sum(0)
        sd += d.sum(0)
        sd2 += (d * d).sum(0)
        cnt += ok.sum(0)
    cnt = np.maximum(cnt, 1)
    mb, md, rd = sb / cnt, sd / cnt, np.sqrt(sd2 / cnt)
    print("%d nodes, %d hours: mean diff %+.4f m, |mean diff| p99 %.3f m, RMS diff median %.3f p99 %.3f max %.3f m"
          % (n, nt, md.mean(), np.percentile(abs(md), 99), np.median(rd), np.percentile(rd, 99), rd.max()))
    np.savez_compressed(out.rsplit(".", 1)[0] + ".npz", lon=lon, lat=lat, mean_b=mb, mean_diff=md, rms_diff=rd)

    tri = np.fromfile(ftri, "<u4").reshape(-1, 3).astype(np.int64)
    glon = np.fromfile(ftri.replace("tri.u32", "lon.f64"), "<f8")
    if tri.max() >= n or len(glon) != n or np.abs(glon - lon).max() > 1e-3:
        sys.exit("triangles do not match the run nodes")
    tr = Triangulation(lon, lat, tri)
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for ax, v, cmap, vmin, vmax, title in (
            (axs[0], mb, "viridis", 0, 6, "mean $H_s$ %s (m)" % lb),
            (axs[1], md, "RdBu_r", -0.1, 0.1, "mean $H_s$ %s $-$ %s (m)" % (lb, la)),
            (axs[2], rd, "magma_r", 0, 0.2, "RMS of hourly $H_s$ %s $-$ %s (m)" % (lb, la))):
        sc = ax.tripcolor(tr, v, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
        fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.set_aspect(1.3)
        ax.set_xlabel("longitude (°)")
    axs[0].set_ylabel("latitude (°)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
