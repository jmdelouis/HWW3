"""Maps of the difference between two WW3 runs on the same nodes: mean Hs of
run B, mean Hs B - A and RMS of the hourly Hs difference, after a spin-up.

usage: python plot_run_diff.py <a/ww3.202601.nc> <label a> <b/ww3.202601.nc> <label b> <grid/raw/tri.u32> <out.png> [spinup_days]

The fields are drawn on the triangles of the grid (linear interpolation).
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from netCDF4 import Dataset


def main():
    fa, la, fb, lb, ftri, out = sys.argv[1:7]
    spin = float(sys.argv[7]) if len(sys.argv) > 7 else 3.0
    a, b = Dataset(fa), Dataset(fb)
    lon = a["longitude"][:].astype(float)
    lat = a["latitude"][:].astype(float)
    if not (np.array_equal(lon, b["longitude"][:]) and np.array_equal(lat, b["latitude"][:])):
        sys.exit("the two runs are not on the same nodes")
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
