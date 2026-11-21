"""Map of the finest cells of grid A that are coarser in other grids, for
grids built from the same configuration with a different explicit-scheme cap.

usage: python plot_level_change.py <out.png> <grid A> <label A> <grid B> <label B> [<grid C> <label C> ...]
          [--buoys buoys_stats.json RUN_A RUN_B RUN_C ...] [--box LON0 LON1 LAT0 LAT1]

Every node of A at its finest level is drawn grey when the same HEALPix cell is
a node of B, red when its parent is, black when an ancestor further up is (or
the cell is no longer wet). With ``--buoys``, the buoys of a
``validate_buoys.py`` statistics file are drawn as circles whose colour is the
change of their Hs RMSE from run RUN_A to the run of each grid. ``--box``
restricts the maps to a longitude-latitude box.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(grid):
    raw = os.path.join(grid, "raw")
    uniq = np.fromfile(os.path.join(raw, "uniq.u64"), "<u8")
    lon = np.fromfile(os.path.join(raw, "lon.f64"), "<f8")
    lat = np.fromfile(os.path.join(raw, "lat.f64"), "<f8")
    assert len(uniq) == len(lon) == len(lat), "uniq.u64 does not match the nodes"
    # cdshealpix NUNIQ: 16 * 4^depth + hash
    depth = ((np.floor(np.log2(uniq.astype(np.float64))).astype(np.int64)) - 4) // 2
    hsh = uniq - (np.uint64(16) << (2 * depth).astype(np.uint64))
    return {"uniq": uniq, "lon": lon, "lat": lat, "depth": depth, "hash": hsh}


def to_uniq(depth, hsh):
    return (np.uint64(16) << np.uint64(2 * depth)) | hsh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("grids", nargs="+")
    ap.add_argument("--buoys", nargs="+")
    ap.add_argument("--box", nargs=4, type=float, default=[-180, 180, -90, 90])
    a = ap.parse_args()
    paths, labels = a.grids[0::2], a.grids[1::2]
    ga = load(paths[0])
    dmax = ga["depth"].max()
    x0, x1, y0, y1 = a.box
    inbox = (ga["lon"] >= x0) & (ga["lon"] <= x1) & (ga["lat"] >= y0) & (ga["lat"] <= y1)
    fine = (ga["depth"] == dmax) & inbox
    # marker area of about one cell at this level
    ms = max(0.05, 0.2 * (60.0 / (x1 - x0)) ** 2) if x1 - x0 < 300 else 0.05
    print(f"{labels[0]}: {fine.sum():,} nodes at depth {dmax}", flush=True)
    stats = None
    if a.buoys:
        import json
        stats = json.load(open(a.buoys[0]))
        runs = a.buoys[1:]
    nb = len(paths) - 1
    w = (min(x1, ga["lon"].max()) - max(x0, ga["lon"].min())) / (min(y1, ga["lat"].max()) - max(y0, ga["lat"].min()))
    w *= np.cos(np.deg2rad(0.5 * (max(y0, ga["lat"].min()) + min(y1, ga["lat"].max()))))
    fig, axs = plt.subplots(1, nb, figsize=((6.5 * w + 2.5) * nb, 7.5), squeeze=False)
    for i in range(nb):
        gb = load(paths[i + 1])
        h = ga["hash"][fine]
        same = np.isin(ga["uniq"][fine], gb["uniq"])
        parent = np.isin(to_uniq(dmax - 1, h >> np.uint64(2)), gb["uniq"])
        parent &= ~same
        other = ~same & ~parent
        lon, lat = ga["lon"][fine], ga["lat"][fine]
        ax = axs[0, i]
        ax.scatter(lon[same], lat[same], s=ms, c="0.75", marker="s", linewidths=0, rasterized=True,
                   label=f"kept: {same.sum():,}")
        ax.scatter(lon[parent], lat[parent], s=ms, c="tab:red", marker="s", linewidths=0, rasterized=True,
                   label=f"one level coarser: {parent.sum():,}")
        ax.scatter(lon[other], lat[other], s=ms, c="k", marker="s", linewidths=0, rasterized=True,
                   label=f"coarser or dry: {other.sum():,}")
        print(f"{labels[i + 1]}: kept {same.sum():,}, parent {parent.sum():,}, other {other.sum():,}", flush=True)
        title = f"{labels[i + 1]}: cells of {labels[0]} at $N_{{side}}={2**dmax}$"
        if stats is not None:
            d = np.array([b[runs[i + 1]]["rmse"] - b[runs[0]]["rmse"] for b in stats["per_buoy"]])
            bl = np.array([b["lon"] for b in stats["per_buoy"]])
            bt = np.array([b["lat"] for b in stats["per_buoy"]])
            ib = (bl >= x0) & (bl <= x1) & (bt >= y0) & (bt <= y1)
            d, bl, bt = d[ib], bl[ib], bt[ib]
            o = np.argsort(abs(d))
            sc = ax.scatter(bl[o], bt[o], s=30, c=d[o], cmap="RdBu_r", vmin=-0.2, vmax=0.2,
                            edgecolors="k", linewidths=0.3, zorder=5)
            fig.colorbar(sc, ax=ax, shrink=0.7, label=f"buoy $H_s$ RMSE {runs[i + 1]} $-$ {runs[0]} (m)")
            title += f"\n{len(d)} buoys: {(d > 0.005).sum()} worse, {(d < -0.005).sum()} better by more than 5 mm"
        ax.set_title(title, fontsize=10)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(0.5 * (max(y0, lat.min()) + min(y1, lat.max())))))
        ax.set_xlabel("longitude (°)")
        ax.legend(loc="lower left", fontsize=8, markerscale=max(1.0, 12 / np.sqrt(ms)))
    axs[0, 0].set_ylabel("latitude (°)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
