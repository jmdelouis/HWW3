"""Figures of scripts/explicit_cost.py.

usage:
  python plot_explicit_cost.py map <grid dir> <prefix> <out.png> [lon0 lon1 lat0 lat1 ...]
      local number of explicit sub-steps per node (max over the spectral
      components) with the binding nodes, on the whole domain and in zooms
  python plot_explicit_cost.py curves <out.png> <prefix a> [<prefix b> ...]
      ITER per frequency, and the propagation sweeps S when the worst nodes
      are relieved or moved one HEALPix level coarser
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.tri import Triangulation


def square(ax, box):
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect(1 / np.cos(np.radians(0.5 * (box[2] + box[3]))))


def do_map(gdir, prefix, out, boxes):
    raw = gdir + "/raw/"
    lon = np.fromfile(raw + "lon.f64")
    lat = np.fromfile(raw + "lat.f64")
    tri = np.fromfile(raw + "tri.u32", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    level = np.fromfile(raw + "level.u8", dtype=np.uint8).astype(int)
    z = np.load(prefix + ".npz")
    res = json.load(open(prefix + ".json"))
    # nodes counted in ITER: every wet node, or the interior ones with
    # --interior-only (files written before that option: interior only)
    interior = z["counted"] if "counted" in z else z["interior"]
    req = np.where(interior, z["reqmax"].astype(float), 0.0)
    worst = float(req.max())
    it1 = int(np.ceil(worst - 1e-9))
    # discrete colours: local sub-steps 1 .. ceil(max)
    bounds = np.concatenate([[0], np.arange(1, it1 + 1) + 1e-9])
    cmap = ListedColormap(plt.get_cmap("turbo")(np.linspace(0.08, 0.95, len(bounds) - 1)))
    norm = BoundaryNorm(bounds, cmap.N)
    fig = plt.figure(figsize=(17, 10.5))
    # whole domain: every node coloured, the costly ones drawn last
    ax = fig.add_axes([0.03, 0.44, 0.52, 0.50])
    o = np.argsort(req)
    ax.scatter(lon[o], lat[o], c=req[o], s=np.where(req[o] > 4, 1.5, 0.3), cmap=cmap, norm=norm,
               linewidths=0, rasterized=True)
    hot = req > it1 - 3
    ax.scatter(lon[hot], lat[hot], s=14, facecolors="none", edgecolors="k", linewidths=0.5)
    square(ax, [lon.min(), lon.max(), lat.min(), lat.max()])
    ax.set_title(f"{res['label']}: sub-steps per 600 s that each node alone would impose (max over the components)\n"
                 f"WW3 sub-cycles the whole domain at the worst node: ITER = {it1} at {res['freqs'][0]:.4f} Hz, "
                 f"S = {res['S']} sweeps per step (1 per component: {res['S_min']})\n"
                 f"circles: the {int(hot.sum())} nodes above {it1 - 3}", fontsize=9)
    for k, bx in enumerate(boxes):
        ax.add_patch(plt.Rectangle((bx[0], bx[2]), bx[1] - bx[0], bx[3] - bx[2], fill=False, ec="m", lw=1.2))
        ax.text(bx[1] + 0.8, bx[2], "ABCDEFGH"[k], color="m", fontsize=12, weight="bold")
    # histogram per nside
    ah = fig.add_axes([0.66, 0.47, 0.32, 0.45])
    bins = np.linspace(0, it1, 8 * it1 + 1)
    for d in sorted(set(level[interior])):
        m = interior & (level == d)
        ah.hist(req[m], bins=bins, histtype="step", lw=1.4,
                label=f"nside {2 ** d} ({m.sum():,} nodes, median {np.median(req[m]):.1f})")
    ah.set_yscale("log")
    ah.set_xlabel("local sub-steps per 600 s (max over the components)")
    ah.set_ylabel("nodes counted in ITER")
    for k in range(1, it1 + 1):
        ah.axvline(k, color="0.85", lw=0.6, zorder=0)
    ah.legend(fontsize=8, loc="upper right")
    ah.set_title("the global ITER is set by the tail of the nside-2048 nodes", fontsize=9)
    # zooms with the triangles
    nz = len(boxes)
    w = 0.94 / max(nz, 1)
    for k, bx in enumerate(boxes):
        a = fig.add_axes([0.03 + k * w, 0.04, w - 0.03, 0.33])
        inb = ((lon[tri] > bx[0] - 0.5) & (lon[tri] < bx[1] + 0.5) & (lat[tri] > bx[2] - 0.5)
               & (lat[tri] < bx[3] + 0.5)).all(1)
        t = tri[inb]
        if len(t):
            tr = Triangulation(lon, lat, t)
            v = np.ma.masked_less_equal(req[t].max(1), 0)
            a.tripcolor(tr, v, cmap=cmap, norm=norm, shading="flat", rasterized=True)
            a.triplot(tr, color="k", lw=0.08, alpha=0.5)
        hb = hot & (lon > bx[0]) & (lon < bx[1]) & (lat > bx[2]) & (lat < bx[3])
        a.scatter(lon[hb], lat[hb], s=18, facecolors="none", edgecolors="k", linewidths=0.7)
        square(a, bx)
        a.tick_params(labelsize=7)
        sel = (lon > bx[0]) & (lon < bx[1]) & (lat > bx[2]) & (lat < bx[3])
        a.set_title(f"{'ABCDEFGH'[k]}: max {req[sel].max():.1f}, {int(hb.sum())} circled", fontsize=9)
    cb = fig.add_axes([0.565, 0.47, 0.01, 0.42])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, cax=cb, orientation="vertical", ticks=0.5 * (bounds[:-1] + bounds[1:]),
                 format=matplotlib.ticker.FuncFormatter(lambda x, p: str(int(np.ceil(x)))))
    cb.tick_params(labelsize=8)
    cb.set_ylabel("local sub-steps (ceil)", fontsize=8)
    fig.savefig(out, dpi=150)
    print("wrote", out)


def do_curves(out, prefixes):
    rs = [json.load(open(p + ".json")) for p in prefixes]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for k, r in enumerate(rs):
        c = f"C{k}"
        f = np.array(r["freqs"])
        axs[0].step(f, r["iter_max_by_freq"], where="mid", color=c, label=f"{r['label']} max over directions")
        axs[0].plot(f, r["iter_mean_by_freq"], "--", color=c, label=f"{r['label']} mean")
        if "with_currents_upper_bound" in r:
            axs[0].plot(f, r["with_currents_upper_bound"]["iter_max_by_freq"], ":", color=c,
                        label=f"{r['label']} + max current (bound)")
        n = np.array(r["removal"]["N"])
        axs[1].plot(np.maximum(n, 0.5), r["removal"]["S"], "o-", ms=3, color=c, label=r["label"])
        axs[2].plot(r["coarsen"]["nodes"], r["coarsen"]["S"], "o-", ms=3, color=c, label=r["label"])
        for K, x, y in zip(r["coarsen"]["K"], r["coarsen"]["nodes"], r["coarsen"]["S"]):
            axs[2].annotate(f">{K}", (x, y), fontsize=7, color=c)
        axs[2].axhline(r["S_geometry_fixed"], color=c, lw=0.8, ls="--",
                       label=f"{r['label']}, CFL length at the median of each level")
    axs[1].axhline(rs[0]["S_min"], color="gray", lw=0.8)
    axs[1].text(1, rs[0]["S_min"] + 30, f"one sweep per component ({rs[0]['S_min']})", fontsize=7, color="gray")
    axs[0].set_xscale("log")
    axs[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[0].set_xticks([0.04, 0.06, 0.1, 0.2, 0.4, 0.7], ["0.04", "0.06", "0.1", "0.2", "0.4", "0.7"])
    axs[0].set_xlabel("frequency (Hz)")
    axs[0].set_ylabel("ITER (sweeps per 600 s)")
    axs[0].legend(fontsize=7)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("worst nodes relieved (ranked by local sub-steps)")
    axs[1].set_ylabel("S = sum of ITER over the components")
    axs[1].legend(fontsize=7)
    axs[2].set_xscale("log")
    axs[2].set_xlabel("nodes moved one HEALPix level coarser (local sub-steps > K)")
    axs[2].set_ylabel("S")
    axs[2].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    if sys.argv[1] == "map":
        b = [float(x) for x in sys.argv[5:]]
        do_map(sys.argv[2], sys.argv[3], sys.argv[4], [b[i : i + 4] for i in range(0, len(b), 4)])
    else:
        do_curves(sys.argv[2], sys.argv[3:])
