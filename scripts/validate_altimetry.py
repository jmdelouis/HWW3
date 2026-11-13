"""Compare WW3 Hs on the HEALPix grid with along-track L3 altimetry.

Usage:
    python scripts/validate_altimetry.py RUN_NC NODES_NPZ OUT_PREFIX [L3_ROOT]

RUN_NC     ww3_ounf output (hourly ``hs`` on the nodes, ``time`` in days).
NODES_NPZ  ``grids/<name>/plot/nodes.npz`` (lon, lat, level of every node,
           same order as the WW3 nodes).
OUT_PREFIX prefix for ``_stats.json``, ``_scatter.png``, ``_maps.png``.
L3_ROOT    CMEMS WAVE_GLO_PHY_SWH_L3_NRT_014_001 directory (default: the
           Datarmor mirror).

Method: every 1 Hz altimeter sample (filtered, bias-corrected ``VAVH``) in
the run period and domain is matched to the nearest wet, unmasked,
ice-free node (3-D chord distance) if that node is closer than
``MAXDIST_FRAC`` times its HEALPix pixel size (``sqrt(pi/3) R / nside``,
51 km at nside 128, 3.2 km at nside 2048); samples in the first
``SPINUP_DAYS`` days (cold start) are skipped. The model Hs is interpolated
linearly in time between the two bracketing hourly fields. Matched pairs are then averaged in blocks of ``BLOCK`` consecutive
samples of the same file (about 7 km per sample, so about 50 km), which
removes most of the 1 Hz altimeter noise. Statistics are given globally,
per mission and per HEALPix level of the matched node.
"""
import glob
import json
import os
import sys

import matplotlib
import netCDF4
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402

MAXDIST_FRAC = 0.75
SPINUP_DAYS = 3.0
BLOCK = 7
EARTH_KM = 6371.0
L3_DEFAULT = "/home/ref-cmems-public/tac/wave/WAVE_GLO_PHY_SWH_L3_NRT_014_001"


def xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def stats(o, m):
    if len(o) < 2:
        return {"n": int(len(o))}
    d = m - o
    return {
        "n": int(len(o)),
        "obs_mean": float(o.mean()),
        "bias": float(d.mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "si": float(np.sqrt(((d - d.mean()) ** 2).mean()) / o.mean()),
        "r": float(np.corrcoef(o, m)[0, 1]),
        "slope": float((o * m).sum() / (o * o).sum()),
    }


def main():
    run_nc, nodes_npz, out = sys.argv[1:4]
    l3 = sys.argv[4] if len(sys.argv) > 4 else L3_DEFAULT

    d = netCDF4.Dataset(run_nc)
    lon = np.asarray(d["longitude"][:], float)
    lat = np.asarray(d["latitude"][:], float)
    tri = np.asarray(d["tri"][:]) - 1
    tunits = d["time"].units
    tm = netCDF4.date2num(netCDF4.num2date(d["time"][:], tunits), "seconds since 2000-01-01")
    print("reading hs", flush=True)
    hs = d["hs"][:].astype(np.float32)  # (nt, nnode), masked on land/ice
    mask = np.ma.getmaskarray(hs)
    hs = np.ma.filled(hs, np.nan)
    nodes = np.load(nodes_npz)
    assert np.allclose(nodes["lon"], lon, atol=1e-4) and np.allclose(nodes["lat"], lat, atol=1e-4)
    level = nodes["level"]

    ok = ~mask.all(0)
    if "ice" in d.variables:
        ice = np.ma.filled(d["ice"][:].max(0), 0.0)
        ok &= ice <= 0.0
    idx = np.nonzero(ok)[0]
    tree = cKDTree(xyz(lon[idx], lat[idx]))
    # Max chord distance per node: MAXDIST_FRAC x pixel size at its level.
    maxchord = MAXDIST_FRAC * np.sqrt(np.pi / 3) / 2.0 ** level
    chord = maxchord.max()
    lonmin, lonmax, latmin, latmax = lon.min(), lon.max(), lat.min(), lat.max()

    t0, t1 = tm[0] + SPINUP_DAYS * 86400.0, tm[-1]
    rows = []  # obs, model, mission, level, lon, lat
    days = sorted({netCDF4.num2date(t, "seconds since 2000-01-01").strftime("%Y/%m") for t in (t0, t1)})
    files = []
    for ym in days:
        files += glob.glob(os.path.join(l3, "*", ym, "*.nc"))
    print(len(files), "L3 files", flush=True)
    for f in sorted(files):
        mission = f.split("swh_nrt_")[1].split("-l3")[0]
        with netCDF4.Dataset(f) as a:
            t = a["time"][:].astype(float)
            la = np.ma.filled(a["latitude"][:].astype(float), np.nan)
            lo = np.ma.filled(a["longitude"][:].astype(float), np.nan)
            h = np.ma.filled(a["VAVH"][:].astype(float), np.nan)
        lo = np.where(lo > 180, lo - 360, lo)
        s = (t >= t0) & (t <= t1) & np.isfinite(h) & (la >= latmin) & (la <= latmax) & (lo >= lonmin) & (lo <= lonmax)
        if s.sum() < BLOCK:
            continue
        t, la, lo, h = t[s], la[s], lo[s], h[s]
        dist, j = tree.query(xyz(lo, la), distance_upper_bound=chord)
        s = np.isfinite(dist)
        s[s] = dist[s] <= maxchord[idx[j[s]]]
        if s.sum() < BLOCK:
            continue
        t, la, lo, h, n = t[s], la[s], lo[s], h[s], idx[j[s]]
        k = np.clip(np.searchsorted(tm, t) - 1, 0, len(tm) - 2)
        w = (t - tm[k]) / (tm[k + 1] - tm[k])
        m = (1 - w) * hs[k, n] + w * hs[k + 1, n]
        s = np.isfinite(m)
        t, la, lo, h, n, m = t[s], la[s], lo[s], h[s], n[s], m[s]
        nb = len(h) // BLOCK
        if nb == 0:
            continue
        cut = nb * BLOCK
        # Keep only blocks of consecutive samples (no gap > 3 s).
        tb = t[:cut].reshape(nb, BLOCK)
        good = (np.diff(tb, axis=1) < 3.0).all(1)
        r = lambda x: x[:cut].reshape(nb, BLOCK)  # noqa: E731
        lvl = np.round(np.median(r(level[n]), 1)).astype(int)
        blocks = np.stack([r(h).mean(1), r(m).mean(1), lvl, r(lo).mean(1), r(la).mean(1), r(t).mean(1)], 1)
        rows.append((mission, blocks[good]))
    missions = sorted({r[0] for r in rows})
    allr = np.concatenate([r[1] for r in rows])
    mis = np.concatenate([[r[0]] * len(r[1]) for r in rows])
    o, m, lvl, blo, bla = allr[:, 0], allr[:, 1], allr[:, 2].astype(int), allr[:, 3], allr[:, 4]
    day = ((allr[:, 5] - tm[0]) // 86400).astype(int) + 1

    res = {"run": run_nc, "maxdist_frac": MAXDIST_FRAC, "spinup_days": SPINUP_DAYS, "block": BLOCK,
           "all": stats(o, m),
           "missions": {k: stats(o[mis == k], m[mis == k]) for k in missions},
           "levels": {str(int(lv)): stats(o[lvl == lv], m[lvl == lv]) for lv in np.unique(lvl)}}
    # Regions: open Atlantic, North Sea, Mediterranean.
    ns = (bla > 51) & (bla < 61) & (blo > -4) & (blo < 10)
    med = (bla > 30) & (bla < 46) & (blo > -5.5) & (blo < 37)
    res["regions"] = {"north_sea": stats(o[ns], m[ns]), "mediterranean": stats(o[med], m[med]),
                      "other": stats(o[~ns & ~med], m[~ns & ~med])}
    res["days"] = {str(k): stats(o[day == k], m[day == k]) for k in np.unique(day)}
    with open(out + "_stats.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res["all"], indent=1))

    # Scatter (density) and bias by level.
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    hb = ax[0].hexbin(o, m, gridsize=80, extent=(0, 12, 0, 12), mincnt=1, bins="log", cmap="viridis")
    ax[0].plot([0, 12], [0, 12], "k--", lw=0.8)
    a = res["all"]
    ax[0].set_title(f"Hs, N={a['n']}, bias={a['bias']:.2f} m, RMSE={a['rmse']:.2f} m, SI={a['si']:.2f}, r={a['r']:.3f}", fontsize=9)
    ax[0].set_xlabel("altimeter Hs (m)"); ax[0].set_ylabel("WW3 Hs (m)")
    fig.colorbar(hb, ax=ax[0], label="count")
    lv = sorted(res["levels"], key=int)
    ns_ = [2 ** int(k) for k in lv]
    ax[1].bar(range(len(lv)), [res["levels"][k].get("bias", np.nan) for k in lv], color="C0", label="bias")
    ax[1].plot(range(len(lv)), [res["levels"][k].get("rmse", np.nan) for k in lv], "o-", color="C3", label="RMSE")
    ax[1].set_xticks(range(len(lv)), [f"{n}\n({res['levels'][k]['n']})" for n, k in zip(ns_, lv)])
    ax[1].set_xlabel("nside of matched node (N)"); ax[1].set_ylabel("m"); ax[1].legend()
    ax[1].axhline(0, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig(out + "_scatter.png", dpi=130)

    # Maps: monthly mean and max Hs, mean bias at block positions.
    T = mtri.Triangulation(lon, lat, tri)
    mean = np.nanmean(hs, 0); mx = np.nanmax(hs, 0)
    mean = np.where(np.isfinite(mean), mean, 0); mx = np.where(np.isfinite(mx), mx, 0)
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    for a_, f_, tt, vm in [(ax[0], mean, "mean Hs (m)", 6), (ax[1], mx, "max Hs (m)", 16)]:
        p = a_.tripcolor(T, f_, shading="gouraud", cmap="viridis", vmin=0, vmax=vm)
        fig.colorbar(p, ax=a_, shrink=0.7, label=tt)
        a_.set_title(tt + ", January 2026")
    sc = ax[2].scatter(blo, bla, c=m - o, s=1, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    fig.colorbar(sc, ax=ax[2], shrink=0.7, label="WW3 - altimeter (m)")
    ax[2].set_title("Hs difference along tracks (50 km blocks)")
    for a_ in ax:
        a_.set_xlim(lonmin, lonmax); a_.set_ylim(latmin, latmax); a_.set_aspect(1.4)
    fig.tight_layout(); fig.savefig(out + "_maps.png", dpi=110)


if __name__ == "__main__":
    main()
