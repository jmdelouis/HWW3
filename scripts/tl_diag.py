"""Where a tangent run grows: hourly statistics of the tangent of H_s.

    python scripts/tl_diag.py RUN [RUN ...] [--grid DIR] [--top 10] [--json OUT] [--op OP.npz] [--bound 10]

For each tangent run directory (dhs_YYYYMMDDHH.f64 of hww3-prop --tangent
--out64), hour by hour: the number of non-finite values, the largest
|dH_s| and its node (longitude, latitude, depth), the 99.99th percentile
and the RMS over the finite values; then the nodes where |dH_s| first
exceeds 100 times the 99.99th percentile of its hour, with the hour, and
the H_s of the run there (hs_ files of the tangent run). A tangent that
stays linear in a stable model has no such nodes. The undefined nodes
(|dH_s| = 999.9 on land, dry and ice nodes, where H_s is undefined) are
left out.

With --op (an operator of cal_obs.py), the derivatives of its altimeter
blocks and buoy hours along each tangent: how many are not finite or above
BOUND metres per unit of the parameter (no physical change of H_s is that
large; cal_obs.py turns tangent values below -900 into NaN, as it does the
undefined value), the largest, and where and when.
"""
import sys
import argparse
import glob
import json
import os

import numpy as np


def op_check(opf, d, bound, lon, lat):
    """Derivatives of the observations of operator OPF along tangent run D."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cal_obs
    o = np.load(opf)
    tags = [str(t) for t in o["tags"]]
    da, dh, _ = cal_obs.apply(o, d, prefix="d")
    res = {}
    for kind, v, k, where in (("alt", da, o["alt_k"][:, 0], o["alt_n"][:, 0]), ("buoy_hs", dh, o["bh_k"], None)):
        bad = ~np.isfinite(v) | (np.abs(np.nan_to_num(v)) > bound)
        good = v[~bad]
        r = {"n": int(len(v)), "bad": int(bad.sum()), "nonfinite": int((~np.isfinite(v)).sum()),
             "rms_good": float(np.sqrt(np.mean(good ** 2))) if len(good) else None,
             "max_abs_finite": float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else None}
        if kind == "alt":
            r["bad_at"] = [{"tag": tags[int(k[i])], "lon": float(lon[where[i]]), "lat": float(lat[where[i]]), "d": float(v[i])}
                           for i in np.flatnonzero(bad)[:20]]
        else:
            names = [str(n) for n in o["buoys"]]
            r["bad_at"] = [{"tag": tags[int(k[i])], "buoy": names[int(o["bh_b"][i])], "d": float(v[i])} for i in np.flatnonzero(bad)[:20]]
        res[kind] = r
        print("  %s: %d of %d observations not finite or above %g m (%d not finite), RMS of the others %.3g m, largest finite %.3g m"
              % (kind, r["bad"], r["n"], bound, r["nonfinite"], r["rms_good"] or 0, r["max_abs_finite"] or 0))
        for b in r["bad_at"][:8]:
            print("     ", b)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--grid", default="/scratch/jmdeloui/HWW3/grids/natl_v5k4")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json")
    ap.add_argument("--op")
    ap.add_argument("--bound", type=float, default=10.0)
    a = ap.parse_args()
    nodes = np.load(os.path.join(a.grid, "plot", "nodes.npz"))
    lon, lat = nodes["lon"], nodes["lat"]
    dep = nodes["depth"] if "depth" in nodes else None
    out = {}
    for d in a.runs:
        tags = sorted(os.path.basename(p)[4:14] for p in glob.glob(os.path.join(d, "dhs_*.f64")))
        rows, first = [], {}
        for t in tags:
            x = np.fromfile(os.path.join(d, "dhs_%s.f64" % t), "<f8")
            hs = os.path.join(d, "hs_%s.f64" % t)
            undef = np.abs(np.abs(x) - 999.9) < 1e-3
            if os.path.exists(hs):
                undef |= np.fromfile(hs, "<f8") < -900
            x = np.where(undef, 0.0, x)
            fin = np.isfinite(x)
            ax = np.abs(np.where(fin, x, 0.0))
            q = float(np.quantile(ax[fin], 0.9999)) if fin.any() else float("nan")
            i = int(np.argmax(ax))
            rows.append({"tag": t, "nonfinite": int((~fin).sum()), "max": float(ax[i]), "node": i, "lon": float(lon[i]),
                         "lat": float(lat[i]), "q9999": q, "rms": float(np.sqrt(np.mean(ax[fin] ** 2)))})
            bad = np.flatnonzero((~fin) | (ax > 100 * q))
            for j in bad:
                if int(j) not in first:
                    hs = os.path.join(d, "hs_%s.f64" % t)
                    h = float(np.fromfile(hs, "<f8")[j]) if os.path.exists(hs) else None
                    first[int(j)] = {"tag": t, "lon": float(lon[j]), "lat": float(lat[j]), "dhs": float(x[j]), "hs": h,
                                     "depth": float(dep[j]) if dep is not None else None}
        out[d] = {"hours": rows, "first_bad": first}
        if a.op:
            out[d]["op"] = op_check(a.op, d, a.bound, lon, lat)
        print("%s: %d hours" % (d, len(rows)))
        for r in rows[:: max(1, len(rows) // 24)] + rows[-1:]:
            print("  %s nonfinite %6d  max %.3e at node %7d (%.3f, %.3f)  q99.99 %.3e  rms %.3e"
                  % (r["tag"], r["nonfinite"], r["max"], r["node"], r["lon"], r["lat"], r["q9999"], r["rms"]))
        print("  %d nodes above 100 x q99.99 or non-finite; the first %d by hour:" % (len(first), a.top))
        for j, f in sorted(first.items(), key=lambda kv: kv[1]["tag"])[: a.top]:
            print("    node %7d %s (%.3f, %.3f) dhs %.3e hs %s depth %s" % (j, f["tag"], f["lon"], f["lat"], f["dhs"], f["hs"], f["depth"]))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
