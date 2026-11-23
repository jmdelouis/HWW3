"""Coastal diagnostics of the buoy H_s scores of the calibration.

    python scripts/coast_diag.py OP.npz GRID_DIR BUOY_DIR RUN_A RUN_B
        [--subgrid SUB.json] [--emit-nodes NODES.json] [--json OUT.json]
        [--fractions 0.25+0.5] [--radius-km 25]

OP.npz is an operator of cal_obs.py (buoy hours, the node each buoy is
matched to: the nearest wet node within 10 km); RUN_A and RUN_B are two
runs (for example the start run and the run after a calibration step).
For each buoy: its water depth (the bottom_depth attribute of its file in
BUOY_DIR), the depth of its node and the distance to it, the distance of
the node to the coast (the nearest mesh boundary node that is not on the
open boundary), its HEALPix level and cell radius (half the mean length of
its edges), a grid descriptor of the sub-grid slope (the depth range over
the node and its neighbours divided by the node depth), and with --subgrid
the spread of the fine bathymetry and the fraction of land inside the cell
(scripts/coast_subgrid.py on GEBCO). Then the H_s bias and RMSE of both runs over the buoy hours,
per buoy and in classes of depth mismatch, node depth, distance to the
coast, level and sub-grid descriptors, and a depth-aware matching tried as
a diagnostic only: the nearest wet node within --radius-km whose depth is
within a fraction of the buoy's (else the node of the operator).
--emit-nodes writes the buoy nodes and cell radii for coast_subgrid.py.
"""
import argparse
import glob
import json
import os
import sys

import netCDF4
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cal_obs  # noqa: E402

EARTH_KM = 6371.0


def km(lon1, lat1, lon2, lat2):
    a = cal_obs.xyz(np.atleast_1d(lon1), np.atleast_1d(lat1))
    b = cal_obs.xyz(np.atleast_1d(lon2), np.atleast_1d(lat2))
    return 2 * np.arcsin(np.clip(np.linalg.norm(a - b, axis=-1) / 2, 0, 1)) * EARTH_KM


def buoy_meta(bdir, name):
    fs = sorted(glob.glob(os.path.join(bdir, name + "_*.nc")))
    if not fs:
        return np.nan, np.nan, np.nan
    d = netCDF4.Dataset(fs[0])
    lon = float(np.nanmedian(np.ma.filled(np.ma.asarray(d["LONGITUDE"][:], dtype=float), np.nan)))
    lat = float(np.nanmedian(np.ma.filled(np.ma.asarray(d["LATITUDE"][:], dtype=float), np.nan)))
    bd = np.nan
    if "bottom_depth" in d.ncattrs():
        try:
            bd = abs(float(str(d.getncattr("bottom_depth")).split()[0]))
        except (ValueError, IndexError):
            pass
    return lon, lat, bd if bd > 0 else np.nan


def stats(m, o):
    e = m - o
    return (float(np.mean(e)), float(np.sqrt(np.mean(e ** 2))), int(len(e))) if len(e) else (np.nan, np.nan, 0)


def table(title, cls, labels, sets, runs):
    """Bias and RMSE per class for each operator of SETS ((label, per) pairs) and run."""
    print("\n%s" % title)
    head = "  %-14s %6s" % ("class", "buoys")
    for lab, _ in sets:
        head += " %7s" % ("hours" if lab == "old" else "h " + lab) + "".join("  %-3s %-4s bias/RMSE (cm)" % (r, lab) for r in runs)
    print(head)
    for c, lab in enumerate(labels):
        s = cls == c
        if not s.any():
            continue
        line = "  %-14s %6d" % (lab, int(s.sum()))
        for _, per in sets:
            line += " %7d" % int(sum(per["hours"][i] for i in np.flatnonzero(s)))
            for r in runs:
                m = np.concatenate([per[r]["m"][i] for i in np.flatnonzero(s)])
                o = np.concatenate([per["o"][i] for i in np.flatnonzero(s)])
                b, e, _ = stats(m, o)
                line += "  %+7.2f / %6.2f          " % (100 * b, 100 * e)
        print(line)


def classes(x, edges, fmt):
    c = np.full(len(x), -1)
    labels = []
    for i in range(len(edges) - 1):
        s = (x >= edges[i]) & (x < edges[i + 1])
        c[s] = i
        labels.append(fmt % (edges[i], edges[i + 1]))
    labels.append("unknown")
    c[~np.isfinite(x)] = len(labels) - 1
    return c, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("op", "grid", "bdir", "run_a", "run_b"):
        ap.add_argument(k)
    ap.add_argument("--subgrid")
    ap.add_argument("--emit-nodes")
    ap.add_argument("--json")
    ap.add_argument("--fractions", default="0.25+0.5")
    ap.add_argument("--radius-km", type=float, default=25.0)
    ap.add_argument("--op-new", help="a depth-aware operator of buoy_op.py, compared with OP class by class")
    a = ap.parse_args()
    op = np.load(a.op)
    tags = [str(t) for t in op["tags"]]
    names = [str(n) for n in op["buoys"]]
    bk, bn, bo, bb = op["bh_k"], op["bh_n"], op["bh_obs"], op["bh_b"]
    nodes = np.load(os.path.join(a.grid, "plot", "nodes.npz"))
    lon, lat, level = nodes["lon"].astype(float), nodes["lat"].astype(float), nodes["level"].astype(int)
    depth = np.fromfile(os.path.join(a.grid, "raw", "depth.f64"), "<f8")
    tri = np.fromfile(os.path.join(a.grid, "raw", "tri.u32"), "<u4").reshape(-1, 3).astype(np.int64)
    obc = set(np.fromfile(os.path.join(a.grid, "raw", "obc.u32"), "<u4").tolist())
    nn = len(lon)
    # edges, neighbours, boundary (edges in one triangle), coast = boundary minus open boundary
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e.sort(1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    bnodes = np.unique(ue[cnt == 1])
    coast = np.array([i for i in bnodes if i not in obc])
    ctree = cKDTree(cal_obs.xyz(lon[coast], lat[coast]))
    elen = km(lon[ue[:, 0]], lat[ue[:, 0]], lon[ue[:, 1]], lat[ue[:, 1]])
    esum = np.bincount(ue[:, 0], elen, nn) + np.bincount(ue[:, 1], elen, nn)
    edeg = np.bincount(ue[:, 0], minlength=nn) + np.bincount(ue[:, 1], minlength=nn)
    radius = 0.5 * esum / np.maximum(edeg, 1)
    # depth range over each node and its neighbours (grid sub-grid slope descriptor)
    dmax = depth.copy()
    dmin = depth.copy()
    for c0, c1 in ((0, 1), (1, 0)):
        np.maximum.at(dmax, ue[:, c0], depth[ue[:, c1]])
        np.minimum.at(dmin, ue[:, c0], depth[ue[:, c1]])
    slope = (dmax - dmin) / np.maximum(depth, 1.0)
    runs = {"A": a.run_a, "B": a.run_b}
    model = {r: cal_obs.model_points(d, tags, "hs", bk, bn) for r, d in runs.items()}
    sub = json.load(open(a.subgrid)) if a.subgrid else {}
    rows = []
    for ib, name in enumerate(names):
        s = bb == ib
        node = int(bn[s][0])
        blon, blat, bd = buoy_meta(a.bdir, name)
        dc, _ = ctree.query(cal_obs.xyz(np.array([lon[node]]), np.array([lat[node]]))[0])
        row = {"name": name, "lon": blon, "lat": blat, "bottom_depth": bd, "node": node,
               "node_depth": float(depth[node]), "dist_km": float(km(blon, blat, lon[node], lat[node])[0]),
               "coast_km": float(2 * np.arcsin(min(dc / 2, 1.0)) * EARTH_KM), "level": int(level[node]),
               "radius_km": float(radius[node]), "slope": float(slope[node]),
               "gebco_std": sub.get(str(node), {}).get("std", np.nan),
               "gebco_range": sub.get(str(node), {}).get("range", np.nan),
               "gebco_land": sub.get(str(node), {}).get("land", np.nan)}
        for r in runs:
            ok = np.isfinite(model[r][s])
            row["bias_" + r], row["rmse_" + r], row["hours_" + r] = stats(model[r][s][ok], bo[s][ok])
        rows.append(row)
    if a.emit_nodes:
        json.dump({str(r["node"]): {"lon": float(lon[r["node"]]), "lat": float(lat[r["node"]]), "radius_km": r["radius_km"]}
                   for r in rows}, open(a.emit_nodes, "w"))
        print("%d buoy nodes written to %s" % (len(rows), a.emit_nodes))
    # per-buoy model and observations, on the hours both runs define
    per = {"o": [], "hours": [], "A": {"m": []}, "B": {"m": []}}
    for ib in range(len(names)):
        s = bb == ib
        ok = np.isfinite(model["A"][s]) & np.isfinite(model["B"][s])
        per["o"].append(bo[s][ok])
        per["hours"].append(int(ok.sum()))
        for r in runs:
            per[r]["m"].append(model[r][s][ok])
    sets = [("old", per)]
    if a.op_new:
        on = np.load(a.op_new)
        mn = {r: cal_obs.buoy_points(on, d, tags, "hs", on["bh_k"], on["bh_b"]) for r, d in runs.items()}
        per2 = {"o": [], "hours": [], "A": {"m": []}, "B": {"m": []}}
        for ib in range(len(names)):
            s = on["bh_b"] == ib
            ok = np.isfinite(mn["A"][s]) & np.isfinite(mn["B"][s])
            per2["o"].append(on["bh_obs"][s][ok])
            per2["hours"].append(int(ok.sum()))
            for r in runs:
                per2[r]["m"].append(mn[r][s][ok])
        sets.append(("new", per2))
        o2 = np.concatenate(per2["o"])
        print("new operator %s: %d buoys with hours, %d buoy hours" % (a.op_new, sum(1 for h in per2["hours"] if h), len(o2)))
        for r in runs:
            b, e, n = stats(np.concatenate(per2[r]["m"]), o2)
            print("  run %s: H_s bias %+.2f cm, RMSE %.2f cm" % (r, 100 * b, 100 * e))
    allm = {r: np.concatenate(per[r]["m"]) for r in runs}
    allo = np.concatenate(per["o"])
    print("%d buoys, %d buoy hours; run A %s, run B %s" % (len(names), len(allo), a.run_a, a.run_b))
    for r in runs:
        b, e, n = stats(allm[r], allo)
        print("  run %s: H_s bias %+.2f cm, RMSE %.2f cm" % (r, 100 * b, 100 * e))
    nd = np.array([r["node_depth"] for r in rows])
    bd = np.array([r["bottom_depth"] for r in rows])
    print("\nbuoys with a bottom_depth: %d of %d; node depth / buoy depth: median %.2f, 10-90 %% %.2f-%.2f; "
          "buoy-node distance median %.1f km, max %.1f km"
          % (np.isfinite(bd).sum(), len(rows), np.nanmedian(nd / bd), np.nanpercentile(nd / bd, 10),
             np.nanpercentile(nd / bd, 90), np.median([r["dist_km"] for r in rows]), max(r["dist_km"] for r in rows)))
    ratio = nd / bd
    c, lab = classes(ratio, [0, 0.5, 0.8, 1.25, 2.0, 1e9], "%.2f-%.2g")
    table("H_s by node depth / buoy depth", c, lab, sets, runs)
    c, lab = classes(nd, [0, 20, 50, 100, 200, 1e9], "%g-%g m")
    table("H_s by node depth", c, lab, sets, runs)
    c, lab = classes(np.array([r["coast_km"] for r in rows]), [0, 5, 10, 25, 50, 1e9], "%g-%g km")
    table("H_s by distance of the node to the coast", c, lab, sets, runs)
    lv = np.array([r["level"] for r in rows])
    levels = sorted(set(lv.tolist()))
    c = np.array([levels.index(v) for v in lv])
    table("H_s by HEALPix level of the node", c, ["level %d" % v for v in levels], sets, runs)
    c, lab = classes(np.array([r["slope"] for r in rows]), [0, 0.1, 0.3, 1.0, 1e9], "%g-%g")
    table("H_s by depth range of the node and its neighbours / node depth", c, lab, sets, runs)
    if sub:
        gs = np.array([r["gebco_std"] for r in rows]) / np.maximum(nd, 1.0)
        c, lab = classes(gs, [0, 0.1, 0.3, 1.0, 1e9], "%g-%g")
        table("H_s by GEBCO depth spread inside the cell / node depth", c, lab, sets, runs)
        c, lab = classes(np.array([r["gebco_land"] for r in rows]), [0, 0.001, 0.1, 0.3, 1.01], "%g-%g")
        table("H_s by GEBCO land fraction inside the cell", c, lab, sets, runs)
    # depth-aware matching (diagnostic only)
    wet = np.isfinite(cal_obs.field(a.run_a, "hs", tags[0]))
    wi = np.flatnonzero(wet)
    wtree = cKDTree(cal_obs.xyz(lon[wi], lat[wi]))
    chord = 2 * np.sin(a.radius_km / EARTH_KM / 2)
    out = {"rows": rows, "depth_aware": {}}
    for f in [float(x) for x in a.fractions.split("+")]:
        moved = 0
        newn = bn.copy()
        for ib, r in enumerate(rows):
            if not np.isfinite(r["bottom_depth"]):
                continue
            if abs(r["node_depth"] - r["bottom_depth"]) <= f * r["bottom_depth"]:
                continue
            cand = wi[wtree.query_ball_point(cal_obs.xyz(np.array([r["lon"]]), np.array([r["lat"]]))[0], chord)]
            cand = cand[np.abs(depth[cand] - r["bottom_depth"]) <= f * r["bottom_depth"]]
            if len(cand) == 0:
                continue
            best = cand[np.argmin(km(r["lon"], r["lat"], lon[cand], lat[cand]))]
            newn[bb == ib] = best
            moved += 1
        res = {}
        for rn, d in runs.items():
            m = cal_obs.model_points(d, tags, "hs", bk, newn)
            ok = np.isfinite(m) & np.isfinite(model["A"]) & np.isfinite(model["B"])
            b0, e0, _ = stats(model[rn][ok], bo[ok])
            b1, e1, n1 = stats(m[ok], bo[ok])
            shallow = ok & np.isin(bb, [i for i, r in enumerate(rows) if r["node_depth"] < 50])
            bs0, es0, _ = stats(model[rn][shallow], bo[shallow])
            bs1, es1, ns = stats(m[shallow], bo[shallow])
            res[rn] = {"bias": [b0, b1], "rmse": [e0, e1], "hours": n1, "shallow_bias": [bs0, bs1], "shallow_rmse": [es0, es1],
                       "shallow_hours": ns}
        out["depth_aware"][str(f)] = {"moved": moved, "runs": res}
        print("\ndepth-aware matching, depth within %.0f %% of the buoy's, within %.0f km: %d buoys moved"
              % (100 * f, a.radius_km, moved))
        for rn in runs:
            q = res[rn]
            print("  run %s: all %+.2f -> %+.2f cm bias, %.2f -> %.2f cm RMSE; nodes < 50 m: %+.2f -> %+.2f cm bias, "
                  "%.2f -> %.2f cm RMSE (%d hours)" % (rn, 100 * q["bias"][0], 100 * q["bias"][1], 100 * q["rmse"][0],
                                                       100 * q["rmse"][1], 100 * q["shallow_bias"][0], 100 * q["shallow_bias"][1],
                                                       100 * q["shallow_rmse"][0], 100 * q["shallow_rmse"][1], q["shallow_hours"]))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
