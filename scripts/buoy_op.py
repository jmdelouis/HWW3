"""Depth-aware buoy operator of the calibration.

    python scripts/buoy_op.py positions OP.npz[+OP2.npz] BUOY_DIR OUT.json
    python scripts/buoy_op.py build OP.npz GRID_DIR BASE_RUN POS.json DEPTHS.json OUT.npz
        [--mode depth|nearest] [--tm02 moments|linear] [--s S] [--tol T] [--cap C] [--radius-km R] [--k-idw K]
        [--json WEIGHTS.json]
    python scripts/buoy_op.py tm02 OP.npz RUN

positions: the buoys of the operators of cal_obs.py (OP.npz, several
joined by +), with their position and the bottom_depth attribute of their
file, for scripts/buoy_depth.py (run where the fine bathymetry is).

build: a new operator OUT.npz in which the buoy H_s and T_m02 at each hour
are fixed weighted sums of the model fields at a few nodes (cal_obs.py
eval uses them; the altimeter part is unchanged). The weights depend only
on geometry and depth, so the operator stays linear in the fields and the
tangents go through it unchanged:
  - the water depth at the buoy is that of the GEBCO cell that holds it
    (DEPTHS.json of buoy_depth.py), else the median of the sea cells of
    the 3 x 3 cells around it, else the bottom_depth of its file: GEBCO
    first, since the node depths of the grid come from GEBCO;
  - the base weights are barycentric in the triangle of the mesh that
    holds the buoy, over its wet nodes, or inverse squared distance over
    the K nearest wet nodes within R km (default 8 and 10 km) when the
    buoy is outside the wet mesh;
  - each is multiplied by exp(-(ln(d_node / d_buoy))^2 / S^2), with both
    depths capped at C metres (default 100 m: deeper water is deep for
    the waves of interest), and the weights are renormalised (default
    S = ln 2);
  - if no base node has a depth within a factor T of the buoy's (default
    3), the K nearest wet nodes within R km are tried with the kernel; if
    none is within the factor either, the buoy is left out of the cost and
    the scores;
  - a buoy without any depth keeps its base weights.
The buoy T_m02 is 2 pi sqrt(M0 / M2) from the weighted moments m0 and m2
of the nodes (--tm02 moments, the default: a linear average of T_m02
differs from it by up to 4.5 s at 1 % of the buoy hours), or the weighted
T_m02 itself (--tm02 linear). The hours are those of OP whose weighted
nodes are all defined in BASE_RUN. --mode nearest gives every buoy the node of OP with weight 1 and
a flat kernel: the operator of OP, to check that the scores are the same.
--json writes the nodes, weights and depths of every buoy.

tm02: the buoy T_m02 of RUN through the weights of OP, interpolated
linearly as the operator does and through the moments m0 and m2 (from
H_s and T_m02 at the nodes), to see whether the difference matters.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cal_obs  # noqa: E402
import coast_diag  # noqa: E402


def positions(a):
    names = []
    for p in a.op.split("+"):
        for n in np.load(p)["buoys"]:
            if str(n) not in names:
                names.append(str(n))
    out = {}
    for n in names:
        lon, lat, bd = coast_diag.buoy_meta(a.bdir, n)
        out[n] = {"lon": lon, "lat": lat, "bottom_depth": None if not np.isfinite(bd) else bd}
    json.dump(out, open(a.out, "w"), indent=1)
    print("%d buoys written to %s" % (len(out), a.out))


def local_xy(lon, lat, lon0, lat0):
    return (lon - lon0) * 111.2 * np.cos(np.radians(lat0)), (lat - lat0) * 111.2


def containing_triangle(tri, lon, lat, ctree, blon, blat, nq=24):
    """Barycentric weights of (blon, blat) in the triangle that holds it, or None."""
    _, cand = ctree.query(cal_obs.xyz(np.array([blon]), np.array([blat]))[0], nq)
    for t in np.atleast_1d(cand):
        v = tri[t]
        x, y = local_xy(lon[v], lat[v], blon, blat)
        det = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if det == 0:
            continue
        l0 = ((y[1] - y[2]) * (0 - x[2]) + (x[2] - x[1]) * (0 - y[2])) / det
        l1 = ((y[2] - y[0]) * (0 - x[2]) + (x[0] - x[2]) * (0 - y[2])) / det
        l2 = 1 - l0 - l1
        if min(l0, l1, l2) >= -1e-9:
            return v, np.array([l0, l1, l2])
    return None


def build(a):
    op = dict(np.load(a.op))
    tags = [str(t) for t in op["tags"]]
    names = [str(n) for n in op["buoys"]]
    pos = json.load(open(a.pos))
    deps = json.load(open(a.depths)) if a.mode == "depth" else {}
    nodes = np.load(os.path.join(a.grid, "plot", "nodes.npz"))
    lon, lat = nodes["lon"].astype(float), nodes["lat"].astype(float)
    depth = np.fromfile(os.path.join(a.grid, "raw", "depth.f64"), "<f8")
    tri = np.fromfile(os.path.join(a.grid, "raw", "tri.u32"), "<u4").reshape(-1, 3).astype(np.int64)
    nn = len(lon)
    # wet nodes: defined at more than half the hours of the base run, as cal_obs.py build
    defined_frac = np.zeros(nn)
    for tag in tags:
        defined_frac += np.isfinite(cal_obs.field(a.base, "hs", tag))
    wet = defined_frac / len(tags) > 0.5
    wi = np.flatnonzero(wet)
    wtree = cKDTree(cal_obs.xyz(lon[wi], lat[wi]))
    cen = cal_obs.xyz(lon[tri].mean(1), lat[tri].mean(1))
    ctree = cKDTree(cen)
    chord = 2 * np.sin(a.radius_km / coast_diag.EARTH_KM / 2)
    dcap = lambda d: np.minimum(np.maximum(d, 0.1), a.cap)  # noqa: E731
    rows, wl = [], []
    for ib, name in enumerate(names):
        p = pos[name]
        row = {"name": name, "lon": p["lon"], "lat": p["lat"]}
        s = op["bh_b"] == ib
        old = int(op["bh_n"][s][0]) if s.any() else int(op["bt_n"][op["bt_b"] == ib][0])
        row["old_node"] = old
        if a.mode == "nearest":
            row.update(base="nearest", depth=None, source="none", kept=True)
            wl.append((np.array([old]), np.array([1.0])))
            rows.append(row)
            continue
        d = deps.get(name) or {}
        if d.get("cell") is not None:
            db, src = d["cell"], "gebco"
        elif d.get("med3") is not None:
            db, src = d["med3"], "gebco3x3"
        elif p.get("bottom_depth"):
            db, src = p["bottom_depth"], "metadata"
        else:
            db, src = None, "none"
        row.update(depth=db, source=src, bottom_depth=p.get("bottom_depth"))
        # base weights
        t = containing_triangle(tri, lon, lat, ctree, p["lon"], p["lat"])
        base_n, base_w, kind = None, None, None
        if t is not None:
            v, bw = t
            ok = wet[v]
            if ok.any():
                base_n, base_w, kind = v[ok], bw[ok], "triangle"
        idx = wtree.query_ball_point(cal_obs.xyz(np.array([p["lon"]]), np.array([p["lat"]]))[0], chord)
        near = wi[idx]
        dist = coast_diag.km(p["lon"], p["lat"], lon[near], lat[near])
        o = np.argsort(dist)[:a.k_idw]
        near, dist = near[o], dist[o]
        idw_w = 1.0 / np.maximum(dist, 0.1) ** 2
        if base_n is None:
            if len(near) == 0:
                row.update(base="none", kept=False, reason="no wet node within %.0f km" % a.radius_km)
                wl.append((np.array([old]), np.array([0.0])))
                rows.append(row)
                continue
            base_n, base_w, kind = near, idw_w, "idw"
        row["base"] = kind
        if db is None:
            n_, w_ = base_n, base_w
        else:
            lr = np.log(dcap(depth[base_n]) / dcap(db))
            if np.abs(lr).min() > np.log(a.tol):
                kind += "+idw"
                lr = np.log(dcap(depth[near]) / dcap(db)) if len(near) else np.array([np.inf])
                if len(near) == 0 or np.abs(lr).min() > np.log(a.tol):
                    row.update(kept=False, reason="no node within a factor %g of the buoy depth" % a.tol,
                               node_depths=[float(x) for x in depth[base_n]])
                    wl.append((np.array([old]), np.array([0.0])))
                    rows.append(row)
                    continue
                base_n, base_w = near, idw_w
                row["base"] = kind
            n_, w_ = base_n, base_w * np.exp(-(lr / a.s) ** 2)
        w_ = w_ / w_.sum()
        keep = w_ > 1e-6
        n_, w_ = n_[keep], w_[keep] / w_[keep].sum()
        row.update(kept=True, nodes=[int(x) for x in n_], weights=[float(x) for x in w_],
                   node_depths=[float(x) for x in depth[n_]],
                   changed=not (len(n_) == 1 and int(n_[0]) == old))
        wl.append((n_, w_))
        rows.append(row)
    K = max(len(n_) for n_, _ in wl)
    bwn = np.zeros((len(names), K), np.int64)
    bww = np.zeros((len(names), K))
    for i, (n_, w_) in enumerate(wl):
        bwn[i, :len(n_)] = n_
        bww[i, :len(w_)] = w_
    kept = np.array([r["kept"] for r in rows])
    # hours whose weighted nodes are all defined in the base run
    involved = np.unique(bwn[bww > 0])
    dmask = np.zeros((len(tags), len(involved)), bool)
    for i, tag in enumerate(tags):
        dmask[i] = np.isfinite(cal_obs.field(a.base, "hs", tag, involved))
    col = {n: j for j, n in enumerate(involved)}

    def ok_hours(k, b):
        good = kept[b].copy()
        for i in np.flatnonzero(good):
            nb = bwn[b[i]][bww[b[i]] > 0]
            good[i] = all(dmask[k[i], col[n]] for n in nb)
        return good
    gh = ok_hours(op["bh_k"], op["bh_b"])
    gt = ok_hours(op["bt_k"], op["bt_b"])
    for key in ("bh_k", "bh_n", "bh_obs", "bh_b"):
        op[key] = op[key][gh]
    for key in ("bt_k", "bt_n", "bt_obs", "bt_b", "bt_cost"):
        op[key] = op[key][gt]
    op.update(bw_nodes=bwn, bw_w=bww, bw_kept=kept, bw_mode=np.array(a.mode), bw_tm02=np.array(a.tm02),
              bw_params=np.array([a.s, a.tol, a.cap, a.radius_km, a.k_idw]))
    np.savez(a.out, **op)
    src = {}
    for r in rows:
        src[r.get("source", "none")] = src.get(r.get("source", "none"), 0) + 1
    print("%s: %d buoys, %d kept, %d left out; %d change node or weights; depth sources %s; "
          "H_s hours %d of %d, T_m02 hours %d of %d; up to %d nodes per buoy"
          % (a.out, len(rows), int(kept.sum()), int((~kept).sum()), sum(1 for r in rows if r.get("changed")),
             src, int(gh.sum()), len(gh), int(gt.sum()), len(gt), K))
    for r in rows:
        if not r["kept"]:
            print("  left out: %s (%.3f, %.3f) depth %s (%s), %s" % (r["name"], r["lon"], r["lat"], r.get("depth"),
                                                                     r.get("source"), r.get("reason")))
    bd = np.array([[r["depth"], r["bottom_depth"]] for r in rows
                   if r.get("depth") is not None and r.get("bottom_depth") and r.get("source", "").startswith("gebco")], float)
    if len(bd):
        q = bd[:, 0] / bd[:, 1]
        print("GEBCO / bottom_depth at %d buoys: median %.2f, 10-90 %% %.2f-%.2f, beyond a factor 2: %d"
              % (len(q), np.median(q), np.percentile(q, 10), np.percentile(q, 90), int(((q > 2) | (q < 0.5)).sum())))
    if a.json:
        json.dump({"params": {"mode": a.mode, "s": a.s, "tol": a.tol, "cap": a.cap, "radius_km": a.radius_km,
                              "k_idw": a.k_idw}, "buoys": rows}, open(a.json, "w"), indent=1, default=float)


def tm02(a):
    o = np.load(a.op)
    tags = [str(t) for t in o["tags"]]
    k, b = o["bt_k"], o["bt_b"]
    lin = cal_obs.buoy_points(o, a.run, tags, "t02", k, b)
    m0s, m2s = np.full(len(k), np.nan), np.full(len(k), np.nan)
    nodes, wts = o["bw_nodes"], o["bw_w"]
    for i, tag in enumerate(tags):
        s = k == i
        if not s.any():
            continue
        hs, t = cal_obs.field(a.run, "hs", tag), cal_obs.field(a.run, "t02", tag)
        m0 = (hs / 4) ** 2
        m2 = m0 * (2 * np.pi / t) ** 2
        nb, wb = nodes[b[s]], wts[b[s]]
        m0s[s] = (np.where(wb > 0, m0[nb], 0.0) * wb).sum(1)
        m2s[s] = (np.where(wb > 0, m2[nb], 0.0) * wb).sum(1)
    mom = 2 * np.pi * np.sqrt(m0s / m2s)
    obs = o["bt_obs"]
    for label, m in (("linear in T_m02", lin), ("through m0 and m2", mom)):
        e = m - obs
        print("%-18s bias %+.4f s, RMSE %.4f s (%d hours)" % (label, np.nanmean(e), np.sqrt(np.nanmean(e ** 2)), int(np.isfinite(e).sum())))
    d = np.abs(mom - lin)
    print("|difference| median %.2e s, 99th percentile %.2e s, largest %.2e s" % (np.nanmedian(d), np.nanpercentile(d, 99), np.nanmax(d)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("positions")
    p.add_argument("op")
    p.add_argument("bdir")
    p.add_argument("out")
    p = sub.add_parser("build")
    for k in ("op", "grid", "base", "pos", "depths", "out"):
        p.add_argument(k)
    p.add_argument("--mode", default="depth", choices=["depth", "nearest"])
    p.add_argument("--tm02", default="moments", choices=["moments", "linear"])
    p.add_argument("--s", type=float, default=float(np.log(2)))
    p.add_argument("--tol", type=float, default=3.0)
    p.add_argument("--cap", type=float, default=100.0)
    p.add_argument("--radius-km", type=float, default=10.0)
    p.add_argument("--k-idw", type=int, default=8)
    p.add_argument("--json")
    p = sub.add_parser("tm02")
    p.add_argument("op")
    p.add_argument("run")
    a = ap.parse_args()
    {"positions": positions, "build": build, "tm02": tm02}[a.cmd](a)


if __name__ == "__main__":
    main()
