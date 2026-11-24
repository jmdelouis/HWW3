"""Representativeness errors of the buoys for the calibration cost.

    python scripts/buoy_sigma.py OP_TRAIN.npz OP_VALID.npz RUN COAST_TRAIN.json COAST_VALID.json SUBGRID.json
        OUT_TRAIN.npz OUT_VALID.npz [--json SIGMA.json] [--min-buoys 5] [--exclude LIST.csv] [--huber DELTA]

OP_TRAIN and OP_VALID are depth-aware operators of scripts/buoy_op.py; RUN
is the start run of the calibration; COAST_*.json are the rows of
scripts/coast_diag.py for the buoys of each window (the node of each buoy
and its distance to the coast) and SUBGRID.json the fraction of land in
the cells of these nodes (scripts/coast_subgrid.py). Each buoy gets a
class from the land fraction of its cell (0, 0.1-10 %, 10-30 %, 30 % or
more) and the distance of its node to the coast (under 5 km, 5 km or more).
If a class has fewer than --min-buoys buoys on the training window, the two
distance classes of its land fraction are merged, and a land class still
too small is merged with the next smaller one. For each class, from the
residuals of RUN on the training window:
    sigma_repr^2 = max(0, mean square residual - sigma_0^2)
for H_s (sigma_0 = 0.30 m) and for T_m02 (1.0 s, over the hours of the
T_m02 term of the cost). The observation error of a buoy hour is then
sqrt(sigma_0^2 + sigma_repr^2) for its class: OUT_TRAIN and OUT_VALID are
the operators with these errors per hour (bh_sigma, bt_sigma), which
cal_obs.py and cal_gn.py use. They are fixed once, from the start run, and
not refitted along the calibration.

--exclude removes the buoys of a list (configs/buoy_exclude.csv: sites the
mesh cannot represent) from both operators before the errors are fitted,
and --huber stores the threshold of the Huber loss (in units of the
observation error of each observation) that cal_obs.py and cal_gn.py then
use for the whole cost (the third cost of the calibration).
"""
import csv
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cal_obs  # noqa: E402

SIGMA0 = {"hs": 0.30, "tm02": 1.0}
LAND = ["0", "0.1-10 %", "10-30 %", ">= 30 %"]
COAST = ["< 5 km", ">= 5 km"]


def land_class(f):
    if not np.isfinite(f) or f <= 0.001:
        return 0
    return 1 if f < 0.1 else (2 if f < 0.3 else 3)


def buoy_classes(names, rows, sub):
    by = {r["name"]: r for r in rows}
    out = []
    for n in names:
        r = by[n]
        land = sub.get(str(r["node"]), {}).get("land", np.nan)
        out.append((land_class(land), 0 if r["coast_km"] < 5 else 1))
    return out


def drop(o, excl):
    """The operator O without the hours of the buoys of EXCL (marked not kept)."""
    if not excl:
        return o
    names = [str(n) for n in o["buoys"]]
    bad = [i for i, n in enumerate(names) if n in excl]
    sh = ~np.isin(o["bh_b"], bad)
    st = ~np.isin(o["bt_b"], bad)
    for k in ("bh_k", "bh_n", "bh_obs", "bh_b"):
        o[k] = o[k][sh]
    for k in ("bt_k", "bt_n", "bt_obs", "bt_b", "bt_cost"):
        o[k] = o[k][st]
    kept = o["bw_kept"].copy()
    kept[bad] = False
    o["bw_kept"] = kept
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("op_train", "op_valid", "run", "coast_train", "coast_valid", "subgrid", "out_train", "out_valid"):
        ap.add_argument(k)
    ap.add_argument("--json")
    ap.add_argument("--min-buoys", type=int, default=5)
    ap.add_argument("--exclude", help="CSV of buoys left out (column name)")
    ap.add_argument("--huber", type=float, default=0.0)
    a = ap.parse_args()
    excl = set()
    if a.exclude:
        excl = {r["name"] for r in csv.DictReader(open(a.exclude))}
    sub = json.load(open(a.subgrid))
    ot = drop(dict(np.load(a.op_train)), excl)
    names_t = [str(n) for n in ot["buoys"]]
    cls_t = buoy_classes(names_t, json.load(open(a.coast_train))["rows"], sub)
    kept = ot["bw_kept"]
    print("left out: %d buoys of the list (%s); %d buoys kept on the training window"
          % (sum(n in excl for n in names_t), ", ".join(sorted(n for n in names_t if n in excl)), int(kept.sum())))
    count = {}
    for (l, c), k in zip(cls_t, kept):
        if k:
            count[(l, c)] = count.get((l, c), 0) + 1
    # merging: a land class with a small distance class becomes one class;
    # a land class still small joins the next smaller land class
    group = {}
    for l in range(4):
        if all(count.get((l, c), 0) >= a.min_buoys for c in range(2)):
            for c in range(2):
                group[(l, c)] = "land %s, %s" % (LAND[l], COAST[c])
        else:
            for c in range(2):
                group[(l, c)] = "land %s" % LAND[l]
    for l in range(1, 4):
        g = group[(l, 0)]
        if g == group[(l, 1)] and sum(count.get((l, c), 0) for c in range(2)) < a.min_buoys:
            for c in range(2):
                group[(l, c)] = group[(l - 1, c)]
    # residuals of the start run on the training window
    _, bh, bt = cal_obs.apply(ot, a.run)
    rh = bh - ot["bh_obs"]
    rt = bt - ot["bt_obs"]
    gb = np.array([group[c] for c in cls_t])
    table = {}
    for g in sorted(set(group.values()), key=lambda s: list(group.values()).index(s)):
        bi = np.flatnonzero((gb == g) & kept)
        sh = np.isin(ot["bh_b"], bi)
        stt = np.isin(ot["bt_b"], bi) & ot["bt_cost"]
        msh = float(np.mean(rh[sh] ** 2)) if sh.any() else 0.0
        mst = float(np.mean(rt[stt] ** 2)) if stt.any() else 0.0
        table[g] = {"buoys": int(len(bi)), "hs_hours": int(sh.sum()), "tm02_hours": int(stt.sum()),
                    "hs_bias": float(np.mean(rh[sh])) if sh.any() else None, "hs_rms": float(np.sqrt(msh)),
                    "tm02_rms": float(np.sqrt(mst)),
                    "sigma_repr_hs": float(np.sqrt(max(0.0, msh - SIGMA0["hs"] ** 2))),
                    "sigma_repr_tm02": float(np.sqrt(max(0.0, mst - SIGMA0["tm02"] ** 2)))}
    print("%-26s %6s %7s %7s %9s %9s %10s %10s" % ("class", "buoys", "H_s h", "T h", "H_s RMS", "T RMS", "s_repr H_s", "s_repr T"))
    for g, t in table.items():
        print("%-26s %6d %7d %7d %8.3fm %8.3fs %9.3fm %9.3fs" % (g, t["buoys"], t["hs_hours"], t["tm02_hours"], t["hs_rms"], t["tm02_rms"],
                                                           t["sigma_repr_hs"], t["sigma_repr_tm02"]))
    for opf, coastf, outf in ((a.op_train, a.coast_train, a.out_train), (a.op_valid, a.coast_valid, a.out_valid)):
        o = drop(dict(np.load(opf)), excl)
        names = [str(n) for n in o["buoys"]]
        cls = buoy_classes(names, json.load(open(coastf))["rows"], sub)
        g = [group[c] for c in cls]
        sh = np.array([np.hypot(SIGMA0["hs"], table[x]["sigma_repr_hs"]) for x in g])
        st = np.array([np.hypot(SIGMA0["tm02"], table[x]["sigma_repr_tm02"]) for x in g])
        o.update(bh_sigma=sh[o["bh_b"]], bt_sigma=st[o["bt_b"]], bs_class=np.array(g), bs_table=np.array(json.dumps(table)),
                 bx_excluded=np.array(sorted(excl)))
        if a.huber > 0:
            o["huber_delta"] = np.array(a.huber)
        np.savez(outf, **o)
        print("%s: %d buoys, sigma of H_s %.3f-%.3f m, of T_m02 %.3f-%.3f s" % (outf, len(names), sh.min(), sh.max(), st.min(), st.max()))
    if a.json:
        json.dump({"sigma0": SIGMA0, "min_buoys": a.min_buoys, "classes": table,
                   "buoys": {n: group[c] for n, c in zip(names_t, cls_t)}}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
