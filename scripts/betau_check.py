"""Compare the check runs of BETAMAX(U) (scripts/betau_check_hpc.slurm).

usage:
    python scripts/betau_check.py DIR

DIR holds the outputs of the check runs. For every hour of the runs:
v_c0 against v_const (BETAMAX(U) with all coefficients 0 against BETAMAX
1.75 constant: H_s and T_m02, the number of nodes that differ and the
largest difference); t_c0_all and t_c0_beta against t_const (the tangent
for all coefficients at once and the BETAMAX tangent with BETAMAX(U),
against the BETAMAX tangent of the constant run: relative L2 difference
of the tangent of H_s and of T_m02, and of the tangent of the total
energy); v_s3_interp against v_s3_exact (BETAMAX 1.37249382 through the
tables interpolated in BETAMAX against the tables computed for it: the
largest and RMS differences of H_s). Writes DIR/check.json.
"""
import glob
import json
import os
import sys

import numpy as np


UNDEF = -999.0


def tags(d, name):
    return sorted(os.path.basename(p)[len(name) + 1:len(name) + 11] for p in glob.glob(os.path.join(d, name + "_*.f64")))


def field(d, name, tag):
    return np.fromfile(os.path.join(d, "%s_%s.f64" % (name, tag)), "<f8")


def summary(d):
    return json.load(open(os.path.join(d, "summary.json")))


def same(o, a, b, name):
    """Hour by hour: nodes that differ and the largest difference of NAME between runs A and B."""
    rows = []
    for t in tags(os.path.join(o, a), name):
        x, y = field(os.path.join(o, a), name, t), field(os.path.join(o, b), name, t)
        # the nodes where both are defined (UNDEF = -999.9 on land, dry and ice nodes)
        ok = np.isfinite(x) & np.isfinite(y) & (x > UNDEF) & (y > UNDEF)
        dif = np.abs(x[ok] - y[ok])
        rows.append({"tag": t, "nodes": int(ok.sum()), "nodes_differ": int(np.sum(dif > 0)), "max_abs": float(dif.max()),
                     "rms": float(np.sqrt(np.mean(dif ** 2))), "rms_field": float(np.sqrt(np.mean(x[ok] ** 2))),
                     "p99": float(np.quantile(dif, 0.99)), "nodes_above_1mm": int(np.sum(dif > 1e-3))})
    return rows


def rel(o, a, b, name):
    """Hour by hour: relative L2 difference of NAME of run A against run B, and the largest difference."""
    rows = []
    for t in tags(os.path.join(o, b), name):
        x, y = field(os.path.join(o, a), name, t), field(os.path.join(o, b), name, t)
        ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < -UNDEF) & (np.abs(y) < -UNDEF)
        n = np.linalg.norm(y[ok])
        rows.append({"tag": t, "rel_l2": float(np.linalg.norm(x[ok] - y[ok]) / n) if n > 0 else 0.0,
                     "max_abs": float(np.abs(x[ok] - y[ok]).max()), "norm": float(n)})
    return rows


def energy(o, a, b):
    """Largest relative difference of the tangent of the total energy (summary.json, every 6 h)."""
    ea = [e.get("denergy") for e in summary(os.path.join(o, a))["energy"]]
    eb = [e.get("denergy") for e in summary(os.path.join(o, b))["energy"]]
    if None in ea or None in eb:
        return None
    va, vb = np.array(ea, float), np.array(eb, float)
    return {"rel_max": float(np.max(np.abs(va - vb) / np.maximum(np.abs(vb), 1e-300))), "last_a": float(va[-1]), "last_b": float(vb[-1])}


def main():
    o = sys.argv[1]
    res = {}
    if os.path.isdir(os.path.join(o, "v_c0")):
        res["c0_value"] = {n: same(o, "v_c0", "v_const", n) for n in ("hs", "t02")}
        last = res["c0_value"]["hs"][-1]
        print("c = 0 against BETAMAX 1.75 constant, H_s: %s" % "; ".join(
            "%s %d nodes differ, max %.3g m" % (r["tag"], r["nodes_differ"], r["max_abs"]) for r in res["c0_value"]["hs"][::3] + [last]))
        print("    T_m02: max over the hours %.3g s" % max(r["max_abs"] for r in res["c0_value"]["t02"]))
    for a in ("t_c0_all", "t_c0_beta"):
        if os.path.isdir(os.path.join(o, a)):
            r = {n: rel(o, a, "t_const", n) for n in ("dhs", "dt02")}
            r["denergy"] = energy(o, a, "t_const")
            res[a] = r
            print("%s against the BETAMAX tangent of the constant run: dH_s rel. L2 max %.3g (last hour %.3g), dT_m02 %.3g, energy %s" % (
                a, max(x["rel_l2"] for x in r["dhs"]), r["dhs"][-1]["rel_l2"], max(x["rel_l2"] for x in r["dt02"]), r["denergy"]))
    if os.path.isdir(os.path.join(o, "v_s3_interp")):
        r = same(o, "v_s3_interp", "v_s3_exact", "hs")
        res["s3_interp"] = r
        print("BETAMAX 1.37249382 interpolated against exact tables, H_s: max %.3g m, last hour RMS %.3g m, 99th percentile %.3g m, "
              "%d of %d nodes above 1 mm (RMS of H_s %.3f m)" % (max(x["max_abs"] for x in r), r[-1]["rms"], r[-1]["p99"],
                                                              r[-1]["nodes_above_1mm"], r[-1]["nodes"], r[-1]["rms_field"]))
        if os.path.isdir(os.path.join(o, "v_const")):
            # the size of the change the interpolation has to follow: step 3 against p0
            q = same(o, "v_s3_exact", "v_const", "hs")
            res["s3_vs_p0"] = q
            print("    for scale, step 3 against p0 (BETAMAX, SDSC2 and NLPROP), H_s: last hour RMS %.3g m, max %.3g m" % (q[-1]["rms"], q[-1]["max_abs"]))
    json.dump(res, open(os.path.join(o, "check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
