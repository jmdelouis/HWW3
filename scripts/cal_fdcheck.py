"""The derivatives of the calibration cost and scores from the tangent runs
against the M2b finite differences (docs/differentiable.md, section 9.5):
48 h from 8 January, window 8-10 January (OP.npz from cal_obs.py build).

For each parameter P, the finite differences of the runs P_p1 / P_m1
(step 1 %) and P_p01 / P_m01 (0.1 %) of CASES (lines "NAME JOBID", outputs
$S/m2/out_JOBID) and the tangent of the run PREFIX_P ($S/cal/PREFIX_P),
both through the operators of OP.npz.

usage:
    python scripts/cal_fdcheck.py OP.npz CASES PREFIX OUT.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cal_obs  # noqa: E402

S = "/scratch/jmdeloui/HWW3"
KEYS = [("J", None), ("alt", "bias"), ("alt", "rmse"), ("buoy_hs", "bias"), ("buoy_hs", "rmse"), ("buoy_tm02", "bias"), ("buoy_tm02", "rmse")]


def flat(res):
    out = {"J": res["J"]}
    for k, s in KEYS[1:]:
        out["%s_%s" % (k, s)] = res["scores"][k][s]
    return out


def main():
    op, cases_f, prefix, out = sys.argv[1:5]
    cases = dict(line.split() for line in open(cases_f) if line.strip())
    base, _, _, _ = cal_obs.evaluate(op, "%s/m2/out_%s" % (S, cases["base"]), [])
    res = {"base": flat(base), "params": {}}
    print("base:", json.dumps(res["base"]))
    for p in ("BETAMAX", "SDSC2", "NLPROP", "BJGAM", "WIND"):
        tdir = "%s/cal/%s_%s" % (S, prefix, p)
        if not os.path.exists(os.path.join(tdir, "summary.json")):
            continue
        t, _, _, _ = cal_obs.evaluate(op, tdir, [(p, tdir)])
        tl = {"J": t["dJ"][p]}
        for k, s in KEYS[1:]:
            tl["%s_%s" % (k, s)] = t["d_scores"][p][k][s]
        row = {"tangent": tl}
        for suf, h in (("1", 0.01), ("01", 0.001)):
            a, b = cases.get("%s_p%s" % (p, suf)), cases.get("%s_m%s" % (p, suf))
            if not (a and b):
                continue
            ra = flat(cal_obs.evaluate(op, "%s/m2/out_%s" % (S, a), [])[0])
            rb = flat(cal_obs.evaluate(op, "%s/m2/out_%s" % (S, b), [])[0])
            row["fd_%s" % h] = {k: (ra[k] - rb[k]) / (2 * h) for k in ra}
        res["params"][p] = row
        for k in tl:
            fds = "  ".join("FD %g %+.4e" % (float(h[3:]), row[h][k]) for h in row if h.startswith("fd_"))
            ref = [row[h][k] for h in row if h.startswith("fd_")]
            rel = abs(tl[k] - ref[-1]) / max(abs(ref[-1]), 1e-30) if ref else float("nan")
            print("%-8s %-15s tangent %+.4e  %s  (relative to the smallest step %.2e)" % (p, k, tl[k], fds, rel), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
