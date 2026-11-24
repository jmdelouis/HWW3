"""One point of the calibration under the three costs, on both windows.

    python scripts/cal_point.py RUN X.json OUT.json [--windows train+valid] [--cal DIR]

RUN is a value run (hourly H_s and T_m02), X.json its control vector
(names and x = ln p - ln p0, written by cal_gn.py; "none" for p0). For
each window (the operators op_<window>.npz of cal_obs.py in DIR, default
$S/cal) and each cost: the first (op_<window>, nearest node, sigma 0.30 m
and 1.0 s), the depth-aware operator alone (op_<window>_depth) and the
second cost (op_<window>_cost2, with the representativeness errors) and the
third (op_<window>_cost3: without the excluded buoys, Huber), it
gives J and its parts (altimeters, buoy H_s, buoy T_m02), the prior term
sum (x / 0.25)^2 separately, and the unweighted scores (bias and RMSE of
the altimeters, and of the buoy H_s and T_m02 through the nearest node and
through the depth-aware operator). A window the run does not cover is
skipped. OUT.json holds everything.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cal_obs  # noqa: E402

COSTS = (("first", ""), ("depth", "_depth"), ("second", "_cost2"), ("third", "_cost3"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("x")
    ap.add_argument("out")
    ap.add_argument("--windows", default="train+valid")
    ap.add_argument("--cal", default="/scratch/jmdeloui/HWW3/cal")
    ap.add_argument("--sigma-p", type=float, default=0.25)
    a = ap.parse_args()
    x = np.zeros(0) if a.x == "none" else np.array(json.load(open(a.x))["x"])
    prior = float(np.sum((x / a.sigma_p) ** 2))
    out = {"run": a.run, "x": x.tolist(), "prior": prior, "windows": {}}
    print("%s: prior %.3f" % (a.run, prior))
    for w in a.windows.split("+"):
        res = {}
        try:
            for name, suf in COSTS:
                opf = os.path.join(a.cal, "op_%s%s.npz" % (w, suf))
                if not os.path.exists(opf):
                    continue
                r, _, _, _ = cal_obs.evaluate(opf, a.run, [])
                res[name] = {"J": r["J"], "J_parts": r["J_parts"], "scores": r["scores"], "J_l2": r["J_l2"]}
        except (FileNotFoundError, KeyError, ValueError, TypeError) as e:
            print("  %s: skipped (%s)" % (w, e))
            continue
        out["windows"][w] = res
        for name, _ in COSTS:
            if name not in res:
                continue
            q = res[name]
            print("  %-5s %-6s J %10.3f (alt %9.3f, buoy H_s %9.3f, buoy T_m02 %9.3f) + prior %.3f = %10.3f"
                  % (w, name, q["J"], q["J_parts"]["alt"], q["J_parts"]["bhs"], q["J_parts"]["btm"], prior, q["J"] + prior))
        if "third" in res:
            s3 = res["third"]["scores"]
            print("  %-5s third-cost scores (without the excluded buoys): buoy H_s %+.2f/%.2f cm, T_m02 %+.3f/%.3f s"
                  % (w, 100 * s3["buoy_hs"]["bias"], 100 * s3["buoy_hs"]["rmse"], s3["buoy_tm02"]["bias"], s3["buoy_tm02"]["rmse"]))
        s1, s2 = res["first"]["scores"], res["depth"]["scores"]
        print("  %-5s scores: alt %+.2f/%.2f cm; buoy H_s %+.2f/%.2f cm (nearest), %+.2f/%.2f cm (depth-aware); "
              "T_m02 %+.3f/%.3f s (nearest), %+.3f/%.3f s (depth-aware)"
              % (w, 100 * s1["alt"]["bias"], 100 * s1["alt"]["rmse"], 100 * s1["buoy_hs"]["bias"], 100 * s1["buoy_hs"]["rmse"],
                 100 * s2["buoy_hs"]["bias"], 100 * s2["buoy_hs"]["rmse"], s1["buoy_tm02"]["bias"], s1["buoy_tm02"]["rmse"],
                 s2["buoy_tm02"]["bias"], s2["buoy_tm02"]["rmse"]))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
