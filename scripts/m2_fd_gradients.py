"""Finite-difference gradients of the GPU model with respect to source
parameters and the wind (milestones M2 and M3 of docs/differentiable.md).

Each case of CASES (lines "NAME JOBID": base, and P_p1 / P_m1 for a
parameter P multiplied by 1 +/- h, h = 0.01 for _p1/_m1 and 0.001 for
_p01/_m01) is the output directory $S/m2/out_JOBID of a hww3-prop run. For
each case the responses are: the total energy and the mean H_s at the last
hour, and the altimeter and buoy statistics of validate_altimetry.py and
validate_buoys.py (run on the GPU fields made into a netCDF file shaped
like TEMPLATE, with VAL_SPINUP_DAYS, VAL_END and VAL_MIN_HOURS from the
environment). The gradient with respect to the relative change of the
parameter is (R(+h) - R(-h)) / (2 h).

usage:
    python scripts/m2_fd_gradients.py CASES TEMPLATE.nc NODES.npz GRID_DIR OUT.json
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

S = "/scratch/jmdeloui/HWW3"
HERE = os.path.dirname(os.path.abspath(__file__))


def responses(name, job, template, nodes, grid, work):
    out = os.path.join(S, "m2", "out_%s" % job)
    nc = os.path.join(work, "%s.nc" % name)
    py = sys.executable
    subprocess.run([py, os.path.join(HERE, "gpu_to_nc.py"), template, out, nc], check=True, capture_output=True)
    subprocess.run([py, os.path.join(HERE, "validate_altimetry.py"), nc, nodes, os.path.join(work, name), os.path.join(S, "obs", "l3")],
                   check=True, capture_output=True)
    subprocess.run([py, os.path.join(HERE, "validate_buoys.py"), os.path.join(work, name + "_buoys"), "%s:%s:%s" % (name, nc, grid),
                    "--buoys", os.path.join(S, "obs", "insitu_MO", "202601")], check=True, capture_output=True)
    alt = json.load(open(os.path.join(work, name + "_stats.json")))["all"]
    b = json.load(open(os.path.join(work, name + "_buoys_stats.json")))
    summ = json.load(open(os.path.join(out, "summary.json")))
    last = sorted(glob.glob(os.path.join(out, "hs_*.f32")))[-1]
    hs = np.fromfile(last, "<f4")
    hs = hs[hs > -900]
    r = {
        "energy": summ["energy"][-1]["energy"],
        "mean_hs": float(hs.mean()),
        "alt_bias": alt["bias"], "alt_rmse": alt["rmse"], "alt_n": alt["n"],
        "buoy_hs_bias": b["hs"]["all"][name]["bias"], "buoy_hs_rmse": b["hs"]["all"][name]["rmse"],
        "buoy_tm02_bias": b["tm02"]["all"][name]["bias"], "buoy_tm02_rmse": b["tm02"]["all"][name]["rmse"],
        "buoy_n": b["hs"]["all"][name]["n"],
    }
    os.remove(nc)
    return r


def main():
    cases_f, template, nodes, grid, out = sys.argv[1:6]
    cases = dict(line.split() for line in open(cases_f) if line.strip())
    work = os.path.join(os.path.dirname(out), "fd_work")
    os.makedirs(work, exist_ok=True)
    res = {}
    for name, job in cases.items():
        if not os.path.exists(os.path.join(S, "m2", "out_%s" % job, "summary.json")):
            print("skip", name, job, "(no output)")
            continue
        res[name] = responses(name, job, template, nodes, grid, work)
        print(name, {k: round(v, 6) for k, v in res[name].items()}, flush=True)
    grads = {}
    for name in res:
        for suf, h in (("_p1", 0.01), ("_p01", 0.001)):
            if not name.endswith(suf):
                continue
            p = name[: -len(suf)]
            m = p + suf.replace("_p", "_m")
            if m not in res:
                continue
            g = {k: (res[name][k] - res[m][k]) / (2 * h) for k in res[name] if not k.endswith("_n")}
            grads["%s (h = %g)" % (p, h)] = g
            print("d/dln", p, "h =", h, {k: "%.4g" % v for k, v in g.items()}, flush=True)
    json.dump({"responses": res, "gradients_per_relative_change": grads}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
