"""Two tangent runs compared at the observations of an operator.

    python scripts/tl_cmp_obs.py OP.npz OLD NEW [--partial]

The derivatives of the altimeter blocks, the buoy H_s and the buoy T_m02
of operator OP (cal_obs.py) along the tangent runs OLD and NEW (same
direction, e.g. without and with a change of the tangent model, or the f32
tangent against the f64 one): the relative L2 difference
||d_new - d_old|| / ||d_old||, the largest difference relative to the
largest derivative, and the correlation, on the observations where both
are finite, with the number where OLD is finite and NEW is not. With
--partial the runs may cover only part of the window of OP (such as the
24 h runs of scripts/opt_bench_hpc.slurm): the observations of the hours
missing from a run are left out.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cal_obs  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 3:
        raise SystemExit(__doc__)
    opf, old, new = args
    if "--partial" in sys.argv:
        field = cal_obs.field
        size = {}

        def partial(d, name, tag, idx=None):
            try:
                return field(d, name, tag, idx)
            except FileNotFoundError:
                if idx is not None:
                    return np.full(len(idx), np.nan)
                if d not in size:
                    # the number of nodes, from any hourly H_s field of the run
                    f = sorted(x for x in os.listdir(d) if x.startswith("hs_"))[0]
                    size[d] = len(field(d, "hs", f[3:].split(".")[0]))
                return np.full(size[d], np.nan)

        cal_obs.field = partial
    o = np.load(opf)
    a = cal_obs.apply(o, old, prefix="d")
    b = cal_obs.apply(o, new, prefix="d")
    for name, x, y in zip(("altimeters", "buoy H_s", "buoy T_m02"), a, b):
        lost = int((np.isfinite(x) & ~np.isfinite(y)).sum())
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) == 0:
            print("%-11s      0 observations in both runs (%d not finite in NEW)" % (name, lost))
            continue
        n = np.linalg.norm(x)
        print("%-11s %6d: relative L2 %.3e, largest difference %.3e of the largest derivative %.3e, correlation %.6f; %d not finite in NEW"
              % (name, len(x), np.linalg.norm(y - x) / max(n, 1e-300), np.abs(y - x).max() / max(np.abs(x).max(), 1e-300),
                 np.abs(x).max(), np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan, lost))


if __name__ == "__main__":
    main()
