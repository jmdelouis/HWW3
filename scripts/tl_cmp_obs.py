"""Two tangent runs compared at the observations of an operator.

    python scripts/tl_cmp_obs.py OP.npz OLD NEW

The derivatives of the altimeter blocks, the buoy H_s and the buoy T_m02
of operator OP (cal_obs.py) along the tangent runs OLD and NEW (same
direction, e.g. without and with a change of the tangent model): the
relative L2 difference ||d_new - d_old|| / ||d_old||, the largest
difference relative to the largest derivative, and the correlation.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cal_obs  # noqa: E402


def main():
    opf, old, new = sys.argv[1:4]
    o = np.load(opf)
    a = cal_obs.apply(o, old, prefix="d")
    b = cal_obs.apply(o, new, prefix="d")
    for name, x, y in zip(("altimeters", "buoy H_s", "buoy T_m02"), a, b):
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        n = np.linalg.norm(x)
        print("%-11s %6d: relative L2 %.3e, largest difference %.3e of the largest derivative %.3e, correlation %.6f"
              % (name, len(x), np.linalg.norm(y - x) / max(n, 1e-300), np.abs(y - x).max() / max(np.abs(x).max(), 1e-300),
                 np.abs(x).max(), np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan))


if __name__ == "__main__":
    main()
