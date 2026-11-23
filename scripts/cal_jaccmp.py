"""Compare the tangent Jacobians of two calibration evaluations.

    python scripts/cal_jaccmp.py EVAL0.npz EVAL1.npz BETAMAX+SDSC2+NLPROP

EVAL0 and EVAL1 are written by ``cal_obs.py eval --npz`` with tangent runs
at two points of the calibration (frozen Jacobian of option A: the first
is the one used for every step). For each parameter: the norm of its
column of the weighted Jacobian (residuals divided by their sigma) at both
points, the relative norm of the difference and the cosine between the
two columns, per block (altimeters, buoy H_s, buoy T_m02) and overall;
then the Gauss-Newton matrices A^T A at both points and the largest
relative change of their entries.
"""
import sys

import numpy as np


def cols(e, names):
    sa, sb, st = e["sigma"]
    out = {}
    for n in names:
        out[n] = {"alt": e["d_alt_" + n] / sa, "bhs": e["d_bhs_" + n] / sb, "btm": e["d_btm_" + n] / st}
        out[n]["all"] = np.concatenate([out[n]["alt"], out[n]["bhs"], out[n]["btm"]])
    return out


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    e0, e1 = np.load(sys.argv[1]), np.load(sys.argv[2])
    names = sys.argv[3].split("+")
    c0, c1 = cols(e0, names), cols(e1, names)
    print("%-8s %-4s %12s %12s %10s %8s" % ("param", "blk", "|A0|", "|A1|", "|A1-A0|/|A0|", "cos"))
    for n in names:
        for b in ("alt", "bhs", "btm", "all"):
            a, c = c0[n][b], c1[n][b]
            if a.shape != c.shape:
                raise SystemExit("%s %s: %s residuals against %s" % (n, b, a.shape, c.shape))
            na, nc = np.linalg.norm(a), np.linalg.norm(c)
            print("%-8s %-4s %12.4e %12.4e %10.4f %8.5f" % (
                n, b, na, nc, np.linalg.norm(c - a) / max(na, 1e-300), a @ c / max(na * nc, 1e-300)))
    A0 = np.stack([c0[n]["all"] for n in names], 1)
    A1 = np.stack([c1[n]["all"] for n in names], 1)
    H0, H1 = A0.T @ A0, A1.T @ A1
    print("A^T A at the first point:\n" + "\n".join("  " + " ".join("%+.4e" % v for v in row) for row in H0))
    print("A^T A at the second point:\n" + "\n".join("  " + " ".join("%+.4e" % v for v in row) for row in H1))
    print("largest relative change of A^T A: %.4f" % (np.abs(H1 - H0).max() / np.abs(H0).max()))


if __name__ == "__main__":
    main()
