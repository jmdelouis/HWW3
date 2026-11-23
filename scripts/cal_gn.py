"""One Gauss-Newton (Levenberg-Marquardt) step of the calibration of the
GPU model (docs/differentiable.md, section 9.5).

EVAL.npz (cal_obs.py eval --npz) holds the residuals r of the observations
at the current parameters and their derivatives along the tangents of
ln p, one per parameter. With x = ln p - ln p0, the cost is
    J(x) = sum (r / sigma)^2 + sum (x / SIGMA_P)^2
and the step solves (H + L diag H) dx = -g with H = A^T A + I / SIGMA_P^2,
g = A^T r + x / SIGMA_P^2, A the derivatives of r / sigma; each component
of dx is limited to MAX_STEP. H^-1 is the posterior covariance of ln p for
Gaussian independent errors of the given sigmas; since the errors of
neighbouring altimeter blocks and buoy hours are correlated, it is also
given scaled by the reduced chi-square J_data / (N - n).

usage:
    python scripts/cal_gn.py EVAL.npz X.json OUT.json [--lambda L] [--sigma-p S] [--max-step M]
X.json: {"names": [...], "x": [...]} (missing: x = 0); OUT.json gets the
step, the new x and factors exp(x), the predicted cost, H, the covariance
and the correlations.
"""
import json
import sys

import numpy as np


def main():
    a = sys.argv[1:]
    opts = {"--lambda": "0", "--sigma-p": "0.25", "--max-step": "0.3"}
    for k in list(opts):
        if k in a:
            i = a.index(k)
            opts[k] = a[i + 1]
            del a[i:i + 2]
    lam, sp, mx = float(opts["--lambda"]), float(opts["--sigma-p"]), float(opts["--max-step"])
    e = np.load(a[0])
    names = [str(n) for n in e["names"]]
    try:
        xs = json.load(open(a[1]))
        x = np.array([xs["x"][xs["names"].index(n)] for n in names])
    except FileNotFoundError:
        x = np.zeros(len(names))
    sa, sb, st = e["sigma"]
    r = np.concatenate([e["r_alt"] / sa, e["r_bhs"] / sb, e["r_btm"] / st])
    A = np.stack([np.concatenate([e["d_alt_" + n] / sa, e["d_bhs_" + n] / sb, e["d_btm_" + n] / st]) for n in names], 1)
    P = np.eye(len(names)) / sp ** 2
    H = A.T @ A + P
    g = A.T @ r + P @ x
    jd, jp = float(r @ r), float(x @ P @ x)
    dx = -np.linalg.solve(H + lam * np.diag(np.diag(H)), g)
    scale = min(1.0, mx / np.abs(dx).max()) if np.abs(dx).max() > 0 else 1.0
    dx *= scale
    xn = x + dx
    rp = r + A @ dx
    jpred = float(rp @ rp + xn @ P @ xn)
    C = np.linalg.inv(H)
    n = len(r)
    chi = jd / max(n - len(names), 1)
    sd = np.sqrt(np.diag(C))
    corr = C / np.outer(sd, sd)
    out = {"names": names, "x": x.tolist(), "J": jd + jp, "J_data": jd, "J_prior": jp, "n_obs": n,
           "lambda": lam, "sigma_p": sp, "max_step": mx, "step_scale": scale,
           "dx": dx.tolist(), "x_new": xn.tolist(), "factor_new": np.exp(xn).tolist(), "J_pred": jpred,
           "gradient": g.tolist(), "H": H.tolist(), "cov": C.tolist(), "sd_ln": sd.tolist(),
           "reduced_chi2": chi, "sd_ln_scaled": (sd * np.sqrt(chi)).tolist(), "corr": corr.tolist()}
    json.dump(out, open(a[2], "w"), indent=1)
    print("J %.6e (data %.6e, prior %.3e) over %d residuals" % (jd + jp, jd, jp, n))
    for i, nm in enumerate(names):
        print("%-8s x %+.5f  grad %+.4e  dx %+.5f  -> factor %.5f  (sd %.4f, scaled %.4f)"
              % (nm, x[i], g[i], dx[i], np.exp(xn[i]), sd[i], sd[i] * np.sqrt(chi)))
    print("predicted J %.6e (%.3e relative change); step scale %.3f" % (jpred, jpred / (jd + jp) - 1, scale))
    print("correlations:\n" + "\n".join("  " + " ".join("%+.3f" % v for v in row) for row in corr))


if __name__ == "__main__":
    main()
