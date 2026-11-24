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
given scaled by the reduced chi-square J_data / (N - n). If EVAL.npz holds a
Huber threshold (third cost, cal_obs.py), the data term is
sum rho(r / sigma) with rho(z) = z^2 for |z| <= delta and 2 delta |z| -
delta^2 beyond, and the step is that of iteratively reweighted least
squares: H = A^T W A + I / SIGMA_P^2, g = A^T W r + x / SIGMA_P^2 with the
weights W = min(1, delta / |r / sigma|) at the current residuals.

BETAMAX(U): the coefficients BETAU0, BETAU1, .. of ln BETAMAX on the knots
of the wind speed (hww3-prop --betau) are already logarithms, x = c_k, and
their tangents are for a change of 1 of c_k; with --smooth S the prior adds
the second differences of consecutive coefficients,
sum_k ((c_{k-1} - 2 c_k + c_{k+1}) / S)^2, to keep the curve smooth.

usage:
    python scripts/cal_gn.py EVAL.npz X.json OUT.json [--lambda L] [--sigma-p S] [--max-step M] [--names A+B+C] [--smooth S]
With --names only those parameters are controlled (the others stay at p0:
BJGAM was frozen at 0.73 after the first step).
X.json: {"names": [...], "x": [...]} (missing: x = 0); OUT.json gets the
step, the new x and factors exp(x), the predicted cost, H, the covariance
and the correlations.
"""
import json
import sys

import numpy as np


def prior(names, sp, smooth=0.0):
    """Inverse prior covariance: I / SP^2, plus the second differences of the
    consecutive BETAMAX(U) coefficients over SMOOTH^2."""
    P = np.eye(len(names)) / sp ** 2
    bu = sorted((int(n[5:]), i) for i, n in enumerate(names) if n.startswith("BETAU") and n[5:].isdigit())
    if smooth > 0 and len(bu) >= 3:
        ks = [k for k, _ in bu]
        if ks != list(range(ks[0], ks[0] + len(ks))):
            sys.exit("the smoothness prior needs consecutive BETAU coefficients, got %s" % ks)
        idx = [i for _, i in bu]
        D = np.zeros((len(idx) - 2, len(names)))
        for r in range(len(idx) - 2):
            D[r, idx[r]], D[r, idx[r + 1]], D[r, idx[r + 2]] = 1.0, -2.0, 1.0
        P = P + D.T @ D / smooth ** 2
    return P


def main():
    a = sys.argv[1:]
    opts = {"--lambda": "0", "--sigma-p": "0.25", "--max-step": "0.3", "--names": "", "--smooth": "0"}
    for k in list(opts):
        if k in a:
            i = a.index(k)
            opts[k] = a[i + 1]
            del a[i:i + 2]
    lam, sp, mx = float(opts["--lambda"]), float(opts["--sigma-p"]), float(opts["--max-step"])
    e = np.load(a[0])
    names = [str(n) for n in e["names"]]
    if opts["--names"]:
        sub = opts["--names"].split("+")
        missing = [n for n in sub if n not in names]
        if missing:
            sys.exit("no tangent for %s in %s" % (missing, a[0]))
        names = sub
    try:
        xs = json.load(open(a[1]))
        x = np.array([xs["x"][xs["names"].index(n)] for n in names])
    except FileNotFoundError:
        x = np.zeros(len(names))
    sa, sb, st = e["sigma"]
    if "s_bhs" in e:
        # observation errors per buoy hour (representativeness)
        sb, st = e["s_bhs"], e["s_btm"]
    r = np.concatenate([e["r_alt"] / sa, e["r_bhs"] / sb, e["r_btm"] / st])
    A = np.stack([np.concatenate([e["d_alt_" + n] / sa, e["d_bhs_" + n] / sb, e["d_btm_" + n] / st]) for n in names], 1)
    P = prior(names, sp, float(opts["--smooth"]))
    delta = float(e["huber"]) if "huber" in e else 0.0
    if delta > 0:
        def rho(z):
            az = np.abs(z)
            return np.where(az <= delta, z * z, 2 * delta * az - delta * delta)
        wt = np.minimum(1.0, delta / np.maximum(np.abs(r), 1e-12))
    else:
        def rho(z):
            return z * z
        wt = np.ones(len(r))
    H = A.T @ (wt[:, None] * A) + P
    g = A.T @ (wt * r) + P @ x
    jd, jp = float(np.sum(rho(r))), float(x @ P @ x)
    dx = -np.linalg.solve(H + lam * np.diag(np.diag(H)), g)
    scale = min(1.0, mx / np.abs(dx).max()) if np.abs(dx).max() > 0 else 1.0
    dx *= scale
    xn = x + dx
    rp = r + A @ dx
    jpred = float(np.sum(rho(rp)) + xn @ P @ xn)
    C = np.linalg.inv(H)
    n = len(r)
    chi = jd / max(n - len(names), 1)
    sd = np.sqrt(np.diag(C))
    corr = C / np.outer(sd, sd)
    out = {"names": names, "x": x.tolist(), "J": jd + jp, "J_data": jd, "J_prior": jp, "n_obs": n,
           "lambda": lam, "sigma_p": sp, "smooth": float(opts["--smooth"]), "max_step": mx, "step_scale": scale, "huber": delta,
           "share_beyond_huber": float(np.mean(np.abs(r) > delta)) if delta > 0 else 0.0,
           "dx": dx.tolist(), "x_new": xn.tolist(), "factor_new": np.exp(xn).tolist(), "J_pred": jpred,
           "gradient": g.tolist(), "H": H.tolist(), "cov": C.tolist(), "sd_ln": sd.tolist(),
           "reduced_chi2": chi, "sd_ln_scaled": (sd * np.sqrt(chi)).tolist(), "corr": corr.tolist()}
    json.dump(out, open(a[2], "w"), indent=1)
    print("J %.6e (data %.6e, prior %.3e) over %d residuals%s" % (jd + jp, jd, jp, n,
          ", Huber delta %.2f (%.1f %% beyond)" % (delta, 100 * np.mean(np.abs(r) > delta)) if delta > 0 else ""))
    for i, nm in enumerate(names):
        print("%-8s x %+.5f  grad %+.4e  dx %+.5f  -> factor %.5f  (sd %.4f, scaled %.4f)"
              % (nm, x[i], g[i], dx[i], np.exp(xn[i]), sd[i], sd[i] * np.sqrt(chi)))
    print("predicted J %.6e (%.3e relative change); step scale %.3f" % (jpred, jpred / (jd + jp) - 1, scale))
    print("correlations:\n" + "\n".join("  " + " ".join("%+.3f" % v for v in row) for row in corr))


if __name__ == "__main__":
    main()
