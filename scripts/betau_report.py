"""Report of the BETAMAX(U) test (docs/differentiable.md, section 9.7).

    python scripts/betau_report.py OUT_DIR GN.json --runs p0:it0_value+s3:it3_value+bu:bu2_trial [--scratch S]
    python scripts/betau_report.py OUT_DIR bu1:gnu1.json:x+bus3:gnus3.json:x_new --runs ...

GN.json is the Gauss-Newton step (cal_gn.py) whose x is the final point of
BETAMAX(U) (its "x" with --at x, or its "x_new" with --at x_new) and whose
covariance gives the posterior uncertainty of ln BETAMAX(U) =
ln 1.75 + sum_k c_k phi_k(U): var = phi(U)^T C phi(U), unscaled and scaled
by the reduced chi-square. Several curves are drawn together when given as
LABEL:GN.json[:x|x_new] separated by +. The runs (directories of $S/cal) are compared on
the training and validation windows through the operators of the third cost
(op_<window>_cost3.npz: depth-aware buoy operator without the excluded
buoys): the bias and RMSE of the buoy H_s and of the altimeter blocks by
observed H_s and by the wind speed of the forcing at the observation (the
weights of the operator applied to the wind of $S/m2/forcing/wind.ww3).

OUT_DIR gets betau_curve.png, betau_bias.png and betau_report.json.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cal_obs  # noqa: E402
from obs_diag import binned, read_wind, robust  # noqa: E402

KNOTS = np.array([2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0])
B0 = 1.75


def phi(u, knots=KNOTS):
    """Hat functions of the knots at the wind speeds U (rows), as the kernels."""
    u = np.atleast_1d(np.asarray(u, float))
    uc = np.clip(u, knots[0], knots[-1])
    n = len(knots)
    out = np.zeros((len(u), n))
    for k in range(n):
        up = np.ones_like(uc) if k + 1 == n else np.maximum(0, (knots[min(k + 1, n - 1)] - uc) / (knots[min(k + 1, n - 1)] - knots[k]))
        lo = np.ones_like(uc) if k == 0 else np.maximum(0, (uc - knots[k - 1]) / (knots[k] - knots[max(k - 1, 0)]))
        out[:, k] = np.where(uc >= knots[k], up, lo)
    return out


def wind_at(o, wind, tags):
    """Wind speed at the altimeter blocks and at the buoy hours of operator O."""
    k, n, w = o["alt_k"], o["alt_n"], o["alt_w"]
    alt = np.full(k.shape, np.nan)
    for i, t in enumerate(tags):
        s = k == i
        if s.any() and t in wind:
            nxt = wind.get(tags[i + 1], wind[t]) if i + 1 < len(tags) else wind[t]
            alt[s] = (1 - w[s]) * wind[t][n[s]] + w[s] * nxt[n[s]]
    alt = alt.mean(1)
    bk, bb = o["bh_k"], o["bh_b"]
    nodes, wts = o["bw_nodes"], o["bw_w"]
    bh = np.full(len(bk), np.nan)
    for i, t in enumerate(tags):
        s = bk == i
        if s.any() and t in wind:
            nb, wb = nodes[bb[s]], wts[bb[s]]
            bh[s] = (np.where(wb > 0, wind[t][nb], 0.0) * wb).sum(1)
    return alt, bh


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("gn", help="GN.json, or LABEL:GN.json[:x|x_new] separated by + for several curves")
    ap.add_argument("--runs", default="p0:it0_value+s3:it3_value")
    ap.add_argument("--at", default="x_new", choices=("x", "x_new"))
    ap.add_argument("--windows", default="train+valid")
    ap.add_argument("--scratch", default="/scratch/jmdeloui/HWW3")
    a = ap.parse_args()
    S = a.scratch
    C = os.path.join(S, "cal")
    os.makedirs(a.out, exist_ok=True)
    runs = dict(r.split(":") for r in a.runs.split("+"))
    rep = {"runs": runs, "knots": KNOTS.tolist()}

    # the curves and their posterior uncertainty
    specs = []
    for s in a.gn.split("+"):
        p = s.split(":")
        specs.append(("BETAMAX(U)", p[0], a.at) if len(p) == 1 else (p[0], p[1], p[2] if len(p) > 2 else a.at))
    u = np.linspace(0, 26, 261)
    P = phi(u)
    curves = {}
    for lab, f, at in specs:
        g = json.load(open(f))
        names = g["names"]
        x = np.array(g[at])
        ib = [names.index("BETAU%d" % k) for k in range(len(KNOTS))]
        cov = np.array(g["cov"])[np.ix_(ib, ib)]
        chi = g["reduced_chi2"]
        lnb = np.log(B0) + P @ x[ib]
        sd = np.sqrt(np.einsum("ik,kl,il->i", P, cov, P))
        entry = {"gn": f, "at": at, "c": x[ib].tolist(), "beta_at_knots": (B0 * np.exp(x[ib])).tolist(),
                 "sd_ln_at_knots": np.sqrt(np.diag(cov)).tolist(), "sd_ln_scaled_at_knots": (np.sqrt(np.diag(cov) * chi)).tolist(),
                 "reduced_chi2": chi, "others": {n: {"x": float(x[i]), "factor": float(np.exp(x[i]))} for i, n in enumerate(names) if i not in ib}}
        curves[lab] = (lnb, sd, chi, B0 * np.exp(x[ib]), entry)
    rep["curves"] = {lab: c[4] for lab, c in curves.items()}
    rep["curve"] = next(iter(rep["curves"].values()))

    # the observations of both windows and the wind at them
    wsamp = []
    tab = {"buoy": {r: {"obs": [], "err": [], "wind": []} for r in runs}, "alt": {r: {"obs": [], "err": [], "wind": []} for r in runs}}
    for w in a.windows.split("+"):
        o = np.load(os.path.join(C, "op_%s_cost3.npz" % w))
        tags = [str(t) for t in o["tags"]]
        wind = read_wind(os.path.join(S, "m2", "forcing", "wind.ww3"), tags)
        walt, wbh = wind_at(o, wind, tags)
        wsamp.append(np.concatenate([np.ravel(v) for v in wind.values()]))
        for r, d in runs.items():
            alt, bh, _ = cal_obs.apply(o, os.path.join(C, d))
            for kind, mod, obs, ww in (("alt", alt, o["alt_obs"], walt), ("buoy", bh, o["bh_obs"], wbh)):
                tab[kind][r]["obs"].append(obs)
                tab[kind][r]["err"].append(mod - obs)
                tab[kind][r]["wind"].append(ww)
            rep.setdefault("scores", {}).setdefault(w, {})[r] = {"alt": robust(alt - o["alt_obs"]), "buoy_hs": robust(bh - o["bh_obs"])}
    eh = {"buoy": np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 10]), "alt": np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 15])}
    ew = np.array([0, 3, 6, 9, 12, 15, 20, 30])
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for i, kind in enumerate(("buoy", "alt")):
        for j, (key, edges, xl) in enumerate((("obs", eh[kind], "observed H_s (m)"), ("wind", ew, "wind speed of the forcing (m/s)"))):
            rows = {}
            for r in runs:
                xx = np.concatenate(tab[kind][r][key])
                ee = np.concatenate(tab[kind][r]["err"])
                rows[r] = binned(xx, ee, edges)
            rep.setdefault("binned", {}).setdefault(kind, {})[key] = rows
            ax = axs[i, j]
            for r, rr in rows.items():
                xc = [(q["lo"] + q["hi"]) / 2 for q in rr]
                ln, = ax.plot(xc, [100 * q["bias"] for q in rr], "o-", label="%s bias" % r)
                ax.plot(xc, [100 * q["rmse"] for q in rr], "--", color=ln.get_color(), lw=0.8, label="%s RMSE" % r)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel(xl)
            ax.set_ylabel("cm")
            ax.set_title("%s H_s, both windows, by %s" % ("buoy" if kind == "buoy" else "altimeter", "observed H_s" if key == "obs" else "wind speed"), fontsize=9)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "betau_bias.png"), dpi=90)
    plt.close(fig)

    # the curve with the wind distribution
    ws = np.concatenate(wsamp)
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, (lab, (lnb, sd, chi, bk, _)) in zip(("C0", "C2", "C4", "C1"), curves.items()):
        ax.fill_between(u, np.exp(lnb - np.sqrt(chi) * sd), np.exp(lnb + np.sqrt(chi) * sd), color=col, alpha=0.15,
                        label="%s: 1 sd, scaled by the reduced chi-square" % lab)
        ax.fill_between(u, np.exp(lnb - sd), np.exp(lnb + sd), color=col, alpha=0.35, label="%s: 1 sd" % lab)
        ax.plot(u, np.exp(lnb), "-", color=col, label=lab)
        ax.plot(KNOTS, bk, "o", color=col)
    ax.axhline(B0, color="k", lw=0.8, ls=":", label="p0 1.75")
    ax.axhline(1.37249382, color="C3", lw=0.8, ls="--", label="step 3 1.3725")
    ax.set_xlabel("wind speed at the node (m/s)")
    ax.set_ylabel("BETAMAX")
    ax2 = ax.twinx()
    ax2.hist(ws, bins=np.arange(0, 27, 1), color="0.6", alpha=0.3, density=True)
    ax2.set_ylabel("share of the node-hours of the windows")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "betau_curve.png"), dpi=100)
    plt.close(fig)
    rep["wind_quantiles"] = {str(q): float(np.quantile(ws, q)) for q in (0.5, 0.9, 0.99, 0.999)}
    json.dump(rep, open(os.path.join(a.out, "betau_report.json"), "w"), indent=1)
    print(json.dumps(rep["curves"], indent=1))
    for kind in ("buoy", "alt"):
        for key in ("obs", "wind"):
            print(kind, "by", key)
            for r, rr in rep["binned"][kind][key].items():
                print("  %-3s " % r + " ".join("[%g,%g) %+.1f" % (q["lo"], q["hi"], 100 * q["bias"]) for q in rr))


if __name__ == "__main__":
    main()
