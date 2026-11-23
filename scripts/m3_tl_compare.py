"""Tangent of a GPU run against central finite differences (milestone M3 of
docs/differentiable.md).

TANGENT is the output directory of `hww3-prop run ... --tangent NAME` (the
hourly tangent of H_s, dhs_YYYYMMDDHH.f64 or .f32, and the tangent of the
total energy, "denergy" in summary.json, for a relative change of NAME).
Each H:PLUS:MINUS names the output directories of the same run with NAME
multiplied by 1 + H and 1 - H; the central difference is
(R(+H) - R(-H)) / (2 H). The H_s files in double precision (--out64) are
used when present.

For the total energy at each time of summary.json the tangent is compared
with every difference. For the hourly H_s fields, on the nodes where all
runs are defined: with one step, the relative L2 difference of the fields,
the difference of their means and the percentiles of the absolute
difference; with two steps H1 > H2, a node is smooth where the two
differences agree to TOL of its scale s = max(|d2|, 1e-3 rms(d2)) (no
discrete choice of the model changes within the steps), and on the smooth
nodes the tangent is compared with the difference of step H2, relative to
s, with the count of nodes above TARGET (default: TOL).

usage:
    python scripts/m3_tl_compare.py TANGENT H1:PLUS1:MINUS1 [H2:PLUS2:MINUS2] [--tol TOL] [--target T] [--json OUT]
"""
import glob
import json
import os
import sys

import numpy as np


def field(d, name, tag):
    for ext, dt in ((".f64", "<f8"), (".f32", "<f4")):
        f = os.path.join(d, "%s_%s%s" % (name, tag, ext))
        if os.path.exists(f):
            return np.fromfile(f, dt).astype(np.float64), ext
    return None, None


def main():
    args = sys.argv[1:]
    tol, out, target = 1e-4, None, None
    if "--target" in args:
        i = args.index("--target")
        target = float(args[i + 1])
        del args[i:i + 2]
    if "--tol" in args:
        i = args.index("--tol")
        tol = float(args[i + 1])
        del args[i:i + 2]
    if "--json" in args:
        i = args.index("--json")
        out = args[i + 1]
        del args[i:i + 2]
    target = tol if target is None else target
    tdir = args[0]
    pairs = []
    for a in args[1:]:
        h, p, m = a.split(":")
        pairs.append((float(h), p, m))
    pairs.sort(key=lambda x: -x[0])
    st = json.load(open(os.path.join(tdir, "summary.json")))
    sums = [(json.load(open(os.path.join(p, "summary.json"))), json.load(open(os.path.join(m, "summary.json")))) for _, p, m in pairs]
    energy = []
    for k, et in enumerate(st["energy"]):
        row = {"time": et["time"], "energy": et["energy"], "tl": et.get("denergy", 0.0), "fd": {}}
        for (h, _, _), (sp, sm) in zip(pairs, sums):
            ep, em = sp["energy"][k], sm["energy"][k]
            if not (et["time"] == ep["time"] == em["time"]):
                sys.exit("energy times differ: %s %s %s" % (et["time"], ep["time"], em["time"]))
            row["fd"][str(h)] = (ep["energy"] - em["energy"]) / (2 * h)
        energy.append(row)
        print("energy %s: E %.6e, dE/dln tangent %.8e, " % (et["time"], et["energy"], row["tl"])
              + ", ".join("h = %g: %.8e (relative %.2e)" % (float(h), v, abs(row["tl"] - v) / max(abs(v), 1e-300)) for h, v in row["fd"].items()))
    hours = []
    for f in sorted(glob.glob(os.path.join(tdir, "dhs_*.f32"))):
        tag = os.path.basename(f)[4:-4]
        tl, ext = field(tdir, "dhs", tag)
        fds, ok = [], tl > -900
        for h, p, m in pairs:
            hp, _ = field(p, "hs", tag)
            hm, _ = field(m, "hs", tag)
            if hp is None or hm is None:
                break
            ok &= (hp > -900) & (hm > -900)
            fds.append((hp - hm) / (2 * h))
        if len(fds) != len(pairs):
            continue
        tl = tl[ok]
        fds = [d[ok] for d in fds]
        d2 = fds[-1]
        rms = np.sqrt(np.mean(d2 ** 2))
        r = {"time": tag, "nodes": int(ok.sum()), "files": ext, "rms_fd": float(rms),
             "mean_tl": float(tl.mean()), "mean_fd": float(d2.mean()),
             "mean_rel": float(abs(tl.mean() - d2.mean()) / max(abs(d2.mean()), 1e-300)),
             "l2_rel": float(np.linalg.norm(tl - d2) / max(np.linalg.norm(d2), 1e-300))}
        if len(fds) == 2:
            s = np.maximum(np.abs(d2), 1e-3 * rms)
            smooth = np.abs(fds[0] - d2) <= tol * s
            e = np.abs(tl - d2)[smooth] / s[smooth]
            r.update({
                "smooth": int(smooth.sum()),
                "e_p50": float(np.percentile(e, 50)) if e.size else float("nan"),
                "e_p99": float(np.percentile(e, 99)) if e.size else float("nan"),
                "e_max": float(e.max()) if e.size else float("nan"),
                "above_target": int((e > target).sum()),
                "mean_tl_smooth": float(tl[smooth].mean()) if e.size else float("nan"),
                "mean_fd_smooth": float(d2[smooth].mean()) if e.size else float("nan"),
                "l2_rel_smooth": float(np.linalg.norm((tl - d2)[smooth]) / max(np.linalg.norm(d2[smooth]), 1e-300)),
            })
        else:
            diff = np.abs(tl - d2)
            r.update({"p50": float(np.percentile(diff, 50)), "p99": float(np.percentile(diff, 99)), "max": float(diff.max())})
        hours.append(r)
    for r in hours[:: max(1, len(hours) // 8)] + hours[-1:]:
        line = ("H_s %s (%s): %d nodes, rms dHs/dln %.3e m; mean tangent %.6e, difference %.6e (relative %.2e), L2 %.2e"
                % (r["time"], r["files"], r["nodes"], r["rms_fd"], r["mean_tl"], r["mean_fd"], r["mean_rel"], r["l2_rel"]))
        if "smooth" in r:
            line += ("; smooth to %g: %d (%.1f %%): tangent against the difference p50 %.2e, p99 %.2e, max %.2e, %d above %g, L2 %.2e"
                     % (tol, r["smooth"], 100.0 * r["smooth"] / max(r["nodes"], 1), r["e_p50"], r["e_p99"], r["e_max"], r["above_target"], target, r["l2_rel_smooth"]))
        else:
            line += "; |diff| p50 %.2e, p99 %.2e, max %.2e m" % (r["p50"], r["p99"], r["max"])
        print(line)
    if out:
        json.dump({"tangent": tdir, "pairs": pairs, "tol": tol, "target": target, "energy": energy, "hs": hours}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
