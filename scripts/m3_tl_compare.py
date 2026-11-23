"""Tangent of a GPU run against central finite differences (milestone M3 of
docs/differentiable.md).

TANGENT is the output directory of `hww3-prop run ... --tangent NAME` (the
hourly tangent of H_s, dhs_YYYYMMDDHH.f32, and the tangent of the total
energy, "denergy" in summary.json, for a relative change of NAME); PLUS and
MINUS those of the same run with NAME multiplied by 1 + H and 1 - H. For
the total energy at each time of summary.json the tangent is compared with
(E(+H) - E(-H)) / (2 H); for the hourly H_s fields, on the nodes where all
three are defined, the relative L2 difference of the fields, the relative
difference of their means and the percentiles of the absolute difference
(m per unit relative change).

usage:
    python scripts/m3_tl_compare.py TANGENT PLUS MINUS H [OUT.json]
"""
import glob
import json
import os
import sys

import numpy as np


def main():
    tdir, pdir, mdir, h = sys.argv[1:5]
    h = float(h)
    st, sp, sm = (json.load(open(os.path.join(d, "summary.json"))) for d in (tdir, pdir, mdir))
    energy = []
    for et, ep, em in zip(st["energy"], sp["energy"], sm["energy"]):
        if not (et["time"] == ep["time"] == em["time"]):
            sys.exit("energy times differ: %s %s %s" % (et["time"], ep["time"], em["time"]))
        fd = (ep["energy"] - em["energy"]) / (2 * h)
        tl = et["denergy"]
        energy.append({"time": et["time"], "energy": et["energy"], "tl": tl, "fd": fd, "rel": abs(tl - fd) / max(abs(fd), 1e-300)})
        print("energy %s: E %.6e, dE/dln tangent %.6e, finite difference %.6e, relative difference %.2e"
              % (et["time"], et["energy"], tl, fd, energy[-1]["rel"]))
    hours = []
    for f in sorted(glob.glob(os.path.join(tdir, "dhs_*.f32"))):
        tag = os.path.basename(f)[4:-4]
        fp, fm = (os.path.join(d, "hs_%s.f32" % tag) for d in (pdir, mdir))
        if not (os.path.exists(fp) and os.path.exists(fm)):
            continue
        d = np.fromfile(f, "<f4").astype(np.float64)
        hp = np.fromfile(fp, "<f4").astype(np.float64)
        hm = np.fromfile(fm, "<f4").astype(np.float64)
        ok = (d > -900) & (hp > -900) & (hm > -900)
        tl, fd = d[ok], (hp[ok] - hm[ok]) / (2 * h)
        diff = np.abs(tl - fd)
        nfd = np.linalg.norm(fd)
        r = {
            "time": tag,
            "nodes": int(ok.sum()),
            "l2_rel": float(np.linalg.norm(tl - fd) / max(nfd, 1e-300)),
            "mean_tl": float(tl.mean()),
            "mean_fd": float(fd.mean()),
            "mean_rel": float(abs(tl.mean() - fd.mean()) / max(abs(fd.mean()), 1e-300)),
            "rms_fd": float(nfd / np.sqrt(max(ok.sum(), 1))),
            "p50": float(np.percentile(diff, 50)),
            "p99": float(np.percentile(diff, 99)),
            "max": float(diff.max()),
            "corr": float(np.corrcoef(tl, fd)[0, 1]) if nfd > 0 else float("nan"),
        }
        hours.append(r)
    for r in hours[:: max(1, len(hours) // 8)] + hours[-1:]:
        print("H_s %s: %d nodes, mean dHs/dln tangent %.5e, finite difference %.5e (relative %.2e); "
              "L2 %.2e of rms %.3e; |diff| p50 %.2e, p99 %.2e, max %.2e m; correlation %.8f"
              % (r["time"], r["nodes"], r["mean_tl"], r["mean_fd"], r["mean_rel"], r["l2_rel"], r["rms_fd"], r["p50"], r["p99"], r["max"], r["corr"]))
    if len(sys.argv) > 5:
        json.dump({"tangent": tdir, "plus": pdir, "minus": mdir, "h": h, "energy": energy, "hs": hours}, open(sys.argv[5], "w"), indent=1)


if __name__ == "__main__":
    main()
