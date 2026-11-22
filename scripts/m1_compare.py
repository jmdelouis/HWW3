"""Compare hourly H_s and T_m02 node fields of the GPU propagation model
(hww3-prop) with those of a WW3 run, as the M1 targets of
docs/differentiable.md (section 6) are defined.

The fields are little-endian float32 files with one value per node and
-999.9 where undefined, as hww3-prop writes them. To compare with a WW3
netCDF output directly, use scripts/m1_compare_nc.py. For each node the RMS difference over the
hours is computed; the targets are an H_s RMS difference below 1 cm at the
median node and below 5 cm at the 99th percentile.

usage:
    python scripts/m1_compare.py GPU_DIR WW3_DIR HOUR [HOUR ...] [--out stats.json]
        GPU_DIR/hs_<YYYYMMDDHH>.f32, GPU_DIR/t02_<...>.f32 against the same
        names in WW3_DIR
    python scripts/m1_compare.py --pair A.f32 B.f32
        one field against another (for instance H_s of a restart)
"""
import json
import os
import sys

import numpy as np

UNDEF = -999.0


def load(p):
    return np.fromfile(p, dtype="<f4")


def pair(a, b):
    va, vb = load(a), load(b)
    if va.size != vb.size:
        sys.exit("sizes differ: %d, %d" % (va.size, vb.size))
    ok = (va > UNDEF) & (vb > UNDEF)
    d = np.abs(va[ok] - vb[ok]).astype(np.float64)
    rel = d / np.maximum(np.abs(vb[ok]), 1e-6)
    print("%s vs %s: %d nodes both defined (%d / %d defined), |diff| median %.3e p99 %.3e max %.3e, "
          "max relative %.3e, mean %.5f / %.5f"
          % (a, b, ok.sum(), (va > UNDEF).sum(), (vb > UNDEF).sum(), np.median(d), np.percentile(d, 99),
             d.max(), rel.max(), va[ok].mean(), vb[ok].mean()))


def series(gdir, wdir, hours, out):
    res = {}
    for var in ("hs", "t02"):
        sq = None
        n = None
        mean_g, mean_w = [], []
        for h in hours:
            g = load(os.path.join(gdir, "%s_%s.f32" % (var, h)))
            w = load(os.path.join(wdir, "%s_%s.f32" % (var, h)))
            ok = (g > UNDEF) & (w > UNDEF)
            d = np.where(ok, (g - w).astype(np.float64), 0.0)
            if sq is None:
                sq = np.zeros(g.size)
                n = np.zeros(g.size)
            sq += d * d
            n += ok
            mean_g.append(float(g[ok].mean()))
            mean_w.append(float(w[ok].mean()))
        good = n > 0
        rms = np.sqrt(sq[good] / n[good])
        res[var] = {
            "nodes": int(good.sum()),
            "hours": len(hours),
            "rms_median": float(np.median(rms)),
            "rms_p99": float(np.percentile(rms, 99)),
            "rms_max": float(rms.max()),
            "mean_gpu_last": mean_g[-1],
            "mean_ww3_last": mean_w[-1],
        }
        print("%s: %d nodes, %d hours, RMS difference median %.4f p99 %.4f max %.4f; last hour mean %.4f (GPU) %.4f (WW3)"
              % (var, good.sum(), len(hours), res[var]["rms_median"], res[var]["rms_p99"], res[var]["rms_max"],
                 mean_g[-1], mean_w[-1]))
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--pair":
        pair(a[1], a[2])
    else:
        out = None
        if "--out" in a:
            i = a.index("--out")
            out = a[i + 1]
            a = a[:i] + a[i + 2:]
        series(a[0], a[1], a[2:], out)
