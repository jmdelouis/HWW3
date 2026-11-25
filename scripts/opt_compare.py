"""Two runs of hww3-prop compared hour by hour (optimisation checks).

    python scripts/opt_compare.py RUN_A RUN_B [--prefix hs,t02] [--hours N]

For each prefix (hs, t02, dhs, dt02), the double-precision hourly files
(PREFIX_YYYYMMDDHH.f64) present in both runs, up to N hours: the number of
hours identical to the bit, the largest difference, and the median and
99th percentile over the hours of the RMS difference over the defined nodes
(not -999.9 in either run), with the 99th percentile over the nodes of the
difference at the last hour, and the seconds per model day of both runs
(summary.json). With --rel (for runs that differ beyond round-off, such as
a single-precision tangent against the double-precision one), also the
hourly RMS difference relative to the RMS of RUN_A (median and largest
over the hours) and the relative difference of the mean over the nodes
(the tangent of the mean H_s for dhs; median and largest).
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--prefix", default="hs,t02")
    ap.add_argument("--hours", type=int, default=0)
    ap.add_argument("--rel", action="store_true")
    a = ap.parse_args()
    for run in (a.a, a.b):
        s = os.path.join(run, "summary.json")
        if os.path.exists(s):
            d = json.load(open(s))
            print("%s: %s steps, %.1f s per model day, setup %.1f s" % (run, d.get("steps"), d.get("seconds_per_model_day", float("nan")),
                                                                     d.get("setup_seconds", float("nan"))))
    for p in a.prefix.split(","):
        ta = {os.path.basename(f)[len(p) + 1:len(p) + 11] for f in glob.glob(os.path.join(a.a, p + "_*.f64"))}
        tb = {os.path.basename(f)[len(p) + 1:len(p) + 11] for f in glob.glob(os.path.join(a.b, p + "_*.f64"))}
        tags = sorted(ta & tb)
        if a.hours:
            tags = tags[: a.hours]
        if not tags:
            print("%-5s no common hour" % p)
            continue
        same, rms, dmax, last, rrel, mrel = 0, [], 0.0, None, [], []
        for t in tags:
            x = np.fromfile(os.path.join(a.a, "%s_%s.f64" % (p, t)), "<f8")
            y = np.fromfile(os.path.join(a.b, "%s_%s.f64" % (p, t)), "<f8")
            ok = (np.abs(np.abs(x) - 999.9) > 1e-3) & (np.abs(np.abs(y) - 999.9) > 1e-3)
            d = np.abs(x - y)[ok]
            same += int(np.array_equal(x, y))
            rms.append(float(np.sqrt(np.mean(d ** 2))) if d.size else 0.0)
            dmax = max(dmax, float(d.max()) if d.size else 0.0)
            last = d
            if a.rel and d.size:
                xs = x[ok]
                n = float(np.sqrt(np.mean(xs ** 2)))
                rrel.append(rms[-1] / n if n > 0 else 0.0)
                m = float(np.mean(xs))
                mrel.append(abs(m - float(np.mean(y[ok]))) / abs(m) if m != 0 else 0.0)
        print("%-5s %d hours (%s to %s): %d identical to the bit, largest difference %.3e, hourly RMS median %.3e p99 %.3e, last hour p99 over the nodes %.3e"
              % (p, len(tags), tags[0], tags[-1], same, dmax, np.median(rms), np.quantile(rms, 0.99), np.quantile(last, 0.99) if last.size else 0.0))
        if rrel:
            print("      relative RMS difference median %.3e largest %.3e; relative difference of the mean over the nodes median %.3e largest %.3e"
                  % (np.median(rrel), max(rrel), np.median(mrel), max(mrel)))


if __name__ == "__main__":
    main()
