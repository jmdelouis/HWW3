"""Compare the hourly H_s of two hww3-prop runs node by node.

    python scripts/pre_hs_cmp.py RUN_A RUN_B [--every 24]

For each hour written by both runs (hs_YYYYMMDDHH.f64 if both have it,
otherwise .f32), over the nodes defined in both: mean H_s of A, median,
99th percentile and largest |B - A|, and the node of the largest; one line
every --every hours and a summary over all hours.
"""
import argparse
import glob
import os

import numpy as np


def load(d, tag):
    for ext, dt in (('f64', '<f8'), ('f32', '<f4')):
        p = os.path.join(d, 'hs_%s.%s' % (tag, ext))
        if os.path.exists(p):
            return ext, np.fromfile(p, dt).astype(np.float64)
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--every', type=int, default=24)
    a = ap.parse_args()
    tags = sorted(set(os.path.basename(p)[3:13] for p in glob.glob(os.path.join(a.a, 'hs_*.f*')))
                  & set(os.path.basename(p)[3:13] for p in glob.glob(os.path.join(a.b, 'hs_*.f*'))))
    if not tags:
        raise SystemExit('no common hour')
    allmed, allp99, allmax = [], [], []
    print('%-10s %4s %9s %10s %10s %10s %8s' % ('hour', 'ext', 'mean Hs', 'median', 'p99', 'max', 'node'))
    for i, t in enumerate(tags):
        ea, x = load(a.a, t)
        eb, y = load(a.b, t)
        if ea != eb:
            ea, x = 'f32', np.fromfile(os.path.join(a.a, 'hs_%s.f32' % t), '<f4').astype(np.float64)
            y = np.fromfile(os.path.join(a.b, 'hs_%s.f32' % t), '<f4').astype(np.float64)
        ok = (x > -999) & (y > -999)
        d = np.abs(y[ok] - x[ok])
        med, p99, mx = np.median(d), np.percentile(d, 99), d.max()
        allmed.append(med)
        allp99.append(p99)
        allmax.append(mx)
        if i % a.every == 0 or i == len(tags) - 1:
            print('%-10s %4s %9.4f %10.3e %10.3e %10.3e %8d' % (
                t, ea, x[ok].mean(), med, p99, mx, np.flatnonzero(ok)[d.argmax()]))
    print('%d common hours: median |dHs| up to %.3e m, p99 up to %.3e m, largest %.3e m' % (
        len(tags), max(allmed), max(allp99), max(allmax)))


if __name__ == '__main__':
    main()
