"""Split the growth of a tangent by part of the step at probed nodes.

    python scripts/tl_probe.py DIR [--factor 3] [--nth 36]

DIR is the output of `hww3-prop run --tangent NAME --probe N1,N2,..`
(probe.csv and probe_spec.f64: after each part of every step, the
spectrum and its tangent at the probed nodes). For every node: the
L2 norm of the tangent spectrum after each part, its growth factor over
each part (prepare: forcing, ice and level updates; ktp3a, geo, ktp3b:
the propagation; srce: W3SRCE), the steps where one part multiplies it by
more than FACTOR, with the node scalars (depth, wind, u*, mean dynamic
step, ice), and at the worst of these steps the frequency-direction bins
where the tangent grew most in that part, with the spectrum there.

With probe_check.csv (--probe-check H): at every step, the tangent of the
source step alone for the incoming tangent spectrum against central
differences of the value kernel with relative steps H and H / 10, per node:
the growth factor of the tangent over the step (amp_tl) and of the
differences (amp_fd), the largest difference between the tangent and the
differences of step H / 10 relative to their largest value (err), and
between the two differences (rough: a discrete choice of the model inside
the steps). The steps where amp_tl exceeds FACTOR are listed.
"""
import argparse
import csv
import collections
import os

import numpy as np

PHASES = ["prepare", "ktp3a", "geo", "ktp3b", "srce"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--factor", type=float, default=3.0)
    ap.add_argument("--nth", type=int, default=36)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(a.dir, "probe.csv"))))
    spec = np.fromfile(os.path.join(a.dir, "probe_spec.f64"), "<f8")
    nspec = len(spec) // (2 * len(rows))
    spec = spec.reshape(len(rows), 2, nspec)
    nk = nspec // a.nth
    by_node = collections.defaultdict(list)
    for r in rows:
        by_node[int(r["node"])].append(r)
    for node, rr in by_node.items():
        print("node %d: %d probes, %d steps" % (node, len(rr), len(rr) // len(PHASES)))
        prev = None
        growth = {p: [] for p in PHASES}
        events = []
        for r in rr:
            l2 = float(r["l2_vad"])
            if prev is not None and prev > 0:
                f = l2 / prev
                growth[r["phase"]].append(f)
                if f > a.factor and l2 > 1e-12:
                    events.append((f, r, prevrow))
            prev, prevrow = l2, r
        for p in PHASES:
            g = np.array(growth[p]) if growth[p] else np.array([1.0])
            print("   %-8s growth factor: median %.3f, max %.3g, steps above %g: %d" % (p, np.median(g), g.max(), a.factor, int((g > a.factor).sum())))
        print("   the parts that multiply the tangent by more than %g:" % a.factor)
        for f, r, pr in events[:25]:
            print("     %s %-8s x%-9.3g |dA| %.3e -> %.3e  sum A %.3e  depth %s m, U10 %s m/s, u* %s, dt %s s, ice %s, mapsta %s, max at (ik %s, ith %s)"
                  % (r["time"], r["phase"], f, float(pr["l2_vad"]), float(r["l2_vad"]), float(r["sum_va"]), r["dw"], r["u10"], r["ust"],
                     r["dtdyn"], r["ice"], r["mapsta"], r["ik_max"], r["ith_max"]))
        if events:
            f, r, pr = max(events, key=lambda e: e[0])
            i1, i0 = int(r["row"]), int(pr["row"])
            va0, d0 = spec[i0, 0], spec[i0, 1]
            va1, d1 = spec[i1, 0], spec[i1, 1]
            ratio = np.abs(d1) / np.maximum(np.abs(d0), 1e-300)
            top = np.argsort(-np.abs(d1))[:12]
            print("   worst: %s %s x%.3g; bins of the largest tangent after it (ik, ith: A before, A after, dA before, dA after):" % (r["time"], r["phase"], f))
            for q in top:
                print("     (%2d, %2d)  %.3e  %.3e  %+.3e  %+.3e" % (q // a.nth, q % a.nth, va0[q], va1[q], d0[q], d1[q]))
            ek = (np.abs(d1).reshape(nk, a.nth).sum(1))
            print("   |dA| by frequency after it:", " ".join("%.2g" % x for x in ek))
            print("   A by frequency after it:   ", " ".join("%.2g" % x for x in va1.reshape(nk, a.nth).sum(1)))
    checks(a)


def checks(a):
    f = os.path.join(a.dir, "probe_check.csv")
    if not os.path.exists(f):
        return
    rows = list(csv.DictReader(open(f)))
    by = collections.defaultdict(list)
    for r in rows:
        by[int(r["node"])].append(r)
    print("one-step checks of the source tangent (%d)" % len(rows))
    for node, rr in by.items():
        amp = np.array([float(r["amp_tl"]) for r in rr])
        err = np.array([float(r["err_tl_fd2"]) for r in rr])
        rough = np.array([float(r["rough_fd1_fd2"]) for r in rr])
        smooth = rough < 1e-3
        print("  node %d: amp_tl median %.3f max %.3g; err on the %d steps with rough < 1e-3: median %.2e max %.2e; %d steps rough"
              % (node, np.median(amp), amp.max(), int(smooth.sum()), np.median(err[smooth]) if smooth.any() else np.nan,
                 err[smooth].max() if smooth.any() else np.nan, int((~smooth).sum())))
        for r in rr:
            if float(r["amp_tl"]) > a.factor:
                print("    %s amp_tl %.3g amp_fd %.3g err %.2e rough %.2e |dA| %.3e -> %.3e, worst bin (ik %s, ith %s)"
                      % (r["time"], float(r["amp_tl"]), float(r["amp_fd"]), float(r["err_tl_fd2"]), float(r["rough_fd1_fd2"]),
                         float(r["l2_in"]), float(r["l2_tl"]), r["ik_err"], r["ith_err"]))


if __name__ == "__main__":
    main()
