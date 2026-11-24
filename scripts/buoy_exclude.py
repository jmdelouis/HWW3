"""Candidates for the list of buoys the mesh cannot represent.

    python scripts/buoy_exclude.py RUN [--scratch S] [--ratio 0.4]

For every buoy of the calibration's training and validation windows, the
median over its hours (both windows) of observed H_s / model H_s of RUN
(the start run), through the depth-aware operator (the nearest node for the
buoys it leaves out), with the metadata of its file (platform name, source,
category), its position and the descriptors of scripts/coast_diag.py
(distance of its node to the coast, land fraction of the cell, depths).
Buoys below --ratio are printed as candidates, the others with a ratio
above 1 / ratio for reference. The list itself (configs/buoy_exclude.csv)
is fixed by hand from these candidates: a buoy is left out only if its
site cannot be represented by the mesh (harbour, barrier, inner Wadden,
archipelago), not because the model is wrong there.
"""
import argparse
import glob
import json
import os
import sys

import netCDF4
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cal_obs  # noqa: E402


def meta(bdir, name):
    fs = sorted(glob.glob(os.path.join(bdir, name + "_*.nc")))
    if not fs:
        return {}
    with netCDF4.Dataset(fs[0]) as d:
        return {a: str(d.getncattr(a)) for a in ("platform_name", "source", "source_platform_category_code", "institution",
                                                  "site_code", "wmo_platform_code") if a in d.ncattrs()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--scratch", default="/scratch/jmdeloui/HWW3")
    ap.add_argument("--ratio", type=float, default=0.4)
    a = ap.parse_args()
    C, X = os.path.join(a.scratch, "cal"), os.path.join(a.scratch, "coast")
    bdir = os.path.join(a.scratch, "obs", "insitu_MO", "202601")
    sub = json.load(open(os.path.join(X, "buoy_subgrid_all.json")))
    obs, desc = {}, {}
    for w in ("train", "valid"):
        rows = {r["name"]: r for r in json.load(open(os.path.join(X, "coast_%s.json" % w)))["rows"]}
        od = np.load(os.path.join(C, "op_%s_depth.npz" % w))
        on = np.load(os.path.join(C, "op_%s.npz" % w))
        _, bhd, _ = cal_obs.apply(od, a.run)
        _, bhn, _ = cal_obs.apply(on, a.run)
        names = [str(n) for n in od["buoys"]]
        assert names == [str(n) for n in on["buoys"]]
        for ib, n in enumerate(names):
            o, bh = (od, bhd) if od["bw_kept"][ib] else (on, bhn)
            s = o["bh_b"] == ib
            if not s.any():
                continue
            d = obs.setdefault(n, {"obs": [], "mod": []})
            d["obs"] += o["bh_obs"][s].tolist()
            d["mod"] += bh[s].tolist()
            desc[n] = rows.get(n, {})
    out = []
    for n, d in obs.items():
        ob, mo = np.array(d["obs"]), np.array(d["mod"])
        ok = mo > 0.01
        r = float(np.median(ob[ok] / mo[ok])) if ok.any() else float("nan")
        c = desc.get(n, {})
        out.append((r, n, len(ob), float(np.median(ob)), float(np.median(mo)), c))
    out.sort()
    print("%-44s %6s %5s %6s %6s %7s %5s %6s %6s  %s" % ("buoy", "ratio", "hours", "obs", "model", "coast", "land", "node d", "buoy d", "metadata"))
    for r, n, h, mo_, mm, c in out:
        if r < a.ratio or r > 1 / a.ratio:
            m = meta(bdir, n)
            print("%-44s %6.2f %5d %6.2f %6.2f %6.1fkm %5.2f %6.1f %6s  %s %s/%s (%.3f, %.3f)" % (
                n, r, h, mo_, mm, c.get("coast_km", float("nan")), sub.get(str(c.get("node")), {}).get("land", float("nan")),
                c.get("node_depth", float("nan")), c.get("bottom_depth"), m.get("platform_name", ""), m.get("source", ""),
                m.get("source_platform_category_code", ""), c.get("lon", float("nan")), c.get("lat", float("nan"))))
    print("%d buoys; %d below %.2f, %d above %.2f" % (len(out), sum(1 for x in out if x[0] < a.ratio), a.ratio,
                                                    sum(1 for x in out if x[0] > 1 / a.ratio), 1 / a.ratio))


if __name__ == "__main__":
    main()
