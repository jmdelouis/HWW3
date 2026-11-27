"""H_s of hww3-pr3 (or hww3-prop) on a structured grid against a ww3_ounf file.

    python scripts/pr3_compare.py RUN_DIR FIELD.nc [--json OUT.json]

RUN_DIR holds hs_YYYYMMDDHH.f32 on the whole NX x NY grid (iy NX + ix,
-999.9 off the sea points), as hww3-pr3 writes them; FIELD.nc is the
ww3_ounf output of the same grid (variable hs on latitude x longitude).
For every hour present in both: the number of sea points, the mean H_s of
both, the largest and the RMS difference; then, over the hours after the
first, the RMS difference of each point in time, with its median, 99th
percentile and largest value over the points, and the same relative to
the point's mean H_s (points with a mean H_s above 1 cm).
"""
import argparse
import glob
import json
import os

import netCDF4
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("nc")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    ds = netCDF4.Dataset(a.nc)
    times = netCDF4.num2date(ds["time"][:], ds["time"].units, only_use_cftime_datetimes=False)
    index = {f"{t.year:04d}{t.month:02d}{t.day:02d}{t.hour:02d}": i for i, t in enumerate(times)}
    ny, nx = ds["hs"].shape[1:]
    rows, diffs, means = [], [], []
    for f in sorted(glob.glob(os.path.join(a.run, "hs_*.f32"))):
        tag = os.path.basename(f)[3:13]
        if tag not in index:
            continue
        g = np.fromfile(f, dtype="<f4").astype(np.float64)
        w = np.ma.filled(ds["hs"][index[tag]].astype(np.float64), np.nan).ravel()
        assert g.size == nx * ny, (g.size, nx, ny)
        ok = (g > -999.0) & np.isfinite(w)
        d = g[ok] - w[ok]
        rows.append({"time": tag, "n": int(ok.sum()), "mean_gpu": float(g[ok].mean()), "mean_ww3": float(w[ok].mean()),
                     "max_abs": float(np.abs(d).max()), "rms": float(np.sqrt((d * d).mean()))})
        r = rows[-1]
        print(f"{tag}: {r['n']} points, mean H_s ours {r['mean_gpu']:.5f} m, WW3 {r['mean_ww3']:.5f} m, max |diff| {r['max_abs']:.3e} m, RMS {r['rms']:.3e} m")
        full = np.full(nx * ny, np.nan)
        full[ok] = g[ok] - w[ok]
        diffs.append(full)
        mw = np.full(nx * ny, np.nan)
        mw[ok] = w[ok]
        means.append(mw)
    if len(diffs) > 1:
        dd = np.array(diffs[1:])
        rms_t = np.sqrt(np.nanmean(dd * dd, axis=0))
        mw = np.nanmean(np.array(means[1:]), axis=0)
        ok = np.isfinite(rms_t)
        rel = rms_t[ok & (mw > 0.01)] / mw[ok & (mw > 0.01)]
        s = {"points": int(ok.sum()), "rms_median": float(np.median(rms_t[ok])), "rms_p99": float(np.percentile(rms_t[ok], 99)),
             "rms_max": float(rms_t[ok].max()), "rel_median": float(np.median(rel)), "rel_p99": float(np.percentile(rel, 99)),
             "rel_max": float(rel.max())}
        print(f"RMS in time per point over {s['points']} points: median {s['rms_median']:.3e} m, p99 {s['rms_p99']:.3e} m, max {s['rms_max']:.3e} m; "
              f"relative to the mean H_s: median {s['rel_median']:.3e}, p99 {s['rel_p99']:.3e}, max {s['rel_max']:.3e}")
        rows.append({"summary": s})
    if a.json:
        with open(a.json, "w") as fo:
            json.dump(rows, fo, indent=1)


if __name__ == "__main__":
    main()
