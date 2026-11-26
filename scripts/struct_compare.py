"""H_s of hww3-prop on a meshed structured grid against WW3's gridded output.

    python scripts/struct_compare.py RUN_OUT GRID_DIR FIELD.nc [--every 6] [--json OUT.json]

RUN_OUT holds the hourly files hs_YYYYMMDDHH.f32 of hww3-prop on the mesh
GRID_DIR (hww3-pre struct); FIELD.nc is the ww3_ounf file of the same
structured grid (variable hs). Every EVERY hours present in both, over the
nodes where both are defined (H_s > 0 in WW3, not -999.9 here): the mean
H_s of both, the bias, the RMS difference, the scatter index (RMS of the
difference over the mean of WW3) and the correlation, overall and by
latitude band of the geographic coordinates (raw/lat_geo.f64).
"""
import argparse
import glob
import json
import os

import netCDF4
import numpy as np


def stats(a, b):
    d = a - b
    return {
        "n": int(a.size),
        "mean_gpu": float(a.mean()),
        "mean_ww3": float(b.mean()),
        "bias": float(d.mean()),
        "rmse": float(np.sqrt((d * d).mean())),
        "si": float(np.sqrt((d * d).mean()) / max(b.mean(), 1e-9)),
        "r": float(np.corrcoef(a, b)[0, 1]) if a.size > 2 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("grid")
    ap.add_argument("nc")
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    raw = os.path.join(a.grid, "raw")
    ij = np.fromfile(os.path.join(raw, "ij.u32"), dtype="<u4").astype(np.int64)
    lat = np.fromfile(os.path.join(raw, "lat_geo.f64"), dtype="<f8")
    ds = netCDF4.Dataset(a.nc)
    times = netCDF4.num2date(ds["time"][:], ds["time"].units, only_use_cftime_datetimes=False)
    index = {f"{t.year:04d}{t.month:02d}{t.day:02d}{t.hour:02d}": i for i, t in enumerate(times)}
    files = sorted(glob.glob(os.path.join(a.run, "hs_*.f32")))
    out = []
    bands = [(-90, -30), (-30, 0), (0, 30), (30, 90)]
    for k, f in enumerate(files):
        tag = os.path.basename(f)[3:13]
        if tag not in index or k % a.every:
            continue
        g = np.fromfile(f, dtype="<f4").astype(np.float64)
        w = np.ma.filled(ds["hs"][index[tag]].astype(np.float64), np.nan).ravel()[ij]
        ok = (g > -999.0) & np.isfinite(w) & (w > 0)
        row = {"time": tag, "all": stats(g[ok], w[ok])}
        for lo, hi in bands:
            m = ok & (lat >= lo) & (lat < hi)
            if m.sum() > 10:
                row[f"{lo}..{hi}"] = stats(g[m], w[m])
        out.append(row)
        s = row["all"]
        print(f"{tag}: {s['n']} nodes, mean Hs gpu {s['mean_gpu']:.3f} m ww3 {s['mean_ww3']:.3f} m, bias {s['bias']:+.3f} m, RMSE {s['rmse']:.3f} m, SI {s['si']:.3f}, r {s['r']:.3f}")
    if a.json:
        with open(a.json, "w") as fo:
            json.dump(out, fo, indent=1)


if __name__ == "__main__":
    main()
