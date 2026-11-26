"""Forcing files of hww3-prop from a WW3 gridded output on a structured grid.

    python scripts/struct_fieldnc.py FIELD.nc GRID_DIR OUT_DIR [--start YYYYMMDDHH] [--stop YYYYMMDDHH]

FIELD.nc is a ww3_ounf file of the structured grid that GRID_DIR meshes
(hww3-pre struct), with the wind (uwnd, vwnd) and currents (ucur, vcur)
as WW3 used them on that grid. Each time between START and STOP is written
on the nodes of the mesh (raw/ij.u32: index iy NX + ix of each node) to
OUT_DIR/wind.ww3 and OUT_DIR/current.ww3, in the format of ww3_prnc (big-
endian Fortran records: the header 'WAVEWATCH III' + field + NX NY GTYPE,
then per time the date and one record per component), with NX the number
of nodes, NY 1 and GTYPE 3 (unstructured), as hww3-prop reads them. Missing
values (land, ice-free fill) become 0. On a mesh in rotated coordinates
(raw/angle.f64 not zero) the vectors are turned into the rotated frame.
"""
import argparse
import datetime as dt
import os

import netCDF4
import numpy as np


def record(f, body):
    n = np.array([len(body)], dtype=">i4").tobytes()
    f.write(n + body + n)


def writer(path, fid, n):
    f = open(path, "wb")
    head = b"WAVEWATCH III" + fid.encode() + np.array([n, 1, 3, 0, 0, 0], dtype=">i4").tobytes()
    record(f, head)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("nc")
    ap.add_argument("grid")
    ap.add_argument("out")
    ap.add_argument("--start", default="")
    ap.add_argument("--stop", default="")
    a = ap.parse_args()
    raw = os.path.join(a.grid, "raw")
    ij = np.fromfile(os.path.join(raw, "ij.u32"), dtype="<u4").astype(np.int64)
    ang = np.fromfile(os.path.join(raw, "angle.f64"), dtype="<f8")
    rotated = bool(np.any(ang != 0.0))
    ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    n = ij.size
    ds = netCDF4.Dataset(a.nc)
    t = netCDF4.num2date(ds["time"][:], ds["time"].units, only_use_cftime_datetimes=False)
    key = lambda s: dt.datetime.strptime(s, "%Y%m%d%H") if s else None
    t0, t1 = key(a.start), key(a.stop)
    os.makedirs(a.out, exist_ok=True)
    files = {"WND": ("uwnd", "vwnd", "wind.ww3"), "CUR": ("ucur", "vcur", "current.ww3")}
    outs = {k: writer(os.path.join(a.out, v[2]), k, n) for k, v in files.items() if v[0] in ds.variables}
    count = 0
    for it, tt in enumerate(t):
        tt = dt.datetime(tt.year, tt.month, tt.day, tt.hour, tt.minute, tt.second)
        if (t0 and tt < t0) or (t1 and tt > t1):
            continue
        stamp = np.array([tt.year * 10000 + tt.month * 100 + tt.day, tt.hour * 10000 + tt.minute * 100 + tt.second], dtype=">i4").tobytes()
        for k, f in outs.items():
            u = np.ma.filled(ds[files[k][0]][it].astype(np.float64), 0.0).ravel()[ij]
            v = np.ma.filled(ds[files[k][1]][it].astype(np.float64), 0.0).ravel()[ij]
            if rotated:
                u, v = u * ca - v * sa, u * sa + v * ca
            record(f, stamp)
            record(f, u.astype(">f4").tobytes())
            record(f, v.astype(">f4").tobytes())
        count += 1
    for f in outs.values():
        f.close()
    print(f"{count} times of {', '.join(files[k][2] for k in outs)} on {n} nodes ({'rotated' if rotated else 'geographic'}) in {a.out}")


if __name__ == "__main__":
    main()
