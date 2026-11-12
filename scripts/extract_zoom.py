"""Extract small plotting subsets of an hww3-grid mesh (numpy only, runs on HPC).

usage:
  python extract_zoom.py <grid_dir> <out.npz> lon0 lon1 lat0 lat1   # zoom with cells + triangles
  python extract_zoom.py <grid_dir> <out.npz> --nodes               # all nodes (float32) for level maps

Zoom subsets need raw/cell_lon.f64 and raw/cell_lat.f64 (`hww3-grid cells <grid_dir>`).
"""

import json
import sys
from pathlib import Path

import numpy as np


def main():
    d, out = Path(sys.argv[1]), sys.argv[2]
    raw = d / "raw"
    lon = np.fromfile(raw / "lon.f64", dtype="<f8")
    lat = np.fromfile(raw / "lat.f64", dtype="<f8")
    level = np.fromfile(raw / "level.u8", dtype="u1")
    obc = np.fromfile(raw / "obc.u32", dtype="<u4")
    summary = json.loads((d / "meta.json").read_text())["summary"]
    if sys.argv[3] == "--nodes":
        np.savez_compressed(out, lon=lon.astype("f4"), lat=lat.astype("f4"), level=level,
                            obc=obc, summary=json.dumps(summary))
        return
    x0, x1, y0, y1 = (float(v) for v in sys.argv[3:7])
    pad = 0.5
    tri = np.fromfile(raw / "tri.u32", dtype="<u4").reshape(-1, 3)
    clon = np.fromfile(raw / "cell_lon.f64", dtype="<f8").reshape(-1, 4)
    clat = np.fromfile(raw / "cell_lat.f64", dtype="<f8").reshape(-1, 4)
    sel = (lon > x0 - pad) & (lon < x1 + pad) & (lat > y0 - pad) & (lat < y1 + pad)
    tri = tri[sel[tri].any(axis=1)]
    used = np.unique(np.concatenate([np.nonzero(sel)[0], tri.ravel()]))
    remap = np.full(lon.size, -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    np.savez_compressed(
        out, lon=lon[used], lat=lat[used], level=level[used], tri=remap[tri],
        cell_lon=clon[used], cell_lat=clat[used], bbox=np.array([x0, x1, y0, y1]),
        summary=json.dumps(summary),
    )


if __name__ == "__main__":
    main()
