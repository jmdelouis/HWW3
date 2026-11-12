"""Extract a GEBCO sub-grid as raw little-endian float32 + meta.json for hww3-grid.

usage: python prep_gebco.py GEBCO_2024.nc OUT_DIR lon_min lon_max lat_min lat_max [--margin 3] [--stride 1]

The output grid is row-major with longitude fastest, latitude increasing,
elevation in metres (positive up). Only numpy and netCDF4 are required.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("gebco")
    p.add_argument("out")
    p.add_argument("lon_min", type=float)
    p.add_argument("lon_max", type=float)
    p.add_argument("lat_min", type=float)
    p.add_argument("lat_max", type=float)
    p.add_argument("--margin", type=float, default=3.0)
    p.add_argument("--stride", type=int, default=1)
    a = p.parse_args()

    ds = Dataset(a.gebco)
    ds.set_auto_mask(False)
    lon_all = ds["lon"][:]
    lat_all = ds["lat"][:]
    m = a.margin
    i = np.nonzero((lon_all >= a.lon_min - m) & (lon_all <= a.lon_max + m))[0]
    j = np.nonzero((lat_all >= max(a.lat_min - m, -90)) & (lat_all <= min(a.lat_max + m, 90)))[0]
    s = a.stride
    i0, i1 = int(i[0]), int(i[-1]) + 1
    j0, j1 = int(j[0]), int(j[-1]) + 1
    lon = lon_all[i0:i1:s]
    lat = lat_all[j0:j1:s]
    var = ds["elevation"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # write band by band to limit memory
    with open(str(out / "elev.f32"), "wb") as f:
        band = 1024 * s
        for jb in range(j0, j1, band):
            z = var[jb:min(jb + band, j1):s, i0:i1:s].astype("<f4")
            f.write(np.ascontiguousarray(z).tobytes())
    meta = {
        "nlon": int(lon.size),
        "nlat": int(lat.size),
        "lon0": float(lon[0]),
        "lat0": float(lat[0]),
        "dlon": float(lon[1] - lon[0]),
        "dlat": float(lat[1] - lat[0]),
        "file": "elev.f32",
        "source": Path(a.gebco).name,
        "stride": s,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
