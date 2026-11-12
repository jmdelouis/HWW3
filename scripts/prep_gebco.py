"""Extract a GEBCO sub-grid as raw little-endian float32 + meta.json for hww3-grid.

usage: python prep_gebco.py GEBCO_2024.nc OUT_DIR lon_min lon_max lat_min lat_max [--margin 3] [--stride 1]

The output grid is row-major with longitude fastest, latitude increasing,
elevation in metres (positive up).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


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

    ds = xr.open_dataset(a.gebco)
    m = a.margin
    sub = ds["elevation"].sel(
        lon=slice(a.lon_min - m, a.lon_max + m), lat=slice(max(a.lat_min - m, -90), min(a.lat_max + m, 90))
    )
    if a.stride > 1:
        sub = sub.isel(lon=slice(None, None, a.stride), lat=slice(None, None, a.stride))
    lon = sub["lon"].values
    lat = sub["lat"].values
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # write band by band to limit memory
    with open(out / "elev.f32", "wb") as f:
        band = 1024
        for j0 in range(0, lat.size, band):
            z = sub.isel(lat=slice(j0, j0 + band)).values.astype("<f4")
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
        "stride": a.stride,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
