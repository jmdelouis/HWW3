"""Write an hww3-grid grid (level, depth, bathymetry slope) as a Zarr V3
HEALPix pyramid for gridlook.

Usage:
    python scripts/grid_pyramid.py GRID_DIR OUT [--levels 7-11]

GRID_DIR   output directory of ``hww3-grid`` (reads ``raw/*``).
OUT        local directory (uploaded afterwards with ``scripts/upload_s3.py``).

Fields, one value per HEALPix cell, same layout and conventions as
``scripts/ww3_pyramid.py`` (zarr DGGS convention v1, WGS84, multiscales):

* ``level``  HEALPix depth of the grid node holding the cell (nside = 2**level);
  coarser pyramid levels keep the finest (max) grid level they contain;
* ``nside``  2**level, same resampling;
* ``cell_size``  square root of the cell area of the grid node (km);
* ``depth``  water depth of the node (m, positive down), area mean;
* ``slope``  bathymetry slope |grad h| (m/m): ``raw/slope.f64`` written by the
  generator when present, otherwise the P1 gradient of ``depth`` on the
  triangles averaged on the nodes (area weights), area mean;
* ``target_size``  cell size required by the depth and slope criteria (km),
  from ``raw/target_km.f64`` when the generator wrote it.

Fields are ``(time, cells)`` with a single time step (the build date of the
grid): gridlook only displays HEALPix variables with a leading dimension.
"""
import argparse
import datetime
import json
import os
import time as _time

import numpy as np
import zarr

from ww3_pyramid import (DGGS_CONVENTION, MULTISCALES_CONVENTION, build_index, dense_layout, dggs_group,
                         level_indices, multiscales, nanmax_reduce, nanmean_reduce, open_store)

R_AUTHALIC_KM = 6371.0072


def node_slope(lon, lat, depth, tri):
    """|grad depth| at the nodes: P1 gradient per triangle on the sphere,
    averaged on the nodes with area weights."""
    lo, la = np.radians(lon), np.radians(lat)
    p = R_AUTHALIC_KM * 1e3 * np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    a, b, c = (p[tri[:, k]] for k in range(3))
    n = np.cross(b - a, c - a)
    area2 = np.linalg.norm(n, axis=1)
    nh = n / area2[:, None]
    f = depth[tri]
    g = (f[:, 0, None] * np.cross(nh, c - b) + f[:, 1, None] * np.cross(nh, a - c)
         + f[:, 2, None] * np.cross(nh, b - a)) / area2[:, None]
    gm = np.linalg.norm(g, axis=1)
    w = 0.5 * area2
    num = np.zeros(lon.size)
    den = np.zeros(lon.size)
    for k in range(3):
        np.add.at(num, tri[:, k], gm * w)
        np.add.at(den, tri[:, k], w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("grid_dir")
    ap.add_argument("out")
    ap.add_argument("--levels", default="7-11")
    a = ap.parse_args()
    t0 = _time.time()
    lmin, lmax = (int(x) for x in a.levels.split("-"))
    raw = os.path.join(a.grid_dir, "raw")

    def rd(name, dt):
        return np.fromfile(os.path.join(raw, name), dt)

    uniq = rd("uniq.u64", "<u8")
    level = rd("level.u8", "u1").astype(np.int64)
    ipix = (uniq - (4 * 4 ** level).astype(np.uint64)).astype(np.int64)
    depth = rd("depth.f64", "<f8")
    lon, lat = rd("lon.f64", "<f8"), rd("lat.f64", "<f8")
    if os.path.exists(os.path.join(raw, "slope.f64")):
        slope, slope_src = rd("slope.f64", "<f8"), "hww3-grid (smoothed GEBCO)"
    else:
        tri = rd("tri.u32", "<u4").reshape(-1, 3).astype(np.int64)
        slope, slope_src = node_slope(lon, lat, depth, tri), "P1 gradient of the node depths"
    area = 4 * np.pi * R_AUTHALIC_KM ** 2 / (12.0 * 4.0 ** level)
    fields = {
        "level": (level.astype(np.float32), "max",
                  {"long_name": "HEALPix depth of the grid node (nside = 2**level)", "units": "1"}),
        "nside": ((2.0 ** level).astype(np.float32), "max", {"long_name": "HEALPix nside of the grid node",
                                                             "units": "1"}),
        "cell_size": (np.sqrt(area).astype(np.float32), "mean",
                      {"long_name": "square root of the grid cell area", "units": "km"}),
        "depth": (depth.astype(np.float32), "mean",
                  {"long_name": "water depth (mean of the wet GEBCO samples of the cell)",
                   "standard_name": "sea_floor_depth_below_geoid", "units": "m", "positive": "down"}),
        "slope": (slope.astype(np.float32), "mean",
                  {"long_name": f"bathymetry slope |grad h| ({slope_src})", "units": "1"}),
    }
    if os.path.exists(os.path.join(raw, "target_km.f64")):
        fields["target_size"] = (rd("target_km.f64", "<f8").astype(np.float32), "mean",
                                 {"long_name": "cell size required by the depth and slope criteria",
                                  "units": "km"})
    names = list(fields)

    fine_cells, fine_node = build_index(ipix, level, lmax)
    per_level = level_indices(fine_cells, lmin, lmax)
    store = open_store(a.out)
    root = zarr.open_group(store, mode="w", zarr_format=3)
    meta = json.load(open(os.path.join(a.grid_dir, "meta.json")))
    root.attrs.update({
        "Conventions": "CF-1.13",
        "title": f"hww3-grid multi-resolution HEALPix grid '{meta['config']['name']}' ({a.grid_dir})",
        "source": "hww3-grid, GEBCO 2024",
        "history": f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%MZ} scripts/grid_pyramid.py (github.com/jmdelouis/HWW3)",
        "hww3_grid_config": json.dumps(meta["config"]),
        "zarr_conventions": [MULTISCALES_CONVENTION, DGGS_CONVENTION],
        "multiscales": multiscales(names, lmin, lmax, {n: f[1] for n, f in fields.items()}),
    })
    built = os.path.getmtime(os.path.join(raw, "uniq.u64"))
    hours = (datetime.datetime.fromtimestamp(built, datetime.timezone.utc)
             - datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)).total_seconds() / 3600
    zstd = zarr.codecs.ZstdCodec(level=5)
    cur = {n: f[0][fine_node] for n, f in fields.items()}
    for lv in range(lmax, lmin - 1, -1):
        if lv < lmax:
            st = per_level[lv]["starts"]
            cur = {n: (nanmax_reduce if fields[n][1] == "max" else nanmean_reduce)(v, st) for n, v in cur.items()}
        all_cells, pos, per_shard = dense_layout(per_level[lv]["cells"], lv)
        g = dggs_group(root, lv, all_cells, per_shard)
        tv = g.create_array("time", shape=(1,), dtype="float64", chunks=(1,), dimension_names=["time"],
                            fill_value=np.nan, attributes={"standard_name": "time", "calendar": "standard",
                                                           "units": "hours since 2000-01-01 00:00:00", "axis": "T"})
        tv[:] = [hours]
        for n in names:
            out = np.full((1, all_cells.size), np.nan, np.float32)
            out[0, pos] = cur[n]
            arr = g.create_array(n, shape=(1, all_cells.size), dtype="float32", chunks=(1, min(per_shard, 16384)),
                                 shards=(1, per_shard), compressors=[zstd], fill_value=np.nan,
                                 dimension_names=["time", "cells"],
                                 attributes={**fields[n][2], "coordinates": "cell_ids"})
            arr[:] = out
        print(f"depth {lv}: {per_level[lv]['cells'].size} data cells ({_time.time() - t0:.0f} s)", flush=True)
    zarr.consolidate_metadata(store)
    print(f"done in {_time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
