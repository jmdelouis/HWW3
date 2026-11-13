"""Write WW3 fields on the HEALPix grid as a Zarr V3 multi-resolution pyramid.

Usage:
    python scripts/ww3_pyramid.py RUN_NC NODES_NPZ OUT [--vars all|hs,t02,...]
        [--levels 7-11] [--tstride 1] [--ntime N]

RUN_NC     ww3_ounf output (fields on the nodes, ``time`` in days).
NODES_NPZ  ``grids/<name>/plot/nodes.npz`` (lon, lat, level of every node).
OUT        local directory (uploaded afterwards with ``scripts/upload_s3.py``).

Every node of the grid is the centre of a NESTED HEALPix pixel of depth
``level`` (nside = 2**level). The pyramid holds one group per depth, from
the coarsest to the finest requested:

* at a depth finer than a node's, the node value is copied to all its
  descendants (each pixel is cut 2x2 per level);
* at a depth coarser than a node's, a pixel is the mean of its 4 children
  that hold data (equal areas, so this is the area mean). Directions are
  averaged as unit vectors.

So the finest level is the model field exactly and each coarser level is
its exact area average.

Fields (``--vars all``): every scalar of the ww3_ounf file (``hs``, ``t02``,
``fp``, ``dir``, ``wlv``, ``ice`` and the components ``uwnd``, ``vwnd``,
``ucur``, ``vcur``), plus the speeds ``wspd`` and ``cspd`` computed from the
components. Fields absent from the file are skipped.

Layout: the zarr DGGS convention v1 (as written by ``xdggs``) and the zarr
multiscales convention v0.1::

    OUT/                     root: multiscales, Conventions, provenance
      11/  hs, t02, ... (time, cells), cell_ids (cells), time   dggs attribute
      10/  ...
      7/   ...

There is no CF ``crs`` grid-mapping variable: readers that find one (easygems
``healpix_nside``/``healpix_order``) take a sphere and ignore the ``dggs``
ellipsoid, which moves every cell south by up to 0.13 degree.

``cells`` holds every cell of the depth-``SHARD_LEVEL`` pixels the domain
touches, in ascending nested order (contiguous identifiers, so a chunk is a
compact area); cells without data hold NaN and chunks with only NaN are not
written. Arrays are sharded: a shard is ``TSHARD`` time steps of one
depth-``SHARD_LEVEL`` pixel, a chunk one time step of at most
``CHUNK_CELLS`` cells, compressed with zstd.

The generator places HEALPix on the WGS84 ellipsoid (geodetic latitudes are
converted to authalic latitudes, as ``healpix-geo`` does), so the cell
identifiers are computed from authalic latitudes and the WGS84 ellipsoid is
declared in ``dggs``.
"""
import argparse
import datetime
import json
import time as _time

import healpy as hp
import numpy as np
import zarr

SHARD_LEVEL = 3
CHUNK_CELLS = 16384
TSHARD = 24

DGGS_CONVENTION = {
    "uuid": "7b255807-140c-42ca-97f6-7a1cfecdbc38",
    "name": "dggs",
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/dggs/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/dggs/blob/v1/README.md",
    "description": "Discrete Global Grid Systems convention for zarr",
}
MULTISCALES_CONVENTION = {
    "uuid": "d35379db-88df-4056-af3a-620245f8e347",
    "name": "multiscales",
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v0.1/README.md",
    "description": "Multiscale layout of zarr datasets",
}

WGS84 = {"name": "WGS84", "semi_major_axis": 6378137.0, "inverse_flattening": 298.257223563}

ALL_VARS = ["hs", "t02", "fp", "dir", "wspd", "uwnd", "vwnd", "cspd", "ucur", "vcur", "wlv", "ice"]
DERIVED = {
    "wspd": ("uwnd", "vwnd", {"long_name": "wind speed at 10 m", "standard_name": "wind_speed", "units": "m s-1"}),
    "cspd": ("ucur", "vcur", {"long_name": "surface current speed",
                              "standard_name": "sea_water_speed", "units": "m s-1"}),
}
ANGLES = {"dir"}


def authalic_deg(lat_deg):
    """Geodetic latitude -> authalic latitude on WGS84 (degrees), exact."""
    f = 1.0 / WGS84["inverse_flattening"]
    e2 = f * (2 - f)
    e = np.sqrt(e2)
    s = np.sin(np.radians(lat_deg))

    def q(sn):
        return (1 - e2) * (sn / (1 - e2 * sn * sn) - np.log((1 - e * sn) / (1 + e * sn)) / (2 * e))

    return np.degrees(np.arcsin(np.clip(q(s) / q(1.0), -1, 1)))


def node_pixels(lon, lat, level):
    """Nested pixel index of every node at its own depth (on the authalic
    sphere); checks that the node is the pixel centre."""
    lat = authalic_deg(lat)
    ipix = np.empty(lon.size, np.int64)
    for lv in np.unique(level):
        s = level == lv
        ns = 2 ** int(lv)
        ipix[s] = hp.ang2pix(ns, lon[s], lat[s], nest=True, lonlat=True)
        clon, clat = hp.pix2ang(ns, ipix[s], nest=True, lonlat=True)
        d = hp.rotator.angdist([lon[s], lat[s]], [clon, clat], lonlat=True)
        res = hp.nside2resol(ns)
        if d.max() > 1e-2 * res:
            raise ValueError(f"depth {lv}: node off pixel centre by {d.max() / res:.3g} pixel")
    return ipix


def build_index(ipix, level, fine):
    """Cells of the finest depth covered by the nodes, sorted, and the node
    that holds each of them."""
    reps = 4 ** (fine - level.astype(np.int64))
    first = ipix * reps
    node = np.repeat(np.arange(ipix.size), reps)
    cells = np.repeat(first, reps) + (np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps))
    order = np.argsort(cells, kind="stable")
    return cells[order], node[order]


def coarsen_index(cells):
    """Parents of sorted cells, and the start of each parent's run."""
    parents = cells >> 2
    starts = np.flatnonzero(np.r_[True, parents[1:] != parents[:-1]])
    return parents[starts], starts


def dense_layout(cells, level):
    """Every cell of the depth-SHARD_LEVEL pixels touched, and the position of
    each data cell in it."""
    per_shard = 4 ** (level - SHARD_LEVEL)
    shards = np.unique(cells // per_shard)
    all_cells = (shards[:, None] * per_shard + np.arange(per_shard)[None, :]).ravel()
    pos = np.searchsorted(all_cells, cells)
    return all_cells, pos, per_shard


def nanmean_reduce(cur, starts):
    """Mean over runs starting at ``starts`` (last axis), ignoring NaN."""
    ok = np.isfinite(cur)
    s = np.add.reduceat(np.where(ok, cur, 0.0), starts, axis=-1)
    c = np.add.reduceat(ok.astype(np.float32), starts, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s / c).astype(np.float32)


def nanmax_reduce(cur, starts):
    """Max over runs starting at ``starts`` (last axis), ignoring NaN."""
    return np.fmax.reduceat(cur, starts, axis=-1).astype(np.float32)


def open_store(path):
    return zarr.storage.LocalStore(path)


def dggs_group(root, lv, all_cells, per_shard):
    """Level group with the DGGS attribute and its ``cell_ids`` coordinate."""
    g = root.create_group(str(lv))
    g.attrs["zarr_conventions"] = [DGGS_CONVENTION]
    g.attrs["dggs"] = {"name": "healpix", "refinement_level": lv, "indexing_scheme": "nested",
                       "ellipsoid": dict(WGS84), "spatial_dimension": "cells",
                       "coordinate": "cell_ids", "compression": "none"}
    ids = g.create_array("cell_ids", shape=(all_cells.size,), dtype="uint64", chunks=(per_shard,),
                         compressors=[zarr.codecs.ZstdCodec(level=3)], fill_value=2 ** 64 - 1,
                         dimension_names=["cells"],
                         attributes={"standard_name": "healpix_index", "units": "1"})
    ids[:] = all_cells.astype(np.uint64)
    return g


def multiscales(names, lmin, lmax, resampling):
    """Root ``multiscales`` attribute: finest level first."""
    return {"layout": [
        {"asset": f"{lv}/{n}"} if lv == lmax else
        {"asset": f"{lv}/{n}", "derived_from": f"{lmax}/{n}",
         "transform": {"scale": [float(2 ** (lmax - lv))]}, "resampling_method": resampling.get(n, "mean")}
        for lv in range(lmax, lmin - 1, -1) for n in names]}


def level_indices(fine_cells, lmin, lmax):
    """Per depth: sorted data cells and the run starts from the finer depth."""
    per_level = {lmax: {"cells": fine_cells}}
    cells = fine_cells
    for lv in range(lmax - 1, lmin - 1, -1):
        parents, starts = coarsen_index(cells)
        per_level[lv] = {"cells": parents, "starts": starts}
        cells = parents
    return per_level


def main():
    import netCDF4

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_nc")
    ap.add_argument("nodes_npz")
    ap.add_argument("out")
    ap.add_argument("--vars", default="all")
    ap.add_argument("--levels", default="7-11")
    ap.add_argument("--tstride", type=int, default=1)
    ap.add_argument("--ntime", type=int, default=0, help="first N steps only (test)")
    a = ap.parse_args()
    t0 = _time.time()
    lmin, lmax = (int(x) for x in a.levels.split("-"))

    d = netCDF4.Dataset(a.run_nc)
    have = set(d.variables)
    want = ALL_VARS if a.vars == "all" else a.vars.split(",")
    names = [n for n in want if n in have or (n in DERIVED and set(DERIVED[n][:2]) <= have)]
    print("fields:", ",".join(names), flush=True)
    lon = np.asarray(d["longitude"][:], float)
    lat = np.asarray(d["latitude"][:], float)
    nodes = np.load(a.nodes_npz)
    assert np.allclose(nodes["lon"], lon, atol=1e-4) and np.allclose(nodes["lat"], lat, atol=1e-4)
    level = nodes["level"].astype(np.int64)
    tsel = np.arange(0, d["time"].size, a.tstride)
    if a.ntime:
        tsel = tsel[: a.ntime]
    times = netCDF4.num2date(d["time"][tsel], d["time"].units)
    hours = np.array([(datetime.datetime(*t.timetuple()[:6]) - datetime.datetime(2000, 1, 1)).total_seconds()
                      / 3600 for t in times])

    ipix = node_pixels(lon, lat, level)
    fine_cells, fine_node = build_index(ipix, level, lmax)
    per_level = level_indices(fine_cells, lmin, lmax)
    print(f"{fine_cells.size} cells at depth {lmax} from {ipix.size} nodes ({_time.time() - t0:.0f} s)", flush=True)

    store = open_store(a.out)
    root = zarr.open_group(store, mode="w", zarr_format=3)
    root.attrs.update({
        "Conventions": "CF-1.13",
        "title": "WAVEWATCH III on a multi-resolution HEALPix grid, North Atlantic, January 2026",
        "source": f"WAVEWATCH III, {a.run_nc}",
        "history": f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%MZ} scripts/ww3_pyramid.py (github.com/jmdelouis/HWW3)",
        "zarr_conventions": [MULTISCALES_CONVENTION, DGGS_CONVENTION],
        "multiscales": multiscales(names, lmin, lmax, {}),
    })

    zstd = zarr.codecs.ZstdCodec(level=5)
    arrays, dense = {}, {}
    for lv in range(lmax, lmin - 1, -1):
        all_cells, pos, per_shard = dense_layout(per_level[lv]["cells"], lv)
        dense[lv] = (all_cells.size, pos)
        g = dggs_group(root, lv, all_cells, per_shard)
        tv = g.create_array("time", shape=(tsel.size,), dtype="float64", chunks=(tsel.size,),
                            dimension_names=["time"], fill_value=np.nan,
                            attributes={"standard_name": "time", "units": "hours since 2000-01-01 00:00:00",
                                        "calendar": "standard", "axis": "T"})
        tv[:] = hours
        chunk = min(per_shard, CHUNK_CELLS)
        for n in names:
            if n in DERIVED:
                attrs = dict(DERIVED[n][2])
            else:
                src = d[n]
                attrs = {k: src.getncattr(k) for k in ("long_name", "standard_name", "units")
                         if k in src.ncattrs()}
            attrs["coordinates"] = "cell_ids"
            if n in ANGLES:
                attrs["cell_methods"] = "area: mean (unit vectors)"
            arrays[(lv, n)] = g.create_array(
                n, shape=(tsel.size, all_cells.size), dtype="float32",
                chunks=(1, chunk), shards=(min(TSHARD, tsel.size), per_shard),
                compressors=[zstd], fill_value=np.nan, dimension_names=["time", "cells"],
                attributes=attrs)
        print(f"depth {lv}: {per_level[lv]['cells'].size} data cells, {all_cells.size} stored", flush=True)

    def read(n, bt):
        if n in DERIVED:
            u, v = (np.ma.filled(d[c][bt].astype(np.float32), np.nan) for c in DERIVED[n][:2])
            return np.hypot(u, v)
        return np.ma.filled(d[n][bt].astype(np.float32), np.nan)

    # Time blocks of one shard: build all depths in memory, write once.
    for b0 in range(0, tsel.size, TSHARD):
        bt = tsel[b0:b0 + TSHARD]
        for n in names:
            v = read(n, bt)[:, fine_node]  # (nt, fine cells)
            if n in ANGLES:
                r = np.radians(v)
                comps = [np.cos(r), np.sin(r)]
            for lv in range(lmax, lmin - 1, -1):
                if n in ANGLES:
                    if lv < lmax:
                        comps = [nanmean_reduce(c, per_level[lv]["starts"]) for c in comps]
                    cur = (np.degrees(np.arctan2(comps[1], comps[0])) % 360).astype(np.float32)
                else:
                    if lv < lmax:
                        v = nanmean_reduce(v, per_level[lv]["starts"])
                    cur = v
                size, pos = dense[lv]
                out = np.full((bt.size, size), np.nan, np.float32)
                out[:, pos] = cur
                arrays[(lv, n)][b0:b0 + bt.size, :] = out
        print(f"steps {b0}-{b0 + bt.size - 1} written ({_time.time() - t0:.0f} s)", flush=True)

    zarr.consolidate_metadata(store)
    print(json.dumps({lv: int(per_level[lv]["cells"].size) for lv in per_level}))
    print(f"done in {_time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
