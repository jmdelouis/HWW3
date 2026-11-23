"""Spread of the fine bathymetry inside the cells of some grid nodes.

    python scripts/coast_subgrid.py GEBCO_DIR NODES_JSON > SUB.json

GEBCO_DIR is a cut of scripts/prep_gebco.py (elev.f32 and meta.json,
elevation positive up, longitude fastest, latitude increasing), read by
memory map; NODES_JSON (a file, or the JSON text itself) maps node ids to
their lon, lat and cell radius in km, as scripts/coast_diag.py
--emit-nodes writes them. For each node, over the GEBCO points within the
radius: the number of points, the fraction of land (elevation above 0),
and over the sea points the mean, standard deviation and range of the
depth (m, positive down). Only numpy is needed, so that it runs where the
bathymetry is, and it writes nothing but its output.
"""
import json
import os
import sys

import numpy as np


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    g = sys.argv[1]
    meta = json.load(open(os.path.join(g, "meta.json")))
    nlon, nlat = meta["nlon"], meta["nlat"]
    z = np.memmap(os.path.join(g, meta["file"]), dtype="<f4", mode="r", shape=(nlat, nlon))
    arg = sys.argv[2]
    nodes = json.load(open(arg)) if os.path.exists(arg) else json.loads(arg)
    out = {}
    for k, v in nodes.items():
        lon, lat, r = v["lon"], v["lat"], max(v["radius_km"], 0.5)
        dlat = r / 111.2
        dlon = dlat / max(np.cos(np.radians(lat)), 1e-3)
        i0 = int(np.floor((lon - dlon - meta["lon0"]) / meta["dlon"]))
        i1 = int(np.ceil((lon + dlon - meta["lon0"]) / meta["dlon"])) + 1
        j0 = int(np.floor((lat - dlat - meta["lat0"]) / meta["dlat"]))
        j1 = int(np.ceil((lat + dlat - meta["lat0"]) / meta["dlat"])) + 1
        i0, j0 = max(i0, 0), max(j0, 0)
        i1, j1 = min(i1, nlon), min(j1, nlat)
        blk = np.asarray(z[j0:j1, i0:i1], dtype=float)
        glon = meta["lon0"] + meta["dlon"] * np.arange(i0, i1)
        glat = meta["lat0"] + meta["dlat"] * np.arange(j0, j1)
        x = (glon[None, :] - lon) * 111.2 * np.cos(np.radians(lat))
        y = (glat[:, None] - lat) * 111.2
        inside = x ** 2 + y ** 2 <= r ** 2
        e = blk[inside]
        sea = -e[e < 0]
        out[k] = {"n": int(e.size), "land": float((e >= 0).mean()) if e.size else float("nan"),
                  "mean": float(sea.mean()) if sea.size else float("nan"),
                  "std": float(sea.std()) if sea.size else float("nan"),
                  "range": float(sea.max() - sea.min()) if sea.size else float("nan")}
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
