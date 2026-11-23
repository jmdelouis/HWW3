"""Water depth at the buoys from the fine bathymetry.

    python scripts/buoy_depth.py GEBCO_DIR BUOYS_JSON > DEPTHS.json

GEBCO_DIR is a cut of scripts/prep_gebco.py (elev.f32 and meta.json,
elevation positive up, longitude fastest, latitude increasing), read by
memory map; BUOYS_JSON (a file, or the JSON text itself) maps buoy names
to their lon and lat, as scripts/buoy_op.py positions writes them. For
each buoy: the depth of the GEBCO cell that holds it (m, positive down,
null if that cell is land), and over the 3 x 3 and 7 x 7 cells around it
(about 1.4 and 3.2 km at 15 arc seconds) the median depth of the sea cells
and the fraction of land. Only numpy is needed, so that it runs where the
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
    buoys = json.load(open(arg)) if os.path.exists(arg) else json.loads(arg)
    out = {}
    for name, b in buoys.items():
        i = int(round((b["lon"] - meta["lon0"]) / meta["dlon"]))
        j = int(round((b["lat"] - meta["lat0"]) / meta["dlat"]))
        if not (0 <= i < nlon and 0 <= j < nlat):
            out[name] = None
            continue
        e = float(z[j, i])
        row = {"cell": -e if e < 0 else None}
        for h in (1, 3):
            blk = np.asarray(z[max(j - h, 0):j + h + 1, max(i - h, 0):i + h + 1], dtype=float).ravel()
            sea = -blk[blk < 0]
            row["med%d" % (2 * h + 1)] = float(np.median(sea)) if sea.size else None
            row["land%d" % (2 * h + 1)] = float((blk >= 0).mean())
        out[name] = row
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
