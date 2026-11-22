"""Cut a lon/lat box out of an HWW3 grid for a nested test domain.

usage: python subdomain.py <grid dir> <out dir> lon0 lon1 lat0 lat1 [NK NTH]

Keeps the triangles whose three nodes are in the box and writes, in <out dir>,
the raw arrays of the sub-mesh (lon, lat, depth, level, tri), a ww3_grid.nml
(with NK frequencies and NTH directions if given) so that explicit_cost.py runs
on it, and subdomain.json: node counts per nside, open-boundary nodes (on the
sub-mesh boundary but interior in the full mesh) and coastal boundary nodes.
"""
import json, os, sys, re
import numpy as np

gdir, out, lon0, lon1, lat0, lat1 = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:7])
nk = int(sys.argv[7]) if len(sys.argv) > 7 else None
nth = int(sys.argv[8]) if len(sys.argv) > 8 else None
raw = gdir + "/raw/"
lon = np.fromfile(raw + "lon.f64"); lat = np.fromfile(raw + "lat.f64")
tri = np.fromfile(raw + "tri.u32", np.uint32).reshape(-1, 3).astype(np.int64)
depth = np.fromfile(raw + "depth.f64"); level = np.fromfile(raw + "level.u8", np.uint8)
inn = (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
tin = inn[tri].all(1)
used = np.unique(tri[tin])
remap = -np.ones(len(lon), np.int64); remap[used] = np.arange(len(used))
st = remap[tri[tin]]
os.makedirs(out + "/raw", exist_ok=True)
for name, a in (("lon.f64", lon[used]), ("lat.f64", lat[used]), ("depth.f64", depth[used]),
                ("level.u8", level[used]), ("tri.u32", st.astype(np.uint32))):
    a.tofile(out + "/raw/" + name)
# boundary of the full mesh and of the sub-mesh
def bnodes(t, n):
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), 1)
    ue, c = np.unique(e, axis=0, return_counts=True)
    b = np.zeros(n, bool); b[ue[c == 1].ravel()] = True
    return b
bfull = bnodes(tri, len(lon))[used]
bsub = bnodes(st, len(used))
openb = bsub & ~bfull
if os.path.exists(raw + "obc.u32"):  # open-boundary nodes of the full grid in the box
    obc = np.zeros(len(lon), bool)
    obc[np.fromfile(raw + "obc.u32", np.uint32)] = True
    openb |= bsub & obc[used]
np.flatnonzero(openb).astype(np.uint32).tofile(out + "/raw/obc.u32")
txt = open(gdir + "/ww3_grid.nml").read()
if nk:
    txt = re.sub(r"(SPECTRUM%NK\s*=\s*)\d+", r"\g<1>%d" % nk, txt)
    txt = re.sub(r"(SPECTRUM%NTH\s*=\s*)\d+", r"\g<1>%d" % nth, txt)
open(out + "/ww3_grid.nml", "w").write(txt)
res = {"grid": gdir, "box": [lon0, lon1, lat0, lat1], "nodes": int(len(used)), "triangles": int(len(st)),
       "nodes_per_nside": {int(2 ** int(l)): int((level[used] == l).sum()) for l in np.unique(level[used])},
       "open_boundary_nodes": int(openb.sum()), "coastal_boundary_nodes": int((bsub & bfull).sum()),
       "depth_median": float(np.median(depth[used])), "share_of_grid": float(len(used) / len(lon))}
json.dump(res, open(out + "/subdomain.json", "w"), indent=1)
print(json.dumps(res))
