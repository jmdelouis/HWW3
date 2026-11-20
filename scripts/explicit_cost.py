"""Cost of the explicit WW3 N scheme on an HWW3 grid, per node and per
spectral component, and the nodes that set it.

WW3 (w3profsmd_pdlib.F90, PDLIB_W3XYPFSN2) sub-cycles every spectral component
c = (f, theta) ITER(c) = ceil(DT / min_i dt_i(c)) times, with

    dt_i(c) = SI_i / sum_T max(0, K_iT),  K_iT = lambda_T . n_iT,
    lambda_T = (c_1 + c_2 + c_3) / 6,

n_iT the inward normal of the edge opposite i (length of that edge), SI_i a
third of the area of the triangles around i, the minimum taken over the
interior nodes of all MPI ranks (MPI_ALLREDUCE). A single node with a short
dt_i therefore sets the number of sweeps of the whole domain for that
component. The propagation work per global step is S = sum_c ITER(c) sweeps
(S = NK * NTH if every component needs one sweep).

The script computes, without currents (and optionally with an upper bound
that adds the largest current speed of each node along every direction):
  - ITER(f, theta) and S,
  - per node, the local number of sub-steps req_i = max_c DT / dt_i(c)
    (the WW3 output field CFX),
  - the nodes that set ITER (binding nodes), their nside, depth, CFL length
    over cell size, level jump and angles,
  - S if the N worst nodes were relieved, if the worst nodes were moved one
    HEALPix level coarser (CFL length doubled), or if every node had the
    median CFL length / cell size of its level (geometry fixed).

usage: python explicit_cost.py <grid dir> <out prefix> [label] [current.nc]
"""

import json
import re
import sys

import numpy as np

G = 9.806
RADIUS = 4.0e7 / (2 * np.pi)  # WW3 constants.F90
DT = 600.0


def group_speed(f, h):
    w = 2 * np.pi * f
    k = np.maximum(w * w / G, w / np.sqrt(G * h))
    for _ in range(60):
        t = np.tanh(k * h)
        fk = G * k * t - w * w
        dk = G * t + G * k * h * (1 - t * t)
        k = k - fk / dk
    kh = k * h
    n = 0.5 * (1 + np.where(kh < 20, 2 * kh / np.sinh(np.minimum(2 * kh, 700)), 0.0))
    return n * w / k


def nml(path, key, default):
    try:
        m = re.search(key + r"\s*=\s*([-0-9.eE+]+)", open(path).read())
        return float(m.group(1)) if m else default
    except OSError:
        return default


def current_max(path, lon, lat):
    """Largest hourly surface current speed at each node (nearest cell)."""
    from netCDF4 import Dataset

    d = Dataset(path)
    names = list(d.variables)
    un = next(n for n in names if n.lower() in ("ucur", "uo", "u", "uuss", "utotal"))
    vn = next(n for n in names if n.lower() in ("vcur", "vo", "v", "vuss", "vtotal"))
    glon = d["longitude"][:].astype(float)
    glat = d["latitude"][:].astype(float)
    nt = d.dimensions["time"].size
    m = np.zeros((len(glat), len(glon)), np.float32)
    for t0 in range(0, nt, 24):
        u = np.ma.filled(d[un][t0 : t0 + 24], 0).astype(np.float32)
        v = np.ma.filled(d[vn][t0 : t0 + 24], 0).astype(np.float32)
        m = np.maximum(m, np.sqrt(u * u + v * v).max(0))
    ix = np.clip(np.rint((lon - glon[0]) / (glon[1] - glon[0])).astype(int), 0, len(glon) - 1)
    iy = np.clip(np.rint((lat - glat[0]) / (glat[1] - glat[0])).astype(int), 0, len(glat) - 1)
    return m[iy, ix].astype(float), nt


def main():
    gdir, out = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else gdir
    curfile = sys.argv[4] if len(sys.argv) > 4 else None
    raw = gdir + "/raw/"
    lon = np.fromfile(raw + "lon.f64")
    lat = np.fromfile(raw + "lat.f64")
    tri = np.fromfile(raw + "tri.u32", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    depth = np.fromfile(raw + "depth.f64")
    level = np.fromfile(raw + "level.u8", dtype=np.uint8).astype(int)
    np_, ne = len(lon), len(tri)
    gnml = gdir + "/ww3_grid.nml"
    f1 = nml(gnml, r"SPECTRUM%FREQ1", 0.0373)
    xfr = nml(gnml, r"SPECTRUM%XFR", 1.1)
    nk = int(nml(gnml, r"SPECTRUM%NK", 32))
    nth = int(nml(gnml, r"SPECTRUM%NTH", 36))
    thoff = nml(gnml, r"SPECTRUM%THOFF", 0.0)
    dmin = nml(gnml, r"GRID%DMIN", 1.0)
    freqs = f1 * xfr ** np.arange(nk)
    ths = (np.arange(nth) + thoff) * 2 * np.pi / nth
    h = np.maximum(depth, dmin)
    print(f"{label}: {np_} nodes, {ne} triangles, depth {depth.min():.1f}..{depth.max():.1f} m, "
          f"NK {nk} from {f1} Hz x {xfr}, NTH {nth}, DMIN {dmin}", flush=True)

    # local tangent-plane coordinates per triangle (the N scheme is affine
    # invariant, so this matches WW3's degree coordinates to first order)
    lo, la = np.radians(lon[tri]), np.radians(lat[tri])
    c0 = np.cos(la.mean(1))[:, None]
    x = RADIUS * c0 * (lo - lo[:, :1])
    y = RADIUS * (la - la.mean(1)[:, None])
    area2 = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    neg = area2 < 0
    if neg.any():
        tri[neg] = tri[neg][:, [0, 2, 1]]
        x[neg] = x[neg][:, [0, 2, 1]]
        y[neg] = y[neg][:, [0, 2, 1]]
        area2 = np.abs(area2)
    area = 0.5 * area2
    # inward normals (WW3 IEN): n1 = (y2 - y3, x3 - x2), cyclic
    nx = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1)
    ny = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1)
    flat = tri.ravel()
    si = np.bincount(flat, weights=np.repeat(area / 3, 3), minlength=np_)

    # interior nodes: every edge around them shared by two triangles
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    boundary = np.zeros(np_, bool)
    boundary[ue[cnt == 1].ravel()] = True
    interior = ~boundary
    # node descriptors
    jump = np.zeros(np_, int)
    dl = np.abs(level[ue[:, 0]] - level[ue[:, 1]])
    np.maximum.at(jump, ue[:, 0], dl)
    np.maximum.at(jump, ue[:, 1], dl)
    near_bnd = np.zeros(np_, bool)
    bb = boundary[ue[:, 0]] | boundary[ue[:, 1]]
    near_bnd[ue[bb].ravel()] = True
    near_bnd &= interior
    ln = [np.hypot(x[:, (i + 2) % 3] - x[:, (i + 1) % 3], y[:, (i + 2) % 3] - y[:, (i + 1) % 3]) for i in range(3)]
    ang = []
    for i in range(3):
        a, b, c = ln[i], ln[(i + 1) % 3], ln[(i + 2) % 3]
        ang.append(np.degrees(np.arccos(np.clip((b * b + c * c - a * a) / (2 * b * c), -1, 1))))
    ang = np.stack(ang, 1)
    amin = np.full(np_, 180.0)
    amax = np.zeros(np_)
    np.minimum.at(amin, flat, np.repeat(ang.min(1), 3))
    np.maximum.at(amax, flat, np.repeat(ang.max(1), 3))
    cell = RADIUS * np.sqrt(np.pi / 3) / (2.0 ** level)

    # geometric CFL length per node and direction: L_i(theta) = SI / sum max(0, n.e)/2
    q = [np.maximum(0.0, np.cos(t) * nx + np.sin(t) * ny).astype(np.float32) for t in ths]
    lgeo = np.full(np_, np.inf)
    for qt in q:
        kk = np.bincount(flat, weights=0.5 * qt.ravel(), minlength=np_)
        lgeo = np.minimum(lgeo, np.where(kk > 0, si / np.maximum(kk, 1e-30), np.inf))
    eff = lgeo / cell

    cur, ncur = (None, 0)
    if curfile:
        try:
            cur, ncur = current_max(curfile, lon, lat)
            print(f"currents: {ncur} hourly fields, node max {cur.max():.2f} m/s, p99 {np.quantile(cur, 0.99):.2f}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print("currents skipped:", ex)
            cur = None

    def sweep(add=None):
        """ITER per component, per-node max requirement, binding counts."""
        it = np.zeros((nk, nth), int)
        arg = np.zeros((nk, nth), int)
        reqmax = np.zeros(np_)
        reqs = []
        for ik, f in enumerate(freqs):
            s = group_speed(f, h)
            if add is not None:
                s = s + add
            st = (s[tri].sum(1) / 6.0).astype(np.float32)[:, None]
            row = []
            for j, qt in enumerate(q):
                kk = np.bincount(flat, weights=(st * qt).ravel(), minlength=np_)
                r = DT * kk / si
                r[~interior] = 0.0
                i = int(np.argmax(r))
                it[ik, j] = max(1, int(np.ceil(r[i] - 1e-9)))
                arg[ik, j] = i
                reqmax = np.maximum(reqmax, r)
                row.append(r.astype(np.float32))
            reqs.append(row)
        return it, arg, reqmax, reqs

    it, arg, reqmax, reqs = sweep()
    S = int(it.sum())
    print(f"S = {S} sweeps per step (minimum {nk * nth}); ITER at f1: {it[0].min()}..{it[0].max()}", flush=True)

    # binding nodes: alone they would force the ITER of a component with ITER > 1
    bind = np.zeros(np_, int)
    for ik in range(nk):
        for j in range(nth):
            if it[ik, j] > 1:
                bind += reqs[ik][j] > it[ik, j] - 1
    nbind = int((bind > 0).sum())

    # S after relieving the N worst nodes (ranked by reqmax)
    order = np.argsort(-np.where(interior, reqmax, -1))
    Ns = [n for n in (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)
          if n < interior.sum()]
    Srem = np.zeros(len(Ns), int)
    for ik in range(nk):
        for j in range(nth):
            r = reqs[ik][j][order]
            suf = np.maximum.accumulate(r[::-1])[::-1]
            Srem += np.maximum(1, np.ceil(suf[Ns] - 1e-9)).astype(int)
    # S if nodes whose local requirement exceeds K were one HEALPix level coarser
    Ks = [2, 3, 4, 5, 6, 7, 8, 9]
    Scoarse, ncoarse = [], []
    for K in Ks:
        fac = np.where(reqmax > K, 0.5, 1.0)
        Scoarse.append(int(sum(max(1, int(np.ceil((reqs[ik][j] * fac).max() - 1e-9))) for ik in range(nk) for j in range(nth))))
        ncoarse.append(int((reqmax > K).sum()))
    # S if every node had at least the median efficiency of its level
    med = {d: float(np.median(eff[(level == d) & interior & np.isfinite(eff)])) for d in np.unique(level[interior])}
    effmed = np.array([med.get(d, np.nan) for d in level])
    gfac = np.where(eff < effmed, eff / effmed, 1.0)
    Sgeom = int(sum(max(1, int(np.ceil((reqs[ik][j] * gfac).max() - 1e-9))) for ik in range(nk) for j in range(nth)))

    res = {
        "label": label, "nodes": np_, "triangles": ne, "interior": int(interior.sum()),
        "nk": nk, "nth": nth, "freqs": freqs.tolist(), "dt": DT,
        "S": S, "S_min": nk * nth,
        "iter_max_by_freq": it.max(1).tolist(), "iter_mean_by_freq": it.mean(1).tolist(),
        "n_binding_nodes": nbind,
        "removal": {"N": Ns, "S": Srem.tolist()},
        "coarsen": {"K": Ks, "S": Scoarse, "nodes": ncoarse},
        "S_geometry_fixed": Sgeom,
        "eff_median_by_nside": {int(2 ** d): v for d, v in med.items()},
    }
    # nodes that set ITER at f1 and the worst ones overall
    top = order[:40]
    res["top_nodes"] = [
        {"lon": round(float(lon[i]), 4), "lat": round(float(lat[i]), 4), "nside": int(2 ** level[i]),
         "depth": round(float(depth[i]), 1), "req": round(float(reqmax[i]), 2), "bind": int(bind[i]),
         "cell_km": round(float(cell[i] / 1e3), 2), "L_km": round(float(lgeo[i] / 1e3), 2),
         "eff": round(float(eff[i]), 3), "eff_med": round(float(effmed[i]), 3), "jump": int(jump[i]),
         "near_boundary": bool(near_bnd[i]), "amin": round(float(amin[i]), 1), "amax": round(float(amax[i]), 1)}
        for i in top]
    # causes among the binding nodes
    b = bind > 0
    res["binding_causes"] = {
        "low_eff (<0.7 x median of level)": int((b & (eff < 0.7 * effmed)).sum()),
        "level_jump>=1": int((b & (jump >= 1)).sum()),
        "near_boundary": int((b & near_bnd).sum()),
        "depth>200m": int((b & (depth > 200)).sum()),
        "depth<50m": int((b & (depth < 50)).sum()),
        "by_nside": {int(2 ** d): int((b & (level == d)).sum()) for d in np.unique(level[b])} if b.any() else {},
    }

    if cur is not None:
        itc, _, reqc, _ = sweep(add=cur)
        res["with_currents_upper_bound"] = {"S": int(itc.sum()), "iter_max_by_freq": itc.max(1).tolist(),
                                            "hourly_fields": ncur}
        print(f"with currents (upper bound): S = {int(itc.sum())}", flush=True)

    np.savez_compressed(out + ".npz", reqmax=reqmax.astype(np.float32), bind=bind.astype(np.int32),
                        eff=eff.astype(np.float32), lgeo=lgeo.astype(np.float32), interior=interior,
                        it=it, arg=arg)
    json.dump(res, open(out + ".json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("S", "S_min", "n_binding_nodes", "removal", "coarsen", "S_geometry_fixed", "binding_causes")}, indent=1))
    print("ITER max by freq:", res["iter_max_by_freq"])
    for t in res["top_nodes"][:15]:
        print(t)


if __name__ == "__main__":
    main()
