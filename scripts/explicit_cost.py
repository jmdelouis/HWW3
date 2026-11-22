"""Cost of the explicit WW3 N scheme on an HWW3 grid, per node and per
spectral component, and the nodes that set it.

WW3 (w3profsmd_pdlib.F90, PDLIB_W3XYPFSN2) sub-cycles every spectral component
c = (f, theta) ITER(c) = ceil(DT / min_i dt_i(c)) times, with

    dt_i(c) = SI_i / sum_T max(0, K_iT),  K_iT = lambda_T . n_iT,
    lambda_T = (c_1 + c_2 + c_3) / 6,

n_iT the inward normal of the edge opposite i (length of that edge), SI_i a
third of the area of the triangles around i, the minimum taken over the nodes
of all MPI ranks (MPI_ALLREDUCE) with IOBP_i = 1 .OR. FSBCCFL. FSBCCFL is
UGBCCFL of the UNST namelist of ww3_grid, true by default and in the HWW3 runs
(ww3_grid.out: UGBCCFL = T), so every wet node (IOBDP = 1, depth >= DMIN +
1e-5), coastal (IOBP = 0) and open-boundary (IOBP = 2) nodes included, counts.
A single node with a short dt_i sets the number of sweeps of the whole domain
for that component, and a coastal node in one or two triangles has a dt_i 2-3
times shorter than an interior node of the same size. The propagation work
per global step is S = sum_c ITER(c) sweeps (S = NK * NTH if every component
needs one sweep); the GPU model hww3-wave sweeps all directions of a
frequency together, sum_f max_theta ITER(f, theta) frequency sweeps per step.

The script computes, without currents (and optionally with an upper bound
that adds the largest current speed of each node along every direction):
  - ITER(f, theta), S and the frequency sweeps, counting every wet node as
    WW3 does (default) or the interior nodes only (--interior-only,
    FSBCCFL = F, the count used to design grid v5), and S for every
    combination of node classes (interior, coast, open boundary);
  - per node, the local number of sub-steps req_i = max_c DT / dt_i(c)
    (the WW3 output field CFX), its class and number of triangles, and the
    nodes of each class above K sub-steps (the cfl_substeps test of
    hww3-grid);
  - the nodes that set ITER (binding nodes), their class, nside, depth, CFL
    length over cell size, level jump and angles;
  - S if the N worst nodes were relieved, if the nodes above K sub-steps
    were one HEALPix level coarser (CFL length doubled) or as many levels as
    they need to reach K, if every node had the median CFL length / cell
    size of its level and class (geometry fixed), and if the coastal nodes
    in a single triangle were removed with their triangle (mesh recomputed).

usage: python explicit_cost.py <grid dir> <out prefix> [label] [current.nc] [--interior-only]
"""

import json
import re
import sys

import numpy as np

G = 9.806
RADIUS = 4.0e7 / (2 * np.pi)  # WW3 constants.F90
DT = 600.0
CLASSES = ("coast", "interior", "open")  # index = WW3 IOBP


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


def mesh(lon, lat, tri, np_, ths, obc):
    """Triangle geometry on local tangent planes (the N scheme is affine
    invariant, so this matches WW3's degree coordinates to first order),
    PDLIB_SI, the positive parts of e_theta . n_iT / 2 per direction, the
    WW3 IOBP of every node and its number of triangles."""
    tri = tri.copy()
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
    ntri = np.bincount(flat, minlength=np_)
    # boundary nodes: on an edge of a single triangle (WW3 IOBP = 0, or 2 on
    # the open boundary)
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    boundary = np.zeros(np_, bool)
    boundary[ue[cnt == 1].ravel()] = True
    iobp = np.where(boundary, 0, 1)
    iobp[obc] = 2
    iobp[ntri == 0] = 0
    q = [np.maximum(0.0, np.cos(t) * nx + np.sin(t) * ny).astype(np.float32) for t in ths]
    return dict(tri=tri, x=x, y=y, flat=flat, si=si, ntri=ntri, iobp=iobp, ue=ue, q=q)


def main():
    args = sys.argv[1:]
    opts = [a for a in args if a.startswith("--")]
    pos = [a for a in args if not a.startswith("--")]
    if len(pos) < 2 or set(opts) - {"--interior-only"}:
        sys.exit(__doc__)
    interior_only = "--interior-only" in opts
    gdir, out = pos[0], pos[1]
    label = pos[2] if len(pos) > 2 else gdir
    curfile = pos[3] if len(pos) > 3 else None
    raw = gdir + "/raw/"
    lon = np.fromfile(raw + "lon.f64")
    lat = np.fromfile(raw + "lat.f64")
    tri0 = np.fromfile(raw + "tri.u32", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    depth = np.fromfile(raw + "depth.f64")
    level = np.fromfile(raw + "level.u8", dtype=np.uint8).astype(int)
    try:
        obc = np.fromfile(raw + "obc.u32", dtype=np.uint32).astype(np.int64)
    except OSError:
        obc = np.zeros(0, np.int64)
    np_, ne = len(lon), len(tri0)
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
    wet = depth >= dmin + 1e-5  # WW3 IOBDP; dry nodes have no velocity
    print(f"{label}: {np_} nodes, {ne} triangles, depth {depth.min():.1f}..{depth.max():.1f} m, "
          f"NK {nk} from {f1} Hz x {xfr}, NTH {nth}, DMIN {dmin}, "
          f"counting {'the interior nodes only (FSBCCFL = F)' if interior_only else 'every wet node (FSBCCFL = T)'}",
          flush=True)

    g = mesh(lon, lat, tri0, np_, ths, obc)
    tri, flat, si, ntri, iobp, ue = g["tri"], g["flat"], g["si"], g["ntri"], g["iobp"], g["ue"]
    x, y = g["x"], g["y"]
    interior = iobp == 1
    cls_mask = [wet & (iobp == c) for c in range(3)]
    counted = cls_mask[1] if interior_only else wet & (ntri > 0)
    print("nodes: " + ", ".join(f"{CLASSES[c]} {int(cls_mask[c].sum())}" for c in range(3))
          + f", dry {int((~wet).sum())}; coastal nodes in 1 / 2 / 3 triangles: "
          + " / ".join(str(int((cls_mask[0] & (ntri == n)).sum())) for n in (1, 2, 3)), flush=True)
    # node descriptors
    jump = np.zeros(np_, int)
    dl = np.abs(level[ue[:, 0]] - level[ue[:, 1]])
    np.maximum.at(jump, ue[:, 0], dl)
    np.maximum.at(jump, ue[:, 1], dl)
    near_bnd = np.zeros(np_, bool)
    bb = ~interior[ue[:, 0]] | ~interior[ue[:, 1]]
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
    lgeo = np.full(np_, np.inf)
    for qt in g["q"]:
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

    def sweep(gm, masks, add=None, keep=None):
        """Largest DT / dt_i per component over each node mask of `masks`,
        the node where it is reached, the largest requirement of every node
        over the components, and with `keep` (a node mask) the requirement of
        every component at the nodes of `keep` (float32, zero elsewhere)."""
        flat_, q_ = gm["flat"], gm["q"]
        inv = np.zeros(np_)
        np.divide(DT, gm["si"], out=inv, where=gm["si"] > 0)
        cmax = np.zeros((len(masks), nk, nth))
        carg = np.zeros((len(masks), nk, nth), int)
        reqmax = np.zeros(np_)
        reqs = []
        for ik, f in enumerate(freqs):
            s = group_speed(f, h)
            if add is not None:
                s = s + add
            s = s * wet
            st = (s[gm["tri"]].sum(1) / 6.0).astype(np.float32)[:, None]
            row = []
            for j, qt in enumerate(q_):
                kk = np.bincount(flat_, weights=(st * qt).ravel(), minlength=np_)
                r = kk * inv
                for m, mask in enumerate(masks):
                    rm = np.where(mask, r, 0.0)
                    i = int(np.argmax(rm))
                    cmax[m, ik, j] = rm[i]
                    carg[m, ik, j] = i
                reqmax = np.maximum(reqmax, r)
                if keep is not None:
                    row.append(np.where(keep, r, 0.0).astype(np.float32))
            if keep is not None:
                reqs.append(row)
        return cmax, carg, reqmax, reqs

    def iters(c):
        return np.maximum(1, np.ceil(c - 1e-9)).astype(int)

    cmax, carg, reqmax, reqs = sweep(g, cls_mask, keep=counted)
    sel = [1] if interior_only else [0, 1, 2]
    it = iters(cmax[sel].max(0))
    arg = np.take_along_axis(carg[sel], cmax[sel].argmax(0)[None], 0)[0]
    S = int(it.sum())
    it_int = iters(cmax[1])
    S_by_classes = {}
    for combo in ((1,), (1, 2), (0, 1), (0, 1, 2), (0,), (2,)):
        S_by_classes["+".join(CLASSES[c] for c in combo)] = int(iters(cmax[list(combo)].max(0)).sum())
    sweeps = int(it.max(1).sum())
    print(f"S = {S} sweeps per step (minimum {nk * nth}), {sweeps} frequency sweeps; ITER at f1: "
          f"{it[0].min()}..{it[0].max()}; interior nodes only: S = {int(it_int.sum())}, "
          f"{int(it_int.max(1).sum())} frequency sweeps, ITER at f1 {it_int[0].min()}..{it_int[0].max()}", flush=True)
    print("S by node classes counted:", S_by_classes, flush=True)
    # class of the node that sets ITER, per component
    setter = iobp[arg]
    set_by = {CLASSES[c]: int((setter == c).sum()) for c in range(3)}

    # binding nodes: alone they would force the ITER of a component with ITER > 1
    bind = np.zeros(np_, int)
    for ik in range(nk):
        for j in range(nth):
            if it[ik, j] > 1:
                bind += reqs[ik][j] > it[ik, j] - 1
    nbind = int((bind > 0).sum())

    # S after relieving the N worst nodes (ranked by reqmax)
    rq = np.where(counted, reqmax, 0.0)
    order = np.argsort(-np.where(counted, reqmax, -1))
    Ns = [n for n in (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)
          if n < counted.sum()]
    Srem = np.zeros(len(Ns), int)
    for ik in range(nk):
        for j in range(nth):
            r = reqs[ik][j][order]
            suf = np.maximum.accumulate(r[::-1])[::-1]
            Srem += np.maximum(1, np.ceil(suf[Ns] - 1e-9)).astype(int)

    def s_scaled(fac):
        return int(sum(max(1, int(np.ceil((reqs[ik][j] * fac).max() - 1e-9))) for ik in range(nk) for j in range(nth)))

    # nodes above K sub-steps (hww3-grid cfl_substeps test), per class
    Ks = [2, 3, 4, 5, 6, 7, 8, 9]
    above = {K: {CLASSES[c]: int((cls_mask[c] & (reqmax > K)).sum()) for c in range(3)} for K in Ks}
    # S if the nodes above K were one HEALPix level coarser, or as many levels
    # as they need to reach K (each level halves the requirement; neighbours
    # and the nside_min floor ignored)
    Scoarse, ncoarse, Sfull, nlev = [], [], [], []
    for K in Ks:
        Scoarse.append(s_scaled(np.where(rq > K, 0.5, 1.0)))
        ncoarse.append(int((rq > K).sum()))
        lev = np.where(rq > K, np.ceil(np.log2(np.maximum(rq, 1e-30) / K) - 1e-9), 0).astype(int)
        Sfull.append(s_scaled(0.5 ** lev))
        nlev.append({int(n): int((lev == n).sum()) for n in np.unique(lev[lev > 0])})
    # S if every node had at least the median efficiency of its level and
    # class (interior / boundary)
    bnd = ~interior
    effmed = np.full(np_, np.nan)
    med = {}
    for d in np.unique(level[counted]):
        for name, m in (("interior", interior), ("boundary", bnd)):
            sel_n = (level == d) & m & counted & np.isfinite(eff)
            if sel_n.any():
                v = float(np.median(eff[sel_n]))
                med.setdefault(name, {})[int(2 ** d)] = v
                effmed[(level == d) & m] = v
    gfac = np.where(eff < effmed, eff / effmed, 1.0)
    Sgeom = s_scaled(gfac)

    res = {
        "label": label, "nodes": np_, "triangles": ne,
        "fsbccfl": not interior_only,
        "classes": {CLASSES[c]: int(cls_mask[c].sum()) for c in range(3)}, "dry": int((~wet).sum()),
        "interior": int(interior.sum()),
        "coastal_by_triangles": {str(n): int((cls_mask[0] & (ntri == n)).sum()) for n in range(1, 7)},
        "nk": nk, "nth": nth, "freqs": freqs.tolist(), "dt": DT,
        "S": S, "S_min": nk * nth, "frequency_sweeps": sweeps,
        "iter_f1": [int(it[0].min()), int(it[0].max())],
        "S_interior": int(it_int.sum()), "frequency_sweeps_interior": int(it_int.max(1).sum()),
        "S_by_classes": S_by_classes, "iter_set_by_class": set_by,
        "iter_max_by_freq": it.max(1).tolist(), "iter_mean_by_freq": it.mean(1).tolist(),
        "iter_max_by_freq_interior": it_int.max(1).tolist(),
        "n_binding_nodes": nbind,
        "above_K": above,
        "removal": {"N": Ns, "S": Srem.tolist()},
        "coarsen": {"K": Ks, "S": Scoarse, "nodes": ncoarse},
        "coarsen_to_K": {"K": Ks, "S": Sfull, "levels": nlev},
        "S_geometry_fixed": Sgeom,
        "eff_median_by_nside": med,
    }
    top = order[:40]
    res["top_nodes"] = [
        {"lon": round(float(lon[i]), 4), "lat": round(float(lat[i]), 4), "class": CLASSES[iobp[i]],
         "ntri": int(ntri[i]), "nside": int(2 ** level[i]),
         "depth": round(float(depth[i]), 1), "req": round(float(reqmax[i]), 2), "bind": int(bind[i]),
         "cell_km": round(float(cell[i] / 1e3), 2), "L_km": round(float(lgeo[i] / 1e3), 2),
         "eff": round(float(eff[i]), 3), "eff_med": round(float(effmed[i]), 3), "jump": int(jump[i]),
         "near_boundary": bool(near_bnd[i]), "amin": round(float(amin[i]), 1), "amax": round(float(amax[i]), 1)}
        for i in top]
    # causes among the binding nodes
    b = bind > 0
    res["binding_causes"] = {
        "by_class": {CLASSES[c]: int((b & (iobp == c)).sum()) for c in range(3)},
        "coastal_by_triangles": {str(n): int((b & (iobp == 0) & (ntri == n)).sum()) for n in range(1, 7)},
        "low_eff (<0.7 x median of level and class)": int((b & (eff < 0.7 * effmed)).sum()),
        "level_jump>=1": int((b & (jump >= 1)).sum()),
        "near_boundary": int((b & near_bnd).sum()),
        "depth>200m": int((b & (depth > 200)).sum()),
        "depth<50m": int((b & (depth < 50)).sum()),
        "by_nside": {int(2 ** d): int((b & (level == d)).sum()) for d in np.unique(level[b])} if b.any() else {},
    }

    del reqs
    if cur is not None:
        cc, _, _, _ = sweep(g, cls_mask, add=cur)
        itc = iters(cc[sel].max(0))
        res["with_currents_upper_bound"] = {"S": int(itc.sum()), "iter_max_by_freq": itc.max(1).tolist(),
                                            "hourly_fields": ncur}
        print(f"with currents (upper bound): S = {int(itc.sum())}", flush=True)

    # coastal nodes in a single triangle removed with their triangle, mesh
    # and classes recomputed (one pass)
    ear = cls_mask[0] & (ntri == 1)
    keep_t = ~ear[tri].any(1)
    g2 = mesh(lon, lat, tri0[keep_t], np_, ths, obc)
    cls2 = [wet & (g2["iobp"] == c) & (g2["ntri"] > 0) for c in range(3)]
    c2, _, _, _ = sweep(g2, cls2)
    it2 = iters(c2[sel].max(0))
    S_ears = {"nodes_removed": int((ntri > 0).sum() - (g2["ntri"] > 0).sum()),
              "triangles_removed": int((~keep_t).sum()),
              "S": int(it2.sum()), "frequency_sweeps": int(it2.max(1).sum()),
              "iter_f1": [int(it2[0].min()), int(it2[0].max())],
              "new_coastal_nodes_in_1_triangle": int((cls2[0] & (g2["ntri"] == 1)).sum())}
    res["remove_single_triangle_coast"] = S_ears
    print("single-triangle coastal nodes removed:", S_ears, flush=True)

    np.savez_compressed(out + ".npz", reqmax=reqmax.astype(np.float32), bind=bind.astype(np.int32),
                        eff=eff.astype(np.float32), lgeo=lgeo.astype(np.float32), interior=interior,
                        counted=counted, iobp=iobp.astype(np.int8), ntri=ntri.astype(np.int16), it=it, arg=arg)
    json.dump(res, open(out + ".json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("S", "S_min", "frequency_sweeps", "S_interior", "S_by_classes",
                                          "iter_set_by_class", "n_binding_nodes", "above_K", "removal",
                                          "coarsen", "coarsen_to_K", "S_geometry_fixed",
                                          "remove_single_triangle_coast", "binding_causes")}, indent=1))
    print("ITER max by freq:", res["iter_max_by_freq"])
    for t in res["top_nodes"][:15]:
        print(t)


if __name__ == "__main__":
    main()
