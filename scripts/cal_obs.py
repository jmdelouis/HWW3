"""Observation operators and cost of the calibration of the GPU model
(docs/differentiable.md, section 9.5).

build: the fixed linear operators from the hourly fields of a run to the
observations of a window, with the matching of validate_altimetry.py and
validate_buoys.py:
  - altimeters: every L3 sample of the window matched to the nearest wet,
    ice-free node within 0.75 pixel, interpolated linearly in time, and
    averaged in blocks of 7 consecutive samples of a file (no gap > 3 s);
  - buoys: hourly means of the good VHM0 and VTM02 samples, the nearest wet
    node within 10 km, buoys with at least MIN_HOURS good hours.
The masks (undefined nodes, wet nodes, ice) come from the base run RUN_DIR
and the ice of the WW3 files ICE_NC over the window, and stay fixed. Buoys
whose T_m02 RMSE in the base run exceeds 3 s are kept for the scores and
left out of the T_m02 term of the cost (docs/differentiable.md, section 7).

eval: applies the operators to the hourly H_s and T_m02 of RUN_DIR
(hs_, t02_YYYYMMDDHH.f64, or .f32) and, for each NAME:DIR, to the tangents
dhs_, dt02_ of DIR: the scores (bias and RMSE), the cost
J = sum (r_alt / SA)^2 + sum (r_hs / SB)^2 + sum (r_t02 / ST)^2 over the
observations, and the derivatives of the scores and of J along each
tangent; with --npz the residuals and their derivatives, for cal_gn.py.

usage:
    python scripts/cal_obs.py build OP.npz RUN_DIR START END ICE_NC[+ICE_NC] GRID_DIR [--l3 DIR] [--buoys DIR] [--min-hours N]
    python scripts/cal_obs.py eval OP.npz RUN_DIR [NAME:DIR ...] [--json OUT] [--npz OUT] [--sigma SA+SB+ST]
START and END are YYYYMMDDHH, the first and last hours of the window.
"""
import datetime
import glob
import json
import os
import sys

import netCDF4
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_buoys as vb  # noqa: E402

S = "/scratch/jmdeloui/HWW3"
L3 = S + "/obs/l3"
BUOYS = S + "/obs/insitu_MO/202601"
MAXDIST_FRAC = 0.75
BLOCK = 7
SIGMA = (0.25, 0.30, 1.0)  # m, m, s
TM02_RMSE_MAX = 3.0
EPOCH = datetime.datetime(2000, 1, 1)


def hours_of(start, end):
    t0 = datetime.datetime.strptime(start, "%Y%m%d%H")
    t1 = datetime.datetime.strptime(end, "%Y%m%d%H")
    n = int((t1 - t0).total_seconds() // 3600) + 1
    return [t0 + datetime.timedelta(hours=i) for i in range(n)]


def field(d, name, tag, idx=None):
    for ext, dt in ((".f64", "<f8"), (".f32", "<f4")):
        p = os.path.join(d, "%s_%s%s" % (name, tag, ext))
        if os.path.exists(p):
            v = np.fromfile(p, dt).astype(np.float64)
            v[v < -900] = np.nan
            return v if idx is None else v[idx]
    raise FileNotFoundError(os.path.join(d, "%s_%s" % (name, tag)))


def xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def l3_files(t0, t1, root):
    """L3 files whose name interval overlaps [t0, t1] (seconds since 2000)."""
    out = []
    for f in sorted(glob.glob(os.path.join(root, "*", "2026", "*", "*.nc"))):
        p = os.path.basename(f).split("_")
        a = (datetime.datetime.strptime(p[-3], "%Y%m%dT%H%M%S") - EPOCH).total_seconds()
        b = (datetime.datetime.strptime(p[-2], "%Y%m%dT%H%M%S") - EPOCH).total_seconds()
        if b >= t0 and a <= t1:
            out.append(f)
    return out


def build(op, rdir, start, end, ice_ncs, grid, l3=L3, bdir=BUOYS, min_hours=24):
    times = hours_of(start, end)
    tags = [t.strftime("%Y%m%d%H") for t in times]
    tm = np.array([(t - EPOCH).total_seconds() for t in times])
    nodes = np.load(os.path.join(grid, "plot", "nodes.npz"))
    lon, lat, level = nodes["lon"].astype(float), nodes["lat"].astype(float), nodes["level"]
    nn = len(lon)
    print("window %s to %s: %d hours, %d nodes" % (start, end, len(tags), nn), flush=True)
    defined = np.zeros((len(tags), nn), bool)
    for i, tag in enumerate(tags):
        defined[i] = np.isfinite(field(rdir, "hs", tag))
    ice = np.zeros(nn)
    for f in ice_ncs.split("+"):
        with netCDF4.Dataset(f) as d:
            t = d["time"]
            ft = netCDF4.num2date(t[:], t.units, only_use_cftime_datetimes=False)
            sel = [i for i, x in enumerate(ft) if times[0] <= x <= times[-1]]
            if sel:
                ice = np.maximum(ice, np.ma.filled(d["ice"][sel, :], 0.0).max(0))
    # altimeters (validate_altimetry.py)
    ok = defined.any(0) & (ice <= 0.0)
    idx = np.nonzero(ok)[0]
    tree = cKDTree(xyz(lon[idx], lat[idx]))
    maxchord = MAXDIST_FRAC * np.sqrt(np.pi / 3) / 2.0 ** level
    chord = maxchord.max()
    box = (lon.min(), lon.max(), lat.min(), lat.max())
    t0, t1 = tm[0], tm[-1]
    files = l3_files(t0, t1, l3)
    print(len(files), "L3 files", flush=True)
    blocks = {"obs": [], "k": [], "n": [], "w": [], "mission": [], "level": [], "lon": [], "lat": [], "t": []}
    for f in files:
        mission = f.split("swh_nrt_")[1].split("-l3")[0]
        with netCDF4.Dataset(f) as a:
            t = a["time"][:].astype(float)
            la = np.ma.filled(a["latitude"][:].astype(float), np.nan)
            lo = np.ma.filled(a["longitude"][:].astype(float), np.nan)
            h = np.ma.filled(a["VAVH"][:].astype(float), np.nan)
        lo = np.where(lo > 180, lo - 360, lo)
        s = (t >= t0) & (t <= t1) & np.isfinite(h) & (la >= box[2]) & (la <= box[3]) & (lo >= box[0]) & (lo <= box[1])
        if s.sum() < BLOCK:
            continue
        t, la, lo, h = t[s], la[s], lo[s], h[s]
        dist, j = tree.query(xyz(lo, la), distance_upper_bound=chord)
        s = np.isfinite(dist)
        s[s] = dist[s] <= maxchord[idx[j[s]]]
        if s.sum() < BLOCK:
            continue
        t, la, lo, h, n = t[s], la[s], lo[s], h[s], idx[j[s]]
        k = np.clip(np.searchsorted(tm, t) - 1, 0, len(tm) - 2)
        w = (t - tm[k]) / (tm[k + 1] - tm[k])
        s = defined[k, n] & defined[k + 1, n]
        t, la, lo, h, n, k, w = t[s], la[s], lo[s], h[s], n[s], k[s], w[s]
        nb = len(h) // BLOCK
        if nb == 0:
            continue
        cut = nb * BLOCK
        r = lambda x: x[:cut].reshape(nb, BLOCK)  # noqa: E731
        good = (np.diff(r(t), axis=1) < 3.0).all(1)
        blocks["obs"].append(r(h).mean(1)[good])
        blocks["k"].append(r(k)[good])
        blocks["n"].append(r(n)[good])
        blocks["w"].append(r(w)[good])
        blocks["mission"] += [mission] * int(good.sum())
        blocks["level"].append(np.round(np.median(r(level[n]), 1)).astype(int)[good])
        blocks["lon"].append(r(lo).mean(1)[good])
        blocks["lat"].append(r(la).mean(1)[good])
        blocks["t"].append(r(t).mean(1)[good])
    alt = {k: (np.concatenate(v) if k != "mission" else np.array(v)) for k, v in blocks.items()}
    print("altimeters: %d blocks" % len(alt["obs"]), flush=True)
    # buoys (validate_buoys.py, one run)
    vb.MIN_HOURS = min_hours
    hrs = tm / 3600.0
    buoys = vb.read_buoys(bdir, hrs, box)
    wet = defined.mean(0) > 0.5
    wi = np.flatnonzero(wet)
    tr = cKDTree(xyz(lon[wi], lat[wi]))
    hs_k, hs_n, hs_o, hs_b, t_k, t_n, t_o, t_b, names = [], [], [], [], [], [], [], [], []
    for b in buoys:
        dist, j = tr.query(xyz(np.array([b["lon"]]), np.array([b["lat"]]))[0])
        if 2 * np.arcsin(dist / 2) * vb.EARTH_KM > vb.MAXDIST_KM:
            continue
        node = int(wi[j])
        okh = np.isfinite(b["hs"]) & defined[:, node]
        if okh.sum() < min_hours:
            continue
        ib = len(names)
        names.append(b["name"])
        kk = np.flatnonzero(okh)
        hs_k.append(kk); hs_n.append(np.full(len(kk), node)); hs_o.append(b["hs"][kk]); hs_b.append(np.full(len(kk), ib))
        if b["tm02"] is not None:
            okt = okh & np.isfinite(b["tm02"])
            kt = np.flatnonzero(okt)
            t_k.append(kt); t_n.append(np.full(len(kt), node)); t_o.append(b["tm02"][kt]); t_b.append(np.full(len(kt), ib))
    cat = lambda x, dt: np.concatenate(x).astype(dt) if x else np.zeros(0, dt)  # noqa: E731
    bh = {"k": cat(hs_k, int), "n": cat(hs_n, int), "obs": cat(hs_o, float), "b": cat(hs_b, int)}
    bt = {"k": cat(t_k, int), "n": cat(t_n, int), "obs": cat(t_o, float), "b": cat(t_b, int)}
    # T_m02 of the base run: buoys above 3 s RMSE stay out of the cost
    mt = model_points(rdir, tags, "t02", bt["k"], bt["n"])
    rmse = np.array([np.sqrt(np.nanmean((mt[bt["b"] == i] - bt["obs"][bt["b"] == i]) ** 2)) if (bt["b"] == i).any() else 0.0
                     for i in range(len(names))])
    bt["cost"] = rmse[bt["b"]] <= TM02_RMSE_MAX
    print("buoys: %d, %d hours of H_s, %d of T_m02 (%d buoys above %.0f s of T_m02 RMSE left out of the cost)"
          % (len(names), len(bh["obs"]), len(bt["obs"]), int((rmse > TM02_RMSE_MAX).sum()), TM02_RMSE_MAX), flush=True)
    np.savez(op, tags=np.array(tags), alt_obs=alt["obs"], alt_k=alt["k"], alt_n=alt["n"], alt_w=alt["w"],
             alt_mission=alt["mission"], alt_level=alt["level"], alt_lon=alt["lon"], alt_lat=alt["lat"], alt_t=alt["t"],
             bh_k=bh["k"], bh_n=bh["n"], bh_obs=bh["obs"], bh_b=bh["b"],
             bt_k=bt["k"], bt_n=bt["n"], bt_obs=bt["obs"], bt_b=bt["b"], bt_cost=bt["cost"],
             buoys=np.array(names), tm02_rmse0=rmse)


def model_points(d, tags, name, k, n):
    """Values of the hourly fields NAME of run D at (hour k, node n)."""
    out = np.full(len(k), np.nan)
    for i, tag in enumerate(tags):
        s = k == i
        if s.any():
            out[s] = field(d, name, tag, n[s])
    return out


def apply(o, d, prefix=""):
    """Model values at the observations from run (or tangent) directory D:
    altimeter blocks, buoy H_s and buoy T_m02."""
    tags = list(o["tags"])
    k, n, w = o["alt_k"], o["alt_n"], o["alt_w"]
    # the two bracketing hours of every sample
    kk = np.concatenate([k.ravel(), k.ravel() + 1])
    nn = np.concatenate([n.ravel(), n.ravel()])
    v = model_points(d, tags, prefix + "hs", kk, nn)
    a, b = v[: k.size].reshape(k.shape), v[k.size:].reshape(k.shape)
    alt = ((1 - w) * a + w * b).mean(1)
    bh = model_points(d, tags, prefix + "hs", o["bh_k"], o["bh_n"])
    bt = model_points(d, tags, prefix + "t02", o["bt_k"], o["bt_n"])
    return alt, bh, bt


def scores(r):
    return {"n": int(len(r)), "bias": float(np.mean(r)), "rmse": float(np.sqrt(np.mean(r ** 2)))}


def dscores(r, dr):
    rm = np.sqrt(np.mean(r ** 2))
    return {"bias": float(np.mean(dr)), "rmse": float(np.mean(r * dr) / rm)}


def evaluate(op, rdir, tans, sigma=SIGMA):
    o = np.load(op)
    alt, bh, bt = apply(o, rdir)
    r = {"alt": alt - o["alt_obs"], "bhs": bh - o["bh_obs"], "btm": bt - o["bt_obs"]}
    cost = o["bt_cost"]
    for key, v in r.items():
        if not np.isfinite(v).all():
            raise ValueError("%s: %d undefined model values" % (key, int((~np.isfinite(v)).sum())))
    sa, sb, st = sigma
    jparts = {"alt": float(np.sum((r["alt"] / sa) ** 2)), "bhs": float(np.sum((r["bhs"] / sb) ** 2)),
              "btm": float(np.sum((r["btm"][cost] / st) ** 2))}
    res = {"run": rdir, "sigma": list(sigma), "J": sum(jparts.values()), "J_parts": jparts,
           "scores": {"alt": scores(r["alt"]), "buoy_hs": scores(r["bhs"]), "buoy_tm02": scores(r["btm"]),
                      "buoy_tm02_cost": scores(r["btm"][cost])},
           "n_buoys": int(len(o["buoys"]))}
    jac = {}
    for name, d in tans:
        da, dh, dt = apply(o, d, "d")
        jac[name] = (da, dh, dt)
        res.setdefault("d_scores", {})[name] = {"alt": dscores(r["alt"], da), "buoy_hs": dscores(r["bhs"], dh),
                                                 "buoy_tm02": dscores(r["btm"], dt)}
        res.setdefault("dJ", {})[name] = float(2 * (np.sum(r["alt"] * da) / sa ** 2 + np.sum(r["bhs"] * dh) / sb ** 2
                                                   + np.sum(r["btm"][cost] * dt[cost]) / st ** 2))
    return res, r, jac, cost


def main():
    a = sys.argv[1:]
    opts = {}
    for key in ("--json", "--npz", "--sigma", "--l3", "--buoys", "--min-hours"):
        if key in a:
            i = a.index(key)
            opts[key] = a[i + 1]
            del a[i:i + 2]
    if a[0] == "build":
        build(a[1], a[2], a[3], a[4], a[5], a[6], opts.get("--l3", L3), opts.get("--buoys", BUOYS), int(opts.get("--min-hours", 24)))
        return
    sigma = tuple(float(x) for x in opts["--sigma"].split("+")) if "--sigma" in opts else SIGMA
    tans = [tuple(x.split(":", 1)) for x in a[3:]]
    res, r, jac, cost = evaluate(a[1], a[2], tans, sigma)
    print(json.dumps({k: res[k] for k in ("J", "J_parts", "scores")}, indent=1))
    for name in res.get("dJ", {}):
        print(name, "dJ/dln %.6e" % res["dJ"][name], json.dumps(res["d_scores"][name]))
    if "--json" in opts:
        json.dump(res, open(opts["--json"], "w"), indent=1)
    if "--npz" in opts:
        names = [t[0] for t in tans]
        np.savez(opts["--npz"], names=np.array(names), sigma=np.array(sigma), r_alt=r["alt"], r_bhs=r["bhs"], r_btm=r["btm"][cost],
                 **{"d_%s_%s" % (key, nm): v for nm in names for key, v in zip(("alt", "bhs", "btm"),
                                                                                 (jac[nm][0], jac[nm][1], jac[nm][2][cost]))})


if __name__ == "__main__":
    main()
