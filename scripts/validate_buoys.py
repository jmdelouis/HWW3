"""Compare WW3 Hs and Tm02 of several runs with moored buoys.

Usage:
    python scripts/validate_buoys.py OUT_PREFIX LABEL:RUN_NC:GRID_DIR [...]
        [--buoys DIR] [--match nearest|depth --weights WEIGHTS.json]

LABEL     short run name used in the outputs (for example v2, v3).
RUN_NC    ww3_ounf output (hourly ``hs`` and ``t02`` on the nodes).
GRID_DIR  ``grids/<name>`` of that run (``plot/nodes.npz`` and
          ``raw/depth.f64``, same node order as the WW3 output).
DIR       monthly moored-buoy files of the Copernicus Marine in situ TAC
          (default: the January 2026 directory of the Datarmor mirror of
          INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030).

Method: every buoy file with ``VHM0`` inside the domain is read; samples
with quality flag 1 (good) in the run period, after ``SPINUP_DAYS`` days of
cold start, are averaged in hourly bins. Buoys at the same position (files
of several networks for one platform) are kept once, the one with most
hours. With ``--match nearest`` (the default, that of the earlier tables of
the notes) each buoy is matched in every run to the nearest wet node (3-D
chord distance); the buoy is kept only if this node is closer than
``MAXDIST_KM`` in every run, so all runs are compared on the same buoys and
hours. With ``--match depth`` the model at a buoy is the depth-aware
interpolation of the calibration (``scripts/buoy_op.py build --json``
writes WEIGHTS.json for one grid, which all runs must share): H_s is the
weighted sum over the buoy's nodes, T_m02 is 2 pi sqrt(M0 / M2) from the
weighted moments m0 and m2 of the nodes, and the buoys the weights leave
out are skipped; the class of a buoy is that of its node of largest weight.
The model is interpolated linearly in time to the hourly bins. Statistics are
given for all pairs, per buoy, and per class of the buoy (the HEALPix depth
of its node in the last run, the model depth, the region).

Outputs: OUT_PREFIX_stats.json, OUT_PREFIX_scatter.png and, with two runs or
more, OUT_PREFIX_series.png and OUT_PREFIX_series_tm02.png (Hs and Tm02 at
the four buoys whose RMSE changes most between the last two runs).
"""
import argparse
import datetime
import glob
import json
import os

import matplotlib
import netCDF4
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# VAL_SPINUP_DAYS shortens it for runs started from a restart
SPINUP_DAYS = float(os.environ.get("VAL_SPINUP_DAYS", 3.0))
MAXDIST_KM = 10.0
# VAL_MIN_HOURS lowers it for short runs (the 92 hours of the explicit runs after spin-up)
MIN_HOURS = int(os.environ.get("VAL_MIN_HOURS", 100))
EARTH_KM = 6371.0
BUOY_DEFAULT = ("/home/ref-copernicus-insitu/INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030/"
                "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr_202311/monthly/MO/202601")
REGIONS = [  # (name, lon0, lon1, lat0, lat1), first match wins
    # Channel and Biscay before the Mediterranean box, which also covers the
    # Bay of Biscay south of 46N (the Basque and Landes coasts)
    ("Channel and Biscay", -10.0, 2.0, 43.0, 51.0),
    ("Mediterranean", -5.6, 45.0, 30.0, 46.0),
    ("Baltic", 13.0, 31.0, 53.0, 66.0),
    ("North Sea", -4.0, 13.0, 51.0, 62.0),
    ("Iberia and Canaries", -20.0, -5.6, 26.0, 43.0),
    ("Norwegian and Arctic", -10.0, 45.0, 62.0, 80.0),
    ("Irish and Celtic seas", -12.0, -4.0, 51.0, 56.0),
    ("US and Canada", -100.0, -50.0, 0.0, 80.0),
]


def xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def stats(o, m):
    if len(o) < 2:
        return {"n": int(len(o))}
    d = m - o
    return {
        "n": int(len(o)),
        "obs_mean": float(o.mean()),
        "bias": float(d.mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "si": float(np.sqrt(((d - d.mean()) ** 2).mean()) / o.mean()),
        "r": float(np.corrcoef(o, m)[0, 1]),
    }


def region(lon, lat):
    for name, a, b, c, e in REGIONS:
        if a <= lon <= b and c <= lat <= e:
            return name
    return "other"


def read_var(d, name, t, hours):
    """Hourly means of the good samples of ``name`` (first depth with data)."""
    if name not in d.variables:
        return None
    v = np.ma.filled(np.ma.asarray(d[name][:], dtype=float), np.nan)
    q = np.ma.filled(np.ma.asarray(d[name + "_QC"][:]), 9) if name + "_QC" in d.variables else np.ones_like(v)
    if v.ndim == 2:
        good = np.isfinite(v) & (q == 1)
        k = int(np.argmax(good.sum(0)))
        v, q = v[:, k], q[:, k]
    v = np.where((q == 1) & (v > 0) & (v < 30), v, np.nan)
    h = np.floor(t / 3600.0 + 0.5)
    ok = np.isfinite(v) & np.isfinite(h)
    idx = np.searchsorted(hours, h[ok])
    inside = (idx < len(hours)) & (hours[np.minimum(idx, len(hours) - 1)] == h[ok])
    s = np.zeros(len(hours)); c = np.zeros(len(hours))
    np.add.at(s, idx[inside], v[ok][inside]); np.add.at(c, idx[inside], 1)
    out = np.full(len(hours), np.nan)
    out[c > 0] = s[c > 0] / c[c > 0]
    return out


def read_buoys(bdir, hours, bbox):
    buoys = []
    for f in sorted(glob.glob(os.path.join(bdir, "*.nc"))):
        try:
            d = netCDF4.Dataset(f)
        except OSError:
            continue
        if "VHM0" not in d.variables:
            continue
        lon = np.ma.filled(np.ma.asarray(d["LONGITUDE"][:], dtype=float), np.nan).ravel()
        lat = np.ma.filled(np.ma.asarray(d["LATITUDE"][:], dtype=float), np.nan).ravel()
        lon, lat = float(np.nanmedian(lon)), float(np.nanmedian(lat))
        if not (bbox[0] <= lon <= bbox[1] and bbox[2] <= lat <= bbox[3]):
            continue
        t = netCDF4.date2num(netCDF4.num2date(d["TIME"][:], d["TIME"].units,
                                              only_use_cftime_datetimes=False),
                             "seconds since 2000-01-01")
        t = np.ma.filled(np.ma.asarray(t, dtype=float), np.nan)
        hs = read_var(d, "VHM0", t, hours)
        if hs is None or np.isfinite(hs).sum() < MIN_HOURS:
            continue
        tm = read_var(d, "VTM02", t, hours)
        name = os.path.basename(f).rsplit("_", 1)[0]
        buoys.append({"name": name, "lon": lon, "lat": lat, "hs": hs, "tm02": tm})
    # one buoy per position (several networks distribute the same platform)
    buoys.sort(key=lambda b: -np.isfinite(b["hs"]).sum())
    kept, seen = [], set()
    for b in buoys:
        key = (round(b["lon"], 2), round(b["lat"], 2))
        if key not in seen:
            seen.add(key)
            kept.append(b)
    return kept


def load_run(label, run_nc, grid):
    d = netCDF4.Dataset(run_nc)
    lon = np.asarray(d["longitude"][:], float)
    lat = np.asarray(d["latitude"][:], float)
    th = netCDF4.date2num(netCDF4.num2date(d["time"][:], d["time"].units,
                                           only_use_cftime_datetimes=False),
                          "seconds since 2000-01-01") / 3600.0
    nodes = np.load(os.path.join(grid, "plot", "nodes.npz"))
    assert np.allclose(nodes["lon"], lon, atol=1e-4) and np.allclose(nodes["lat"], lat, atol=1e-4)
    depth = np.fromfile(os.path.join(grid, "raw", "depth.f64"), "<f8")
    assert len(depth) == len(lon)
    print(f"{label}: reading hs and t02 ({len(th)} x {len(lon)})", flush=True)
    hs = np.ma.filled(d["hs"][:].astype(np.float32), np.nan)
    t02 = np.ma.filled(d["t02"][:].astype(np.float32), np.nan) if "t02" in d.variables else None
    wet = np.isfinite(hs).mean(0) > 0.5
    return {"label": label, "lon": lon, "lat": lat, "level": nodes["level"], "depth": depth,
            "th": th, "hs": hs, "t02": t02, "wet": wet}


def interp_time(field, th, node, hours):
    if field is None:
        return np.full(len(hours), np.nan)
    return np.interp(hours, th, field[:, node], left=np.nan, right=np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--buoys", default=BUOY_DEFAULT)
    ap.add_argument("--match", choices=["nearest", "depth"], default="nearest",
                    help="nearest wet node (the earlier tables) or the depth-aware weights of --weights")
    ap.add_argument("--weights", help="JSON of scripts/buoy_op.py build --json, for --match depth")
    a = ap.parse_args()
    if a.match == "depth" and not a.weights:
        ap.error("--match depth needs --weights")
    W = {}
    if a.match == "depth":
        W = {r["name"]: r for r in json.load(open(a.weights))["buoys"]}

    runs = [load_run(*r.split(":")) for r in a.runs]
    th = runs[-1]["th"]
    hours = np.arange(np.ceil(th[0] + 24 * SPINUP_DAYS), np.floor(th[-1]) + 1)
    lon, lat = runs[-1]["lon"], runs[-1]["lat"]
    bbox = (lon.min(), lon.max(), lat.min(), lat.max())
    buoys = read_buoys(a.buoys, hours, bbox)
    print(f"{len(buoys)} buoys with at least {MIN_HOURS} good hours in the box", flush=True)

    trees = [cKDTree(xyz(r["lon"][r["wet"]], r["lat"][r["wet"]])) for r in runs]
    wet_idx = [np.flatnonzero(r["wet"]) for r in runs]
    rows = []
    left_out = 0
    for b in buoys:
        p = xyz(np.array([b["lon"]]), np.array([b["lat"]]))[0]
        if a.match == "depth":
            wr = W.get(b["name"])
            if wr is None or not wr["kept"]:
                left_out += 1
                continue
            nodes, wts = np.array(wr["nodes"]), np.array(wr["weights"])
            main_n = int(nodes[np.argmax(wts)])
            row = {"name": b["name"], "lon": b["lon"], "lat": b["lat"],
                   "region": region(b["lon"], b["lat"]), "obs_hs": b["hs"], "obs_tm02": b["tm02"]}
            for r in runs:
                if len(r["lon"]) <= nodes.max():
                    raise SystemExit("--weights: node %d beyond the %d nodes of run %s" % (nodes.max(), len(r["lon"]), r["label"]))
                hs_n = np.array([interp_time(r["hs"], r["th"], n, hours) for n in nodes])
                t_n = np.array([interp_time(r["t02"], r["th"], n, hours) for n in nodes])
                m0 = (hs_n / 4) ** 2
                m2 = m0 * (2 * np.pi / t_n) ** 2
                q = xyz(np.array([r["lon"][main_n]]), np.array([r["lat"][main_n]]))[0]
                row[r["label"]] = {"dist_km": float(2 * np.arcsin(np.linalg.norm(p - q) / 2) * EARTH_KM), "node": main_n,
                                   "level": int(r["level"][main_n]), "depth": float(r["depth"][main_n]),
                                   "hs": (wts[:, None] * hs_n).sum(0),
                                   "tm02": 2 * np.pi * np.sqrt((wts[:, None] * m0).sum(0) / (wts[:, None] * m2).sum(0))}
            ok = np.isfinite(row["obs_hs"])
            for r in runs:
                ok &= np.isfinite(row[r["label"]]["hs"])
            if ok.sum() < MIN_HOURS:
                continue
            row["ok"] = ok
            rows.append(row)
            continue
        match = []
        for tr, wi in zip(trees, wet_idx):
            dist, k = tr.query(p)
            match.append((2 * np.arcsin(dist / 2) * EARTH_KM, int(wi[k])))
        if max(m[0] for m in match) > MAXDIST_KM:
            continue
        row = {"name": b["name"], "lon": b["lon"], "lat": b["lat"],
               "region": region(b["lon"], b["lat"]), "obs_hs": b["hs"], "obs_tm02": b["tm02"]}
        for r, (dist, node) in zip(runs, match):
            row[r["label"]] = {"dist_km": dist, "node": node, "level": int(r["level"][node]),
                               "depth": float(r["depth"][node]),
                               "hs": interp_time(r["hs"], r["th"], node, hours),
                               "tm02": interp_time(r["t02"], r["th"], node, hours)}
        ok = np.isfinite(row["obs_hs"])
        for r in runs:
            ok &= np.isfinite(row[r["label"]]["hs"])
        if ok.sum() < MIN_HOURS:
            continue
        row["ok"] = ok
        rows.append(row)
    if a.match == "depth":
        print(f"{len(rows)} buoys through the depth-aware weights of {a.weights}, {left_out} left out by them", flush=True)
    else:
        print(f"{len(rows)} buoys matched within {MAXDIST_KM} km in every run", flush=True)

    labels = [r["label"] for r in runs]
    last = labels[-1]

    def pooled(sel, var="hs"):
        out = {}
        for lab in labels:
            o, m = [], []
            for row in sel:
                obs = row["obs_" + var]
                if obs is None:
                    continue
                ok = row["ok"] & np.isfinite(obs) & np.isfinite(row[lab][var])
                o.append(obs[ok]); m.append(row[lab][var][ok])
            if o:
                out[lab] = stats(np.concatenate(o), np.concatenate(m))
            out[lab]["n_buoys"] = int(sum(1 for row in sel if row["obs_" + var] is not None))
        return out

    classes = {
        "all": rows,
        f"{last} nside 2048": [r for r in rows if r[last]["level"] == 11],
        f"{last} nside 1024": [r for r in rows if r[last]["level"] == 10],
        f"{last} nside <= 512": [r for r in rows if r[last]["level"] <= 9],
        "depth < 50 m": [r for r in rows if r[last]["depth"] < 50],
        "depth 50-200 m": [r for r in rows if 50 <= r[last]["depth"] < 200],
        "depth >= 200 m": [r for r in rows if r[last]["depth"] >= 200],
    }
    for reg in sorted({r["region"] for r in rows}):
        classes[reg] = [r for r in rows if r["region"] == reg]
    summary = {
        "runs": {lab: path for lab, path in (r.split(":")[:2] for r in a.runs)},
        "buoy_dir": a.buoys, "match": a.match, "weights": a.weights, "spinup_days": SPINUP_DAYS, "maxdist_km": MAXDIST_KM,
        "min_hours": MIN_HOURS, "n_buoys": len(rows),
        "hs": {k: pooled(v) for k, v in classes.items() if v},
        "tm02": {k: pooled([r for r in v if r["obs_tm02"] is not None], "tm02")
                 for k, v in classes.items() if any(r["obs_tm02"] is not None for r in v)},
        "per_buoy": [],
    }
    for row in rows:
        e = {"name": row["name"], "lon": row["lon"], "lat": row["lat"], "region": row["region"]}
        for lab in labels:
            m = row[lab]
            ok = row["ok"]
            e[lab] = {"dist_km": round(m["dist_km"], 2), "level": m["level"],
                      "depth": round(m["depth"], 1), **stats(row["obs_hs"][ok], m["hs"][ok])}
            if row["obs_tm02"] is not None:
                okt = ok & np.isfinite(row["obs_tm02"]) & np.isfinite(m["tm02"])
                for other in labels:
                    okt &= np.isfinite(row[other]["tm02"])
                e[lab]["tm02"] = stats(row["obs_tm02"][okt], m["tm02"][okt])
        summary["per_buoy"].append(e)
    with open(a.out + "_stats.json", "w") as f:
        json.dump(summary, f, indent=1)

    # figures: scatter of the last run, map of the RMSE change, time series
    o = np.concatenate([r["obs_hs"][r["ok"]] for r in rows])
    m = np.concatenate([r[last]["hs"][r["ok"]] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    hb = ax[0].hexbin(o, m, gridsize=80, bins="log", cmap="viridis", mincnt=1)
    top = np.nanpercentile(np.r_[o, m], 99.9)
    ax[0].plot([0, top], [0, top], "k--", lw=0.8)
    ax[0].set(xlim=(0, top), ylim=(0, top), xlabel="buoy $H_s$ (m)", ylabel=f"WW3 {last} $H_s$ (m)",
              title=f"{len(rows)} buoys, {len(o)} hours")
    fig.colorbar(hb, ax=ax[0], label="count")
    s = summary["hs"]["all"][last]
    ax[0].text(0.03, 0.97, f"bias {s['bias']:+.3f} m\nRMSE {s['rmse']:.3f} m\nSI {s['si']:.3f}\nr {s['r']:.4f}",
               transform=ax[0].transAxes, va="top", fontsize=9)
    if len(labels) > 1:
        prev = labels[-2]
        dr = np.array([e[last]["rmse"] - e[prev]["rmse"] for e in summary["per_buoy"]])
        bl = np.array([e["lon"] for e in summary["per_buoy"]])
        bt = np.array([e["lat"] for e in summary["per_buoy"]])
        lim = max(0.05, np.nanpercentile(np.abs(dr), 95))
        r0 = runs[-1]
        sub = slice(None, None, 7)
        ax[1].scatter(r0["lon"][r0["wet"]][sub], r0["lat"][r0["wet"]][sub], s=0.05, c="0.85", rasterized=True)
        sc = ax[1].scatter(bl, bt, c=dr, s=18, cmap="RdBu_r", vmin=-lim, vmax=lim, edgecolors="k", linewidths=0.3)
        fig.colorbar(sc, ax=ax[1], label=f"RMSE {last} $-$ RMSE {prev} (m)")
        sel = np.isfinite(bl)
        ax[1].set(xlim=(bl[sel].min() - 3, bl[sel].max() + 3), ylim=(bt[sel].min() - 2, bt[sel].max() + 2),
                  xlabel="longitude", ylabel="latitude", title="$H_s$ RMSE change per buoy")
        ax[1].set_aspect(1.5)
    fig.tight_layout()
    fig.savefig(a.out + "_scatter.png", dpi=150)
    plt.close(fig)

    # time series at the buoys where the runs differ most, for Hs and for Tm02
    if len(labels) > 1:
        days = hours / 24.0 - (netCDF4.date2num(datetime.datetime(2026, 1, 1), "hours since 2000-01-01") / 24.0) + 1
        for var, unit, suffix in (("hs", "$H_s$ (m)", ""), ("tm02", "$T_{m02}$ (s)", "_tm02")):
            def change(e):
                x, y = (e[last], e[labels[-2]]) if var == "hs" else (e[last].get("tm02", {}), e[labels[-2]].get("tm02", {}))
                return abs(x["rmse"] - y["rmse"]) if "rmse" in x and "rmse" in y else -1.0
            dch = np.array([change(e) for e in summary["per_buoy"]])
            order = [i for i in np.argsort(dch)[::-1][:4] if dch[i] >= 0]
            if not order:
                continue
            fig, axs = plt.subplots(len(order), 1, figsize=(11, 2.3 * len(order)), sharex=True)
            for axk, i in zip(np.atleast_1d(axs), order):
                row = rows[i]
                axk.plot(days, row["obs_" + var], "k.", ms=2, label="buoy")
                for lab in labels:
                    ok = row["ok"]
                    axk.plot(days, np.where(ok, row[lab][var], np.nan), lw=1,
                             label=f"{lab} (nside {2 ** row[lab]['level']}, {row[lab]['dist_km']:.1f} km)")
                axk.set_ylabel(unit)
                axk.set_title(f"{row['name']} ({row['lon']:.2f}, {row['lat']:.2f}), {row['region']}", fontsize=9)
                axk.legend(fontsize=7, loc="upper right")
            np.atleast_1d(axs)[-1].set_xlabel("day of January 2026")
            fig.tight_layout()
            fig.savefig(a.out + "_series" + suffix + ".png", dpi=150)
            plt.close(fig)
    print(json.dumps({k: v for k, v in summary["hs"].items()}, indent=1))


if __name__ == "__main__":
    main()
