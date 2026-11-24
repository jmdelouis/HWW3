"""Observation diagnostics of the calibration: buoys, altimeters, loss.

    python scripts/obs_diag.py OUT_DIR [--scratch S]

From the stored outputs only, on a CPU: the start run (cal/it0_value, "p0")
and the run at step 3 (cal/it3_value, "s3"), on the training and validation
windows of the calibration, through its operators (cal_obs.py; the buoys
through the depth-aware operator of buoy_op.py, and through the nearest
node for comparison).

A. Buoys: a table per buoy (platform from its file, position, depths,
   distance to the coast, land fraction and GEBCO spread of its cell,
   HEALPix level, hours, mean observed H_s, bias, RMSE, standard
   deviation, correlation, median error, 1.4826 MAD, change from p0 to s3,
   share of the buoy H_s cost of the second cost), the concentration of the
   squared error over the buoys (top 5, 10, 20; scores without the top
   10), and figures: per-buoy scores against the descriptors, maps, time
   series of the 10 worst and 3 good buoys, model against observation and
   the error by observed H_s and by wind speed at the node.
B. Altimeters: residual statistics (bias, standard deviation, RMSE,
   1.4826 MAD, excess kurtosis, share of the mean square from the largest
   1 % and 5 %), histograms and normal QQ plots, the residual by observed
   H_s, mission, distance to the coast and to the ice, a map of the
   residual on a 2-degree grid, and the spread of the 7 samples of each
   block (the operators rebuilt with alt_sd, op_<window>_sd.npz).
C. Loss: from the residuals of the second cost and the Jacobian A1 (the
   tangent runs of step 1), the Gauss-Newton step of the L2 cost and the
   one-reweighting (IRLS) steps of Huber losses (delta 1, 1.5 and 2 times
   the observation error) and of an L1 loss, at p0 and at s3, with their
   direction against the L2 step.

OUT_DIR gets the figures (PNG), the per-buoy tables (CSV) and summary.json.
"""
import argparse
import csv
import datetime
import glob
import json
import os
import struct
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cal_obs  # noqa: E402

EARTH_KM = 6371.0
WINDOWS = ("train", "valid")
RUNS = ("p0", "s3")
NAMES = ("BETAMAX", "SDSC2", "NLPROP")


def km_chord(d):
    return 2 * np.arcsin(np.clip(d / 2, 0, 1)) * EARTH_KM


def robust(e):
    e = np.asarray(e, float)
    if len(e) == 0:
        return {}
    med = float(np.median(e))
    return {"n": int(len(e)), "bias": float(e.mean()), "std": float(e.std()), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "median": med, "mad": float(1.4826 * np.median(np.abs(e - med)))}


def platform(bdir, name):
    fs = sorted(glob.glob(os.path.join(bdir, name + "_*.nc")))
    if not fs:
        return {}
    with netCDF4.Dataset(fs[0]) as d:
        at = {a: str(d.getncattr(a)) for a in d.ncattrs()}
    return {"source": at.get("source", ""), "category": at.get("source_platform_category_code", ""),
            "network": name.split("_")[0], "platform_name": at.get("platform_name", "")}


def read_wind(path, tags):
    """Wind speed per node at the hours TAGS (YYYYMMDDHH) from a ww3_prnc file (big endian)."""
    want = {(int(t[:8]), int(t[8:]) * 10000): t for t in tags}
    out = {}
    with open(path, "rb") as f:
        def rec():
            h = f.read(4)
            if len(h) < 4:
                return None
            n = struct.unpack(">i", h)[0]
            b = f.read(n)
            f.read(4)
            return b
        rec()
        while True:
            t = rec()
            if t is None:
                break
            d, s = struct.unpack(">2i", t[:8])
            u, v = rec(), rec()
            if (d, s) in want:
                uu = np.frombuffer(u, ">f4").astype(float)
                vv = np.frombuffer(v, ">f4").astype(float)
                out[want[(d, s)]] = np.hypot(uu, vv)
            if len(out) == len(want):
                break
    return out


def binned(x, y, edges):
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (x >= a) & (x < b) & np.isfinite(y)
        if s.sum() >= 5:
            rows.append({"lo": float(a), "hi": float(b), "n": int(s.sum()), "bias": float(y[s].mean()),
                         "rmse": float(np.sqrt(np.mean(y[s] ** 2))), "std": float(y[s].std())})
    return rows


def plot_binned(ax, rows_by_run, xlabel, title):
    for run, rows in rows_by_run.items():
        if not rows:
            continue
        xc = [(r["lo"] + r["hi"]) / 2 for r in rows]
        ax.plot(xc, [100 * r["bias"] for r in rows], "o-", label="%s bias" % run)
        ax.plot(xc, [100 * r["rmse"] for r in rows], "s--", label="%s RMSE" % run)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cm")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)


def load_grid(grid):
    nodes = np.load(os.path.join(grid, "plot", "nodes.npz"))
    lon, lat = nodes["lon"].astype(float), nodes["lat"].astype(float)
    tri = np.fromfile(os.path.join(grid, "raw", "tri.u32"), "<u4").reshape(-1, 3).astype(np.int64)
    obc = set(np.fromfile(os.path.join(grid, "raw", "obc.u32"), "<u4").tolist())
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e.sort(1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    coast = np.array([i for i in np.unique(ue[cnt == 1]) if i not in obc])
    return lon, lat, coast


# --------------------------------------------------------------------------
# A. buoys
# --------------------------------------------------------------------------

def buoys(a, out, summary):
    S, C, X = a.scratch, os.path.join(a.scratch, "cal"), os.path.join(a.scratch, "coast")
    run_dir = {"p0": os.path.join(C, "it0_value"), "s3": os.path.join(C, "it3_value")}
    bdir = os.path.join(S, "obs", "insitu_MO", "202601")
    lon, lat, coast = load_grid(os.path.join(S, "grids", "natl_v5k4"))
    depths = json.load(open(os.path.join(X, "buoy_depths.json")))
    sub = json.load(open(os.path.join(X, "buoy_subgrid_all.json")))
    plat = {}
    series = {}
    allrows = {}
    wind_tags = set()
    for w in WINDOWS:
        od = np.load(os.path.join(C, "op_%s_depth.npz" % w))
        oc = np.load(os.path.join(C, "op_%s_cost2.npz" % w))
        on = np.load(os.path.join(C, "op_%s.npz" % w))
        tags = [str(t) for t in od["tags"]]
        wind_tags |= set(tags)
        names = [str(n) for n in od["buoys"]]
        rows_c = {r["name"]: r for r in json.load(open(os.path.join(X, "coast_%s.json" % w)))["rows"]}
        md, mn = {}, {}
        for run in RUNS:
            _, md[run], _ = cal_obs.apply(od, run_dir[run])
            _, mn[run], _ = cal_obs.apply(on, run_dir[run])
        sig = oc["bh_sigma"]
        rd = {run: md[run] - od["bh_obs"] for run in RUNS}
        rn = {run: mn[run] - on["bh_obs"] for run in RUNS}
        cost_tot = {run: float(np.sum((rd[run] / sig) ** 2)) for run in RUNS}
        rows = []
        for ib, name in enumerate(names):
            if name not in plat:
                plat[name] = platform(bdir, name)
            s = od["bh_b"] == ib
            sn = on["bh_b"] == ib
            c = rows_c.get(name, {})
            node = c.get("node")
            g = depths.get(name) or {}
            row = {"window": w, "name": name, **plat[name], "lon": c.get("lon"), "lat": c.get("lat"),
                   "buoy_depth_gebco": g.get("cell") if g.get("cell") is not None else g.get("med3"),
                   "buoy_depth_meta": c.get("bottom_depth"), "node": node, "node_depth": c.get("node_depth"),
                   "dist_node_km": c.get("dist_km"), "coast_km": c.get("coast_km"), "level": c.get("level"),
                   "land": sub.get(str(node), {}).get("land"), "gebco_std": sub.get(str(node), {}).get("std"),
                   "kept": bool(od["bw_kept"][ib]), "hours": int(s.sum()),
                   "obs_mean": float(od["bh_obs"][s].mean()) if s.any() else None}
            for run in RUNS:
                e = rd[run][s]
                q = robust(e)
                row.update({"%s_%s" % (k, run): v for k, v in q.items() if k != "n"})
                row["corr_%s" % run] = float(np.corrcoef(md[run][s], od["bh_obs"][s])[0, 1]) if s.sum() > 2 else None
                row["sse_%s" % run] = float(np.sum(e ** 2))
                row["cost_share_%s" % run] = float(np.sum((e / sig[s]) ** 2) / cost_tot[run])
                qn = robust(rn[run][sn])
                row["bias_nearest_%s" % run], row["rmse_nearest_%s" % run] = qn.get("bias"), qn.get("rmse")
            if s.any():
                row["dbias"] = row["bias_s3"] - row["bias_p0"]
                row["drmse"] = row["rmse_s3"] - row["rmse_p0"]
            rows.append(row)
            if s.any():
                k = od["bh_k"][s]
                ser = series.setdefault(name, {"t": [], "obs": [], "p0": [], "s3": [], "node": node})
                ser["t"] += [tags[i] for i in k]
                ser["obs"] += od["bh_obs"][s].tolist()
                ser["p0"] += md["p0"][s].tolist()
                ser["s3"] += md["s3"][s].tolist()
        allrows[w] = rows
        # pooled scores, both operators, and the concentration of the squared error
        pooled = {}
        for run in RUNS:
            pooled[run] = {"depth_aware": robust(rd[run]), "nearest": robust(rn[run])}
        act = [r for r in rows if r["hours"]]
        order = sorted(act, key=lambda r: -r["sse_p0"])
        tot = sum(r["sse_p0"] for r in act)
        conc = {"top%d" % k: sum(r["sse_p0"] for r in order[:k]) / tot for k in (5, 10, 20)}
        top10 = {r["name"] for r in order[:10]}
        keep = np.isin(od["bh_b"], [i for i, n in enumerate(names) if n not in top10])
        for run in RUNS:
            conc["without_top10_%s" % run] = robust(rd[run][keep])
        conc["top10_names"] = [r["name"] for r in order[:10]]
        summary["buoys_%s" % w] = {"pooled": pooled, "concentration": conc}
        with open(os.path.join(out, "buoys_%s.csv" % w), "w", newline="") as f:
            keys = list(rows[0].keys())
            for r in rows:
                for k in r:
                    if k not in keys:
                        keys.append(k)
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for r in sorted(rows, key=lambda r: -(r.get("sse_p0") or 0)):
                wr.writerow(r)
        # per-buoy scores against the descriptors
        desc = [("buoy_depth_gebco", "buoy depth (m)", True), ("node_depth", "node depth (m)", True),
                ("coast_km", "node to coast (km)", True), ("land", "land fraction of the cell", False),
                ("gebco_std", "GEBCO std in the cell (m)", True), ("level", "HEALPix level", False),
                ("lat", "latitude", False), ("obs_mean", "mean observed H_s (m)", False), ("hours", "hours", False)]
        ys = [("bias_p0", "bias p0 (cm)"), ("rmse_p0", "RMSE p0 (cm)"), ("dbias", "bias s3 - p0 (cm)"), ("drmse", "RMSE s3 - p0 (cm)")]
        fig, axs = plt.subplots(len(ys), len(desc), figsize=(3 * len(desc), 2.6 * len(ys)), squeeze=False)
        for j, (k, lab, logx) in enumerate(desc):
            x = np.array([r.get(k) if r.get(k) is not None else np.nan for r in act], float)
            for i, (yk, ylab) in enumerate(ys):
                y = 100 * np.array([r.get(yk) if r.get(yk) is not None else np.nan for r in act], float)
                ax = axs[i, j]
                ax.scatter(np.maximum(x, 0.1) if logx else x, y, s=6)
                if logx:
                    ax.set_xscale("log")
                ax.axhline(0, color="k", lw=0.5)
                if i == len(ys) - 1:
                    ax.set_xlabel(lab, fontsize=8)
                if j == 0:
                    ax.set_ylabel(ylab, fontsize=8)
        fig.suptitle("Buoy H_s (depth-aware operator), %s window, %d buoys" % (w, len(act)))
        fig.tight_layout()
        fig.savefig(os.path.join(out, "buoy_scatter_%s.png" % w), dpi=90)
        plt.close(fig)
        # by platform type
        types = sorted({(r.get("source") or "?") + "/" + (r.get("category") or "?") for r in act})
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (yk, ylab) in zip(axs, ys[:3]):
            for i, ty in enumerate(types):
                y = [100 * r[yk] for r in act if (r.get("source") or "?") + "/" + (r.get("category") or "?") == ty and r.get(yk) is not None]
                ax.scatter(np.full(len(y), i) + np.random.uniform(-0.15, 0.15, len(y)), y, s=8)
            ax.set_xticks(range(len(types)))
            ax.set_xticklabels(["%s (%d)" % (t, sum(1 for r in act if (r.get('source') or '?') + '/' + (r.get('category') or '?') == t))
                                for t in types], rotation=40, ha="right", fontsize=7)
            ax.set_ylabel(ylab)
            ax.axhline(0, color="k", lw=0.5)
        fig.suptitle("Buoy H_s by platform type (source / category code of the file), %s window" % w)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "buoy_platform_%s.png" % w), dpi=90)
        plt.close(fig)
        # maps
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        cs = np.arange(0, len(coast), 7)
        for ax, (yk, lab, cmap, lim) in zip(axs, (("bias_p0", "bias p0 (cm)", "RdBu_r", 40), ("rmse_p0", "RMSE p0 (cm)", "viridis", None),
                                                  ("dbias", "bias s3 - p0 (cm)", "RdBu_r", 5))):
            ax.scatter(lon[coast[cs]], lat[coast[cs]], s=0.1, c="0.7")
            v = np.array([100 * r[yk] for r in act])
            sc = ax.scatter([r["lon"] for r in act], [r["lat"] for r in act], c=v, s=18, cmap=cmap,
                            vmin=-lim if lim else None, vmax=lim if lim else np.percentile(v, 95), edgecolors="k", linewidths=0.2)
            plt.colorbar(sc, ax=ax, shrink=0.7)
            ax.set_title(lab)
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
        fig.suptitle("Buoy H_s, %s window" % w)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "buoy_map_%s.png" % w), dpi=90)
        plt.close(fig)
    # time series over both windows: the 10 worst and 3 good buoys at p0
    sse = {n: np.sum((np.array(s["p0"]) - np.array(s["obs"])) ** 2) for n, s in series.items()}
    rms = {n: np.sqrt(np.mean((np.array(s["p0"]) - np.array(s["obs"])) ** 2)) for n, s in series.items() if len(s["obs"]) >= 60}
    worst = sorted(sse, key=lambda n: -sse[n])[:10]
    good = sorted(rms, key=lambda n: rms[n])[:3]
    fig, axs = plt.subplots(13, 1, figsize=(12, 30))
    for ax, n in zip(axs, worst + good):
        s = series[n]
        o = np.argsort(s["t"])
        t = [datetime.datetime.strptime(s["t"][i], "%Y%m%d%H") for i in o]
        ax.plot(t, np.array(s["obs"])[o], "k.", ms=3, label="observed")
        ax.plot(t, np.array(s["p0"])[o], "b-", lw=1, label="p0")
        ax.plot(t, np.array(s["s3"])[o], "r-", lw=1, label="step 3")
        p = plat.get(n, {})
        ax.set_title("%s %s (%s/%s) SSE %.1f m2, RMSE %.2f m" % ("worst" if n in worst else "good", n, p.get("source"), p.get("category"),
                                                                sse[n], np.sqrt(sse[n] / len(o))), fontsize=8)
        ax.set_ylabel("H_s (m)")
        ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "buoy_series.png"), dpi=80)
    plt.close(fig)
    summary["series_buoys"] = {"worst": worst, "good": good}
    # model against observed, and the error by observed H_s and by wind speed at the node
    wind = read_wind(os.path.join(S, "m2", "forcing", "wind.ww3"), sorted(wind_tags))
    obs, err, ws = {r: [] for r in RUNS}, {r: [] for r in RUNS}, []
    mod = {r: [] for r in RUNS}
    for n, s in series.items():
        for i, t in enumerate(s["t"]):
            ws.append(wind[t][s["node"]] if t in wind and s["node"] is not None else np.nan)
            for r in RUNS:
                obs[r].append(s["obs"][i])
                mod[r].append(s[r][i])
                err[r].append(s[r][i] - s["obs"][i])
    ws = np.array(ws)
    fig, axs = plt.subplots(1, 4, figsize=(22, 5))
    for ax, r in zip(axs[:2], RUNS):
        ax.hexbin(obs[r], mod[r], gridsize=60, bins="log", extent=(0, 10, 0, 10))
        ax.plot([0, 10], [0, 10], "w-", lw=0.8)
        ax.set_xlabel("observed H_s (m)")
        ax.set_ylabel("model H_s (m)")
        ax.set_title("buoys, both windows, %s (%d hours)" % (r, len(obs[r])))
    eh = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 10])
    plot_binned(axs[2], {r: binned(np.array(obs[r]), np.array(err[r]), eh) for r in RUNS}, "observed H_s (m)", "buoy H_s error by observed H_s")
    ew = np.array([0, 3, 6, 9, 12, 15, 20, 30])
    plot_binned(axs[3], {r: binned(ws, np.array(err[r]), ew) for r in RUNS}, "wind speed at the node (m/s)", "buoy H_s error by wind speed")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "buoy_model_obs.png"), dpi=90)
    plt.close(fig)
    summary["buoy_by_obs_hs"] = {r: binned(np.array(obs[r]), np.array(err[r]), eh) for r in RUNS}
    summary["buoy_by_wind"] = {r: binned(ws, np.array(err[r]), ew) for r in RUNS}


# --------------------------------------------------------------------------
# B. altimeters
# --------------------------------------------------------------------------

def altimeters(a, out, summary):
    S, C = a.scratch, os.path.join(a.scratch, "cal")
    run_dir = {"p0": os.path.join(C, "it0_value"), "s3": os.path.join(C, "it3_value")}
    lon, lat, coast = load_grid(os.path.join(S, "grids", "natl_v5k4"))
    ctree = cKDTree(cal_obs.xyz(lon[coast], lat[coast]))
    ice = np.zeros(len(lon))
    for f in (os.path.join(S, "refs", "natl_v5k4_jan2026_exp", "ww3.202601.nc"), os.path.join(S, "runs", "m2_dev_full", "ww3.202601.nc")):
        if os.path.exists(f):
            with netCDF4.Dataset(f) as d:
                nt = d["ice"].shape[0]
                for i in range(0, nt, 48):
                    ice = np.maximum(ice, np.ma.filled(d["ice"][i:i + 48, :], 0.0).max(0))
    inodes = np.flatnonzero(ice > 0)
    itree = cKDTree(cal_obs.xyz(lon[inodes], lat[inodes])) if len(inodes) else None
    for w in WINDOWS:
        o = np.load(os.path.join(C, "op_%s.npz" % w))
        osd = os.path.join(C, "op_%s_sd.npz" % w)
        res = {}
        for run in RUNS:
            alt, _, _ = cal_obs.apply(o, run_dir[run])
            res[run] = alt - o["alt_obs"]
        p = cal_obs.xyz(o["alt_lon"], o["alt_lat"])
        dc = km_chord(ctree.query(p)[0])
        di = km_chord(itree.query(p)[0]) if itree is not None else np.full(len(dc), np.inf)
        st = {}
        for run in RUNS:
            e = res[run]
            q = robust(e)
            sq = np.sort(e ** 2)[::-1]
            q.update(kurtosis=float(stats.kurtosis(e)), skew=float(stats.skew(e)),
                     mse_share_top1=float(sq[: max(1, len(e) // 100)].sum() / sq.sum()),
                     mse_share_top5=float(sq[: max(1, len(e) // 20)].sum() / sq.sum()),
                     rmse_without_bias=float(np.sqrt(np.mean((e - e.mean()) ** 2))))
            st[run] = q
        # binned: observed H_s, distance to the coast, to the ice; by mission
        eh = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 15])
        ec = np.array([0, 10, 25, 50, 100, 200, 500, 5000])
        ei = np.array([0, 25, 50, 100, 200, 500, 20000])
        st["by_obs_hs"] = {r: binned(o["alt_obs"], res[r], eh) for r in RUNS}
        st["by_coast_km"] = {r: binned(dc, res[r], ec) for r in RUNS}
        st["by_ice_km"] = {r: binned(di, res[r], ei) for r in RUNS}
        miss = [str(m) for m in o["alt_mission"]]
        st["by_mission"] = {}
        for m in sorted(set(miss)):
            s = np.array([x == m for x in miss])
            st["by_mission"][m] = {r: robust(res[r][s]) for r in RUNS}
        if os.path.exists(osd):
            sd = np.load(osd)
            if len(sd["alt_obs"]) == len(o["alt_obs"]) and np.array_equal(sd["alt_obs"], o["alt_obs"]):
                s7 = sd["alt_sd"]
                st["noise"] = {"sample_sd_median": float(np.median(s7)), "sample_sd_mean": float(np.mean(s7)),
                               "block_mean_noise_rms": float(np.sqrt(np.mean(s7 ** 2) / 7)),
                               "rmse_p0": st["p0"]["rmse"]}
            else:
                st["noise"] = "op_%s_sd.npz does not match op_%s.npz" % (w, w)
        summary["altimeters_%s" % w] = st
        # figures
        fig, axs = plt.subplots(2, 4, figsize=(22, 9))
        for i, run in enumerate(RUNS):
            e = res[run]
            axs[i, 0].hist(100 * e, bins=120, range=(-200, 200), log=True)
            axs[i, 0].set_title("%s residual (cm): bias %.2f, std %.2f, RMSE %.2f, 1.4826 MAD %.2f, kurtosis %.1f"
                                % (run, 100 * st[run]["bias"], 100 * st[run]["std"], 100 * st[run]["rmse"], 100 * st[run]["mad"],
                                   st[run]["kurtosis"]), fontsize=7)
            stats.probplot(e / e.std(), dist="norm", plot=axs[i, 1])
            axs[i, 1].set_title("normal QQ plot of the standardised residual, %s" % run, fontsize=8)
            plot_binned(axs[i, 2], {run: st["by_obs_hs"][run]}, "observed H_s (m)", "residual by observed H_s")
            plot_binned(axs[i, 3], {run: st["by_coast_km"][run]}, "distance to the coast (km)", "residual by distance to the coast")
            axs[i, 3].set_xscale("symlog", linthresh=10)
        fig.suptitle("Altimeter blocks, %s window (%d blocks)" % (w, len(o["alt_obs"])))
        fig.tight_layout()
        fig.savefig(os.path.join(out, "alt_residuals_%s.png" % w), dpi=90)
        plt.close(fig)
        fig, axs = plt.subplots(1, 4, figsize=(24, 5.5))
        ms = sorted(st["by_mission"])
        for k, run in enumerate(RUNS):
            axs[0].bar(np.arange(len(ms)) + 0.4 * k, [100 * st["by_mission"][m][run]["bias"] for m in ms], 0.4, label="%s bias" % run)
            axs[1].bar(np.arange(len(ms)) + 0.4 * k, [100 * st["by_mission"][m][run]["rmse"] for m in ms], 0.4, label="%s RMSE" % run)
        for ax in axs[:2]:
            ax.set_xticks(np.arange(len(ms)) + 0.2)
            ax.set_xticklabels(["%s (%d)" % (m, st["by_mission"][m]["p0"]["n"]) for m in ms], rotation=40, ha="right", fontsize=7)
            ax.set_ylabel("cm")
            ax.legend(fontsize=7)
        plot_binned(axs[2], st["by_ice_km"], "distance to the nearest ice node (km)", "residual by distance to the ice")
        axs[2].set_xscale("log")
        if isinstance(st.get("noise"), dict):
            sd = np.load(osd)["alt_sd"]
            axs[3].hist(100 * sd, bins=100, range=(0, 150), log=True)
            axs[3].set_title("spread of the 7 samples of a block (cm): median %.1f; noise of the block mean %.1f; RMSE %.1f"
                             % (100 * st["noise"]["sample_sd_median"], 100 * st["noise"]["block_mean_noise_rms"], 100 * st["p0"]["rmse"]),
                             fontsize=7)
        fig.suptitle("Altimeter blocks by mission, by distance to the ice, and the noise inside the blocks, %s window" % w)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "alt_mission_ice_noise_%s.png" % w), dpi=90)
        plt.close(fig)
        # map of the residual on a 2-degree grid
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        bx = np.arange(-104, 50, 2.0)
        by = np.arange(-4, 84, 2.0)
        ix = np.digitize(o["alt_lon"], bx) - 1
        iy = np.digitize(o["alt_lat"], by) - 1
        for ax, (lab, f, cmap, lim) in zip(axs, (("residual std p0 (cm)", lambda e: e.std(), "viridis", None),
                                                  ("residual mean p0 (cm)", lambda e: e.mean(), "RdBu_r", 30),
                                                  ("residual mean s3 (cm)", lambda e: e.mean(), "RdBu_r", 30))):
            run = "s3" if "s3" in lab else "p0"
            g = np.full((len(by), len(bx)), np.nan)
            for j in range(len(by)):
                for i in range(len(bx)):
                    s = (ix == i) & (iy == j)
                    if s.sum() >= 10:
                        g[j, i] = 100 * f(res[run][s])
            im = ax.pcolormesh(bx, by, g, cmap=cmap, vmin=-lim if lim else None, vmax=lim if lim else np.nanpercentile(g, 98))
            plt.colorbar(im, ax=ax, shrink=0.7)
            ax.set_title(lab)
        fig.suptitle("Altimeter residual on a 2-degree grid (cells with 10 blocks or more), %s window" % w)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "alt_map_%s.png" % w), dpi=90)
        plt.close(fig)


# --------------------------------------------------------------------------
# C. loss
# --------------------------------------------------------------------------

def loss(a, out, summary):
    C = os.path.join(a.scratch, "cal")
    tans = [(n, os.path.join(C, "it1_%s" % n)) for n in NAMES]
    sp, mx = 0.25, 0.3
    res = {}
    for label, run, xf in (("p0", "it0_value", None), ("s3", "it3_value", "x3.json")):
        r0, r, jac, cost = cal_obs.evaluate(os.path.join(C, "op_train_cost2.npz"), os.path.join(C, run), tans)
        o = np.load(os.path.join(C, "op_train_cost2.npz"))
        sa = 0.25
        sb, st = o["bh_sigma"], o["bt_sigma"][cost]
        z = np.concatenate([r["alt"] / sa, r["bhs"] / sb, r["btm"][cost] / st])
        A = np.stack([np.concatenate([jac[n][0] / sa, jac[n][1] / sb, jac[n][2][cost] / st]) for n in NAMES], 1)
        x = np.zeros(3) if xf is None else np.array([json.load(open(os.path.join(C, xf)))["x"][i] for i in range(3)])
        P = np.eye(3) / sp ** 2
        blocks = {"alt": slice(0, len(r["alt"])), "bhs": slice(len(r["alt"]), len(r["alt"]) + len(r["bhs"])),
                  "btm": slice(len(r["alt"]) + len(r["bhs"]), len(z))}
        out_l = {"x": x.tolist(), "blocks_kurtosis": {k: float(stats.kurtosis(z[s])) for k, s in blocks.items()}}

        def step(wt):
            H = A.T @ (wt[:, None] * A) + P
            g = A.T @ (wt * z) + P @ x
            dx = -np.linalg.solve(H, g)
            sc = min(1.0, mx / np.abs(dx).max())
            return dx * sc, sc

        def huber(v, d):
            av = np.abs(v)
            return np.where(av <= d, v ** 2, 2 * d * av - d ** 2)
        dx2, sc2 = step(np.ones(len(z)))
        zp = z + A @ dx2
        out_l["L2"] = {"J": float(np.sum(z ** 2) + x @ P @ x), "dx": dx2.tolist(), "scale": sc2,
                       "factors_from_p0": np.exp(x + dx2).tolist(),
                       "J_pred": float(np.sum(zp ** 2) + (x + dx2) @ P @ (x + dx2))}
        for d in (1.0, 1.5, 2.0):
            wt = np.where(np.abs(z) <= d, 1.0, d / np.abs(z))
            dx, sc = step(wt)
            zp = z + A @ dx
            out_l["Huber_%.1f" % d] = {"J": float(np.sum(huber(z, d)) + x @ P @ x), "share_beyond": float(np.mean(np.abs(z) > d)),
                                       "dx": dx.tolist(), "scale": sc, "factors_from_p0": np.exp(x + dx).tolist(),
                                       "J_pred": float(np.sum(huber(zp, d)) + (x + dx) @ P @ (x + dx)),
                                       "cos_with_L2": float(dx @ dx2 / np.linalg.norm(dx) / np.linalg.norm(dx2))}
        wt = 1.0 / np.maximum(np.abs(z), 0.05)
        dx, sc = step(wt)
        zp = z + A @ dx
        out_l["L1"] = {"J": float(np.sum(np.abs(z))), "dx": dx.tolist(), "scale": sc, "factors_from_p0": np.exp(x + dx).tolist(),
                       "J_pred": float(np.sum(np.abs(zp))), "cos_with_L2": float(dx @ dx2 / np.linalg.norm(dx) / np.linalg.norm(dx2))}
        res[label] = out_l
    summary["loss"] = res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--scratch", default="/scratch/jmdeloui/HWW3")
    ap.add_argument("--parts", default="A+B+C")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summary = {}
    parts = a.parts.split("+")
    if "B" in parts:
        altimeters(a, a.out, summary)
    if "C" in parts:
        loss(a, a.out, summary)
    if "A" in parts:
        buoys(a, a.out, summary)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    # a short text summary
    for w in WINDOWS:
        if "altimeters_%s" % w in summary:
            s = summary["altimeters_%s" % w]
            for r in RUNS:
                q = s[r]
                print("altimeters %s %s: bias %+.2f std %.2f RMSE %.2f (without bias %.2f) MAD %.2f cm, kurtosis %.2f, "
                      "MSE share of top 1 %% %.3f, top 5 %% %.3f" % (w, r, 100 * q["bias"], 100 * q["std"], 100 * q["rmse"],
                                                                     100 * q["rmse_without_bias"], 100 * q["mad"], q["kurtosis"],
                                                                     q["mse_share_top1"], q["mse_share_top5"]))
            if isinstance(s.get("noise"), dict):
                print("  noise: sample spread median %.1f cm, block-mean noise %.1f cm" % (100 * s["noise"]["sample_sd_median"],
                                                                                     100 * s["noise"]["block_mean_noise_rms"]))
        if "buoys_%s" % w in summary:
            s = summary["buoys_%s" % w]
            for r in RUNS:
                for op in ("depth_aware", "nearest"):
                    q = s["pooled"][r][op]
                    print("buoys %s %s %-11s: bias %+.2f std %.2f RMSE %.2f MAD %.2f median %+.2f cm (%d hours)"
                          % (w, r, op, 100 * q["bias"], 100 * q["std"], 100 * q["rmse"], 100 * q["mad"], 100 * q["median"], q["n"]))
            c = s["concentration"]
            print("  squared error at p0 from the top 5 / 10 / 20 buoys: %.3f / %.3f / %.3f" % (c["top5"], c["top10"], c["top20"]))
            for r in RUNS:
                q = c["without_top10_%s" % r]
                print("  without the top 10, %s: bias %+.2f RMSE %.2f cm" % (r, 100 * q["bias"], 100 * q["rmse"]))
    if "loss" in summary:
        for lab, q in summary["loss"].items():
            print("loss at %s (kurtosis of the normalised residuals %s):" % (lab, {k: round(v, 2) for k, v in q["blocks_kurtosis"].items()}))
            for k in ("L2", "Huber_1.0", "Huber_1.5", "Huber_2.0", "L1"):
                v = q[k]
                print("  %-9s factors from p0 %s, scale %.2f, J %.1f -> %.1f%s" % (
                    k, ["%.3f" % f for f in v["factors_from_p0"]], v["scale"], v["J"], v["J_pred"],
                    "" if k == "L2" else ", cos with L2 %.3f" % v["cos_with_L2"]))


if __name__ == "__main__":
    main()
