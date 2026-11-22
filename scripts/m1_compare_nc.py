"""Compare the hourly H_s and T_m02 of the GPU propagation model (hww3-prop)
with a WW3 netCDF output, as the M1 targets of docs/differentiable.md
(section 6) are defined, without numpy (for the hpc login nodes).

The WW3 fields are read with ncdump (ww3_ounf output, dimensions (time,
node)); the GPU fields are the little-endian float32 files hs_YYYYMMDDHH.f32
and t02_YYYYMMDDHH.f32 written by hww3-prop (-999.9 where undefined). For
each node the RMS difference over the hours is computed; the targets are an
H_s RMS difference below 1 cm at the median node and below 5 cm at the 99th
percentile, and the total energy within 0.5 %. The energy is compared here
as sum_i (H_s,i / 4)^2 over the nodes defined in both runs (node sum, not
area-weighted), at every hour.

usage:
    python3 scripts/m1_compare_nc.py WW3.nc GPU_DIR [--out stats.json]
        [--hs-min H] [--worst N --grid GRID_DIR] [--vars hs,t02,uwnd,vwnd]
  --hs-min H  keep only the nodes where H_s reaches H (m) in either run
              (idealised swell: most nodes have no waves)
  --vars V    the fields to compare (default hs,t02; the GPU files are
              V_YYYYMMDDHH.f32, the energy is computed when hs is listed)
  --worst N   list the N nodes with the largest H_s RMS difference, with
              their position and depth from GRID_DIR/raw/{lon,lat,depth}.f64
              and whether they are on the open boundary (GRID_DIR/raw/obc.u32)
"""
import array
import datetime
import json
import math
import os
import re
import subprocess
import sys

UNDEF = -900.0


def header(nc):
    h = subprocess.run(["ncdump", "-h", nc], capture_output=True, text=True, check=True).stdout
    m = re.search(r"node\s*=\s*(\d+)", h)
    if not m:
        sys.exit("no node dimension in %s" % nc)
    units = re.search(r'time:units\s*=\s*"days since ([0-9-]+)[ T]?([0-9:]*)', h)
    base = datetime.datetime.strptime(units.group(1) + " " + (units.group(2) or "00:00:00"), "%Y-%m-%d %H:%M:%S")
    return int(m.group(1)), base


def read_var(nc, var, code="f"):
    """Values of `var` in file order, as array(code), fill values as -999.9."""
    p = subprocess.Popen(["ncdump", "-v", var, "-p", "9,17", nc], stdout=subprocess.PIPE, text=True)
    out = array.array(code)
    data = False
    on = False
    for line in p.stdout:
        if not data:
            data = line.strip() == "data:"
            continue
        if not on:
            if re.match(r"\s*%s\s*=" % var, line):
                on = True
                line = line.split("=", 1)[1]
            else:
                continue
        end = ";" in line
        for tok in line.replace(";", " ").replace(",", " ").split():
            if tok == "}":
                continue
            out.append(-999.9 if tok == "_" else float(tok))
        if end:
            break
    p.stdout.close()
    p.wait()
    return out


def load_f32(path):
    a = array.array("f")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    return a


def raw(path, code):
    a = array.array(code)
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    return a


def main():
    args = sys.argv[1:]
    opt = {}
    for k in ("--out", "--hs-min", "--worst", "--grid", "--vars"):
        if k in args:
            i = args.index(k)
            opt[k] = args[i + 1]
            args = args[:i] + args[i + 2:]
    out = opt.get("--out")
    hs_min = float(opt.get("--hs-min", "-1"))
    nworst = int(opt.get("--worst", "0"))
    nc, gdir = args
    np_, base = header(nc)
    # times in double precision (days), rounded to the second
    times = read_var(nc, "time", "d")
    tags = [(base + datetime.timedelta(seconds=round(t * 86400.0))).strftime("%Y%m%d%H") for t in times]
    if len(set(tags)) != len(tags):
        sys.exit("repeated output hours in %s: %s" % (nc, tags))
    res = {"ww3": nc, "gpu": gdir, "nodes": np_, "hs_min": hs_min}
    energy = []
    hsmax = array.array("f", bytes(4 * np_))
    for var in opt.get("--vars", "hs,t02").split(","):
        w_all = read_var(nc, var)
        if len(w_all) != len(tags) * np_:
            sys.exit("%s: %d values for %d times x %d nodes" % (var, len(w_all), len(tags), np_))
        sq = array.array("d", bytes(8 * np_))
        cnt = array.array("i", bytes(4 * np_))
        dmax = array.array("f", bytes(4 * np_))
        tmax = array.array("i", bytes(4 * np_))
        gmax = array.array("f", bytes(4 * np_))
        wmax = array.array("f", bytes(4 * np_))
        hours = 0
        for it, tag in enumerate(tags):
            gp = os.path.join(gdir, "%s_%s.f32" % (var, tag))
            if not os.path.exists(gp):
                continue
            g = load_f32(gp)
            w = w_all[it * np_:(it + 1) * np_]
            hours += 1
            eg = ew = 0.0
            for i in range(np_):
                a, b = g[i], w[i]
                if a > UNDEF and b > UNDEF:
                    d = a - b
                    sq[i] += d * d
                    cnt[i] += 1
                    if abs(d) > dmax[i]:
                        dmax[i], tmax[i], gmax[i], wmax[i] = abs(d), it, a, b
                    if var == "hs":
                        eg += a * a
                        ew += b * b
                        m = a if a > b else b
                        if m > hsmax[i]:
                            hsmax[i] = m
            if var == "hs":
                energy.append({"time": tag, "gpu": eg / 16.0, "ww3": ew / 16.0,
                               "relative": (eg - ew) / ew if ew > 0 else 0.0})
        keep = [i for i in range(np_) if cnt[i] > 0 and hsmax[i] >= hs_min]
        rmsn = {i: math.sqrt(sq[i] / cnt[i]) for i in keep}
        rms = sorted(rmsn.values())
        n = len(rms)
        if n == 0:
            sys.exit("no common node for %s (no GPU file matches the WW3 times %s..)" % (var, tags[:2]))
        q = lambda f: rms[min(n - 1, int(round(f * (n - 1))))]
        res[var] = {"hours": hours, "nodes": n, "rms_median": q(0.5), "rms_p99": q(0.99), "rms_max": rms[-1]}
        print("%s: %d hours, %d nodes, RMS difference median %.4g, p99 %.4g, max %.4g"
              % (var, hours, n, q(0.5), q(0.99), rms[-1]))
        if var == "hs" and nworst > 0:
            gd = opt.get("--grid")
            lon = raw(os.path.join(gd, "raw", "lon.f64"), "d") if gd else None
            lat = raw(os.path.join(gd, "raw", "lat.f64"), "d") if gd else None
            dep = raw(os.path.join(gd, "raw", "depth.f64"), "d") if gd else None
            obc = set(raw(os.path.join(gd, "raw", "obc.u32"), "I")) if gd else set()
            top = sorted(keep, key=lambda i: -rmsn[i])[:nworst]
            res["hs_worst"] = []
            print("  node      rms     max|d|  at          GPU     WW3     lon      lat     depth  obc")
            for i in top:
                e = {"node": i, "rms": rmsn[i], "max_diff": dmax[i], "time": tags[tmax[i]], "gpu": gmax[i], "ww3": wmax[i]}
                if gd:
                    e.update({"lon": lon[i], "lat": lat[i], "depth": dep[i], "obc": i in obc})
                res["hs_worst"].append(e)
                print("  %-8d %7.4f %7.4f  %s %7.3f %7.3f %8.3f %8.3f %7.1f  %s"
                      % (i, rmsn[i], dmax[i], tags[tmax[i]], gmax[i], wmax[i],
                         lon[i] if gd else 0, lat[i] if gd else 0, dep[i] if gd else 0, i in obc))
    if not energy:
        if out:
            with open(out, "w") as f:
                json.dump(res, f, indent=1)
        return
    worst = max(energy, key=lambda e: abs(e["relative"]))
    res["energy"] = energy
    res["energy_worst_relative"] = worst["relative"]
    print("energy sum (H_s/4)^2: largest relative difference %.3e at %s (GPU %.6g, WW3 %.6g); last hour %.3e"
          % (worst["relative"], worst["time"], worst["gpu"], worst["ww3"], energy[-1]["relative"]))
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
