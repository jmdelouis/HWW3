"""Add the FES2014 tide to the hourly Copernicus Marine currents and sea level
of scripts/prep_cmems.py, for ww3_prnc.

usage: python fes_tides.py IN_DIR OUT_DIR [FES_ROOT]

IN_DIR    current.nc (uo, vo) and level.nc (zos) of prep_cmems.py; the
          GLOBAL_ANALYSISFORECAST_PHY_001_024 hourly fields have no tide.
OUT_DIR   current.nc and level.nc, same grid and times, with the tide added.
FES_ROOT  FES2014 directory with fes2014a_currents/{eastward,northward}_velocity
          (Ua, Va in cm/s, Ug, Vg in degrees) and
          fes2014b_elevations_extrapolated/ocean_tide_extrapolated
          (amplitude cm, phase degrees); default: the Datarmor copy.

Method: for every constituent except Sa and Ssa (seasonal, already in the
ocean model), the complex constant A exp(-i g) is interpolated bilinearly
from the 1/16 deg FES grid to the 1/12 deg grid of IN_DIR. The current
constants have gaps near the coast; they are filled first by NFILL passes
of the mean of the valid neighbours (about 7 km per pass); the elevations
are the extrapolated FES2014b fields. The tide at time t is
Re(sum_k f_k A_k exp(i (G_k + u_k - g_k))), with the equilibrium arguments
G and the nodal corrections f, u of pyTMD.constituents.arguments (FES
corrections, TT - UT1 = 69.2 s), as in pyTMD for the FES models. Points
without any FES value keep the ocean-model field.
"""
import datetime as dt
import os
import sys
import time

import numpy as np
from netCDF4 import Dataset, date2num, num2date

import pyTMD.constituents

FES_DEFAULT = "/home/datawork-WW3/FORCING/FES2014/SRC_GLOBAL_RECT"
SKIP = {"sa", "ssa"}
NFILL = 8
DELTAT = 69.2 / 86400.0  # TT - UT1 in 2026 (days)
CHUNK = 24  # hours per write
PYTMD_NAME = {"la2": "lambda2"}  # FES file name -> pyTMD constituent
VARS = [  # (output file, variable, FES directory, amplitude, phase, scale to m or m/s)
    ("current.nc", "uo", "fes2014a_currents/eastward_velocity", "Ua", "Ug", 0.01),
    ("current.nc", "vo", "fes2014a_currents/northward_velocity", "Va", "Vg", 0.01),
    ("level.nc", "zos", "fes2014b_elevations_extrapolated/ocean_tide_extrapolated", "amplitude", "phase", 0.01),
]


def fes_block(path, amp, pha, box):
    """Complex constants A exp(-i g) of one FES file on the lon/lat box (nan where missing)."""
    lat0, lat1, lon0, lon1 = box
    with Dataset(path) as d:
        lat = np.asarray(d["lat"][:], float)
        lon = np.asarray(d["lon"][:], float)
        lonw = np.where(lon >= 180.0, lon - 360.0, lon)
        jy = np.nonzero((lat >= lat0) & (lat <= lat1))[0]
        ix = np.nonzero((lonw >= lon0) & (lonw <= lon1))[0]
        ix = ix[np.argsort(lonw[ix])]
        a = np.ma.filled(d[amp][jy[0]:jy[-1] + 1, :].astype(np.float64), np.nan)[:, ix]
        g = np.ma.filled(d[pha][jy[0]:jy[-1] + 1, :].astype(np.float64), np.nan)[:, ix]
    a[(a > 1e10) | (a < 0)] = np.nan
    return lat[jy], lonw[ix], a * np.exp(-1j * np.radians(g))


def fill(c, npass):
    """Fill nan cells next to valid ones with the mean of their valid neighbours."""
    for _ in range(npass):
        bad = np.isnan(c)
        if not bad.any():
            break
        v = np.where(bad, 0, c)
        w = (~bad).astype(float)
        vp = np.pad(v, 1)
        wp = np.pad(w, 1)
        s = np.zeros_like(v)
        n = np.zeros_like(w)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                if dy == 1 and dx == 1:
                    continue
                s += vp[dy:dy + v.shape[0], dx:dx + v.shape[1]]
                n += wp[dy:dy + v.shape[0], dx:dx + v.shape[1]]
        new = bad & (n > 0)
        c = c.copy()
        c[new] = s[new] / n[new]
    return c


def bilinear(flat, flon, c, lat, lon):
    """Bilinear interpolation of c (flat x flon, regular) to the lat x lon grid, nan neighbours skipped."""
    fy = (lat - flat[0]) / (flat[1] - flat[0])
    fx = (lon - flon[0]) / (flon[1] - flon[0])
    j0 = np.clip(np.floor(fy).astype(int), 0, len(flat) - 2)
    i0 = np.clip(np.floor(fx).astype(int), 0, len(flon) - 2)
    wy = np.clip(fy - j0, 0, 1)[:, None]
    wx = np.clip(fx - i0, 0, 1)[None, :]
    num = np.zeros((len(lat), len(lon)), complex)
    den = np.zeros((len(lat), len(lon)))
    for dj, wj in ((0, 1 - wy), (1, wy)):
        for di, wi in ((0, 1 - wx), (1, wx)):
            v = c[(j0 + dj)[:, None], (i0 + di)[None, :]]
            w = wj * wi * np.isfinite(v)
            num += w * np.where(np.isfinite(v), v, 0)
            den += w
    out = np.full(num.shape, np.nan, complex)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    return out


def create(path, src, names):
    d = Dataset(path, "w")
    d.createDimension("time", None)
    d.createDimension("latitude", len(src["latitude"]))
    d.createDimension("longitude", len(src["longitude"]))
    t = d.createVariable("time", "f8", ("time",))
    t.units, t.calendar = src["time"].units, getattr(src["time"], "calendar", "standard")
    for n in ("latitude", "longitude"):
        v = d.createVariable(n, "f4", (n,))
        v.units = src[n].units
        v[:] = src[n][:]
    for n in names:
        v = d.createVariable(n, "f4", ("time", "latitude", "longitude"), zlib=True, complevel=1,
                             fill_value=-9999.0, chunksizes=(1, len(src["latitude"]), len(src["longitude"])))
        v.units = src[n].units
        v.comment = "Copernicus Marine GLOBAL_ANALYSISFORECAST_PHY_001_024 plus the FES2014 tide (scripts/fes_tides.py)"
    return d


def main():
    ind, out = sys.argv[1], sys.argv[2]
    fes = sys.argv[3] if len(sys.argv) > 3 else FES_DEFAULT
    os.makedirs(out, exist_ok=True)
    src = {f: Dataset(os.path.join(ind, f)) for f in ("current.nc", "level.nc")}
    lat = np.asarray(src["level.nc"]["latitude"][:], float)
    lon = np.asarray(src["level.nc"]["longitude"][:], float)
    tv = src["level.nc"]["time"]
    dates = num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"), only_use_cftime_datetimes=False)
    mjd = np.asarray(date2num(list(dates), "days since 1858-11-17 00:00:00"), float)
    nt = len(mjd)
    print(f"{len(lat)} x {len(lon)} points, {nt} hours from {dates[0]} to {dates[-1]}", flush=True)

    cons = sorted({f[:-3] for f in os.listdir(os.path.join(fes, VARS[0][2])) if f.endswith(".nc")}
                  & {f[:-3] for f in os.listdir(os.path.join(fes, VARS[2][2])) if f.endswith(".nc")})
    cons = [c for c in cons if c not in SKIP]
    pu, pf, G = pyTMD.constituents.arguments(mjd, [PYTMD_NAME.get(c, c) for c in cons],
                                             deltat=DELTAT, corrections="FES")
    F = (pf * np.exp(1j * (np.radians(G) + pu))).astype(np.complex64)  # (nt, ncons)
    print(len(cons), "constituents:", " ".join(cons), flush=True)

    box = (lat.min() - 0.25, lat.max() + 0.25, lon.min() - 0.25, lon.max() + 0.25)
    hc = {}
    for _, name, sub, amp, pha, scale in VARS:
        t0 = time.time()
        h = np.empty((len(lat) * len(lon), len(cons)), np.complex64)
        for k, c in enumerate(cons):
            flat, flon, cb = fes_block(os.path.join(fes, sub, c + ".nc"), amp, pha, box)
            if name != "zos":
                cb = fill(cb, NFILL)
            h[:, k] = (scale * bilinear(flat, flon, cb, lat, lon)).ravel()
        miss = np.isnan(h[:, 0])
        h[np.isnan(h)] = 0
        hc[name] = h
        m2 = np.abs(h[:, cons.index("m2")])
        print(f"{name}: {time.time() - t0:.0f} s, {miss.mean() * 100:.1f} % of the points without FES value, "
              f"M2 amplitude max {m2.max():.2f}, p99 {np.percentile(m2[~miss], 99):.2f}", flush=True)

    files = {}
    for f, names in (("current.nc", ["uo", "vo"]), ("level.nc", ["zos"])):
        files[f] = create(os.path.join(out, f), src[f], names)
        files[f]["time"][:] = src[f]["time"][:]
    t0 = time.time()
    stats = {n: [0.0, 0.0] for n in hc}
    for k0 in range(0, nt, CHUNK):
        k1 = min(k0 + CHUNK, nt)
        for f, name, *_ in VARS:
            tide = (hc[name] @ F[k0:k1].T).real.T.reshape(k1 - k0, len(lat), len(lon))
            base = np.ma.filled(src[f][name][k0:k1].astype(np.float32), 0.0)
            files[f][name][k0:k1] = base + tide
            stats[name][0] = max(stats[name][0], float(np.abs(tide).max()))
            stats[name][1] += float((tide ** 2).sum())
        print(f"hours {k0}-{k1 - 1}: {time.time() - t0:.0f} s", flush=True)
    for name, (mx, s2) in stats.items():
        print(f"tide {name}: max |tide| {mx:.2f}, RMS {np.sqrt(s2 / (nt * len(lat) * len(lon))):.3f}")
    for d in list(files.values()) + list(src.values()):
        d.close()


if __name__ == "__main__":
    main()
