"""Open-boundary spectra for ww3_bounc from the partitions of a global WW3 run.

Usage:
    python scripts/boundary_spectra.py GRID_DIR OUT_DIR [--glob DIR]
        [--start 20260101T00Z] [--stop 20260201T00Z]

GRID_DIR  ``grids/<name>`` (``raw/obc.u32``, ``raw/lon.f64``, ``raw/lat.f64``).
OUT_DIR   directory for one netCDF spectrum file per station and
          ``spec.list`` (the file list read by ``ww3_bounc``).
DIR       fields of the Ifremer GLOBMULTI global run at 0.5 deg, every 3 h
          (default: the 2026 best estimate on Datarmor). Each file holds the
          height, peak period, mean direction (from), directional spread and
          wind-sea fraction of six partitions (``phs0``..``phs5``, ``ptp*``,
          ``pdir*``, ``pspr*``, ``pws*``).

Stations: for every open-boundary node of the grid, the nearest cell of the
global grid with a valid ``hs`` in the first field; the cells are kept once.
``ww3_bounc`` then interpolates linearly between the two nearest stations
of every boundary node (``BOUND%INTERP = 2``).

Spectrum: the sum over the partitions of a JONSWAP frequency spectrum
(``GAMMA_WIND`` for a partition whose wind-sea fraction exceeds 0.5,
``GAMMA_SWELL`` otherwise) with peak frequency 1/Tp, scaled so that
4 sqrt(m0) is the partition height on the model frequencies, times a
cos^(2s)((theta - theta_m)/2) directional distribution with
s = 2 / sigma^2 - 1 for the spread sigma (rad). The spectra are written on
the model grid (32 frequencies from 0.0373 Hz with ratio 1.1, 36
directions), in the layout of ww3_ounp: ``efth(time, station, frequency,
direction)`` in m2 s rad-1 and ``direction`` as the nautical direction the
waves travel to, ordered so that ww3_bounc needs no conversion.
"""
import argparse
import glob
import os

import netCDF4
import numpy as np

NK, NTH, F1, XFR = 32, 36, 0.0373, 1.1
GAMMA_WIND, GAMMA_SWELL = 3.3, 3.3
NPART = 6
GLOB_DEFAULT = "/home/datawork-WW3/FORECAST/GLOBMULTI/GLOB-30M/FIELD_NC/best_estimate/2026"


def model_grid():
    f = F1 * XFR ** np.arange(NK)
    df = f * (XFR - 1.0 / XFR) / 2.0  # WW3 bin widths (DSII / 2 pi)
    th = np.arange(NTH) * 2 * np.pi / NTH  # WW3 cartesian "to" directions, THOFF = 0
    dth = 2 * np.pi / NTH
    nautical_to = np.mod(90.0 - np.degrees(th), 360.0)  # ww3_bounc: TH = 2.5 pi - D
    return f, df, th, dth, nautical_to


def jonswap(f, fp, gamma):
    sig = np.where(f <= fp, 0.07, 0.09)
    r = np.exp(-((f - fp) ** 2) / (2 * sig ** 2 * fp ** 2))
    return f ** -5.0 * np.exp(-1.25 * (fp / f) ** 4) * gamma ** r


def partition_spectrum(hs, tp, dfrom, spr, wsf, f, df, th, dth):
    """E(f, theta) of one partition on the model grid, m2 s rad-1."""
    fp = max(1.0 / tp, f[0])
    ef = jonswap(f, fp, GAMMA_WIND if wsf > 0.5 else GAMMA_SWELL)
    ef *= (hs / 4.0) ** 2 / np.sum(ef * df)
    sigma = np.radians(max(spr, 5.0))
    s = max(2.0 / sigma ** 2 - 1.0, 0.5)
    th_to = np.radians(270.0 - dfrom)  # cartesian direction the waves travel to
    dd = np.cos(0.5 * (th - th_to)) ** 2
    dd = dd ** s
    dd /= np.sum(dd) * dth
    return ef[:, None] * dd[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("out")
    ap.add_argument("--glob", default=GLOB_DEFAULT)
    ap.add_argument("--start", default="20260101T00Z")
    ap.add_argument("--stop", default="20260201T00Z")
    a = ap.parse_args()

    obc = np.fromfile(os.path.join(a.grid, "raw", "obc.u32"), "<u4")
    lon = np.fromfile(os.path.join(a.grid, "raw", "lon.f64"), "<f8")[obc]
    lat = np.fromfile(os.path.join(a.grid, "raw", "lat.f64"), "<f8")[obc]
    files = sorted(glob.glob(os.path.join(a.glob, "MARC_WW3-GLOB-30M_*.nc")))
    tag = lambda p: p.rsplit("_", 1)[1][:-3]  # noqa: E731
    files = [p for p in files if a.start <= tag(p) <= a.stop]
    print(f"{len(obc)} open-boundary nodes, {len(files)} global fields "
          f"{tag(files[0])} .. {tag(files[-1])}", flush=True)

    d0 = netCDF4.Dataset(files[0])
    glon = np.asarray(d0["longitude"][:], float)
    glat = np.asarray(d0["latitude"][:], float)
    valid = ~np.ma.getmaskarray(d0["hs"][0])
    jj, ii = np.nonzero(valid)
    vlon, vlat = glon[ii], glat[jj]
    cells = []
    for x, y in zip(lon, lat):
        dx = (vlon - x) * np.cos(np.radians(y))
        k = int(np.argmin(dx ** 2 + (vlat - y) ** 2))
        cells.append((int(jj[k]), int(ii[k])))
    cells = sorted(set(cells))
    sj = np.array([c[0] for c in cells])
    si = np.array([c[1] for c in cells])
    nst = len(cells)
    print(f"{nst} stations (global cells); distance to the boundary nodes up to "
          f"{max(np.hypot((glon[si][:, None] - lon) * np.cos(np.radians(lat)), glat[sj][:, None] - lat).min(0)):.2f} deg",
          flush=True)

    f, df, th, dth, naut = model_grid()
    nt = len(files)
    times = np.zeros(nt)
    efth = np.zeros((nt, nst, NK, NTH), np.float32)
    hs_glob = np.full((nt, nst), np.nan)
    for it, p in enumerate(files):
        d = netCDF4.Dataset(p)
        times[it] = netCDF4.date2num(netCDF4.num2date(d["time"][0], d["time"].units,
                                                      only_use_cftime_datetimes=False),
                                     "days since 1990-01-01 00:00:00")
        hs_glob[it] = np.ma.filled(d["hs"][0][sj, si].astype(float), np.nan)
        parts = {}
        for k in range(NPART):
            for v in ("phs", "ptp", "pdir", "pspr", "pws"):
                parts[v, k] = np.ma.filled(d[f"{v}{k}"][0][sj, si].astype(float), np.nan)
        for s in range(nst):
            for k in range(NPART):
                h, tp = parts["phs", k][s], parts["ptp", k][s]
                if not (np.isfinite(h) and np.isfinite(tp)) or h <= 0.01 or tp <= 0:
                    continue
                dfrom = parts["pdir", k][s]
                spr = parts["pspr", k][s] if np.isfinite(parts["pspr", k][s]) else 30.0
                wsf = parts["pws", k][s] if np.isfinite(parts["pws", k][s]) else 0.0
                efth[it, s] += partition_spectrum(h, tp, dfrom, spr, wsf, f, df, th, dth).astype(np.float32)
        d.close()
        if it % 40 == 0:
            print(f"  {tag(p)}", flush=True)

    hs_spec = 4 * np.sqrt(np.einsum("tskd,k->ts", efth.astype(float), df) * dth)
    ok = np.isfinite(hs_glob) & (hs_glob > 0.05)
    print(f"Hs of the rebuilt spectra / global hs: mean ratio "
          f"{np.mean(hs_spec[ok] / hs_glob[ok]):.3f}, mean global hs {np.mean(hs_glob[ok]):.2f} m, "
          f"max {np.nanmax(hs_glob):.2f} m", flush=True)

    os.makedirs(a.out, exist_ok=True)
    names = []
    for s in range(nst):
        name = f"bnd{s:04d}"
        fn = os.path.join(a.out, f"{name}_spec.nc")
        with netCDF4.Dataset(fn, "w") as o:
            o.createDimension("time", None)
            o.createDimension("station", 1)
            o.createDimension("string16", 16)
            o.createDimension("frequency", NK)
            o.createDimension("direction", NTH)
            v = o.createVariable("time", "f8", ("time",))
            v.units = "days since 1990-01-01 00:00:00"
            v.calendar = "standard"
            v[:] = times
            v = o.createVariable("station", "i4", ("station",))
            v[:] = [1]
            v = o.createVariable("station_name", "S1", ("station", "string16"))
            v[:] = netCDF4.stringtochar(np.array([name.ljust(16)], "S16"))
            v = o.createVariable("longitude", "f4", ("station",))
            v.units = "degree_east"
            v[:] = [glon[si[s]]]
            v = o.createVariable("latitude", "f4", ("station",))
            v.units = "degree_north"
            v[:] = [glat[sj[s]]]
            v = o.createVariable("frequency", "f4", ("frequency",))
            v.units = "s-1"
            v[:] = f
            v = o.createVariable("direction", "f4", ("direction",))
            v.units = "degree"
            v.standard_name = "sea_surface_wave_to_direction"
            v[:] = naut
            v = o.createVariable("efth", "f4", ("time", "station", "frequency", "direction"),
                                 fill_value=np.float32(9.96921e36))
            v.units = "m2 s rad-1"
            v.long_name = "sea surface wave directional variance spectral density"
            v[:] = efth[:, s:s + 1]
            o.source = ("rebuilt from the partitions of the Ifremer GLOBMULTI GLOB-30M run "
                        "by HWW3 scripts/boundary_spectra.py")
        names.append(os.path.basename(fn))
    with open(os.path.join(a.out, "spec.list"), "w") as fl:
        for n in names:
            fl.write(f"{n}\n")
    print(f"wrote {nst} files and spec.list in {a.out}", flush=True)


if __name__ == "__main__":
    main()
