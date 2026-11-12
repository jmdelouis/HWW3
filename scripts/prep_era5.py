"""Rewrite the ERA5 files for ww3_prnc (netCDF4 + numpy only).

- wind.nc: hourly u10, v10 on ascending latitudes, CF time in days since 1990.
- ice.nc : monthly mean sea-ice concentration (land -> 0), repeated at the
  first and last time so that the ice obstruction is fixed over the month.

usage: python prep_era5.py ERA5_DIR YYYYMM OUT_DIR
"""
import os
import sys
import numpy as np
from netCDF4 import Dataset, num2date, date2num

src, ym, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
UNITS = 'days since 1990-01-01 00:00:00'


def read(short):
    d = Dataset(os.path.join(src, f'era5_{short}_{ym}.nc'))
    tname = 'valid_time' if 'valid_time' in d.variables else 'time'
    t = d[tname]
    dates = num2date(t[:], t.units, getattr(t, 'calendar', 'standard'))
    lat = d['latitude'][:]
    lon = d['longitude'][:]
    v = d[short][:]
    d.close()
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        v = v[:, ::-1, :]
    return dates, np.asarray(lat), np.asarray(lon), v


def write(path, dates, lat, lon, fields):
    d = Dataset(path, 'w')
    d.createDimension('time', None)
    d.createDimension('latitude', len(lat))
    d.createDimension('longitude', len(lon))
    t = d.createVariable('time', 'f8', ('time',))
    t.units, t.calendar = UNITS, 'standard'
    t[:] = date2num(list(dates), UNITS, 'standard')
    y = d.createVariable('latitude', 'f4', ('latitude',))
    y.units = 'degrees_north'
    y[:] = lat
    x = d.createVariable('longitude', 'f4', ('longitude',))
    x.units = 'degrees_east'
    x[:] = lon
    for name, (val, units) in fields.items():
        v = d.createVariable(name, 'f4', ('time', 'latitude', 'longitude'),
                             zlib=True, complevel=1, fill_value=-9999.0)
        v.units = units
        v[:] = val
    d.close()


dates, lat, lon, u = read('u10')
_, _, _, v = read('v10')
u = np.ma.filled(u.astype('f4'), 0.0)
v = np.ma.filled(v.astype('f4'), 0.0)
write(os.path.join(out, 'wind.nc'), dates, lat, lon,
      {'u10': (u, 'm s-1'), 'v10': (v, 'm s-1')})
print('wind', len(dates), dates[0], dates[-1], 'speed max %.1f' % np.hypot(u, v).max())

idates, _, _, ci = read('siconc')
ci = np.ma.filled(ci.astype('f4'), np.nan)
cm = np.nanmean(ci, axis=0)
cm = np.where(np.isfinite(cm), cm, 0.0).astype('f4')
write(os.path.join(out, 'ice.nc'), [idates[0], idates[-1]], lat, lon,
      {'siconc': (np.stack([cm, cm]), '1')})
print('ice mean field: cells >0.5 =', int((cm > 0.5).sum()), 'of', cm.size)
