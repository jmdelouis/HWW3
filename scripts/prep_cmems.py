"""Extract hourly surface currents and sea level from the Datarmor mirror of
Copernicus Marine GLOBAL_ANALYSISFORECAST_PHY_001_024 (PT1H-m, 1/12 deg) for
the HWW3 North Atlantic domain, and write current.nc and level.nc for ww3_prnc.

For each day the most recent model run (largest R date) is used.
Land / missing values are set to 0.

usage: python prep_cmems.py YEAR MONTH OUT_DIR
"""
import calendar
import datetime as dt
import glob
import os
import sys
import numpy as np
from netCDF4 import Dataset, num2date, date2num

MIRROR = ('/home/ref-ocean-model-public/multiparameter/physic/global/cmems/'
          'global-analysisforecast-phy-001-024/cmems_mod_glo_phy_anfc_0.083deg_PT1H-m')
S, N, W, E = -2.0, 82.0, -102.0, 47.0
UNITS = 'days since 1990-01-01 00:00:00'

year, month, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
os.makedirs(out, exist_ok=True)
ndays = calendar.monthrange(year, month)[1]


def create(path, lat, lon, names):
    d = Dataset(path, 'w')
    d.createDimension('time', None)
    d.createDimension('latitude', len(lat))
    d.createDimension('longitude', len(lon))
    t = d.createVariable('time', 'f8', ('time',))
    t.units, t.calendar = UNITS, 'standard'
    y = d.createVariable('latitude', 'f4', ('latitude',))
    y.units = 'degrees_north'
    y[:] = lat
    x = d.createVariable('longitude', 'f4', ('longitude',))
    x.units = 'degrees_east'
    x[:] = lon
    for n, u in names:
        v = d.createVariable(n, 'f4', ('time', 'latitude', 'longitude'),
                             zlib=True, complevel=1, fill_value=-9999.0,
                             chunksizes=(1, len(lat), len(lon)))
        v.units = u
    return d


cur = lev = None
nt = 0
for day in range(1, ndays + 1):
    doy = dt.date(year, month, day).timetuple().tm_yday
    files = sorted(glob.glob(f'{MIRROR}/{year}/{doy:03d}/glo12_rg_1h-m_*.nc'))
    if not files:
        sys.exit(f'no file for {year}-{month:02d}-{day:02d}')
    f = files[-1]
    d = Dataset(f)
    lat = d['latitude'][:]
    lon = d['longitude'][:]
    jy = np.where((lat >= S) & (lat <= N))[0]
    ix = np.where((lon >= W) & (lon <= E))[0]
    ys, ye, xs, xe = jy[0], jy[-1] + 1, ix[0], ix[-1] + 1
    if cur is None:
        cur = create(os.path.join(out, 'current.nc'), lat[ys:ye], lon[xs:xe],
                     [('uo', 'm s-1'), ('vo', 'm s-1')])
        lev = create(os.path.join(out, 'level.nc'), lat[ys:ye], lon[xs:xe],
                     [('zos', 'm')])
    tv = d['time']
    dates = num2date(tv[:], tv.units, getattr(tv, 'calendar', 'standard'))
    tnum = date2num(list(dates), UNITS, 'standard')
    k = len(tnum)
    for name, dst in (('uo', cur), ('vo', cur), ('zos', lev)):
        a = d[name][:, 0, ys:ye, xs:xe]
        dst[name][nt:nt + k] = np.ma.filled(a.astype('f4'), 0.0)
    cur['time'][nt:nt + k] = tnum
    lev['time'][nt:nt + k] = tnum
    nt += k
    d.close()
    print(day, os.path.basename(f), k, flush=True)
cur.close()
lev.close()
print('hours', nt)
