"""Download ERA5 hourly 10 m wind and sea-ice concentration for the HWW3
North Atlantic domain (one netCDF file per variable and month).

usage: python get_era5.py YEAR MONTH OUTDIR
Needs ~/.cdsapirc (CDS API key).
"""
import calendar
import os
import sys
import cdsapi

year, month, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
os.makedirs(out, exist_ok=True)
ndays = calendar.monthrange(year, month)[1]
# North, West, South, East: grid extent (0-80 N, 100 W-45 E) plus a 2 deg margin
AREA = [82, -102, -2, 47]
VARS = {'u10': '10m_u_component_of_wind',
        'v10': '10m_v_component_of_wind',
        'siconc': 'sea_ice_cover'}
c = cdsapi.Client(quiet=True)
for short, name in VARS.items():
    f = os.path.join(out, f'era5_{short}_{year}{month:02d}.nc')
    if os.path.exists(f):
        print('exists', f)
        continue
    c.retrieve('reanalysis-era5-single-levels', {
        'product_type': ['reanalysis'], 'variable': [name],
        'year': [str(year)], 'month': [f'{month:02d}'],
        'day': [f'{d:02d}' for d in range(1, ndays + 1)],
        'time': [f'{h:02d}:00' for h in range(24)],
        'area': AREA, 'data_format': 'netcdf',
        'download_format': 'unarchived'}, f + '.part')
    os.rename(f + '.part', f)
    print('done', f, os.path.getsize(f))
