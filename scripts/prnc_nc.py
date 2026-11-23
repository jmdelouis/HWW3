"""Write a netCDF forcing file in the form that ``hww3-pre prnc`` reads.

The Rust port of ww3_prnc (hww3_wave::prnc) does the interpolation; this
script only reads the netCDF file, since the crate has no netCDF reader. It
reads the values as ww3_prnc does: as stored (no unpacking, no masking),
converted to single precision, with ``_FillValue``, ``scale_factor`` and
``add_offset`` of each variable in single precision, the axes in single
precision and the times converted from the ``time`` (or ``MT``) variable.

    python scripts/prnc_nc.py wind.nc u10+v10 OUTDIR [--start 20260101] [--stop 20260110230000]

OUTDIR gets meta.json, lon.f32, lat.f32 and one <var>.f32 per variable
(little endian, [time][lat][lon]; the first level of a 4-D variable).
"""
import argparse
import json
import os

import netCDF4
import numpy as np


def f32_bits(v):
    return int(np.asarray(v, dtype=np.float32).reshape(-1)[0].view(np.uint32))


def ww3_time(s):
    d = ''.join(c for c in s if c.isdigit())
    return (int(d[:8]), int((d[8:] + '000000')[:6]))


def first_var(ds, names):
    for n in names:
        if n in ds.variables:
            return ds.variables[n]
    raise SystemExit('none of %s in the file' % (names,))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('nc')
    ap.add_argument('vars', help='VAR or VAR1+VAR2')
    ap.add_argument('out')
    ap.add_argument('--start', help='first time, yyyymmdd[hhmmss]')
    ap.add_argument('--stop', help='last time, yyyymmdd[hhmmss]')
    a = ap.parse_args()
    ds = netCDF4.Dataset(a.nc)
    ds.set_auto_maskandscale(False)
    names = a.vars.split('+')
    lon = first_var(ds, ['longitude', 'lon', 'Longitude', 'x', 'X'])[:]
    lat = first_var(ds, ['latitude', 'lat', 'Latitude', 'y', 'Y'])[:]
    # ww3_prnc: X0I = ALO(1,1), XNI = ALO(NXI,1), Y0I = ALA(1,1), YNI = ALA(1,NYI)
    lon = np.asarray(lon[0] if lon.ndim == 2 else lon, dtype=np.float32)
    lat = np.asarray(lat[:, 0] if lat.ndim == 2 else lat, dtype=np.float32)
    tv = first_var(ds, ['time', 'MT'])
    cal = getattr(tv, 'calendar', 'standard')
    dates = netCDF4.num2date(tv[:], tv.units, calendar=cal,
                             only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    times = [(d.year * 10000 + d.month * 100 + d.day, d.hour * 10000 + d.minute * 100 + d.second)
             for d in dates]
    t0 = ww3_time(a.start) if a.start else (0, 0)
    t1 = ww3_time(a.stop) if a.stop else (99999999, 0)
    keep = [i for i, t in enumerate(times) if t0 <= t <= t1]
    if not keep:
        raise SystemExit('no time between %s and %s' % (t0, t1))
    os.makedirs(a.out, exist_ok=True)
    lon.astype('<f4').tofile(os.path.join(a.out, 'lon.f32'))
    lat.astype('<f4').tofile(os.path.join(a.out, 'lat.f32'))
    meta = {'source': os.path.abspath(a.nc), 'nxi': int(lon.size), 'nyi': int(lat.size), 'vars': [],
            'times': [list(times[i]) for i in keep]}
    for n in names:
        v = ds.variables[n]
        if '_FillValue' not in v.ncattrs():
            raise SystemExit('%s has no _FillValue (ww3_prnc stops)' % n)
        # ww3_prnc keeps the _FillValue of the last variable
        meta['fill'] = f32_bits(v.getncattr('_FillValue'))
        scale = v.getncattr('scale_factor') if 'scale_factor' in v.ncattrs() else 1.0
        offset = v.getncattr('add_offset') if 'add_offset' in v.ncattrs() else 0.0
        meta['vars'].append({'name': n, 'scale': f32_bits(scale), 'offset': f32_bits(offset),
                             'dtype': str(v.dtype), 'dims': list(v.dimensions)})
        with open(os.path.join(a.out, n + '.f32'), 'wb') as f:
            for i in keep:
                x = v[i, 0] if v.ndim == 4 else v[i]
                if x.shape != (lat.size, lon.size):
                    raise SystemExit('%s: record shape %s, expected %s' % (n, x.shape, (lat.size, lon.size)))
                np.asarray(x).astype('<f4').tofile(f)
    with open(os.path.join(a.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('%s: %s, %d x %d, %d of %d times (%s to %s), fill %r' % (
        a.nc, '+'.join(names), lon.size, lat.size, len(keep), len(times),
        times[keep[0]], times[keep[-1]], np.uint32(meta['fill']).view(np.float32)))


if __name__ == '__main__':
    main()
