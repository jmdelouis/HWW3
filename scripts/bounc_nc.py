"""Write the station spectrum files of ww3_bounc in the form that ``hww3-pre bounc`` reads.

The Rust port of ww3_bounc (hww3_wave::bounc) does the rest; this script
only reads the netCDF files, since the crate has no netCDF reader. For each
file of the list (``BOUND%FILE`` of ww3_bounc.nml, paths relative to the
list) it reads, as ww3_bounc does, the station longitude and latitude, the
frequencies and directions, and ``efth`` as stored (no unpacking) with its
``_FillValue``, ``scale_factor`` and ``add_offset`` and whether it is stored
as short integers, all in single precision; the times are those of the
first file.

    python scripts/bounc_nc.py spec.list OUTDIR

OUTDIR gets meta.json and efth.f32 (little endian,
[station][time][frequency][direction]).
"""
import argparse
import json
import os

import netCDF4
import numpy as np


def f32_bits(v):
    return int(np.asarray(v, dtype=np.float32).reshape(-1)[0].view(np.uint32))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('list')
    ap.add_argument('out')
    a = ap.parse_args()
    base = os.path.dirname(os.path.abspath(a.list))
    with open(a.list) as f:
        files = [os.path.join(base, l.strip()) for l in f if l.strip()]
    os.makedirs(a.out, exist_ok=True)
    meta = {'nst': len(files), 'lon': [], 'lat': [], 'coding': []}
    with open(os.path.join(a.out, 'efth.f32'), 'wb') as out:
        for k, fn in enumerate(files):
            ds = netCDF4.Dataset(fn)
            ds.set_auto_maskandscale(False)
            st = 'station_name' in ds.variables
            lon = ds.variables['longitude' if st else 'x'][:]
            lat = ds.variables['latitude' if st else 'y'][:]
            meta['lon'].append(f32_bits(np.asarray(lon).reshape(-1)[0]))
            meta['lat'].append(f32_bits(np.asarray(lat).reshape(-1)[0]))
            v = ds.variables['efth'] if 'efth' in ds.variables else ds.variables['Efth']
            att = v.ncattrs()
            if '_FillValue' not in att:
                raise SystemExit('%s: efth has no _FillValue' % fn)
            meta['coding'].append([f32_bits(v.getncattr('_FillValue')),
                                   f32_bits(v.getncattr('scale_factor') if 'scale_factor' in att else 1.0),
                                   f32_bits(v.getncattr('add_offset') if 'add_offset' in att else 0.0),
                                   v.dtype == np.int16])
            freq = np.asarray(ds.variables['frequency'][:], dtype=np.float32)
            dirs = np.asarray(ds.variables['direction'][:], dtype=np.float32)
            x = np.asarray(v[:, 0] if v.ndim == 4 else v[:])
            if k == 0:
                tv = ds.variables['time']
                dates = netCDF4.num2date(tv[:], tv.units, calendar=getattr(tv, 'calendar', 'standard'),
                                         only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                meta['times'] = [[d.year * 10000 + d.month * 100 + d.day,
                                  d.hour * 10000 + d.minute * 100 + int(round(d.second + d.microsecond / 1e6))]
                                 for d in dates]
                meta['nk'], meta['nth'] = int(freq.size), int(dirs.size)
                meta['freq'] = [f32_bits(q) for q in freq]
                meta['dir'] = [f32_bits(q) for q in dirs]
                shape = x.shape
            elif x.shape != shape:
                raise SystemExit('%s: efth shape %s, first file %s' % (fn, x.shape, shape))
            x.astype('<f4').tofile(out)
            ds.close()
    with open(os.path.join(a.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('%d stations, %d x %d spectra, %d times (%s to %s)' % (
        meta['nst'], meta['nk'], meta['nth'], len(meta['times']), meta['times'][0], meta['times'][-1]))


if __name__ == '__main__':
    main()
