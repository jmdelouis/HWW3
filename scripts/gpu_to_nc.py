"""Write the hourly H_s and T_m02 of the GPU model (hww3-prop) as a netCDF
file shaped like the ww3_ounf output of the same grid and period, so that
validate_altimetry.py and validate_buoys.py read both runs alike.

The node coordinates, the triangles, the time axis and the ice
concentration are copied from the WW3 file; the GPU fields are the
little-endian float32 files hs_YYYYMMDDHH.f32 and t02_YYYYMMDDHH.f32
(-999.9 where undefined). Hours without a GPU file and undefined values are
masked, as WW3 masks land and ice.

usage:
    python scripts/gpu_to_nc.py WW3.nc GPU_DIR OUT.nc
"""
import os
import sys

import netCDF4
import numpy as np

COPY = ("longitude", "latitude", "tri", "time", "ice")


def main():
    ww3, gdir, out = sys.argv[1:4]
    src = netCDF4.Dataset(ww3)
    dst = netCDF4.Dataset(out, "w", format="NETCDF4")
    for name, dim in src.dimensions.items():
        dst.createDimension(name, None if dim.isunlimited() else len(dim))
    dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
    dst.setncattr("source_of_hs_t02", "hww3-prop (GPU model), files of " + os.path.abspath(gdir))
    for v in COPY:
        if v not in src.variables:
            continue
        s = src[v]
        fill = getattr(s, "_FillValue", None)
        o = dst.createVariable(v, s.dtype, s.dimensions, zlib=True, fill_value=fill)
        o.setncatts({k: s.getncattr(k) for k in s.ncattrs() if k != "_FillValue"})
        o[:] = s[:]
    t = src["time"]
    tags = [d.strftime("%Y%m%d%H") for d in netCDF4.num2date(t[:], t.units, only_use_cftime_datetimes=False)]
    nn = len(src.dimensions["node"])
    for var in ("hs", "t02"):
        s = src[var]
        fill = getattr(s, "_FillValue", np.float32(9.96921e36))
        o = dst.createVariable(var, "f4", s.dimensions, zlib=True, fill_value=fill)
        o.setncatts({k: s.getncattr(k) for k in s.ncattrs() if k not in ("_FillValue", "scale_factor", "add_offset")})
        found = 0
        for it, tag in enumerate(tags):
            p = os.path.join(gdir, "%s_%s.f32" % (var, tag))
            if os.path.exists(p):
                g = np.fromfile(p, "<f4")
                if len(g) != nn:
                    sys.exit("%s: %d values for %d nodes" % (p, len(g), nn))
                o[it, :] = np.ma.masked_less(g, -900.0)
                found += 1
            else:
                o[it, :] = np.ma.masked_all(nn, "f4")
        print("%s: %d of %d hours from %s" % (var, found, len(tags), gdir))
    dst.close()
    src.close()


if __name__ == "__main__":
    main()
