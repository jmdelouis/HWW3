# Configuration

A grid is described by one JSON file, passed to the program:

```bash
hww3-grid configs/natl_datarmor.json > grids/natl_summary.json
```

Relative paths are resolved against the directory of the configuration
file. The program prints a JSON summary on stdout and fails if the mesh is
invalid or, when `quality.enforce` is true, if a triangle angle is outside
the accepted range.

## Example: North Atlantic

```json
{
  "name": "natl",
  "domain": { "lon_min": -100.0, "lon_max": 45.0, "lat_min": 0.0, "lat_max": 80.0 },
  "ellipsoid": "wgs84",
  "nside_min": 128,
  "nside_max": 2048,
  "max_level_span": 4,
  "refine": [
    { "max_dist_km": 10.0,  "nside": 2048 },
    { "max_dist_km": 50.0,  "nside": 1024 },
    { "max_dist_km": 200.0, "nside": 512 }
  ],
  "min_depth_m": 5.0,
  "wet_fraction": 0.5,
  "keep_largest_component": true,
  "bathy": "../data/gebco/natl/meta.json",
  "coast": { "source": "shapefile", "path": "/home1/datahome/marc/data/GSHHS/gshhs_land.shp" },
  "output_dir": "../grids/natl",
  "quality": { "min_angle_deg": 20.0, "max_angle_deg": 130.0, "flips": true,
               "max_level_jump": 1, "trim_boundary": true, "enforce": true },
  "ww3": { "nth": 36, "nk": 32, "freq1": 0.0373, "xfr": 1.1, "dir_offset": 0.0,
           "dtmax": 600.0, "dtxy": 600.0, "dtkth": 300.0, "dtmin": 10.0,
           "zlim": -0.1, "dmin": 2.0 }
}
```

## Keys

| Key | Default | Meaning |
|---|---|---|
| `name` | required | grid name; output files `<name>.msh`, `<name>.obc`, `namelists_<name>.nml` |
| `domain` | required | box `lon_min, lon_max, lat_min, lat_max` (geodetic degrees, lon in (-180, 180]). Cells whose centre is outside are not wet; wet cells touching them are open-boundary nodes |
| `ellipsoid` | `"wgs84"` | `"wgs84"`: latitudes converted to authalic latitudes for HEALPix; `"sphere"`: no conversion |
| `nside_min` | required | coarsest nside (power of two) |
| `nside_max` | required | finest nside (power of two) |
| `max_level_span` | 4 | maximum number of levels between `nside_min` and `nside_max` |
| `refine` | required | list of rules `{max_dist_km, nside}`: a cell closer than `max_dist_km` to the coast gets at least `nside` |
| `min_depth_m` | 5 | cells whose mean wet depth is below this are dry |
| `wet_fraction` | 0.5 | minimum fraction of wet bathymetry samples for a wet cell |
| `keep_largest_component` | true | keep only the largest connected wet region |
| `bathy` | required | `meta.json` of the bathymetry (from `scripts/prep_gebco.py`) |
| `coast` | required | `{"source": "shapefile", "path": ...}`, `{"source": "gshhg", "path": ..., "levels": [1]}` or `{"source": "bathy"}` |
| `output_dir` | required | output directory |
| `quality` | see below | triangle-quality control |
| `ww3` | see below | parameters of `ww3_grid.nml` |

### `quality`

| Key | Default | Meaning |
|---|---|---|
| `min_angle_deg` | 20 | smallest accepted triangle angle |
| `max_angle_deg` | 130 | largest accepted triangle angle |
| `flips` | true | Delaunay edge flips (max-min angle) inside the mesh |
| `max_level_jump` | 1 | largest level difference allowed across a flipped edge |
| `trim_boundary` | true | remove boundary triangles outside the angle range |
| `enforce` | true | fail if an angle is still outside the range |

### `ww3`

| Key | Default | WW3 namelist |
|---|---|---|
| `nth` | 36 | `SPECTRUM%NTH` (directions) |
| `nk` | 32 | `SPECTRUM%NK` (frequencies) |
| `freq1` | 0.0373 | `SPECTRUM%FREQ1` (Hz) |
| `xfr` | 1.1 | `SPECTRUM%XFR` |
| `dir_offset` | 0 | `SPECTRUM%THOFF` |
| `dtmax`, `dtxy`, `dtkth`, `dtmin` | 600, 600, 300, 10 | `TIMESTEPS%*` (s) |
| `zlim` | -0.1 | `GRID%ZLIM` (m) |
| `dmin` | 2 | `GRID%DMIN` (m), also the floor of the depths written to the mesh |

## Bathymetry input

`scripts/prep_gebco.py` cuts GEBCO 2024 to the domain plus a margin and
writes `elev.f32` (little-endian float32, elevation in m, positive up,
row-major with longitude fastest and latitude increasing) and `meta.json`:

```text
{"nlon": <int>, "nlat": <int>, "lon0": <deg>, "lat0": <deg>,
 "dlon": 0.004166666666662877, "dlat": 0.004166666666662877,
 "file": "elev.f32", "source": "GEBCO_2024.nc", "stride": 1}
```

`lon0`, `lat0` are the centre of the first sample; the GEBCO step is 15
arc-seconds.

```bash
python scripts/prep_gebco.py data/gebco/GEBCO_2024.nc data/gebco/natl -100 45 0 80 --margin 3
```
