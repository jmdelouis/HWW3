# Outputs

`hww3-grid <config.json>` writes to `output_dir`:

| File | Content |
|---|---|
| `<name>.msh` | Gmsh 2.2 ASCII mesh: nodes `(lon, lat, depth)` with depth positive down, triangles of type 2, 1-based |
| `<name>.obc` | open-boundary nodes in the `READMSHOBC` format of WW3 (see below) |
| `ww3_grid.nml` | WW3 grid namelist (`GRID%TYPE = 'UNST'`, `UNST%SF = -1`) |
| `namelists_<name>.nml` | numerical scheme: fully implicit PDLIB (`IMPTOTAL = T`, JGS solver) |
| `raw/*.{u64,u8,f64,u32}` | little-endian arrays, see below |
| `meta.json` | configuration, summary and description of the raw arrays |

and prints the summary on stdout (also stored in `meta.json`).

## Raw arrays

Nodes are in NESTED (Morton) order.

| File | Type | Content |
|---|---|---|
| `uniq.u64` | uint64 | HEALPix NESTED uniq index of each node as in cdshealpix, $16\cdot4^{d} + i_{\rm pix}$ |
| `level.u8` | uint8 | HEALPix depth $d$ of each node (nside $= 2^d$) |
| `lon.f64`, `lat.f64` | float64 | geodetic degrees |
| `depth.f64` | float64 | mean wet depth of the cell (m, positive down), at least `ww3.dmin` |
| `tri.u32` | uint32 | triangles, 3 × 0-based node indices, counter-clockwise |
| `obc.u32` | uint32 | 0-based open-boundary node indices |
| `slope.f64` | float64 | mean bathymetry slope of the cell (with `refine_bathy`) |
| `target_km.f64` | float64 | cell size required by the size functions, km (with `refine_bathy`) |

`hww3-grid cells <output_dir>` adds `raw/cell_lon.f64` and
`raw/cell_lat.f64` (n × 4 cell vertices S, E, N, W) for plots.

Reading them in Python:

```python
import numpy as np
d = "grids/natl/raw/"
uniq = np.fromfile(d + "uniq.u64", "<u8")
level = np.fromfile(d + "level.u8", "u1")
tri = np.fromfile(d + "tri.u32", "<u4").reshape(-1, 3)
ipix = uniq - 16 * 4 ** level.astype(np.uint64)  # NESTED pixel at nside 2**level (cdshealpix uniq)
```

## Open-boundary file

WW3 reads the list with `READMSHOBC` (`w3triamd.F90`): the first line
starts with the comment character, then comment lines, then lines
`index ntag [tags] node` (here `ntag = 0`, 1-based node), and a final
comment line. A bare list of node numbers is read silently as empty.

```text
$ Comments
$ Open boundary points written by hww3-grid (971 nodes)
1 0 <node 1>
2 0 <node 2>
...
971 0 <node 971>
$ End of file
```

## Summary (JSON)

Main keys of the summary: `n_nodes`, `n_open_boundary`, `nodes_per_nside`,
`n_leaves`, `n_wet_leaves`, `n_balance_splits`, `n_components`,
`n_pinch_tris_removed`, `quality_before` (angles of the raw
triangulation), `n_flips`, `n_trimmed`, `mesh` (the validation report, see
[Validation](validation.md)) and `timings_s`.

## Model output as a HEALPix pyramid

Because every node is a HEALPix pixel, the WW3 fields on the nodes are a
multi-resolution HEALPix map. They are written as a Zarr V3 pyramid, one
level per nside from `nside_min` to `nside_max`: at a level finer than a
node, the node value is copied to all its descendants (each coarse pixel
is split 2×2 per level); at a coarser level, the values of the fine nodes
are averaged (directions as unit vectors). Every level is then a complete
regular HEALPix map of the domain and the repeated values compress well.

`scripts/ww3_pyramid.py` writes every scalar output of `ww3_ounf` (`hs`,
`t02`, `fp`, `dir`, `wlv`, `ice`, `uwnd`, `vwnd`, `ucur`, `vcur`) plus the
speeds `wspd` and `cspd`; `scripts/grid_pyramid.py` writes the grid itself
(`level`, `nside`, `cell_size`, `depth`, `slope`, `target_size`, one time
step). `scripts/natl_pyramid.pbs` runs both on Datarmor:

```bash
qsub -v GRID=grids/natl_v3 scripts/natl_pyramid.pbs                    # grid only, 20 s
qsub -v RUN=natl_v2_jan2026_cur,GRID=grids/natl_v2 scripts/natl_pyramid.pbs
zvenv/bin/python scripts/upload_s3.py grids/natl_v3/grid-l11-l7.zarr \
  s3://grid4earth/public/trials/jmdelouis-2026-11-12-hww3/grid_natl_v3_fix/grid-l11-l7.zarr
```

The pyramids are built on a compute node (zarr 3 needs Python ≥ 3.11:
`/appli/anaconda/latest/bin/python -m venv zvenv && zvenv/bin/pip install
--only-binary=:all: zarr s3fs netCDF4 healpy`); the upload runs on a login
node, since the compute nodes have no internet access, with the `grid4earth`
AWS profile.

Layout: the zarr DGGS convention v1, as written by `xdggs`, and the zarr
multiscales convention v0.1:

- root group: `multiscales` (finest level first, `resampling_method: mean`,
  `max` for `level` and `nside`), `Conventions: CF-1.13`, provenance;
- one group per depth (`11`, `10`, …, `7`) with the `dggs` attribute
  (`name: healpix`, `refinement_level`, `indexing_scheme: nested`,
  `ellipsoid: {name: WGS84, semi_major_axis, inverse_flattening}`,
  `spatial_dimension: cells`, `coordinate: cell_ids`), `cell_ids` (uint64,
  ascending nested order), `time` (hours since 2000-01-01) and the fields
  as `(time, cells)` float32;
- `cells` covers every cell of the depth-3 pixels the domain touches
  (contiguous identifiers); cells without data hold NaN and chunks holding
  only NaN are not written;
- chunks of one time step and at most 16,384 cells, shards of 24 time steps
  and one depth-3 pixel, zstd level 5.

Cell identifiers are computed from authalic latitudes, because the generator
places HEALPix on the WGS84 ellipsoid; the script checks that every node is
its pixel centre. There is no CF `crs` variable: the first stores had one
with `healpix_nside`/`healpix_order` (easygems convention), which gridlook
reads before `dggs` and then draws on a sphere, so every cell appeared up to
0.13° (14 km) south of its place.

January 2026, run `natl_jan2026` (Hs only, hourly, depths 7–11): 4,991,755
data cells at nside 2048, 20,991 at nside 128; 9 min on one core, 2.5 GB.
With the 12 fields, 24 hours take 225 s and 941 MB. The full month of
run `natl_v2_jan2026_cur` (12 fields, 744 hours, PBS job 7905049) takes
7,112 s on one core and 29 GB, with 4,991,473 data cells at nside 2048.

Stores on S3 (trials, can be removed at any time):

- <https://data.grid4earth.eu/trials/jmdelouis-2026-11-12-hww3/grid_natl_v2_fix/grid-l11-l7.zarr>
- <https://data.grid4earth.eu/trials/jmdelouis-2026-11-12-hww3/grid_natl_v3_fix/grid-l11-l7.zarr>
- <https://data.grid4earth.eu/trials/jmdelouis-2026-11-12-hww3/natl_v2_jan2026_cur/ww3-l11-l7.zarr>
  (12 fields, January 2026, grid v2, `BETAMAX` 1.75)
- <https://data.grid4earth.eu/trials/jmdelouis-2026-11-12-hww3/natl_v3_jan2026_cur/ww3-l11-l7.zarr>
  (same on grid v3, PBS job 7905051: 7,337 s, 37 GB)

All four are drawn by the gridlook preview (draft pull request
d70-t/gridlook#235), on the WGS84 ellipsoid, for example
<https://grid4earth.eu/gridlook-preview/#https://data.grid4earth.eu/trials/jmdelouis-2026-11-12-hww3/natl_v3_jan2026_cur/ww3-l11-l7.zarr>.
The first grid stores (`grid_natl_v2`, `grid_natl_v3`) were drawn empty:
`grid_pyramid.py` decoded `raw/uniq.u64` with $4\cdot4^d$ instead of the
cdshealpix $16\cdot4^d$, so every cell identifier was $12\cdot4^d$ too large
(beyond the sphere); the script now checks that the decoded pixels are
valid. The first Hs stores (`natl_jan2026/hs-l11-l7.zarr`,
`natl_jan2026_cur/hs-l11-l7.zarr`) are displayed but shifted, as explained
above.
