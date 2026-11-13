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
| `uniq.u64` | uint64 | HEALPix NESTED uniq index of each node, $4\cdot4^{d} + i_{\rm pix}$ |
| `level.u8` | uint8 | HEALPix depth $d$ of each node (nside $= 2^d$) |
| `lon.f64`, `lat.f64` | float64 | geodetic degrees |
| `depth.f64` | float64 | mean wet depth of the cell (m, positive down), at least `ww3.dmin` |
| `tri.u32` | uint32 | triangles, 3 × 0-based node indices, counter-clockwise |
| `obc.u32` | uint32 | 0-based open-boundary node indices |

`hww3-grid cells <output_dir>` adds `raw/cell_lon.f64` and
`raw/cell_lat.f64` (n × 4 cell vertices S, E, N, W) for plots.

Reading them in Python:

```python
import numpy as np
d = "grids/natl/raw/"
uniq = np.fromfile(d + "uniq.u64", "<u8")
level = np.fromfile(d + "level.u8", "u1")
tri = np.fromfile(d + "tri.u32", "<u4").reshape(-1, 3)
ipix = uniq - 4 * 4 ** level.astype(np.uint64)   # NESTED pixel at nside 2**level
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
are averaged. Every level is then a complete regular HEALPix map of the
domain, the repeated values compress well, and gridlook displays any level
without interpolation.

`scripts/hs_pyramid.py` writes it (`scripts/natl_pyramid.pbs` on Datarmor):

```bash
qsub -v RUN=natl_jan2026,GRID=grids/natl,OUT=runs/natl_jan2026/hs-l11-l7.zarr scripts/natl_pyramid.pbs
zvenv/bin/python scripts/upload_s3.py runs/natl_jan2026/hs-l11-l7.zarr \
  s3://grid4earth/public/trials/jmdelouis-2026-11-12-hww3/natl_jan2026/hs-l11-l7.zarr
```

The first command runs on a compute node (zarr 3 needs Python ≥ 3.11:
`/appli/anaconda/latest/bin/python -m venv zvenv && zvenv/bin/pip install
--only-binary=:all: zarr s3fs netCDF4 healpy`); the upload runs on a login
node, since the compute nodes have no internet access, with the `grid4earth`
AWS profile.

Layout (GRID4EARTH production layout, as in `convert_highway`):

- root group: `multiscales` (zarr-conventions v0.1, finest level first,
  `resampling_method: mean`), `Conventions: CF-1.13`, provenance;
- one group per depth (`11`, `10`, …, `7`) with the `dggs` attribute
  (HEALPix, nested, WGS84 ellipsoid), a CF `crs` grid mapping, `cell_ids`
  (uint64, ascending nested order), `time` (hours since 2000-01-01) and the
  fields (`hs`, …) as `(time, cells)` float32;
- `cells` covers every cell of the depth-3 pixels the domain touches
  (contiguous identifiers); cells without data hold NaN and chunks holding
  only NaN are not written;
- chunks of one time step and at most 16,384 cells, shards of 24 time steps
  and one depth-3 pixel, zstd level 5.

Cell identifiers are computed from authalic latitudes, because the generator
places HEALPix on the WGS84 ellipsoid; the script checks that every node is
its pixel centre.

January 2026, run `natl_jan2026` (Hs, hourly, depths 7–11): 4,991,755 data
cells at nside 2048, 20,991 at nside 128; 9 min on one core, 2.5 GB in
19,551 objects (1.6 times the raw node field, 1.5 GB), uploaded in 7.5 min.
View it in the gridlook preview (zoom-driven level choice, time slider):

<https://grid4earth.eu/gridlook-preview/#https://data.grid4earth.eu/trials/jmdelouis-2026-11-12-hww3/natl_jan2026/hs-l11-l7.zarr>

The run with currents and sea level is at the same path with
`natl_jan2026_cur` (PBS job 7898390, 561 s, 2.5 GB, uploaded in 472 s):

<https://grid4earth.eu/gridlook-preview/#https://data.grid4earth.eu/trials/jmdelouis-2026-11-12-hww3/natl_jan2026_cur/hs-l11-l7.zarr>

They are trials under `public/trials/` and can be removed at any time.
