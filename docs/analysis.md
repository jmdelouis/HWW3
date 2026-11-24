# Analysis tools

The analysis scripts are in `scripts/`. They need Python with `numpy`,
`scipy`, `netCDF4` and `matplotlib` (see [Installation](installation.md));
on the hpc cluster they run with `$R/pyenv/bin/python` (see
[Reproducing](reproducing.md), section 8). Each script describes its
arguments at its top; the full list is in [API and scripts](api.md).

Hands-on: [Validate a run](tutorials/validate_run.md).

## Scores against the altimeters

```text
python scripts/validate_altimetry.py RUN_NC NODES_NPZ OUT_PREFIX [L3_ROOT]
```

`RUN_NC` is the `ww3_ounf` output of a run, `NODES_NPZ` the
`grids/<name>/plot/nodes.npz` of its grid, `L3_ROOT` the directory of the
Copernicus Marine L3 altimeter files (default: the Datarmor mirror). Every
1 Hz sample takes the nearest wet, ice-free node if it lies within 0.75 of
that node's cell size, interpolated linearly in time; consecutive samples
are averaged in blocks of 7 (about 50 km). The first 3 days of a run are
skipped as spin-up (`VAL_SPINUP_DAYS` changes this), and `VAL_END` ends
the window. Writes `OUT_PREFIX_stats.json` (bias, RMSE, scatter index,
correlation: in total, per mission, per HEALPix level, per region and per
day), `OUT_PREFIX_scatter.png` and `OUT_PREFIX_maps.png`. On Datarmor,
`scripts/natl_validate.pbs` runs it for a run directory.

## Scores against the buoys

```text
python scripts/validate_buoys.py OUT_PREFIX LABEL:RUN_NC:GRID_DIR [...]
       [--buoys DIR] [--match nearest|depth --weights WEIGHTS.json]
```

$H_s$ and $T_{m02}$ of one or several runs against the moored buoys of
Copernicus Marine, averaged per hour. With several runs every run is
scored on the same buoys and hours, so the numbers can be compared; a buoy
is kept with a wet node within 10 km and at least 100 good hours
(`VAL_MIN_HOURS` lowers this for short runs). As for the altimeters, the
first 3 days are skipped (`VAL_SPINUP_DAYS`). `--match depth` uses the depth-aware operator of the calibration
instead of the nearest node. Writes `OUT_PREFIX_stats.json` (in total, per
buoy, per class of buoy) and the figures `OUT_PREFIX_scatter.png` and, with
two runs or more, `OUT_PREFIX_series.png` and `OUT_PREFIX_series_tm02.png`.
On Datarmor, `scripts/natl_buoys.pbs` runs it.

## GPU runs against WW3

| Script | What it does |
|---|---|
| `m1_compare_nc.py WW3.nc GPU_DIR` | the hourly $H_s$ and $T_{m02}$ of a GPU run against a WW3 run, node by node: RMS difference at the median node, at the 99th percentile and at the worst nodes (`--worst N --grid DIR`), and the total energy; no numpy needed |
| `gpu_to_nc.py WW3.nc GPU_DIR OUT.nc` | the GPU fields as a netCDF file shaped like the WW3 output, so that the scoring scripts read them |
| `m2_scores_hpc.slurm` | the altimeter and buoy scores of a WW3 run and, with `GPU=`, of a GPU run on the same observations (hpc cluster) |
| `pre_hs_cmp.py` | the hourly $H_s$ of two GPU runs, node by node |
| `m3_tl_compare.py` | a tangent run against the central differences of perturbed runs |

## Maps and figures

| Script | Figure |
|---|---|
| `plot_hs.py RUN_NC out.png` | a map of $H_s$ 12 h after the start of a run |
| `plot_run_diff.py A.nc "label A" B.nc "label B" grid/raw/tri.u32 out.png [spinup_days]` | two runs and their difference, on the same grid or on two grids (`DIFF_TRI_A`) |
| `plot_level_change.py` | the cells made coarser or finer from one grid to another, with the change of the buoy scores |
| `extract_zoom.py`, `plot_levels.py`, `plot_cells_tris.py`, `plot_mesh.py` | the HEALPix level of every node, and zooms on the cells and triangles of a grid |
| `explicit_cost.py`, `plot_explicit_cost.py` | the sub-steps of the explicit scheme per node, as a map and as curves |

## HEALPix pyramids for gridlook

`ww3_pyramid.py` writes the fields of a run, and `grid_pyramid.py` the
grid itself, as a multi-resolution HEALPix Zarr pyramid that gridlook
draws in a web browser without interpolation; `natl_pyramid.pbs` runs both
on Datarmor and `upload_s3.py` publishes a store. See
[Outputs](outputs.md), section *Model output as a HEALPix pyramid*.

## Diagnostics of the observations

Tools written for the calibration, to understand why some observations
resist the model:

| Script | What it does |
|---|---|
| `obs_diag.py` (`obs_diag_hpc.slurm`) | per-buoy tables and figures, the noise of the altimeter blocks, the bias by sea state and by wind, robust-loss steps against the least-squares step |
| `coast_diag.py`, `coast_subgrid.py` | the buoy scores by depth mismatch, node depth, distance to the coast, HEALPix level and land fraction of the cell |
| `buoy_op.py`, `buoy_depth.py` | the depth-aware buoy operator and the GEBCO depth at the buoys |
| `buoy_sigma.py`, `buoy_exclude.py` | the representativeness errors per class of buoy, and the candidates for the buoys the mesh cannot represent |
| `cal_point.py` (`cal_monitor_hpc.slurm`) | one point of the calibration under every cost, on both windows |

Their results are in [GPU model results](wave_results.md), sections
*Coastal diagnostics of the buoy scores*, *Third cost and the spin-up
rule* and *Observation diagnostics*.
