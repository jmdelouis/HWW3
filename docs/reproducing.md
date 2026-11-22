# Reproducing

This page lists every step, from the raw data to the January 2026 runs, as
they were run on Datarmor (Ifremer, PBS). `R` is the working directory on
DATAWORK and the repository is copied there:

```bash
R=/home1/datawork/$USER/HWW3
git clone git@github.com:jmdelouis/HWW3.git $R     # or rsync the repository
cd $R
mkdir -p logs grids runs forcing
```

All PBS scripts write their log to `$R/logs/<script>.log`. Timings and job
ids are those of the reference run.

## 0. Sources and data

```bash
bash scripts/fetch_sources.sh $R      # login node with internet access
```

downloads WW3 `develop` (pinned to `dea641c`), netCDF-Fortran 4.6.1,
ParMETIS 4.0.3 and GEBCO 2024 (`data/gebco/GEBCO_2024.nc`). The GSHHS
coastline is the shapefile `gshhs_land.shp` (full resolution, level 1),
path in `configs/natl_datarmor.json`.

## 1. Grid generator tests (any machine)

```bash
cd hww3-grid && cargo test --release
```

All tests must pass (see [Validation](validation.md)).

## 2. Bathymetry and grid

```bash
qsub scripts/natl_grid.pbs                       # grid v1: coast distance, no angle control
qsub -v CFG=configs/natl_v2_datarmor.json,GRID=grids/natl_v2 scripts/natl_grid.pbs   # v2: + angle control
qsub -v CFG=configs/natl_v3_datarmor.json,GRID=grids/natl_v3 scripts/natl_grid.pbs   # v3: depth and slope
qsub -v CFG=configs/natl_v4_datarmor.json,GRID=grids/natl_v4 scripts/natl_grid.pbs   # v4: + alternated diagonals, CFL flips
```

Grid v3 (PBS job 7899977, 70 s on 8 cores, 7.0 GB) is refined from the
bathymetry with the OceanMesh2D size functions (`wl` 100, `slp` 10 above
300 m, Rossby-radius filter, grade 0.2): 794,613 nodes (8,917, 15,050,
40,980, 119,503 and 610,163 at nside 128 to 2048), 1,341 open-boundary
nodes, angles 20.0°–125.9°. Other parameter sets are compared with

```bash
qsub -v SCAN="128:10:0:0.2 100:10:0:0.2 100:10:-1:0.2 100:20:20:0.2" scripts/natl_grid_scan.pbs
```

(`wl:slp:fl:grade`; 1,104,782, 1,091,206, 794,613 and 967,167 nodes,
job 7899758). The grid pyramid for gridlook (level, depth, slope, required
size) is written by `qsub -v GRID=grids/natl_v3 scripts/natl_pyramid.pbs`
(see [Outputs](outputs.md)).

The script cuts GEBCO to the domain (`scripts/prep_gebco.py`, 30 s, only
the first time), builds `hww3-grid` with Cargo (Rust under
`$HOME/stl-mpi/{rustup,cargo}`), builds the grid (8 threads, 152 s, peak
5.8 GB for the first build), writes the cell vertices and the plotting
subsets (`scripts/natl_extract.sh`). Outputs: `grids/natl/`,
`grids/natl_summary.json`, `grids/natl/plot/*.npz`.

Figures (any machine with matplotlib, after copying `grids/natl/plot`):

```bash
python scripts/plot_levels.py grids/natl/plot/nodes.npz tex/hww3/fig/natl_levels.png
for z in brittany iroise gibraltar openocean; do
  python scripts/plot_cells_tris.py grids/natl/plot/$z.npz tex/hww3/fig/natl_$z.png
done
```

## 3. WAVEWATCH III build

```bash
qsub scripts/build_ww3.pbs        # 5.5 min, 8 cores
```

On the Ifremer hpc cluster, from `/scratch/jmdeloui/HWW3` with the WW3
source cloned at `dea641c` and `parmetis-4.0.3.tar.gz` in `src/`:

```bash
sbatch scripts/build_ww3_hpc.slurm   # 2.3 min, 8 cores
```

See [WW3 setup](ww3.md) for the toolchain and switches.

## 4. WW3 grid preprocessing

```bash
qsub scripts/natl_ww3_grid.pbs    # 35 s: runs/natl/mod_def.ww3 (260 MB)
```

`ww3_grid` must report 502,690 sea points and 971 active boundary points.

## 5. Homogeneous-wind test

```bash
qsub scripts/natl_test.pbs        # 2 nodes, 56 ranks, 22 min
python scripts/plot_hs.py runs/natl_test/ww3.202601.nc tex/hww3/fig/natl_test_hs.png
```

12 h from rest with a 15 m/s westerly wind (`ww3/natl_test/ww3_shel.nml`).
Expected: 1337 s for 72 steps, median $H_s$ 2.75 m after 12 h.

## 6. Forcing

ERA5 (needs `~/.cdsapirc`; 7 min):

```bash
$R/venv/bin/python scripts/get_era5.py 2026 1 forcing/era5
qsub scripts/natl_jan2026_prep.pbs        # wind.ww3 (3.0 GB, 72 s), ice.ww3
```

Copernicus Marine currents and sea level (Datarmor mirror, no account):

```bash
qsub scripts/natl_jan2026_cur_prep.pbs    # 14 min: current.ww3, level.ww3
```

## 7. January 2026 runs

```bash
qsub scripts/natl_jan2026_run.pbs         # wind + ice, 4 nodes, 112 ranks
qsub scripts/natl_jan2026_cur_run.pbs     # wind + ice + currents + sea level
```

Each run covers 1 January 00 UTC to 31 January 23 UTC (the last ERA5
hour), writes hourly fields (`HS T02 FP DIR WND ICE`, plus `CUR WLV`) and
weekly restarts, then `ww3_ounf` converts them to `ww3.202601.nc`. Speed:
about 10.2 s per 600 s step on 112 ranks, i.e. about 13 h for the month.

The reference January runs (PBS jobs 7894000 and 7894069) use grid v1
(`grids/natl`, built without the triangle-quality step). All later runs use
grid v2 (`grids/natl_v2`, step 3 with `configs/natl_v2_datarmor.json`); with
v2, rerun `ww3_grid` and the `ww3_prnc` steps above in the v2 run directory,
since the forcing files are interpolated on the nodes of one grid only.

### Grid v2 run with BETAMAX for ERA5

```bash
J=$(qsub -v BETAMAX=1.75,RUN=natl_v2_jan2026_cur scripts/natl_v2_setup.pbs)   # 4 min
qsub -W depend=afterok:$J -v RUN=natl_v2_jan2026_cur scripts/natl_v2_run.pbs
```

`natl_v2_setup.pbs` runs `ww3_grid` on `grids/natl_v2` with
`&SIN4 BETAMAX = 1.75 /` appended to the generated namelists (the v1 runs
used the default 1.43; check `betamax : 1.750` in `ww3_grid.out`), then
`ww3_prnc` of wind, ice, currents and sea level on the v2 nodes from the
netCDF files of the v1 runs (61, 1, 93 and 48 s). The run uses the
currents-and-sea-level `ww3_shel.nml`.

The same scripts run on grid v3 with `GRID`:

```bash
J=$(qsub -v BETAMAX=1.75,RUN=natl_v3_jan2026_cur,GRID=grids/natl_v3 -o logs/natl_v3_setup.log scripts/natl_v2_setup.pbs)
qsub -W depend=afterok:$J -v RUN=natl_v3_jan2026_cur -o logs/natl_v3_run.log scripts/natl_v2_run.pbs
```

(PBS jobs 7901052 and 7901053; `ww3_grid` 44 s, 794,613 sea points,
`ww3_prnc` 80, 1, 133 and 58 s). With the same forcing and `BETAMAX` as
the v2 run, the difference between the two runs is the grid alone.

### Grid v3 run with open-boundary spectra

```bash
J=$(qsub scripts/natl_bnd_setup.pbs)        # 2 min: spectra from GLOBMULTI GLOB-30M, ww3_bounc
qsub -W depend=afterok:$J -v RUN=natl_v3_jan2026_bnd -o logs/natl_v3_bnd_run.log scripts/natl_v2_run.pbs
```

`natl_bnd_setup.pbs` links the grid and forcing of `runs/natl_v3_jan2026_cur`
into `runs/natl_v3_jan2026_bnd`, rebuilds one spectrum file per global cell
next to the open boundary in `spec/` (`scripts/boundary_spectra.py`) and runs
`ww3_bounc`, which writes `nest.ww3`; `ww3_shel` reads it at start (column
`b` of `log.ww3`).

The same run on grid v4 (PBS jobs 7935602–7935604, 19.5 h):

```bash
J1=$(qsub -v BETAMAX=1.75,RUN=natl_v4_jan2026_cur,GRID=grids/natl_v4 -o logs/natl_v4_setup.log scripts/natl_v2_setup.pbs)
J2=$(qsub -W depend=afterok:$J1 -v SRC=natl_v4_jan2026_cur,RUN=natl_v4_jan2026_bnd,GRID=grids/natl_v4 -o logs/natl_v4_bnd_setup.log scripts/natl_bnd_setup.pbs)
qsub -W depend=afterok:$J2 -v RUN=natl_v4_jan2026_bnd -o logs/natl_v4_bnd_run.log scripts/natl_v2_run.pbs
```

### FES2014 tide

The hourly Copernicus Marine fields have no tide. `scripts/fes_tides.py`
adds the FES2014 tide (Datarmor copy in
`/home/datawork-WW3/FORCING/FES2014/SRC_GLOBAL_RECT`) to `current.nc` and
`level.nc` of `prep_cmems.py`; it needs pyTMD 3.1, installed once without
its dependencies:

```bash
$R/venv/bin/pip install --target $R/pylib --no-deps pyTMD timescale
J=$(qsub -v R=$R scripts/natl_tide_setup.pbs)   # 21 min: fes_tides.py, ww3_prnc current and level
qsub -W depend=afterok:$J -v R=$R,RUN=natl_v4_jan2026_tide -o $R/logs/natl_v4_tide_run.log scripts/natl_v2_run.pbs
```

`natl_tide_setup.pbs` links the grid, wind, ice and `nest.ww3` of
`runs/natl_v4_jan2026_bnd` into `runs/natl_v4_jan2026_tide` (PBS jobs
7958070 and 7958071, 19 h). Validation (PBS jobs 7962685 and 7963439):

```bash
V=runs/natl_v4_jan2026_tide/val
python scripts/validate_altimetry.py runs/natl_v4_jan2026_tide/ww3.202601.nc grids/natl_v4/plot/nodes.npz $V/natl_v4_jan2026_tide
python scripts/validate_buoys.py $V/buoys v4bnd:runs/natl_v4_jan2026_bnd/ww3.202601.nc:grids/natl_v4   v4tide:runs/natl_v4_jan2026_tide/ww3.202601.nc:grids/natl_v4
python scripts/plot_run_diff.py runs/natl_v4_jan2026_bnd/ww3.202601.nc "v4 without tide"   runs/natl_v4_jan2026_tide/ww3.202601.nc "v4 with FES2014 tide" grids/natl_v4/raw/tri.u32 $V/natl_v4_tide_minus_bnd_diff.png 3
```

Expected: altimeter RMSE 0.348 m and bias +0.041 m as without tide; buoy
$H_s$ RMSE 0.373 → 0.372 m and $T_{m02}$ RMSE 1.316 → 1.263 s.

### Explicit scheme

The same run with the explicit N scheme (PBS jobs 7951887 and 7951889,
about 4,300 s per model day instead of 2,100 s). Every PBS script takes
`R` (default `/home1/datawork/jmdeloui/HWW3`) to run from another file
system, for example `qsub -v R=/home1/scratch/jmdeloui/HWW3,...`.

```bash
for v in 3 4; do
  J=$(qsub -v SRC=natl_v${v}_jan2026_bnd,CUR=natl_v${v}_jan2026_cur,RUN=natl_v${v}_jan2026_exp,GRID=grids/natl_v$v -o logs/natl_v${v}_exp_setup.log scripts/natl_exp_setup.pbs)
  qsub -W depend=afterok:$J -v RUN=natl_v${v}_jan2026_exp -o logs/natl_v${v}_exp_run.log scripts/natl_v2_run.pbs
done
```

Scores on the common window (altimeters with `VAL_END=2026-01-07T21:00`,
buoys with `VAL_MIN_HOURS=48`), the explicit cost per node and the figures:

```bash
VAL_END=2026-01-07T21:00 python scripts/validate_altimetry.py runs/natl_v4_jan2026_exp/ww3.202601.nc grids/natl_v4/plot/nodes.npz runs/natl_v4_jan2026_exp/val/natl_v4_jan2026_exp
# implicit runs on the window of the v3 explicit run (runs/natl_v{3,4}_jan2026_bnd_w7 link ww3.202601.nc of the _bnd runs)
VAL_END=2026-01-07T20:00 python scripts/validate_altimetry.py runs/natl_v4_jan2026_bnd_w7/ww3.202601.nc grids/natl_v4/plot/nodes.npz natl_v4_jan2026_bnd_w7
VAL_MIN_HOURS=48 python scripts/validate_buoys.py runs/natl_v4_jan2026_exp/val/buoys \
  v3imp:runs/natl_v3_jan2026_bnd/ww3.202601.nc:grids/natl_v3 v3exp:runs/natl_v3_jan2026_exp/ww3.202601.nc:grids/natl_v3 \
  v4imp:runs/natl_v4_jan2026_bnd/ww3.202601.nc:grids/natl_v4 v4exp:runs/natl_v4_jan2026_exp/ww3.202601.nc:grids/natl_v4
python scripts/explicit_cost.py grids/natl_v4 results/natl_v4 "grid v4" runs/natl_jan2026_cur/current.nc   # 5 min, 40 GB
python scripts/plot_explicit_cost.py map grids/natl_v4 results/natl_v4 natl_v4_explicit_cost_map.png \
  -97 -80 70 79  4.4 7.6 59 63  -7.2 -5.2 56.4 57.8  10 22 76 80.5
python scripts/plot_explicit_cost.py curves natl_explicit_cost_curves.png results/natl_v3 results/natl_v4
python scripts/plot_run_diff.py runs/natl_v4_jan2026_bnd/ww3.202601.nc "v4 implicit" \
  runs/natl_v4_jan2026_exp/ww3.202601.nc "v4 explicit" grids/natl_v4/raw/tri.u32 natl_v4_jan2026_exp_diff.png 3
```

### Grid v5 (explicit sub-steps capped)

Grids with at most 5 and 4 explicit sub-steps (about 20 min each for 10
builds), their explicit cost, and 7-day runs with both schemes:

```bash
S=/home1/scratch/jmdeloui/HWW3
for k in 5 4; do
  G=grids/natl_v5k$k
  J=$(qsub -v R=$S,CFG=configs/natl_v5k${k}_datarmor.json,GRID=$G -o $S/logs/grid_v5k$k.log scripts/natl_grid.pbs)
  qsub -W depend=afterok:$J -v R=$S,GRID=$G,NAME=natl_v5k$k -o $S/logs/xcost_v5k$k.log scripts/natl_explicit_cost.pbs
  J1=$(qsub -W depend=afterok:$J -v R=$S,BETAMAX=1.75,RUN=natl_v5k${k}_jan2026_cur,GRID=$G scripts/natl_v2_setup.pbs)
  J2=$(qsub -W depend=afterok:$J1 -v "R=$S,SRC=natl_v5k${k}_jan2026_cur,RUN=natl_v5k${k}_jan2026_bnd,GRID=$G,STOP=20260108 000000" scripts/natl_bnd_setup.pbs)
  J3=$(qsub -W depend=afterok:$J2 -v "R=$S,SRC=natl_v5k${k}_jan2026_bnd,CUR=natl_v5k${k}_jan2026_cur,RUN=natl_v5k${k}_jan2026_exp,GRID=$G,STOP=20260108 000000" scripts/natl_exp_setup.pbs)
  qsub -W depend=afterok:$J3 -v R=$S,RUN=natl_v5k${k}_jan2026_exp scripts/natl_v2_run.pbs
  qsub -W depend=afterok:$J2 -v R=$S,RUN=natl_v5k${k}_jan2026_bnd scripts/natl_v2_run.pbs
done
```

Scores on the window of the grid-v4 explicit run, the cells made coarser
and the differences with grid v4 (run v4 interpolated on the v5 nodes):

```bash
for k in 5 4; do for s in bnd exp; do
  VAL_END=2026-01-07T21:00 python scripts/validate_altimetry.py runs/natl_v5k${k}_jan2026_$s/ww3.202601.nc \
    grids/natl_v5k$k/plot/nodes.npz runs/natl_v5k${k}_jan2026_$s/val/natl_v5k${k}_jan2026_$s
done; done
VAL_END=2026-01-07T21:00 python scripts/validate_altimetry.py runs/natl_v4_jan2026_bnd_w7/ww3.202601.nc \n  grids/natl_v4/plot/nodes.npz natl_v4_jan2026_bnd_to21h   # grid v4 on the same window
VAL_MIN_HOURS=48 python scripts/validate_buoys.py runs/natl_v5k4_jan2026_exp/val/buoys \
  v4imp:runs/natl_v4_jan2026_bnd/ww3.202601.nc:grids/natl_v4 \
  v5k5imp:runs/natl_v5k5_jan2026_bnd/ww3.202601.nc:grids/natl_v5k5 v5k5exp:runs/natl_v5k5_jan2026_exp/ww3.202601.nc:grids/natl_v5k5 \
  v5k4imp:runs/natl_v5k4_jan2026_bnd/ww3.202601.nc:grids/natl_v5k4 v5k4exp:runs/natl_v5k4_jan2026_exp/ww3.202601.nc:grids/natl_v5k4 \
  v4exp:runs/natl_v4_jan2026_exp/ww3.202601.nc:grids/natl_v4      # the last run sets the hours
python scripts/plot_level_change.py natl_v5_level_change.png grids/natl_v4 "grid v4" grids/natl_v5k5 "v5 K=5" \
  grids/natl_v5k4 "v5 K=4" --buoys runs/natl_v5k4_jan2026_exp/val/buoys_stats.json v4imp v5k5imp v5k4imp --box -20 15 27 62
DIFF_TRI_A=grids/natl_v4/raw/tri.u32 python scripts/plot_run_diff.py runs/natl_v4_jan2026_bnd/ww3.202601.nc "v4 implicit" \
  runs/natl_v5k4_jan2026_bnd/ww3.202601.nc "v5 K=4 implicit" grids/natl_v5k4/raw/tri.u32 natl_v5k4_jan2026_diff.png 3
```

### Altimeter validation

```bash
qsub -v RUN=natl_jan2026,GRID=grids/natl scripts/natl_validate.pbs   # 2 min
qsub -v RUN=natl_jan2026,GRID=grids/natl scripts/natl_pyramid.pbs    # grid and run pyramids
```

Writes `runs/natl_jan2026/val/natl_jan2026_{stats.json,scatter.png,maps.png}`.
Expected for the wind-and-ice run: 331,154 blocks, bias −0.19 m,
RMSE 0.45 m, SI 0.14, r 0.972 (see Validation). Needs `scipy` and
`matplotlib` in the venv (`pip install --only-binary=:all: scipy matplotlib`).

### Buoy validation

```bash
qsub -o logs/natl_buoys.log -j oe scripts/natl_buoys.pbs   # 5 min, needs the v2 and v3 runs
```

Writes `runs/natl_v3_jan2026_cur/val/buoys_{stats.json,scatter.png,series.png}`
from the Copernicus Marine moored buoys of January 2026
(`/home/ref-copernicus-insitu/.../monthly/MO/202601`). Expected: 267 buoys,
166,626 hours, $H_s$ bias −0.015 / −0.004 m and RMSE 0.368 / 0.375 m for
v2 / v3 (see Validation).

## 8. Note

```bash
cd tex/hww3 && latexmk -pdf main.tex
```

`tex/hww3/main.tex` gathers the grid statistics, the figures, the WW3
build, the run costs and the results; it is updated after every run.
It is mirrored on Overleaf (project `6af6a96c4ed66a0cad905cb6`) by
`scripts/sync_overleaf.sh`, which needs an Overleaf git token in the git
credential store.
