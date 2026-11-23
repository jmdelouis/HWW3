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

The sub-cycling counted as WW3 does (every wet node, `UGBCCFL = T`;
`--interior-only` gives the counts of `results/natl_v{3,4,5k5,5k4}_explicit_cost.json`),
without current, 3 to 6 min per grid on a workstation:

```bash
for g in v3 v4 v5k5 v5k4; do
  python scripts/explicit_cost.py grids/natl_$g results/natl_${g}_explicit_cost_bccfl "grid $g"
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

## 8. GPU wave model

The GPU model runs on the A100s of the Ifremer hpc cluster (Slurm), where
WW3 is also built (section 3). There `R=/scratch/jmdeloui/HWW3`, holding
the repository scripts, the grids and the WW3 build. See
[GPU wave model](wave_model.md) for the programs and
[GPU model results](wave_results.md) for the expected numbers.

```bash
cd hww3-wave && cargo test --no-default-features     # unit tests, any machine, under a minute
module load cuda/13.0.0
CARGO_TARGET_DIR=$R/target_wave cargo build --release
```

### Milestone M0: sweep times and dot-product tests

```bash
sbatch scripts/wave_m0_hpc.slurm     # one A100, 30 min at most: box, v4, v5 K=4
```

Writes `$R/results/wave_m0_{box_v4,v4,v5k4}_<job>.json`. Expected:
0.93 ms per frequency sweep on grid v4 with `two_phase_x4`, dot-product
tests to round-off. On Datarmor, the CUDA C backend is built with
`CUDARC_CUDA_VERSION=11040 cargo build --release --no-default-features --features cuda-c`
and run by `qsub scripts/wave_m0.pbs`.

### Milestone M1: propagation-only runs against WW3

The inputs are those of the grid v5 $K=4$ explicit run of Datarmor:
`restart001.ww3` (8 January 2026 00 h), `current.ww3`, `level.ww3`,
`ice.ww3` and `nest.ww3`, copied to `$R/m1/v5k4`, and the `ww3_grid`
inputs of the hpc check of section 3 in `$R/runs/test_grid_v5k4`.

```bash
# WW3 references, 32 cores, 12-15 min each; ideal first, it writes the ww3_strt restart
for m in ideal prop geo; do sbatch --export=ALL,MODE=$m scripts/m1_ww3_prop_hpc.slurm; done
# GPU model, one A100, 58-66 s each
for m in ideal prop geo; do sbatch --export=ALL,MODE=$m scripts/m1_prop_hpc.slurm; done
# comparison (no numpy needed), with the GPU output directory of each job
python3 scripts/m1_compare_nc.py $R/runs/m1_prop/ww3.202601.nc $R/m1/out_<job> --out m1_prop.json
python3 scripts/m1_compare_nc.py $R/runs/m1_ideal/ww3.202601.nc $R/m1/out_<job> --hs-min 0.5
python3 scripts/m1_compare_nc.py $R/runs/m1_prop/ww3.202601.nc $R/m1/out_<job> --worst 20 --grid $R/grids/natl_v5k4
```

`CHECK=1` adds the `check-ktp3` and `check-geo` comparisons with the CPU
transcriptions, `NORUN=1` skips the run, `HOURS` sets its length and
`EXTRA='--profile'` times each part of a step:
`sbatch --export=ALL,MODE=prop,HOURS=6,CHECK=1 scripts/m1_prop_hpc.slurm`.

Step-by-step comparison (the diagnosis of the coast nodes): WW3 for one
hour with a restart after each of the first three steps, then
`hww3-prop check-steps` against those restarts (`STEPS_REF` separated by
`:`, `NODES` by `+`, since `sbatch --export` splits at commas):

```bash
sbatch --export=ALL,MODE=prop,HOURS=1,FSTRIDE=600,RSTRIDE=600,RCOUNT=3,TAG=_diag scripts/m1_ww3_prop_hpc.slurm
D=$R/runs/m1_prop_diag
sbatch --export=ALL,MODE=prop,STEPS_REF=$D/restart001.ww3:$D/restart002.ww3:$D/restart003.ww3,NODES=169562+165663 scripts/m1_prop_hpc.slurm
# WW3 depth and currents every 600 s, to compare with check-steps --dump (DUMP=dir)
sbatch --export=ALL,MODE=geo,HOURS=1,FSTRIDE=600,TAG=_fld,"FLIST=HS DPT CUR" scripts/m1_ww3_prop_hpc.slurm
```

### Milestone M2a: source terms against WW3

The forcing of 1–10 January (wind, currents, levels, ice from
`natl_v5k4_jan2026_cur`, boundary spectra from `natl_v5k4_jan2026_bnd`) is
cut from the Datarmor runs with `ww3_subset.py` (two fields per record for
the wind and the currents) and copied to `$R/m2/forcing`. Then, on the hpc
cluster:

```bash
# WW3 with the simplified ST4 (SINTABLE = 0, SDSCUM = 0): 48 h from 8 January with
# restarts after the first three steps, and the cold-started week of 1-8 January
sbatch --export=ALL,MODE=dev,RSTRIDE=600,RCOUNT=3 scripts/m2_ww3_ref_hpc.slurm
sbatch --ntasks=64 --mem=160G -t 24:00:00 --export=ALL,MODE=week scripts/m2_ww3_ref_hpc.slurm
# the explicit run with sources, 12 h from 8 January, with UGBCCFL = T then F on the same node
sbatch scripts/m2_ww3_bccfl_hpc.slurm
# source-term harness on the restart of 8 January (in runs/m2_dev, after ww3_grid)
sbatch --export=ALL,OUT=src_8jan_c.bin,NODES=169562+130464+165663+229656+147650 scripts/m2_src_harness_hpc.slurm
# CPU transcriptions against the harness
cargo build --release --no-default-features --bin hww3-src
target/release/hww3-src check $R/m2/harness/src_8jan_c.bin
```

The GPU source kernel and the model with the source terms (the harness
file `src_8jan_c.bin.params` of job 4138986 also holds `SED_D50` and
`SED_PSIC` per sea point):

```bash
module load cuda/13.0.0
CARGO_TARGET_DIR=$R/target_wave cargo build --release --bin hww3-src --bin hww3-prop
# the kernel on the harness nodes, f32 and f64, and its timing on 200 copies of them
sbatch scripts/m2_src_gpu_hpc.slurm
# the first three steps against the WW3 restarts of the dev reference
D=$R/runs/m2_dev
sbatch --export=ALL,STEPS_REF=$D/restart001.ww3:$D/restart002.ww3:$D/restart003.ww3 scripts/m2_prop_hpc.slurm
# 48 h, then the comparison with the hourly fields of the dev reference
sbatch --export=ALL,HOURS=48,EXTRA=--wind-out scripts/m2_prop_hpc.slurm
python3 scripts/m1_compare_nc.py $D/ww3.202601.nc $R/m2/out_JOBID --worst 20 --grid $R/grids/natl_v5k4
python3 scripts/m1_compare_nc.py $D/ww3.202601.nc $R/m2/out_JOBID --vars uwnd,vwnd
```

The week on the GPU starts from the restart that a 1 h cold-started WW3
run writes after its first step, since the GPU model has no `W3UINI`:

```bash
sbatch -t 00:30:00 --export=ALL,MODE=week,HOURS=1,RSTRIDE=600,RCOUNT=1,TAG=_start,FLIST=HS scripts/m2_ww3_ref_hpc.slurm
sbatch -t 02:00:00 --export=ALL,RESTART=$R/runs/m2_week_start/restart001.ww3,HOURS=168,EXTRA=--wind-out scripts/m2_prop_hpc.slurm
```

**Scores on hpc.** The hpc cluster has no numpy and does not see the
Datarmor mirrors of the observations. A Python environment on scratch and a
copy of the January observations (as agreed, 22 November 2026):

```bash
# Python 3.9 of the system, packages through the proxy (304 MB):
# numpy 2.0.2, scipy 1.13.1, netCDF4 1.7.2, matplotlib 3.9.4 and their dependencies
python3 -m venv $R/pyenv
$R/pyenv/bin/pip install --no-cache-dir numpy scipy netCDF4 matplotlib
# from a machine that reaches both clusters: the January 2026 buoys (303 MB),
# the L3 altimeter files of January 2026 (607 MB) and the node files
B=/home/ref-copernicus-insitu/INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030/cmems_obs-ins_glo_phybgcwav_mynrt_na_irr_202311/monthly/MO
L=/home/ref-cmems-public/tac/wave/WAVE_GLO_PHY_SWH_L3_NRT_014_001
G=/home1/scratch/jmdeloui/HWW3/grids
ssh datarmor "tar -cf - -C $B 202601" | ssh hpc "mkdir -p $R/obs/insitu_MO && tar -m -xf - -C $R/obs/insitu_MO"
ssh datarmor "cd $L && tar -cf - */2026/01" | ssh hpc "mkdir -p $R/obs/l3 && tar -m -xf - -C $R/obs/l3"
ssh datarmor "tar -cf - -C $G natl_v5k4/plot/nodes.npz natl_v4/plot/nodes.npz natl_v4/raw/depth.f64" | ssh hpc "tar -m -xf - -C $R/grids"
```

The GPU fields become a netCDF file shaped like the WW3 one (coordinates,
triangles, time and ice copied from it, `scripts/gpu_to_nc.py`), then both
go through the validation scripts on the window of the explicit runs
(`VAL_END=2026-01-07T21:00`, `VAL_MIN_HOURS=48`):

```bash
sbatch --export=ALL,WW3NC=$R/runs/m2_week/ww3.202601.nc,GPU=$R/m2/out_JOBID,OUT=$R/m2/val_week scripts/m2_scores_hpc.slurm
```

The same 48 h in double precision on the GPU, and WW3 restarts every
10 minutes from 9 January 00:50 to 02:00 (around the spike of node 79649):

```bash
sbatch -t 01:30:00 --export=ALL,HOURS=48,EXTRA=--f64 scripts/m2_prop_hpc.slurm
sbatch -t 03:00:00 --export=ALL,MODE=dev,HOURS=26,RSTART=89400,RSTRIDE=600,RCOUNT=8,TAG=_diag9,FLIST=HS scripts/m2_ww3_ref_hpc.slurm
```

### Milestone M2b: the full ST4

The calibrated namelists unchanged (`ST4=full`), the harness on their
`mod_def.ww3`, then the GPU with the parameter file of that harness:

```bash
sbatch --export=ALL,MODE=dev,ST4=full,TAG=_full,RSTRIDE=600,RCOUNT=3 scripts/m2_ww3_ref_hpc.slurm
sbatch --export=ALL,RUN=$R/runs/m2_dev_full,OUT=src_8jan_full.bin,NODES=169562+130464+165663+229656+147650 scripts/m2_src_harness_hpc.slurm
target/release/hww3-src check $R/m2/harness/src_8jan_full.bin
sbatch --export=ALL,HARNESS=src_8jan_full.bin scripts/m2_src_gpu_hpc.slurm
D=$R/runs/m2_dev_full
sbatch --export=ALL,PARAMS=src_8jan_full.bin.params,STEPS_REF=$D/restart001.ww3:$D/restart002.ww3:$D/restart003.ww3 scripts/m2_prop_hpc.slurm
sbatch --export=ALL,PARAMS=src_8jan_full.bin.params,HOURS=48,EXTRA=--wind-out scripts/m2_prop_hpc.slurm
python3 scripts/m1_compare_nc.py $D/ww3.202601.nc $R/m2/out_JOBID --worst 20 --grid $R/grids/natl_v5k4
# the week: the restart after the first step of a cold-started run, then against the
# explicit run natl_v5k4_jan2026_exp (its ww3.202601.nc copied from Datarmor to $R/refs)
sbatch -t 00:30:00 --export=ALL,MODE=week,ST4=full,HOURS=1,RSTRIDE=600,RCOUNT=1,TAG=_start_full,FLIST=HS scripts/m2_ww3_ref_hpc.slurm
sbatch -t 02:00:00 --export=ALL,PARAMS=src_8jan_full.bin.params,RESTART=$R/runs/m2_week_start_full/restart001.ww3,HOURS=168,EXTRA=--wind-out scripts/m2_prop_hpc.slurm
sbatch --export=ALL,WW3NC=$R/refs/natl_v5k4_jan2026_exp/ww3.202601.nc,GPU=$R/m2/out_JOBID,OUT=$R/m2/val_week_full scripts/m2_scores_hpc.slurm
```

Grid v4: its set-up (`natl.msh`, `natl.obc`, `ww3_grid.nml`,
`namelists_natl.nml` of `natl_v4_jan2026_exp`) to `$R/runs/setup_v4`, its
forcing of 1–8 January to `$R/m2/forcing_v4` (cut with `ww3_subset.py` as
for grid v5 $K=4$, from `natl_v4_jan2026_cur` and `natl_v4_jan2026_bnd`),
its hourly fields to `$R/refs/natl_v4_jan2026_exp`, then:

```bash
V="SETUP=$R/runs/setup_v4,FORCING=$R/m2/forcing_v4"
sbatch -t 01:00:00 --mem=160G --export=ALL,MODE=week,ST4=full,$V,HOURS=1,RSTRIDE=600,RCOUNT=1,TAG=_v4start_full,FLIST=HS scripts/m2_ww3_ref_hpc.slurm
D=$R/runs/m2_week_v4start_full
sbatch --export=ALL,RUN=$D,RESTART=$D/restart001.ww3,WIND=$R/m2/forcing_v4/wind.ww3,OUT=src_v4_full.bin,STEP=400 scripts/m2_src_harness_hpc.slurm
sbatch -t 02:00:00 --mem=96G --export=ALL,GRID=$R/grids/natl_v4,FORCING=$R/m2/forcing_v4,PARAMS=src_v4_full.bin.params,RESTART=$D/restart001.ww3,HOURS=165 scripts/m2_prop_hpc.slurm
python3 scripts/m1_compare_nc.py $R/refs/natl_v4_jan2026_exp/ww3.202601.nc $R/m2/out_JOBID --worst 20 --grid $R/grids/natl_v4
sbatch --mem=96G --export=ALL,WW3NC=$R/refs/natl_v4_jan2026_exp/ww3.202601.nc,GPU=$R/m2/out_JOBID,OUT=$R/m2/val_v4_full,GRID=$R/grids/natl_v4 scripts/m2_scores_hpc.slurm
```

Finite-difference gradients of M2b: 48 h in double precision from the
restart of 8 January on grid v5 $K=4$, each parameter multiplied by
$1 \pm 0.01$ (and $1 \pm 0.001$ for the wind and `BETAMAX`). `BETAMAX`
enters the stress tables of `ww3_grid`, so it goes through perturbed
`mod_def.ww3` files and their harness parameter files; `hww3-src tables`
checks the derivative of `TABU_TAUHF2` against them. One line `NAME JOBID`
per run in `$R/m2/fd/cases.txt`, then the gradients of the energy, mean
$H_s$ and scores:

```bash
F=src_8jan_full.bin.params
sbatch -t 01:00:00 "--export=ALL,PARAMS=$F,HOURS=48,EXTRA=--f64" scripts/m2_prop_hpc.slurm                              # base
sbatch -t 01:00:00 "--export=ALL,PARAMS=$F,HOURS=48,EXTRA=--f64 --scale-par SDSC2:1.01" scripts/m2_prop_hpc.slurm       # likewise NLPROP, BJGAM, 0.99
sbatch -t 01:00:00 "--export=ALL,PARAMS=$F,HOURS=48,EXTRA=--f64 --wind-factor 1.01" scripts/m2_prop_hpc.slurm           # likewise 0.99, 1.001, 0.999
sbatch --export=ALL,BETAMAX=1.7675,TAG=bp1 scripts/m2_moddef_hpc.slurm                                                     # likewise 1.7325, 1.75175, 1.74825
sbatch --export=ALL,RUN=$R/runs/m2_moddef_bp1,OUT=par_bp1.bin,STEP=1000000,NODES=169562 scripts/m2_src_harness_hpc.slurm
sbatch -t 01:00:00 "--export=ALL,PARAMS=par_bp1.bin.params,HOURS=48,EXTRA=--f64" scripts/m2_prop_hpc.slurm
target/release/hww3-src tables $R/m2/harness/$F $R/m2/harness/par_bp1.bin.params $R/m2/harness/par_bm1.bin.params
sbatch --dependency=afterany:JOBIDS scripts/m2_fd_hpc.slurm
```

Milestone M3, with the binaries built in `$R/target_wave_m3`: the tangent
kernel of the source terms on the harness nodes, short tangent runs with
their own finite differences, and the 48 h tangent runs against the runs
above:

```bash
sbatch --export=ALL,HARNESS=src_8jan_full.bin scripts/m3_tl_src_hpc.slurm
sbatch --export=ALL,HARNESS=src_v4_full.bin scripts/m3_tl_src_hpc.slurm
sbatch -t 00:40:00 --export=ALL,NAME=INIT,NOSRC=1,HOURS=3,H=1e-4:1e-5 scripts/m3_tl_run_hpc.slurm
sbatch -t 00:40:00 --export=ALL,NAME=INIT,HOURS=3,H=1e-4:1e-5 scripts/m3_tl_run_hpc.slurm
sbatch -t 00:40:00 --export=ALL,NAME=WIND,HOURS=3,H=1e-4:1e-5 scripts/m3_tl_run_hpc.slurm
O=$R/m3/tl_JOBID; python3 scripts/m3_tl_compare.py $O/tangent 1e-4:$O/plus_1e-4:$O/minus_1e-4 1e-5:$O/plus_1e-5:$O/minus_1e-5 --tol 1e-6 --target 1e-4
sbatch --export=ALL,NAME=BETAMAX,HOURS=48 scripts/m3_tl_run_hpc.slurm            # one per direction; lines NAME JOBID in $R/m3/tl48.txt
sbatch --dependency=afterok:JOBIDS scripts/m3_tl_cmp_hpc.slurm
```

Calibration of the ST4 parameters: the stress tables of `INSIN4` computed
in Rust against those of WW3 in the base and perturbed parameter files, a
parameter file for other values, and two short GPU runs with WW3's tables
and with the Rust tables:

```bash
H=$R/m2/harness
target/release/hww3-src tables $H/src_8jan_full.bin.params $H/par_bp01.bin.params $H/par_bm01.bin.params $H/par_bp1.bin.params $H/par_bm1.bin.params
target/release/hww3-src params $H/src_8jan_full.bin.params $R/cal/rust_bp1.params --betamax 1.7675 --scale-par SDSC2:1.01+NLPROP:0.99+BJGAM:1.02
sbatch -t 00:20:00 --export=ALL,TAG=tabchk_ww3,F64=1,HOURS=3,RESTART=$R/m1/v5k4/restart001.ww3 scripts/cal_run_hpc.slurm
sbatch -t 00:20:00 --export=ALL,TAG=tabchk_rust,BETAMAX=1.75,F64=1,HOURS=3,RESTART=$R/m1/v5k4/restart001.ww3 scripts/cal_run_hpc.slurm
```

The calibration (all runs from the restart after the first step of the
cold start of 1 January, `runs/m2_week_start_full`): the start run to
10 January, the observation operators of the two windows, then per
iteration K four tangent runs to 7 January 12 h, the cost and its
Jacobian, the Gauss–Newton step and a value run at the trial point,
kept if its cost is lower (`trialK.env` holds the parameters of the
step, `xK.json` the accepted point):

```bash
sbatch -t 01:00:00 --export=ALL,TAG=it0_value,BETAMAX=1.75,HOURS=216 scripts/cal_run_hpc.slurm
ICE=$R/refs/natl_v5k4_jan2026_exp/ww3.202601.nc+$R/runs/m2_dev_full/ww3.202601.nc
sbatch --export=ALL,SCRIPT=cal_obs.py,ARGS="build $R/cal/op_train.npz $R/cal/it0_value 2026010512 2026010712 $ICE $R/grids/natl_v5k4" scripts/cal_py_hpc.slurm
sbatch --export=ALL,SCRIPT=cal_obs.py,ARGS="build $R/cal/op_valid.npz $R/cal/it0_value 2026010713 2026011000 $ICE $R/grids/natl_v5k4" scripts/cal_py_hpc.slurm
for n in BETAMAX SDSC2 NLPROP BJGAM; do      # iteration K, parameters of xK (K = 0: BETAMAX=1.75)
  sbatch -t 02:00:00 --export=ALL,TAG=itK_$n,HOURS=156,TANGENT=$n,BETAMAX=...,FSDSC2=...,FNLPROP=...,FBJGAM=... scripts/cal_run_hpc.slurm
done
D=$R/cal/itK
python3 scripts/cal_obs.py eval $R/cal/op_train.npz ${D}_BETAMAX BETAMAX:${D}_BETAMAX SDSC2:${D}_SDSC2 NLPROP:${D}_NLPROP BJGAM:${D}_BJGAM --npz $R/cal/itK_eval.npz
python3 scripts/cal_gn.py $R/cal/itK_eval.npz $R/cal/xK.json $R/cal/gnK.json
sbatch -t 00:40:00 --export=ALL,TAG=itK+1_trial,HOURS=156,BETAMAX=...,FSDSC2=...,FNLPROP=...,FBJGAM=... scripts/cal_run_hpc.slurm
python3 scripts/cal_obs.py eval $R/cal/op_train.npz $R/cal/itK+1_trial
# the parameter file built in Rust from the namelists and the grid, against the harness file
target/release/hww3-src build-params $R/grids/natl_v5k4 $R/runs/test_grid_v5k4/namelists_natl.nml $R/cal/rust_full.params --grid-nml $R/runs/test_grid_v5k4/ww3_grid.nml --check $R/m2/harness/src_8jan_full.bin.params
# control only: the start-point derivatives against the M2b finite differences of 8-10 January
python3 scripts/cal_fdcheck.py $R/cal/op_0810.npz $R/m2/fd/cases.txt fdchk $R/cal/fdcheck.json
```

## 9. Note

```bash
cd tex/hww3 && latexmk -pdf main.tex
```

`tex/hww3/main.tex` gathers the grid statistics, the figures, the WW3
build, the run costs and the results; it is updated after every run.
It is mirrored on Overleaf (project `6af6a96c4ed66a0cad905cb6`) by
`scripts/sync_overleaf.sh`, which needs an Overleaf git token in the git
credential store.
