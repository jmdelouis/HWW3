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
qsub scripts/natl_grid.pbs                       # grid without angle control
qsub -v CFG=configs/natl_v2_datarmor.json,GRID=grids/natl_v2 scripts/natl_grid.pbs
```

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

### Altimeter validation

```bash
qsub -v RUN=natl_jan2026,GRID=grids/natl scripts/natl_validate.pbs   # 2 min
```

Writes `runs/natl_jan2026/val/natl_jan2026_{stats.json,scatter.png,maps.png}`.
Expected for the wind-and-ice run: 331,154 blocks, bias −0.19 m,
RMSE 0.45 m, SI 0.14, r 0.972 (see Validation). Needs `scipy` and
`matplotlib` in the venv (`pip install --only-binary=:all: scipy matplotlib`).

## 8. Note

```bash
cd tex/hww3 && latexmk -pdf main.tex
```

`tex/hww3/main.tex` gathers the grid statistics, the figures, the WW3
build, the run costs and the results; it is updated after every run.
It is mirrored on Overleaf (project `6af6a96c4ed66a0cad905cb6`) by
`scripts/sync_overleaf.sh`, which needs an Overleaf git token in the git
credential store.
