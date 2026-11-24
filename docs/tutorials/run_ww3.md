# Tutorial: prepare the inputs and run WW3

This tutorial builds the North Atlantic grid v5 $K=4$ (the grid of the
GPU model), prepares its inputs and runs WW3 on it for 1–8 January 2026
on Datarmor, with the implicit scheme, the ERA5 wind and ice, the
Copernicus Marine currents and water level, and open-boundary spectra.

## Prerequisites

- A Datarmor account, and the repository in the working directory:

  ```bash
  R=/home1/datawork/$USER/HWW3
  git clone git@github.com:jmdelouis/HWW3.git $R
  cd $R
  mkdir -p logs grids runs forcing
  ```

- A Python environment with `cdsapi`, `numpy` and `netCDF4` in
  `$R/venv`, and a Copernicus Climate Data Store key in `~/.cdsapirc`
  (see [Installation](../installation.md)).
- The steps that are done once for every grid, from
  [Reproducing](../reproducing.md): the sources, the WW3 build, and the
  forcing of January 2026. The forcing scripts interpolate onto the first
  North Atlantic grid (`grids/natl`, in `runs/natl`) and leave the netCDF
  fields in `runs/natl_jan2026` and `runs/natl_jan2026_cur`, which every
  later grid reuses:

  ```bash
  bash scripts/fetch_sources.sh $R                         # login node: WW3, ParMETIS, GEBCO
  $R/venv/bin/python scripts/get_era5.py 2026 1 forcing/era5   # login node: ERA5, 7 min
  JB=$(qsub scripts/build_ww3.pbs)                         # WW3, 5.5 min
  JG=$(qsub scripts/natl_grid.pbs)                         # grid v1
  JW=$(qsub -W depend=afterok:$JB:$JG scripts/natl_ww3_grid.pbs)
  qsub -W depend=afterok:$JW scripts/natl_jan2026_prep.pbs       # wind and ice
  qsub -W depend=afterok:$JW scripts/natl_jan2026_cur_prep.pbs   # currents and water level, 14 min
  ```

As written above, the PBS scripts work in `/home1/datawork/jmdeloui/HWW3`
and write their logs there: another user adds `-v R=$R` and
`-o logs/<name>.log` to each `qsub`, as the commands below do.

## 1. Build the grid

```bash
cd $R
G=grids/natl_v5k4
J=$(qsub -v R=$R,CFG=configs/natl_v5k4_datarmor.json,GRID=$G -o logs/grid_v5k4.log scripts/natl_grid.pbs)
```

The job builds the generator, the grid (about 17 min on 8 cores: this
grid caps the sub-steps of the explicit scheme, which takes several
builds), the cell vertices and the plotting files `grids/natl_v5k4/plot/`.

## 2. Prepare the run directory

```bash
J1=$(qsub -W depend=afterok:$J -v R=$R,BETAMAX=1.75,RUN=natl_v5k4_jan2026_cur,GRID=$G \
     -o logs/natl_v5k4_setup.log scripts/natl_v2_setup.pbs)
```

`natl_v2_setup.pbs` makes `runs/natl_v5k4_jan2026_cur`, runs `ww3_grid` on
the new mesh with `BETAMAX = 1.75` appended to the namelists, then
`ww3_prnc` on the wind, ice, currents and water level of January 2026.

## 3. Add the open-boundary spectra

```bash
J2=$(qsub -W depend=afterok:$J1 \
     -v "R=$R,SRC=natl_v5k4_jan2026_cur,RUN=natl_v5k4_jan2026_bnd,GRID=$G,STOP=20260108 000000" \
     -o logs/natl_v5k4_bnd_setup.log scripts/natl_bnd_setup.pbs)
```

`natl_bnd_setup.pbs` links the grid and the forcing of the previous
directory into `runs/natl_v5k4_jan2026_bnd`, rebuilds the spectra of the
global run next to the open boundary, runs `ww3_bounc` (which writes
`nest.ww3`), and sets the end of the run to 8 January 00 h.

## 4. Run WW3

```bash
qsub -W depend=afterok:$J2 -v R=$R,RUN=natl_v5k4_jan2026_bnd -o logs/natl_v5k4_bnd_run.log scripts/natl_v2_run.pbs
qstat -u $USER
```

`natl_v2_run.pbs` runs `ww3_shel` on 112 MPI ranks (4 nodes), then
`ww3_ounf`.

## Expected output

| Step | Time | Output |
|---|---|---|
| Grid | about 17 min | `grids/natl_v5k4/` and `grids/natl_v5k4_summary.json`: 424,123 nodes |
| Run directory | a few minutes | `runs/natl_v5k4_jan2026_cur/mod_def.ww3`, `wind.ww3`, `ice.ww3`, `current.ww3`, `level.ww3` |
| Boundary spectra | a few minutes | `runs/natl_v5k4_jan2026_bnd/nest.ww3` |
| Run | about 1,060 s per model day, about 2 h for the week | `runs/natl_v5k4_jan2026_bnd/ww3.202601.nc` |

## Check it

1. `ww3_grid` took the new `BETAMAX` and the whole mesh:

   ```bash
   grep -i betamax runs/natl_v5k4_jan2026_cur/ww3_grid.out     # betamax : 1.750
   grep -iE "sea points|boundary" runs/natl_v5k4_jan2026_cur/ww3_grid.out
   ```

   The number of sea points is the number of nodes of the grid, and the
   number of active boundary points that of the nodes listed in
   `grids/natl_v5k4/natl.obc`.

2. The run went to its end: the run log gives the wall times of
   `ww3_shel` and `ww3_ounf`, and lists the run directory with
   `ww3.202601.nc`:

   ```bash
   grep wall logs/natl_v5k4_bnd_run.log
   ```

3. The output holds defined fields at every hour:

   ```bash
   $R/venv/bin/python - <<'EOF'
   import netCDF4, numpy as np
   d = netCDF4.Dataset("runs/natl_v5k4_jan2026_bnd/ww3.202601.nc")
   hs = d["hs"]
   print("hours x nodes:", hs.shape)
   print("median Hs at the last hour: %.2f m" % np.ma.median(hs[-1]))
   EOF
   ```

   The second dimension is 424,123 nodes. The real check is the
   comparison with the observations: [Validate a run](validate_run.md).

## Variants

- **Explicit scheme** (the scheme of the GPU model, about 2,600 s per
  model day): prepare a third directory with `natl_exp_setup.pbs` after
  step 3 and run it, as in [Reproducing](../reproducing.md), section
  *Grid v5 (explicit sub-steps capped)*.
- **The whole month**: leave out `STOP` in step 3.
- **The tide**: `natl_tide_setup.pbs` adds the FES2014 tide to the
  currents and the water level ([Reproducing](../reproducing.md), section
  *FES2014 tide*).
- **The hpc cluster**: WW3 is built there by `build_ww3_hpc.slurm`; the
  runs made there are the references of the GPU model
  ([Reproducing](../reproducing.md), section 8).
