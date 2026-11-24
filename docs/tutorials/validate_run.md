# Tutorial: validate a run against altimeters and buoys

This tutorial scores a run against the observations of January 2026: the
significant wave height $H_s$ of the satellite altimeters, and $H_s$ and
the mean period $T_{m02}$ of the moored buoys. Part A does it on the hpc
cluster for the WW3 explicit run of grid v5 $K=4$ and, if you have one, a
GPU run of the same week; part B on Datarmor for the run of the tutorial
[Prepare the inputs and run WW3](run_ww3.md).

The method is the same in both parts (see [Analysis tools](../analysis.md)):
each altimeter sample takes the model at the nearest node, and 7
consecutive samples are averaged into a block of about 50 km; each buoy
takes the nearest node within 10 km, hour by hour. The first 3 days of a
run starting from rest are skipped, and the window ends on 7 January
21 h, so that runs of different lengths are scored on the same
observations.

## Part A: on the hpc cluster

### Prerequisites

Under `R=/scratch/jmdeloui/HWW3` ([Reproducing](../reproducing.md),
section 8, *Scores on hpc*):

- the Python environment `$R/pyenv` (`numpy`, `scipy`, `netCDF4`,
  `matplotlib`);
- the observations of January 2026 copied from Datarmor: the L3 altimeter
  files in `$R/obs/l3` and the buoy files in `$R/obs/insitu_MO/202601`;
- the grid files `grids/natl_v5k4/plot/nodes.npz` and
  `grids/natl_v5k4/raw/depth.f64`;
- the hourly fields of the WW3 explicit run of grid v5 $K=4$, copied from
  Datarmor to `$R/refs/natl_v5k4_jan2026_exp/ww3.202601.nc`;
- optionally, a GPU run of 1–8 January (the week of
  [Run the GPU model](run_gpu.md), *Going further*) in `$R/m2/out_JOBID`.

### Commands

The WW3 run alone:

```bash
R=/scratch/jmdeloui/HWW3
cd $R
sbatch --export=ALL,WW3NC=$R/refs/natl_v5k4_jan2026_exp/ww3.202601.nc,OUT=$R/val/v5k4_exp scripts/m2_scores_hpc.slurm
```

The WW3 run and a GPU run together (the GPU fields are first written as a
netCDF file shaped like WW3's by `scripts/gpu_to_nc.py`):

```bash
sbatch --export=ALL,WW3NC=$R/refs/natl_v5k4_jan2026_exp/ww3.202601.nc,GPU=$R/m2/out_JOBID,OUT=$R/val/week scripts/m2_scores_hpc.slurm
```

The job takes a few minutes on one CPU node. `VAL_END` (default
`2026-01-07T21:00`) and `VAL_MIN_HOURS` (default 48) can be passed in
`--export` to change the window and the number of hours a buoy needs.

### Expected output

In the `OUT` directory:

| File | Content |
|---|---|
| `ww3_stats.json`, `ww3_scatter.png`, `ww3_maps.png` | the altimeter scores of the WW3 run (and `gpu_*` for the GPU run) |
| `buoys_stats.json`, `buoys_scatter.png` | the buoy scores of every run, on the same buoys and hours |
| `buoys_series.png`, `buoys_series_tm02.png` | with two runs: $H_s$ and $T_{m02}$ at the four buoys that differ most |

and these scores for the WW3 run (and the same for the GPU run, to the
digits printed):

| Observations | Number | Bias | RMSE |
|---|---|---|---|
| Altimeters, $H_s$ | 45,518 blocks | −1.50 cm | 28.25 cm |
| Buoys, $H_s$ | 252 buoys | −5.38 cm | 32.46 cm |
| Buoys, $T_{m02}$ | 171 buoys | −0.424 s | 1.210 s |

### Check it

The log `$R/logs/scores_JOBID.log` prints the scores as the scripts compute
them. To read them from the files:

```bash
$R/pyenv/bin/python - <<'EOF'
import json
d = "/scratch/jmdeloui/HWW3/val/v5k4_exp/"
a = json.load(open(d + "ww3_stats.json"))["all"]
print("altimeters: %d blocks, bias %+.2f cm, RMSE %.2f cm" % (a["n"], 100 * a["bias"], 100 * a["rmse"]))
b = json.load(open(d + "buoys_stats.json"))
for var, unit in (("hs", "m"), ("tm02", "s")):
    s = b[var]["all"]["ww3"]
    print("buoys %s: %d buoys, bias %+.3f %s, RMSE %.3f %s" % (var, s["n_buoys"], s["bias"], unit, s["rmse"], unit))
EOF
```

Open the figures: the points of the scatter plots should lie along the
diagonal, and the maps show where along the tracks the model is too high
or too low.

## Part B: on Datarmor

### Prerequisites

- The run `runs/natl_v5k4_jan2026_bnd` of the tutorial
  [Prepare the inputs and run WW3](run_ww3.md), and its grid
  `grids/natl_v5k4` with `plot/nodes.npz`.
- `scipy` and `matplotlib` in the Python environment of `$R/venv`:

  ```bash
  $R/venv/bin/pip install --only-binary=:all: scipy matplotlib
  ```

The observations are read from the Datarmor mirrors, the default of both
scripts.

### Commands

```bash
R=/home1/datawork/$USER/HWW3
cd $R
RUN=natl_v5k4_jan2026_bnd
V=runs/$RUN/val
mkdir -p $V
VAL_END=2026-01-07T21:00 $R/venv/bin/python scripts/validate_altimetry.py \
    runs/$RUN/ww3.202601.nc grids/natl_v5k4/plot/nodes.npz $V/$RUN
VAL_MIN_HOURS=48 $R/venv/bin/python scripts/validate_buoys.py \
    $V/buoys v5k4imp:runs/$RUN/ww3.202601.nc:grids/natl_v5k4
```

Each script takes a few minutes. `scripts/natl_validate.pbs` and
`scripts/natl_buoys.pbs` run the same scripts as PBS jobs.

### Expected output

`runs/natl_v5k4_jan2026_bnd/val/` holds `natl_v5k4_jan2026_bnd_stats.json`,
`_scatter.png` and `_maps.png` for the altimeters, and `buoys_stats.json`
and `buoys_scatter.png` for the buoys. The altimeter scores of this run
(implicit scheme, 4 January 00 h to 7 January 21 h) are 45,518 blocks, a
bias of −0.029 m and an RMSE of 0.283 m. For the buoys, expect a bias of
about −0.05 m and an RMSE of about 0.33 m in $H_s$, and an RMSE of about
1.2 s in $T_{m02}$: the exact numbers depend on the buoys kept, which
depend on the runs scored together (see [Validation](../validation.md),
section *Grid v5: explicit sub-steps capped*).

## Going further

- Score several runs at once: give `validate_buoys.py` several
  `LABEL:RUN_NC:GRID_DIR` arguments; they are then compared on the same
  buoys and hours.
- Map the difference of two runs:
  `scripts/plot_run_diff.py A.nc "label A" B.nc "label B" grids/<grid>/raw/tri.u32 out.png 3`.
- The scores of every reference run, and what they mean:
  [Validation](../validation.md) and [GPU model results](../wave_results.md).
