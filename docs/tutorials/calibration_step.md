# Tutorial: take one calibration step from tangent runs

This tutorial repeats the first step of the calibration of the ST4
parameters `BETAMAX`, `SDSC2` and `NLPROP` on the hpc cluster: a run at the
start point, the observation operators, one tangent run per parameter, the
cost and its derivatives, a Gauss–Newton step, and a run at the new point
to check that the cost decreased. What each piece does is explained in
[Learning the parameters](../learning.md).

The tutorial uses the **first cost** of the calibration (nearest-node
buoys, observation errors 0.25 m for the altimeters, 0.30 m and 1.0 s for
the buoys), whose numbers are known. The later costs add the depth-aware
buoy operator, representativeness errors, excluded buoys and a Huber loss
([GPU model results](../wave_results.md), *Calibration of the ST4
parameters*).

## Prerequisites

- Everything the tutorial [Run the GPU model for 48 h](run_gpu.md) needs,
  plus the restart written by WW3 after the first step of a cold start on
  1 January, `$R/runs/m2_week_start_full/restart001.ww3`.
- The model built in `$R/target_wave_m3`, where
  `scripts/cal_run_hpc.slurm` looks for it:

  ```bash
  cd $R/hww3-wave
  module load cuda/13.0.0
  CARGO_TARGET_DIR=$R/target_wave_m3 cargo build --release --bin hww3-prop
  ```

- What the tutorial [Validate a run](validate_run.md) needs on the hpc
  cluster (the Python environment and the observations), and the WW3 files
  that give the ice over the two windows,
  `$R/refs/natl_v5k4_jan2026_exp/ww3.202601.nc` and
  `$R/runs/m2_dev_full/ww3.202601.nc`.

```{warning}
`$R/cal` holds the runs and operators of the reference calibration
(`it0_value`, `op_train.npz`, ...). The commands below write new names
starting with `tuto_`: do not reuse the reference names.
```

## 1. Run at the start point

```bash
R=/scratch/jmdeloui/HWW3
cd $R
sbatch -t 01:00:00 --export=ALL,TAG=tuto_it0_value,BETAMAX=1.75,HOURS=216 scripts/cal_run_hpc.slurm
```

The model runs 1 to 10 January (216 h) from the restart of 1 January, with
`BETAMAX` 1.75 and the other parameters at their values in the parameter
file. **Expected:** hourly `hs_` and `t02_` fields in `$R/cal/tuto_it0_value`,
in about 20 minutes.

## 2. Observation operators

Once the run is done:

```bash
ICE=$R/refs/natl_v5k4_jan2026_exp/ww3.202601.nc+$R/runs/m2_dev_full/ww3.202601.nc
G=$R/grids/natl_v5k4
sbatch --export=ALL,SCRIPT=cal_obs.py,ARGS="build $R/cal/tuto_op_train.npz $R/cal/tuto_it0_value 2026010512 2026010712 $ICE $G" scripts/cal_py_hpc.slurm
sbatch --export=ALL,SCRIPT=cal_obs.py,ARGS="build $R/cal/tuto_op_valid.npz $R/cal/tuto_it0_value 2026010713 2026011000 $ICE $G" scripts/cal_py_hpc.slurm
```

The training window is 5 January 12 h to 7 January 12 h, the validation
window 7 January 13 h to 10 January 00 h: both start after the 4 days the
cold start of 1 January takes to settle, and `cal_obs.py` refuses a window
that starts in the first 30 % of its run. **Expected:** the log
`$R/logs/calpy_JOBID.log` of the training window counts 24,153 altimeter
blocks, 12,247 buoy hours of $H_s$ and 8,292 of $T_{m02}$.

## 3. One tangent run per parameter

```bash
for n in BETAMAX SDSC2 NLPROP; do
  sbatch -t 02:00:00 --export=ALL,TAG=tuto_it0_$n,HOURS=156,TANGENT=$n,BETAMAX=1.75 scripts/cal_run_hpc.slurm
done
```

Each job runs the model in double precision to 7 January 12 h, the end of
the training window, together with its tangent for a relative change of
one parameter: besides `hs_` and `t02_`, it writes `dhs_` and `dt02_`, the
derivatives of $H_s$ and $T_{m02}$ at every node and hour. `BJGAM` is left
out: it was frozen at 0.73 after the first step of the reference
calibration, since few observations constrain it.

## 4. Cost and derivatives

```bash
D=$R/cal/tuto_it0
sbatch --export=ALL,SCRIPT=cal_obs.py,ARGS="eval $R/cal/tuto_op_train.npz ${D}_BETAMAX BETAMAX:${D}_BETAMAX SDSC2:${D}_SDSC2 NLPROP:${D}_NLPROP --npz $R/cal/tuto_it0_eval.npz" scripts/cal_py_hpc.slurm
```

The operators are applied to the fields of the first tangent run (the
model at the start point) and to the three tangents. **Expected,** in the
log: the scores of the start point and the cost $J$ ≈ 47,250 over 44,595
residuals, with the derivatives $\partial J/\partial \ln p$ ≈ +25,100
(`BETAMAX`), −10,600 (`SDSC2`) and +5,900 (`NLPROP`); the residuals and
their derivatives go to `tuto_it0_eval.npz`.

## 5. The Gauss–Newton step

```bash
$R/pyenv/bin/python scripts/cal_gn.py $R/cal/tuto_it0_eval.npz $R/cal/tuto_x0.json $R/cal/tuto_gn0.json
```

`tuto_x0.json` does not exist, so the step starts from the start point
($x = 0$). The script prints $J$, for each parameter its gradient, its
step and the new factor, the predicted $J$ and the correlations of the
parameters, and writes them, with the posterior covariance, to
`tuto_gn0.json`. **Expected:** factors of about 0.976 for `BETAMAX`
(1.7077), 1.183 for `SDSC2` and 1.255 for `NLPROP`, and a predicted
decrease of $J$ of about 1.1 %.

## 6. Check the step with a run

With the factors printed by `cal_gn.py` (`BETAMAX` is 1.75 times its
factor):

```bash
sbatch -t 00:40:00 --export=ALL,TAG=tuto_it1_trial,HOURS=156,BETAMAX=1.7077,FSDSC2=1.183,FNLPROP=1.255 scripts/cal_run_hpc.slurm
```

then, once it is done:

```bash
sbatch --export=ALL,SCRIPT=cal_obs.py,ARGS="eval $R/cal/tuto_op_train.npz $R/cal/tuto_it1_trial" scripts/cal_py_hpc.slurm
```

**Expected:** a training cost of about 46,630 against 47,250 at the start,
−1.3 %, more than predicted: the step is accepted. The next step repeats
steps 3 to 6 at the new point; `cal_gn.py` then reads that point from its
second argument, a file `{"names": [...], "x": [...]}` with
$x = \ln(p / p_0)$.

## Check it

- The cost of step 4 and that of step 6 are evaluated on the same
  observations (the same operator file): only then can they be compared.
- The validation window is the honest test: evaluate both runs with
  `tuto_op_valid.npz` (the start run of step 1 covers it; run the trial
  point for 216 h to cover it too). In the reference calibration the
  validation cost fell from 69,351 to 68,857 at this step.
- A WW3 run with the same parameters gives the same costs to 0.1
  ([GPU model results](../wave_results.md), *Steps of the calibration*).
