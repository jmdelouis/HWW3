# Tutorial: run the GPU model for 48 h

This tutorial builds the GPU model `hww3-prop` on the Ifremer hpc cluster,
runs it for 48 h from 8 January 2026 on grid v5 $K=4$ with the full ST4
physics on one A100, and compares it with WW3 run on the same inputs.

## Prerequisites

- An account on the hpc cluster with access to the `gpu` partition (A100).
- The working directory `R=/scratch/jmdeloui/HWW3`, which holds the
  repository and the inputs prepared once from the Datarmor runs
  ([Reproducing](../reproducing.md), section 8):

  | Input | Path under `$R` |
  |---|---|
  | the grid | `grids/natl_v5k4/` |
  | the WW3 restart of 8 January 00 h | `m1/v5k4/restart001.ww3` |
  | wind, currents, water level, ice and boundary spectra of 1–10 January | `m2/forcing/{wind,current,level,ice,nest}.ww3` |
  | the parameter file of the source terms (full ST4) | `m2/harness/src_8jan_full.bin.params` |
  | the WW3 run of the same 48 h, for the comparison | `runs/m2_dev_full/ww3.202601.nc` |

- A Rust toolchain (stable) on the login node.

The job scripts write their logs to `/scratch/jmdeloui/HWW3/logs` and read
from `S=/scratch/jmdeloui/HWW3`: another user changes these two lines at
the top of `scripts/m2_prop_hpc.slurm`.

## 1. Build the model

On the login node:

```bash
R=/scratch/jmdeloui/HWW3
cd $R/hww3-wave
module load cuda/13.0.0
CARGO_TARGET_DIR=$R/target_wave cargo build --release --bin hww3-prop
```

**Expected:** `$R/target_wave/release/hww3-prop`, the program the job
script runs. The first build downloads and compiles the crates (CubeCL,
Burn) and takes several minutes. The GPU kernels themselves are compiled
when the model first launches them, by the NVRTC of the CUDA toolkit.

## 2. Run 48 h on one A100

```bash
cd $R
sbatch --export=ALL,PARAMS=src_8jan_full.bin.params,HOURS=48 scripts/m2_prop_hpc.slurm
squeue -u $USER
```

The script loads the CUDA module, starts from the restart of 8 January and
reads the forcing, the boundary spectra and the parameter file above.

**Expected:** `JOBID` below is the number that `sbatch` printed. The log
`$R/logs/m2gpu_JOBID.log` starts with the GPU (`NVIDIA A100-SXM4-80GB`)
and the inputs, and ends with the time of the run: about 254 s for the
48 h, that is 127 s per model day. The output directory
`$R/m2/out_JOBID` holds one `hs_YYYYMMDDHH.f32` and one
`t02_YYYYMMDDHH.f32` per hour (little-endian `float32`, one value per node,
−999.9 where the model has no value) and `summary.json`.

## 3. Compare with WW3

```bash
python3 scripts/m1_compare_nc.py $R/runs/m2_dev_full/ww3.202601.nc $R/m2/out_JOBID \
    --worst 20 --grid $R/grids/natl_v5k4
```

The script needs no numpy (the hpc login nodes have none). For every node
it computes the RMS difference over the hours between the two runs, then
prints its median and its 99th percentile over the nodes, the 20 nodes
that differ most, and the difference of the total energy.

**Expected:**

| | Median node | 99th percentile | Worst node |
|---|---|---|---|
| $H_s$ | 2.0 µm | 77 µm | 2.1 cm |
| $T_{m02}$ | 4.9e-6 s | 3.2e-4 s | 0.16 s |

and a total energy within 1.1e-6 (relative) of WW3's. The worst node,
134247, is on the Normandy coast at 6.8 m of depth. The targets of the
model were 1 cm at the median node, 5 cm at the 99th percentile and 0.5 %
of energy: differences of a few micrometres are round-off.

## Going further

| To do | How |
|---|---|
| Double precision | add `EXTRA=--f64` and `-t 01:30:00` to `sbatch` |
| A week from 1 January, scored against the altimeters and buoys | `--export=ALL,PARAMS=src_8jan_full.bin.params,RESTART=$R/runs/m2_week_start_full/restart001.ww3,HOURS=168` with `-t 02:00:00` (about 15 min), then [Validate a run](validate_run.md) |
| Other values of the ST4 parameters | `scripts/cal_run_hpc.slurm` with `BETAMAX`, `FSDSC2`, `FNLPROP`, `FBJGAM` (see [Take one calibration step](calibration_step.md)) |
| Inputs made in Rust only, from calm conditions | `scripts/pre_allrust_hpc.slurm` (see [Preparing a run](../preparing.md)) |
| Every option of the program | [GPU wave model](../wave_model.md) |
