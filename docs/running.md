# Running the model

Two models run on the same grids:

- **WAVEWATCH III** (Fortran, many CPU cores with MPI), unmodified, with
  the implicit or the explicit scheme. It is the reference.
- **The GPU model** of the `hww3-wave` crate (Rust, one A100), a rewrite
  of the explicit configuration of WW3 that also computes derivatives. It
  is checked against WW3 at every stage.

## WW3 on Datarmor and on the hpc cluster

On Datarmor, `scripts/natl_v2_run.pbs` runs `ww3_shel` on 112 MPI ranks
(4 nodes of 28 cores) in a run directory prepared by the set-up scripts
(see [Preparing a run](preparing.md)), then `ww3_ounf`, which writes the
hourly fields to `ww3.202601.nc`: $H_s$, $T_{m02}$, peak frequency, mean
direction, wind and ice, plus currents and water level when they force the
run. Weekly restarts are written too.

Wall time per model day on 112 ranks, 2 to 7 January 2026:

| Grid | Nodes | Implicit | Explicit |
|---|---|---|---|
| v4 | 794,613 | 2,115 s | 4,318 s |
| v5 $K=4$ | 424,123 | 1,058 s | 2,601 s |

The same WW3 source and switches are built on the hpc cluster, where they
make the references of the GPU model (`scripts/m1_ww3_prop_hpc.slurm`,
`scripts/m2_ww3_ref_hpc.slurm`); the two builds agree to 1.4 mm in $H_s$
after 6 h (see [WW3 setup](ww3.md)).

- Commands of every reference run: [Reproducing](reproducing.md),
  section 7 (Datarmor) and section 8 (hpc cluster).
- Hands-on: [Prepare the inputs and run WW3](tutorials/run_ww3.md).

## The GPU model `hww3-prop`

`hww3-prop run` steps the spectra of grid v5 $K=4$ or v4 on one A100 with
everything the explicit WW3 configuration has: propagation in space,
refraction and frequency shift, currents, water levels, ice, open-boundary
spectra and the full ST4 source terms. With the full ST4 it takes 124 s
per model day on grid v5 $K=4$ and 220 s on grid v4, and over a week it
stays within 2.2 µm of WW3 in $H_s$ at the median node, with the same
altimeter and buoy scores.

| Item | Description |
|---|---|
| Inputs | a WW3 restart (or `calm`), the forcing files `wind.ww3`, `current.ww3`, `level.ww3`, `ice.ww3`, the boundary file `nest.ww3`, and the parameter file of the source terms |
| Outputs | hourly $H_s$ and $T_{m02}$ per node, `hs_YYYYMMDDHH.f32` and `t02_YYYYMMDDHH.f32`, and `summary.json`; on request a WW3 restart at the end |
| Options | double precision (`--f64`), other parameter values (`--betamax`, `--scale-par`), tangent runs (`--tangent`), switches to turn parts off, profiling |
| Job scripts | `m2_prop_hpc.slurm` (48 h from 8 January, or a step-by-step check), `cal_run_hpc.slurm` (the runs of the calibration, from 1 January), `pre_allrust_hpc.slurm` (a run with Rust inputs only) |

- Build, every program and every option: [GPU wave model](wave_model.md).
- How the schemes are written for the GPU:
  [Numerics of the GPU model](wave_numerics.md), in the Reference section.
- Hands-on: [Run the GPU model for 48 h](tutorials/run_gpu.md).

## Every command of the reference runs

[Reproducing](reproducing.md) lists every step as it was run, from the
raw data to the January 2026 runs and the calibration, with the job
scripts, their options, the job numbers and the times. The tutorials pick
from it the commands an ordinary run needs.

```{toctree}
:hidden:
:maxdepth: 2

reproducing
wave_model
```
