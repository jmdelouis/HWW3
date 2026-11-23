# GPU wave model (`hww3-wave`)

`hww3-wave` is a Rust crate that rewrites, for one GPU, the parts of
WAVEWATCH III that the HWW3 explicit configuration runs. Its purpose is a
**differentiable WW3**: a model that sits inside a learning chain and
returns the gradient of a misfit to altimeters and buoys with respect to
the forcing fields and the model parameters. The design, the cost and the
milestones are in [Differentiable WW3](differentiable.md).

WW3 itself is unchanged: the Fortran runs of the previous pages are the
reference that every part of the GPU model is checked against.

**Status (22 November 2026).**

- **Milestone M0 (done, go).** The explicit N-scheme sweep of the
  geographic propagation and its transpose run on one A100: 0.93 ms per
  frequency on grid v4. The dot-product tests pass to round-off, and a
  sub-step is an operation of Burn's autodiff tape.
- **Milestone M1 (targets met).** The propagation-only model `hww3-prop`
  runs 48 h of grid v5 $K=4$ in 65 s on one A100, against 813 s for WW3 on
  32 cores. It stays within 1.5 µm of WW3 in $H_s$ at the median node and
  0.71 mm at the worst node, once the sub-cycling count includes the coast
  nodes as WW3 does.
- **Milestone M2a (targets met)**, the simplified ST4 source terms: over
  48 h and over 1–8 January the GPU model stays within 2.5 µm of WW3 in
  $H_s$ at the median node and 0.11 mm at the 99th percentile, and its
  altimeter and buoy scores are WW3's, at 146 s per model day on one A100.
- **Milestone M2b (targets met)**, the calibrated full ST4 (stress tables,
  cumulative breaking term): against the explicit runs of grids v5 $K=4$
  and v4, 2.2 and 3.6 µm at the median node, the same altimeter and buoy
  scores, at 124 and 220 s per model day on one A100 (target 430 s on grid
  v4).

Details and numbers are in [GPU model results](wave_results.md), the
schemes in [Numerics of the GPU model](wave_numerics.md).

## What is implemented

| Component | WW3 routine | Module | Milestone |
|---|---|---|---|
| Geometry, boundary flags, sub-cycling count | `PDLIB_INIT`, `SET_UG_IOBP_PDLIB_INIT`, `PDLIB_W3XYPFSN2` (`LCALC`) | `geom`, `prop` | M0, M1 |
| Explicit N scheme, geographic propagation | `PDLIB_W3XYPFSN2`, `PDLIB_W3XYPUG` | `kernels`, `prop` | M0, M1 |
| Transpose of the geographic sub-step | (none in WW3) | `kernels`, `cuda_c`, `burn_ext` | M0 |
| Refraction and frequency shift | `W3KTP3`, `W3QCK1`, `W3QCK2` | `prop` | M1 |
| Currents: time interpolation and gradients | `W3UCUR`, `UG_GRADIENTS` | `prop`, `model` | M1 |
| Water levels: depth, wavenumbers, spectrum remapping | `W3ULEV`, `WAVNU3` | `prop`, `spec`, `model` | M1 |
| Ice as an obstruction (IC0) | `W3UICE` | `model` | M1 |
| Open-boundary spectra | `W3IOBC`, `W3UBPT`, `APPLY_BOUNDARY_CONDITION_VA` | `ww3io`, `prop`, `model` | M1 |
| Outputs $H_s$, $T_{m02}$ | `W3OUTG` | `prop` | M1 |
| Restart, forcing and boundary files | `W3IORS`, `W3FLDG`, `W3IOBC` | `ww3io` | M1 |
| Mean parameters, u* (`SINTABLE = 0`: Newton iterations) | `W3SPR4`, `CALC_USTAR` | `sources` (CPU, checked), `srce` | M2a |
| Wind input, swell damping, tail stress in line | `W3SIN4` | `sources` (CPU, checked), `srce` | M2a |
| Wind forcing: time interpolation (`WNT2`), wind relative to the current (`RWND`) | `W3UWND` | `srce`, `model` | M2a |
| Whitecapping without the cumulative term (`SDSCUM = 0`) | `W3SDS4` | `sources` (CPU, checked), `srce` | M2a |
| Nonlinear interactions (DIA), linear input, depth breaking, bottom friction | `W3SNL1`, `W3SLN1`, `W3SDB1`, `W3SBT4` | `sources` (CPU, checked), `srce` | M2a |
| Integration: dynamic step, limiter, tail, `FHMAX`, ice scaling | `W3SRCE` | `sources` (CPU, checked), `srce` | M2a |
| Source terms in the time loop, after the propagation | `W3WAVE` | `model` | M2a |
| Stress tables (`SINTABLE = 1`): u* from `TAUT`, tail stress from `TAUHFT2` | `INSIN4`, `CALC_USTAR`, `W3SIN4` | `sources` (CPU, checked), `srce` | M2b |
| Directional cumulative breaking term (`SDSCUM < 0`, `CUMULW`), turbulent and viscous terms | `W3SDS4`, `INSIN4` | `sources` (CPU, checked), `srce` | M2b |
| Unit harness of the source terms | (Fortran program on `libww3.a`) | `harness/src_harness.F90`, `harness` | M2a |
| Stress tables for any `BETAMAX`, without `ww3_grid`: `TAUT`, `TAUHFT`, `TAUHFT2` and the derivative of the last two | `TABU_STRESS`, `TABU_TAUHF`, `TABU_TAUHF2` (`INSIN4`) | `sources` (`tabu_stress`, `tabu_tauhf2`, `rebuild_stress_tables`), `hww3-src params`, `hww3-prop --betamax` | calibration |
| Tangent of the model (forward mode) for the wind, a source parameter or the initial spectra | (none in WW3) | `srce_tl`, `prop_tl`, `model` | M3 |

## Crate layout

| Module | GPU | Role |
|---|---|---|
| `grid` | no | read an HWW3 grid (`raw/*`, `ww3_grid.nml`); cut a lon/lat box |
| `geom` | no | what WW3 builds for the PDLIB scheme: normals, `PDLIB_SI`, `IOBP`, `IOBPA`, `IOBDP`, `IOBPD`, velocities, `ITER` |
| `spec` | no | spectral grid of `W3GRID`; `WAVNU3` |
| `ww3io` | no | big-endian WW3 files: restart (read and write), forcing fields, boundary spectra |
| `reference` | no | `PDLIB_W3XYPFSN2` on the CPU in double precision, as WW3 writes it |
| `cpu` | no | CPU transcriptions of `W3KTP3` and `PDLIB_W3XYPUG` |
| `harness` | no | reader of the WW3 source-term harness: inputs, outputs and parameters of the source terms |
| `sources` | no | CPU transcriptions in double precision of the source terms (`W3SPR4`, `CALC_USTAR`, `W3SIN4`, `W3SLN1`, `W3SNL1`, `W3SDS4`, `W3SDB1`, `W3SBT4`) and of the `W3SRCE` integration |
| `backend` | no | the `Sweeper` trait shared by the GPU backends |
| `kernels` | CubeCL | N-scheme sub-step and its transpose, three variants |
| `device` | CubeCL | the grid on the device; `Sweeper` for CubeCL |
| `burn_ext` | CubeCL | a sub-step as an operation of Burn's autodiff tape |
| `prop` | CubeCL | the kernels of the propagation-only model |
| `srce` | CubeCL | the source kernel: the whole `W3SRCE` integration of a global step, one cube per node with the spectrum in shared memory; the wind interpolation of `W3UWND` |
| `srce_tl` | CubeCL | the source kernel with its tangent (forward mode) |
| `prop_tl` | CubeCL | the tangents of `W3KTP3`, of the geographic node update, of the level remap, of the wind and of the outputs |
| `ww3grid` | no | the source-term part of `ww3_grid`: the parameter file of the source terms from the namelists and the grid |
| `model` | CubeCL | the time loop of `W3WAVE`, propagation only or with the source terms |
| `cuda_c` | CUDA C | the M0 kernels in CUDA C, through `cudarc` and NVRTC |

Two programs come with the crate: `hww3-wave` (milestone M0: geometry,
timings, dot-product tests) and `hww3-prop` (milestone M1: the
propagation-only model). The full API is in the
<a href="rust-wave/hww3_wave/index.html">Rust API reference of hww3-wave</a>.

## Backends and platforms

| Backend | Cargo feature | Kernels | Runs on |
|---|---|---|---|
| CubeCL with Burn | `cubecl` (default) | written in Rust (`#[cube]`), compiled at first launch by the NVRTC of the CUDA toolkit | Ifremer hpc cluster: A100-SXM4-80GB, CUDA 13.0, RHEL 9.4 |
| CUDA C | `cuda-c` | CUDA C strings compiled by NVRTC, launched through `cudarc` | Datarmor A100 nodes (driver limited to CUDA 11.4) |

CubeCL 0.11 depends on `cubecl-llvm`, whose build script downloads an LLVM
bundle that needs glibc 2.34 or later: it does not build on Datarmor
(glibc 2.19). The CUDA C backend covers M0 only; M1 needs CubeCL.

## Build

On the Ifremer hpc cluster, with the default feature:

```bash
module load cuda/13.0.0
cd hww3-wave
CARGO_TARGET_DIR=/scratch/jmdeloui/HWW3/target_wave cargo build --release
```

On Datarmor, CUDA C backend, with the NVRTC of CUDA 11.4 that the driver of
the GPU nodes accepts:

```bash
CUDARC_CUDA_VERSION=11040 CARGO_TARGET_DIR=/home1/scratch/jmdeloui/HWW3/target_wave_c \
    cargo build --release --no-default-features --features cuda-c
```

The modules without a GPU build and test anywhere:

```bash
cargo test --no-default-features      # unit tests, no GPU needed
```

The CI also type-checks and documents the GPU backends on a machine
without CUDA or LLVM; see [Tests and continuous integration](development.md).

## Running

### `hww3-wave` (milestone M0)

```text
hww3-wave info  <grid dir> [--box lon0 lon1 lat0 lat1]
hww3-wave bench <grid dir> [--box lon0 lon1 lat0 lat1] [--backend cubecl|cuda-c]
                [--reps N] [--block B] [--f64] [--reference] [--burn] [--out results.json]
```

`info` prints the geometry and the sub-cycling of a grid without a GPU.
`bench` times the three kernel variants, runs the dot-product tests of the
transpose (`--reference` adds the check against the CPU loop of WW3,
`--burn` the test through Burn's `backward()`) and writes a JSON summary.
The job scripts are `scripts/wave_m0_hpc.slurm` (CubeCL, hpc) and
`scripts/wave_m0.pbs` (CUDA C, Datarmor).

### `hww3-prop` (milestones M1 and M2a)

```text
hww3-prop run <grid dir> --restart FILE|calm [--current FILE] [--level FILE] [--ice FILE]
          [--nest FILE] [--wind FILE --sources PARAMS] [--start YYYYMMDDHHMMSS]
          --hours H|--stop YYYYMMDDHHMMSS --out DIR [--restart-out FILE]
          [--f64] [--block B] [--units U] [--itstep0 N]
          [--no-cx] [--no-cth] [--no-ck]
          [--host-level] [--no-bccfl] [--profile]
hww3-prop restart-hs <grid dir> --restart FILE --out FILE.f32
hww3-prop restart-copy RESTART --out FILE
hww3-prop check-ktp3 <grid dir> --restart FILE [forcing as for run]
hww3-prop check-geo  <grid dir> --restart FILE [forcing as for run]
hww3-prop check-steps <grid dir> --restart FILE [forcing as for run]
          --ref RESTART1,RESTART2,... [--nodes N1,N2,...] [--no-cth] [--no-ck]
          [--no-bccfl] [--dump DIR]
```

| Option | Meaning |
|---|---|
| `--restart` | WW3 restart: `FULL` (from `ww3_shel` or `--restart-out`) or `COLD` (from `ww3_strt`, then `--start` is needed); `calm`: zero spectra at `--start`, as WW3 starts without a restart file |
| `--stop` | stop time instead of `--hours` |
| `--restart-out` | write the state at the end as a `FULL` WW3 restart; continue with `--itstep0` set to the `ITSTEP` printed |
| `--current`, `--level`, `--ice` | forcing files of `ww3_prnc` (`current.ww3`, `level.ww3`, `ice.ww3`) |
| `--nest` | open-boundary spectra of `ww3_bounc` (`nest.ww3`) |
| `--wind`, `--sources` | wind of `ww3_prnc` (`wind.ww3`) and the parameter file of the source-term harness (`*.params`, from the `mod_def.ww3` of the run): the source terms of the simplified ST4 run after the propagation (milestone M2a) |
| `--units` | units per cube of the source kernel (default 128) |
| `--no-cx`, `--no-cth`, `--no-ck` | switch off the geographic propagation, refraction or the frequency shift (`RUN%FLCX`, `RUN%FLCTH`, `RUN%FLCK`) |
| `--f64` | double precision |
| `--host-level` | water-level update on the host instead of the device |
| `--no-bccfl` | count the interior nodes only in `ITER` (`FSBCCFL = F`); by default every wet node counts, as WW3 does with `UGBCCFL = T` |
| `--profile` | synchronise after each part of a step and report the time of each part |
| `--scale-par NAME:F+...`, `--wind-factor F`, `--init-factor F` | multiply `SDSC2`, `NLPROP` or `BJGAM`, the wind speed of the forcing or the restart spectra by a factor (in double precision) |
| `--betamax B` | `BETAMAX` with the stress tables `TAUT`, `TAUHFT`, `TAUHFT2` computed for it by the Rust port instead of those of the parameter file |
| `--tangent NAME` | the tangent of the model for a relative change of `WIND`, `BETAMAX`, `SDSC2`, `NLPROP`, `BJGAM` or `INIT` (the restart spectra): `dhs_` and `dt02_` files and the tangent of the total energy (milestone M3) |
| `--out64` | also write $H_s$, $T_{m02}$ and their tangents in double precision (`.f64`) |

`run` writes hourly $H_s$ and $T_{m02}$ to `DIR/hs_YYYYMMDDHH.f32` and
`DIR/t02_YYYYMMDDHH.f32` (little-endian `f32`, one value per node, −999.9
where `MAPSTA <= 0`) and a `DIR/summary.json`. `restart-hs` gives $H_s$
and $T_{m02}$ of the restart spectra, to check the reader, and
`restart-copy` reads a restart and writes it back, to check the writer.
`check-ktp3`
and `check-geo` compare one `W3KTP3` call and one geographic step on the
GPU with the CPU transcriptions. `check-steps` steps from the restart and
compares the full spectra with WW3 restarts written at the end of some
steps: $H_s$ differences by node class and by distance to the coast, the
worst nodes, and for the nodes of `--nodes` the frequencies and directions
that differ, with `IOBPD`, `ITER` and the local CFL number; `--dump`
writes the depth and currents of those steps. The job scripts are
`scripts/m1_prop_hpc.slurm` (GPU) and `scripts/m1_ww3_prop_hpc.slurm`
(Fortran reference); see [Reproducing](reproducing.md).

### `hww3-src` (milestone M2a)

```text
hww3-src check HARNESS        (reads HARNESS and HARNESS.params)
hww3-src check-gpu HARNESS [--f64] [--units N] [--replicate R]
hww3-src check-tl HARNESS [--units N] [--h H]
hww3-src tables BASE.params [OTHER.params ...]
hww3-src params BASE.params OUT.params [--betamax B] [--scale-par NAME:F+...]
hww3-src build-params GRID_DIR NAMELISTS.nml OUT.params [--grid-nml WW3_GRID.nml] [--check BASE.params]
```

Runs the CPU transcriptions of the source terms (`sources`) on the inputs of
the WW3 source-term harness (`harness/src_harness.F90`, built and run by
`scripts/m2_src_harness_hpc.slurm`) and prints, for every output, the
largest difference over the nodes. Each routine gets the inputs WW3 had;
then the whole `W3SRCE` integration over the global step runs from the same
state. It also counts the nodes where the wind-sea flags, the
depth-breaking flag or the limiter decision differ from WW3. It needs no
GPU. `check-gpu` runs the GPU source kernel (`srce`) on the same nodes
and compares the spectra after the step with WW3 and with the CPU
transcription; `--replicate R` repeats every node `R` times for the
timing (`scripts/m2_src_gpu_hpc.slurm`). `check-tl` compares the tangent
kernel `srce_tl` with central differences of `srce` (milestone M3).

`tables` computes the stress tables of `INSIN4` for the `BETAMAX` of each
parameter file and compares them with those WW3 wrote there: `TAUT`
(`TABU_STRESS`, iterated in single precision as WW3 does) and `TAUHFT`,
`TAUHFT2` (`TABU_TAUHF`, `TABU_TAUHF2`, in double precision, with their
derivative with respect to `BETAMAX`). `params` writes the parameter file
of other values of `BETAMAX` (its tables computed here), `SDSC2`, `NLPROP`
and `BJGAM`, copying everything else from `BASE`: no other entry of the
file depends on these four parameters (the saturation weights,
`CUMULW` and the DIA coefficients do not), so a new value needs neither
`ww3_grid` nor the harness. `hww3-prop --betamax` builds the same tables in
memory. `build-params` builds the whole parameter file from the namelists
of the run and the grid (`ww3grid`, the source-term part of `ww3_grid`),
and with `--check` compares it record by record with a harness file.
