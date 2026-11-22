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
  runs 48 h of grid v5 $K=4$ in 32 s on one A100, against 813 s for WW3 on
  32 cores. It stays within 1 mm of WW3 in $H_s$ at the median node.
  Differences at about fifty coast nodes are still being diagnosed.
- **Next: milestone M2a**, the simplified ST4 source terms.

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
| Source terms (ST4, DIA, LN1, DB1, BT4) | `W3SRCE` and its routines | — | M2a, M2b |

## Crate layout

| Module | GPU | Role |
|---|---|---|
| `grid` | no | read an HWW3 grid (`raw/*`, `ww3_grid.nml`); cut a lon/lat box |
| `geom` | no | what WW3 builds for the PDLIB scheme: normals, `PDLIB_SI`, `IOBP`, `IOBPA`, `IOBDP`, `IOBPD`, velocities, `ITER` |
| `spec` | no | spectral grid of `W3GRID`; `WAVNU3` |
| `ww3io` | no | big-endian WW3 files: restart, forcing fields, boundary spectra |
| `reference` | no | `PDLIB_W3XYPFSN2` on the CPU in double precision, as WW3 writes it |
| `cpu` | no | CPU transcriptions of `W3KTP3` and `PDLIB_W3XYPUG` |
| `backend` | no | the `Sweeper` trait shared by the GPU backends |
| `kernels` | CubeCL | N-scheme sub-step and its transpose, three variants |
| `device` | CubeCL | the grid on the device; `Sweeper` for CubeCL |
| `burn_ext` | CubeCL | a sub-step as an operation of Burn's autodiff tape |
| `prop` | CubeCL | the kernels of the propagation-only model |
| `model` | CubeCL | the time loop of `W3WAVE` with the source terms off |
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

### `hww3-prop` (milestone M1)

```text
hww3-prop run <grid dir> --restart FILE [--current FILE] [--level FILE] [--ice FILE]
          [--nest FILE] [--start YYYYMMDDHHMMSS] --hours H --out DIR
          [--f64] [--block B] [--itstep0 N] [--no-cx] [--no-cth] [--no-ck]
          [--host-level] [--profile]
hww3-prop restart-hs <grid dir> --restart FILE --out FILE.f32
hww3-prop check-ktp3 <grid dir> --restart FILE [forcing as for run]
hww3-prop check-geo  <grid dir> --restart FILE [forcing as for run]
```

| Option | Meaning |
|---|---|
| `--restart` | WW3 restart: `FULL` (from `ww3_shel`) or `COLD` (from `ww3_strt`, then `--start` is needed) |
| `--current`, `--level`, `--ice` | forcing files of `ww3_prnc` (`current.ww3`, `level.ww3`, `ice.ww3`) |
| `--nest` | open-boundary spectra of `ww3_bounc` (`nest.ww3`) |
| `--no-cx`, `--no-cth`, `--no-ck` | switch off the geographic propagation, refraction or the frequency shift (`RUN%FLCX`, `RUN%FLCTH`, `RUN%FLCK`) |
| `--f64` | double precision |
| `--host-level` | water-level update on the host instead of the device |
| `--profile` | synchronise after each part of a step and report the time of each part |

`run` writes hourly $H_s$ and $T_{m02}$ to `DIR/hs_YYYYMMDDHH.f32` and
`DIR/t02_YYYYMMDDHH.f32` (little-endian `f32`, one value per node, −999.9
where `MAPSTA <= 0`) and a `DIR/summary.json`. `restart-hs` gives $H_s$
and $T_{m02}$ of the restart spectra, to check the reader. `check-ktp3`
and `check-geo` compare one `W3KTP3` call and one geographic step on the
GPU with the CPU transcriptions. The job scripts are
`scripts/m1_prop_hpc.slurm` (GPU) and `scripts/m1_ww3_prop_hpc.slurm`
(Fortran reference); see [Reproducing](reproducing.md).
