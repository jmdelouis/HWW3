# hww3-wave: milestone M0 of the differentiable WW3

Milestone M0 of [`docs/differentiable.md`](../docs/differentiable.md)
(section 9, and section 4 for the bandwidth estimate): the explicit N scheme
of the WW3 PDLIB geographic propagation (`PDLIB_W3XYPFSN2` in
`w3profsmd_pdlib.F90`) and its transpose, as GPU kernels on the HWW3 grids,
timed on one A100, with dot-product tests.

## Results (21–22 November 2026)

### Toolchain

- **Datarmor.** CubeCL 0.11.4, the version Burn 0.22.1 needs, does not
  build. Its CUDA backend depends on `cubecl-llvm`, whose build script
  downloads a prebuilt LLVM 23 (`tracel-llvm-bundler`, 323 MB in
  `~/.local/share/tracel`, removed since). That `llvm-config` needs
  glibc 2.34 and Datarmor has glibc 2.19, so CubeCL and Burn cannot be used
  on Datarmor.
- **Datarmor GPU driver.** The A100-SXM4-80GB nodes run driver 470.239.06,
  which supports CUDA 11.4 at most. The PTX of NVRTC 12.1 is rejected
  (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, PBS job 7964674). The fallback
  backend therefore compiles its kernels with the NVRTC of the cuda/11.4u1
  module (`CUDARC_CUDA_VERSION=11040`).
- **Fallback on Datarmor.** The CUDA C kernels, compiled by NVRTC and
  launched through `cudarc` 0.19.10, build with stable Rust 1.99.1 and run
  (PBS jobs 7964684, 7964764, 7964774).
- **CubeCL and Burn on the Ifremer hpc cluster.** CubeCL 0.11.4 and
  Burn 0.22.1 build there: RHEL 9.4, glibc 2.34, Slurm, A100, cuda/13.0.0
  module. Rust 1.99.1 was copied from Datarmor, and the LLVM bundle was
  downloaded at build time, as agreed with Jean-Marc. They also run there
  (Slurm job 4115275) on an A100-SXM4-80GB with driver 595.71.05. It is
  the same GPU model, with the same 1,593 MHz memory clock, as the Datarmor
  nodes.
  - CubeCL's double precision works.
  - CubeCL compiles each kernel at its first launch.
  - Burn's autodiff tape takes a sub-step as one of its operations, with
    the transpose kernel as its backward.

### Sub-cycling

`ITER(f, θ)` follows WW3 (`LCALC` branch of `PDLIB_W3XYPFSN2`, degrees,
interior nodes). `S` is the sum over the components of the sub-steps per
600 s step. A frequency sweep is one sub-step of all directions of one
frequency, and a step needs `Σ_f max_θ ITER` of them.

| grid | nodes | triangles | interior / coast / open | S (`explicit_cost.py`) | sweeps per step | ITER at f1 |
|---|---|---|---|---|---|---|
| Channel–Biscay box of v4 | 21,425 | 41,313 | 19,872 / 1,358 / 195 | 3,079 (3,080) | 92 | 6–8 |
| v4 | 794,613 | 1,531,542 | 734,829 / 58,443 / 1,341 | 3,678 (3,678) | 110 | 8–10 |
| v5 K=4 | 424,123 | 807,832 | 382,296 / 40,775 / 1,052 | 2,837 (2,835) | 83 | 4–5 |

A coastal node propagates 18.9 of the 36 directions on average.

### Sweep times, CUDA C fallback, one Datarmor A100 (PBS job 7964774)

Three kernel variants (single precision, 256 units per block; times are
means over 200 sweeps, 400 for the box, of frequency 1 with all directions
active):

- `fused`: one unit per (node, direction), recomputing `K`, `NM` and
  `FLALL` of the ~6 triangles around the node;
- `two_phase`: twelve coefficients per triangle and frequency
  (`K` and the Simpson part of `FLALL` are linear in cos θ and sin θ),
  one pass over (triangle, direction) for `Ũ_T`, then the node gather;
- `two_phase_x4`: the same with four directions per unit, so the triangle
  decoding and the coefficients are loaded once for four directions.

| grid | variant | forward sweep | transpose sweep | 600 s step, forward / transpose | forward s per model day |
|---|---|---|---|---|---|
| v4 | fused | 2.065 ms | 2.695 ms | 0.227 / 0.296 s | 32.7 |
| v4 | two_phase | 1.421 ms | 1.598 ms | 0.157 / 0.176 s | 22.5 |
| v4 | **two_phase_x4** | **0.947 ms** | **1.067 ms** | **0.105 / 0.118 s** | **15.1** |
| v5 K=4 | fused | 1.103 ms | 1.436 ms | 0.091 / 0.119 s | 13.1 |
| v5 K=4 | two_phase | 0.758 ms | 0.851 ms | 0.063 / 0.071 s | 9.1 |
| v5 K=4 | **two_phase_x4** | **0.502 ms** | **0.572 ms** | **0.042 / 0.047 s** | **6.0** |
| box | two_phase_x4 | 0.033 ms | 0.035 ms | 3.0 / 3.2 ms | 0.43 |

- **Bandwidth estimate** of section 4 for one frequency sweep on v4: about
  794,613 × 800 B at 1 TB/s, i.e. 0.64 ms. **Go threshold:** 1.3 ms.
- **v4:** the fused gather takes 2.07 ms, the two-phase kernels 1.42 ms
  and the four-directions variant **0.95 ms, under the threshold**
  (1.5 times the estimate).
- **v5 K=4:** 0.50 ms against an estimate of 0.34 ms.
- **What limits the kernels.** The bytes a sweep must move at least (state
  read and written, nodal and triangle arrays read once) give 156, 226 and
  339 GB/s for the three variants on v4. The kernels are bound by
  instruction count and gather latency, not by DRAM bandwidth.
- **Coefficients.** Computing the per-triangle coefficients of all 32
  frequencies takes 0.012 s on v4. That is 11 % of a step if the
  velocities change at every step (currents, water level). The CubeCL
  figure, 0.028 s, also includes compiling the kernel at its first launch.
- **Double precision.** It costs 1.9 times single precision with
  `two_phase_x4` (1.78 ms on v4) and 1.5 times with the fused kernel.
- **Model-day figures.** They cover the geographic propagation only (144
  steps); WW3's spectral propagation and source terms are not included.
- **Node numbering.** Edges join nodes whose indices differ by 3 at the
  median and by under 3,703 for 99 % of them, so renumbering the nodes
  was not tried.

### Sweep times, CubeCL kernels with Burn, one hpc A100 (Slurm job 4115275)

The CubeCL kernels (`kernels.rs`) are the same three variants, written in
Rust. They run on the CubeCL client of Burn's `Device::cuda(0)` and are
compiled by the NVRTC of CUDA 13.0. Measurements as above; the CUDA C
times are those of PBS job 7964774.

| grid | variant | forward sweep, CubeCL / CUDA C | transpose sweep, CubeCL / CUDA C | 600 s step (CubeCL), forward / transpose | forward s per model day (CubeCL) |
|---|---|---|---|---|---|
| v4 | fused | 1.946 / 2.065 ms | 2.690 / 2.695 ms | 0.214 / 0.296 s | 30.8 |
| v4 | two_phase | 1.259 / 1.421 ms | 1.554 / 1.598 ms | 0.139 / 0.171 s | 20.0 |
| v4 | **two_phase_x4** | **0.931** / 0.947 ms | **1.029** / 1.067 ms | **0.103 / 0.114 s** | **14.8** |
| v5 K=4 | fused | 1.036 / 1.103 ms | 1.434 / 1.436 ms | 0.086 / 0.119 s | 12.4 |
| v5 K=4 | two_phase | 0.673 / 0.758 ms | 0.830 / 0.851 ms | 0.056 / 0.069 s | 8.0 |
| v5 K=4 | **two_phase_x4** | **0.515** / 0.502 ms | **0.583** / 0.572 ms | **0.043 / 0.048 s** | **6.2** |
| box | two_phase_x4 | 0.033 / 0.033 ms | 0.034 / 0.035 ms | 3.0 / 3.1 ms | 0.43 |

- **CubeCL against CUDA C, single precision, v4 and v5 K=4.** The CubeCL
  sweeps are at most 3 % slower than the CUDA C sweeps (`two_phase_x4` on
  v5 K=4) and up to 11 % faster (`two_phase` forward, 1.26 against
  1.42 ms on v4). Writing the kernels in Rust with CubeCL costs nothing
  measurable here.
- **CubeCL against CUDA C, double precision.** On v4 the sweeps are within
  2 % of CUDA C, except the `two_phase` transpose (8 % slower). On
  v5 K=4 the CubeCL f64 runs used 512 units per block, carried over from
  the f32 `two_phase_x4` timing, against 256 for CUDA C. There the
  forward sweeps are within 5 % but the transposes are 9–33 % slower
  (fused 3.00 against 2.25 ms). The block size of the f64 kernels was not
  tuned.
- **Go threshold met with CubeCL:** **0.93 ms** on v4 with
  `two_phase_x4`, against 1.3 ms.
- **Double precision with CubeCL:** 1.77 ms on v4 and 0.98 ms on v5 K=4
  with `two_phase_x4`, 1.9 times single precision, as with CUDA C
  (1.78 and 0.94 ms).

### Dot-product tests

`⟨L δx, ȳ⟩ = ⟨δx, Lᵀ ȳ⟩` with `δx` uniform in [0, 1] and `ȳ` in [−1, 1],
products summed in double precision on the CPU. `L` is one sub-step of
frequency 1, or a 600 s step (`max_θ ITER` sub-steps, each direction with
its own `ITER`) of frequencies 1, 17 and 32. The table gives
`|a − b| / max(|a|, |b|)` and the normwise error
`|a − b| / (‖L δx‖ ‖ȳ‖)`, worst case over the four operators and the three
variants:

| grid | backend | f32 relative | f32 normwise | f64 relative | f64 normwise |
|---|---|---|---|---|---|
| box | CUDA C | 2.1e-7 | 2.2e-10 | 4.2e-14 | 4.4e-17 |
| box | CubeCL | 1.5e-7 | 1.6e-10 | 4.3e-14 | 4.6e-17 |
| v4 | CUDA C | 1.7e-7 | 2.9e-11 | 1.4e-12 | 9.6e-17 |
| v4 | CubeCL | 1.7e-7 | 2.9e-11 | 1.4e-12 | 9.7e-17 |
| v5 K=4 | CUDA C | 3.1e-6 | 1.4e-10 | 4.6e-13 | 5.2e-17 |
| v5 K=4 | CubeCL | 1.3e-6 | 5.9e-11 | 4.4e-13 | 5.2e-17 |

- **Single precision:** the tests pass to round-off. The 3.1e-6 and
  1.3e-6 on v5 K=4 come from cancellation in `⟨L δx, ȳ⟩` for the 600 s
  step of frequency 1; their normwise errors are 1.4e-10 and 5.9e-11.
- **Double precision:** the tests pass to 1e-12 or better.
- **Through Burn's autodiff.** The test is also run with `L` (one sub-step
  of frequency 1) as an operation of Burn's tape:
  - `x` is a Burn tensor with `require_grad`, the loss is
    `sum(L(x) ȳ)`, and `backward()` gives `Lᵀ ȳ`;
  - the test passes with a relative error of 3.8e-8 on v4 (normwise
    1.2e-11) and 1.0e-7 on the box;
  - the gradient Burn returns is identical, bit for bit, to the output of
    the transpose kernel launched directly.

### Check against WW3's loop

`reference.rs` runs `PDLIB_W3XYPFSN2` on the CPU in double precision, as
WW3 writes it (loop over triangles with scatter, `max(0, .)` clip). It is
compared with the GPU kernels after a 600 s step of frequencies 1 and 32,
from a positive random state:

- f32: largest difference 1.6e-6 of the largest value, rms 3e-8 or less;
- f64: largest difference 2.4e-15;
- these bounds hold for all variants and grids, with both backends.

### Verdict for M0

**Go.** The three criteria of section 9 are met.

1. **The kernels run on one A100.** CubeCL kernels, driven through Burn,
   run the N-scheme sweep and its transpose on one A100 of the hpc
   cluster (Slurm job 4115275). On Datarmor, where CubeCL cannot be built,
   the same kernels in CUDA C run through `cudarc` (PBS job 7964774).
2. **The sweep is under the threshold.** One frequency sweep on grid v4
   takes **0.93 ms** in single precision with CubeCL (0.95 ms with
   CUDA C), under the 1.3 ms threshold and 1.5 times the bandwidth
   estimate of 0.64 ms. On grid v5 K=4 it takes 0.51 ms (0.50 ms).
3. **The dot-product tests pass.** They pass to single-precision
   round-off (normwise ≤ 2.2e-10) and, since CubeCL's f64 works, in
   double precision to 1.4e-12 or better. The test also passes through
   Burn's `backward()`, which returns the transpose kernel's output
   exactly.

**What this means for the next milestones:**

- **Cost of the geographic propagation alone** (f32, `two_phase_x4`):
  - forward, per model day: 14.8 s on v4 and 6.2 s on v5 K=4;
  - transpose, per model day: 16.4 s and 7.0 s;
  - recomputing the coefficients at every step, if the currents or the
    water level change, adds about 11 %.

  These figures do not include the spectral propagation, the source terms
  or the checkpointing.
- **Platform.** CubeCL and Burn, the choice of the design note, need the
  hpc cluster: Datarmor has glibc 2.19, and its A100 driver supports
  CUDA 11.4 at most. The CUDA C backend stays as a cross-check and as the
  way to use the Datarmor A100s.
- **Left for M3.** The forward kernel clips negative values (`max(0, ·)`)
  and the transpose here is that of the linear part. The adjoint of the
  full step must also apply the clip mask, which it can rebuild from the
  stored forward state.
- **Not tuned.** The block size of the f64 kernels, and the transposes,
  which are 10–14 % slower than the forward sweeps with `two_phase_x4` on
  v4 and v5 K=4.

## Milestone M1, propagation-only model (in progress, 22 November 2026)

`hww3-prop` runs the time loop of `W3WAVE` with the source terms off
(`RUN%FLSOU = F`) on one GPU, from a WW3 restart, with the currents, water
levels, ice and boundary spectra of a WW3 run:

- `ww3io.rs` reads the big-endian WW3 files: restart (`W3IORS`), forcing
  fields (`W3FLDG`) and boundary spectra (`W3IOBC`);
- `spec.rs` holds the spectral constants of `W3GRID` and `WAVNU3`;
- `prop.rs` holds the CubeCL kernels:
  - the geographic N-scheme sweep with the current terms and open-boundary
    values re-imposed after every sub-iteration;
  - `ITER` recomputed at every step, as `PDLIB_W3XYPFSN2` does with
    currents;
  - `W3KTP3`: refraction and frequency shift with ULTIMATE QUICKEST;
  - `UG_GRADIENTS`, the current interpolation of `W3UCUR`, the spectrum
    remapping of `W3ULEV`, and H_s and T_m02 of `W3OUTG`;
- `model.rs` is the driver. It follows the order of `W3WAVE` and reads the
  forcing as `ww3_shel` does; the Copernicus Marine fields are stamped at
  hh:30, so levels switch on the hour;
- `cpu.rs` transcribes `W3KTP3` (with WW3's `VQ`, `MAPTH2`, `MAPWN2`) and
  `PDLIB_W3XYPUG` (triangle loop with scatter, internals in double) to
  check the kernels.

Checks on grid v5 K=4 from `restart001.ww3` of the explicit run (8 January
2026 00 h), on one hpc A100:

| check | result | Slurm job |
|---|---|---|
| H_s and T_m02 of the restart spectra against the WW3 output at the restart time | 371,278 nodes (52,845 ice nodes excluded), largest difference 2.4e-5 m and 8e-5 s | 4116035 |
| one `W3KTP3` call, GPU against the CPU transcription (20,972 nodes) | largest difference 3.8e-7 of the node maximum; the call changes the state by 2.2 % rms | 4116118 |
| geographic step of frequencies 1, 17, 32 with currents and boundary spectra, GPU against the CPU transcription | largest difference 3.9e-7, rms 1.1e-7, `ITER` identical in all directions | 4116118 |
| 48 h run, first version | 288 steps in 110.5 s, i.e. 55 s per model day; mean H_s 1.45 to 1.00 m | 4116119 |
| 48 h run, current version | 288 steps in 32.0 s, i.e. 16.0 s per model day; H_s within 3e-4 m of the first version | 4117585 |

Speed. Starting from the first 48 h run (110.5 s), each change was checked
against the run before it, on all 49 hourly H_s and T_m02 fields:

| change | 48 h run | H_s against the previous run | Slurm job |
|---|---|---|---|
| `ITER` of all 32 frequencies in one launch (32 launches of 18,432 units before); level update on 8 host threads | 61.8 s | identical (bitwise) | 4117276 |
| `W3KTP3` kernels with a register window of four values instead of local arrays (3 × 36 and 5 × 34 values per unit) | 42.4 s | largest difference 2.9e-4 m, rms 8e-8 m (order of the fused multiply-adds); `check-ktp3` 3.3e-7 | 4117344 |
| `W3ULEV` on the device (kernel `ulev_node`; `--host-level` keeps the host path); energy budget from per-node sums on the device instead of a 2 GB download | 34.3 s, with the first 6 h of the rerun 4117419 (8.3 s; they took 43.2 s once in 4117391) | largest difference 6.3e-5 m, rms 3e-8 m | 4117391 |
| theta kernel staged through shared memory: coalesced loads and stores, rows padded to 37 values | 32.0 s | identical (bitwise) | 4117585 |

The 32.0 s include about 4.5 s of NVRTC compilation of `ulev_node` at the
first level update; after it, 6 model hours take 3.4 s (13.6 s per model
day). A 6 h run synchronised after each part (job 4117586) splits the
time into: geographic sweeps 2.0 s (3,001 sweeps, 0.67 ms each, against
0.52 ms for the M0 sweep without currents or open boundary); `W3KTP3`
1.0 s (theta 0.4 s, k 0.6 s, i.e. 5.6 and 8.3 ms per call); `ITER` 0.4 s;
outputs 0.1 s; forcing 4.9 s, almost all of it the one-time compilation
(the non-synchronised 48 h run spends 2.0 s in forcing in all).

Against the Fortran. `scripts/m1_ww3_prop_hpc.slurm` ran WW3 dea641c
(hpc build, 32 ranks of the `cpu` partition) with `RUN%FLSOU = F` for 48 h
from 8 January 2026 00 h, and `scripts/m1_prop_hpc.slurm` ran `hww3-prop`
with the same `MODE`:

- `prop`: restart001.ww3, currents, levels, ice and boundary spectra;
- `geo`: the same with `RUN%FLCTH = RUN%FLCK = F` (`--no-cth --no-ck`);
- `ideal`: a Gaussian swell packet from `ww3_strt` (0.08 Hz, 25W 47N,
  3° spread, H_s max 3 m), no forcing and no boundary spectra. The COLD
  restart of `ww3_strt` has no time and no masks; `hww3-prop` takes
  `--start` and the grid's `MAPSTA`.

`scripts/m1_compare_nc.py` (no numpy, reads the WW3 netCDF with `ncdump`)
compares the 49 hourly fields. Below, "RMS" is the RMS difference over the
hours at a node; the energy is `Σ_i (H_s,i / 4)²` over the nodes, at the
worst hour:

| mode | nodes | H_s RMS median | H_s RMS p99 | H_s RMS max | energy | Fortran, 32 cores | GPU | jobs (Fortran, GPU) |
|---|---|---|---|---|---|---|---|---|
| prop | 371,278 | 0.94 mm | 6.0 mm | 0.48 m | −0.047 % | 813 s | 32.0 s | 4118215, 4117585 |
| geo | 371,278 | 0.63 mm | 3.1 mm | 0.47 m | −0.018 % | 702 s | 24.7 s | 4118220, 4118253 |
| ideal, nodes reaching H_s 0.5 m | 22,015 | 1.5 mm | 4.8 mm | 7.0 cm | −0.45 % (13 h), +0.12 % at 48 h | 887 s | 32.9 s | 4118221, 4118252 |

The three M1 targets of section 6 (H_s RMS below 1 cm at the median node
and 5 cm at the 99th percentile, energy within 0.5 %) are met in all three
modes. T_m02 differs by 3.2 ms (median) and 0.06 s (p99) in `prop`; in
`ideal` it is not meaningful where the spectrum is nearly empty.

The largest differences are all on coast nodes (`IOBP = 0`), most of them
in a single triangle: in `prop`, 1,074 of 371,278 nodes differ by more than
1 cm RMS, 49 by more than 5 cm and 7 by more than 20 cm (node 169562, Nova
Scotia, 6.1 m: 2.11 against 1.84 m one hour after the restart, 3.52 against
2.94 m at the peak). What was ruled out:

- the masks: `IOBP` and `IOBPD` read from `mod_def.ww3` are identical to
  those of `geom.rs` on all nodes, including the 40,775 coast nodes;
- the depths: `ZB` of `mod_def.ww3` is within 0.8 mm of the grid depths;
- precision: the `geo` run in double precision on the GPU (job 4118461)
  is within 0.6 mm of the single-precision run on every node;
- the code path: WW3 calls `PDLIB_W3XYPUG` per spectral bin with
  `PDLIB_W3XYPFSN2`, recomputes `ITER` when the currents or levels are
  updated (`UGDTUPDATE`), and builds `PDLIB_SI` and `PDLIB_IEN` as
  `geom.rs` does.

Coast nodes are left out of the CFL count, so a single-triangle coast node
can run at a local CFL above 1, where the result depends on the number of
sub-steps much more than elsewhere. A one-hour Fortran run with fields
every 600 s and point spectra at a few of these nodes would show where the
first step differs.

**Still to do:**

- The diagnosis of the coast-node differences above.
- Speed, if needed later: the k kernel of `W3KTP3` runs at about a third
  of the memory bandwidth, and the level kernel could be split to compile
  faster.

## What the crate does

- `grid.rs` reads an HWW3 grid (`raw/{lon,lat,depth}.f64`, `tri.u32`,
  `obc.u32`, `ww3_grid.nml`) and cuts a lon/lat box as
  `scripts/subdomain.py` does (open boundary = boundary nodes of the sub-mesh
  that are interior in the full mesh).
- `geom.rs` builds what WW3 builds, in degrees as WW3 does for a spherical
  grid: counter-clockwise triangles, `PDLIB_IEN` (inward normals
  `n1 = (y2 - y3, x3 - x2)`), `PDLIB_SI`, `IOBP` (0 coast, 1 interior,
  2 open boundary), `IOBPA`, `IOBDP` (depth below `DMIN + 1e-5`), `IOBPD`
  with the `eDet` test of `SET_UG_IOBP_PDLIB_INIT`, the nodal velocities
  `C = (FACX cg cos θ / CLATS, FACY cg sin θ) IOBDP`, and
  `ITER(f, θ) = ceil(DT max_i KKSUM_i / SI_i)` over the interior nodes, as the
  `LCALC` branch of `PDLIB_W3XYPFSN2` does.
- `kernels.rs` (CubeCL) and `cuda_c.rs` (CUDA C, NVRTC, `cudarc`) hold the
  same kernels, in three variants (`fused`, `two_phase`, `two_phase_x4`,
  described with the timings). All are gathers over the triangles around a
  node, with no atomics. The state of a frequency is `[np, nth]`, with the
  directions contiguous.
  - forward: `U_i ← max(0, U_i − DTSI_i Σ_T K⁺_iT (U_i − Ũ_T) (1 − IOBPA_i)) IOBPD IOBDP`,
    `Ũ_T = NM_T Σ_q FLALL_qT U_q`;
  - transpose (linear part, without the clip):
    `z_j = m_j ȳ_j − Σ_{T∋j} [K⁺_jT v_j − W_jT Σ_{i∈T} K⁺_iT v_i]`,
    `v_i = m_i (1 − IOBPA_i) DTSI_i ȳ_i`, `W = NM FLALL`, `m = IOBPD IOBDP`.
  - Each direction keeps its own `ITER(f, θ)` and time step `DT / ITER`:
    a launch is one sub-step of all directions of one frequency (a
    *frequency sweep*), and a direction whose `ITER` is used up is copied.
    One 600 s step of a frequency takes `max_θ ITER(f, θ)` sweeps.
- `reference.rs` is `PDLIB_W3XYPFSN2` in double precision on the CPU,
  written as in WW3 (loop over triangles, scatter into the nodes), to check
  the gather kernels.
- `burn_ext.rs` (CubeCL backend): the states are Burn tensors on
  `Device::cuda(0)`, the kernels run on the CubeCL client and buffers behind
  them (`Tensor::try_into_primitive`), and a sub-step is an operation of
  Burn's autodiff tape (`burn_autodiff::ops::Backward`) whose backward is the
  transpose kernel.
- `main.rs` times the sweeps, runs the dot-product tests
  `⟨L δx, ȳ⟩ = ⟨δx, Lᵀ ȳ⟩` (one sub-step, and the 600 s step of three
  frequencies, i.e. `(L)^ITER`), compares the GPU step with the CPU
  reference, and runs the dot-product test through Burn's `backward()`.

## Build and run

CubeCL and Burn (default feature `cubecl`), on the Ifremer hpc cluster
(RHEL 9.4, glibc 2.34):

```bash
module load cuda/13.0.0
CARGO_TARGET_DIR=/scratch/jmdeloui/HWW3/target_wave cargo build --release
sbatch scripts/wave_m0_hpc.slurm
```

Fallback (feature `cuda-c`), on Datarmor, with the NVRTC of CUDA 11.4 that
the driver of the GPU nodes accepts (`LD_LIBRARY_PATH=/appli/cuda/11.4u1/lib64`
at run time, set in the PBS script):

```bash
CUDARC_CUDA_VERSION=11040 CARGO_TARGET_DIR=/home1/scratch/jmdeloui/HWW3/target_wave_c \
    cargo build --release --no-default-features --features cuda-c
```

```bash
qsub scripts/wave_m0.pbs
```

`hww3-wave info <grid>` prints the geometry and the sub-cycling without a
GPU; `hww3-wave bench <grid> [--box lon0 lon1 lat0 lat1] [--backend cubecl|cuda-c]
[--reps N] [--block B] [--f64] [--reference] [--burn] [--out file.json]`
runs the M0 measurements.
