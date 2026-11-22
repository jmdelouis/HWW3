# GPU model results

Results of milestones M0 and M1 of the [differentiable WW3](differentiable.md),
21–22 November 2026, on A100-SXM4-80GB GPUs: those of the Ifremer hpc
cluster (CubeCL with Burn, CUDA 13.0) and of Datarmor (CUDA C fallback,
CUDA 11.4). Both have the same 1,593 MHz memory clock. Job numbers are
Slurm jobs on hpc and PBS jobs on Datarmor.

## Milestone M0: the N-scheme sweep and its transpose

### Sub-cycling of the test grids

`ITER(f, θ)` follows WW3 (`LCALC` branch of `PDLIB_W3XYPFSN2`). $S$ is the
sum of the sub-steps over the components for one 600 s step; a *frequency
sweep* is one sub-step of all directions of one frequency, and a step
needs $\sum_f \max_\theta \mathrm{ITER}$ of them.

| Grid | Nodes | Triangles | Interior / coast / open | $S$ (`explicit_cost.py`) | Sweeps per step | ITER at $f_1$ |
|---|---|---|---|---|---|---|
| Channel–Biscay box of v4 | 21,425 | 41,313 | 19,872 / 1,358 / 195 | 3,079 (3,080) | 92 | 6–8 |
| v4 | 794,613 | 1,531,542 | 734,829 / 58,443 / 1,341 | 3,678 (3,678) | 110 | 8–10 |
| v5 $K=4$ | 424,123 | 807,832 | 382,296 / 40,775 / 1,052 | 2,837 (2,835) | 83 | 4–5 |

A coast node propagates 18.9 of the 36 directions on average.

### Sweep times

Single precision, means over 200 sweeps (400 for the box) of frequency 1
with all directions active. CubeCL kernels with Burn on one hpc A100
(job 4115275), against the same kernels in CUDA C on one Datarmor A100
(job 7964774):

| Grid | Variant | Forward sweep, CubeCL / CUDA C | Transpose sweep, CubeCL / CUDA C | 600 s step (CubeCL), forward / transpose | Forward, s per model day (CubeCL) |
|---|---|---|---|---|---|
| v4 | `fused` | 1.946 / 2.065 ms | 2.690 / 2.695 ms | 0.214 / 0.296 s | 30.8 |
| v4 | `two_phase` | 1.259 / 1.421 ms | 1.554 / 1.598 ms | 0.139 / 0.171 s | 20.0 |
| v4 | **`two_phase_x4`** | **0.931** / 0.947 ms | **1.029** / 1.067 ms | **0.103 / 0.114 s** | **14.8** |
| v5 $K=4$ | `fused` | 1.036 / 1.103 ms | 1.434 / 1.436 ms | 0.086 / 0.119 s | 12.4 |
| v5 $K=4$ | `two_phase` | 0.673 / 0.758 ms | 0.830 / 0.851 ms | 0.056 / 0.069 s | 8.0 |
| v5 $K=4$ | **`two_phase_x4`** | **0.515** / 0.502 ms | **0.583** / 0.572 ms | **0.043 / 0.048 s** | **6.2** |
| box | `two_phase_x4` | 0.033 / 0.033 ms | 0.034 / 0.035 ms | 3.0 / 3.1 ms | 0.43 |

- **Against the threshold.** The bandwidth estimate of one sweep on grid v4
  is 794,613 × 800 B at 1 TB/s, 0.64 ms; the go threshold was twice that,
  1.3 ms. `two_phase_x4` takes 0.93 ms, 1.5 times the estimate.
- **CubeCL against CUDA C.** In single precision, the CubeCL sweeps are at
  most 3 % slower and up to 11 % faster than the CUDA C ones: writing the
  kernels in Rust costs nothing measurable.
- **Double precision** costs 1.9 times single precision with
  `two_phase_x4` (1.77 ms on v4, 0.98 ms on v5 $K=4$), with both backends.
- **What limits the kernels.** The bytes a sweep must move give 156, 226
  and 339 GB/s for the three variants on v4: the kernels are bound by
  instruction count and gather latency, not by memory bandwidth.
- **Coefficients.** The twelve coefficients per triangle of all 32
  frequencies take 0.012 s on v4, 11 % of a step if the velocities change
  at every step (currents, water levels).
- **Not tuned yet.** The block size of the double-precision kernels, and
  the transposes, 10–14 % slower than the forward sweeps.

These figures cover the geographic propagation alone (144 steps per model
day), not the spectral propagation or the source terms.

### Dot-product tests

$\langle L\,\delta x, \bar y\rangle = \langle \delta x, L^\mathsf{T} \bar y\rangle$
with $\delta x$ uniform in $[0, 1]$ and $\bar y$ in $[-1, 1]$, products
summed in double precision on the CPU. $L$ is one sub-step of frequency 1,
or a 600 s step of frequencies 1, 17 and 32. Worst case over the four
operators and the three variants, as the relative error
$|a - b| / \max(|a|, |b|)$ and the normwise error
$|a - b| / (\lVert L\,\delta x\rVert\,\lVert \bar y\rVert)$:

| Grid | Backend | f32 relative | f32 normwise | f64 relative | f64 normwise |
|---|---|---|---|---|---|
| box | CUDA C | 2.1e-7 | 2.2e-10 | 4.2e-14 | 4.4e-17 |
| box | CubeCL | 1.5e-7 | 1.6e-10 | 4.3e-14 | 4.6e-17 |
| v4 | CUDA C | 1.7e-7 | 2.9e-11 | 1.4e-12 | 9.6e-17 |
| v4 | CubeCL | 1.7e-7 | 2.9e-11 | 1.4e-12 | 9.7e-17 |
| v5 $K=4$ | CUDA C | 3.1e-6 | 1.4e-10 | 4.6e-13 | 5.2e-17 |
| v5 $K=4$ | CubeCL | 1.3e-6 | 5.9e-11 | 4.4e-13 | 5.2e-17 |

The tests pass to round-off; the larger relative errors on v5 $K=4$ come
from cancellation in $\langle L\,\delta x, \bar y\rangle$ for the step of
frequency 1. Through Burn's tape, with $L$ (one sub-step) as an operation
and the loss $\sum L(x)\,\bar y$, `backward()` passes the test with a
relative error of 3.8e-8 on v4, and its gradient is identical, bit for bit,
to the output of the transpose kernel.

### Check against the loop of WW3

`reference.rs` runs `PDLIB_W3XYPFSN2` on the CPU in double precision as
WW3 writes it (triangle loop with scatter, `max(0, ·)` clip). After a 600 s
step of frequencies 1 and 32 from a positive random state, the GPU kernels
agree with it to 1.6e-6 of the largest value in single precision (RMS
3e-8 or less) and to 2.4e-15 in double precision, for all variants and
grids, with both backends.

### Verdict: go

1. The kernels run on one A100: CubeCL with Burn on the hpc cluster, and
   CUDA C through `cudarc` on Datarmor, where CubeCL cannot be built.
2. One frequency sweep on grid v4 takes 0.93 ms, under the 1.3 ms
   threshold.
3. The dot-product tests pass in both precisions, and through Burn's
   `backward()`.

## Milestone M1: the propagation-only model

`hww3-prop` runs the time loop of `W3WAVE` with the source terms off
(`RUN%FLSOU = F`) on one GPU, from a WW3 restart, with the currents, water
levels, ice and boundary spectra of a WW3 run. All checks are on grid
v5 $K=4$, from `restart001.ww3` of the explicit run (8 January 2026 00 h),
on one hpc A100.

### Unit checks against the CPU transcriptions

| Check | Result | Job |
|---|---|---|
| $H_s$ and $T_{m02}$ of the restart spectra against the WW3 output at the restart time | 371,278 nodes (52,845 ice nodes excluded), largest difference 2.4e-5 m and 8e-5 s | 4116035 |
| one `W3KTP3` call, GPU against `cpu::ktp3` (20,972 nodes) | largest difference 3.8e-7 of the node maximum, for a call that changes the state by 2.2 % RMS | 4116118 |
| geographic step of frequencies 1, 17 and 32 with currents and boundary spectra, GPU against `cpu::geo_freq` | largest difference 3.9e-7, RMS 1.1e-7, `ITER` identical in all directions | 4116118 |

### Speed

The first 48 h run took 110.5 s (job 4116119). Each change below was
checked against the run before it on all 49 hourly $H_s$ and $T_{m02}$
fields:

| Change | 48 h run | $H_s$ against the previous run | Job |
|---|---|---|---|
| `ITER` of all 32 frequencies in one launch; level update on 8 host threads | 61.8 s | identical (bitwise) | 4117276 |
| `W3KTP3` kernels with a register window of four values instead of local arrays | 42.4 s | largest difference 2.9e-4 m, RMS 8e-8 m (order of the fused multiply-adds) | 4117344 |
| `W3ULEV` on the device; energy budget from per-node sums on the device | 34.3 s | largest difference 6.3e-5 m, RMS 3e-8 m | 4117391 |
| direction kernel staged through shared memory (coalesced loads and stores) | 32.0 s | identical (bitwise) | 4117585 |

The 32.0 s include about 4.5 s of NVRTC compilation at the first level
update; after it, 6 model hours take 3.4 s, i.e. 13.6 s per model day. A
6 h run synchronised after each part (job 4117586) spends 2.0 s in the
geographic sweeps (3,001 sweeps of 0.67 ms, against 0.52 ms for the M0
sweep without currents or open boundary), 1.0 s in `W3KTP3`, 0.4 s in
`ITER`, 0.1 s in the outputs and 4.9 s in the forcing, almost all of it the
one-time compilation.

### Against WW3

`scripts/m1_ww3_prop_hpc.slurm` ran WW3 `dea641c` (hpc build, 32 ranks)
with `RUN%FLSOU = F` for 48 h from 8 January 2026 00 h, and
`scripts/m1_prop_hpc.slurm` ran `hww3-prop` in the same mode:

- `prop`: restart, currents, levels, ice and boundary spectra;
- `geo`: the same with refraction and frequency shift off
  (`RUN%FLCTH = RUN%FLCK = F`, `--no-cth --no-ck`);
- `ideal`: a Gaussian swell packet from `ww3_strt` (0.08 Hz, centred at
  25° W 47° N, 3° spread, $H_s$ up to 3 m), without forcing or boundary
  spectra.

`scripts/m1_compare_nc.py` compares the 49 hourly fields. "RMS" is the RMS
difference over the hours at a node; the energy is $\sum_i (H_{s,i}/4)^2$
over the nodes, at the worst hour:

| Mode | Nodes | $H_s$ RMS, median | $H_s$ RMS, p99 | $H_s$ RMS, max | Energy | WW3, 32 cores | GPU | Jobs (WW3, GPU) |
|---|---|---|---|---|---|---|---|---|
| `prop` | 371,278 | 0.94 mm | 6.0 mm | 0.48 m | −0.047 % | 813 s | 32.0 s | 4118215, 4117585 |
| `geo` | 371,278 | 0.63 mm | 3.1 mm | 0.47 m | −0.018 % | 702 s | 24.7 s | 4118220, 4118253 |
| `ideal`, nodes reaching $H_s$ = 0.5 m | 22,015 | 1.5 mm | 4.8 mm | 7.0 cm | −0.45 % (13 h), +0.12 % at 48 h | 887 s | 32.9 s | 4118221, 4118252 |

The three M1 targets (an $H_s$ RMS difference below 1 cm at the median node
and below 5 cm at the 99th percentile, the energy within 0.5 %) are met in
the three modes. $T_{m02}$ differs by 3.2 ms at the median node and 0.06 s
at the 99th percentile in `prop`.

### Open question: coast nodes

The largest differences are all on coast nodes (`IOBP = 0`), most of them
with a single triangle. In `prop`, 1,074 of the 371,278 nodes differ by
more than 1 cm RMS, 49 by more than 5 cm and 7 by more than 20 cm. At the
worst node (169562, Nova Scotia, 6.1 m deep), $H_s$ is 2.11 m on the GPU
against 1.84 m in WW3 one hour after the restart, from the same 1.37 m.

Ruled out so far:

- the masks: `IOBP` and `IOBPD` read from `mod_def.ww3` are identical to
  those of `geom` on all nodes, including the 40,775 coast nodes;
- the depths: `ZB` of `mod_def.ww3` is within 0.8 mm of the grid depths;
- the precision: the `geo` run in double precision (job 4118461) is within
  0.6 mm of the single-precision run on every node;
- the code path: WW3 calls `PDLIB_W3XYPUG` per spectral bin with
  `PDLIB_W3XYPFSN2`, recomputes `ITER` after the current and level updates,
  and builds `PDLIB_SI` and `PDLIB_IEN` as `geom` does.

Coast nodes are left out of the CFL count, so a coast node with a single
triangle can run at a local CFL number above 1, where the result depends
much more on the number of sub-steps than elsewhere. A one-hour WW3 run
with fields every 600 s and point spectra at a few of these nodes would
show at which step and in which components the difference appears.
