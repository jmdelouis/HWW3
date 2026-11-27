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

| Grid | Nodes | Triangles | Interior / coast / open | $S$ (`explicit_cost.py`) | Sweeps per step | ITER at $f_1$ | $S$, every wet node | Sweeps per step, every wet node | ITER at $f_1$, every wet node |
|---|---|---|---|---|---|---|---|---|---|
| Channel–Biscay box of v4 | 21,425 | 41,313 | 19,872 / 1,358 / 195 | 3,079 (3,080) | 92 | 6–8 | 6,308 | 208 | 8–17 |
| v4 | 794,613 | 1,531,542 | 734,829 / 58,443 / 1,341 | 3,678 (3,678) | 110 | 8–10 | 8,326 | 258 | 18–24 |
| v5 $K=4$ | 424,123 | 807,832 | 382,296 / 40,775 / 1,052 | 2,837 (2,835) | 83 | 4–5 | 9,003 | 287 | 19–26 |

A coast node propagates 18.9 of the 36 directions on average.

The M0 counts (columns 5 to 7) take the interior nodes only, as
`scripts/explicit_cost.py --interior-only`. WW3 runs with `FSBCCFL = T`
(the default of `UGBCCFL`) and counts the coast and open-boundary nodes as
well (see [Coast nodes](#coast-nodes-the-sub-cycling-count)), which gives
the last three columns (`hww3-wave info` and `scripts/explicit_cost.py`,
which agree exactly, without currents).

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
day), not the spectral propagation or the source terms. The step and
per-day times use the interior sweep counts (110 and 83 per step). With the
count of WW3 (258 and 287), `two_phase_x4` takes 0.240 s per step on v4
(34.6 s per model day, 38.2 s for the transpose) and 0.148 s on v5 $K=4$
(21.3 s, 24.1 s for the transpose); the sweep times do not change.

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

With the sub-cycling of WW3 (every wet node, see
[Coast nodes](#coast-nodes-the-sub-cycling-count)), the 48 h run makes
82,386 sweeps instead of 23,917 and takes 65.1 s, 32.5 s per model day
(job 4120563). The rest of this section describes the runs before that
correction.

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
| `prop` | 371,278 | 1.5 µm | 24 µm | 0.71 mm | −2.4e-7 | 813 s | 65.1 s | 4118215, 4120563 |
| `geo` | 371,278 | 0.23 µm | 5.9 µm | 0.71 mm | −2.8e-8 | 702 s | 57.6 s | 4118220, 4120564 |
| `ideal`, nodes reaching $H_s$ = 0.5 m | 22,002 | 0.11 µm | 4.9 µm | 22 µm | −1.3e-7 | 887 s | 66.3 s | 4118221, 4120565 |

These runs count every wet node in `ITER`, as WW3 does. The three M1
targets (an $H_s$ RMS difference below 1 cm at the median node and below
5 cm at the 99th percentile, the energy within 0.5 %) are met in the three
modes with a margin of three to four orders of magnitude: no node differs
by more than 1 mm RMS, and $T_{m02}$ differs by 5.7 µs at the median node
and 0.11 ms at the 99th percentile in `prop`. The GPU runs the 48 h 12.5
times faster than WW3 on 32 cores.

Before that correction, with `ITER` counted on the interior nodes only
(jobs 4117585, 4118253 and 4118252), the median and 99th-percentile RMS
differences were 0.94 / 6.0 mm in `prop`, 0.63 / 3.1 mm in `geo` and
1.5 / 4.8 mm in `ideal`, with up to 0.48 m on coast nodes.

### Coast nodes: the sub-cycling count

**The question.** With `ITER` counted on the interior nodes only, the
largest differences were all on coast nodes (`IOBP = 0`), most of them in a
single triangle: in `prop`, 1,074 of the 371,278 nodes differed by more
than 1 cm RMS, 49 by more than 5 cm and 7 by more than 20 cm. At node
169562 (Nova Scotia, 6.1 m deep), $H_s$ was 2.11 m on the GPU against
1.84 m in WW3 one hour after the restart, from the same 1.37 m.

**Ruled out first.**

- the masks: `IOBP` and `IOBPD` read from `mod_def.ww3` are identical to
  those of `geom` on all nodes, including the 40,775 coast nodes;
- the depths: `ZB` of `mod_def.ww3` is within 0.8 mm of the grid depths,
  and the triangles of `natl.msh` are those of the grid;
- the precision: the `geo` run in double precision (job 4118461) is within
  0.6 mm of the single-precision run on every node;
- the code path: WW3 calls `PDLIB_W3XYPUG` per spectral bin with
  `PDLIB_W3XYPFSN2`, recomputes `ITER` after the current and level updates
  (`UGDTUPDATE`), and builds `PDLIB_SI` and `PDLIB_IEN` as `geom` does.

**Step by step.** WW3 was run for one hour from the same restart with a
restart file written after each of the first three 600 s steps (`prop`
and `geo`, Slurm jobs 4119699 and 4119701; the PDLIB restart writer needs
`ulimit -s unlimited`). `hww3-prop check-steps` compares the full spectra
of every node with those files, by node class and by graph distance to the
coast, and gives the frequencies and directions that differ at chosen
nodes with `IOBPD`, `ITER` and the local CFL number
$\Delta t / \mathrm{ITER} \sum_T K^+_{iT} / \mathrm{SI}_i$ (jobs 4119700 and
4119702):

- the difference appears in the first step, in the geographic part
  (`geo` as much as `prop`): 902 coast nodes and 182 interior nodes differ
  by more than 1 cm in $H_s$, up to 13 cm;
- it is not confined to the coast: 12 nodes 8 or more edges away from any
  coast differ by more than 1 cm (node 165663, on the Sable Island bank,
  5.4 m deep, 24 edges from the coast: 3.3 % more energy in all directions
  of the peak frequencies), and most of the worst nodes are 5–7 m deep;
- the inputs agree: the depths and currents of WW3 (fields `DPT` and `CUR`
  every 600 s, job 4120131) are within 1 mm and 2e-6 m/s of those of the
  GPU on every node;
- the coast nodes in a single triangle run at local CFL numbers of 2.5–4.8
  (node 169562: 3.05 with `ITER` = 1 at frequency 22), where the N scheme
  is no longer positive;
- adding one sub-step to every component on the GPU cuts the coast nodes
  above 1 cm from 902 to 95 and the interior ones from 182 to 16: WW3
  takes more sub-steps.

**The cause.** In `PDLIB_W3XYPFSN2`, the time-step limit `DTMAX` is taken
over the nodes with `IOBP_LOC(IP) .EQ. 1 .OR. FSBCCFL`. `FSBCCFL` comes
from `UGBCCFL` in the `UNST` namelist of `ww3_grid`, whose default is true
(`ww3_grid.out` of the runs: `UGBCCFL = T`). WW3 therefore counts every wet
node, coast and open boundary included, and the small cells of the coast
set the sub-cycling: on grid v5 $K=4$, `ITER` at the lowest frequency is
19–26 instead of 4–5, and a step needs 287 frequency sweeps instead of 83
(258 instead of 110 on grid v4).

**The fix.** `ITER` now counts every wet node in the GPU kernels
(`iter_partial`), in `cpu::geo_freq` and in `geom` (`Config::bccfl`,
`Geometry::set_bccfl`, `--no-bccfl` for the interior count). After it, the
local CFL number stays below 1 everywhere (0.96 at node 169562 with
`ITER` = 5), the first three steps agree with WW3 within 0.1 mm in `geo`
and 1.1 mm in `prop` on every node (jobs 4120561 and 4120562), and the
48 h runs agree within 0.71 mm (table above). The GPU run takes twice as
long (65.1 s instead of 32.0 s per 48 h), since it makes 3.4 times more
sweeps.

**Consequences.**

- The M0 per-day times (14.8 s on grid v4, 6.2 s on grid v5 $K=4$) used
  the interior count; with that of WW3 they are 34.6 and 21.3 s.
- The sub-step cap `cfl_substeps` of `hww3-grid`, which shaped grid v5,
  counted the interior nodes only, as M0 did; the coast cells set the
  explicit cost of WW3 on these grids. `scripts/explicit_cost.py` now
  counts every wet node (`--interior-only` for the former count) and
  `hww3-grid` counts them with `refine_bathy.cfl_boundary`; see
  [Validation](validation.md), grid v5, for the consequences.

## Milestone M2a: the source terms (targets met)

M2a adds the source terms of the HWW3 configuration with the simplified
ST4: `SINTABLE = 0` in `&SIN4` (no stress tables: u* by Newton iterations
in `CALC_USTAR`, the tail stress integrated in line over 50 wavenumbers in
`W3SIN4`) and `SDSCUM = 0` in `&SDS4` (no cumulative breaking term), with
the DIA (`W3SNL1`), `W3SLN1`, `W3SDB1`, `W3SBT4` and the integration of
`W3SRCE`. Everything else is as in the grid v5 $K=4$ explicit run
(`BETAMAX = 1.75`, `UGBCCFL` at its default).

### WW3 references

`scripts/m2_ww3_ref_hpc.slurm` runs WW3 `dea641c` (hpc build) with the
simplified ST4; `ww3_grid.out` confirms `SINTABLE = 0`, `SDSCUM = 0.00`
and no stress table. The forcing of 1–10 January 2026 (wind, currents,
levels, ice, boundary spectra) was copied from the Datarmor run directories
to `$R/m2/forcing` with `ww3_subset.py`.

| Run | Period | Start | Outputs | Slurm job | Cost |
|---|---|---|---|---|---|
| `dev` | 8–10 January, 48 h | `restart001.ww3` of the explicit run (8 January 00 h) | hourly fields; restarts after the first three 600 s steps | 4122569 (32 ranks) | 3,064 s, 27 core-hours |
| `week` | 1–8 January | cold start, as the explicit run | hourly fields, for the final targets and the scores | 4122614 (64 ranks) | 5,003 s, 88 core-hours |
| `week_start` | 1 January, 1 h | cold start | the restart after the first 600 s step, to start the GPU week from | 4138952 (32 ranks) | 398 s, 3 core-hours |

The `dev` run takes 64 s per model hour on 32 ranks, against 17 s for the
propagation-only reference of M1 on the same grid and cores: the source
terms cost about three times the propagation in WW3.

The altimeter and buoy scores run on hpc with the validation scripts
unchanged (`scripts/m2_scores_hpc.slurm`, a Python environment and the
January observations on scratch, see [Reproducing](reproducing.md)). The
matching is the one of the Datarmor runs: 45,518 altimeter blocks over
4 January 00 h – 7 January 21 h, as for `natl_v5k4_jan2026_exp`. The `week`
reference with the simplified ST4 (Slurm job 4139219) scores:

| | Bias | RMSE $H_s$ | Scatter index | Correlation |
|---|---|---|---|---|
| Altimeters (45,518 blocks) | +1.2 cm | 28.4 cm | 0.121 | 0.971 |
| Buoys (252) | −3.6 cm | 32.2 cm | 0.276 | 0.919 |
| Calibrated full ST4 (`natl_v5k4_jan2026_exp`), altimeters | −1.5 cm | 28.3 cm | 0.121 | 0.970 |

### Source-term harness

`hww3-wave/harness/src_harness.F90` is linked against the hpc WW3 build
(`libww3.a` and its modules, the same compiler flags) by
`scripts/m2_src_harness_hpc.slurm`. It sets WW3 up as `ww3_shel` does,
reads the `mod_def.ww3` of the simplified-ST4 reference (which runs
`INSIN4` and `INSNL1`), takes the spectra, level, ice, u*, u* direction and
`ASF` of a restart and the wind of `wind.ww3` at the same time, and calls,
node by node, the routines of the first iteration of `W3SRCE` (`W3SPR4`,
`W3SIN4`, `W3SPR4`, `W3SLN1`, `W3SIN4`, `W3SNL1`, `W3SDS4`, `W3SDB1`,
`W3SBT4`) and then a full `W3SRCE` (600 s). It writes their inputs and
outputs, and a parameter file with what those routines read from the WW3
modules: the spectral grid, the ST4 constants, `FWTABLE`, the saturation
indices and weights of `W3SDS4`, the DIA tables of `INSNL1` (`IP11` …
`IC82`, `AWG`, `SWG`, `AF11`), the LN1, DB1 and BT4 coefficients. The ports
read that file instead of computing the tables again
(`hww3_wave::harness`).

On the restart of 8 January 00 h, every 200th sea point and five chosen
nodes give 1,869 nodes (Slurm job 4123121, 5 s). The values are physical:
drag coefficients of 0.8–1.6e-3, Charnock coefficients of 0.0095–0.023, u*
from `W3SPR4` close to the restart value, dynamic steps of 77–212 s at
shallow windy nodes where bottom friction dominates.

### Ports checked against the harness

CPU transcriptions in double precision (`hww3_wave::sources`), each call
fed with the inputs WW3 had in the harness (`hww3-src check`); the error is
the largest difference over the 1,869 nodes, relative to WW3 (to the
largest value of the node for spectra):

| Routine | Outputs | Largest relative difference |
|---|---|---|
| `W3SPR4`, `CALC_USTAR` (10 Newton iterations) | EMEAN, FMEAN, FMEAN1, WNMEAN, FMEANWS | 1.1e-6 |
| | u*, $C_d$, $z_0$, Charnock | 2.6e-6 |
| | mean direction of the long waves | 6e-5 rad (absolute) |
| `W3SIN4` (tail stress in line) | S, D | 3.2e-6 |
| | wave-supported stress, stress of swell damping | 3.4e-6 |
| | wind-sea flags `LLWS` | 1 bin of 2,153,088 differs |
| `W3SLN1` | S | 5.3e-6 |
| `W3SNL1` (DIA, tables of `INSNL1`) | S, D | 1.1e-6 |
| `W3SDS4` (`SDSBCHOICE = 1`, no cumulative term) | S, D | 1.7e-5 |
| `W3SDB1` | S, D; breaking flag | 3.6e-8; no node differs |
| `W3SBT4` (movable bed) | S, D, bottom stress | 5.0e-5 |
| | ripple roughness | 1.0e-4 |
| `W3SRCE` (600 s, dynamic step, limiter, tail, `FHMAX`, ice scaling) | spectrum after the step | 4.3e-6 (mean 6.6e-8) |
| | mean dynamic step, cut-off frequency, u*, Charnock, mean wavenumber | 1.6e-5 |
| | stresses (to the waves, to the ocean, swell damping, bottom) | 2.9e-4 |
| | energy flux from the wind | 5.3e-4 |
| | shaved steps of the limiter | no node differs |

The larger differences of `W3SDS4` and `W3SBT4` come from the
single-precision tables WW3 interpolates (error function, friction
factor); those of the integrated stresses and fluxes from sums of terms of
both signs. The flux to the ocean, a difference of the integrated fluxes,
agrees to 1.2e-4 on average but to 4.4e-2 at the node where it nearly
cancels. The number of dynamic steps goes from 1 to 60 (2.4 on average);
the mean step agrees to 8.9e-6, so both codes take the same steps at every
node.

These are the round-off differences of the single-precision WW3.

### GPU source kernel

`hww3_wave::srce` runs the whole `W3SRCE` integration of a global step in
one CubeCL launch, one cube of 128 units per node. The spectrum and the
work arrays of the node (source sums, wind-sea flags, the extended spectrum
and the three interaction arrays of the DIA, the saturation of `W3SDS4`)
sit in shared memory, about 39 kB per cube in single precision and 78 kB in
double precision (CubeCL allocates it dynamically, up to the 163 kB of an
A100). The units share the loops over the 1,152 bins. The scalar parts run
on unit 0: the mean parameters, `CALC_USTAR`, the sheltering loop over the
frequencies of `W3SIN4`, the 50-point tail, the three depths of `W3SBT4`
and the dynamic step. Their sums go in the order of WW3. The parameters and
tables are those of the harness parameter file, written by WW3 itself from
the `mod_def.ww3` of the run (`SrceTables`).

On the 1,869 harness nodes (`hww3-src check-gpu`, Slurm job 4138915, one
A100):

| Run | Spectrum after 600 s: largest difference relative to the node maximum | $H_s$ | Mean dynamic step | u* |
|---|---|---|---|---|
| GPU single precision against WW3 | 3.1e-6 (mean 4.1e-8) | 0.9 µm | 1.8e-5 | 3.3e-6 |
| GPU double precision against WW3 | 4.3e-6 (mean 6.6e-8) | 1.0 µm | 8.8e-6 | 2.6e-6 |
| GPU double precision against the CPU transcription | 4.6e-8 | 0.01 µm | | |

No node takes a different number of dynamic steps. The double-precision
kernel departs from the CPU transcription by 4.6e-8 because it keeps the
single-precision constants of WW3 (`TPIINV`, 0.99 in the step count and the
other literals). On 200 copies of the harness nodes (373,800 nodes, about
the size of grid v5 $K=4$), one 600 s step of source terms takes 0.97 s in
single precision, which would add 280 s to the 65 s of the 48 h
propagation.

### Time loop and the first steps

`hww3-prop` with `--wind` and `--sources` runs the source kernel after the
propagation of every step, at the nodes with `MAPSTA = 1`, as `W3WAVE`
does. The wind is read like the currents (the two records around the time)
and interpolated by `W3UWND` with `WNT2` (speed corrected so that the
energy varies linearly) and `RWND` (`RWNDC = 1`: the current of the step
is removed). The ice concentration of `W3UICE` scales the source terms.
`SED_D50` and `SED_PSIC` come per sea point from the harness parameter
file (WW3 allocates them from 0, hence a 0 in their range). The start call
of `W3WAVE` (`IT = 0`, `DTG = 1 s`) is skipped. It only sets u*, and
`CALC_USTAR` does not use the previous u*.

`hww3-prop check-steps` against the restarts the `dev` reference wrote
after its first three steps (Slurm job 4138987, one A100):

| After | Coast nodes, largest $\lvert\Delta H_s\rvert$ | Interior nodes above 1 mm | Largest $\lvert\Delta H_s\rvert$ | Above 1 cm |
|---|---|---|---|---|
| 1 step (00:10) | 0.8 mm | 7 of 335,539 | 3.9 mm (node 30735, Tyrrhenian Sea, 922 m) | 0 |
| 2 steps (00:20) | 0.9 mm | 11 | 3.3 mm | 0 |
| 3 steps (00:30) | 0.6 mm | 13 | 7.9 mm (node 17597, Aegean Sea, 78 m) | 0 |

The open-boundary nodes agree to 0.01 mm. The propagation alone agreed to
about 1 µm (M1), and the source kernel reproduces WW3 to 1 µm on the
harness nodes. These millimetre differences at a few nodes out of 371,000
come from somewhere else. At the worst nodes they sit in the
high-frequency bands, around the cut-off: bands 24–27 differ by 1–2 % at
node 398721. This points to discrete decisions taken differently at a few
nodes: the tail index `NKH`, or the number of dynamic steps when the
limiting step falls close to a threshold. The wind of the step, which the
harness does not test, is the other input that differs. This is not
verified yet. The dynamic step and cut-off frequency of WW3 (`DTD`, `FC`)
at these steps would confirm it.

### 48 h and the week against WW3

Hourly $H_s$ and $T_{m02}$ of the GPU model against the WW3 references
(`scripts/m1_compare_nc.py`; RMS difference over the hours at each of the
371,278 nodes where both are defined, then its median, 99th percentile and
maximum over the nodes):

| Run | GPU (one A100) | WW3 | $H_s$: median / p99 / max | $T_{m02}$: median / p99 / max | Energy |
|---|---|---|---|---|---|
| 48 h from 8 January 00 h (jobs 4139023, 4139369) | 291.7 s, 146 s per model day | 3,064 s on 32 ranks | 2.1 µm / 86 µm / 2.3 cm | 5.0e-6 / 3.3e-4 / 0.57 s | 1.4e-6 |
| 1–8 January, from the WW3 restart of 00:10 (jobs 4139117, 4139370) | 1,002 s, 143 s per model day | 5,003 s on 64 ranks | 2.5 µm / 0.11 mm / 1.9 cm | 6.4e-6 / 4.9e-4 / 0.63 s | 3.7e-6 |

The section 6 targets are 2.1 cm at the median node and 20 cm at the 99th
percentile, the difference between the explicit and implicit runs of grid
v4. The GPU model is four orders of magnitude below them. The hourly wind
it applies (written with `--wind-out`) is WW3's to 1e-6 m/s at the median
node and 5.5e-6 m/s at worst. One A100 runs the 48 h ten times as fast as
32 cores of the hpc cluster.

The scores of the week (job 4139371, window of the explicit runs) are the
same for both:

| | WW3 | GPU |
|---|---|---|
| Altimeters, 45,518 blocks: bias / RMSE of $H_s$ | +1.1586 / 28.3882 cm | +1.1585 / 28.3882 cm |
| Buoys, 252: bias / RMSE of $H_s$ | −3.5877 / 32.2130 cm | −3.5877 / 32.2130 cm |
| Buoys, 171: bias / RMSE of $T_{m02}$ | −0.4666 / 1.2116 s | −0.4666 / 1.2116 s |

**Nodes above 1 cm.** Eight nodes in each run have an RMS difference above
1 cm. In the 48 h run they are in the fjords of Lofoten and Troms
(68–71° N), with $H_s$ of 0.05–0.2 m under 3–4 m/s of wind, and in the
western Channel and on the Bahamas banks. In the week they are in the
archipelago of the Gulf of Finland, in Passamaquoddy Bay (Bay of Fundy), in
Ungava Bay (ice concentration up to 0.4) and, once, in a Labrador Sea storm
($H_s$ 8 m, 26 m/s of wind). Each is a spike of one or two hours: at node
79649, 0.055 m on the GPU against 0.178 m in WW3 on 9 January 02 h, then
within 1 mm an hour later. In the Labrador Sea, 7.92 against 8.07 m for one
hour.

The same 48 h in double precision on the GPU (job 4162035) has the same
spikes at the same nodes and hours. The two GPU runs differ by 0.14 µm in
$H_s$ at the median node, and at most by 4.2 mm RMS (2.6 cm in one hour) at
the same Lofoten node, 77687. WW3 then wrote restarts every 10 minutes from
9 January 00:50 to 02:00 (job 4162053, 11 core-hours), and `check-steps`
compared the spectra at those times (job 4162054):

- At node 79649, a coast node of a fjord at 180 m depth, the difference
  is entirely in the highest bands, 26 to 31. These bands carry energy in
  one model and almost none in the other, in turn: band 27 holds 1.2e-3
  on the GPU against 1.7e-7 in WW3 at 00:50, and 5e-5 against 1.6e-3 at
  01:50. Above the index `NKH` of the cut-off frequency, `W3SRCE`
  replaces the spectrum by the tail of the band below. So when `NKH`
  changes, these bands are either grown by the wind or reset.
- From WW3's own state at 00:50, the harness at these nodes (Slurm job
  4162062) finds that the CPU transcription of `W3SRCE` reproduces WW3 to
  4.8e-7. The cut-off frequency agrees to 8e-8 and the limiter decisions
  are the same, and the GPU kernel reproduces the CPU transcription to
  4.6e-8. These nodes take up to 47 dynamic steps in a 600 s step, against
  2.4 on average, so the high frequencies there grow fast under 3–4 m/s of
  wind.

So the source terms are right at these nodes. Where the young sea grows
this fast near the cut-off, differences of the order of the
single-precision round-off decide, from step to step, whether the top
bands are grown or reset. The order of the operations, fused
multiply-adds and the elementary functions differ between WW3 and the GPU
and are the same in its two precisions. So the GPU runs follow each other
more closely than they follow WW3. This is the high-frequency cause seen
in the first steps, amplified where the dynamic step is shortest. It
leaves the median and the 99th percentile four orders of magnitude below
the targets.

## Milestone M2b: the full ST4 (targets met)

M2b runs the source terms of the calibrated configuration: the stress
tables of `INSIN4` (`SINTABLE = 1`: u* interpolated in `TAUT` by
`CALC_USTAR`, the tail stress in `TAUHFT2`, with sheltering, by `W3SIN4`)
and the directional cumulative breaking term of `W3SDS4`
(`SDSCUM = -0.40`, through the matrix `CUMULW`). Both are WW3 defaults,
so the namelists of `natl_v5k4_jan2026_exp` are used unchanged
(`scripts/m2_ww3_ref_hpc.slurm` with `ST4=full`).

### Harness and CPU ports

The harness also writes the stress tables (`TAUT`, `TAUHFT`, `TAUHFT2`
with their steps), `DIKCUMUL` and `CUMULW` when WW3 holds them. On the
restart of 8 January with the `mod_def.ww3` of the calibrated
configuration (Slurm job 4139132), the CPU transcriptions with these
branches reproduce WW3 on the 1,869 nodes as closely as in M2a:

| Routine | Largest relative difference |
|---|---|
| `CALC_USTAR` from `TAUT`: u* | 1.5e-7 |
| `W3SIN4` with the tail stress from `TAUHFT2`: S, D; wave-supported stress | 4.1e-6; 2.7e-6 |
| `W3SDS4` with the cumulative term: S, D | 1.7e-5 |
| `W3SRCE`, 600 s: spectrum after the step (mean) | 3.9e-6 (6.5e-8) |
| `W3SRCE`: mean dynamic step | 1.3e-5 |

The wind-sea flags, the depth-breaking flags and the limiter decisions
are the same as WW3's at every node. Dynamic steps: 1 to 60 per node, 1.8
on average.

The GPU kernel has the same branches. `CUMULW` depends only on the two
frequencies and on the direction difference, so it goes to the device as
a table of $32 \times 32 \times 36$ values, checked against every used
entry of the matrix. On the harness nodes (job 4139172), the kernel
reproduces the full `W3SRCE` of WW3 to 4.0e-6 in single precision
(0.9 µm in $H_s$), with no node taking another number of dynamic steps, and
the CPU transcription to 4.6e-8 in double precision. A step on 373,800
nodes takes 0.81 s, against 0.97 s with the simplified ST4: the table
look-ups cost less than the Newton iterations and the 50-point tail, and
more than pay for the cumulative term.

### First steps and 48 h

`check-steps` against the restarts of the calibrated reference (job
4162033) finds at most 6 interior nodes above 1 mm and none above 1 cm
after each of the first three steps (largest 4.4 mm). Over 48 h from
8 January (GPU job 4162034 against the WW3 reference 4139131, which took
2,982 s on 32 ranks, 26 core-hours):

| $H_s$: median / p99 / max | $T_{m02}$: median / p99 / max | Energy | Wind | GPU |
|---|---|---|---|---|
| 2.0 µm / 77 µm / 2.1 cm | 4.9e-6 / 3.2e-4 / 0.16 s | 1.1e-6 | 1e-6 m/s (median) | 254 s, 127 s per model day |

The worst node, 134247 on the Normandy coast at 6.8 m of depth, has the
same one-hour spikes as in M2a (1.94 against 2.08 m on 9 January 01 h). One
A100 runs the 48 h 11.7 times as fast as 32 cores.

### The week against the explicit run of grid v5 $K=4$

The GPU model with the full ST4 ran 1–8 January from the restart written
after the first step of a 1 h cold-started WW3 run with the calibrated
namelists (Slurm job 4162036, 360 s on 32 ranks, 3 core-hours). It is
compared with `natl_v5k4_jan2026_exp`, the explicit run made on Datarmor,
whose hourly fields were copied to hpc (GPU job 4162038, comparison
4162039, scores 4162040):

| $H_s$: median / p99 / max | $T_{m02}$: median / p99 / max | Energy | GPU |
|---|---|---|---|
| 2.2 µm / 0.10 mm / 3.0 cm | 6.2e-6 / 4.8e-4 / 0.25 s | 1.2e-5 | 868 s, 124 s per model day |

| | WW3 (`natl_v5k4_jan2026_exp`) | GPU |
|---|---|---|
| Altimeters, 45,518 blocks: bias / RMSE of $H_s$ | −1.5034 / 28.2527 cm | −1.5034 / 28.2527 cm |
| Buoys, 252: bias / RMSE of $H_s$ | −5.3770 / 32.4627 cm | −5.3770 / 32.4627 cm |
| Buoys, 171: bias / RMSE of $T_{m02}$ | −0.4238 / 1.2095 s | −0.4238 / 1.2095 s |

The scores of the WW3 run are those of its validation on Datarmor
(`results/natl_v5k4_jan2026_exp_val_stats.json`). The nodes above 1 cm are
the same kind as in M2a: Ungava Bay, the Gulf of Finland, Svalbard, the
Canadian Arctic, the Orinoco delta.

### Grid v4: comparison and speed

Grid v4 (794,613 nodes) with the full ST4, from the restart written after
the first step of a 1 h cold-started WW3 run on hpc (Slurm job 4162072,
630 s on 32 ranks, 5 core-hours), to 7 January 21 h, against the explicit
run `natl_v4_jan2026_exp` of Datarmor. Its set-up, its forcing of
1–8 January and its hourly fields were copied to hpc. The harness on the
v4 state of 1 January 00:10 (job 4162073) finds the `W3SRCE` of WW3 to
3.5e-6 on 1,695 nodes, with the same decisions.

| $H_s$: median / p99 / max | $T_{m02}$: median / p99 / max | Energy | GPU (one A100) |
|---|---|---|---|
| 3.6 µm / 0.14 mm / 5.4 cm | 9.8e-6 / 7.1e-4 / 0.63 s | 1.7e-5 | 1,514 s for 165 h: 220 s per model day |

| | WW3 (`natl_v4_jan2026_exp`) | GPU |
|---|---|---|
| Altimeters, 44,670 blocks: bias / RMSE of $H_s$ | −1.7402 / 28.2533 cm | −1.7403 / 28.2534 cm |
| Buoys, 252: bias / RMSE of $H_s$ | −5.6577 / 30.1013 cm | −5.6577 / 30.1013 cm |
| Buoys, 171: bias / RMSE of $T_{m02}$ | −0.4151 / 1.2189 s | −0.4151 / 1.2189 s |

(Jobs 4162075 to 4162077.) The WW3 scores are those of its Datarmor
validation (`results/natl_v4_jan2026_exp_val_stats.json`, `…_buoys_stats.json`).
The GPU model takes 220 s per model day, under the 430 s of the M2b target
(ten times the explicit WW3).

The largest differences, up to 5.4 cm RMS and 44 cm in one hour, are in a
cluster of nodes at the mouth of Ungava Bay (60.2–60.6° N, 64.2–64.6° W),
from 4 to 7 January. There the ice concentration is 0.35–0.43, which
scales the source terms, the wind 15–17 m/s and the tidal current
0.2–0.3 m/s. Unlike the fjord nodes, the differences last several hours and
change $T_{m02}$ by up to 1.3 s, so they reach more than the top bands. The
same nodes stand out on grid v5 $K=4$ (up to 28 cm in one hour).

They come from the history of the run, not from a step that differs. On
grid v5 $K=4$, the calibrated WW3 week was rerun with restarts every
10 minutes from 5 January 00:00 to 02:00 (Slurm job 4162775, 64 cores,
1 h). Against them, the GPU run from 1 January differs by 13 cm at node
238271 at 00:00 and the difference grows by about 2.5 cm per step, in the
peak bands 16 to 20, where in several directions the GPU holds half of
WW3's energy or less (job 4162776). Started from the WW3 restart of 00:00
itself (`check-steps --itstep0 576`, job 4163960), the GPU stays within
5 mm of WW3 over the 12 steps everywhere, and the Ungava nodes are no
longer among the largest differences. On the same state, the harness finds
the CPU `W3SRCE` within 2.1e-7 of WW3 at six nodes at the mouth of the bay,
with the same limiter and breaking decisions and 3 to 40 dynamic steps
(job 4163959). Each step of the model is therefore WW3's there; the
difference builds up over days in a fast-growing sea in partial ice, where
small differences of state are amplified, as with the fjord spikes. The M3
tangent runs show how fast this happens: a change of $10^{-4}$ of the wind
flips some discrete choice of `W3SRCE` at most nodes within hours (section
"The tangent through the time loop").

Round-off alone is enough. The same GPU run in double precision from
1 January (job 4165288) agrees with the single-precision run to 1 mm at
these nodes until 4 January 18 h, then differs from it as much as WW3
does, at the same nodes and hours:

| 5 January | $H_s$ at node 238271: single / double precision / WW3 | nodes above 1 cm: single − double | single − WW3 |
|---|---|---|---|
| 00 h | 5.061 / 5.225 / 5.194 m | 18 | 16 |
| 01 h | 5.108 / 5.342 / 5.337 m | 22 | 28 |
| 02 h | 5.052 / 5.060 / 5.148 m | 38 | 37 |

The Ungava cluster is therefore the sensitivity of the model to round-off
there, not an error of the GPU model.

### Finite-difference gradients

Gradients of the GPU model with the calibrated full ST4 with respect to
the parameters of M2: 48 h in double precision from the restart of
8 January 00 h on grid v5 $K=4$, with each parameter multiplied by
$1 \pm 0.01$, and $1 \pm 0.001$ for the wind and `BETAMAX` (15 runs, Slurm
jobs 4162941 to 4162955 and 4163276, 20 min each on one A100). `BETAMAX`
enters the stress tables, so its runs read the tables of perturbed
`mod_def.ww3` files (`scripts/m2_moddef_hpc.slurm`); the wind is the
forcing wind times a factor. The responses are the total energy and the
mean $H_s$ at 10 January 00 h and the scores of 8–10 January against the
altimeters (22,933 blocks) and the buoys (12,153 hours), job 4163939.
Derivatives per unit relative change, $\partial R / \partial \ln p$
(base values: energy 1.0028e5, mean $H_s$ 1.7358 m, altimeters
+6.23 / 25.73 cm, buoys −3.89 / 42.35 cm in $H_s$ and −0.376 / 1.196 s in
$T_{m02}$):

| $p$ (step) | energy | mean $H_s$ (m) | altimeters: bias / RMSE (m) | buoys $H_s$: bias / RMSE (m) | buoys $T_{m02}$: bias / RMSE (s) |
|---|---|---|---|---|---|
| wind (1 %) | 3.347e5 | 2.769 | 2.675 / 0.847 | 1.833 / −0.018 | 1.294 / 0.521 |
| wind (0.1 %) | 3.346e5 | 2.769 | 2.672 / 0.850 | 1.835 / −0.021 | 1.266 / 0.466 |
| `BETAMAX` (1 %) | 6.700e4 | 0.564 | 0.530 / 0.159 | 0.444 / −0.017 | 0.193 / 0.116 |
| `BETAMAX` (0.1 %) | 6.707e4 | 0.564 | 0.530 / 0.159 | 0.442 / −0.017 | 0.183 / 0.097 |
| `SDSC2` (1 %) | −3.15e4 | −0.264 | −0.254 / −0.072 | −0.206 / 0.018 | −0.091 / 0.003 |
| `NLPROP` (1 %) | 1.66e4 | 0.146 | 0.146 / 0.038 | 0.094 / −0.010 | 0.238 / −0.070 |
| `BJGAM` (1 %) | 310 | 0.003 | 0.000 / 0.000 | 0.017 / −0.012 | 0.004 / 0.002 |

The two steps agree to about $10^{-3}$ on the energy, the mean $H_s$ and
the biases, and to 2–20 % on the RMSEs, which are differences of nearly
equal numbers. Over two days the energy goes as the wind to the power
3.3; 1 % more `BETAMAX` raises $H_s$ by 5.6 mm on average and the
altimeter bias by 5.3 mm, 1 % more `SDSC2` lowers them by 2.6 and 2.5 mm,
and depth-induced breaking (`BJGAM`) acts only at the buoys of shallow
water. The $H_s$ RMSE against the buoys falls slightly with the wind,
`BETAMAX`, `NLPROP` and `BJGAM` and rises with `SDSC2`, while the
$T_{m02}$ RMSE rises with all but `NLPROP`. The 15 runs take 5 GPU-hours.

## Milestone M3: tangent kernels (target met where the model is smooth)

The tangent kernels differentiate the GPU model in forward mode for a
relative change of the wind speed or of a source parameter (`BETAMAX`,
`SDSC2`, `NLPROP`, `BJGAM`). They are written by hand, each line of the
value kernel with its derivative next to it: `srce_tl` for the whole
`W3SRCE` integration, `prop_tl` for `W3KTP3`, the geographic node update,
the level remap, the wind and the outputs. The discrete choices of the
model (number of dynamic steps, `NKH`, the limiter, the table cells, the
clamps at zero) are taken from the values and carry no derivative, so the
tangent is exact where the model is smooth. For `BETAMAX` the derivative
of the tail-stress table `TAUHFT2` comes from `TABU_TAUHF2` differentiated
in double precision (`hww3-src tables` checks it against the tables of
perturbed `mod_def.ww3` files).

### Tangent of the source terms

`hww3-src check-tl` runs the tangent kernel and the value kernel in double
precision on the harness nodes, one 600 s step from the harness state, and
compares the tangent with central differences of the value kernel with
relative steps $10^{-4}$ and $10^{-5}$. Where the two differences agree to
$10^{-6}$ of the largest derivative of the node (no limiter decision or
table cell changes within the step), the tangent is compared with the
$10^{-5}$ difference: largest difference over the spectrum relative to the
largest derivative, per node. The spectrum direction is a random relative
change of every bin.

| harness (nodes) | direction | smooth / sensitive nodes | median | p99 | max |
|---|---|---|---|---|---|
| v5 $K=4$, 8 January 00 h (1,869) | spectrum | 1,854 / 1,869 | 1.6e-11 | 1.0e-9 | 4.7e-7 |
| | wind | 1,705 / 1,869 | 2.8e-9 | 3.4e-7 | 9.5e-5 |
| | `BETAMAX` | 1,586 / 1,861 | 2.2e-9 | 6.6e-7 | 9.5e-7 |
| | `SDSC2` | 1,587 / 1,621 | 8.9e-10 | 3.2e-7 | 9.9e-7 |
| | `NLPROP` | 1,811 / 1,864 | 1.9e-9 | 4.6e-7 | 9.2e-7 |
| | `BJGAM` | 2 / 3 | 7.8e-9 | 7.8e-9 | 7.8e-9 |
| v4, 1 January 00:10 (1,695) | spectrum | 1,608 / 1,652 | 6.4e-11 | 5.0e-9 | 9.8e-7 |
| | wind | 1,400 / 1,658 | 2.3e-10 | 7.5e-9 | 2.1e-6 |
| | `BETAMAX` | 1,418 / 1,647 | 7.8e-11 | 6.1e-7 | 3.3e-6 |
| | `SDSC2` | 191 / 251 | 1.3e-10 | 4.8e-7 | 5.6e-7 |
| | `NLPROP` | 561 / 1,472 | 3.0e-8 | 9.8e-7 | 1.5e-6 |

(Jobs 4163750 and 4163751.) No smooth node is above the $10^{-4}$ of the
M3 target. The values of the tangent kernel are those of the value kernel
to 1.3e-16, and it takes 1.2 to 1.7 times its time. Nodes are sensitive
where the derivative exceeds $10^{-9}$ of the largest spectral density of
the node, below what the differences resolve. Depth-induced breaking acts
at three harness nodes of grid v5 $K=4$ and at none of grid v4, and the
saturation term of `SDSC2` at 251 nodes of the grid v4 state, ten minutes
after the cold start.
The first version of the comparison, with steps $10^{-3}$ and $10^{-4}$ and
agreement to $10^{-4}$ (job 4163415), counted more nodes as smooth and
found up to 1.1e-3 for the wind: there the larger step crosses cell edges
of the bilinear tables `TAUT` and `TAUHFT2` where the derivative jumps.

### The tangent through the time loop

`hww3-prop run --tangent NAME --f64 --out64` carries the tangent through
the whole step: forcing, boundary (zero tangent), ice and level updates,
`W3KTP3`, the geographic propagation and `W3SRCE`. Over 3 h from the
restart of 8 January 00 h on grid v5 $K=4$ it is compared with central
differences of runs with relative steps $10^{-4}$ and $10^{-5}$, written in
double precision. Per node, the hourly $H_s$ derivative is compared on the
nodes where the two differences agree to $10^{-6}$ of the node's scale
$s = \max(|d|, 10^{-3}\,\mathrm{rms}(d))$; the `INIT` direction is a
relative change of the restart spectra.

| direction | after | smooth nodes | median | p99 | above $10^{-4}$ | max |
|---|---|---|---|---|---|---|
| `INIT`, propagation only | 1 h | 100 % | 1.5e-11 | 4.9e-11 | 0 | 3.1e-7 |
| | 3 h | 100 % | 1.5e-11 | 5.5e-11 | 1 | 1.1e-4 |
| `INIT`, with the source terms | 1 h | 64 % | 7.6e-11 | 8.7e-7 | 7 | 1.0e-3 |
| | 3 h | 24 % | 4.0e-9 | 3.6e-6 | 14 | 4.8e-4 |
| `WIND` | 1 h | 19 % | 6.9e-9 | 6.6e-6 | 18 | 7.5e-4 |
| | 3 h | 3.4 % | 6.6e-8 | 3.5e-5 | 28 | 2.5e-2 |

(Jobs 4163934 to 4163936; percentages of the 371,278 nodes with $H_s$.)
Without the source terms the model is piecewise linear in the spectrum and
the tangent agrees with the differences to round-off on every node, and the
total energy to 2e-9; a change of all spectra by the same factor keeps the
limiter ratios of `W3KTP3`, so this checks the linear part of the
propagation tangent, not the limiter branches. With the source terms, where the model is smooth the
tangent agrees to $10^{-4}$ on all but 7 to 30 nodes.

But the model with the source terms is smooth at these scales at fewer and
fewer nodes: a change of $10^{-4}$ in the wind flips, within three hours, at
least one discrete choice of the model (the number of dynamic steps, the
cut-off index `NKH` and the limiter of `W3SRCE`, the limiter branches of
`W3KTP3`) at 97 % of the nodes, and each flip makes a small jump that the
tangent, a derivative along the path of the values, does not see.
On aggregates the jumps add up instead of vanishing with the step: the
derivative of the total energy after 3 h is 8.416e4 for the tangent
against 8.291e4 and 8.280e4 for the differences with steps $10^{-4}$ and
$10^{-5}$ (1.5 %), 4.384e4 against 4.373e4 and 4.378e4 for `INIT` (0.2 %).
The finite differences of aggregates do not converge either (0.13 % and
0.10 % between the two steps). The tangent is therefore the exact
derivative of the model where it is smooth, and the finite differences
with a step of 1 % include the mean effect of the jumps, which a
calibration sees.

### 48 h against the finite differences of M2b

The tangent runs of the five directions over the 48 h of the M2b finite
differences (Slurm jobs 4163952 to 4163956, comparison job 4163958) take
645 s per model day on one A100 in double precision, 1.5 times the 433 s
of a value run: one tangent run costs less than the two runs of a central
difference. Against the differences with steps of 1 % (and 0.1 % for the
wind and `BETAMAX`):

| direction | energy, 9 January 00 h | energy, 10 January 00 h | mean $H_s$, 10 January 00 h |
|---|---|---|---|
| wind | 2.0 % (2.0 %) | 1.0 % (1.05 %) | 0.92 % |
| `BETAMAX` | 0.92 % (1.2 %) | 0.20 % (0.09 %) | 0.30 % |
| `SDSC2` | 0.19 % | 0.11 % | 0.15 % |
| `NLPROP` | 3.2 % | 1.1 % | 1.1 % |
| `BJGAM` | 4.3 % | 3.4 % | 3.6 % |

(Relative differences between the tangent and the finite difference of
step 1 %, and of step 0.1 % in brackets.) The tangent gives the
derivatives of the aggregates to 0.1–4 %: closest for the parameters that
scale a term smoothly (`SDSC2`, `BETAMAX`), farthest for depth-induced
breaking, whose fraction of breaking waves `QB` has thresholds at
$H_{rms}/H_m$ = 0.2, 0.5 and 1 and acts at a few shallow nodes. The wind
tangent is 1–2 % above the differences over the whole run, as over 3 h,
and the differences of steps 1 % and 0.1 % agree with each other to
$10^{-3}$, so the gap is the mean effect of the discrete choices of the
model, whose share of a difference does not shrink with the step, and not
an error of the tangent. Node by node over 48 h, the two finite differences
agree to $10^{-3}$ at only 19–31 % of the nodes, where the tangent differs
by 0.2 % at the median node after 48 h.

## Calibration of the ST4 parameters

### Stress tables in Rust

The calibration changes `BETAMAX`, which enters the stress tables that
`ww3_grid` computes once and writes in `mod_def.ww3` (`INSIN4`): `TAUT`
(`TABU_STRESS`, the friction velocity as a function of the wave-supported
stress and the wind, which does not use `BETAMAX`), `TAUHFT` and `TAUHFT2`
(`TABU_TAUHF`, `TABU_TAUHF2`, the stress of the high-frequency tail). They
are now computed by the Rust port (`hww3_wave::sources::tabu_stress`,
`tabu_tauhf2`). `hww3-src tables` compares them with the tables WW3 wrote
in the parameter files of the base configuration and of the four
perturbed `mod_def.ww3` files of the M2b finite differences:

| `BETAMAX` | `TAUT` | `TAUHFT` | `TAUHFT2` |
|---|---|---|---|
| 1.75 | 7.9e-8 | 1.13e-6 | 1.13e-6 |
| 1.75175 | 7.9e-8 | 9.7e-7 | 9.7e-7 |
| 1.74825 | 7.9e-8 | 1.07e-6 | 1.07e-6 |
| 1.7675 | 7.9e-8 | 9.1e-7 | 9.1e-7 |
| 1.7325 | 7.9e-8 | 8.7e-7 | 8.7e-7 |

(Largest difference relative to the largest value of the table.) `TAUT`
is iterated in single precision as WW3 does: in double precision it would
differ from WW3's by up to 1.2e-4 in 474 of its 40,401 cells, near the
largest wave-supported stress at winds of 43–45 m/s, where `1 - TAUW /
TAU` cancels in WW3's single precision. The three tables take 1.8 s on 8
threads (16 ms for `TAUT`). The derivative of `TAUHFT` and `TAUHFT2` with
respect to `BETAMAX` agrees with the central difference of the WW3 tables
of `BETAMAX` ± 0.1 % to 6e-4 and 1e-3 of its largest value, the precision
of single-precision tables over a 0.2 % step.

No other entry of the parameter file depends on the calibrated parameters:
`SDSC2`, `NLPROP` (`SNLC1`) and `BJGAM` (`SDBC2`) are scalars read at
each call, and the saturation weights, `CUMULW` and `DIKCUMUL` of
`INSIN4` and the DIA coefficients of `INSNL1` depend on the spectral grid
and other parameters only; the wind factor enters no table. `hww3-src
params` therefore writes the parameter file of any values of the four
parameters from the base file (a file written for `BETAMAX` 1.7675 reads
back with tables equal to the port's to 6e-8, their single-precision
rounding), and `hww3-prop --betamax` builds the tables in memory: a new
value needs neither `ww3_grid` nor the harness.

Two GPU runs of 3 h in double precision from the restart of 8 January,
one with the tables of the parameter file and one with the Rust tables
(Slurm jobs 4182413 and 4182414), give the same $H_s$ to 5.0e-8 m after
1 h and 6.5e-8 m after 2 h (RMS 1.2e-9 and 1.4e-9 m over 371,278 nodes);
after 3 h one node differs by 4.3e-5 m (RMS 1.1e-7 m), a discrete choice
of the source integration flipped by the 1e-6 difference of the tables,
as in the M3 checks. In the GPU job (2 cores) the tables take 9.6 s.

### Set-up of the calibration

- **Control vector.** $x = \ln p - \ln p_0$ for `BETAMAX`, `SDSC2`,
  `NLPROP` and `BJGAM`, from the calibrated values 1.75, −2.2e-5, 2.5e7
  and 0.73; the wind is that of the forcing.
- **Model.** Grid v5 $K=4$ with the calibrated full ST4 and the forcing of
  1–10 January, from the WW3 restart after the first step of the cold
  start of 1 January; `BETAMAX` with the stress tables of the Rust port.
- **Windows.** The cold start takes about four days to settle: the
  altimeter bias of the start run is −93, −51, −24, −12, −1.6, +3.7, +5.3,
  +5.9 and +6.5 cm on 1 to 9 January. Training on 5 January 12 h – 7
  January 12 h (24,153 altimeter blocks, 12,247 buoy hours of $H_s$, 8,292
  of $T_{m02}$), validation on 7 January 13 h – 10 January 00 h (27,935,
  14,989 and 10,096), both from one run from 1 January, so that the
  parameters act from the start.
- **Cost.** $J = \sum (r_{alt}/0.25\,\mathrm{m})^2 + \sum (r_{H_s}/0.30\,\mathrm{m})^2
  + \sum (r_{T_{m02}}/1.0\,\mathrm{s})^2 + \sum (x/0.25)^2$ over the
  altimeter blocks and buoy hours of the window, with the matching of the
  validation scripts (`scripts/cal_obs.py`, which reproduces their scores
  exactly); the sigmas are close to the RMSEs of the start, so that each
  kind of observation counts with its number. Buoys whose $T_{m02}$ RMSE
  exceeds 3 s at the start (2 in training) are kept in the scores and left
  out of the $T_{m02}$ term.
- **Method.** Gauss–Newton with the Jacobian of the residuals from four
  tangent runs per iteration in double precision (`hww3-prop --tangent`,
  `dhs_` and `dt02_`), each component of the step limited to 0.3; a step
  is kept only after a value run confirms that $J$ decreased. No finite
  difference enters the optimisation.

At the start, $J_0 = 47{,}250$ over 44,595 residuals, with
$\partial J/\partial \ln p$ = +25,108 (`BETAMAX`), −10,585 (`SDSC2`),
+5,913 (`NLPROP`) and −118 (`BJGAM`).

As a control only, the tangent derivatives of the cost and scores over
8–10 January (48 h tangent runs from the restart of 8 January) were
compared with the M2b finite differences of the same window, which needed
no new run (`scripts/cal_fdcheck.py`):

| relative difference | `BETAMAX` (0.1 %) | `SDSC2` | `NLPROP` | `BJGAM` |
|---|---|---|---|---|
| $J$ | 4.6 % | 0.4 % | 4.9 % | 7.8 % |
| altimeter bias / RMSE | 0.7 / 0.3 % | 0.15 / 0.2 % | 0.4 / 1.1 % | 3 / 16 % |
| buoy $H_s$ bias / RMSE | 0.9 / 44 % | 0.24 / 0.6 % | 5.5 / 13 % | 4 / 8 % |
| buoy $T_{m02}$ bias | 17 % | 6 % | 3.5 % | 21 % |

The derivatives of $J$ and of the $H_s$ biases agree to a few per cent,
as the M3 checks predict for aggregates, and those of the RMSEs, small
differences of nearly equal numbers, less well; the $T_{m02}$ RMSE
derivatives do not agree (three times larger for `BETAMAX`). The
$T_{m02}$ tangent itself is right: on the 41–44 % of the nodes where the
`BETAMAX` differences of steps 1 % and 0.1 % agree, it agrees with them
to 3–6e-3 at the median node, and the mean of $dT_{m02}$ over these nodes
to 0.3–0.7 % (5 % at 9 January 00 h, where the mean is near zero). $T_{m02}$ weighs the high-frequency tail by $\sigma^2$, and the
tail reset above the cut-off index `NKH` makes discrete jumps that the
finite differences of the aggregates include and the tangent does not.

### Steps of the calibration

- **BJGAM frozen.** After the first step Jean-Marc froze `BJGAM` at 0.73:
  only a few shallow buoys constrain it, the altimeters are insensitive to
  it, its full step of 0.75 in $\ln$ scaled the whole step down to 0.40,
  and its tangent is the least reliable. The trial with `BJGAM` x1.35
  (training $J$ 46,872.0) is kept as information only. The control vector
  is $\ln$ `BETAMAX`, `SDSC2` and `NLPROP`.
- **Step 1.** From the four tangent runs of the start (Jacobian $A_0$
  without its `BJGAM` column): `BETAMAX` x0.976 (1.7077), `SDSC2` x1.183,
  `NLPROP` x1.255, predicted −1.11 %. Its value run (job 4189550) gives a
  training $J$ of 46,628.5 + 1.3 (prior) against 47,249.7: −1.31 %,
  accepted. A WW3 run with these parameters from the cold start (job
  4269275, 64 cores, 1 h 48 min) gives 46,628.4 on training and 68,856.6
  on validation, against the GPU's 46,628.5 and 68,856.7.
- **Frozen Jacobian (options A and C).** Each further step uses one frozen
  Jacobian and one value run, and the calibration stops when an accepted
  step gains less than 0.2 % or no Levenberg–Marquardt damping gives a
  decrease. From step 1, $A_0$ predicts −0.008 % only; the three f64
  tangent runs at step 1 (jobs 4269271 to 4269273, Jacobian $A_1$) change
  $A^TA$ by 4.5 % but the `BETAMAX` column by 26 % (67 % in its $T_{m02}$
  block), which turns the gradient from −157 to +2,406, and the step of
  $A_1$, along the `BETAMAX`–`SDSC2` valley (correlation 0.89), is
  `BETAMAX` x0.867, `SDSC2` x0.876 (limited to 0.3), `NLPROP` x1.216 from
  $p_0$, predicted −1.32 %. $A_1$ is the frozen Jacobian from then on; the
  value run of step 2 is job 4295934.
- **Depth-aware buoy operator, evaluated** (Jean-Marc's request, see
  "Depth-aware buoy operator" below). The cost of the calibration stays the
  nearest-node one until he has seen these results. The cost of every point
  so far under both operators, on the training window (data part; the
  prior adds 0 at $p_0$ and 1.29 at step 1):

  | point | nearest node | depth-aware |
  |---|---|---|
  | $p_0$ | 47,249.7 | 45,265.2 |
  | step 1 | 46,628.5 | 44,728.2 |
  | change | −1.31 % | −1.18 % |

  and on the validation window 69,351.1 → 68,856.7 (nearest node) and
  66,610.9 → 66,224.7 (depth-aware). Step 1 would be accepted under both.
  Under the new operator the step of $A_1$ (its buoy rows through the new
  operator, from the stored `dhs_`, `dt02_` fields) would be `BETAMAX`
  x0.868, `SDSC2` x0.876, `NLPROP` x1.211 from $p_0$, predicted −0.90 %:
  the point of value run 4295934 to 0.005 in the factors, so a switch of
  the cost would not waste that run.
- **Second cost** (Jean-Marc, 24 November: "Oui les deux"). From step 2 the
  buoy terms use the depth-aware interpolation, and each buoy hour has the
  error $\sigma_b = \sqrt{\sigma_0^2 + \sigma_{repr}^2}$, with
  $\sigma_0$ = 0.30 m for $H_s$ and 1.0 s for $T_{m02}$ and a
  representativeness error per class of buoy,
  $\sigma_{repr}^2 = \max(0,\ \overline{r^2} - \sigma_0^2)$ over the
  residuals of the start run on the training window under the new operator
  (`scripts/buoy_sigma.py`). The classes are the land fraction of the cell
  of the buoy's node (GEBCO) and its distance to the coast; a land class
  with fewer than 5 buoys in one of its distance classes is not split by
  distance. The errors are fixed from the start run and kept for the rest
  of the calibration. The altimeter terms are unchanged.

  | class (training window) | buoys | RMS $H_s$ | $\sigma_{repr}$ $H_s$ | RMS $T_{m02}$ | $\sigma_{repr}$ $T_{m02}$ |
  |---|---|---|---|---|---|
  | no land in the cell, node under 5 km from the coast | 65 | 0.303 m | 0.043 m | 0.910 s | 0 |
  | no land, 5 km or more | 135 | 0.245 m | 0 | 0.814 s | 0 |
  | land 0.1–10 %, under 5 km | 15 | 0.237 m | 0 | 1.336 s | 0.886 s |
  | land 0.1–10 %, 5 km or more | 6 | 0.644 m | 0.570 m | 1.770 s | 1.461 s |
  | land 10–30 % | 17 | 0.461 m | 0.350 m | 0.995 s | 0 |
  | land 30 % or more | 14 | 0.506 m | 0.407 m | 0.976 s | 0 |

  The cost of every stored point under the three costs (data part, then
  the prior; training window, and validation for the 216 h runs):

  | point | first cost (nearest node) | depth-aware operator | second cost | prior |
  |---|---|---|---|---|
  | $p_0$, training | 47,249.7 | 45,265.2 | 41,085.9 | 0 |
  | step 1, training | 46,628.5 | 44,728.2 | 40,606.6 | 1.29 |
  | WW3 at step 1, training | 46,628.4 | 44,728.1 | 40,606.5 | 1.29 |
  | step 2, training | 45,800.1 | 44,037.6 | 39,927.7 | 1.22 |
  | $p_0$, validation | 69,351.1 | 66,610.9 | 59,449.2 | |
  | step 1, validation | 68,856.7 | 66,224.7 | 59,201.2 | |
  | WW3 at step 1, validation | 68,856.6 | 66,224.6 | 59,201.1 | |

  The unweighted scores (bias / RMSE) of the same points, the buoys
  through the nearest node and through the depth-aware operator:

  | point | altimeters (cm) | buoy $H_s$, nearest (cm) | buoy $H_s$, depth-aware (cm) | $T_{m02}$, nearest (s) | $T_{m02}$, depth-aware (s) |
  |---|---|---|---|---|---|
  | $p_0$, training | +2.86 / 25.46 | −4.85 / 31.30 | −4.77 / 31.25 | −0.399 / 1.131 | −0.467 / 0.999 |
  | step 1, training | +1.04 / 25.27 | −6.38 / 31.64 | −6.32 / 31.61 | −0.352 / 1.100 | −0.416 / 0.969 |
  | step 2, training | +0.16 / 24.92 | −6.56 / 31.66 | −6.50 / 31.65 | −0.362 / 1.090 | −0.421 / 0.966 |
  | $p_0$, validation | +5.76 / 26.46 | −3.30 / 41.43 | −3.03 / 40.79 | −0.393 / 1.148 | −0.411 / 1.024 |
  | step 1, validation | +3.67 / 26.28 | −5.60 / 41.68 | −5.35 / 41.12 | −0.344 / 1.123 | −0.357 / 0.999 |

- **Step 2.** The value run of the step of $A_1$ (job 4295934: `BETAMAX`
  x0.867, 1.5164, `SDSC2` x0.876, `NLPROP` x1.216 from $p_0$) lowers the
  second cost from 40,607.8 to 39,928.9 with the prior (−1.67 %; −1.78 %
  under the first cost and −1.54 % with the depth-aware operator alone),
  and is accepted. The step of $A_1$ from step 1 under the second cost
  would have been `BETAMAX` x0.871, `SDSC2` x0.876, `NLPROP` x1.198,
  predicted −0.99 %.
- **Step 3.** From step 2, with $A_1$ frozen under the second cost:
  `BETAMAX` x0.784 (1.3725), `SDSC2` x0.649, `NLPROP` x1.137 from $p_0$
  (the `SDSC2` component limited to 0.3 again), predicted −0.73 %. Its
  value run (job 4310651) lowers the training cost with the prior from
  39,928.9 to 39,809.4 (−0.30 %), and is accepted.
- **Validation monitoring.** Since step 3 every value run covers 216 h,
  and the step-2 and step-3 points were run again over 216 h (jobs
  4310674 and 4310675), so that every accepted point has its validation
  cost (`scripts/cal_point.py`, `scripts/cal_monitor_hpc.slurm`); the
  calibration pauses if an accepted step lowers the training cost but
  raises the validation cost. Second cost, data part, with the prior
  apart:

  | point | training $J$ | altimeters / buoy $H_s$ / buoy $T_{m02}$ | prior | validation $J$ |
  |---|---|---|---|---|
  | $p_0$ | 41,085.9 | 25,049.2 / 9,683.7 / 6,353.0 | 0 | 59,449.2 |
  | step 1 | 40,606.6 | 24,673.9 / 10,002.7 / 5,929.9 | 1.29 | 59,201.2 |
  | step 2 | 39,927.7 | 24,007.4 / 10,019.7 / 5,900.6 | 1.22 | 58,094.4 |
  | step 3 | 39,805.2 | 23,630.3 / 10,137.8 / 6,037.1 | 4.19 | 57,475.3 |

  The validation cost falls at every step (−1.07 % at step 3). The
  unweighted scores on the validation window go from p0 to step 3:
  altimeters +5.76 / 26.46 → +1.36 / 25.66 cm, buoy $H_s$ through the
  depth-aware operator −3.03 / 40.79 → −6.46 / 40.75 cm, buoy $T_{m02}$
  −0.411 / 1.024 → −0.395 / 1.000 s. On the training window the buoy terms
  rise again at step 3 while the altimeter term falls.
- **Step 4.** From step 3, with $A_1$ frozen under the second cost:
  `BETAMAX` x0.721 (1.2619), `SDSC2` x0.481, `NLPROP` x1.067 from $p_0$,
  predicted −1.25 % (prior about 10.4); value run job 4318533 (216 h). The
  frozen linearisation is losing accuracy: step 3 gained 0.30 % of the
  0.73 % predicted.

### Result of the calibration (finalised at step 3)

Jean-Marc finalised the calibration at step 3. The final parameters are
`BETAMAX` 1.37249382 (x0.784), `SDSC2` x0.649237785 and `NLPROP`
x1.13718427 from $p_0$, with `BJGAM` 0.73. The final runs are:

- the f64 tangents over 216 h (Slurm jobs 4326004–4326006);
- the WW3 run at step 3 (job 4326007, `$S/runs/m2_week_cal3`);
- the evaluation (job 4326008, `scripts/cal_final2_hpc.slurm`).

Cost (data part, with the prior apart: 0 at $p_0$, 4.19 at step 3) on both
windows. The step-3 column is the GPU run, with the WW3 run in brackets:

| cost | window | $p_0$ | step 3 (WW3) | change |
|---|---|---|---|---|
| first (nearest node, 0.30 m and 1.0 s) | training | 47,249.7 | 45,526.0 (45,525.4) | −3.65 % |
| | validation | 69,351.1 | 66,680.1 (66,679.8) | −3.85 % |
| second (depth-aware, representativeness) | training | 41,085.9 | 39,805.2 (39,804.9) | −3.12 % |
| | validation | 59,449.2 | 57,475.3 (57,474.9) | −3.32 % |
| third (8 buoys left out, Huber) | training | 35,543.4 | 34,432.2 (34,431.9) | −3.13 % |
| | validation | 50,242.2 | 48,173.9 (48,173.6) | −4.12 % |

Unweighted scores, bias / RMSE. The buoys use the depth-aware operator,
with all buoys and without the 8 excluded ones; $T_{m02}$ is through the
moments. The scores of the WW3 run at step 3 are the same to 0.01 cm and
0.001 s:

| | training $p_0$ | training step 3 | validation $p_0$ | validation step 3 |
|---|---|---|---|---|
| altimeter $H_s$ (cm) | +2.86 / 25.46 | −1.38 / 24.73 | +5.76 / 26.46 | +1.36 / 25.66 |
| buoy $H_s$, all (cm) | −4.77 / 31.25 | −6.88 / 31.81 | −3.03 / 40.79 | −6.46 / 40.75 |
| buoy $H_s$, without the 8 (cm) | −7.48 / 27.68 | −9.60 / 28.52 | −5.45 / 38.44 | −8.89 / 38.67 |
| buoy $T_{m02}$, all (s) | −0.467 / 0.999 | −0.438 / 0.975 | −0.411 / 1.024 | −0.395 / 1.000 |

The calibration lowers the altimeter bias and RMSE on both windows and the
$T_{m02}$ error a little. It makes the buoy $H_s$ bias more negative: the
RMSE rises 0.6–0.8 cm on the training window and changes by −0.04 to
+0.23 cm on the validation window. The history of the steps under the
second cost is the table above. Step 4, run for information only, would
give 39,907.6 + 10.35 (training) and 57,232.0 (validation): its training
cost rises while its validation cost falls.

**Posterior.** The posterior comes from $A_1$ (the tangents at $p_0$)
applied to the residuals at step 3 under the second cost (`gn3.json`). It
is a lower bound, since the Jacobian is not refreshed and the errors are
correlated:

| | `BETAMAX` | `SDSC2` | `NLPROP` |
|---|---|---|---|
| standard deviation of $\ln p$ | 0.0083 | 0.0243 | 0.0149 |
| scaled by the reduced $\chi^2$ (0.897) | 0.0078 | 0.0230 | 0.0142 |

The correlations are `BETAMAX`–`SDSC2` +0.88, `SDSC2`–`NLPROP` +0.63 and
`BETAMAX`–`NLPROP` +0.20.

**Tangents at step 3.** The step-3 tangents are not used:

- **Buoy blocks.** They agree with $A_1$, with cosines of 0.84 (`BETAMAX`),
  0.997 (`SDSC2`) and 0.88 (`NLPROP`) for $H_s$, and 0.30, 0.82 and 0.85
  for $T_{m02}$.
- **Altimeter blocks.** They are corrupted by a tangent that grows without
  bound at a few nodes near the sea-ice edge in the training window
  (`scripts/tl_diag.py`):
  - The growth appears first at single nodes of Ungava Bay, Labrador, West
    Greenland and Newfoundland, up to $2 \times 10^6$ m per unit of $\ln p$
    on 6 January 16 h for `BETAMAX`.
  - It reaches the altimeter passes of 7 January 00 h and 08 h: 527 (32
    not finite), 87 and 221 of the 24,153 blocks above 10 m per unit for
    `BETAMAX`, `SDSC2` and `NLPROP`, the largest 917, 73 and 306 m.
    `cal_obs.py` turned tangent values below −900 into NaN along with the
    undefined value; it now removes only the undefined value −999.9.
  - After the training window, the growth starts again in the Wadden Sea
    near the Elbe on 7 January 13–17 h, up to $3 \times 10^{10}$ m on
    7 January 19 h, and spreads along the North Sea coasts in the
    validation window.
- **At $p_0$.** The tangents of $A_1$ have isolated spikes at single nodes
  (Gulf of Finland, $4 \times 10^7$ m on 4 January 01 h), but none reaches
  an observation of the training window: the largest derivative is 7.5 m
  per unit for `BETAMAX`, 2.2 m and 1.4 m for the others.

Steps 1–3 are therefore not affected. The local instability of the tangent
at these parameters is listed among the limits
([Differentiable WW3 study](differentiable.md), section 9.5).

**Where the step-3 tangent grows** (24–25 November, `scripts/tl_probe_hpc.slurm`,
`tl_fix_check_hpc.slurm`, runs under `cal/tlfix`).

- **Source of the blocks.** Tracking the field back hour by hour, the
  derivatives of the 7 January 08 h pass off Newfoundland (−10 to +73 m per
  unit, buoy 44078 −11 to −14 m) come from nodes 238059 and 238060 on the
  coast of northern Labrador (63.9°W 59.74°N):
  - ice 0.44, depth 32 m, $U_{10}$ 16 m/s, $H_s$ 3–4 m;
  - the tangent goes from 5 m on 4 January 19 h to $1.3 \times 10^6$ m at
    22 h;
  - the field then drifts south along Labrador and decays to about
    $10^3$ m off Newfoundland.

  Smaller events start at 221168 south of Greenland (no ice, 194 m) and in
  Ungava Bay.
- **A growth of the model, not an error of the tangent.** A double-precision
  tangent from a restart of 4 January 06 h, probed after each part of every
  step (`hww3-prop run --probe --probe-check`, job 4388139), puts the growth
  in the source step:
  - at 238059 the source step multiplies the tangent by 3 to 4.3 per 600 s
    step on 8 of 108 steps, in the young wind-sea bins (0.25–0.3 Hz,
    downwind) under a swell peak at 0.1 Hz;
  - central differences of the source step along the same tangent
    spectrum (relative step $10^{-4}$) give the same factor where the two
    steps agree: 3.4–4.6 against 3.2–4.3;
  - the two step sizes disagree on 67 of the 108 steps, at discrete
    choices inside the step.

  Leaving parts out of the tangent on these steps gives:

  | Part left out | Factor at 238059 |
  |---|---|
  | none | 3.4 |
  | the DIA | 1.2–1.75 |
  | the wind input | 1.5–2.4 |
  | dissipation | 4–6 |

  The amplification comes from the DIA coupling the bins of a young wind
  sea together with the wind input; dissipation damps it.
- **Leaving out the DIA's tangent** (`--tl-nlcut C`: in the bins where the
  global step times the DIA's own diagonal exceeds $C$). The step-3
  `BETAMAX` tangent over 216 h gives:

  | Cut $C$ | Job | Altimeter blocks above 10 m (training) | Largest | Buoy hours above 10 m | At 238059 | Wadden Sea |
  |---|---|---|---|---|---|---|
  | none | 4326004 | 527 | 3,030 m | 5 | $1.3 \times 10^6$ m | $3.6 \times 10^{10}$ m |
  | 1 | 4374294 | 540 | 3,430 m | 6 | $1.4 \times 10^6$ m | 1–3 m |
  | 0.5 | 4374295 (156 h) | 78 | 67 m | 0 | $3.7 \times 10^4$ m | |
  | 0.25 | 4392466 | 0 | 7.2 m | 0 | 307 m | under 3.1 m |

  With cut 0.25:
  - The validation window is also clean: 0 of 27,935 altimeter blocks
    above 10 m (1,903 without the cut) and one buoy hour (Keeten,
    21.5 m) against 2,260.
  - The cut creates a new spike at one node of the Colombian coast
    ($H_s$ 0.19 m, $3.6 \times 10^6$ m on 8 January 13 h, gone by 18 h).
  - It moves the step-3 derivatives at the training altimeter blocks
    east of 40°W by a relative L2 of 0.66 from the tangent without the
    cut, which has spikes of its own there (0.10 for cut 1).

  The DIA's diagonal misses most of the amplification: at 238059 cut 1
  does not change the factor (3.4), cut 0.5 gives 3.3 and cut 0.25 gives
  2.6. Leaving the DIA's tangent out over the whole spectrum of a node
  where one bin meets the criterion (`--tl-nlnode`) gives 2.8 at 0.5 and
  1.7 at 0.25 (job 4392595).
- **The checks of M3 with cut 0.25** (job 4393681, at $p_0$). On the
  harness nodes of 8 January, the tangent of one source step departs from
  the central differences:
  - by more than $10^{-4}$ on 41–54 of the 1,586–1,854 nodes where the
    model is smooth, up to 100 %, for the spectrum, the wind and the
    parameters; without the cut none departs by more than 9.5e-5;
  - so the cut also triggers at $p_0$, on about 3 % of the nodes.

  Through the time loop, the tangent of $H_s$ departs from the differences
  by more than $10^{-4}$ at many more smooth nodes:

  | Direction | After 1 h, cut 0.25 | After 1 h, no cut | After 3 h, cut 0.25 | After 3 h, no cut |
  |---|---|---|---|---|
  | `WIND` | 1,310 | 18 | 91 | 28 |
  | `INIT` | 5,715 | 7 | 1,436 | 14 |

  The aggregates do not move:
  - Over 3 h the total energy is within 1.5 % (`WIND`) and 0.25 %
    (`INIT`) of the differences, as without the cut.
  - Over 48 h against the differences of M2b, the energy on 10 January is
    within 0.12 % (0.009 % for step 0.1 %) for `BETAMAX` and 1.0 % for
    `NLPROP`, against 0.20 % (0.09 %) and 1.1 % without the cut.
  - The mean $H_s$ is within 0.07 % and 0.9 %, against 0.30 % and 1.1 %.

The cut removes the blow-up but changes the tangent node by node where the
model is smooth, also at $p_0$, so no cut is adopted. The tangents at
$p_0$, which the `BETAMAX`$(U)$ test uses, reach no observation above
10 m per unit (largest 5.0 m for the seven coefficients).

**Cost of the calibration.** 33 GPU runs took 16.6 GPU-hours on the A100s
from 23 November 07:27 to 24 November 11:47 (28 h). The CPU jobs took
8.6 h: operators, evaluations and the WW3 runs.

### `BETAMAX`$(U)$ test (25 November)

**Set-up.** The test runs on the same windows and under the third cost
([Differentiable WW3 study](differentiable.md), section 9.7):
- control: $\ln\beta = \ln 1.75 + \sum_k c_k\phi_k(U)$ on the knots 2–20 m/s,
  with `SDSC2` and `NLPROP`;
- start: $p_0$;
- Jacobian: frozen from the tangents at $p_0$ (`BETAU0`–`BETAU6`, jobs
  4332611–4332617, with those of `SDSC2` and `NLPROP`);
- no cut of the DIA's tangent (Jean-Marc's choice, see "Where the step-3
  tangent grows" above);
- prior: $\mathcal N(0, 0.25^2)$ with a smoothness term of scale 0.1
  (`cal_gn.py --smooth 0.1`);
- each trial is a 216 h value run (`cal_run_hpc.slurm` with `BETAU`,
  then `cal_monitor_hpc.slurm`).

**Steps.**
- Step 1 (job 4411318, 21 min; monitoring job 4411319): the training cost
  with its prior went from 35,543.4 to 34,934.2 (−1.71 %, against −1.68 %
  predicted). The validation data term went from 50,242.2 to 49,772.0
  (−0.94 %).
- Step 2 is predicted to gain only 0.11 %, below the 0.2 % of the stopping
  rule, so it was not run.

The final point is step 1. The posterior below is the inverse Hessian of
the $p_0$ Jacobian at the step-1 residuals: a lower bound.

| Knot (m/s) | 2 | 5 | 8 | 11 | 14 | 17 | 20 |
|---|---|---|---|---|---|---|---|
| $c_k$ | +0.004 | +0.110 | −0.079 | +0.029 | −0.059 | −0.083 | +0.072 |
| $\beta$ | 1.756 | 1.953 | 1.617 | 1.801 | 1.650 | 1.610 | 1.880 |
| sd of $\ln\beta$ | 0.065 | 0.020 | 0.013 | 0.007 | 0.010 | 0.011 | 0.012 |

- `SDSC2` becomes ×1.137 (sd of the logarithm 0.017) and `NLPROP` ×1.209
  (0.016).
- The reduced chi-square is 0.80, which scales the standard deviations by
  0.89.
- The largest correlations are those of `SDSC2` with `NLPROP` (+0.68) and
  with the knots at 8–20 m/s (+0.22 to +0.48).

The curve alternates between the knots by ±0.1 in $\ln\beta$ around a mean
near 1.75. This is far more than its standard deviations, but those are
lower bounds, and the point is a minimum of the linear model at $p_0$. It
lies in another valley than the constant step 3 (`BETAMAX` 1.37,
`SDSC2` ×0.649): it raises `SDSC2` and `NLPROP` instead.

**Cost and scores against $p_0$ and step 3** (third cost, data term; the
prior is 26.4 for $\beta(U)$ and 4.2 for step 3; altimeters and buoy $H_s$
in cm, bias / RMSE, through the operators of the third cost):

| Window | Run | $J$ | Altimeters | Buoy $H_s$ | Buoy $T_{m02}$ (s) |
|---|---|---|---|---|---|
| training | $p_0$ | 35,543.4 | +2.86 / 25.46 | −7.48 / 27.68 | −0.490 / 0.996 |
| | step 3 | 34,432.2 (−3.13 %) | −1.38 / 24.73 | −9.60 / 28.52 | −0.460 / 0.972 |
| | $\beta(U)$ | 34,907.8 (−1.79 %) | +1.23 / 25.05 | −8.45 / 28.30 | −0.460 / 0.971 |
| validation | $p_0$ | 50,242.2 | +5.76 / 26.46 | −5.45 / 38.44 | −0.422 / 1.017 |
| | step 3 | 48,173.9 (−4.12 %) | +1.36 / 25.66 | −8.89 / 38.67 | −0.406 / 0.993 |
| | $\beta(U)$ | 49,772.0 (−0.94 %) | +3.95 / 26.27 | −7.35 / 38.76 | −0.394 / 1.005 |

$\beta(U)$ lowers the cost by half as much as step 3 on the training
window, and by a quarter as much on validation. Compared with step 3:
- it keeps more of the altimeter bias of $p_0$;
- its buoy $H_s$ bias is 1.2–1.5 cm less negative;
- on validation, the altimeter cost carries most of the shortfall.

**Bias by sea state and by wind** (both windows, cm, $p_0$ / step 3 /
$\beta(U)$, `scripts/betau_report.py`, `cal/betau_report` on hpc):

| Observations | Class | Number | $p_0$ | Step 3 | $\beta(U)$ |
|---|---|---|---|---|---|
| buoy $H_s$, by observed $H_s$ | 2–3 m | 3,076 | −15.6 | −21.4 | −18.8 |
| | 4–6 m | 635 | −31.7 | −45.0 | −42.2 |
| | 6–10 m | 131 | −54.7 | −74.6 | −64.8 |
| buoy $H_s$, by wind | 9–12 m/s | 4,750 | −6.7 | −11.8 | −8.9 |
| | 12–15 m/s | 2,026 | −6.8 | −17.3 | −13.2 |
| | 15–20 m/s | 701 | −12.4 | −26.3 | −23.7 |
| | 20–30 m/s | 14 | −22.1 | −54.9 | −13.2 |
| altimeters, by observed $H_s$ | 2–3 m | 17,341 | +7.1 | +3.4 | +5.9 |
| | 4–6 m | 4,084 | −5.0 | −17.1 | −13.2 |
| | 6–8 m | 496 | −24.0 | −43.6 | −32.8 |
| | 8–15 m | 39 | −73.7 | −104.9 | −65.3 |
| altimeters, by wind | 9–12 m/s | 12,422 | +7.8 | +2.6 | +5.9 |
| | 12–15 m/s | 4,586 | +12.0 | +0.6 | +5.6 |
| | 15–20 m/s | 1,389 | +14.2 | −6.1 | +0.9 |
| | 20–30 m/s | 62 | +19.2 | −19.3 | +21.8 |

- At 15–20 m/s, $\beta(U)$ brings the altimeter bias to +0.9 cm, against
  +14.2 at $p_0$ and −6.1 at step 3.
- Its bias for the buoys above 4 m stays between those of $p_0$ and step 3.
- With $\beta$ rising to 1.88 above 20 m/s, the 62 altimeter blocks there
  get +21.8 cm, more than at $p_0$, and the 14 buoy hours −13.2 cm.

Below 6 m and 20 m/s, the RMSEs of a class differ by at most 3 cm
between the three runs.

**What it shows.** With the Jacobian frozen at $p_0$, $\beta(U)$ does not
reach the cost of the constant step 3, although the constant is in its
family: all $c_k = \ln(1.3725/1.75) = -0.243$. The linear model at $p_0$ is
not accurate that far, as in the calibration, which needed the Jacobian at
step 1.

The same Jacobian applied at the residuals of step 3, from step 3 written
in the $c_k$ (CPU job 4415564, no run), predicts a further −1.31 % of the
cost with its prior (34,442.0 to 33,989.3):
- $\ln\beta$ moves by +0.17, +0.20, −0.03, +0.05, −0.01, −0.06 and +0.11
  at the knots;
- `SDSC2` moves to ×0.636.

**From step 3** (Jean-Marc: "Oui, l'essayer"). The step predicted above
was run (value run job 4415590, 20 min; monitoring job 4415591), to be
accepted only if both windows fall against step 3:

- **Point.** $\beta$ is 1.625, 1.684, 1.336, 1.438, 1.355, 1.298 and
  1.525 at the knots, with `SDSC2` ×0.636 and `NLPROP` ×1.136.
- **Training.** The cost with its prior went from 34,442.0 to 34,048.6
  (−1.14 %, against −1.31 % predicted). The data term went from 34,432.2
  to 34,018.0 (−1.20 %), or −4.29 % from $p_0$.
- **Validation.** The data term rose from 48,173.9 to 48,592.1 (+0.87 %;
  −3.28 % from $p_0$, against −4.12 % at step 3).
  - The altimeters rose by 401 and $T_{m02}$ by 151.
  - The buoy $H_s$ fell by 134.

The step is therefore rejected, and with it the test of `BETAMAX`$(U)$ on
these windows ends at step 3.

| Window | Run | $J$ | Altimeters | Buoy $H_s$ | Buoy $T_{m02}$ (s) |
|---|---|---|---|---|---|
| training | from step 3 | 34,018.0 (−4.29 %) | +0.48 / 24.48 | −8.23 / 28.34 | −0.481 / 0.968 |
| validation | from step 3 | 48,592.1 (−3.28 %) | +3.14 / 25.77 | −7.64 / 38.41 | −0.425 / 1.004 |

(Percentages from $p_0$; the other rows are in the table above.)

- **Training scores.** Against step 3, the training altimeters go from
  −1.38 / 24.73 to +0.48 / 24.48 cm and the buoys from −9.60 / 28.52 to
  −8.23 / 28.34 cm.
- **Validation scores.** The altimeters move away from step 3
  (+1.36 / 25.66 to +3.14 / 25.77 cm), while the buoys improve
  (−8.89 / 38.67 to −7.64 / 38.41 cm).
- **Shape of the curve.** It repeats that of the first point, 0.15–0.23
  lower in $\ln\beta$ from 5 m/s up: high at 5 and 11 m/s, low at 8 and
  14–17 m/s, rising at 20 m/s. Both points come from the same Jacobian at
  $p_0$, so the shape is not independent evidence.
- **Bias by class** (both windows, cm, from step 3, against step 3):
  - altimeters at 15–20 m/s: −6.3 against −6.1;
  - altimeters at 12–15 m/s: +2.5 against +0.6;
  - buoys above 6 m: −67.9 against −74.6;
  - the RMSEs are equal or lower in 22 of the 31 classes, most in the
    rare ones: altimeters above 8 m (90.1 against 116.7 cm, 39 blocks)
    and above 20 m/s (51.5 against 60.0 cm, 62 blocks);
  - the altimeters at 1.5–3 m, where most blocks are, get 0.4–0.5 cm
    more RMSE (19.3 and 23.2 against 18.9 and 22.7 cm).

`scripts/betau_report.py` now draws several curves together
(`bu1:gnu1.json:x+bus3:gnus3.json:x_new`); the report of the four runs is
in `cal/betau_report_s3` on hpc (CPU job 4415662).

**Cost.** Each 216 h value run took 0.33–0.34 GPU-hours. The GPU tally
from 23 November 07:27 is 35.4 GPU-hours.

## Preprocessing in Rust (in progress)

The aim is a GPU run that needs no WW3 tool; WW3 then only makes the
references.

### Parameter file of the source terms

`hww3_wave::ww3grid` builds the whole parameter file of the source terms
from the namelists of the run and the grid, the source-term part of
`ww3_grid` (`W3GRID`): the namelist defaults of the HWW3 switches (ST4, NL1,
LN1, BT4, DB1, MISC) and the derived constants, the spectral grid, the
saturation weights and `CUMULW` of `INSIN4`, the DIA geometry, weights and
index tables of `INSNL1`, the friction-factor table of `TABU_FW` (with the
Kelvin functions of `KZEONE`), the error-function table of `TABU_ERF`, the
critical Shields number of the sediment, and the stress tables. `hww3-src
build-params` writes it and compares it record by record with the file of
the harness (grid v5 $K=4$, calibrated namelists):

| records | difference relative to the largest value |
|---|---|
| 121 of 137 (the 48 integer records included) | identical |
| `AF11` | 1.5e-6 |
| `TAUHFT`, `TAUHFT2` | 1.1e-6 |
| `SSTXFTWN`, `SWG`, `AWG`, `CUMULW`, `SATWEIGHTS`, `FTE`, `FACP`, `FACTI2`, `SLNC1`, `SNLC1`, `FFXFM`, `ERFTABLE`, `TAUT` | 6e-8 to 1.8e-7 |

No record is missing or shaped otherwise; the differences are the
single-precision round-off of WW3 (powers and logarithms computed in
double precision here). Two WW3 details are kept: `TABU_ERF` computes
`(1 + erf x) / 2` (its `2**(1/2)` is 1 in integer arithmetic), and the
saturation tables have the `2 (NTH/2 - 1) + 1` rows that `W3DIMS`
allocates, of which `INSIN4` fills 17. The file takes 1.8 s to build.

### Initial state

- **Start from calm conditions.** Without a restart file, WW3 with the
  `LN1` switch starts from calm conditions (`W3IORS`: `TYPE = 'CALM'`;
  `W3INIT`: zero spectra; levels and ice zero until the first fields are
  read). `hww3-prop run --restart calm --start T` does the same
  (`Restart::calm`). After a cold or calm start `W3IORS` leaves
  `TLEV(1) = TICE(1) = -1`, and the first step of `W3WAVE` then applies the
  first level and ice fields whatever their time (action 'I'); the first
  call of `W3FLDG` keeps a field at the start time. Two fixes made
  `hww3-prop` do the same: its times of the level and ice fields start
  unset, and its first step does not read the level and ice files again
  (without them the ice of 1 January 00 h was never applied, and the first
  step gave up to 1.02 m of $H_s$ under the ice of Hudson Bay, Slurm jobs
  4191399 and 4269338). Against the WW3 restart written after the first
  step of the cold start of 1 January (`runs/m2_week_start_full`, 00:10,
  `scripts/pre_restart_hpc.slurm`, job 4295937) the $H_s$ of the first step
  differs by at most 1.7 mm (one of 370,000 nodes above 1 mm, 0.7 mm at
  most at the coast).
- **Restarts.** `Restart::write` writes a `FULL` restart in the layout of
  `W3IORS`: header, time, one record per spectrum, the times of the
  fields (`TLEV`, `TICE`, `TRHO`, `TIC1`, `TIC5`), then `WLV`, `ICE`,
  `MAPSTA + 8 MAPST2`, `UST`, `USTDIR`, `ASF`, `FPIS`, and whatever follows
  (the coupling fields, none here). The reader now keeps all of these.
  `hww3-prop restart-copy` reads and writes back the WW3 restarts of 1
  January 00:10 and 8 January 00:00 (1.97 GB each, `NSEA` 424,123, `NSPEC`
  1152) identical byte for byte, in 1.4 s. `hww3-prop run --restart-out
  FILE` writes the state at the end of a run: the spectra in single
  precision, levels, ice and masks with the times of their fields, and
  u* of the last source call. `USTDIR` and `FPIS` are written as zero and
  `ASF` as one, since the GPU model recomputes u* at each source call and
  does not keep the others. A run continues from such a restart with
  `--itstep0` set to the `ITSTEP` the first run prints (its parity sets the
  order of `W3KTP3`). A run from calm to 01 h in two pieces joined by a
  restart at 00:10 gives the $H_s$ and $T_{m02}$ of the run in one piece
  to 2.4e-7 m and 9.5e-7 s (at 288 nodes, identical elsewhere): the run in
  one piece keeps the wavenumbers of the level update on the device at the
  first step, and the restarted run computes them on the host. Reading
  such a restart with WW3 has not been tried.

### Forcing fields (`ww3_prnc`)

`hww3_wave::prnc` ports what `ww3_prnc` does on an HWW3 grid
(`FORCING%GRID%LATLON = T`, an unstructured target): for each node, the cell
of the regular source grid that holds it, bilinear weights (scaled by the
factor `EFAC`, which differs from 1 only outside the source grid),
`INTERP`, which leaves out corners at `_FillValue` and divides by the sum
of the other weights, and for the vector fields the rescaling of the
interpolated vector to the interpolated root-mean-square speed for the
wind (`WNT2`) or mean speed for the currents (`CRT1`), at most by 1.5. The
arithmetic is in single precision in WW3's order, including
`MOD(360 + x - X0I, 360)` on the longitudes. The crate has no netCDF reader
and the hpc cluster has no Rust netCDF crate, so `scripts/prnc_nc.py`
reads the netCDF file as `ww3_prnc` does (values as stored, `_FillValue`,
`scale_factor`, `add_offset` and the axes in single precision) and
`hww3-pre prnc` does the rest and writes the forcing file
(`ww3io::FieldWriter`, the format of `W3FLDG`). With `--msh` the node
coordinates are read from the gmsh file, as WW3 reads them; on grid v5
$K=4$ they agree with the grid's raw arrays to 5e-11 degree.

On synthetic fields on a 0.25-degree grid covering grid v5 $K=4$, a water
level linear in longitude and latitude comes out within 7.5e-7 m of its
exact values at the 424,123 nodes, and a uniform wind (10, 5 m/s) within
1.9e-6 m/s; each file takes 0.2 s.

The netCDF inputs of the January runs of grid v5 $K=4$ (ERA5 wind and ice,
Copernicus Marine currents and sea level, 597 x 337 and 1789 x 1009
points; the boundary station files below) were copied from Datarmor
(`runs/natl_v5k4_jan2026_{cur,bnd}`, 329 files, 7.7 GB, md5 checked) to
`$S/pre` on the hpc cluster. From them `scripts/pre_inputs_hpc.slurm`
builds the forcing files of 1 to 11 January in Rust and compares them with
those of `ww3_prnc` used by the M2 runs (Slurm jobs 4197806 and 4201526):

| field | times compared | values that differ | time to build |
|---|---|---|---|
| wind (2 components) | 220 | 0 of 2 x 93,307,060 | 2.4 s |
| ice | 1 | 0 of 424,123 | 0.1 s |
| currents (2 components) | 220 | 0 of 2 x 93,307,060 | 11.0 s |
| water level | 220 | 0 of 93,307,060 | 3.4 s |

The files are identical to the bit: the single-precision arithmetic of
`ww3_prnc` is reproduced, including the rescaling of the wind (`WNT2`) and
of the currents (`CRT1`).

### Boundary spectra (`ww3_bounc`)

`hww3_wave::bounc` ports `ww3_bounc` in `WRITE` mode: the spectra of the
station files (on the spectral grid of the model, as
`scripts/boundary_spectra.py` writes them, so that the conversion `W3CSPC`
is not needed; `hww3-pre bounc` stops if it would be) decoded and divided by
$2\pi$, and for each open-boundary node, in sea-point order, the two
nearest stations by `DIST_HAVERSINE` (in degrees, single precision) with
the linear weights of `INTERP = 2`. `ww3io::NestWriter` writes the boundary
file (`nest.ww3`), and `scripts/bounc_nc.py` reads the netCDF files of
the stations. With four synthetic stations, the file built for grid v5
$K=4$ has the header (`NK` 32, `NTH` 36, `XFR` 1.1, `FR1` 0.0373, `TH(1)`
0) and the 1052 boundary points of the WW3 file of the M2 runs identical
to the bit (coordinates and order), in 0.2 s.

From the 318 station files of the WW3 run (31 days every 3 h), the file
built in 0.9 s is identical to the bit to that of `ww3_bounc` over the 74
times the M2 file holds (1 to 10 January): header, boundary points,
station indices, weights and spectra. One detail of gfortran was needed:
`MIN(1., MAX(0., COS1))` returns its first argument unless the second is
larger, so a `COS1` of -0, which occurs at 25 nodes on 45 E that lie on
the meridian of their nearest station, gives a weight of -0 in WW3's file.
`station_weights` evaluates it in the same order.

### A run with Rust inputs only

`scripts/pre_allrust_hpc.slurm` runs grid v5 $K=4$ for 216 h from calm
conditions on 1 January with the parameter file of `build-params` and the
forcing and boundary files above, so with no WW3 tool, and
`scripts/pre_allrust_cmp_hpc.slurm` compares it with the start run of the
calibration (`cal/it0_value`, from WW3's restart after the first step with
the parameter file of the harness): hourly $H_s$ node by node
(`scripts/pre_hs_cmp.py`) and the cost and scores of the calibration on its
training and validation windows (Slurm jobs 4269339 and 4269340, 19 min
for the run). Over the 216 hours the median $|\Delta H_s|$ is at most
9.5e-7 m and its 99th percentile at most 3.8e-5 m; the largest difference,
0.30 m at node 238271 in Ungava Bay on 5 January, is the amplification of
round-off seen there before. The cost and all the scores are identical to
the digits printed on both windows (training $J$ 47,249.7, validation
69,351.1). A GPU run therefore needs no WW3 tool: the parameter file, the
forcing, the boundary spectra and the initial state all come from Rust.

## Optimisation (from 25 November)

### Profile of the present build

The profile uses `scripts/opt_profile_hpc.slurm`, on one A100 of hpc,
from the restart of 1 January 00:10 with the full ST4 on grid v5 $K=4$.
- Nsight Systems ran on 6 h in f32 and on 3 h in f64 and for the
  `BETAMAX` tangent (job 4415688), with the same runs under `--profile`.
- The units per cube of the source kernel were scanned (job 4416363).
- Nsight Compute cannot read the counters on hpc: `ERR_NVGPUCTRPERM`
  (job 4416360). The performance counters are reserved to the
  administrators.

**Host.** After the start the GPU is busy 98.0 % (f32), 97.3 % (f64) and
99.9 % (tangent) of the time:
- The gaps are the compilation of a kernel at its first launch: 1.0 s in
  f32, 4.2 s in f64, and 13 s for the tangent kernel.
- The 657 launches per step take 2.7 ms of the host.
- The copies of the forcing take 0.2 s per 6 h, and the hourly outputs
  take nothing measurable.

The timers `iter` and `outputs` of a run without `--profile` measure the
wait for the GPU at the download of the sub-cycling count and of the
outputs, not work of their own. Overlapping the outputs with the steps can
therefore gain less than 1 %.

**Kernels.** In the 6 h runs (the first hours are the most expensive, 1.66
s per step for the source kernel in f32 against about 0.6 s on average
over 216 h):

| Kernel | f32 | f64 | f64 tangent |
|---|---|---|---|
| source terms (`srce`, `srce_tl`) | 88.7 % (1.66 s per step) | 95.6 % (8.75 s) | 95.1 % (14.0 s) |
| geographic sweeps (`geo_tri_x4`, `geo_node_x4`) | 8.5 % (0.56 ms per sweep) | 3.3 % (1.05 ms) | 3.8 % (1.94 ms) |
| `W3KTP3` (`ktp3_k`, `ktp3_theta`) | 1.7 % | 0.7 % | 0.8 % |
| the rest | 1.1 % | 0.4 % | 0.3 % |

With 287 sweeps per step, the propagation costs about 0.21 s per step in
f32, 0.40 s in f64 and 0.72 s for the tangent. Over the 216 h runs, the
source kernel is therefore about 73 % of a value run in f32
(0.81 s per step), 87 % in f64 and 83 % of a tangent run.

**Source kernel.** One cube of 128 units per node holds the spectrum and
its work arrays in shared memory:

| Kernel | Registers per unit | Shared memory per node | Nodes per SM | Warps per SM (of 64) |
|---|---|---|---|---|
| `srce` f32 | 128 | 41,840 B | 3 (shared memory) | 12 |
| `srce` f64 | 254 | 83,680 B | 1 (shared memory) | 4 |
| `srce_tl` f64 | 255 | 156,832 B | 1 (shared memory) | 4 |

The results are identical to the bit for 64, 128, 256 and 512 units:

| Units | f32, 2 h (s per model day) | f64, 1 h (s per model day) |
|---|---|---|
| 64 | 525 | 2026 |
| 128 | 448 | 1690 |
| 256 | 969 | 1748 |
| 512 | 903 | 1858 |

More units per cube do not help:
- In f32 with 256 units, the registers allow only 2 nodes per SM instead
  of 3, and the run is 2.2 times slower.
- In f64, with one node per SM in every case, it changes little.
- With 64 units the parallel loops take twice as long, yet the run is only
  17 % (f32) to 20 % (f64) slower. About 80 % of the kernel's time is
  therefore spent in parts that more units do not shorten: the scalar parts
  on unit 0 (mean parameters, $u_*$ by Newton iterations, the tail stress,
  the dynamic step) and the loops over 32 frequencies or 36 directions.

The kernel's speed follows the number of nodes an SM holds at once, set by
the shared memory.

### Optimisation steps (checkpoint of 25 November)

Each step is checked with `scripts/opt_bench_hpc.slurm` against runs of
the present build (commit 35ecb15, kept in `opt/bin_base`) over the same
hours, from the restart of 1 January 00:10 with the parameters of p0:
- the week in f32 (216 h) and 48 h in f64, compared hour by hour
  (`scripts/opt_compare.py`), with the costs and scores of the
  calibration windows for the week (`scripts/opt_scores_hpc.slurm`);
- 24 h of the f64 `BETAMAX` tangent ($H_s$, its tangent and that of
  $T_{m02}$);
- the M3 check of the source tangent (`hww3-src check-tl`) on the harness
  nodes of grids v5 $K=4$ and v4, with both builds.

All runs used A100-SXM4-80GB GPUs. The calibration runs at p0 (`cal/it0_value`,
`cal/it0_BETAMAX`) are not the reference: they came from an older build,
which differs from 35ecb15 at round-off (2 of 216 hours identical to the
bit in f32). `it0_BETAMAX` also covers 156 h, against the first 24 h here,
which cost the most.

The steps, in seconds per model day (1 to 10 January for f32, the first
two days for f64, the first day for the tangent):

| Step | f32, 216 h | f64, 48 h | f64 tangent, 24 h | Against 35ecb15 |
|---|---|---|---|---|
| present build (35ecb15) | 124.1 | 505.1 | 931.4 | – |
| 1. wind-sea flags in bytes | 108.9 | 304.5 | 939.0 | identical to the bit (all three) |
| 2. powers of `W3SPR4` out of its serial loop | 101.7 | 266.4 | 843.8 | round-off |
| 3. constants of `W3SIN4` out of its loop, its sums one per unit | 117.3 | 265.2 | 798.0 | identical to the bit to step 2 |
| 4. tangent: source sums per unit, DIA in two halves | – | – | 590.2 | round-off |

This checkpoint keeps steps 1–2 in the value kernel: 101.7 s per model day
in f32 and 266.4 s in f64, against 124.1 and 505.1. The tangent kernel
has steps 1–4: 590.2 s against 931.4.

**1. Flags in bytes.** The 0/1 wind-sea flags of `W3SPR4` and `W3SIN4`
took one float per bin. In bytes the source kernel needs 38,384 B instead
of 41,840 B per node in f32 and 75,616 B instead of 83,680 B in f64. An SM
then holds 4 nodes instead of 3 in f32, and 2 instead of 1 in f64 (with
254 registers per unit the registers also allow 2). The results are
identical to the bit in f32 (all 216 hours), in f64 and for the tangent.

**2. Powers out of the serial loop of `W3SPR4`.** The sums of `W3SPR4`
over the frequencies run on unit 0. The powers of $\sigma$ and $k$ in
them (`WNMEANPTAIL`, `WNMEANP`) and the ratio `DDEN / CG` are now
computed once per frequency by units 0–31 at the start of the kernel. The
formulas are those of the loop, but the results differ at round-off:
- In f64, $H_s$ differs by at most 9.0e-14 m over 48 h (hourly RMS 1e-16).
- In f32, the differences grow as round-off does over a week, as between
  any two f32 builds: hourly RMS median 9.4e-7 m, 99th percentile
  1.2e-4 m, largest 9.7e-2 m at one node and hour.
- The costs of the calibration windows differ by at most 6e-3 out of
  35,000–69,000 (2e-7), and every bias and RMSE is the same to the
  printed digits.

The three slots per frequency bring the kernel to 38,768 B per node in
f32 and 76,384 B in f64, still 4 and 2 nodes per SM.

**3. Constants out of the loop of `W3SIN4`: in the tangent only.** The
constants per frequency of `W3SIN4` (`CM`, `CONST2`, `CNST`, `ln(k z0)`,
the swell coefficients) computed before its loop, and its four sums over
the directions (`SX`, `SY`, `SNX`, `SNY`) one per unit, give results
identical to the bit to step 2 (all 216 hours in f32, 48 hours in f64).
They make the f32 value kernel slower, though: 117.3 s per model day
instead of 101.7, which the source kernel alone confirms (below). In f64
they change nothing (265.2 s). The value kernel of 35ecb15 already used
128 registers per unit in f32, the most that allows 4 nodes per SM. Step
3 needs 168 (read from its PTX, see the second checkpoint below), so an
SM holds 3 nodes instead of 4. In the tangent kernel step 3 takes the
first day from 843.8 to 798.0 s, also identical to the bit to step 2, so
it is kept there, in slots of its own.

**4. The tangent kernel in two nodes per SM.** The tangent kernel needed
156,832 B per node, so an SM held one node. Now:
- The source sums `VS`, `VD` and their tangents are kept by the unit of
  each bin (bins $r\,n_u + t$ of unit $t$, in registers or local
  memory) instead of shared memory; `W3SIN4` stores its input in a
  scratch buffer and adds it after its loop over the frequencies.
- The DIA computes each of its three passes first for the values, then
  for the tangents, in the same two buffers instead of four, and keeps its
  diagonal per unit.
- `W3SDS4` uses the DIA buffers as scratch.

A node then needs 76,192 B, and an SM holds 2 nodes, as the 255
registers per unit allow. The tangent run takes 590.2 s per model day
instead of 931.4 s (−37 %), with differences at round-off: $H_s$ at most
3.1e-14 m, its tangent at most 9.9e-11 m, that of $T_{m02}$ at most
2.0e-9 s. The M3 check of the source tangent of this checkpoint gives
the same medians, 99th percentiles and maxima against the central
differences as 35ecb15 in every direction on both harnesses (job
4447407). The first launch of the tangent
kernel takes 5.3 s instead of 1.1 s: this is the time `check-tl` gives for
the spectrum direction, and it includes the compilation of the kernel.

**Not kept: the same change in the value kernel.** With the source sums
per unit (on top of step 3), the value kernel needs 25,712 B in f32 and
50,272 B in f64 but is not faster:
- f64: 268.6 s per model day, against 265.2 s for step 3 (2 nodes per SM
  either way, set by the registers);
- f32: 143.2 s, against 117.3 s for step 3;
- with 64 units, for more nodes per SM: 283.2 s in f64 and 174.5 s for
  48 h in f32, against 145.4 s for 35ecb15.

**The source kernel alone.** One global step of the value kernel on the
1,869 harness nodes of grid v5 $K=4$ repeated 200 times (373,800 nodes,
`hww3-src check-gpu`), the builds one after the other on the same GPU,
twice (job 4444291):

| Build | f32 (s) | f64 (s) |
|---|---|---|
| 35ecb15 | 0.805, 0.815 | 2.831, 2.828 |
| step 1 | 0.713, 0.694 | 1.691, 1.689 |
| steps 1–2 | 0.647, 0.641 | 1.459, 1.641 |
| steps 1–3 | 0.804, 0.746 | 1.446, 1.596 |

Between two rounds the same build can differ by 10 % in f64. The
differences against WW3 and the CPU transcription are the same as for
35ecb15 to the printed digits.

### Optimisation steps (second checkpoint, 25 November)

The checks are those of the first checkpoint, against 35ecb15. In
addition, the registers per unit of the source kernels are read in two
ways, which give the same counts:
- from the PTX that CubeCL compiles (`scripts/opt_ptx_hpc.slurm`): it is
  written under `CUBECL_CUDA_DUMP_PTX`, then read by `ptxas -v` of CUDA
  13.0 for sm_80. The NVRTC that CubeCL loads writes PTX version 9.2,
  newer than this `ptxas` reads, so the script lowers the version line
  to 9.0; the sm_80 instructions are the same.
- at run time under Nsight Systems (`scripts/opt_profile_hpc.slurm`,
  `PART=regs`, job 4460958).

| Kernel | Registers per unit | Shared memory per node | Nodes per SM |
|---|---|---|---|
| `srce` f32, first checkpoint | 128 (about 110 B spilled) | 38,768 B | 4 |
| `srce` f32 with step 3 | 168 | – | 3 (registers) |
| `srce` f64 | 255 | 76,384 B | 2 (registers and shared memory) |
| `srce_tl` f64 before step 4 | 255 | 156,832 B | 1 |
| `srce_tl` f64, first checkpoint | 168 (4.3 KB of stack per unit) | 76,192 B | 2 (shared memory) |
| `srce_tl` f64, step 6 | 168 | 53,360 B | 3 |

CubeCL declares only the size of the cube (`.reqntid 128`). The compiler
chooses the registers, and neither `--maxrregcount` nor a launch bound can
cap them without changing CubeCL.

The steps, in seconds per model day, over the same hours as in the first
checkpoint:

| Step | f32, 216 h | f64, 48 h | f64 tangent, 24 h | Against the step before |
|---|---|---|---|---|
| present build (35ecb15) | 124.1 | 505.1 | 931.4 | – |
| first checkpoint (ce6470f) | 101.7 | 266.4 | 590.2 | – |
| 5. square roots in the serial step of `W3SIN4`; tangent of `W3SDB1` kept finite in f32; 4 tangent slots per frequency | 98.8 | 246.1 | 565.8 | round-off |
| 6. tangent: the DIA and `W3SDS4` work in the buffers of the spectrum | – | – | 509.9 | identical to the bit |

**5. Square roots in the serial step of `W3SIN4`.** In the loop of
`W3SIN4` over the frequencies, unit 0 computes the stress left after the
lower frequencies: $|\tau|^{1/4}$ with POW, then the direction with ATAN2
and its cosine and sine. Now $r = \sqrt{\tau_x^2 + \tau_y^2}$,
$|\tau|^{1/4} = \sqrt{r}$, and the cosine and sine are $\tau_x / r$ and
$\tau_y / r$ (1 and 0 at $r = 0$). The value and tangent kernels both do
this. The results differ at round-off:
- In f64, $H_s$ differs from 35ecb15 by at most 1.5e-12 m over 48 h
  (hourly RMS median 1.4e-16 m).
- In f32 over the week: hourly RMS median 3.7e-6 m, 99th percentile
  1.6e-4 m, largest 0.116 m at one node and hour. Step 2 gave 9.4e-7,
  1.2e-4 and 9.7e-2 m.
- The costs of the calibration windows differ from 35ecb15 by at most
  2e-3 (35,543.385 against 35,543.386), and every bias and RMSE is the
  same to the printed digits (job 4460587).
- The M3 check of the source tangent (job 4453109) has no smooth node
  above 1e-4 in any direction. The medians and 99th percentiles are the
  same to two digits; the largest change is the maximum of `SDSC2` on
  grid v4, 5.6e-7 to 6.1e-7.

The source kernel alone takes 0.614–0.636 s instead of 0.653–0.654 s per
step in f32, and 1.335–1.339 s instead of 1.446–1.447 s in f64 (−7.5 %,
job 4453110).

Two changes of the tangent kernel come with step 5:
- `kd` keeps only the tangents of the four slots per frequency that have
  one (`EB`, `EB2`, `CNST`, `BTH0`), instead of 19. This is identical to
  the bit and gains nothing alone (592.0 s, job 4433135), but a node then
  needs 72,352 B, which leaves room for step 6.
- The tangent of `W3SDB1` stays finite in f32 (below). In f64 nothing
  changes: the 24 hours of the tangent are identical to the bit to the
  build without it.

**6. The tangent kernel in three nodes per SM.** At 168 registers per
unit, the registers allow 3 nodes per SM. Shared memory set the limit at
2. Now:
- The extended spectrum of the DIA and its tangent (`UE`, 10,944 B each
  in f64) are held in the buffers of the spectrum and its tangent. These
  are one slot longer than the extended grid; the extra slot holds the
  zero of the bins below the grid. The DIA reads the spectrum only to
  build `UE`.
- The crest lengths of the cumulative term of `W3SDS4` and their tangent
  are held in the same buffers. `W3SDS4` reads the spectrum of other bins
  only for the saturation, before.
- Meanwhile the unit of each bin keeps the spectrum and the tangent of
  its bins, and puts them back after.

A node needs 53,360 B. The tangent run takes 509.9 s per model day
instead of 565.8 s (−10 %), identical to the bit: all 24 hours of $H_s$,
its tangent and that of $T_{m02}$ (job 4461185). The M3 check gives the
same figures as step 5. The gain is smaller than the third node suggests.
The tangent kernel also keeps 4 KB of local memory per unit, which the
extra node shares. The first launch now compiles in 14.4 s, against 5.4 s.

**Not kept: the f32 tangent.** The f32 tangent was not finite after one
hour. Two diagnostics of `hww3-prop run --probe` found where:
- `HWW3_NANCOUNT` counts the nodes with a non-finite spectrum or tangent
  after each part of the step.
- `HWW3_TLFREEZE` leaves out the tangent of one source term (the masks of
  `P_TLFREEZE`); `opt_bench_hpc.slurm` runs it as the case `freeze32`.

The tangent became non-finite in the source step, at 1,282 nodes in
frequencies 12–19 after the first step. The propagation then spread it to
387,701 nodes in six steps. Only leaving out the tangent of `W3SDB1`
removed it (job 4440818). There, below $B = 0.2$, `QB` is 0 and $BB^2$
underflows to 0 in f32 ($H_{rms}$ about 1e-9 m), so the term
`QB FMEAN dBB / BB^2` was 0/0; it is now 0.

The f32 tangent is then finite everywhere and takes 302.4 s per model
day over the first day (326.9 s after step 6, identical to the bit). It
is not accurate enough. Against the f64 tangent:
- Over the first day, the tangent of $H_s$ differs by a relative RMS of
  7.1e-4 (median hour) and 3.5e-2 (largest hour); that of $T_{m02}$ by
  1.2e-2 and 0.15.
- Over the training window of the calibration (156 h, job 4460585,
  against `cal/it0_BETAMAX`), the tangent of $H_s$ reaches 9e9 at some
  nodes.
- At the observations of cost 3, the derivatives differ by a relative
  L2 of 0.26 for the altimeters (correlation 0.91), 7.8e-3 for buoy
  $H_s$ and 4.4e-2 for buoy $T_{m02}$.
- $\partial J / \partial \ln$ `BETAMAX` is 1.921e4 against 1.797e4
  (+6.9 %). The altimeter column of the Jacobian differs by 25.8 %
  (cosine 0.967), and $A^T A$ by 2.8 %.

The f32 tangent is therefore not used. The f64 reference comes from an
older build; over 24 h two f64 builds differ by at most 4e-9 m in the
tangent of $H_s$.

**Where the value kernel spends its time.** A part that only reads its
inputs and rewrites the same values can be called twice per dynamic step
with results identical to the bit. The extra time is its cost (builds for
timing only, not kept; jobs 4459038 and 4462234). `W3SIN4` without
accumulation and `W3SPR4` are such parts; the DIA and `W3SDS4` are, when
called again with a scale of 0. In f64, on the 373,800 nodes of the
source kernel alone:

| Part | Extra time per step | Share of 1.33–1.35 s |
|---|---|---|
| `W3SIN4` (inside the loop) | 0.36–0.38 s | 27 % |
| `W3SDS4` | 0.10–0.11 s | 8 % |
| DIA (`W3SNL1`) | 0.08 s | 6 % |
| `W3SPR4` | 0.03–0.16 s | 2–10 % |

In f32 the same builds vary by 5–10 % between rounds. `W3SIN4` adds
0.10–0.21 s of 0.63–0.72 s, and the DIA nothing measurable. The serial
loop of `W3SIN4` over the frequencies is the largest single part.

### Optimisation done (26 November)

The last items of the plan, checked as before against 35ecb15 and the
second checkpoint (778b1e7). They end the optimisation.

**Not kept: the geographic sub-step as one kernel.** In the fused kernel,
each node computes the upwind values $\tilde U_T$ of its triangles itself
instead of reading them from the triangle kernel's array. Each triangle is
then computed by its three nodes, and each node gathers the spectra of its
triangles' nodes. Under Nsight Systems over 2 h (job 4491113) the fused
kernel is slower than the two kernels it replaces:

| Run | Fused | Two kernels (triangles + nodes) |
|---|---|---|
| f32 | 0.649 ms | 0.560 ms (0.292 + 0.268) |
| f64 | 1.435 ms | 1.043 ms (0.534 + 0.509) |
| f64 tangent | 2.393 ms | 1.919 ms (0.497 + 0.497 + 0.924) |

It was not identical to the bit either: the compiler contracts the
multiplications and additions of the same expression differently in the
two kernels (job 4491114). The two kernels stay.

**Option: parallel reductions (`--par-red`, off by default).** The value
source kernel can do three reductions over the warps instead of on unit 0:
- the four sums of `W3SIN4` over the directions, at each step of its loop
  over the frequencies (`plane_sum`);
- the sums of `W3SPR4` over the 32 frequencies (`plane_sum`);
- the minimum of the dynamic step (`plane_min`, which gives the same
  value in any order).

The option is a compile-time constant of the kernel. Without it the
kernel is that of the second checkpoint: all 216 hours of the f32 week,
48 hours in f64 and 24 hours of the tangent are identical to the bit
(job 4495444). With it (job 4495446):
- f64 48 h takes 228.6 s per model day instead of 245.3 (−6.8 %). The
  source kernel alone takes 1.244–1.251 s per step instead of
  1.346–1.405 s (job 4472571).
- The f32 week takes 99.3 s instead of 98.3: no gain. The source kernel
  alone takes 0.631–0.649 s instead of 0.622–0.640 s.
- The order of the additions changes, so the results differ at
  round-off. In f64, $H_s$ differs by at most 7.6e-12 m over 48 h. In f32
  over the week, the hourly RMS median is 3.4e-6 m, as between two f32
  builds. The costs of the calibration windows differ by at most 4e-3
  (69,351.110 against 69,351.114), and every bias and RMSE is the same to
  the printed digits (job 4498053).
- On the harness nodes, the differences against WW3 and the CPU
  transcription are the same to the printed digits.

The tangent kernel keeps the order of WW3, so the option has no effect on
a tangent run.

**Not done: the vector tangent.** Jean-Marc dropped it on 26 November
("Oui, passer à M4"). One tangent run of the first day takes 508.5 s per
model day, of which about 245 s is the value path (the f64 value run) and
about 263 s the tangent. With the value path shared, the best cases are:
- two directions: about 770 s instead of 1,017 s for two runs (1.3 times);
- nine directions: about 2,610 s instead of 4,580 s (1.75 times).

Those figures assume the nodes per SM stay as they are, and the shared
memory does not allow that:
- Each further direction needs its tangent spectrum, the tangent of the
  extended spectrum of the DIA and the tangent slots: about 11–15 KB per
  node in f64.
- With two directions a node no longer fits in a third of an SM (54,954
  B). The kernel goes back to 2 nodes per SM, and step 6's gain is lost.

The adjoint of M4 gives the gradient of the cost with respect to all the
parameters in one backward run, so it replaces the vector tangent for the
nine directions of `BETAMAX`$(U)$, `SDSC2` and `NLPROP`.

**Not done: asynchronous outputs.** The profile showed that overlapping
the outputs with the steps can gain less than 1 %.

**Speed at the end of the optimisation.** One A100 of hpc against WW3
`dea641c` on grid v5 $K=4$ with the full ST4, in seconds per model day.
The GPU runs start from the restart of 1 January 00:10 (8 January 00 h
for the last two rows) with the parameters of p0. The 64-core WW3 run is
the cold start of 1 January with the parameters of calibration step 1:
a slightly different model, at the same cost.

| Run | Hours | 35ecb15 | Final build | WW3 | Final build against WW3 |
|---|---|---|---|---|---|
| f32 | 1–10 Jan (216 h) | 124.1 | 98.3 | 717.1 on 64 cores of hpc (6,453.9 s for 1–10 Jan, job 4269275) | 7.3 times faster |
| f64 | 1–3 Jan | 505.1 | 245.3 (228.6 with `--par-red`) | – | – |
| f64 `BETAMAX` tangent | 1 Jan | 931.4 | 508.5 | – | – |
| f32 | 8–10 Jan | 127.7 | 101.5 | 1,491 on 32 ranks of hpc (2,982 s, job 4139131, same hours) | 14.7 times faster |
| f64 | 8–10 Jan | 433.3 | 214.8 | the same | 6.9 times faster |

The last two rows are jobs 4495445 (GPU) and 4139131 (WW3). The final
build is identical to the bit to the second checkpoint. Against 35ecb15,
the gains are:
- f32: −21 % over the week and −21 % over 8–10 January;
- f64: −51 % over 1–3 January and −50 % over 8–10 January;
- f64 tangent: −45 %.

On Datarmor, 112 cores ran 2–7 January in 2,601 s per model day with the
explicit scheme and 1,058 s with the implicit one. The f32 week of the
final build is 26 and 10.8 times faster than these. The days differ, and
the first days are the most expensive on the GPU.

The GPU time from 23 November 07:27 is 47.0 GPU-hours at this checkpoint.

## Milestone M4: adjoint prototypes (26 November)

Before the full adjoint is written, two pieces of the model are written
three ways:
- the DIA (`W3SNL1`), the nonlinear source term with the gathers of the
  interaction tables;
- one geographic sub-step (`geo_tri_x4` then `geo_node_x4`), with its
  clip at 0, mask and open boundary.

The three ways are:
- a hand-written adjoint in CubeCL (`hww3_wave::adjoint`);
- the same adjoint in CUDA C, compiled with NVRTC and launched through
  `cudarc` with `__launch_bounds__` (`hww3_wave::cuda_m4`, feature
  `cuda-c`);
- the piece written as Burn tensor operations, its gradient taken by Burn's
  tape (`hww3_wave::burn_m4`).

Each is checked with true gradients only: the dot-product test
$\langle L x, y\rangle = \langle x, L^T y\rangle$ against the tangent
kernels of M3, and the gradients of the three ways against each other.

`hww3-src adj-dia` and `hww3-prop adj-geo` run them. The job script
`scripts/m4_proto_hpc.slurm` samples the GPU memory with `nvidia-smi`
during each run (job 4531322, A100, f64).

**How the hand-written adjoints work.** Neither needs atomics nor an
intermediate of the forward pass beyond its input:
- The DIA adjoint rebuilds the extended spectrum `UE` from the spectrum.
  Each bin of the interaction grid gathers the cotangents of its three
  passes from those of $S$ and $D$ through the transposed index tables of
  `DIA_IC`. Each bin of `UE` gathers its cotangent through the transposed
  tables of `DIA_IP`. The transposed tables are built once on the host
  (39,168 entries).
- The adjoint of a sub-step runs in three passes after the triangle
  values are recomputed:
  - a node pass: the cotangent of the upwind sum and of the node's own
    value, through the clip and the mask;
  - a triangle pass: the cotangent of $\tilde U_T$ from its three nodes;
  - a node pass gathering through the slot lists of the forward kernel.

**DIA** (`W3SNL1` alone, harness nodes of grid v5 $K=4$ repeated 100 times,
186,900 nodes, one cube of 128 units per node):

| | Time per launch | Against the value kernel | Shared memory per node |
|---|---|---|---|
| value (`srce::snl1`) | 0.0420 s | 1 | 73,024 B |
| tangent (`srce_tl::snl1_tl`) | 0.1196 s | 2.85 | 47,120 B |
| adjoint, CubeCL | 0.0974 s | 2.32 | 52,096 B |
| adjoint, CUDA C (102 registers) | 0.0598 s | 1.42 | 52,096 B |
| Burn, forward and backward | 17.2 µs per node (7,476 nodes) | 77 | about 0.6 MB of GPU memory per node |

- The dot-product test agrees to 1.2e-12 of $\langle L x, y\rangle$
  (1.8e-17 of $|Lx|\,|y|$).
- The CUDA C adjoint equals the CubeCL one to 2.8e-16. The result is the
  same with 1 or 2 nodes per SM asked in the launch bounds.
- Burn's gradient equals the hand-written adjoint to 4.0e-16, and its own
  dot-product test agrees to 6.5e-15. Its forward pass alone takes 38
  times the value kernel. Its tape held about 5 GB for 7,476 nodes: the
  whole grid at once would need about 215 GB.

**One geographic sub-step** (frequency 12, `sub = 0`, the whole grid:
424,123 nodes, 807,832 triangles, 36 directions):

| | Time per sub-step | Against the value kernels |
|---|---|---|
| value (triangle and node kernels) | 1.073 ms | 1 |
| tangent | 1.976 ms | 1.84 |
| adjoint, CubeCL (triangle values recomputed, three passes) | 3.865 ms | 3.60 |
| adjoint, CUDA C (56, 68, 72, 46 registers) | 3.134 ms | 2.92 |
| Burn, forward alone | 87 ms | 81 |
| Burn, forward and backward | 4,276 ms | about 4,000 |

- The dot-product test agrees to 4.0e-15. The CUDA C adjoint equals the
  CubeCL one to the bit.
- Burn's gradient equals them to 3.3e-16 (dot-product test 3.8e-15).
- In Burn, the gathers of the nodes of each triangle and of each
  (node, triangle) pair become scatter-adds of repeated indices in the
  backward pass, with atomics in f64. That is why the backward takes 50
  times the forward.
- Burn's intermediates of one sub-step take about 6.1 GB (14.5 KB per
  node), and a step has 287 sub-steps.
- The hand-written adjoint needs 0.6 GB of scratch per sub-step, reused
  from one sub-step to the next.

**Step-level checkpointing over 156 h on one A100** (f64, 936 steps of
600 s):
- A state is the spectra, $424{,}123 \times 1152 \times 8$ B = 3.9 GB. A
  forward step takes 1.70 s (245.3 s per model day): about 0.31 s of
  sweeps (287 × 1.07 ms) and 1.39 s of the rest, mostly the source
  kernel.
- **Cost of the adjoint of a step.** The sweeps are recomputed once and
  reversed at 2.9–3.6 times their cost: about 1.2 s. The source kernel
  is recomputed once (to rebuild its dynamic sub-steps) and reversed at
  1.4–2.3 times its cost, the DIA's ratios taken for the whole `W3SRCE`:
  3.3–4.6 s. The adjoint of a step is then 4.5–5.8 s, 2.7–3.4 forward
  steps.
- **Two levels in host RAM.** A checkpoint every 31 steps (31 states)
  is written during the forward run. In the backward pass, each segment of
  31 steps is recomputed from its checkpoint, keeping its 31 step starts.
  That is 62 states, 242 GB of host RAM: an eighth of an A100 node of
  hpc, which has 2 TB and 8 GPUs. Moving them over PCIe takes about 550 s
  in all, which can overlap the computation.
- **Total.** One forward run, one recomputation and the adjoint: 4.7–5.4
  forward runs of 1,590 s, so 2.1–2.4 h for the gradient of a 156 h cost
  with respect to all the parameters and forcing fields at once.
- **Within a step.** The GPU keeps the three states after `W3KTP3`, the
  propagation and the second `W3KTP3` (12 GB), the inputs of the
  sub-steps of the frequency being reversed (122 MB each, up to `ITER`),
  and the spectra of the dynamic sub-steps of the source kernel, by
  batches of nodes (9.2 KB per node and dynamic sub-step).
- **With the GPU alone.** 12 states (47 GB), recomputed by binomial
  checkpointing (4 repetitions for 936 steps), take about 3.2–3.5 h.
- **Against tangents.** The nine tangents of `BETAMAX`$(U)$, `SDSC2` and
  `NLPROP` over the same window take about 5.7 h, at the 156 h rate of
  the calibration's p0 tangent (642 s per model day) scaled by the
  optimisation's gain on the first day (931.4 to 508.5 s). The adjoint is
  2.4–2.7 times cheaper for the nine parameters, and it also gives the
  gradient with respect to the forcing fields, which tangents cannot.

**Register limits.** CubeCL declares only the size of the cube, so its
kernels cannot be capped. Adding `.maxnreg` to the PTX it writes and
reading `ptxas -v` gives:

| Kernel | Cap | Spill | Effect |
|---|---|---|---|
| f32 value kernel with the hoisted `W3SIN4` constants (step 3, now 168 registers) | 128 | 112 B, as today's kernel | 4 nodes per SM, which would make step 3 usable |
| f64 value kernel | 168 | 160 B | 3 nodes per SM, if its shared memory also fell under 54,954 B (the variant with per-unit sums had 50,272 B) |
| tangent kernel | 128 | 12 KB | not useful |

A cap needs the kernel in CUDA C, which `__launch_bounds__(128, n)`
gives, or a patched CubeCL.

The GPU time from 23 November 07:27 is 51.5 GPU-hours.

## Milestone M4: the adjoint in CUDA C (27 November)

Jean-Marc chose CUDA C for the adjoint. The kernels are compiled at run
time with NVRTC and launched through `cudarc` (feature `cuda-c`). The
forward and tangent kernels stay in CubeCL, and the CI builds both.

### Source terms (checkpoint a)

The adjoint of `W3SRCE` covers the configurations of the M3 tangent:
- `W3SIN4` with `SINTABLE = 1`, with or without sheltering, with the tables
  at one `BETAMAX` or in `BETAMAX` for `BETAMAX`$(U)$;
- `W3SDS4` with `SDSBCHOICE = 1` and its cumulative term, or 3 (Romero);
- the DIA, `W3SLN1`, `W3SDB1`, `W3SBT4`, and the dynamic sub-steps with
  the FHMAX limit and the tail.

It works in three passes (`hww3_wave::cuda_adj`, sources
`src/cuda/w3srce*.cu*`):
1. **Value kernel.** A CUDA C port of the CubeCL kernel writes, for each
   node, the spectrum and 40 scalars at the start of each dynamic
   sub-step: the tape, 9.2 KB per node and sub-step.
2. **Adjoint kernel.** It takes the sub-steps in reverse. For each one it
   recomputes the forward pass from its record: the same sources, step
   and update as the value kernel, so the same discrete choices. Then it
   runs the transposes of the tail, the FHMAX limit, FHIGH, `W3SPR4`, the
   update and each source term.
3. **Start of `W3SRCE`.** Last come `W3SPR4`, `W3SIN4` without sources,
   `W3SPR4`, and `BETAMAX`$(U)$.

The DIA gathers through the transposed index tables (`dia_adj`), with no
atomics. Nodes go in batches whose tape fits a given size.

**Checks** (`hww3-src adj-src`, jobs 4555442 and 4555775 on one A100, f64):

| Check | Harness | Result |
|---|---|---|
| Value port against the CubeCL kernel | HWW3 settings, 1,869 nodes | 4.2e-15 of each node's largest density |
| | IRI settings with Romero, 1,864 nodes | 5.4e-15 |
| | `BETAMAX`$(U)$ | 2.3e-15 |
| | all three | u* within 3.3e-16, mean dynamic step within 1.8e-14 |
| Dot products against the tangent kernel `srce_tl` | all three | sums agree to 3.0e-14 or better, every node to 2.8e-11, none above 1e-8 |

The dot products were run:
- for each freeze mask: all the terms, each term alone, no term, u* and
  z0 frozen, the denominators frozen;
- for each class of directions: the spectrum, the wind speed and
  direction, `BETAMAX`, `SDSC2`, `NLPROP`, `BJGAM`, and the six
  coefficients of `BETAMAX`$(U)$.

**Cost** on the harness repeated 10 times (18,690 nodes):
- The value kernel takes 0.068 s.
- The two value passes of the adjoint (the count of the sub-steps, then the
  tape) take 0.135 s.
- The adjoint kernel takes 0.616 s, 9.0 times the value kernel, with 1.81
  sub-steps per node and a tape of 322 MB.
- The adjoint kernel uses 255 registers and 114,680 B of shared memory per
  node, so one node runs per SM. The value kernel runs two.
- This is well above the 1.4 of the DIA prototype. The causes are the
  lower occupancy, the serial parts on thread 0, and `W3SIN4` computed
  twice per sub-step.

### Propagation (checkpoint b)

The adjoint of the propagation (`hww3_wave::cuda_prop`, source
`src/cuda/w3prop.cu`) runs on the raw pointers of the CubeCL model's
buffers: CubeCL and `cudarc` share the primary context of the device, so
nothing is copied. It covers:
- the refraction and the frequency shift of `W3KTP3`: each row or column
  recomputes its ULTIMATE QUICKEST face values, then applies their
  transpose with the limiter branches of the values;
- the geographic N-scheme with its sub-cycling: the sub-steps of the
  frequency are recomputed and kept on the device, then reversed by a node,
  a triangle and a node pass;
- the conversions between `VA` and `AC`, with their clamps at zero.

The kernels of the level remap of `W3ULEV` (device and host paths) and of
the outputs $H_s$ and $T_m02$ are written as well.

**Checks** (`hww3-prop adj-prop`, job 4555442): dot products against the
tangent kernels of `prop_tl` on grid v5 K=4 (424,123 nodes), after the
forcing of the first step (1 January 00:20; two `W3KTP3` calls, one
before the geographic part):

| Part | Relative difference of the sums |
|---|---|
| refraction | 5.4e-13 |
| frequency shift | 4.7e-13 |
| geographic, frequency 17 | 2.7e-16 |
| geographic, frequency 32 | 6.8e-14 |
| whole propagation of the step | 3.3e-13 |

Frequency 1 agrees to the bit. The random direction is relative to the
spectra, which are close to zero at the lowest frequency, so that test
says nothing.

### Whole steps (first part of checkpoint c)

`hww3-prop adj-step` (job 4555775) runs six steps of grid v5 K=4 from the
restart of 1 January 00:10. It compares the tangent of each direction with
the adjoint of the six steps. Each step is recomputed from its checkpoint,
which holds the spectra, the wavenumbers and group speeds, the host state
and the times of the forcing records (the files are reopened). The adjoint
then transposes the source terms, the propagation and the parts of the
forcing that change the spectra:
- the open boundary;
- the ice transitions;
- the drying and wetting nodes;
- the level remap.

| Direction | Relative difference of the sums |
|---|---|
| spectrum at the start | 1.5e-12 |
| `BETAMAX` | 7.8e-13 |
| `SDSC2` | 3.3e-14 |
| `NLPROP` | 7.7e-13 |
| `BJGAM` | 2.0e-14 |

Times and memory:
- The tangent takes 74 s per direction, 12.3 s per step.
- The value run that keeps the checkpoints takes 95 s.
- The adjoint takes 501 s: 429 s for the source terms, 14 s for the
  propagation and the forcing, 58 s for the recomputed steps.
- A checkpoint holds 4.17 GB, and the GPU peak is 55.7 GB.

At this state, the start of the run, the source terms take many dynamic
sub-steps. The tangent step costs 3.5 times its mean over the first day
(508.5 s per model day).

The adjoint of the source terms, at 71 s per step, is what to optimise
before the 156 h gradient. At this rate 936 steps would take about 22 h,
or about 6 h at the mean rate of the first day, against the 2.1–2.4 h of
the plan.

The GPU time from 23 November 07:27 is 53.3 GPU-hours, of which 0.64
for the adjoint jobs of this section (4553862, 4555442, 4555775).

## Coastal diagnostics of the buoy scores

At the start of the calibration the buoy $H_s$ is too low (bias −4.85 cm on
the training window), and the first step, which brought the altimeter bias
from +2.86 to +1.04 cm, made it −6.38 cm. `scripts/coast_diag.py`
looks at where this bias comes from, for the start run (`cal/it0_value`,
"p0") and the run after step 1 (`cal/it1_value`), over the buoy hours of
the calibration's operators (255 buoys and 12,247 hours on the training
window, 257 buoys and 14,989 hours on the validation window). Each buoy is
matched to the nearest wet node within 10 km. For each buoy it uses the
water depth of its file (attribute `bottom_depth`, present for 174
buoys) and, for its node, the depth, the distance to the coast (the
nearest mesh boundary node not on the open boundary), the HEALPix level,
the depth range over the node and its neighbours, and inside the node's
cell (radius half its mean edge length) the spread of the GEBCO depth and
the fraction of land, computed on Datarmor where GEBCO is
(`scripts/coast_subgrid.py`, which writes nothing there). It needs no GPU.

**The matching is not the cause.** The node depth is the buoy's to a
median ratio of 1.00 (10–90 %: 0.71–1.82), at a median distance of
1.8 km (at most 9.8 km). The bias does not grow with the depth mismatch:

| node depth / buoy depth | buoys | training: p0 → step 1 (cm) |
|---|---|---|
| 0.5–0.8 | 26 | −12.3 → −14.4 |
| 0.8–1.25 | 105 | −5.9 → −7.3 |
| 1.25–2 | 24 | −8.3 → −9.7 |
| above 2 | 16 | −5.5 → −5.7 |
| no depth in the file | 81 | 0.0 → −1.8 |

A depth-aware matching, tried as a diagnostic only (the nearest wet node
within 25 km whose depth is within 25 % or 50 % of the buoy's), moves 58
or 29 buoys and makes the scores worse: on the training window the p0
bias goes from −4.85 to −5.97 or −7.00 cm and the RMSE from 31.3 to
33.0 or 33.4 cm, and likewise on the validation window and at step 1.
The operator of the calibration is unchanged.

**The bias sits at the coast, in cells with land.** H_s bias in cm, p0 →
step 1:

| class | buoys | training | validation |
|---|---|---|---|
| node 0–5 km from the coast | 113 | −8.7 → −9.8 | −10.6 → −12.5 |
| 5–10 km | 30 | −2.0 → −3.9 | +1.5 → −0.6 |
| 10–25 km | 49 | −0.6 → −2.1 | +6.4 → +3.7 |
| 25–50 km | 39 | −2.2 → −4.6 | +3.6 → +0.2 |
| over 50 km | 24 | −3.3 → −4.9 | −5.1 → −7.4 |
| no land in the cell (GEBCO) | 200 | −5.6 → −7.3 | −4.0 → −6.6 |
| land 0.1–10 % of the cell | 23 | +6.6 → +5.6 | +13.3 → +11.5 |
| land 10–30 % | 18 | +5.1 → +4.4 | +8.0 → +6.9 |
| land 30 % or more | 14 | −27.2 → −28.3 | −35.4 → −36.7 |
| GEBCO spread / node depth below 0.1 | 82 | −3.2 → −4.8 | −0.6 → −3.1 |
| 0.1–0.3 | 89 | −4.6 → −6.2 | −2.2 → −4.6 |
| 0.3–1 | 81 | −6.7 → −8.2 | −7.1 → −9.2 |
| HEALPix level 10 | 130 | −4.2 → −6.0 | +0.5 → −2.4 |
| level 11 | 121 | −5.8 → −7.1 | −7.5 → −9.2 |

The 14 buoys whose cell is at least 30 % land (RMSE 58–70 cm) give
−1.4 cm of the −4.85 cm of the training window and −2.0 cm of the
−3.30 cm of the validation window: sites inside bays or behind islands
that the mesh does not resolve. The bias also grows with the spread of the
bathymetry inside the cell, while it does not depend clearly on the node
depth (−4.8 cm at 0–20 m, −2.1 cm at 50–100 m, −10.0 cm at 100–200 m
on the training window). The first calibration step lowers $H_s$ by 1–2.5 cm
in every class: it is the same reduction everywhere that brought the
altimeters closer, and the buoy term of the cost, dominated by these
coastal sites, pulls the other way. Coastal parameters as functions of
these grid descriptors (plan, section 9.7 of the
[differentiable WW3 note](differentiable.md)) are the way to treat them;
leaving these buoys out of the cost, or weighting them down, would be a
decision for the calibration, not taken here.

### Depth-aware buoy operator

Jean-Marc asked for a buoy operator that interpolates the grid taking the
depth into account, instead of the nearest wet node (`scripts/buoy_op.py`,
read by `cal_obs.py` when an operator file holds its weights, and by
`validate_buoys.py --match depth`). It stays linear in the model fields,
with weights fixed by the geometry and the depths, so the tangents go
through it unchanged. It is built and evaluated below; the cost of the
calibration keeps the nearest node until he has seen these results.

- **Depth at the buoy.** The GEBCO cell that holds the buoy (the
  bathymetry from which hww3-grid takes the node depths), else the median
  of the sea cells of the 3 x 3 around it, else the `bottom_depth` of the
  buoy file: GEBCO for 250 buoys, its 3 x 3 median for 3, none for 2 (they
  keep the mesh weights). Where both exist (174 buoys) GEBCO and
  `bottom_depth` agree to a median ratio of 1.00 (10–90 %: 0.77–1.26),
  with 6 buoys beyond a factor 2. GEBCO is read on Datarmor by
  `scripts/buoy_depth.py`, which writes nothing there; only the depths
  come back.
- **Weights.** Barycentric in the mesh triangle that holds the buoy, over
  its wet nodes, or inverse squared distance over the 8 nearest wet nodes
  within 10 km outside the wet mesh; each times
  $\exp(-(\ln(d_{node}/d_{buoy}))^2/s^2)$ with $s = \ln 2$ and both depths
  capped at 100 m (deeper water is deep for these waves), then
  renormalised. If no node of the triangle is within a factor 3 of the
  buoy's depth, the 8 nearest wet nodes within 10 km are tried; if none is,
  the buoy is left out of the cost and the scores. 3 buoys are left out
  (GL_TS_MO_41053 at 1 m, NO_TS_MO_6202603 at 4 m,
  NO_TS_MO_BlankenbergeBuoy at 1 m, in 1–4 m of water where the mesh is
  deeper), and 252 of the 255 buoys of the training window change node or
  weights; a buoy uses up to 8 nodes.
- **$T_{m02}$.** Interpolated through the moments: $2\pi\sqrt{M_0/M_2}$
  with $M_0$, $M_2$ the weighted sums of $m_0 = (H_s/4)^2$ and
  $m_2 = m_0 (2\pi/T_{m02})^2$ at the nodes, which are linear in the
  spectrum. A linear average of $T_{m02}$ differs from it by 0.002 s at
  the median buoy hour but by up to 4.5 s at 1 % of them, and has an RMSE
  0.06 s larger.
- **Check.** With the nearest node and a flat kernel the operator gives
  the cost of the old one to all digits printed (47,249.686709 at $p_0$,
  46,628.543598 at step 1), with $T_{m02}$ linear or through the moments,
  and the same derivatives from the tangent runs of step 1 (to the 7
  digits printed).

| | $p_0$, nearest | $p_0$, depth-aware | step 1, nearest | step 1, depth-aware |
|---|---|---|---|---|
| training buoy $H_s$ bias / RMSE (cm) | −4.85 / 31.30 | −4.77 / 31.25 | −6.38 / 31.64 | −6.32 / 31.61 |
| training $T_{m02}$ bias / RMSE (s) | −0.399 / 1.131 | −0.467 / 0.999 | −0.352 / 1.100 | −0.416 / 0.969 |
| validation buoy $H_s$ bias / RMSE (cm) | −3.30 / 41.43 | −3.03 / 40.79 | −5.60 / 41.68 | −5.35 / 41.12 |
| validation $T_{m02}$ bias / RMSE (s) | −0.393 / 1.148 | −0.411 / 1.024 | −0.344 / 1.123 | −0.357 / 0.999 |

Against the two other ways of matching the buoys, buoy $H_s$ bias / RMSE
in cm (the depth-aware matching is the diagnostic of the coastal section:
the nearest wet node within 25 km whose depth is within 25 % of the
buoy's):

| | nearest node | depth-aware matching | depth-aware interpolation |
|---|---|---|---|
| training, $p_0$ | −4.85 / 31.30 | −5.97 / 33.00 | −4.77 / 31.25 |
| training, step 1 | −6.38 / 31.64 | −7.49 / 33.29 | −6.32 / 31.61 |
| validation, $p_0$ | −3.30 / 41.43 | −4.92 / 45.17 | −3.03 / 40.79 |
| validation, step 1 | −5.60 / 41.68 | −7.20 / 45.35 | −5.35 / 41.12 |

Moving a buoy to a single node of the right depth, farther away, makes the
scores worse; interpolating over the nodes around it with depth weights
makes them slightly better, and much better for $T_{m02}$.

The new operator leaves the buoy $H_s$ bias nearly unchanged overall and
lowers its RMSE a little, and the $T_{m02}$ RMSE by 0.12–0.13 s. Class by
class (start run, training / validation, cm), it halves the bias of the
buoys whose cell is at least 30 % land (−27.2 → −16.4 and −35.4 → −17.8)
and of the finest cells (level 11: −5.8 → −4.5 and −7.5 → −5.5) and of
the nodes shallower than 20 m on validation (−8.6 → −6.0), and changes the
other classes by less than 2 cm; the buoys within 5 km of the coast stay
at −8.4 and −9.7 cm. On the WW3 run at step 1 over 5–10 January,
`validate_buoys.py --match depth` gives a buoy $H_s$ bias of −6.02 cm and
RMSE of 36.5 cm (nearest node: −6.17 and 36.8 cm) and a $T_{m02}$ RMSE of
1.00 s (1.12 s) with a correlation of 0.79 (0.72), over the 254 buoys
that have weights (the weights are built for the buoys of the
calibration's windows; the others are skipped).

## Third cost and the spin-up rule

Jean-Marc's decisions after the observation diagnostics: leave out the
buoys the mesh cannot represent, by a fixed list; use a Huber loss; and
never use the first part of a run.

**Excluded buoys** (`configs/buoy_exclude.csv`). The criterion is the median
over both windows of observed / model $H_s$ at the start run (through the
depth-aware operator, `scripts/buoy_exclude.py`) below 0.45, and a site
confirmed by hand on the map and in the metadata as a harbour, a barrier,
the inner Wadden or an archipelago. A buoy is not left out because the
model is wrong there.

| buoy | site | observed / model |
|---|---|---|
| Roompotsluis | sluice of the Oosterschelde storm-surge barrier | 0.06 |
| Maeslantkering Zeezijde Zuid | storm-surge barrier, Nieuwe Waterweg | 0.10 |
| Maeslantkering Zeezijde Noord | storm-surge barrier, Nieuwe Waterweg | 0.14 |
| Uithuizerwad2 | tidal flats of the inner Wadden | 0.10 |
| Schiermonnikoog Waggen | Wadden Sea behind Schiermonnikoog | 0.10 |
| Zeebrugge Zandopvangkade buoy | inside the port of Zeebrugge | 0.31 |
| Utö | coastal structure, outer Archipelago Sea | 0.43 |
| St Mary's Sound (6201053) | inside the Isles of Scilly | 0.41 |

Checked and kept: IL2-Finnis (0.53) and Penzance (0.51), partly sheltered
by islands or a bay but not inside them; the Sevenstones lightship (0.75);
the exposed buoys of Biscay and Bonifacio where the model misses the storms
(Bilbao station 1.54, Saint-Jean-de-Luz 1.40, Bonifacio 1.54); and
Wielingen Noord, Morecambe Bay, Weymouth and Folkestone, where the model is
2.7–3.4 times too low (open coastal waters, a model error).

**Third cost.** The depth-aware operator without these 8 buoys,
representativeness errors per class fitted again from the start run
without them (a land class with fewer than 5 buoys in one distance class
is not split by distance; $\sigma_{repr}$ 0.425 m for $H_s$ in cells with
30 % land or more and 1.10 s for $T_{m02}$ in cells with 0.1–10 % land, 0
elsewhere), and the Huber loss $\rho(z) = z^2$ for $|z| \le 1.5$ and
$3|z| - 2.25$ beyond, $z$ the residual over its observation error, for the
altimeters and the buoys; the Gauss–Newton steps use the iteratively
reweighted form (`cal_gn.py`). Cost (data part, prior apart) and the
unweighted buoy scores without the excluded buoys (`scripts/cal_cost3_hpc.slurm`,
job 4327059):

| point | third cost, training | validation | second cost, training | validation | buoy $H_s$ without / with the 8 (training, cm) |
|---|---|---|---|---|---|
| $p_0$ | 35,543.4 | 50,242.2 | 41,085.9 | 59,449.2 | −7.48 / 27.68 against −4.77 / 31.25 |
| step 3 | 34,432.2 (prior 4.19) | 48,173.9 | 39,805.2 | 57,475.3 | −9.60 / 28.52 against −6.88 / 31.81 |

Without the 8 buoys the buoy $H_s$ RMSE at $p_0$ falls from 31.25 to
27.68 cm (training) and from 40.79 to 38.44 cm (validation), and the bias
becomes more negative (−7.48 and −5.45 cm against −4.77 and −3.03 cm):
these sites were where the model was much too high.

**Spin-up rule.** A window must start after at least 30 % of its run,
counted from the cold start of the chain of restarts (1 January 00 h):
`cal_obs.py` refuses an operator built, or a run evaluated, on a window
that starts earlier (`check_spinup`). The windows of the calibration
satisfy it: training starts 108 h into the 156 h trial runs (69 %) and
the 216 h value runs (50 %), validation 157 h into the 216 h runs (73 %).

## Observation diagnostics

Why the buoys cannot be fitted with the three parameters, and whether the
loss should be robust: `scripts/obs_diag.py` (`scripts/obs_diag_hpc.slurm`,
Slurm job 4326487, CPU only) looks at every observation of the calibration
from the stored runs, at the start ($p_0$) and at step 3, on both windows,
the buoys through the depth-aware operator and the nearest node. The
figures, the per-buoy tables (`buoys_<window>.csv`) and `summary.json` are
in `$S/diag` on the hpc cluster.

**Altimeters.** The scatter is model error, not instrument noise: the
spread of the 7 samples of a block is 3.6–3.9 cm at the median, so the
noise of a block mean is 3.0–3.3 cm, against a residual of 25 cm (the
operators rebuilt with this spread, `op_<window>_sd.npz`, have the same
blocks). The RMSE is almost all standard deviation, which three global
parameters barely change:

| window | bias $p_0$ → step 3 | std | RMSE | 1.4826 MAD | excess kurtosis |
|---|---|---|---|---|---|
| training | +2.86 → −1.38 cm | 25.30 → 24.69 cm | 25.46 → 24.73 cm | 20.1 → 19.4 cm | 3.2 → 3.5 |
| validation | +5.76 → +1.36 cm | 25.83 → 25.63 cm | 26.46 → 25.66 cm | 21.0 → 20.8 cm | 3.7 → 4.3 |

Removing the bias alone would bring the validation RMSE at $p_0$ to
25.83 cm. The tails are moderately heavy: the largest 1 % of the residuals
carry 16–18 % of the mean square and the largest 5 % 40–44 % (8.5 % and
28 % for a Gaussian). The residual depends on the sea state: +5 to +9 cm
at observed $H_s$ of 1.5–3 m, where most blocks are, and −12 to −28 cm
above 4 m on validation (−28 to −77 cm above 6 m, a few hundred blocks);
step 3 lowers every class by 2–20 cm. It is +9 cm beyond 500 km from the
coast and −4 to +3 cm within 200 km, its standard deviation is larger
within 100 km of the ice (30–38 cm), and the mission biases spread over
10 cm (CFOSAT −4.1 cm, HY-2C +6.6 cm on training).

**Buoys.** The residual has heavy tails (1.4826 MAD 20–24 cm against a
standard deviation of 31–41 cm, excess kurtosis 3.9–4.2), and a few
buoys dominate it: the 5, 10 and 20 buoys with the largest squared error
carry 26 %, 37 % and 50 % of it on the training window (18 %, 30 % and
47 % on validation). Without the 10 worst the RMSE at $p_0$ falls from
31.25 to 25.30 cm (40.79 to 34.74 cm on validation), but the bias does
not improve (−6.06 and −4.19 cm) and step 3 still makes it worse (−8.21
and −7.46 cm): the outliers do not cause the trend of the bias. The worst
buoys are mostly sites the grid cannot represent: the storm-surge barrier
of the Maeslantkering (two stations, observed $H_s$ 0.1 m, model about
1.1 m), the Roompot sluice, the Wadden station of Schiermonnikoog, the
Bilbao station (40 % land in its cell, −1.0 m), the coastal structure of
Utö, the Barcelona coastal buoy, and a few offshore buoys with biases of
±1 m on one window (6201053, IL2-Finnis, 6200079, 6100023) to check for
data or sheltering (time series in `buoy_series.png`). The platform type
of the files (73 moorings, 168 moored surface buoys, 6 offshore
structures, 4 vessels at fixed position, 1 coastal structure) does not
separate them clearly.

The bias of the buoys depends on the sea state, as that of the
altimeters: at $p_0$ it is +9 cm for observed $H_s$ below 0.5 m, −1 to
−3 cm at 0.5–1.5 m, −15 cm at 2–3 m, −23 cm at 3–4 m, −32 cm at 4–6 m and
−55 cm above 6 m (both windows). Step 3 lowers $H_s$ in every class, so it
makes the high sea states worse (−21, −32, −45 and −75 cm), and by wind
speed at the node it leaves the bias at −8 to −2 cm below 12 m/s but takes
it from −4 to −14 cm at 12–15 m/s and from −8 to −22 cm at 15–20 m/s. The
mean $H_s$ of the buoys is lower than that of the altimeters, and more of
their hours are in the low and the high classes, so a constant change of
the growth that lowers the moderate seas, which the altimeters ask for,
moves the buoys the wrong way. This is what a `BETAMAX` depending on the
wind, or a wind correction $g(U)$, should address (plan, section 9.7 of
the [differentiable WW3 note](differentiable.md)).

**Loss.** With the residuals of the second cost and the Jacobian $A_1$,
one Gauss–Newton step of a Huber loss (iteratively reweighted, $\delta$ in
units of the observation error) or of an L1 loss, against the L2 step
(factors from $p_0$ for `BETAMAX`, `SDSC2`, `NLPROP`):

| point | L2 | Huber $\delta = 1$ | Huber 1.5 | Huber 2 | L1 |
|---|---|---|---|---|---|
| from $p_0$ | 0.820, 0.746, 1.167 | 0.876, 0.873, 1.182 (cos 0.94) | 0.862, 0.840, 1.179 (0.97) | 0.849, 0.810, 1.174 (0.99) | 0.943, 0.938, 1.072 (0.96) |
| from step 3 | 0.721, 0.481, 1.067 | 0.741, 0.565, 1.174 (0.91) | 0.735, 0.537, 1.140 (0.98) | 0.731, 0.519, 1.114 (0.99) | 0.772, 0.635, 1.173 (0.44) |

The robust losses move the same way along the `BETAMAX`–`SDSC2` valley
(cosines 0.91–0.99 with the L2 step), only less far; L1 nearly stops at
step 3. A robust loss would limit the weight of the unrepresentative
sites, but it would not remove the conflict between the sea states, which
comes from the model and its forcing, not from outliers.
