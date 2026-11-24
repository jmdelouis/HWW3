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
