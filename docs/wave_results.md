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

| Grid | Nodes | Triangles | Interior / coast / open | $S$ (`explicit_cost.py`) | Sweeps per step | ITER at $f_1$ | Sweeps per step, every wet node | ITER at $f_1$, every wet node |
|---|---|---|---|---|---|---|---|---|
| Channel–Biscay box of v4 | 21,425 | 41,313 | 19,872 / 1,358 / 195 | 3,079 (3,080) | 92 | 6–8 | 208 | 8–17 |
| v4 | 794,613 | 1,531,542 | 734,829 / 58,443 / 1,341 | 3,678 (3,678) | 110 | 8–10 | 258 | 18–24 |
| v5 $K=4$ | 424,123 | 807,832 | 382,296 / 40,775 / 1,052 | 2,837 (2,835) | 83 | 4–5 | 287 | 19–26 |

A coast node propagates 18.9 of the 36 directions on average.

The M0 counts (columns 5 to 7), like `scripts/explicit_cost.py`, take the
interior nodes only. WW3 runs with `FSBCCFL = T` (the default of `UGBCCFL`)
and counts the coast and open-boundary nodes as well (see
[Coast nodes](#coast-nodes-the-sub-cycling-count)), which gives the last
two columns (`hww3-wave info`, without currents).

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
- `scripts/explicit_cost.py` and the sub-step cap `cfl_substeps` of
  `hww3-grid`, which shaped grid v5, count the interior nodes only, as M0
  did; the coast cells now set the explicit cost of WW3 on these grids.

## Milestone M2a: the source terms (in progress)

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
| `week` | 1–8 January | cold start, as the explicit run | hourly fields, for the final targets and the scores | 4122614 (64 ranks) | running: 41 s per model hour, about 2 h and 120 core-hours expected |

The `dev` run takes 64 s per model hour on 32 ranks, against 17 s for the
propagation-only reference of M1 on the same grid and cores: the source
terms cost about three times the propagation in WW3.

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

These are the round-off differences of the single-precision WW3. Still to
port: the wind forcing (`W3UWND`), then the GPU kernels and the time loop.
