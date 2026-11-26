# Differentiable WW3 on GPU in Rust: design and cost

*Design note, November 2026. Milestone M0 has been run (section 9.1):
the N-scheme sweep and its transpose on one A100. Nothing else described
here is implemented yet.*

The aim is the gradient of the misfit between the HWW3 WAVEWATCH III
configuration and the observations already used for validation: along-track
L3 altimeter $H_s$, and moored-buoy $H_s$ and $T_{m02}$. Gradients are
needed with respect to the forcing and to the model parameters, so that the
wave model can sit inside a learning chain: gradient-based calibration,
learned corrections of the forcing, and hybrid physics/ML source terms
trained end to end.

The route chosen is a re-implementation in Rust of the HWW3 configuration
of WW3, explicit scheme only, running on one GPU on the HEALPix triangle
mesh, with derivatives. This note estimates what it costs to develop. The
estimates use:

- the WW3 source used by HWW3 (NOAA-EMC `develop`, commit `dea641c`),
  measured routine by routine;
- the runs of [Validation](validation.md);
- a sub-domain computation made for this note.

## Summary

| Question | Answer |
|---|---|
| Scope | The HWW3 explicit configuration only: N scheme (`EXPFSN`), ST4, DIA (`NL1`), `LN1`, `DB1`, `BT4`, `MLIM`, ice as an obstruction (`IC0`), currents, sea level, open-boundary spectra, on the `hww3-grid` mesh, on one A100. |
| Fortran to port | About 7,400 active lines (19,200 raw lines) out of the 259,054 lines of `model/src`. The physics (ST4, DIA, LN1, DB1, BT4, source integration) is 2,157 active lines; the propagation is 1,432. |
| Decisions (21 November 2026) | Kernels in CubeCL, with Burn for tensors, tape and checkpointing. Only the CUDA crates are installed (no Rust nightly with Enzyme for now). ST4 is simplified first, then ported in full: the aim is a complete WW3. The first target is M2, with finite-difference gradients, before any adjoint work. |
| Milestone M0 (22 November 2026) | **Go.** The CubeCL kernels, driven through Burn, run the N-scheme sweep and its transpose on one A100 of the Ifremer hpc cluster. One frequency sweep on grid v4 takes 0.93 ms in single precision, under the 1.3 ms threshold. The dot-product tests pass, and Burn's `backward()` returns the transpose kernel's output exactly. CubeCL does not build on Datarmor (glibc 2.19), so development continues on the hpc cluster. |
| Smallest useful version | The forward model on GPU, validated against the Fortran explicit runs. Gradients with respect to a handful of parameters (`BETAMAX` first) then come from finite differences in double precision, with no AD code. |
| Cost to that point | **M2a**, with a simplified ST4 that WW3 also runs (`SINTABLE = 0`, `SDSCUM = 0`): 26–52 person-weeks for one developer who knows WW3, Rust and GPUs (central value about 35), or 10–28 weeks of Jean-Marc's time with coding assistants writing most of the code (central value about 16). **M2b**, with the full ST4: 30–59 person-weeks (about 40), or 11–31 weeks of his time (about 18). |
| Cost to reverse mode for the fields (milestone M4) | 52–98 person-weeks for one developer (central value about 70). For Jean-Marc with assistants: 17–48 weeks (central value about 30). |
| AD in Rust on GPU today | No tool differentiates Rust GPU kernels automatically. The kernels are written in CubeCL, with Burn for tensors and the tape. The adjoint of the propagation is written by hand: it is the transposed operator. The adjoints of the source terms are written by hand too, or the source terms are written as Burn tensor operations so that Burn differentiates them. `std::autodiff` (Enzyme) works on the CPU only; it is not installed for now, and the reference derivatives come from finite differences in double precision. |
| Expected speed | 40–550 s per model day on one A100 for grid v4, against 4,318 s for the Fortran explicit scheme on 112 cores: 8 to 100 times faster. Milestone M0 measured the geographic propagation alone at 1.5 times the bandwidth estimate per sweep: 34.6 s per model day on grid v4 and 21.3 s on grid v5 $K=4$ with the sub-cycling of WW3, which counts the coast nodes (14.8 and 6.2 s with the interior count used at M0). |
| Biggest cost drivers | How faithful the ST4 port must be; hand-written adjoints of the source terms; debugging against the Fortran. |

## 1. Scope

**What is ported.** The configuration of the explicit runs
(`scripts/natl_exp_setup.pbs`). The switches are those of
`ww3/switch_hww3`:

- `ST4` (wind input, whitecapping, swell dissipation);
- `NL1` (DIA);
- `LN1` (linear wind input);
- `DB1` (Battjes–Janssen depth-induced breaking);
- `BT4` (SHOWEX bottom friction);
- `MLIM` (Miche limiter);
- `IC0` (ice as an obstruction);
- `WNT2` and `CRT1` (time interpolation of wind and currents);
- `RWND` (wind taken relative to the current).

The namelists split the explicit step into spatial propagation, refraction
with frequency shift, and sources (`EXPFSN = T`, everything `IMP* = F`). The
spectrum is 36 directions by 32 frequencies from 0.0373 Hz, with a 600 s
global step. The mesh and its arrays come from `hww3-grid` (`raw/*`).

`BT4` is the bottom friction and `DB1` the depth-induced breaking; both are
ported.

**What is left out:**

- the MPI and PDLIB domain decomposition (one GPU holds the whole grid);
- the implicit scheme;
- other physics switches;
- structured grids;
- WW3 output formats other than hourly $H_s$ and $T_{m02}$ (plus spectra at
  chosen points).

**Reference timings.** The Fortran explicit scheme takes 4,318 s per model
day on grid v4 (794,613 nodes) and 2,601 s on grid v5 $K=4$ (424,123
nodes), both on 112 cores. The explicit runs scored as well as or better
than the implicit ones:

- altimeter bias −0.017 m instead of −0.031 m;
- buoy $H_s$ RMSE 0.301 m instead of 0.310 m (252 buoys, 4–7 January).

### 1.1 Controls

| input or parameter | in the HWW3 runs | size per week (v4) | differentiable? |
|---|---|---|---|
| 10 m wind | ERA5 hourly 0.25°, put on the nodes by `ww3_prnc`, linear in time, taken relative to the current; enters ST4 through $u_*$ | $2.7 \times 10^8$ | yes, piecewise smooth (stress table) |
| currents, sea level | Copernicus Marine hourly 1/12° plus FES2014: advection, Doppler shift, refraction, total depth | $2.7 \times 10^8$, $1.3 \times 10^8$ | yes; the explicit sub-cycling (an integer) stays fixed |
| sea ice | `IC0` with `CICE0 = CICEN = 0.5` | – | no: a binary mask, kept fixed |
| open-boundary spectra | 1,341 nodes every 3 h, rebuilt from six GLOBMULTI partitions by `scripts/boundary_spectra.py` | $3.8 \times 10^8$ values, or $1.9 \times 10^6$ partition parameters | yes; partition parameters are the natural controls |
| bathymetry | node depths | 794,613 | yes for a fixed mesh; low priority |
| ST4 | `BETAMAX` 1.75, `ZALP` 0.006, `ALPHA0` 0.0095, `TAUWSHELTER` 0.3, `SWELLF*`, `SDSC2` $-2.2\times10^{-5}$, `SDSBR` $9\times10^{-4}$, `SDSCUM` −0.4 | `W3SIN4`, `W3SDS4`, stress tables `TAUHFT`, `TAUHFT2` | yes, if the tables are rebuilt inside the differentiated code (they depend on `BETAMAX`, `ZALP`, `ALPHA0`); the laminar/turbulent swell switch is a step |
| DIA | `LAMBDA` 0.25, `NLPROP` $2.5\times10^7$, `KDCONV`, `KDMIN`, `SNLCS1–3` | `W3SNL1` | yes, except `LAMBDA`, which sets the interpolation indices |
| DB1, BT4 | `BJALFA` 1, `BJGAM` 0.73; `RIPFAC1–4`, $D_{50}$ 0.2 mm | `W3SDB1`, `W3SBT4` | piecewise (branches, Shields threshold) |

Values are those of `runs/natl_v4_jan2026_cur/ww3_grid.out`.

The one sensitivity measured so far is the change of `BETAMAX` from 1.43 to
1.75. It moved the altimeter bias from −0.225 to +0.005 m (331,154 blocks),
about 7 cm per 0.1.

**Non-smooth points.** These exist whatever the tool:

- the per-frequency sub-cycling count (fixed);
- the dynamic source step and the action limiter, $\Delta N_{\max} \propto
  \mathrm{FACP}/(\sigma k^3)$ (the backward pass freezes the step count of each
  node);
- $\max(0, N)$;
- the third-order limiter of the spectral advection (`W3KTP3`), which is
  replaced by first-order upwind on the differentiated path if it gets in
  the way;
- the branches of ST4, DB1 and BT4.

## 2. Development cost

### 2.1 Size of the Fortran to port

`scripts/ww3_loc.py` preprocesses each WW3 file with `cpp` and
`-DW3_<switch>` for every switch of `ww3/switch_hww3`. It then counts the
non-blank, non-comment lines of each routine; declarations and continuation
lines are included. Raw lines are those between `SUBROUTINE` and
`END SUBROUTINE` in the source. With the HWW3 switches the whole `model/src`
has 104,908 active lines out of 259,054 raw.

| component | WW3 routines | raw / active lines |
|---|---|---|
| mesh geometry, boundary flags | `PDLIB_INIT`, `PDLIB_MAPSTA_INIT`, `PDLIB_IOBP_INIT`, `SET_IOBDP_PDLIB`, `SET_IOBPA_PDLIB`, `SET_UG_IOBP_PDLIB_INIT`, `SPATIAL_GRID`, `NVECTRI`, `COUNT`, `COORDMAX`, `AREA_SI`, `SET_IOBP`, `SET_UG_IOBP`, `GET_BOUNDARY` | 2,077 / 848 |
| geographic propagation (N scheme, sub-cycling) | `PDLIB_W3XYPUG`, `PDLIB_W3XYPFSN2` | 574 / 224 |
| spectral propagation (refraction, frequency shift) | `W3KTP3`, `W3QCK1`, `W3QCK2`, `UG_GRADIENTS` | 1,318 / 360 |
| ST4 | `W3SPR4`, `W3SIN4`, `CALC_USTAR`, `W3SDS4` | 1,604 / 737 |
| ST4 tables | `INSIN4`, `TABU_STRESS`, `TABU_TAUHF`, `TABU_TAUHF2` | 778 / 356 |
| DIA | `W3SNL1`, `INSNL1` | 663 / 261 |
| LN1 | `W3SLN1` | 156 / 32 |
| DB1 | `W3SDB1` | 287 / 97 |
| BT4 | `INSBT4`, `TABU_ERF`, `W3SBT4` | 455 / 149 |
| source integration, limiter, MLIM | `W3SRCE` | 2,223 / 525 |
| forcing, ice, boundary spectra | `W3UWND`, `W3UCUR`, `W3ULEV`, `W3UICE`, `W3UBPT`, `W3IOBC` | 2,403 / 640 |
| interpolation of the forcing on the nodes | `ww3_prnc` | 2,698 / 1,487 |
| outputs (all fields) | `W3OUTG` | 1,180 / 665 |
| time loop | `W3WAVE` | 2,769 / 1,039 |
| total | | 19,185 / 7,420 |

The active count still holds code that the Rust port does not need:

- MPI and parallel bookkeeping;
- the many output fields of `W3OUTG`, of which $H_s$ and $T_{m02}$ need
  about 50 lines;
- the general file handling of `ww3_prnc`, which reduces to bilinear
  weights and a netCDF reader.

The `hww3-grid` crate (3,828 lines of Rust) already provides much of the
setup:

- the mesh and its NESTED ids (`cdshealpix`);
- the WW3 group speed;
- the explicit time step of each node, computed like `PDLIB_W3XYPFSN2`
  (`cfl.rs`), from which the sub-cycling per frequency follows.

### 2.2 Effort per component

A person-week is a week of focused work by someone who knows the WW3
physics, Rust and GPU programming. Each figure includes the unit tests
against the Fortran. The *forward* column covers the GPU forward model. The
*derivatives* column covers tangent and adjoint kernels, with hand-written
adjoints of fused kernels.

| component | forward (pw) | derivatives (pw) | notes |
|---|---|---|---|
| mesh, geometry, boundary flags | 1–2 | 0 | reuses `hww3-grid`; node-to-triangle lists in compressed sparse rows for the GPU |
| geographic propagation | 2–3 | 2–3 | per-frequency sub-cycling, open-boundary values, coastal no-inflow masks; the adjoint is the transposed sweep (1 pw); derivatives with respect to depth and currents need the sub-step states (1–2 pw) |
| spectral propagation | 2–4 | 1–3 | 1 pw with first-order upwind (linear, transposed); 3 pw to keep the WW3 limiter |
| ST4 and its tables | 6–11 | 5–10 | the largest item: $u_*$ iteration, tail-stress tables, swell dissipation, saturation, cumulative term on GPU; tables rebuilt in the differentiated code for `BETAMAX`. Done in two steps (section 2.5): 2–5 pw for the simplified ST4 of M2a, then 4–6 pw for the tables and the directional cumulative term |
| DIA | 1–2 | 1 | fixed gathers and scatters; the adjoint swaps them |
| LN1 | 0.5 | 0.2 | |
| DB1 | 0.5–1 | 0.5 | |
| BT4 | 1–2 | 1 | analytic erf instead of the table |
| source integration, limiter, MLIM | 2–3 | 1–2 | dynamic step per node; the backward pass replays the recorded step count |
| forcing, ice, boundary spectra, restart reader | 3–5 | 0.5–1 | netCDF, bilinear weights, time interpolation, relative wind; WW3 restart files as initial state |
| outputs $H_s$, $T_{m02}$ with the tail | 0.5–1 | 0.3 | |
| time loop, splitting, configuration | 2–3 | 2–3 | checkpointing driver (revolve), cotangents injected at the output times |
| observation operators | 1–2 | 0.3 | the matching of the validation scripts as sparse matrices (section 7) |
| PyTorch bridge | 1–2 | 1 | PyO3 and DLPack, the model wrapped as a `torch.autograd.Function` for FOSCAT and the networks |
| GPU performance work | 2–4 | 1–2 | |
| total | 25–46 | 17–28 | |

The Rust code would be about 5,000–7,000 lines for the forward model,
2,500–4,000 for the derivatives and 2,000–3,000 for the tests.

With the source terms written as Burn tensor operations instead of fused
kernels, Burn's tape differentiates them. That replaces 9–15 pw of
hand-written adjoints (ST4, DIA, LN1, DB1, BT4, integration) by 2–3 pw of
glue, so the derivatives column becomes 10–17 pw. The price:

- unfused operations make the source terms 3–10 times slower;
- the tape of one source evaluation holds tens of state-sized arrays, so
  the nodes must be processed in chunks;
- the per-node dynamic step of WW3 becomes a fixed number of sub-steps,
  which departs from the Fortran.

### 2.3 Validation

| item | pw | compute |
|---|---|---|
| Fortran unit harness: drive `W3SIN4`, `W3SDS4`, `W3SNL1`, `W3SDB1`, `W3SBT4`, `W3SRCE` on spectra read from restart files and dump the tendencies | 1–2 | minutes |
| propagation-only Fortran runs: sources off (`RUN%FLSOU = F` in `ww3_grid.nml`), from `restart001.ww3` of the grid v5 $K=4$ explicit run (8 January 00 h), 48 h; an idealised swell from `ww3_strt`; comparison tools | 1–2 | under 300 core-hours |
| full runs against the existing explicit runs, and debugging | 3–8 | GPU, plus one WW3 run. The references are on scratch: `runs/natl_v5k4_jan2026_exp` (hourly $H_s$, $T_{m02}$, 1–8 January, restart at 8 January) and `runs/natl_v4_jan2026_exp` (PBS job 7951889, 1 January – 7 January 21 h). M2a needs one more reference: the grid v5 $K=4$ explicit run repeated with `SINTABLE = 0` and `SDSCUM = 0`, under 570 core-hours |
| gradient checks (section 6) | 2–4 | Fortran finite differences: 3,000–6,000 core-hours |
| total | 7–16 | |

Debugging is the least predictable item. Typical causes are races in the
scatter of the N scheme, the order of the split steps, the boundary flags,
and single-precision accumulation.

### 2.4 Totals

| scenario | to M2a: GPU forward model with the simplified ST4, finite-difference gradients | to M2b: full ST4 | to M4: reverse mode for the fields |
|---|---|---|---|
| one developer (WW3, Rust, GPU), hand-written adjoints | 26–52 pw (6–12 months); central value about 35 | 30–59 pw (7–14 months); central value about 40 | 52–98 pw (12–23 months); central value about 70 |
| same, with Burn tensor operations for the source terms | same | same | 46–87 pw |
| Jean-Marc directing coding assistants (his own time) | 10–28 pw; central value about 16 | 11–31 pw; central value about 18 | 17–48 pw; central value about 30 |

Going through the simplified ST4 first costs about the same as porting the
full ST4 at once: up to 1 pw more, for the extra reference run and the
rework of the source kernels.

The assistant scenario assumes the following share of Jean-Marc's time
compared with the solo estimate:

- 25–40 % on coding tasks: porting Fortran routines that a unit harness
  checks, writing kernels, adjoints and tests, where assistants write most
  of the code and he reviews it;
- 60–80 % on validation, debugging, GPU tuning and physics decisions.

Calendar time depends on his availability. At half time, M2a takes roughly
5–13 months, M2b 5–14 months and M4 8–22 months.

### 2.5 What moves the range

- **Fidelity to ST4.** The decision is to port a simplified ST4 first
  (M2a), then the full ST4 (M2b). WW3 itself runs the simplified version, so
  the comparison with the Fortran at the centimetre level is kept at both
  stages. The simplified version takes two namelist settings:
  - `SINTABLE = 0` (`&SIN4`). The HWW3 runs use the tables (`SINTABLE = 1`).
    With 0, `INSIN4` builds none of the tables `TAUT`, `TAUHFT` and
    `TAUHFT2`. `CALC_USTAR` then finds $u_*$ by at most 10 Newton iterations
    on the Charnock relation, and `W3SIN4` integrates the tail stress over
    50 wavenumbers in line. Both computations are short, per node and
    directly differentiable with respect to `BETAMAX`, `ZALP` and `ALPHA0`,
    with no table to rebuild.
  - `SDSCUM = 0` (`&SDS4`). This removes the cumulative breaking term of
    `W3SDS4`. The HWW3 runs use −0.40: a negative value selects the
    directional version, a product with the `CUMULW` matrix over the lower
    frequencies, which dominates the flops of a source evaluation. A
    positive value selects a cheap isotropic version, an intermediate step
    if needed.

  The simplified ST4 costs 2–5 pw instead of 6–11. The tables and the
  directional cumulative term then add 4–6 pw at M2b. The scores of the
  simplified configuration are not those of the calibrated one (`BETAMAX`
  1.75 was set with the cumulative term), so scientific use starts at M2b.
- **AD route.**
  - Stopping at forward mode (finite differences or tangent kernels for at
    most about 20 parameters) removes M4, 18–32 pw. It gives no gradients
    with respect to the fields.
  - Burn tensor operations for the source terms save 6–11 pw against fused
    hand-written adjoints, at a cost in speed, memory and fidelity.
  - GPU Enzyme through `std::autodiff` and `std::offload` would remove most
    hand-written adjoints. Re-assessed at M3 (section 3): offload still has
    no NVIDIA device side, and it could not differentiate CubeCL kernels in
    any case, so the tangents and adjoints stay hand-written.
- **Sharing of the work.** Coding assistants cut Jean-Marc's time by a
  factor of 2–3. Validation and physics judgement stay with him.
- **Scope.** Not included, each costing extra:
  - multi-GPU, needed for reverse mode on grid v4 at 36 × 32 without
    host-memory snapshots: +2–4 pw;
  - the implicit scheme with its adjoint by the implicit function theorem
    (one transposed solve per step): +6–10 pw;
  - other physics switches.

## 3. Automatic differentiation in Rust on GPU (November 2026)

| tool | state | use here |
|---|---|---|
| Burn 0.22 with the CubeCL backends | Tape-based reverse mode over tensor operations (`Autodiff<B>`). Balanced gradient checkpointing is available. Custom CubeCL kernels are opaque to the tape and need a hand-written backward. CUDA, ROCm, Vulkan and WebGPU. | **chosen**: tensors and tape; custom kernels for the propagation (hand backward); source terms either as tensor operations (automatic backward) or as fused kernels (hand backward). M0: a sub-step of the N scheme works as a tape operation whose backward is the transpose kernel |
| CubeCL | Rust GPU kernels (`#[cube]`), compiled just in time to CUDA, HIP or WGSL, generic over the float type (f32 or f64). Version 0.11 needs glibc 2.34: it builds on the Ifremer hpc cluster, not on Datarmor (glibc 2.19) | **chosen**: language of all kernels. M0: as fast as the same kernels in CUDA C (3 % slower to 11 % faster), and f64 works |
| `std::autodiff` (Enzyme) | Nightly Rust: `rustup component add enzyme`, `lto = "fat"`, `-Zautodiff=Enable`; experimental, CPU. Re-assessed at M3 (23 November 2026): the CPU side ships with nightly as a rustup component; `std::offload`, which would run Rust functions on the GPU, still lacks its NVIDIA device side (under review, waiting on LLVM 22; its CI tests the CPU side only). Neither can see inside CubeCL kernels, which are compiled at run time from CubeCL's own representation. | not installed (it would need a nightly toolchain, which needs Jean-Marc's agreement). It could give reference derivatives of the CPU transcriptions of the source terms; the M3 tangents are hand-written CubeCL kernels checked against finite differences in double precision |
| `rustc_codegen_nvvm` (Rust-CUDA) | Rebooted in 2025, pinned nightly, last crates.io release 2022 | not recommended |
| NVIDIA CUDA Rust: `cuda-oxide` (SIMT) and `cutile-rs` (tiles) | Announced September 2026, experimental. `cuda-oxide` 0.3 (October 2026) needs a pinned nightly and CUDA 12 or later; `cutile-rs` needs CUDA 13.3 (the driver of the Datarmor A100 nodes supports CUDA 11.4 at most; the hpc cluster has CUDA 13.0). Both need compute capability 8.0 (A100). | a later option for kernels in ordinary generic Rust, where dual numbers would work |
| `cudarc` with CUDA C kernels (NVRTC) | Stable and widely used; CubeCL's CUDA backend is built on it | M0 fallback, not needed. It runs the same kernels on the Datarmor A100s, compiled with the NVRTC of CUDA 11.4, as a cross-check |

**What must be written by hand, in every option:**

- **The adjoint of the geographic propagation.**
  - For fixed velocities the N scheme is linear in $N$:
    $N^{m+1} = L_c N^m$ for each component $c$. The $I_c$ sub-steps of a
    600 s step use the same $L_c$, so the adjoint is $(L_c^T)^{I_c}$. It is
    the same triangle loop with gather and scatter swapped, and it needs no
    stored sub-steps. Dot-product tests check it to machine precision. At
    M0 they pass to round-off: normwise errors of at most
    $2.2 \times 10^{-10}$ in single precision, relative errors of at most
    $1.4 \times 10^{-12}$ in double precision (section 9.1).
  - Derivatives with respect to depth, sea level and currents go through
    $c_g$ and need the sub-step states, which are recomputed per frequency.
- **The adjoint of the spectral propagation.** It is linear and transposed
  with first-order upwind; the WW3 limiter needs its own adjoint.
- **The time-loop driver.** It handles binomial checkpointing over the
  1,008 steps of 7 days, the cotangents of the observations at the hourly
  outputs, and the recorded per-node source step counts.
- **The source-term adjoints**, if they are fused kernels. They are local to
  each node, so they are the classical case for hand derivation. Their
  forward-mode counterparts are mechanical and come first.

**Forward mode for parameters.**

- Finite differences in double precision on the GPU cost no code: CubeCL
  kernels are generic over f64. For $p$ parameters they need $p+1$ forward
  runs. At M0 a double-precision sweep costs 1.9 times a single-precision
  one.
- Tangent kernels give the exact directional derivative in one run with
  $p$ tangent lanes and need no checkpointing.
- Dual numbers carried by generic kernels would avoid writing tangent
  kernels, but need a kernel language that accepts a user-defined number
  type: ordinary Rust through `cuda-oxide`, or a CubeCL custom type. A
  CubeCL dual type was tried at M3 and does not work: CubeCL structs cannot
  be reassigned and their values move on use, so the arithmetic of the
  kernels cannot be written over them. The M3 tangent kernels are written
  by hand, each line of the value kernel with its derivative next to it
  (`srce_tl`, `prop_tl`). Hand-written tangent kernels cost about half of
  the forward effort of the source terms.

**Reverse mode for the fields** (wind at every node and time step) is the
only way to get gradients with respect to $10^8$ controls. It needs the
adjoints above and checkpointing (section 5).

## 4. Expected speed

All estimates assume one A100 80 GB, single precision and fused kernels.
The A100 has a peak bandwidth of 2.0 TB/s; the estimates take 1.0 TB/s as
the effective bandwidth for gather-heavy kernels, and 8 TFLOP/s of FP32
(19.5 peak).

| per 600 s step, grid v4 | work | time |
|---|---|---|
| geographic propagation | $\sum_f \max_\theta I = 110$ frequency sweeps × 794,613 nodes × about 800 B (row of 36 directions read and written, neighbour rows, coefficients) = 70 GB | 0.07 s (measured at M0: 0.103 s) |
| spectral propagation | about four passes over the state, 30 GB | 0.03 s |
| one source evaluation | about $1.5 \times 10^6$ flop per node, dominated by the ST4 cumulative term (a matrix product over the lower frequencies, circulant in direction) = 1.2 TFLOP | 0.15 s |
| sources | 1 to 4 evaluations per step on average (WW3's dynamic step) | 0.15–0.6 s |
| total | | 0.25–0.7 s, i.e. 40–100 s per model day |

A second estimate scales the measured WAM6-GPU timing (Yuan et al. 2024):
65.2 s per model day on eight A100 for 2,923,286 grid points with
24 directions × 25 frequencies, a 43 s propagation step and a 5 min source
step. That is 522 A100-seconds per model day. Per model day, grid v4 needs:

- 0.12 times the WAM6 propagation work: 460 full-state sweeps of
  $9.15 \times 10^8$ values, against 2,009 steps of $1.75 \times 10^9$;
- 0.26 to 1.05 times its source work: 144 to 576 evaluations against 288.

Depending on how the WAM6-GPU time splits between propagation and sources
(not given in its text), this gives 60–550 s per model day.

**Result.**

- **Grid v4:** 40–550 s per model day on one A100, against 4,318 s for the
  Fortran explicit scheme on 112 cores: 8–100 times faster.
- **Grid v5 $K=4$:** about half of the v4 figure (fewer nodes, 83 frequency
  sweeps instead of 110), against 2,601 s.
- **Measured at M0 (geographic propagation only, single precision).**
  - One 600 s step takes 0.103 s on grid v4 (14.8 s per model day) and
    0.043 s on grid v5 $K=4$ (6.2 s). The transposed step takes 16.4 and
    7.0 s per model day.
  - That is 1.5 times the bandwidth estimate. The kernels move 156 to
    339 GB/s: instruction count and gather latency bound them, not DRAM
    bandwidth.
  - Computing the coefficients of the 32 frequencies again at every step,
    when the currents or the sea level change, adds about 11 %.
  - With this time the bandwidth estimate for grid v4 becomes 41–106 s
    per model day; the full range stays 40–550 s.
  - These times count the interior nodes in `ITER`. WW3 counts every wet
    node (`FSBCCFL = T`, found at M1, section 9.1): 258 frequency sweeps
    per step on grid v4 and 287 on grid v5 $K=4$ instead of 110 and 83.
    The step then takes 0.240 s on grid v4 (34.6 s per model day) and
    0.148 s on grid v5 $K=4$ (21.3 s), and the estimates that use the M0
    time grow by 19.8 s per model day on grid v4 (61–126 s instead of
    41–106 s, 42–51 s instead of 22–31 s below).
- **The lower end** is reached with fused kernels and one or two source
  evaluations per step.
- **Burn tensor operations for the sources** put the speed-up nearer to the
  upper bound of 550 s per model day.
- **The simplified ST4 of M2a** has no directional cumulative term. A source
  evaluation is then limited by memory traffic (about 20 GB, 0.02 s) rather
  than by flops. The bandwidth estimate falls to 17–26 s per model day on
  grid v4, set by the propagation; with the geographic propagation
  measured at M0 it is 22–31 s.
- **A 7-day forward run on grid v5 $K=4$** should take between 2 and
  30 minutes.

## 5. Memory and checkpointing

| configuration | values per state | single precision | all 1,008 steps of 7 days |
|---|---|---|---|
| grid v4, 36 × 32 | $9.15 \times 10^8$ | 3.66 GB | 3.7 TB |
| grid v5 $K=4$, 36 × 32 | $4.89 \times 10^8$ | 1.95 GB | 2.0 TB |
| grid v5 $K=4$, 24 × 30 | $3.05 \times 10^8$ | 1.22 GB | 1.2 TB |
| Channel–Biscay box (v4), 36 × 32 | $2.47 \times 10^7$ | 0.099 GB | 100 GB |

**Forward model.** The state, the forcing of two hours and the coefficients
take 10–15 GB on grid v4, so the forward model fits easily.

**Reverse mode.** Binomial checkpointing (revolve) with $s$ snapshots over
$n = 1{,}008$ steps uses the smallest $r$ with $\binom{s+r}{s} \ge n$ and
needs $r\,n - \binom{s+r}{r-1}$ forward steps:

| snapshots $s$ | $r$ | forward steps / $n$ | v4 36 × 32 | v5 $K=4$ 36 × 32 |
|---|---|---|---|---|
| 8 | 5 | 4.29 | 29 GB | 16 GB |
| 15 | 4 | 3.04 | 55 GB | 29 GB |
| 30 | 3 | 2.48 | 110 GB | 59 GB |

The working set of one step is 3–6 states. Grid v4 at 36 × 32 therefore
reaches about 75 GB with 15 snapshots: at the limit. It needs 8 snapshots,
snapshots in host memory (3.66 GB over PCIe in about 0.15 s, overlapped with
the next step), or two GPUs. Grid v5 $K=4$ fits with 15–30 snapshots.

A gradient costs about 3 recomputations plus a backward pass of about twice
a forward run, i.e. 5–6 forward runs: 10 min to 3 h on grid v5 $K=4$ for
7 days.

### Test case: the Channel–Biscay box

`scripts/subdomain.py` keeps the triangles of grid v4 with all three nodes
in 13°W–2.5°E, 42.5°–51.5°N (PBS job 7963807); `scripts/explicit_cost.py`
then gives the sweeps per step:

```bash
python scripts/subdomain.py grids/natl_v4 sub_v4 -13 2.5 42.5 51.5
python scripts/explicit_cost.py sub_v4 sub_v4_cost "v4 Channel-Biscay 36x32"
python scripts/subdomain.py grids/natl_v4 sub_v4_s2430 -13 2.5 42.5 51.5 30 24   # 30 frequencies, 24 directions
python scripts/explicit_cost.py sub_v4_s2430 sub_v4_s2430_cost "v4 Channel-Biscay 24x30"
```

(every wet node counted, as WW3 with `UGBCCFL = T`; `--interior-only` for
the interior count used at first).

| | grid v4 | grid v5 $K=4$ |
|---|---|---|
| nodes (nside 128 / 256 / 512 / 1024 / 2048) | 21,425 (19 / 281 / 1,211 / 5,211 / 14,703) | 11,492 (19 / 281 / 1,211 / 8,558 / 1,423) |
| triangles | 41,313 | 21,796 |
| open-boundary / coastal boundary nodes | 195 / 1,358 | 146 / 1,050 |
| $S$ per 600 s step, 36 × 32 (ITER at 0.0373 Hz), interior nodes | 3,080 (8) | 2,619 (5) |
| $S$ per 600 s step, 36 × 32 (ITER at 0.0373 Hz), every wet node as WW3 | 6,308 (17) | 6,041 (15) |
| $S$ per 600 s step, 24 × 30, interior nodes / every wet node | 2,006 / 4,152 | 1,698 / 3,977 |

The box holds 58 of the 268 buoys (46 Channel and Biscay, 9 southern North
Sea, 3 Celtic Sea; 49 with $T_{m02}$). This is where the buoy errors are
largest ($H_s$ RMSE 0.535 m, $T_{m02}$ RMSE 1.436 s with the tide) and the
tidal currents strongest. A state takes 99 MB, so a 7-day trajectory needs
no special checkpointing.

Its boundary spectra come from a full-domain run with point output at the
195 nodes. Since the swell enters through the boundary, the box tests the
local wind sea and the coastal physics. It is the development and
gradient-check case; calibrating `BETAMAX` needs the whole domain.

## 6. Validation and gradient checks

**Forward.**

1. **Kernels.** Each kernel is checked against the Fortran routine on the
   same spectra, through the unit harness (section 2.3).
2. **Propagation alone.** Starting from `restart001.ww3` of the grid v5
   $K=4$ explicit run, with sources off (`RUN%FLSOU = F`), the GPU model is
   run for 48 h with the same currents, sea level and boundary spectra.
   Targets: $H_s$ RMS difference below 1 cm at the median node and 5 cm at
   the 99th percentile, total energy within 0.5 %.
3. **With sources.** At M2a, the GPU model with the simplified ST4 is
   compared with a WW3 explicit run made with the same settings
   (`SINTABLE = 0`, `SDSCUM = 0`, grid v5 $K=4$, 1–8 January). At M2b, the
   full model is compared with the explicit runs already on scratch (grid v5
   $K=4$, 1–8 January; grid v4, 1 January – 7 January 21 h). The targets are
   the same at both stages:
   - a node-by-node difference below the WW3 explicit-minus-implicit
     difference on grid v4 (2.1 cm at the median node, 20 cm at the 99th
     percentile).
   - altimeter and buoy scores of `validate_altimetry.py` and
     `validate_buoys.py` within 5 mm of those of the WW3 run.

**Gradients.**

1. **Dot-product tests** for each linear kernel and its adjoint,
   $\langle L\,\delta x, \bar y\rangle = \langle \delta x, L^T\bar y\rangle$.
2. **Tangent against finite differences** of the GPU model in double
   precision, for `BETAMAX`, `SDSC2`, `NLPROP`, `BJGAM` and a wind factor.
3. **Taylor test** of the reverse mode on the box in double precision:
   $|J(x+\varepsilon d) - J(x) - \varepsilon\, g\cdot d|$ must fall as
   $\varepsilon^2$ over three decades.
4. **Comparison with WW3.** Central finite differences of Fortran runs,
   with two step sizes and the noise floor of a repeated run:
   - for parameters, the parameter is perturbed;
   - for fields, a smooth wind perturbation (one nside-32 cell over the Bay
     of Biscay, 6 h) is added to `wind.nc` before `ww3_prnc`. The WW3
     `WCOR` correction is not compiled in.

   The comparison uses the response of every buoy-hour and altimeter block,
   not only $J$. Target: correlation above 0.9 and regression slope within
   20 %.
5. **Large step.** The measured `BETAMAX` change from 1.43 to 1.75 shows
   how far the linearisation holds.

## 7. Observation operators

The validation scripts define linear operators on the hourly fields:

- **Altimetry.** In `validate_altimetry.py`, each sample takes the nearest
  wet, ice-free node within 0.75 pixel and is interpolated linearly in time;
  blocks are means of 7 samples. This gives a fixed sparse matrix
  $\mathbf{A}$ with at most 14 non-zeros per row: 45,493 rows over 4–7
  January on grid v4, 328,902 over 4–31 January.
- **Buoys.** In `validate_buoys.py`, each buoy takes the nearest wet node
  within 10 km and the hourly bins: matrices $\mathbf{B}$ for $H_s$ (268
  buoys, 167,298 hours) and $\mathbf{B}'$ for $T_{m02}$ (180 buoys, 113,744
  hours).

$H_s = 4\sqrt{m_0}$ and $T_{m02} = \sqrt{m_0/m_2}$, with the tail of
`W3OUTG`. Their derivatives are

$$
\frac{\partial H_s}{\partial E_{kj}} = \frac{8}{H_s}\,w_{kj}, \qquad
\frac{\partial T_{m02}}{\partial E_{kj}} = \frac{T_{m02}}{2}\Big(\frac{1}{m_0} - \frac{f_k^2}{m_2}\Big) w_{kj}.
$$

The loss is a weighted sum of squares, or a Huber loss, with the known
exclusions: the nine buoys whose $T_{m02}$ RMSE exceeds 3 s. The matching is
computed once and stays fixed. A test applies the matrices to the stored
`ww3.202601.nc` and must reproduce `results/natl_v4_jan2026_tide_*_stats.json`.

A week holds about 85,000 observations against $2.7 \times 10^8$ wind
values, so corrections of the forcing must be low-dimensional (section 8).

## 8. HEALPix structure for learned corrections

Every node is a NESTED pixel $(d_i, p_i)$ (`raw/uniq.u64`, `raw/level.u8`).
A field $c$ on a uniform NESTED map at depth $d_c$ reaches the nodes through
an exact sparse operator: the parent value
$c[p_i \gg 2(d_i - d_c)]$ for nodes finer than $d_c$, and the mean of the
$4^{d_c-d_i}$ children for coarser nodes. Only integer shifts are involved:
there is no interpolation, and the transpose brings node gradients back to
the map.

The wet domain covers 9.9 % of the sphere: 78,000 pixels at nside 256 (the
ERA5 scale) and $5.0 \times 10^6$ at nside 2048 (20 MB per field). Even
coastal-scale corrections can therefore be computed on uniform maps.

- **Multi-scale fields.** $\delta u = \sum_l P_l\,c_l$ with penalties
  growing with the level: about $4 \times 10^6$ coefficients per week from
  nside 16 to 256 at 6-hourly knots, instead of $2.7 \times 10^8$.
- **Learned corrections.** FOSCAT operators on the uniform map
  (convolutions on the pixel and its eight neighbours, NESTED 4-to-1
  pooling, `HealpixUNet`), mapped to the nodes by $P$, through the PyTorch
  bridge of section 2.2.
- **Statistical prior.** The scattering covariance of FOSCAT can keep the
  multi-scale statistics of the corrected wind close to those of ERA5 or of
  scatterometer winds.
- **Spectral corrections** (hybrid source terms) act per node along the
  spectral axes: circular convolutions in $\theta$, conditioned on $u_*$,
  $kd$ and the wave age, with zero net energy and momentum for a correction
  of $S_{nl}$.

## 9. Staged plan with go/no-go milestones

| milestone | content | go if | pw (one developer) |
|---|---|---|---|
| M0 | Rust stable with the CubeCL and Burn crates and their CUDA backend on one A100; N-scheme sweep and its transpose on grids v4 and v5 $K=4$ with the sub-cycling of `cfl.rs` | a frequency sweep on grid v4 within twice the bandwidth estimate (about 1.3 ms); dot-product test passes; the toolchain builds and runs on the GPU nodes. Otherwise switch the kernels to `cudarc` and CUDA C. **Done: go** (section 9.1) | 1–2 |
| M1 | propagation-only model: geographic and spectral propagation, depth, currents, sea level, boundary spectra, ice mask, restart reader | the propagation-only targets of section 6. **Targets met** on 22 November 2026: once `ITER` counts the coast nodes as WW3 does, $H_s$ is within 1.5 µm (median node) and 24 µm (99th percentile) RMS of WW3 over 48 h, energy within 2.4e-7, 65 s on one A100 against 813 s on 32 cores (section 9.1, [GPU model results](wave_results.md)) | 10–18 |
| M2a | forward model with the simplified ST4 (`SINTABLE = 0`, `SDSCUM = 0`), DIA, LN1, DB1, BT4, integration and limiter, outputs, observation operators | the targets with sources of section 6 against the WW3 run with the same settings. **First gradients:** finite differences in double precision for `BETAMAX` and a few parameters, at GPU speed, to build and test the calibration chain | 15–32 |
| M2b | full ST4: the stress tables `TAUT`, `TAUHFT`, `TAUHFT2` and the directional cumulative term (`SDSCUM = −0.40`) | the targets with sources of section 6 against the existing explicit runs, and at least 10 times the Fortran explicit speed (under 430 s per model day on grid v4). **First scientific gradients** with respect to the parameters of the calibrated configuration. **Done** on 23 November 2026: targets met on grids v5 $K=4$ and v4 at 220 s per model day on grid v4, and finite-difference gradients of the energy, mean $H_s$ and scores for `BETAMAX`, `SDSC2`, `NLPROP`, `BJGAM` and the wind (section 9.3) | 4–7 |
| M3 | tangent kernels for parameters (sources and stress tables) | agreement with the double-precision finite differences to $10^{-4}$ where the model is smooth. **Met** on 23 November 2026 for one source step and over 3 h; over 48 h the model is smooth almost nowhere and the tangent gives the derivatives of the aggregates to 0.1–4 % (section 9.4) | 4–7 (optional) |
| M4 | reverse mode for the fields: adjoint kernels, revolve driver, adjoint of the observation operators, PyTorch bridge | the Taylor test and the comparison with WW3 of section 6; one wind gradient over 7 days on grid v5 $K=4$ in under 3 h | 18–32 (hand-written adjoints) or 12–21 (Burn source terms) |
| M5 | learning: a regional wind bias, $\beta$ from a small network of wind features, coastal parameters as functions of grid descriptors (section 9.7) | an independent month improves | open |

Two decision points follow from the plan:

- **After M2b,** decide whether gradients with respect to the parameters
  (finite differences or M3) are enough for the first scientific results.
- **Before M4,** choose between hand-written fused adjoints and Burn tensor
  operations for the source terms, based on the M0 and M2 timings.

### 9.1 Result of M0 (21–22 November 2026)

**Go: the three criteria are met.** M1 started on 22 November 2026. The
code is described in [GPU wave model](wave_model.md) and all the results,
M1 included, in [GPU model results](wave_results.md).

**What was built.** The crate `hww3-wave` reads an HWW3 grid and builds
what WW3 builds for `PDLIB_W3XYPFSN2`: inward normals, the boundary flags,
the nodal velocities and `ITER(f, θ)`. It holds the explicit N-scheme
sweep and its transpose in three variants, each in CubeCL and in CUDA C.

- All variants are gathers over the triangles around a node, with no
  atomics.
- `two_phase_x4` computes the triangle values first, then gathers them at
  the nodes, four directions per thread.
- The sub-cycling matches `scripts/explicit_cost.py`: $S$ = 3,678 on grid
  v4 (110 frequency sweeps per step) and 2,837 on grid v5 $K=4$ (83).
  Both count the interior nodes only. The M1 comparison with WW3 found that
  WW3 counts every wet node: `PDLIB_W3XYPFSN2` takes `DTMAX` over the nodes
  with `IOBP = 1 .OR. FSBCCFL`, and `FSBCCFL` comes from `UGBCCFL` in the
  `UNST` namelist of `ww3_grid`, true by default and in the HWW3 runs. The
  small coast cells then set the sub-cycling: 258 frequency sweeps per step
  on grid v4 and 287 on grid v5 $K=4$ (`ITER` at the lowest frequency 18–24
  and 19–26 instead of 8–10 and 4–5). With the interior count, the coast
  nodes in a single triangle ran at local CFL numbers up to 4.8, which gave
  differences up to 0.48 m in $H_s$ at the coast (Slurm jobs 4119699 to
  4120565). With the count of WW3 the sweep times below stand, and the step
  times and per-day times grow by 2.3 on grid v4 and 3.5 on grid v5 $K=4$
  (0.240 and 0.148 s per step, 34.6 and 21.3 s per model day).
  `scripts/explicit_cost.py` now counts them too ($S$ = 8,326 sub-steps
  per step on grid v4 and 9,003 on grid v5 $K=4$, against 3,678 and 2,835),
  and the `cfl_substeps` cap of `hww3-grid` does with
  `refine_bathy.cfl_boundary`.

**Platforms.**

- **Datarmor.** CubeCL 0.11, the version Burn 0.22 needs, does not build:
  its CUDA backend needs a prebuilt LLVM that requires glibc 2.34, and
  Datarmor has glibc 2.19. The driver of the A100 nodes (470.239.06)
  supports CUDA 11.4 at most, so PTX from the NVRTC of the CUDA 12.1
  module is rejected (PBS job 7964674).
- **Ifremer hpc cluster.** CubeCL and Burn build and run there: RHEL 9.4,
  glibc 2.34, A100-SXM4-80GB (the same GPU as Datarmor), CUDA 13.0. This
  answers decision 1: CubeCL with Burn, developed on the hpc cluster.
- **Fallback.** The same kernels in CUDA C, compiled by the NVRTC of
  CUDA 11.4 and launched through `cudarc`, run on a Datarmor A100 (PBS
  job 7964774). They stay as a cross-check.

**Sweep times**, single precision, `two_phase_x4`, one A100. CubeCL is
Slurm job 4115275 on the hpc cluster; CUDA C is PBS job 7964774 on
Datarmor.

| grid | forward sweep, CubeCL / CUDA C | transpose sweep, CubeCL / CUDA C | 600 s step (CubeCL), forward / transpose | s per model day (CubeCL), forward / transpose |
|---|---|---|---|---|
| v4 | 0.93 / 0.95 ms | 1.03 / 1.07 ms | 0.103 / 0.114 s | 14.8 / 16.4 |
| v5 $K=4$ | 0.51 / 0.50 ms | 0.58 / 0.57 ms | 0.043 / 0.048 s | 6.2 / 7.0 |
| Channel–Biscay box (v4) | 0.033 / 0.033 ms | 0.034 / 0.035 ms | 3.0 / 3.1 ms | 0.43 / 0.45 |

- **Against the threshold.** On grid v4, 0.93 ms against 1.3 ms, and
  1.5 times the bandwidth estimate of 0.64 ms. A simpler kernel with one
  thread per (node, direction) takes 1.95 ms; computing the triangle
  values first brings it to 1.26 ms.
- **CubeCL against CUDA C.** From 3 % slower to 11 % faster in single
  precision. Writing the kernels in Rust costs nothing measurable here.
- **Double precision.** 1.9 times single precision (1.77 ms on grid v4).
  It works in CubeCL; the block size of the f64 kernels was not tuned.
- **What the times cover.** The geographic propagation only: no spectral
  propagation, no source terms, no checkpointing.

**Checks.**

- **Dot-product tests**
  $\langle L\,\delta x, \bar y\rangle = \langle \delta x, L^T \bar y\rangle$
  for one sub-step and for a 600 s step of frequencies 1, 17 and 32, on
  the three grids, with both backends:
  - single precision: normwise errors of at most $2.2 \times 10^{-10}$
    (relative errors up to $3.1 \times 10^{-6}$ where the products
    cancel, on grid v5 $K=4$);
  - double precision: relative errors of at most $1.4 \times 10^{-12}$.
- **Through Burn.** With a sub-step as an operation of Burn's tape, the
  test passes with a relative error of $3.8 \times 10^{-8}$ on grid v4.
  The gradient that `backward()` returns is identical, bit for bit, to the
  output of the transpose kernel.
- **Against WW3's loop.** The GPU step is compared with
  `PDLIB_W3XYPFSN2` written as in WW3 (loop over triangles with scatter)
  and run on the CPU in double precision. The largest difference is
  $1.6 \times 10^{-6}$ of the largest value in single precision and
  $2.4 \times 10^{-15}$ in double precision.

**Left for later milestones.**

- **The clip.** The forward step clips negative values (`max(0, ·)`); the
  transpose of M0 is that of the linear part. The tangent and adjoint
  steps (M3, M4) must also apply the clip mask, rebuilt from the stored
  forward state.
- **Coefficients.** Recomputing the triangle coefficients of the 32
  frequencies takes 0.012 s on grid v4, about 11 % of a step, when the
  currents or the sea level change at every step.
- **Tuning.** The transposes are 10–14 % slower than the forward sweeps.

### 9.2 Progress of M2a (from 22 November 2026)

- **References.** WW3 with `SINTABLE = 0` and `SDSCUM = 0`, everything
  else as the grid v5 $K=4$ explicit run: 48 h from the restart of
  8 January (Slurm job 4122569, 27 core-hours on 32 ranks, with restarts
  after the first three steps) and the cold-started week of 1–8 January
  (job 4122614, 5,003 s and 88 core-hours on 64 ranks), on the hpc
  cluster.
- **Harness.** The unit harness of section 2.3 is linked against the hpc
  WW3 build. It calls the source terms node by node and writes their
  inputs, outputs and the WW3 parameters and tables they read.
- **CPU ports.** The double-precision transcriptions of `W3SPR4`,
  `CALC_USTAR`, `W3SIN4` with the tail stress in line, `W3SLN1`,
  `W3SNL1`, `W3SDS4` without the cumulative term, `W3SDB1`, `W3SBT4` and
  the whole `W3SRCE` integration over a 600 s step reproduce WW3 on 1,869
  nodes to the round-off of the single-precision WW3: 1e-6 to 5e-5 for
  each source term, 4.3e-6 for the spectrum after the step, the same
  dynamic steps and limiter decisions at every node (Slurm job 4123121).
- **GPU kernel.** One CubeCL launch runs the whole `W3SRCE` integration of a
  global step, one cube per node with the spectrum in shared memory. On
  the harness nodes it reproduces WW3 to 3.1e-6 in single precision
  (0.9 µm in $H_s$) with the same dynamic steps at every node, and the CPU
  transcription to 4.6e-8 in double precision (job 4138915). A step of
  source terms on 373,800 nodes takes 0.97 s on one A100.
- **Time loop.** `hww3-prop` runs the source kernel after the propagation
  of each step, with the wind of `W3UWND` (time interpolation of `WNT2`,
  wind relative to the current). After the first three steps from the
  restart of 8 January, the spectra agree with the WW3 restarts to within
  1 mm in $H_s$ at all but 13 of 371,000 nodes, at most 7.9 mm (job
  4138987).
- **Targets met.** Over 48 h from 8 January and over 1–8 January, the RMS
  difference in $H_s$ against WW3 is 2.1 and 2.5 µm at the median node,
  0.09 and 0.11 mm at the 99th percentile (targets 2.1 cm and 20 cm), and
  the altimeter and buoy scores of the week are WW3's to 1 µm (jobs
  4139369 to 4139371). One A100 takes 146 s per model day. A few nodes
  (8 of 371,000) have one-hour spikes of up to 23 cm in the highest
  frequency bands. These are young seas that take up to 47 dynamic steps
  per step, where round-off-level differences decide whether the bands
  above the cut-off are grown or reset to the tail. From WW3's own state
  the source terms agree to 4.8e-7 there (jobs 4162035, 4162053, 4162054
  and 4162062).

Details and the list of what remains are in
[GPU model results](wave_results.md).

### 9.3 Progress of M2b (from 22 November 2026)

- **Reference.** WW3 with the calibrated namelists (stress tables,
  `SINTABLE = 1`; cumulative breaking term, `SDSCUM = -0.40`): 48 h from
  the restart of 8 January with restarts after the first three steps
  (Slurm job 4139131).
- **Harness and CPU ports.** The harness writes the stress tables and
  `CUMULW` from the calibrated `mod_def.ww3`. The CPU transcriptions of
  `CALC_USTAR` with `TAUT`, the tail of `W3SIN4` with `TAUHFT2` and the
  cumulative term of `W3SDS4` reproduce WW3 to 1.5e-7, 2.7e-6 and 1.7e-5
  on 1,869 nodes, and the whole `W3SRCE` step to 3.9e-6 (job 4139132).
- **GPU.** The kernel with the same branches reproduces WW3 to 4.0e-6 on
  the harness nodes (job 4139172). Over 48 h from 8 January, the RMS
  difference in $H_s$ is 2.0 µm at the median node, 77 µm at the 99th
  percentile and 2.1 cm at worst (job 4162034 against 4139131), at
  127 s per model day on one A100, 11.7 times 32 cores.
- **Week on grid v5 $K=4$.** Against the explicit run
  `natl_v5k4_jan2026_exp` over 1–8 January, the RMS difference in $H_s$
  is 2.2 µm at the median node and 0.10 mm at the 99th percentile. The
  altimeter and buoy scores are those of WW3 to 1 µm, at 124 s per model
  day (jobs 4162038 to 4162040).
- **Grid v4, targets met.** Against the explicit run `natl_v4_jan2026_exp`
  from 1 January 00:10 to 7 January 21 h, the RMS difference in $H_s$ is
  3.6 µm at the median node and 0.14 mm at the 99th percentile. The
  altimeter and buoy scores are WW3's to 1 µm, and one A100 takes 220 s per
  model day against the 430 s of the target (jobs 4162075 to 4162077). The
  largest differences (5.4 cm RMS) are at the mouth of Ungava Bay in
  partial ice, over several hours. Started from a WW3 restart of
  5 January 00:00 in such an episode on grid v5 $K=4$, the GPU stays within
  5 mm of WW3 over 2 h, and the harness finds `W3SRCE` within 2.1e-7 of WW3
  there (jobs 4163959 and 4163960): each step is WW3's, and the difference
  is built up over days, as for the fjord spikes. The GPU run in double
  precision differs from the single-precision one there as much as WW3
  does, at the same nodes (16 cm at 5 January 00 h, job 4165288): round-off
  alone makes the cluster, and the point is closed.
- **Finite-difference gradients.** 15 runs of 48 h in double precision from
  8 January on grid v5 $K=4$ (5 GPU-hours, jobs 4162941 to 4162955 and
  4163276) give the derivatives of the energy, the mean $H_s$ and the
  scores of 8–10 January with respect to `BETAMAX`, `SDSC2`, `NLPROP`,
  `BJGAM` and the wind (job 4163939, [GPU model results](wave_results.md)).
  Steps of 1 % and 0.1 % agree to about $10^{-3}$ on the energy, the mean
  $H_s$ and the biases. Per 1 % change, the altimeter bias moves by
  +26.8 mm for the wind, +5.3 mm for `BETAMAX`, −2.5 mm for `SDSC2`,
  +1.5 mm for `NLPROP` and 0 for `BJGAM`. M2b is complete.

### 9.4 Progress of M3 (from 23 November 2026)

- **Tangent kernels.** `srce_tl` differentiates the whole `W3SRCE`
  integration with the full ST4; `prop_tl` differentiates `W3KTP3` (the
  ULTIMATE QUICKEST face values are linear in the cell values once the
  limiter branch is chosen), the geographic node update with zero boundary
  tangent, the level remap, the wind relative to the current and the
  outputs. `hww3-prop run --tangent NAME` carries the tangent through the
  whole time loop for a relative change of the wind or of `BETAMAX`,
  `SDSC2`, `NLPROP` or `BJGAM`.
- **Sources checked.** On the harness nodes of grids v5 $K=4$ and v4, the
  tangent of one source step agrees with double-precision central
  differences to at most 9.5e-5 on every node where the model is smooth,
  with medians from 1.6e-11 to 3.0e-8, for the spectrum, the wind and the
  four parameters (jobs 4163750 and 4163751, [GPU model
  results](wave_results.md)). It costs 1.2 to 1.7 times the value kernel.
- **Through the time loop.** Without the source terms the tangent of the
  model with respect to the initial spectra agrees with double-precision
  differences to round-off (1.5e-11 at the median node over 3 h, job
  4163934). With them, it agrees to $10^{-4}$ on all but 7 to 30 nodes
  wherever the model is smooth (jobs 4163935 and 4163936), but within three
  hours a wind change of $10^{-4}$ flips some discrete choice of the model
  (dynamic steps, `NKH` and limiter of `W3SRCE`, limiter branches of
  `W3KTP3`) at 97 % of the nodes. On the total energy the tangent then differs from the finite
  differences by 1.5 % for the wind and 0.2 % for the initial spectra, and
  the differences themselves do not converge with the step. The tangent is
  the derivative along the path of the values; finite differences with a
  1 % step include the mean effect of the jumps.
- **Over 48 h.** Against the M2b finite differences (jobs 4163952 to
  4163958), the tangent gives the derivative of the total energy at
  10 January to 0.1 % for `SDSC2`, 0.1–0.2 % for `BETAMAX`, 1.0 % for the
  wind and `NLPROP` and 3.4 % for `BJGAM`, and that of the mean $H_s$ to
  0.15–3.6 %, at 645 s per model day in double precision (1.5 times a
  value run, less than the two runs of a central difference). For a
  calibration this is a gradient within a few per cent of the 1 %
  finite differences, which include the mean effect of the jumps; the
  reverse mode of M4 will see the same jumps.

### 9.5 Calibration of the ST4 parameters (from 23 November 2026)

- **Tables in Rust.** A new value of `BETAMAX` changes the stress tables
  that `ww3_grid` writes in `mod_def.ww3` (`INSIN4`). They are now
  computed by the Rust port: `TABU_STRESS` (`TAUT`, iterated in single
  precision as WW3 does, since `1 - TAUW / TAU` cancels near `TAUWMAX`),
  `TABU_TAUHF` and `TABU_TAUHF2` (`TAUHFT`, `TAUHFT2`, in double precision,
  with their derivative with respect to `BETAMAX` for the tangent). They
  reproduce the tables of WW3 to 7.9e-8 (`TAUT`) and 0.87–1.13e-6
  (`TAUHFT`, `TAUHFT2`) of their largest value at `BETAMAX` 1.75 and at the
  four values of the M2b finite differences, the single-precision
  round-off of WW3, and take 1.8 s. No other precomputed entry depends on
  `BETAMAX`, `SDSC2`, `NLPROP`, `BJGAM` or the wind factor (the saturation
  weights and `CUMULW` of `INSIN4`, the DIA coefficients of `INSNL1`):
  `hww3-src params` writes the parameter file of any values, and
  `hww3-prop --betamax` builds the tables in memory, so the calibration
  needs neither `ww3_grid` nor the harness. A GPU run with the Rust tables
  gives the $H_s$ of a run with WW3's tables to 6.5e-8 m over 2 h (jobs
  4182413 and 4182414, [GPU model results](wave_results.md)).
- **Set-up.** Control vector $\ln p - \ln p_0$ for `BETAMAX`, `SDSC2`,
  `NLPROP`, `BJGAM` (wind as given); training on 5 January 12 h – 7
  January 12 h and validation on 7 January 13 h – 10 January 00 h, after
  the four days the cold start of 1 January takes to settle; cost over
  the altimeter blocks and buoy hours of $H_s$ and $T_{m02}$ with sigmas
  0.25 m, 0.30 m and 1.0 s and a prior of 0.25 on $\ln p$; Gauss–Newton
  with the Jacobian from four tangent runs per iteration (no finite
  difference), each step confirmed by a value run. At the start the
  derivatives of $J$ and of the $H_s$ scores agree with the M2b finite
  differences to 0.2–8 % (control only), those of the $T_{m02}$ scores
  less well because of the jumps of the tail reset, while the $T_{m02}$
  tangent agrees with them node by node to 3–6e-3 where the model is
  smooth.
- **Steps.** `BJGAM` frozen at 0.73 after the first step (few shallow
  buoys constrain it, the altimeters are insensitive to it, it scaled the
  step down to 0.40, its tangent is the least reliable); step 1 on the
  three other parameters accepted (training $J$ −1.31 %; a WW3 run at that
  point gives the GPU's cost to 0.1 on both windows); then options A and C
  of Jean-Marc: one frozen Jacobian ($A_1$, from the f64 tangents at step
  1) and one value run per step, stop when an accepted step gains less
  than 0.2 % ([GPU model results](wave_results.md)).
- **Second cost.** From step 2 (Jean-Marc's decision) the buoy terms use a
  depth-aware interpolation of the grid, linear in the fields ($T_{m02}$
  through the moments $m_0$, $m_2$), and a representativeness error per
  class of buoy (land fraction of the cell, distance to the coast), fixed
  from the residuals of the start run: up to 0.57 m for $H_s$ and 1.46 s
  for $T_{m02}$ above the 0.30 m and 1.0 s of the others. The costs of all
  points are given under the first and the second cost. Steps 2 and 3 are
  accepted (−1.67 % and −0.30 % of the training cost), the validation cost
  falls at each (59,449 at $p_0$, 57,475 at step 3), and step 4 was run
  for information only; every value run now covers both windows
  ([GPU model results](wave_results.md)).
- **Third cost.** After the observation diagnostics (Jean-Marc): 8 buoys
  at sites the mesh cannot represent (barriers, harbour, inner Wadden,
  archipelagos) are left out by a fixed list (`configs/buoy_exclude.csv`),
  the representativeness errors are fitted again without them, and the
  cost uses a Huber loss with $\delta = 1.5$ observation errors. It is
  the cost of the `BETAMAX`$(U)$ test.
- **Spin-up rule.** Never use the first part of a run while the
  simulation settles: a window starts after at least 30 % of its run,
  counted from the cold start of the chain, and `cal_obs.py` refuses
  others. The month-long calibration will start from a spun-up restart or
  drop at least the first 30 % of each run.
- **Result: finalised at step 3 (Jean-Marc, 24 November).** The final
  parameters are `BETAMAX` 1.3725 (x0.784), `SDSC2` x0.649 and `NLPROP`
  x1.137, with `BJGAM` 0.73. From $p_0$ to step 3 the cost falls on both
  windows under every cost, and more on the validation window under the
  third cost: training −3.12 % and validation −3.32 % under the second cost,
  −3.13 % and −4.12 % under the third. A WW3 run at step 3 gives the GPU's
  cost to 0.001 %. The table, the history and the posterior are in
  [GPU model results](wave_results.md). The run took 33 GPU runs and
  16.6 GPU-hours on the A100s, from 23 November 07:27 to 24 November 11:47,
  plus 8.6 h of CPU jobs.
- **Limits of the trial.**
  - One week of January, one region, two short windows (48 h training,
    59 h validation), and three parameters (`BJGAM` frozen after step 1).
  - One Jacobian ($A_1$, at $p_0$) for steps 2–3 (options A and C):
    step 3 gained 0.30 % of the 0.73 % predicted.
  - The posterior standard deviations (0.008–0.024 on $\ln p$, from $A_1$
    at the step-3 residuals) are a lower bound: the Jacobian is not
    refreshed, and the errors of neighbouring blocks and hours are
    correlated.
  - Constant parameters cannot follow a bias that depends on the sea state:
    step 3 fits the moderate seas the altimeters see and worsens the
    storms (buoy bias −75 cm above 6 m). This is what `BETAMAX`$(U)$ now
    tests.
  - The f64 tangents at step 3 grow without bound at a few nodes near the
    sea-ice edge (Newfoundland–Labrador, West Greenland, Ungava Bay) in the
    training window. After it they also grow in the Wadden Sea of the
    German Bight, up to $3 \times 10^{10}$ m per unit of $\ln p$ on 7
    January 19 h, spreading along the North Sea coasts. They reach 87–527 of
    the 24,153 altimeter blocks above 10 m per unit, against none at $p_0$
    (largest 7.5 m), so the step-3 Jacobian is not used.
  - The blocks come from a young wind sea at the ice edge on the coast of
    northern Labrador (ice 0.44, 16 m/s, $H_s$ 3–4 m). There the source
    step multiplies the tangent by 3 to 4.3 per step on 8 of 108 steps,
    and central differences of the step give the same factor. The model
    itself amplifies small perturbations there: the DIA couples the bins
    of the wind sea, and together with the wind input this outweighs the
    damping of the dissipation.
  - Leaving the DIA's tangent out where its own diagonal is large
    (`--tl-nlcut`, `--tl-nlnode`) removes the blow-up at a cut of 0.25:
    no training block above 10 m. But the tangent of one source step then
    departs from the differences by up to 100 % on about 3 % of the
    harness nodes at $p_0$, so the cut is not used.
  - The tangents at $p_0$, on which the `BETAMAX`$(U)$ test relies, reach
    no observation above 10 m. A Jacobian refreshed away from $p_0$ needs
    either a tangent that follows the path of the values through such
    nodes (M4 will meet the same growth) or observations kept away from
    their wake ([GPU model results](wave_results.md),
    `scripts/tl_probe_hpc.slurm`).

### 9.6 Rest of the port to Rust (from 23 November 2026)

- **Parameter file.** The source-term part of `ww3_grid` is ported
  (`hww3_wave::ww3grid`, `hww3-src build-params`): from the namelists and
  the grid it builds the 137 records of the harness parameter file, 121
  identical and the others at WW3's single-precision round-off (1.5e-6 at
  most), in 1.8 s ([GPU model results](wave_results.md)).
- **Initial state.** `hww3-prop` starts from calm conditions as WW3 does
  without a restart file (`--restart calm`, with the first level and ice
  fields applied at the first step), within 1.7 mm of WW3's first step,
  and writes its state as a `FULL` WW3 restart (`--restart-out`); the
  writer gives back WW3's own restarts byte for byte, and a run continued
  from a GPU restart gives the run in one piece to 2.4e-7 m.
- **Forcing fields.** The interpolation of `ww3_prnc` is ported
  (`hww3_wave::prnc`, `hww3-pre prnc`, with `scripts/prnc_nc.py` reading
  the netCDF files). From the netCDF inputs of the January runs, copied
  from Datarmor, the wind, ice, currents and water level of 1 to 10
  January are identical to the bit to the files of `ww3_prnc`.
- **Boundary spectra.** `ww3_bounc` is ported (`hww3_wave::bounc`,
  `hww3-pre bounc`, with `scripts/bounc_nc.py`); from the 318 station
  files the boundary file is identical to the bit to that of `ww3_bounc`.
- **A run with Rust inputs only** (calm start, `build-params`, the files
  above) gives the calibration's start run to a median of 1e-6 m and the
  same cost and scores on both windows: no WW3 tool is needed for a GPU
  run.

### 9.7 Plan after the calibration (23 November 2026)

Order of the work, approved by Jean-Marc: the calibration until "calibration
done", the preprocessing in Rust (section 9.6), the optimisation (profile
first; an f32 tangent checked against the f64 one, a vector tangent,
asynchronous outputs), `BETAMAX`$(U)$ extended by degrees, then M4. M4 does
not start before the "BETAMAX(U) done" checkpoint, and before its adjoints
Jean-Marc chooses between hand-written fused adjoints and Burn tensor
operations for the source terms from a comparison based on the M2 and
optimisation timings.

**Step 4, `BETAMAX` from the wind, with the tangent.** The form is
$\ln\beta = \ln\beta_0 + \sum_k c_k\,\phi_k(\text{features})$ with
smooth bases, a prior centred on the calibrated constant and smoothness
penalties.

1. First $\beta(U)$ in the local ERA5 $U_{10}$ (a smooth basis of about 7
   knots from 0 to 26 m/s, denser between 10 and 20 m/s), from $p_0$ with
   `SDSC2` and `NLPROP`, under the third cost, on the present windows
   before the optimisation (Jean-Marc, 24 November).

   *Implementation* (`hww3-prop --betau KNOTS:COEFS`). Hat functions
   $\phi_k$ on the knots 2, 5, 8, 11, 14, 17 and 20 m/s, constant below the
   first knot and above the last, adding up to 1 (99 % of the wind speeds
   of the two windows are below 15.7 m/s and 99.9 % below 20 m/s; the
   median is 6.6 m/s). $U$ is the wind speed the source terms see at the
   node and step, and $\beta = \beta_0 \exp\sum_k c_k\phi_k(U)$ enters
   `W3SIN4` in place of `BETAMAX`. The tail-stress table (`TAUHFT2`, or
   `TAUHFT` without sheltering) depends on `BETAMAX`: it is computed with its
   derivative by the Rust `TABU_TAUHF2` at 13 values of `BETAMAX` from 0.75
   to 3.75 (`--betau-grid 0.75:0.25:13`) and interpolated in $\beta$ by
   cubic Hermite polynomials, exact at the grid values (1.75 is one). The
   tangent kernel gets $d\beta = \beta\,(d\beta_0/\beta_0 + \sum_k dc_k
   \phi_k(U) + \sum_k c_k \phi_k'(U)\,dU)$ per node, so that the tangent for
   $c_k$ (`--tangent BETAU<k>`) is the field $\beta\phi_k(U)$, and the
   derivative of the tail stress in $\beta$ comes from the interpolation.
   Without `--betau` the kernels take the former path. Checks
   (`scripts/betau_check_hpc.slurm`): with all $c_k = 0$ the run reproduces
   the constant `BETAMAX` 1.75; the tangent for all $c_k$ at once
   (`BETAU`) reproduces the `BETAMAX` tangent; a constant $\beta$ of
   1.37249382 through the interpolation against the tables computed for it;
   the tangent for $c_3$ at $c_3 = 0.2$ against central differences. On
   12 h (Slurm job 4327297): with all $c_k = 0$ the hourly $H_s$ and
   $T_{m02}$ are bitwise those of `BETAMAX` 1.75 (no node differs); the
   tangents `BETAU` and `BETAMAX` with $\beta(U)$ match the `BETAMAX`
   tangent of the constant run to $4 \times 10^{-16}$ (relative L2 of the
   tangent of $H_s$, the tangent of the total energy to $1.3 \times
   10^{-16}$); the interpolation in $\beta$ at 1.37249382 differs from the
   tables computed for it by $1.3 \times 10^{-6}$ m RMS in $H_s$ at the
   last hour (no node above 1 mm; 4.5 mm at most over the 12 h), against
   4.9 cm RMS for the change of step 3. A tangent run with the 13 tables
   takes about 2 min more (building the tables), 72 min for 156 h.

   *Prior.* $c_k \sim \mathcal N(0, 0.25^2)$ as the logarithms of the other
   parameters, plus the second differences of consecutive coefficients,
   $\sum_k ((c_{k-1} - 2c_k + c_{k+1})/0.1)^2$ (`cal_gn.py --smooth 0.1`):
   a constant or linear $\ln\beta(U)$ pays only the first term, and an
   alternation of $\pm 0.05$ between neighbouring knots costs about 5. The
   tangents of `SDSC2` and `NLPROP` at $p_0$ are those of the first step,
   since $\beta(U)$ with all $c_k = 0$ is the constant model.

   *Result* (25 November, with the Jacobian frozen at $p_0$, no cut of the
   tangent):
   - One Gauss-Newton step lowered the cost by 1.79 % on the training
     window and 0.94 % on validation. The next step was predicted to gain
     0.11 %, so the test stopped there.
   - $\beta$ goes 1.76, 1.95, 1.62, 1.80, 1.65, 1.61 and 1.88 at the
     knots, with `SDSC2` ×1.137 and `NLPROP` ×1.209.
   - The standard deviations of $\ln\beta$ are 0.007–0.02 between 5 and
     20 m/s, lower bounds.
   - This is less than the constant step 3 (−3.13 % and −4.12 %), which
     belongs to the same family. The linear model at $p_0$ does not reach
     it, and the alternating curve is a minimum of that linear model, not
     a measured shape of $\beta(U)$.
   - Against step 3, the buoy $H_s$ bias is 1.2–1.5 cm less negative.
     On validation the altimeter bias is 2.6 cm larger (+3.95 against
     +1.36 cm) and its RMSE 0.6 cm larger.
   - At 15–20 m/s of wind the altimeter bias is +0.9 cm, against +14.2 cm
     at $p_0$ and −6.1 cm at step 3.
   - The same Jacobian at the residuals of step 3 predicts a further
     −1.31 % for a $\beta(U)$ step from step 3. Run on Jean-Marc's
     request, it lowers the training cost by 1.20 % against step 3, but
     the validation cost rises by 0.87 %, mostly in the altimeters, so it
     is rejected ([GPU model results](wave_results.md)).
   - On these windows, then, $\beta(U)$ does not beat the constant step 3.
     The month after the optimisation, with a refreshed Jacobian, will be
     the real test.
2. Then one or two features of the wind history, computed from the forcing
   only: $dU/dt$, the turning rate of the wind direction and $|\nabla U|$,
   with a few spline coefficients each, using the vector tangent once the
   optimisation has made it available.
3. On the validation month, for altimeter $H_s$ and buoy $H_s$ and
   $T_{m02}$, compare the constant $\beta$, $\beta(U)$, $\beta(U,$
   history$)$ and a wind correction $g(U)$; a feature is kept only if the
   validation improves.
4. ERA5 gusts, the air–sea temperature difference and the boundary-layer
   height need new data: candidates, fetched only with Jean-Marc's
   agreement.

Features of the wave state (wave age) would enter the adjoint of the
source terms and wait until after M4.

**After M4 (M5, planned, not started).**

1. A regional wind bias: a multiplicative factor on a HEALPix map of nside
   8–16, constant per month, possibly in classes of $U$, constrained by the
   scatterometer winds (ASCAT, to download only with Jean-Marc's agreement)
   together with the altimeters, to separate a wrong wind from a wrong
   growth.
2. $\beta$ from a small neural network of the features above, through the
   adjoint.
3. Coastal parameters (`BJGAM`, bottom friction, a transparency for
   unresolved obstacles) as functions of the grid descriptors of the
   coastal diagnostics ([GPU model results](wave_results.md): distance to
   the coast, land fraction and bathymetric spread inside the cell,
   HEALPix level), not as free fields per region, validated by leaving out
   whole regions of buoys.

**Guardrails for step 4 and M5.** The learned $\beta$ absorbs the errors of
the wind and of the dissipation and is specific to ERA5, as the present
1.75 is; priors and smoothness penalties; physical constraints ($\beta > 0$,
smooth); an independent validation month.

### 9.8 Optimisation (from 25 November 2026)

Jean-Marc started the optimisation on 25 November ("Oui, lancer"). A
factor of 10 over WW3 is too little for him. The rule is that results
stay unchanged to round-off; anything else goes behind an option, off
by default.

**Profile** (Nsight Systems on hpc; Nsight Compute has no access to the
counters there; [GPU model results](wave_results.md)):
- The GPU is busy 97–99.9 % of a run, so the host and the outputs have
  nothing to gain.
- The source kernel takes about 73 % of a value run in f32, 87 % in f64
  and 83 % of a tangent run. The geographic sweeps take most of the rest.
- In the source kernel, each node keeps its spectrum and work arrays in
  shared memory (42 KB in f32, 84 KB in f64, 157 KB in the tangent). An
  SM therefore runs 3, 1 and 1 nodes at once.
- About 80 % of the kernel's time is in scalar or narrow parts that more
  units per node do not shorten. Its speed follows the number of nodes
  per SM.

**Plan, in order** (expected gain on the seconds per model day):
1. Hold the 0/1 wind-sea flags of the source kernel in bytes instead of
   floats (identical to the bit). This gives 4 nodes per SM in f32 and 2 in
   f64: about −15–20 % in f32 and −35–45 % in f64.
2. More shared memory saved in the source kernel by aliasing arrays that
   are not alive at the same time (identical to the bit): +1 node per SM
   in f32 for each 5.8 KB saved.
3. The scalar parts that do not depend on the order of the operations
   (the minimum of the dynamic step, the maxima, the flags) done by all
   units of the cube: a few per cent each.
4. The tangent kernel down to 83 KB (2 nodes per SM): about −40 % on
   the tangent runs.
5. A vector tangent: one value path for several directions. For the nine
   directions of `BETAMAX`$(U)$ it would cost about 2.2 times less than
   nine runs.
6. An f32 tangent, behind an option until it passes the checks of M3 in
   double precision: about 4 times faster.
7. The sums of the scalar parts as parallel reductions. This changes the
   order of the sums (round-off), so it goes behind an option and is
   checked over 48 h and the week. Up to −40 % of the source kernel.
8. Geographic sweeps: the triangle and node kernels fused, without the
   triangle array in global memory. About −5 % in f32.
9. Asynchronous outputs: under 1 %, done last if at all.

**Progress (checkpoint of 25 November;
[GPU model results](wave_results.md)).**
- Step 1 is done and identical to the bit: f32 week 108.9 s per model day
  instead of 124.1, f64 304.5 s instead of 505.1.
- Part of step 3 is done: the powers in the serial loop of `W3SPR4` are
  computed before it by all units, with differences at round-off. The
  value kernel then takes 101.7 s per model day in f32 and 266.4 s in f64.
  The same for the constants of `W3SIN4` is identical to the bit but slows
  the f32 value kernel (117.3 s), so only the tangent kernel has it.
- Step 4 is done: 2 nodes per SM for the tangent, 590 s per model day
  instead of 931 s over the first day (−37 %), at round-off, with the M3
  check of the source tangent unchanged.
- Step 2 brings nothing more. With the flags in bytes, the registers, not
  the shared memory, set the nodes per SM (4 in f32, 2 in f64): the value
  kernel with its sums per unit needs far less shared memory and is not
  faster.
- The costs and scores of the calibration windows are unchanged to 2e-7.

**Progress (second checkpoint, 25 November;
[GPU model results](wave_results.md)).**
- The registers per unit are now read from the PTX and at run time:
  128 for the f32 value kernel (4 nodes per SM), 255 in f64 (2 nodes) and
  168 for the tangent kernel. The constants of `W3SIN4` take the f32
  value kernel to 168 registers and 3 nodes per SM, which is why they
  slow it.
- Square roots replace POW, ATAN2, COS and SIN in the serial step of
  `W3SIN4`, at round-off: f32 week 98.8 s per model day, f64 246.1 s.
  The scores are the same to the printed digits.
- Step 2 (aliasing) is done for the tangent: the DIA and `W3SDS4` work in
  the buffers of the spectrum, 53 KB per node and 3 nodes per SM. The
  tangent takes 509.9 s per model day over the first day, identical to
  the bit (565.8 s before, 931.4 s for 35ecb15).
- Step 6 fails: the f32 tangent, once a 0/0 of `W3SDB1` in f32 is fixed,
  takes 302 s per model day but grows wrong over the calibration window.
  $\partial J / \partial \ln$ `BETAMAX` is off by 6.9 % and the
  altimeter column of the Jacobian by 26 %. It is not used.
- `W3SIN4` is about 27 % of the f64 value source kernel, `W3SDS4` 8 % and
  the DIA 6 %.

**Optimisation done (26 November;
[GPU model results](wave_results.md)).** In seconds per model day on one
A100, final build against 35ecb15:

| Run | 35ecb15 | Final build |
|---|---|---|
| f32 week | 124.1 | 98.3 |
| f64, 48 h | 505.1 | 245.3 |
| f64 tangent, first day | 931.4 | 508.5 |

Against WW3 on the same hours of grid v5 $K=4$ with the full ST4:
- 64 cores of hpc: 717.1 s; the GPU in f32 is 7.3 times faster.
- 32 ranks of hpc: 1,491 s; the GPU is 14.7 times faster in f32 and 6.9
  in f64.
- 112 cores of Datarmor (other days): 2,601 s explicit and 1,058 s
  implicit, against which the f32 week is 26 and 10.8 times faster.

How the plan ended:
- Step 8 is not kept: the geographic sub-step as one kernel is slower
  (+16 % in f32, +38 % in f64).
- Step 7 is an option, `--par-red`, off by default: −6.8 % in f64, nothing
  in f32, round-off.
- Step 5, the vector tangent, was dropped for M4 ("Oui, passer à M4"). It
  would save at best 1.3 times for two directions and 1.75 times for
  nine, and only if the nodes per SM stayed as they are, which the shared
  memory does not allow.
- Step 9, asynchronous outputs, is skipped: under 1 %.

## 10. Other routes, briefly

| route | effort | why not the main route |
|---|---|---|
| Tapenade or Enzyme on the Fortran | 6–12 person-months for a specialist | pointer aliasing of the WW3 data modules, MPI halo exchanges, Enzyme needs LLVM Flang (assumed-shape arrays not supported); the adjoint runs on CPU at 3–10 forward runs per gradient (12–40 h on 112 cores for 7 days on grid v4) |
| the same twin in PyTorch or JAX | about 20–30 % less than the Rust route, mostly in the adjoints, which the framework provides | slower kernels and more memory for the same model; the Rust route keeps one language with `hww3-grid` and full control of the kernels |
| neural emulator of WW3 | 2–4 person-months plus many training runs | Jacobians not trustworthy for forcing corrections; useful as a component (for example an $S_{nl}$ network) |
| finite differences and ensembles with the Fortran | days | at most about 20 scalars, 263 core-hours per 8-day run on grid v5 $K=4$; still worth running now, as reference gradients |

**Implicit scheme.** If the implicit scheme were kept: with
`JGS_SOURCE_NONLINEAR = F`, each step solves a linear system
$A(N^n, F, \theta)\,N^{n+1} = b(N^n, F, \theta)$. The sources are
linearised once per step in `CALCARRAY_JACOBI_SOURCE_2`. Its adjoint is one
transposed solve per step, by the implicit function theorem (Christianson
1994; Giles and Pierce 2000), not the derivative of the block Gauss–Seidel
iterations. On a GPU, though, the explicit scheme costs fewer sweeps: 3.2
full-spectrum sweeps per step on grid v4.

## 11. Prior work

The literature search of November 2026 found:

- **No WAVEWATCH III adjoint.** Only a tangent-linear model exists, at the
  Naval Research Laboratory (Orzech et al. 2025).
- **No end-to-end differentiable phase-averaged wave model**, and no hybrid
  wave model trained through its solver.
- **No complete GPU port of WW3.**

Earlier work worth knowing:

- **Wave-model adjoints.**
  - Hersbach (1998) generated the adjoint of the full WAM from its Fortran
    and estimated source-term parameters (weaker wind input and
    whitecapping).
  - de las Heras et al. (1995) and Bauer et al. (1996) corrected the wind.
  - SWAN adjoints use the boundary spectrum or the wind as controls
    (Veeramony et al. 2010; Orzech et al. 2013, 2014; Muscarella
    et al. 2021).
  - Panteleev et al. (2015) did adjoint-free 4D-Var.
  - WW3 parameters have been calibrated without adjoints (Timmermans 2015;
    Goeury et al. 2026; Li et al. 2026). Green's functions calibrate ocean
    models in the same spirit (Menemenlis et al. 2005).
- **Algorithmic differentiation.**
  - TAF and Tapenade adjoints of MITgcm (Giering and Kaminski 1998;
    Heimbach et al. 2005; Hascoët and Pascual 2013; Gaikwad et al. 2025).
  - NEMOTAM (Vidard et al. 2015).
  - Enzyme (Moses and Churavy 2020; Moses et al. 2021, 2022).
  - Differentiable Earth-system models in Julia (Moses et al. 2026) and JAX
    (Häfner et al. 2021; Kochkov et al. 2024).
  - Revolve (Griewank and Walther 2000).
  - Implicit differentiation (Christianson 1994; Giles and Pierce 2000;
    Blondel et al. 2022).
- **GPU wave models.**
  - The OpenACC port of the WW3 source terms was limited by CPU–GPU
    transfers (Ikuyajolu et al. 2023).
  - WAM6-GPU runs 7 days of a global 1/10° model in 7.6 min on eight A100,
    37 times a 32-core node, in single precision (Yuan et al. 2024).
- **Machine learning.**
  - $S_{nl}$ networks: Krasnopolsky et al. 2002; Tolman et al. 2005;
    Puscasu 2014; Chen et al. 2025; NLML, which runs inside WW3 through
    FTorch (Ikuyajolu et al. 2026).
  - Data-driven forecasts (James et al. 2018; Zhang et al. 2025; Wang
    et al. 2025) and output bias correction (Sun et al. 2022).
  - Adjoints of networks (Hatfield et al. 2021).
- **HEALPix networks.** Perraudin et al. 2019; Krachmalnicoff and Tomasi
  2019; Karlbauer et al. 2024; Cresswell-Clay et al. 2025; Delouis
  et al. 2022; FOSCAT, whose only active backend is PyTorch.

### References

- Bauer, E., Hasselmann, K., Young, I. R., Hasselmann, S. (1996). Assimilation of wave data into the wave model WAM using an impulse response function method. *J. Geophys. Res.* 101(C2), 3801–3816. doi:10.1029/95JC03306
- Blondel, M., et al. (2022). Efficient and modular implicit differentiation. *NeurIPS* 35. arXiv:2105.15183
- Burn, deep-learning framework in Rust, version 0.22 (custom CubeCL kernels with hand-written backward; gradient checkpointing). https://burn.dev
- Chen, J., Adcock, T. A. A., Liu, Q., Clark, R., Tang, T. (2025). Machine learning approximation for fast and accurate prediction of nonlinear four-wave interactions in spectral wave models. *JGR Mach. Learn. Comput.* 3(1), e2025JH000864. doi:10.1029/2025JH000864
- Christianson, B. (1994). Reverse accumulation and attractive fixed points. *Optim. Methods Softw.* 3(4), 311–326. doi:10.1080/10556789408805572
- Cresswell-Clay, N., et al. (2025). DLESyM. *AGU Advances* 6, e2025AV001706. doi:10.1029/2025AV001706
- CubeCL, GPU kernels in Rust. https://github.com/tracel-ai/cubecl
- CUDA Rust (`cuda-oxide`, `cutile-rs`), NVIDIA, September 2026. https://developer.nvidia.com/cuda/rust
- de las Heras, M. M., Burgers, G., Janssen, P. A. E. M. (1995). Wave data assimilation in the WAM wave model. *J. Mar. Syst.* 6(1–2), 77–85. doi:10.1016/0924-7963(94)00019-8
- Delouis, J.-M., Allys, E., Gauvrit, E., Boulanger, F. (2022). Non-Gaussian modelling and statistical denoising of Planck dust polarisation full-sky maps using scattering transforms. *A&A* 668, A122. doi:10.1051/0004-6361/202244566
- FOSCAT. https://github.com/jmdelouis/FOSCAT
- Gaikwad, S. S., et al. (2025). MITgcm-AD v2: open source tangent linear and adjoint modeling framework for the oceans and atmosphere enabled by the AD tool Tapenade. *Future Gener. Comput. Syst.* 163, 107512. doi:10.1016/j.future.2024.107512
- Giering, R., Kaminski, T. (1998). Recipes for adjoint code construction. *ACM Trans. Math. Softw.* 24(4), 437–474. doi:10.1145/293686.293695
- Giles, M. B., Pierce, N. A. (2000). An introduction to the adjoint approach to design. *Flow Turbul. Combust.* 65, 393–415. doi:10.1023/A:1011430410075
- Goeury, C., Fouquet, T., Teles, M., Benoit, M. (2026). Bayesian optimization for reanalysis and calibration of highly energetic sea state events simulated with a spectral third-generation wave model. *Appl. Ocean Res.* 173, 105144. doi:10.1016/j.apor.2026.105144
- Griewank, A., Walther, A. (2000). Algorithm 799: revolve. *ACM Trans. Math. Softw.* 26(1), 19–45. doi:10.1145/347837.347846
- Häfner, D., Nuterman, R., Jochum, M. (2021). Fast, cheap, and turbulent—global ocean modeling with GPU acceleration in Python. *JAMES* 13. doi:10.1029/2021MS002717
- Hascoët, L., Pascual, V. (2013). The Tapenade automatic differentiation tool: principles, model, and specification. *ACM Trans. Math. Softw.* 39(3). doi:10.1145/2450153.2450158
- Hatfield, S., Chantry, M., Dueben, P., Lopez, P., Geer, A., Palmer, T. (2021). Building tangent-linear and adjoint models for data assimilation with neural networks. *JAMES* 13, e2021MS002521. doi:10.1029/2021MS002521
- Heimbach, P., Hill, C., Giering, R. (2005). An efficient exact adjoint of the parallel MIT general circulation model, generated via automatic differentiation. *Future Gener. Comput. Syst.* 21(8), 1356–1371. doi:10.1016/j.future.2004.11.010
- Hersbach, H. (1998). Application of the adjoint of the WAM model to inverse wave modeling. *J. Geophys. Res.* 103(C5), 10469–10487. doi:10.1029/97JC03554
- Ikuyajolu, O. J., Van Roekel, L., Brus, S. R., Thomas, E. E., Deng, Y., Sreepathi, S. (2023). Porting the WAVEWATCH III (v6.07) wave action source terms to GPU. *Geosci. Model Dev.* 16, 1445–1458. doi:10.5194/gmd-16-1445-2023
- Ikuyajolu, O. J., Van Roekel, L., Brus, S. R., Thomas, E. E. (2026). NLML: a deep neural network emulator for the exact nonlinear interactions in a wind wave model. *JGR Mach. Learn. Comput.* 3(1), e2025JH000699. doi:10.1029/2025JH000699
- James, S. C., Zhang, Y., O'Donncha, F. (2018). A machine learning framework to forecast wave conditions. *Coast. Eng.* 137, 1–10. doi:10.1016/j.coastaleng.2018.03.004
- Karlbauer, M., et al. (2024). Advancing parsimonious deep learning weather prediction using the HEALPix mesh. *JAMES* 16, e2023MS004021. doi:10.1029/2023MS004021
- Kochkov, D., et al. (2024). Neural general circulation models for weather and climate. *Nature* 632, 1060–1066. doi:10.1038/s41586-024-07744-y
- Krachmalnicoff, N., Tomasi, M. (2019). Convolutional neural networks on the HEALPix sphere. *A&A* 628, A129. doi:10.1051/0004-6361/201935211
- Krasnopolsky, V. M., Chalikov, D. V., Tolman, H. L. (2002). A neural network technique to improve computational efficiency of numerical oceanic models. *Ocean Model.* 4, 363–383. doi:10.1016/S1463-5003(02)00010-0
- Li, Z., et al. (2026). A generalized framework for multi-parameter optimization of numerical wind–wave model. *EGUsphere* preprint. doi:10.5194/egusphere-2026-895
- Menemenlis, D., Fukumori, I., Lee, T. (2005). Using Green's functions to calibrate an ocean general circulation model. *Mon. Weather Rev.* 133, 1224–1240. doi:10.1175/MWR2912.1
- Moses, W. S., Churavy, V. (2020). Instead of rewriting foreign code for machine learning, automatically synthesize fast gradients. *NeurIPS* 33. arXiv:2010.01709
- Moses, W. S., et al. (2021). Reverse-mode automatic differentiation and optimization of GPU kernels via Enzyme. *SC'21*. doi:10.1145/3458817.3476165
- Moses, W. S., et al. (2022). Scalable automatic differentiation of multiple parallel paradigms through compiler augmentation. *SC22*. doi:10.1109/SC41404.2022.00065
- Moses, W. S., et al. (2026). DJ4Earth: differentiable, and performance-portable Earth system modeling via program transformations. *JAMES* 18, e2025MS005615. doi:10.1029/2025MS005615
- Muscarella, P., Brunner, K., Walker, D. (2021). Estimating coastal winds by assimilating high-frequency radar spectrum data in SWAN. *Sensors* 21, 7811. doi:10.3390/s21237811
- Orzech, M. D., Veeramony, J., Ngodock, H. (2013). A variational assimilation system for nearshore wave modeling. *J. Atmos. Oceanic Technol.* 30, 953–970. doi:10.1175/JTECH-D-12-00097.1
- Orzech, M., Veeramony, J., Flampouris, S. (2014). Optimizing spectral wave estimates with adjoint-based sensitivity maps. *Ocean Dyn.* 64, 487–505. doi:10.1007/s10236-014-0700-2
- Orzech, M., et al. (2025). Development and testing of tangent linear model for WAVEWATCH III. NRL Memorandum Report NRL/7320/MR--2025/2.
- Panteleev, G., Yaremchuk, M., Rogers, W. E. (2015). Adjoint-free variational data assimilation into a regional wave model. *J. Atmos. Oceanic Technol.* 32, 1386–1399. doi:10.1175/JTECH-D-14-00174.1
- Perraudin, N., Defferrard, M., Kacprzak, T., Sgier, R. (2019). DeepSphere. *Astron. Comput.* 27, 130–146. doi:10.1016/j.ascom.2019.03.004
- Puscasu, R. M. (2014). Integration of artificial neural networks into operational ocean wave prediction models for fast and accurate emulation of exact nonlinear interactions. *Procedia Comput. Sci.* 29, 1156–1170. doi:10.1016/j.procs.2014.05.104
- Rust automatic differentiation (`std::autodiff`), rustc development guide. https://rustc-dev-guide.rust-lang.org/autodiff/installation.html
- Sun, D., et al. (2022). A deep learning-based bias correction method for predicting ocean surface waves in the Northwest Pacific Ocean. *Geophys. Res. Lett.* 49, e2022GL100916. doi:10.1029/2022GL100916
- Timmermans, B. (2015). *Uncertainty in numerical wind-wave models.* PhD thesis, University of Southampton.
- Tolman, H. L., Krasnopolsky, V. M., Chalikov, D. V. (2005). Neural network approximations for nonlinear interactions in wind wave spectra: direct mapping for wind seas in deep water. *Ocean Model.* 8, 253–278. doi:10.1016/j.ocemod.2003.12.008
- Veeramony, J., Walker, D., Hsu, L. (2010). A variational data assimilation system for nearshore applications of SWAN. *Ocean Model.* 35, 206–214. doi:10.1016/j.ocemod.2010.07.008
- Vidard, A., Bouttier, P.-A., Vigilant, F. (2015). NEMOTAM: tangent and adjoint models for the ocean modelling platform NEMO. *Geosci. Model Dev.* 8, 1245–1257. doi:10.5194/gmd-8-1245-2015
- Wang, X., et al. (2025). Data-driven rolling model for global wave height. *Geosci. Model Dev.* 18, 5101–5114. doi:10.5194/gmd-18-5101-2025
- Yuan, Y., et al. (2024). Towards a real-time modeling of global ocean waves by the fully GPU-accelerated spectral wave model WAM6-GPU v1.0. *Geosci. Model Dev.* 17, 6123–6136. doi:10.5194/gmd-17-6123-2024
- Zhang, Z., et al. (2025). Ocean wave forecasting with deep learning as alternative to conventional models. *JAMES* 17, e2025MS005285. doi:10.1029/2025MS005285

## Risks

| risk | mitigation |
|---|---|
| the Rust port departs from WW3 (splitting order, boundary flags, coastal nodes, source integration) | the propagation-only test before the sources; kernel-by-kernel harness; comparison with the existing explicit runs |
| GPU races and single-precision accumulation in the N-scheme scatter | node-centred gather formulation (no atomics) or graph colouring; double-precision build for the checks |
| Rust GPU tools still young (CubeCL, Burn; `cuda-oxide` experimental; GPU Enzyme not available) | M0 passed with CubeCL and Burn (section 9.1); the same kernels in CUDA C through `cudarc` stay as a cross-check |
| CubeCL and Burn do not build on Datarmor (glibc 2.19, A100 driver limited to CUDA 11.4) | development on the Ifremer hpc cluster (A100, glibc 2.34, CUDA 13.0); the CUDA C kernels run on the Datarmor A100s |
| hand-written adjoints of ST4 are long and error-prone | the simplified ST4 first; forward-mode tangents first, checked against double-precision finite differences; CPU `std::autodiff` derivatives of the same per-node functions if the nightly toolchain is installed later; or Burn tensor operations for the source terms |
| parameters and forcing compensate each other (`BETAMAX` was raised to make up for ERA5 winds that are too weak in storms) | calibrate the parameters with the forcing fixed, and the reverse, before any joint fit; $T_{m02}$ and wave-age-dependent errors; multi-scale priors |
| memory for reverse mode on grid v4 | grid v5 $K=4$ for learning, host-memory snapshots, two GPUs (at most two per user) |
| over-fitting January 2026 | train and test on separate halves, then other months (downloads to be agreed) |

## Decisions (21 November 2026)

1. **Kernel language.** CubeCL for the kernels, with Burn for tensors, tape
   and checkpointing. `cudarc` with CUDA C kernels remains the fallback if
   M0 fails. M0 passed on 22 November 2026 (section 9.1): CubeCL with
   Burn, developed on the Ifremer hpc cluster because CubeCL does not
   build on Datarmor. The CUDA C kernels stay as a cross-check on
   Datarmor.
2. **Installs.** The CUDA crates only: Burn, CubeCL and their CUDA backend,
   with the stable Rust toolchain, on the Datarmor and hpc scratch spaces.
   On the hpc cluster the CubeCL build also downloads its prebuilt LLVM,
   as agreed. No Rust nightly with the Enzyme component for now, so the
   reference derivatives come from finite differences in double precision.
3. **Fidelity to ST4.** A simplified ST4 first (M2a: `SINTABLE = 0`,
   `SDSCUM = 0`, which WW3 also runs), then the full ST4 (M2b). The aim is a
   complete WW3.
4. **First target.** M2, with finite-difference gradients, before any
   adjoint work.
