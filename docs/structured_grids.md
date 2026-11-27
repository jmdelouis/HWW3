# WW3 structured grids on the GPU model

`hww3-prop` runs the explicit N scheme of WW3's unstructured grids
(`PDLIB_W3XYPFSN2`) on triangle meshes. The module `structgrid` of
`hww3-wave` and the command `hww3-pre struct` turn a WW3 structured grid,
regular (`RECT`) or curvilinear (`CURV`), into such a mesh, so that the
GPU model runs the grids of an existing WW3 configuration and its speed
can be compared with WW3 on the same grids. The scheme stays the
first-order N scheme: results are not those of WW3's structured schemes
(third-order `UQ` with the garden-sprinkler correction in the IRI
configuration), only close to them.

## Reading the grid

`StructGrid::read` reads `ww3_grid.nml` (namelist version) and the files
it names, as `ww3_grid` does:

| Item | Rule |
|---|---|
| `RECT` coordinates | `X0 + ix SX`, `Y0 + iy SY` (degrees); `SX`, `SY` divided by `RECT%SF`, `X0`, `Y0` by `RECT%SF0` |
| `CURV` coordinates | `CURV%XCOORD%FILENAME`, `CURV%YCOORD%FILENAME`, times `SF`, plus `OFF` |
| bottom level | `DEPTH%FILENAME` times `DEPTH%SF` (negative at sea) |
| status map | `MASK%FILENAME`: 0 land, 1 sea, 2 active boundary, 3 excluded; without a mask, sea where `ZB <= ZLIM` |
| grid edges | sea points of the first and last rows excluded, and of the first and last columns unless `CLOS = 'SMPL'` |
| file layout | free format (`IDFM = 1`) or Fortran unformatted records (`IDFM = 3`, either byte order); rows from the south (`IDLA` 1, 2) or the north (`IDLA` 3, 4) |

The counts match `ww3_grid` on the three grids of the IRI configuration
(`ww3_grid.<name>.out`):

| Grid | Type | Points | Sea | Active boundary | Excluded | Land |
|---|---|---|---|---|---|---|
| IRIGLOB-7M | `RECT` 2880 × 562, 0.125°, `SMPL` | 1,618,560 | 1,189,867 | 0 | 4,304 | 424,389 |
| ARC-15KM | `CURV` 958 × 958 | 917,764 | 358,184 | 1,455 | 209,657 | 349,923 |
| ANTARC-15KM | `CURV` 888 × 888 | 788,544 | 529,414 | 2,172 | 181,612 | 77,518 |

The 4,304 excluded points of IRIGLOB-7M are the sea points of its two edge
rows (2,658 + 1,646), excluded by `ww3_grid` although the mask file marks
them as sea.

## Triangulation

The sea points (`MAPSTA` 1 or 2) become the nodes, in WW3's order (`ISEA`:
`ix` fastest, then `iy`). Each cell whose four corners are sea gives two
triangles cut along its shorter diagonal (on the sphere); a cell with
three sea corners gives one triangle, which follows the coast more closely
than the staircase of the structured grid. With `CLOS = 'SMPL'` the last
column is joined to the first: the geometry (`geom`) takes longitude
differences across the date line when shorter (`wrap_lon`). Sea points in
no triangle (single points between land, connected to the sea only by a
corner) are dropped: 33 on IRIGLOB-7M, 84 on ARC-15KM, 10 on ANTARC-15KM.
The active-boundary points become the open boundary of the mesh (`IOBP =
2`): they keep their spectra unless a boundary file feeds them.

A grid around a pole cannot be meshed in longitude and latitude. With
`--rotate auto` (the default) a grid with a sea point within 10° of a
pole is meshed in rotated coordinates in which that pole lies on the
equator at the rotated origin, the rotated north pole at (0° N, 180° E)
for the north (`north`) and at (0° N, 0° E) for the south (`south`). The
sphere is unchanged, so the scheme, the great-circle turning and the
sub-cycling are exact in the rotated frame; only the spectral directions
are measured from the rotated north, as on WW3's rotated-pole grids. The
angle from the rotated to the geographic east at each node is written
with the mesh (`raw/angle.f64`): vector forcing (wind, currents) must be
turned by it before it reaches the model (`structgrid::rotate_vector`;
`hww3-pre ideal-wind` does it), and directional outputs turned back.
ARC-15KM is meshed this way (rotated latitudes −58.5° to 59.8°);
ANTARC-15KM has no sea point south of 78.4° S and stays geographic.

## Files

`hww3-pre struct` writes an HWW3 grid directory that `hww3-prop` reads
unchanged: `raw/{lon,lat,depth}.f64`, `raw/tri.u32`, `raw/obc.u32` and the
namelist as `ww3_grid.nml` (spectral grid and `DMIN`), plus
`raw/ij.u32` (index `iy NX + ix` of each node in the structured grid, to
put the outputs back on it), `raw/lon_geo.f64`, `raw/lat_geo.f64` and
`raw/angle.f64`.

```text
hww3-pre struct WW3_GRID.nml --data DIR --out GRID_DIR [--rotate auto|none|north|south]
hww3-pre ideal-wind GRID_DIR --out wind.ww3 --start YYYYMMDDHHMMSS --hours H [--speed U]
hww3-src build-params GRID_DIR NAMELISTS.nml OUT.params
hww3-prop run GRID_DIR --restart calm --start T --wind wind.ww3 --sources OUT.params --hours H --out DIR
```

`scripts/struct_run_hpc.slurm` runs these on one A100 of the hpc cluster; the whole recipe, from the copy of the inputs to the comparison with WW3, is in [Run a WW3 structured-grid configuration](tutorials/run_structured_grid.md).

## Physics of the IRI configuration left out

The IRI configuration (`IRIGLOB-7M`, `ARC-15KM`, `ANTARC-15KM` under
`ww3_multi`, 140 days from 12 December 2018) uses terms and options that
`hww3-wave` does not have. The GPU runs leave them out:

| WW3 | IRI setting | GPU model |
|---|---|---|
| propagation | `PR3` (`UQ`, third order, with GSE correction `WDTHCG = WDTHTH = 1.5`) | first-order N scheme on triangles (`PDLIB_W3XYPFSN2`) |
| whitecapping | `SDSBCHOICE = 3` with the isotropic cumulative term `SDSCUM = 0.3` | ported since (see below); the runs above used `SDSBCHOICE = 1`, `SDSCUM = 0` (the HWW3 variant of ST4), the rest of `&SDS4` and `&SIN4` kept (`BETAMAX 1.57`, `SDSC2 -3.8`, ...) |
| ice | `IC2` (under-ice boundary layer) and `IS2` (ice scattering and break-up), ice from ERA5 | not ported: ice only as WW3 masks it without these terms |
| reflection | `REF1` (coast and iceberg reflection) | not ported |
| obstructions | `OBST%FILENAME` (sub-grid blocking) | not used by the triangle scheme |
| grid coupling | three grids exchanging spectra in `ww3_multi` | each grid alone; the active-boundary points of the polar grids keep their spectra |
| time steps | `DTMAX 2520`, `DTXY 420`, `DTKTH 210` | global step 600 s, `DTKTH 300`, geographic sub-steps from the CFL of each component (`ITER`) |

The spectral grid is the configuration's (`NK = 36`, `NTH = 36`, `FREQ1 =
0.0339`, `XFR = 1.1`), with the source parameters of its namelists. The
option `--par-red` of the source kernel needs `NK <= 32` and is off.

## Results on IRIGLOB-7M (one A100, single precision)

Runs of `scripts/struct_run_hpc.slurm` and `scripts/struct_probe_hpc.slurm`
on an A100-SXM4-80GB of hpc, with the wind and currents that WW3 used on
the grid (`FIELD_NC/WW3-IRIGLOB-7M_201901.nc`, 1-8 January 2019, written
on the nodes by `scripts/struct_fieldnc.py`), from calm conditions on
1 January 00 h:

| Run | Job | Hours | s per model day |
|---|---|---|---|
| idealised wind, from calm | 4509420 | 24 | 4,153 |
| FIELD_NC wind and currents, from calm | 4514626 | 72 | 3,500 |

The cost of the 72 h run stays at 840-900 s per 6 h while the mean H_s
grows from 0.97 to 2.16 m: it does not come from the calm start. More than
99 % of it is the source kernel. The source integration of `W3SRCE` takes
dynamic sub-steps limited by the largest allowed change of the spectrum,
with at least `DTMIN = 10` s: probes of 1 h from the restart of 4 January
00 h (mean raw step `dtdyn` of the source kernel at every 1000th node;
sub-steps per 600 s step `min(60, 600 / max(dtdyn, DTMIN))`) give

| Grid, namelist | Job | Sub-steps per step (mean, median, p90) | s per model day (1 h) |
|---|---|---|---|
| IRIGLOB-7M, IRI namelist (`FXFM3 20`, `SDSBR 0.005`) | 4530985 | 36.6, 60, 60 (51 % of the nodes at `DTMIN`) | 3,492 |
| IRIGLOB-7M, `SDSBR 0.0009` | 4533686 | 25.7, 18.1, 60 | 2,342 |
| IRIGLOB-7M, `FXFM3 2.5` | 4532957 | 15.3, 11.8, 36.9 | 1,561 |
| IRIGLOB-7M, `FXFM3 2.5` and `SDSBR 0.0009` | 4533687 | 10.6, 10.9, 17.8 | 1,196 |
| IRIGLOB-7M, HWW3 namelist (`BETAMAX 1.75`, `SINTABLE 0`, ST4 defaults) | 4532958 | 1.5, 1.0, 2.9 | 343 |
| v5 K=4 (NK 32), HWW3 namelist | 4530985 | 1.7, 1.0, 2.4 | 263 (101.5 over 48 h) |

So the cost is set by the source parameters of the configuration, not
by the grid or by `NK = 36`: the prognostic range up to 20 times the mean
frequency (`FXFM3 = 20`) with frequencies up to 0.96 Hz, and the
saturation threshold `SDSBR = 0.005`, make the limiter take 20 times more
sub-steps than in the HWW3 runs. WW3 integrates the sources with the same
limiter. Per node and sub-step the kernel costs about 0.56 µs on
IRIGLOB-7M (NK 36) and 0.7 µs on v5 K=4 (NK 32); with the HWW3 namelist
the cost per node is about 1.2 times that of v5 K=4 (36 frequencies
instead of 32, 3 nodes per SM instead of 4 in the source kernel). The
replacement of `SDSBCHOICE = 3` and `SDSCUM = 0.3` changes the
dissipation and so possibly the number of sub-steps WW3 takes.

## The polar grids

ARC-15KM (meshed in rotated coordinates) and ANTARC-15KM run 12 h from
calm on 1 January 2019 with the ERA5 wind and ice and the CMEMS currents of
the same week (interpolated by `hww3-pre prnc`, the vectors of ARC-15KM
turned into the rotated frame), with the IRI namelist as on IRIGLOB-7M.
Their active-boundary points keep calm spectra (no coupling between the
grids):

| Grid | Nodes | Job | s per model day | ms per node and model day |
|---|---|---|---|---|
| IRIGLOB-7M | 1,189,834 | 4514626 | 3,500 | 2.94 |
| ARC-15KM | 358,100 | 4537364 | 1,127 | 3.15 |
| ANTARC-15KM | 529,404 | 4537365 | 1,805 | 3.41 |

The three grids one after the other take 6,432 s per model day on one
A100 with the IRI source parameters, against 985 s per model day for
WW3 on 252 processes (about 69 core-hours per model day, the A100 1.8
hours); with the source parameters of the HWW3 runs IRIGLOB-7M alone takes
343 s per model day.

At NK 36 the source kernel holds about 43,150 B per node in single
precision (3 nodes per SM, against 4 at NK 32) and about 85,000 B in
double precision (1 node per SM, against 2).

H_s against `FIELD_NC` (`scripts/struct_compare.py`; WW3 started 20 days
earlier, so the GPU run lacks the swell at first):

| Time | Mean H_s GPU / WW3 (m) | Bias (m) | RMSE (m) | Correlation |
|---|---|---|---|---|
| 1 Jan 12 h | 1.30 / 2.12 | -0.82 | 1.06 | 0.70 |
| 2 Jan 12 h | 1.81 / 2.12 | -0.31 | 0.67 | 0.82 |
| 3 Jan 12 h | 2.09 / 2.13 | -0.05 | 0.49 | 0.88 |
| 4 Jan 00 h | 2.16 / 2.13 | +0.03 | 0.48 | 0.89 |

The WW3 reference took 137,953 s of wall time on 252 MPI processes of
Datarmor for the 140 days (`log.mww3`), 985 s per model day for the three
grids together (2,077,465 sea points).

## Romero's whitecapping (`SDSBCHOICE = 3`)

`W3SDS4` with `SDSBCHOICE = 3` (Romero 2019) and the isotropic cumulative
term (`SDSCUM > 0`), as the IRI configuration sets them, is ported to the
CPU reference (`sources::sds4`), the value kernel (`sds4_romero` in
`srce`) and the tangent kernel (`sds4_romero_tl` in `srce_tl`): the
breaking-crest density from the directional saturation, modulated by the
cumulative mean square slope in the direction of the long waves
(`SDSFACMTF`, `SDSNMTF`, `SPMSS`, with `DLWMEAN` of `W3SPR4`), the
breaking strength from `SDSBT`, the pseudo-diagonal, and the isotropic
cumulative rate summed over the longer waves (`DIKCUMUL = NINT(SDSBRF1 /
(XFR - 1))`, set by `W3SDS4` itself). The kernels choose the branch at
compile time from the parameters (`SourceKernel::romero`), so the kernels
of `SDSBCHOICE = 1` are generated as before.

Checks (`scripts/st4_romero_ref_hpc.slurm`, `scripts/st4_romero_gpu_hpc.slurm`):
a `mod_def.ww3` of grid v5 K=4 with the `&SIN4`, `&SDS4` and `&SNL1` of the
IRI namelists, and the source-term harness of M2 on 1,864 nodes of the
restart of 8 January 2026.

| Check | Result |
|---|---|
| CPU against WW3, `W3SDS4` | S max 8.6e-6, RMS 3.9e-7 (relative, per node) |
| CPU against WW3, whole `W3SRCE` | spectrum max 4.7e-6; same dynamic sub-steps (1 to 60, mean 3.72), `LLWS`, `LBREAK`, `SHAVE` identical |
| GPU against CPU and WW3, f32 and f64 | spectrum max 7.5e-5 at one node (ISEA 9401), mean 3.3e-7; H_s within 8.8e-5 m |
| tangent against central differences (M3 check, h 1e-4) | no smooth node above 1e-4 in any direction (largest 9.9e-6) |
| the default path (`SDSBCHOICE = 1`) | 2 h of v5 K=4 identical to the bit to main in f32 and f64; the M3 check prints the same numbers |

The largest GPU difference is at a node where the source integration takes
the `DTMIN` floor: with the same `&SIN4` and `SDSBCHOICE = 1`, the
established kernel differs there by 1.6e-3 from the CPU, and the CPU by
2.8e-3 from WW3. The tangent is not defined where the breaking strength
vanishes (`BTOVER = 0`, where Lambda jumps to zero, so that the cumulative
rate of the shorter waves jumps too) and at the floor of the saturation
(`1e-14`); it takes the one-sided derivative there.
