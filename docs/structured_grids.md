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

`scripts/struct_run_hpc.slurm` runs these on one A100 of the hpc cluster.

## Physics of the IRI configuration left out

The IRI configuration (`IRIGLOB-7M`, `ARC-15KM`, `ANTARC-15KM` under
`ww3_multi`, 140 days from 12 December 2018) uses terms and options that
`hww3-wave` does not have. The GPU runs leave them out:

| WW3 | IRI setting | GPU model |
|---|---|---|
| propagation | `PR3` (`UQ`, third order, with GSE correction `WDTHCG = WDTHTH = 1.5`) | first-order N scheme on triangles (`PDLIB_W3XYPFSN2`) |
| whitecapping | `SDSBCHOICE = 3` with the isotropic cumulative term `SDSCUM = 0.3` | `SDSBCHOICE = 1`, `SDSCUM = 0` (the HWW3 variant of ST4); the rest of `&SDS4` and `&SIN4` kept (`BETAMAX 1.57`, `SDSC2 -3.8`, ...) |
| ice | `IC2` (under-ice boundary layer) and `IS2` (ice scattering and break-up), ice from ERA5 | not ported: ice only as WW3 masks it without these terms |
| reflection | `REF1` (coast and iceberg reflection) | not ported |
| obstructions | `OBST%FILENAME` (sub-grid blocking) | not used by the triangle scheme |
| grid coupling | three grids exchanging spectra in `ww3_multi` | each grid alone; the active-boundary points of the polar grids keep their spectra |
| time steps | `DTMAX 2520`, `DTXY 420`, `DTKTH 210` | global step 600 s, `DTKTH 300`, geographic sub-steps from the CFL of each component (`ITER`) |

The spectral grid is the configuration's (`NK = 36`, `NTH = 36`, `FREQ1 =
0.0339`, `XFR = 1.1`), with the source parameters of its namelists. The
option `--par-red` of the source kernel needs `NK <= 32` and is off.

The WW3 reference took 137,953 s of wall time on 252 MPI processes of
Datarmor for the 140 days (`log.mww3`), 985 s per model day for the three
grids together (2,077,465 sea points).
