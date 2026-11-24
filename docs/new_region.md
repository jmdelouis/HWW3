# A grid for another region

This page explains how to build a grid for a region other than the North
Atlantic, and how to make a grid finer, either everywhere or only near the
coast. The keys are described in [Configuration](configuration.md) and the
size functions in [Algorithm](algorithm.md); here they are put in order,
with the numbers needed to choose them.

## 1. The four steps

```bash
# 1. cut GEBCO to the region (a margin of 3° is kept around the box)
python scripts/prep_gebco.py data/gebco/GEBCO_2024.nc data/gebco/biscay -10 0 43 49 --margin 3

# 2. write configs/biscay.json (see below), then build the generator once
cargo build --release --manifest-path hww3-grid/Cargo.toml

# 3. build the grid (RAYON_NUM_THREADS sets the number of threads)
RAYON_NUM_THREADS=8 hww3-grid/target/release/hww3-grid configs/biscay.json > grids/biscay_summary.json

# 4. optional: cell vertices and figures
hww3-grid/target/release/hww3-grid cells grids/biscay
```

The paths in the configuration (`bathy`, `output_dir`) are relative to the
directory of the configuration file. Step 3 writes the mesh, the
open-boundary file and the WW3 namelists into `output_dir`
(see [Outputs](outputs.md)); `ww3_grid` then reads them as described in
[WW3 setup](ww3.md). On Datarmor, `scripts/natl_grid.pbs` runs steps 2–4;
it cuts GEBCO only for the North Atlantic, so run step 1 by hand for a new
region and pass the configuration with
`qsub -v CFG=configs/biscay.json,GRID=grids/biscay scripts/natl_grid.pbs`.

## 2. Choosing the domain

- `domain` is a longitude/latitude box in geodetic degrees, with
  longitudes in (−180°, 180°] and `lon_min < lon_max`. **A domain cannot
  cross the antimeridian (180°)** in the current version; a Pacific grid
  has to stop at 180° or be split in two.
- Wet cells whose centre is inside the box and which touch the box edge
  become **open-boundary nodes**. Put the edges in deep water, away from
  the area of interest, and plan the boundary spectra (section 6). Edges
  on land produce no boundary nodes.
- Cells whose centre is outside the box are dry, so the grid follows the
  box edges in steps of one coarse cell.
- With `keep_largest_component` (default), only the largest connected wet
  region is kept: lakes and seas joined to it only through cells drier
  than `min_depth_m` or narrower than a cell disappear. Check
  `n_components` in the summary.

## 3. Choosing the resolution

The size of a HEALPix cell is $R\sqrt{\pi/3}/\mathrm{nside}$, the same
everywhere on the sphere:

| nside | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|---|
| cell size (km) | 102 | 51 | 25.5 | 12.7 | 6.4 | 3.2 | 1.6 | 0.80 | 0.40 |

- `nside_min` sets the coarsest cells (open ocean) and `nside_max` the
  finest; both are powers of two.
- `nside_max / nside_min` must be at most $2^{\texttt{max\_level\_span}}$
  (default 4 levels, for example 128 → 2048). Raise `max_level_span` to go
  further, for example to 5 for 256 → 8192.
- **`nside_max` is only a cap.** The size functions decide where the cells
  are fine: a cell is refined until it is smaller than the required size
  $\Delta$, or until it reaches `nside_max`. Raising `nside_max` alone
  refines only the places where $\Delta$ is already smaller than the
  current finest cell.

### The wavelength criterion (`wl`)

$\Delta = T_{M2}\sqrt{g h}/\alpha_{wl}$, with $T_{M2}$ = 44,712 s, gives
for `wl` = 100 ($\Delta \approx 1.40\,\mathrm{km}\times\sqrt{h/1\,\mathrm{m}}$):

| depth $h$ | 1 m | 5 m | 20 m | 100 m | 1000 m | 4000 m |
|---|---|---|---|---|---|---|
| $\Delta$ (km), `wl` = 100 | 1.4 | 3.1 | 6.3 | 14 | 44 | 89 |
| $\Delta$ (km), `wl` = 200 | 0.70 | 1.6 | 3.1 | 7.0 | 22 | 44 |

$\Delta$ scales as $1/\alpha_{wl}$: doubling `wl` halves every cell size
and so moves every depth contour one level finer. The coarsest cells are
reached only where $\Delta$ exceeds the `nside_min` cell size (51 km at
nside 128: deeper than about 1,300 m with `wl` = 100).

### The slope criterion (`slp`)

$\Delta = 2\pi h/(\alpha_{slp}|\nabla \tilde h|)$ refines shelf breaks,
canyons and seamounts. `slope_max_depth_m` (300 m in grid v3) restricts it
to the depths where wind waves feel the bottom; `fl` filters the
bathymetry before the gradient (`fl` = −1: Rossby radius) so that small
features do not refine large areas. Raising `slp` (10 → 20) refines the
slopes without changing the shallow areas.

### The transition (`grade`)

`grade` (0.2) limits the growth of the size per unit distance: the fine
cells are surrounded by bands of intermediate sizes. A smaller `grade`
(0.1) widens these bands, a larger one narrows them. With `grade` = 0 the
level can change as soon as the 2:1 balance allows.

## 4. Making a grid finer

**Finer near every coast.** Raise `nside_max` by one level and `wl` by a
factor of 2: for example `nside_max` 4096 and `wl` 200 give 1.6 km at 5 m
depth instead of 3.2 km. Raising only `nside_max` refines only where
$\Delta$ is below 1.6 km, that is shallower than about 1.3 m with
`wl` = 100: in practice a thin ring of cells along the coast, whose
bathymetry blocks contain land (counted as `min_depth_m`).

**Finer in one area only.** There is no key restricting the refinement to
a sub-box. Two ways:

- build a **separate regional grid** on a smaller domain with a larger
  `nside_max` (and `nside_min`), and give it open-boundary spectra from
  the larger run (section 6). This is the usual WW3 nesting; each grid
  stays small;
- use the **coastal distance rules** `refine` (grids v1 and v2): cells
  closer than `max_dist_km` to the coast get at least `nside`. They apply
  to every coast of the domain, so they make sense mainly on a small
  domain. They need a coastline (`coast`: GSHHS shapefile, GSHHG binary
  or the zero contour of the bathymetry).

**Bathymetry resolution.** GEBCO 2024 has a step of 15″ (460 m in
latitude). By default the bathymetry is averaged on blocks of half the
finest cell (`smooth_km`), so each finest cell holds about four blocks;
at nside 8192 (0.8 km) a block holds about one GEBCO sample, and nside
16384 is finer than GEBCO. For such grids, provide a finer bathymetry in
the same format (`elev.f32` and `meta.json`, see [Configuration](configuration.md)).

**Cost.** Each extra level near the coast multiplies the number of coastal
nodes by about 2 to 4, depending on how wide the refined band is. Grid v3
(794,613 nodes, nside 128 → 2048, North Atlantic) takes 70 s on 8 cores
and 7 GB; the WW3 run time grows with the number of nodes. Start with a
small domain, read `nodes_per_nside` in the summary, then enlarge.

## 5. Example: a finer grid of the Bay of Biscay

The configuration below is an illustration, not a tested grid: it
refines from 12.7 km (nside 512) in the deep Bay of Biscay to 800 m
(nside 8192) on the French coast, using the twice-finer wavelength
criterion.

```json
{
  "name": "biscay",
  "domain": { "lon_min": -10.0, "lon_max": 0.0, "lat_min": 43.0, "lat_max": 49.0 },
  "ellipsoid": "wgs84",
  "nside_min": 512,
  "nside_max": 8192,
  "max_level_span": 4,
  "refine_bathy": { "wl": 200.0, "slp": 20.0, "fl": -1.0, "grade": 0.2,
                    "slope_max_depth_m": 300.0, "min_depth_m": 1.0 },
  "min_depth_m": 2.0,
  "wet_fraction": 0.5,
  "keep_largest_component": true,
  "bathy": "../data/gebco/biscay/meta.json",
  "output_dir": "../grids/biscay"
}
```

The keys left out (`quality`, `ww3`) take their defaults. The western and
northern edges are at sea and become open boundaries; the southern and
eastern edges are mostly on land. `min_depth_m` is
lowered to 2 m so that the narrow estuaries stay wet.

## 6. What changes in WW3

- **Time steps.** The implicit scheme (PDLIB, `IMPTOTAL = T`) has no CFL
  limit, so `dtxy` does not have to follow the finest cell; the North
  Atlantic runs use 600 s with 3.2 km cells. For much finer grids, compare
  a run with shorter steps before choosing.
- **Open boundaries.** `ww3_grid` must report as many active boundary
  points as the `.obc` file holds. `scripts/boundary_spectra.py` rebuilds
  boundary spectra for any grid from the partitions of the Ifremer
  `GLOBMULTI` global run (`--glob`, `--start`, `--stop` for another
  period); a regional grid nested in an HWW3 run can instead take point
  spectra of the parent run at its boundary nodes (`ww3_ounp`), as in any
  WW3 nesting. Without spectra the boundaries are open and empty: the
  swell coming from outside is missing (see [Validation](validation.md)).
- **Forcing.** The North Atlantic box is written as constants in the
  forcing scripts: `AREA` in `scripts/get_era5.py` (ERA5 request, with a
  2° margin) and the bounds `W`, `E`, `S`, `N` in `scripts/prep_cmems.py`
  (Copernicus currents and sea level). Change them to the new domain plus
  a margin before downloading the forcing.
- **Outputs.** `scripts/ww3_pyramid.py` writes the fields of any grid as
  a HEALPix pyramid; set `--levels` to the depths of `nside_min` to
  `nside_max` (for example `9-13` for nside 512 to 8192).

## 7. Checks

After step 3, read the summary (`grids/<name>_summary.json`):

- `n_nodes` and `nodes_per_nside`: the number of nodes per level;
- `n_open_boundary`: the open-boundary nodes;
- `n_components`: 1 with `keep_largest_component`;
- `mesh`: the validation report (orientation, manifold edges, angles);
  with `quality.enforce` the program stops if an angle is outside
  [`min_angle_deg`, `max_angle_deg`].

Then draw the levels (`scripts/plot_levels.py`) or a zoom
(`scripts/plot_cells_tris.py`, after `scripts/extract_zoom.py`), or
write the grid pyramid (`scripts/grid_pyramid.py`) and look at it in
gridlook.
