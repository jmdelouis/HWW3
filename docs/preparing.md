# Preparing a run

A run needs three things: a **grid**, the **inputs** that drive the model
on that grid, and the **programs** (the WW3 build, or the GPU model). This
section covers the first two and the WW3 build; the GPU model is built as
described in [GPU wave model](wave_model.md).

## The grid

`hww3-grid` reads one JSON configuration and a bathymetry cut from GEBCO,
refines the HEALPix cells where the depth and the slope ask for it, and
writes the triangle mesh, the list of open-boundary nodes and the WW3
namelists.

- [Configuration](configuration.md): every key of the JSON file, with the
  North Atlantic example, and the format of the bathymetry input.
- [A grid for another region](new_region.md): the four commands, how to
  choose the domain and the resolution, how to make a grid finer, and
  what changes in WW3.
- [Outputs](outputs.md): the files written (mesh, open boundary, raw
  arrays, summary), how to read them in Python, and the HEALPix pyramid of
  the model output for gridlook.
- How the generator works (refinement, 2:1 balance, triangulation, triangle
  quality, explicit time step): [Algorithm](algorithm.md), in the
  Reference section.

Hands-on: [Build a grid](tutorials/build_grid.md).

## External data

| Data | Source | Scripts | Described in |
|---|---|---|---|
| Bathymetry | GEBCO 2024, 15″ | `fetch_sources.sh`, `prep_gebco.py` | [Configuration](configuration.md), *Bathymetry input* |
| Coastline (distance rules of grids v1 and v2 only) | GSHHS shapefile | path in `configs/natl_datarmor.json` | [Configuration](configuration.md) |
| 10 m wind, hourly | ERA5 single levels, through `cdsapi` | `get_era5.py`, `prep_era5.py`, `natl_jan2026_prep.pbs` | [WW3 setup](ww3.md), *Forcing* |
| Sea-ice concentration | ERA5, January mean, fixed in time (ice as an obstruction) | the same | [WW3 setup](ww3.md), *Forcing* |
| Surface currents and water level, hourly | Copernicus Marine `GLOBAL_ANALYSISFORECAST_PHY_001_024`, 1/12°, Datarmor mirror | `prep_cmems.py`, `natl_jan2026_cur_prep.pbs` | [WW3 setup](ww3.md), *Forcing* |
| Tide | FES2014, added to the currents and the water level | `fes_tides.py` (pyTMD), `natl_tide_setup.pbs` | [Reproducing](reproducing.md), *FES2014 tide* |
| Open-boundary spectra | partitions of the Ifremer `GLOBMULTI` GLOB-30M global run | `boundary_spectra.py`, `natl_bnd_setup.pbs` | [Reproducing](reproducing.md), *Grid v3 run with open-boundary spectra* |

The forcing scripts hold the North Atlantic box as constants (`AREA` in
`get_era5.py`, `W`, `E`, `S`, `N` in `prep_cmems.py`): change them for
another region, as [A grid for another region](new_region.md) explains in
its section 6. The observations used to score a run (altimeters, buoys)
are described in [Analysis tools](analysis.md).

## The WW3 build

[WW3 setup](ww3.md) gives the build on Datarmor (`build_ww3.pbs`, Intel
2020) and on the hpc cluster (`build_ww3_hpc.slurm`, oneAPI 2025), the
check of one against the other, the switches, the numerical settings
written by `hww3-grid`, the format the forcing files must have, and the
pitfalls met on Datarmor.

## Preprocessing with the WW3 tools

WW3 does not read the grid and the forcing directly. In a run directory:

1. `ww3_grid` reads the mesh and the namelists and writes `mod_def.ww3`
   (the ST4 parameter `BETAMAX` is appended to the namelists by the set-up
   script);
2. `ww3_prnc` interpolates each forcing field onto the nodes:
   `wind.ww3`, `current.ww3`, `level.ww3`, `ice.ww3`. These files belong
   to one grid: a new grid needs `ww3_prnc` again;
3. `ww3_bounc` writes the boundary spectra `nest.ww3`.

The set-up scripts `natl_v2_setup.pbs` (steps 1 and 2),
`natl_bnd_setup.pbs` (step 3), `natl_exp_setup.pbs` (the explicit scheme)
and `natl_tide_setup.pbs` (the tide) do this on Datarmor; their commands
are in [Reproducing](reproducing.md), sections 4, 6 and 7. Hands-on:
[Prepare the inputs and run WW3](tutorials/run_ww3.md).

## Preprocessing in Rust

The GPU model can make its own inputs, so that a GPU run needs no WW3
tool; WW3 then only makes the references. A 216 h run with Rust inputs
only gives the same calibration cost and scores as the run with the
inputs of the WW3 tools (`scripts/pre_allrust_hpc.slurm`).

| Program | Replaces | Role |
|---|---|---|
| `hww3-src build-params` | the source-term part of `ww3_grid` | the parameter file of the source terms from the namelists and the grid |
| `hww3-src params` | `ww3_grid` with other parameters | a parameter file for other values of `BETAMAX`, `SDSC2`, `NLPROP`, `BJGAM`, with the stress tables computed in Rust |
| `hww3-pre prnc` with `scripts/prnc_nc.py` | `ww3_prnc` | the forcing files from the netCDF fields |
| `hww3-pre bounc` with `scripts/bounc_nc.py` | `ww3_bounc` | the boundary file from the station spectra |
| `hww3-prop run --restart calm` | a WW3 restart | a start from calm conditions, as WW3 starts without a restart file |

Their options are in [GPU wave model](wave_model.md) (sections on
`hww3-src` and `hww3-pre`), their checks against the WW3 tools in
[GPU model results](wave_results.md), section *Preprocessing in Rust*, and
the commands in [Reproducing](reproducing.md), section 8.

```{toctree}
:hidden:
:maxdepth: 2

configuration
new_region
outputs
ww3
```
