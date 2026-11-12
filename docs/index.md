# HWW3: HEALPix unstructured grids for WAVEWATCH III

HWW3 builds WAVEWATCH III (WW3) unstructured grids whose nodes are the
centres of **multi-resolution HEALPix cells**. The grid is refined near the
coast (from `nside_min` in the open ocean to `nside_max` at the shoreline),
every node is a NESTED HEALPix pixel, and the WW3 output is therefore
already a HEALPix map: it is written as a GRID4EARTH Zarr V3 pyramid and
displayed in gridlook without interpolation.

WW3 itself is **not modified**: the grid is a standard Gmsh triangle mesh
read by `ww3_grid` with `GRID%TYPE = 'UNST'` and run with the implicit
PDLIB scheme.

```{figure} _static/fig/natl_levels.png
:width: 90%

North Atlantic grid: 502,690 nodes, HEALPix nside 128 (open ocean) to 2048
(coast).
```

The demonstration case is the North Atlantic, from the equator to 80°N,
with the Mediterranean and the North Sea, forced by ERA5 and Copernicus
Marine for January 2026 and run on Datarmor (Ifremer).

::::{grid} 2
:::{grid-item-card} Build a grid
[Installation](installation.md), [configuration](configuration.md) and
[outputs](outputs.md) of the `hww3-grid` generator.
:::
:::{grid-item-card} How it works
The [algorithm](algorithm.md): refinement, 2:1 balance, bathymetry,
triangulation from the HEALPix adjacency, triangle quality.
:::
:::{grid-item-card} Is it right?
[Validation](validation.md): unit and end-to-end tests, mesh checks,
statistics of the North Atlantic grid.
:::
:::{grid-item-card} Redo everything
[Reproducing](reproducing.md) the grid, the WW3 build, the forcing and the
runs, step by step, with the job scripts.
:::
::::

```{toctree}
:maxdepth: 2
:hidden:

installation
configuration
algorithm
outputs
validation
ww3
reproducing
api
```
