# HWW3: HEALPix unstructured grids for WAVEWATCH III

HWW3 builds WAVEWATCH III (WW3) unstructured grids whose nodes are the
centres of **multi-resolution HEALPix cells**. The grid is refined from the
bathymetry, with the depth and slope criteria of the classical wave-model
grid generators (from `nside_min` over the deep ocean to `nside_max` in
shallow water and over steep slopes),
every node is a NESTED HEALPix pixel, and the WW3 output is therefore
already a HEALPix map: it is written as a GRID4EARTH Zarr V3 pyramid and
displayed in gridlook without interpolation.

WW3 itself is **not modified**: the grid is a standard Gmsh triangle mesh
read by `ww3_grid` with `GRID%TYPE = 'UNST'` and run with the PDLIB
schemes, implicit or explicit.

Alongside, the `hww3-wave` crate rewrites in Rust, for one GPU, the parts
of WW3 that the explicit configuration runs, towards a **differentiable
WW3** that returns the gradients of the misfit to altimeters and buoys
for learning chains. Its propagation-only model runs 48 h of grid
v5 $K=4$ in 32 s on one A100 and agrees with WW3 to about 1 mm in $H_s$ at the median
node.

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
[outputs](outputs.md) of the `hww3-grid` generator, and how to build
[a grid for another region or a finer grid](new_region.md).
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
:::{grid-item-card} GPU wave model
The [GPU wave model](wave_model.md) `hww3-wave`: its
[numerics](wave_numerics.md), its [results](wave_results.md) against WW3,
and the [design and cost](differentiable.md) of the differentiable WW3.
:::
:::{grid-item-card} Reference
The [API and scripts](api.md), the Rust API references of both crates,
and the [tests and continuous integration](development.md).
:::
::::

```{toctree}
:caption: Grids
:maxdepth: 2
:hidden:

installation
configuration
new_region
algorithm
outputs
validation
```

```{toctree}
:caption: WAVEWATCH III runs
:maxdepth: 2
:hidden:

ww3
reproducing
```

```{toctree}
:caption: GPU wave model
:maxdepth: 2
:hidden:

wave_model
wave_numerics
wave_results
differentiable
```

```{toctree}
:caption: Reference
:maxdepth: 2
:hidden:

api
development
```
