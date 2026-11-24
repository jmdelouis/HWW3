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
for learning chains. With the full ST4 physics it runs a week of grid
v5 $K=4$ in 868 s on one A100 (124 s per model day), agrees with WW3 to
2.2 µm in $H_s$ at the median node, and its tangent runs drive the
calibration of the ST4 parameters.

```{figure} _static/fig/natl_levels.png
:width: 90%

North Atlantic grid: 502,690 nodes, HEALPix nside 128 (open ocean) to 2048
(coast).
```

The demonstration case is the North Atlantic, from the equator to 80°N,
with the Mediterranean and the North Sea, forced by ERA5 and Copernicus
Marine for January 2026 and run on Datarmor and on the Ifremer hpc cluster.

## Where to start

New to HWW3? [Getting started](getting_started.md) explains in plain
words what each step does and on which machine it runs. Then five short
tutorials take you through the main tasks, each with its prerequisites,
its commands, the output to expect and how to check it:

1. [Build a grid](tutorials/build_grid.md) for a region, on a workstation;
2. [Prepare the inputs and run WW3](tutorials/run_ww3.md) on Datarmor;
3. [Run the GPU model for 48 h](tutorials/run_gpu.md) on the hpc cluster;
4. [Validate a run](tutorials/validate_run.md) against altimeters and buoys;
5. [Take one calibration step](tutorials/calibration_step.md) from tangent runs.

The words of the field (spectrum, $H_s$, forcing, restart, tangent...) are
explained in the [Glossary](glossary.md).

## The sections of this site

::::{grid} 1 2 2 2
:::{grid-item-card} Preparing a run
[The grid](preparing.md) (`hww3-grid`, its configuration, a grid for
another region, its outputs), the external data (bathymetry, ERA5 wind,
currents, water levels and tide, sea ice, open-boundary spectra), the WW3
build and the preprocessing, by WW3 or in Rust.
:::
:::{grid-item-card} Running the model
[WW3 runs](running.md) on Datarmor (PBS) and on the hpc cluster (Slurm),
and the GPU model `hww3-prop` on one A100.
:::
:::{grid-item-card} Learning the parameters
[The differentiable model](learning.md): tangent runs, observation
operators, the cost and the Gauss–Newton calibration of the ST4
parameters.
:::
:::{grid-item-card} Analysis tools
[Scripts](analysis.md) that score a run against altimeters and buoys,
compare two runs, draw maps and diagnose the observations.
:::
:::{grid-item-card} Results
[What the runs give](results.md): the grids and the WW3 runs against the
observations, the GPU model against WW3, the calibration steps.
:::
:::{grid-item-card} Reference
[The API and scripts](reference.md), the Rust API references, the design
notes (grid algorithm, numerics of the GPU model) and the tests.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

getting_started
preparing
running
learning
analysis
results
reference
```
