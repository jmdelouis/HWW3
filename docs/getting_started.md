# Getting started

This page is for a first reading: what HWW3 does, step by step, on which
machine each step runs, and where to go next. No knowledge of WW3 is
assumed; the words in italics are explained in the [Glossary](glossary.md).

## What HWW3 does

A wave model computes, at every point of a mesh and at every time step,
the *wave spectrum*: how the wave energy is spread over frequencies and
directions. From the spectrum come the numbers that people use and that
satellites and buoys measure: the *significant wave height* $H_s$ and the
*mean period* $T_{m02}$. HWW3 does this for the North Atlantic with
WAVEWATCH III (WW3) on meshes whose nodes are HEALPix cells, and with a GPU
rewrite of WW3 that can also return derivatives, so that the model can
learn its parameters from the observations.

The work goes in five steps, each with its section of this site:

| Step | What happens | Section |
|---|---|---|
| 1. Grid | `hww3-grid` places the nodes, small near the coast and large over the deep ocean, from the bathymetry, and writes the triangle mesh that WW3 reads | [Preparing a run](preparing.md) |
| 2. Inputs | The *forcing* (wind, currents, water level, sea ice) and the waves entering at the *open boundary* are interpolated onto the nodes of the grid | [Preparing a run](preparing.md) |
| 3. Run | WW3 on many CPU cores, or the GPU model on one GPU, steps the spectrum in time and writes hourly fields | [Running the model](running.md) |
| 4. Comparison | The hourly $H_s$ and $T_{m02}$ are compared with satellite *altimeters* and moored *buoys* | [Analysis tools](analysis.md) |
| 5. Learning | *Tangent runs* give the derivatives of the misfit with respect to the model parameters, and a *Gauss–Newton step* moves the parameters to reduce it | [Learning the parameters](learning.md) |

What each step gave on the North Atlantic in January 2026 is in
[Results](results.md); the programs, scripts and design notes are in
[Reference](reference.md).

## Where things run

| Machine | Scheduler | Used for | Working directory `R` |
|---|---|---|---|
| A workstation | none | building a grid, figures, reading the results | the repository |
| Datarmor (Ifremer) | PBS: `qsub`, `qstat` | WW3 runs, forcing preparation, the mirrors of GEBCO, Copernicus Marine, altimeters and buoys | `/home1/datawork/$USER/HWW3` |
| hpc cluster (Ifremer) | Slurm: `sbatch`, `squeue` | the GPU model on A100s, the calibration, WW3 references | `/scratch/jmdeloui/HWW3` |

The GPU model does not build on Datarmor (glibc too old for CubeCL), so
everything that needs it runs on the hpc cluster. The hpc cluster has no
access to the Datarmor data mirrors: the observations of January 2026 and
the inputs of the GPU runs were copied there (see
[Reproducing](reproducing.md), section 8).

The job scripts in `scripts/` hold the paths of the reference set-up. The
PBS scripts take the working directory as `qsub -v R=...` and write their
log where `qsub -o` says. The Slurm scripts set
`S=/scratch/jmdeloui/HWW3` near their top and write their logs to
`/scratch/jmdeloui/HWW3/logs`: another user changes these two lines.

## Install

[Installation](installation.md) lists what each part needs: a Rust
toolchain for the grid generator and the GPU model (with the CUDA toolkit
for the GPU), Python with `numpy`, `netCDF4`, `scipy` and `matplotlib` for
the scripts, and the WW3 build.

## Tutorials

Each tutorial states its prerequisites, gives the commands in order, says
what output to expect and how to check it. They follow one another, but
each can be run alone when its inputs exist.

| Tutorial | Machine | Time |
|---|---|---|
| [Build a grid](tutorials/build_grid.md) | workstation | under an hour, most of it for the bathymetry download |
| [Prepare the inputs and run WW3](tutorials/run_ww3.md) | Datarmor | about 3 h of jobs for a week of model time |
| [Run the GPU model for 48 h](tutorials/run_gpu.md) | hpc cluster | about 5 min on one A100, plus the build |
| [Validate a run](tutorials/validate_run.md) | hpc cluster or Datarmor | a few minutes per run |
| [Take one calibration step](tutorials/calibration_step.md) | hpc cluster | a few hours of GPU jobs |

```{toctree}
:hidden:
:maxdepth: 2

installation
tutorials/build_grid
tutorials/run_ww3
tutorials/run_gpu
tutorials/validate_run
tutorials/calibration_step
glossary
```
