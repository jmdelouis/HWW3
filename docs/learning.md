# Learning the parameters

The GPU model exists so that the wave model can **learn from the
observations**: adjust its parameters (and later corrections of its
forcing) so that its $H_s$ and $T_{m02}$ match the altimeters and the
buoys better. That needs the derivatives of the misfit with respect to
what is adjusted. HWW3 computes them with tangent runs of the GPU model;
finite differences served only to check the tangents and never enter the
calibration.

## The idea in four steps

1. **Run.** The GPU model runs from a restart and writes the hourly $H_s$
   and $T_{m02}$ at every node.
2. **Observation operators.** Fixed linear operators pick the model values
   where and when the observations are: each altimeter block (7 samples,
   about 50 km) takes the nearest wet, ice-free node, interpolated in time;
   each buoy hour takes the nearest node, or a depth-aware weighting of the
   nodes around the buoy. They reproduce the matching of the validation
   scripts exactly.
3. **Cost.** $J$ sums the squared misfits, each divided by its expected
   error, plus a term that keeps the parameters near their start
   values.
4. **Derivatives and step.** One tangent run per parameter
   (`hww3-prop --tangent NAME`) gives the derivatives of every model value,
   hence of $J$; a Gauss–Newton step proposes new parameters, and a value
   run at that point keeps them only if $J$ decreased.

Hands-on: [Take one calibration step](tutorials/calibration_step.md).

## What is calibrated

| Parameter | Role in the ST4 physics | Start value |
|---|---|---|
| `BETAMAX` | growth of the waves by the wind | 1.75 |
| `SDSC2` | whitecapping dissipation | −2.2e-5 |
| `NLPROP` | nonlinear interactions between waves (DIA) | 2.5e7 |
| `BJGAM` | breaking in shallow water | 0.73, frozen after the first step |

The control vector is $\ln p - \ln p_0$. A new `BETAMAX` changes the
stress tables that `ww3_grid` writes; they are computed in Rust
(`hww3-src params`, `hww3-prop --betamax`), so the calibration needs
neither `ww3_grid` nor any other WW3 tool.

## Where to read more

| Topic | Page |
|---|---|
| The design of the differentiable model: scope, development cost, automatic differentiation in Rust on GPU, speed, memory and checkpointing, gradient checks, the observation operators (section 7), learned corrections on the HEALPix structure, the milestones, the calibration (section 9.5) and the plan after it (section 9.7) | [Differentiable WW3](differentiable.md) |
| The tangent kernels and their checks against finite differences | [GPU model results](wave_results.md), *Milestone M3* |
| The set-up and the steps of the calibration, the three costs, the spin-up rule, the diagnostics of the observations | [GPU model results](wave_results.md), *Calibration of the ST4 parameters* and the sections after it |
| The scripts: `cal_run_hpc.slurm`, `cal_obs.py`, `cal_gn.py`, `buoy_op.py`, `buoy_sigma.py`, `buoy_exclude.py`, `cal_point.py` | [API and scripts](api.md) |
| The commands as they were run | [Reproducing](reproducing.md), section 8 |

```{toctree}
:hidden:
:maxdepth: 2

differentiable
```
