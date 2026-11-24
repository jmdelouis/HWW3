# Results

What the grids, the WW3 runs and the GPU model gave on the North Atlantic
in January 2026. The same numbers, with the figures, are gathered in the
LaTeX note `tex/hww3/main.tex` (see [Reproducing](reproducing.md),
section 9).

## Grids and WW3 runs

[Validation](validation.md):

- the automatic tests of the grid generator and the checks of every mesh;
- the North Atlantic grids v1 to v5: nodes per level, triangle angles,
  the explicit time step and what it costs, and the grids that cap the
  explicit sub-steps (v5);
- the first WW3 run (homogeneous wind), then the January 2026 runs against
  the altimeters and the moored buoys, with the open-boundary spectra and
  with the FES2014 tide.

## The GPU model

[GPU model results](wave_results.md), milestone by milestone:

- M0: the propagation sweep and its transpose on one A100;
- M1: the propagation-only model against WW3;
- M2a and M2b: the source terms, simplified then the full ST4, against WW3
  over 48 h and a week, with the same altimeter and buoy scores;
- M3: the tangent runs against finite differences;
- the calibration of the ST4 parameters, the preprocessing in Rust, and the
  diagnostics of the buoy and altimeter observations.

## A few numbers

| What | Number | Page |
|---|---|---|
| Grid v4 (reference grid) | 794,613 nodes, nside 128 to 2048 | [Validation](validation.md) |
| Grid v5 $K=4$ (grid of the GPU model) | 424,123 nodes | [Validation](validation.md) |
| WW3, explicit scheme, grid v5 $K=4$, 112 cores | 2,601 s per model day | [Validation](validation.md) |
| GPU model, full ST4, grid v5 $K=4$, one A100 | 124 s per model day | [GPU model results](wave_results.md) |
| GPU model against WW3, 1–8 January | 2.2 µm in $H_s$ at the median node; the same scores | [GPU model results](wave_results.md) |
| WW3, explicit scheme, grid v5 $K=4$, against 45,518 altimeter blocks, 4–7 January | bias −1.5 cm, RMSE 28.3 cm | [GPU model results](wave_results.md) |

```{toctree}
:hidden:
:maxdepth: 2

validation
wave_results
```
