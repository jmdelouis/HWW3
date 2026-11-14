# Validation

## Automatic tests

```bash
cd hww3-grid
cargo test --release -- --nocapture     # prints the JSON summaries
```

They run in the CI on every push (`.github/workflows/ci.yml`).

| Test | What it checks |
|---|---|
| `geo::tests::authalic_round_trip` | geodetic → authalic → geodetic latitude to $10^{-12}$ rad; WGS84 authalic radius 6371007.18 m; 0.128° difference at 45°N |
| `shapefile::tests::polyline_round_trip` | a two-part PolyLine record is read back exactly; bounding-box filtering |
| `ear_clip_square` | ear clipping of a convex pentagon gives 3 counter-clockwise triangles |
| `bow_tie_pinch_removed` | a bow-tie node (2 fans) is detected and the smaller fan removed |
| `delaunay_flip_short_diagonal` | a flat quadrilateral split along its long diagonal is flipped to the short one and the minimum angle increases; a second call does nothing |
| `trim_boundary_sliver` | a boundary sliver is removed, the good triangle kept |
| `uniform_open_ocean` | uniform nside-64 grid across a HEALPix base-face corner (45°E, 41.8°N): valid mesh, disc (Euler 1), no isolated node, level jump 0, ≈ 2 triangles per node, open boundary present |
| `island_multiscale` | circular island on the base-face corner, nside 32–512: valid mesh, one hole (Euler 0), level jump 1, every level used, raw angles 17.0°–143.7°, brought to 28.9°–113.6° by the quality step (37 flips, 1 boundary triangle removed) |
| doc tests | the examples in the Rust documentation compile and run |

## Mesh checks

Every grid is checked by `validate::check`; the program fails if any of
the first four counts is not zero.

| Check | Meaning |
|---|---|
| `n_nonmanifold_edges` | edges shared by more than two triangles |
| `n_inconsistent_edges` | interior edges traversed twice in the same direction |
| `n_negative` | clockwise triangles (seen from outside the sphere) |
| `n_pinch_nodes` | nodes with two or more triangle fans (rejected by WW3) |
| `n_isolated_nodes` | nodes used by no triangle |
| `euler` | $V - E + F$: 1 for a disc, $1 - h$ with $h$ holes (islands) |
| `min_angle_deg`, `max_angle_deg`, `min_angle_hist` | angle statistics |
| `max_level_jump` | largest level difference across an edge (1 with the 2:1 balance) |

## North Atlantic grid

Built on Datarmor (`scripts/natl_grid.pbs`, PBS job 7893701, 8 cores,
152 s) before the triangle-quality step:

| Quantity | Value |
|---|---|
| nodes per nside 128 / 256 / 512 / 1024 / 2048 | 10,160 / 6,274 / 73,088 / 135,561 / 277,607 |
| nodes / triangles | 502,690 / 949,199 |
| open-boundary nodes | 971 |
| non-manifold / inconsistent / clockwise / pinch | 0 / 0 / 0 / 0 |
| Euler characteristic | −693 (694 islands) |
| max level jump | 1 |
| angles | 17.5°–139.7°; 3,781 triangles < 20°, 3,070 > 120° |
| pinch triangles removed | 671 |

With the triangle-quality step (`configs/natl_v2_datarmor.json`, PBS job
7894086, 8 cores, 130 s; quality step 8 s):

| Quantity | v1 (no quality step) | v2 (flips + boundary trim) |
|---|---|---|
| nodes / triangles | 502,690 / 949,199 | 502,408 / 947,850 |
| nodes at nside 2048 | 277,607 | 277,325 |
| edge flips / boundary triangles removed | – | 14,750 / 1,275 |
| angles | 17.5°–139.7° | 20.7°–127.7° |
| triangles < 20° / > 120° | 3,781 / 3,070 | 0 / 3 |
| Euler characteristic | −693 | −682 |
| non-manifold / inconsistent / clockwise / pinch | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| open-boundary nodes, max level jump | 971, 1 | 971, 1 |

```{figure} _static/fig/natl_gibraltar.png
:width: 80%

Strait of Gibraltar: HEALPix cells (blue) and triangles (red).
```

```{figure} _static/fig/natl_openocean.png
:width: 80%

Open ocean (32°W–24°W, 38°N–44°N): sheared cells near the boundary
between HEALPix polar and equatorial base faces.
```

## First WW3 run

12 h cold start with a homogeneous 15 m/s westerly wind
(`ww3/natl_test`, PBS job 7893925): 56 MPI ranks, 18.6 s per 600 s step;
median $H_s$ 1.0 / 1.85 / 2.75 m after 3 / 6 / 12 h, no NaN.

```{figure} _static/fig/natl_test_hs.png
:width: 100%

$H_s$ after 12 h of uniform wind from rest.
```

## January 2026 against altimetry

Run `natl_jan2026` (grid v1, ERA5 wind and ice, no boundary spectra, PBS job
7894000: 40,754 s on 112 ranks) compared with the L3 NRT along-track $H_s$
of ten altimeters (CMEMS `WAVE_GLO_PHY_SWH_L3_NRT_014_001`, Datarmor
mirror) by `scripts/validate_altimetry.py`
(`qsub -v RUN=natl_jan2026,GRID=grids/natl scripts/natl_validate.pbs`,
2 min). Each 1 Hz sample takes the value of the nearest wet, ice-free node
if it lies within 0.75 of that node's pixel size, linear in time; days 1–3
(cold start) are skipped and samples are averaged in blocks of 7 (≈ 50 km).

| subset | N | mean obs (m) | bias (m) | RMSE (m) | SI | r |
|---|---|---|---|---|---|---|
| all | 331,154 | 2.92 | −0.19 | 0.45 | 0.14 | 0.972 |
| nside 128 | 183,701 | 3.20 | −0.20 | 0.47 | 0.13 | 0.972 |
| nside 256 | 31,675 | 2.99 | −0.20 | 0.46 | 0.14 | 0.970 |
| nside 512 | 82,837 | 2.65 | −0.18 | 0.43 | 0.15 | 0.970 |
| nside 1024 | 29,615 | 2.05 | −0.14 | 0.36 | 0.16 | 0.968 |
| nside 2048 | 3,326 | 1.82 | −0.12 | 0.33 | 0.17 | 0.967 |
| North Sea | 5,567 | 2.90 | −0.31 | 0.53 | 0.15 | 0.978 |
| Mediterranean | 14,539 | 2.19 | −0.20 | 0.40 | 0.16 | 0.967 |

The ten missions agree within 0.08 m in bias. The negative bias comes from
the missing South Atlantic swell at the equatorial open boundary and from
the North-West Atlantic storms (ERA5 winds with default ST4 `BETAMAX`), not
from the coarse cells: errors decrease with nside. The script also writes
the statistics per mission and per day (`runs/<run>/val/<run>_stats.json`).

Run `natl_jan2026_cur` (same forcing plus Copernicus Marine surface
currents and sea level, PBS job 7894069, 41,654 s; validation job 7898389)
on the same 331,154 blocks: bias −0.225 m, RMSE 0.462 m, SI 0.138,
r 0.9738. Currents lower the scatter index and raise the correlation at
every nside, but make the bias 0.035 m more negative (WW3 uses the wind
relative to the current, and the mean current mostly runs with the
westerlies). Statistics: `results/natl_jan2026_cur_val_stats.json`.

Run `natl_v2_jan2026_cur` (grid v2, same forcing as `natl_jan2026_cur`,
ST4 `BETAMAX` 1.75 instead of 1.43, PBS job 7899026: 41,927 s on 112 ranks,
`ww3_ounf` 718 s; validation job 7905048) on the same 331,154 blocks:

| subset | v1 + currents, `BETAMAX` 1.43: bias / RMSE / SI / r | v2 + currents, `BETAMAX` 1.75: bias / RMSE / SI / r |
|---|---|---|
| all | −0.225 / 0.462 / 0.138 / 0.9738 | +0.005 / 0.384 / 0.131 / 0.9748 |
| nside 128 | −0.240 / 0.486 / 0.132 / 0.9730 | +0.012 / 0.406 / 0.127 / 0.9741 |
| nside 256 | −0.232 / 0.468 / 0.136 / 0.9719 | +0.006 / 0.384 / 0.128 / 0.9728 |
| nside 512 | −0.212 / 0.437 / 0.145 / 0.9717 | −0.002 / 0.360 / 0.136 / 0.9728 |
| nside 1024 | −0.175 / 0.370 / 0.159 / 0.9701 | −0.016 / 0.313 / 0.153 / 0.9714 |
| nside 2048 | −0.143 / 0.337 / 0.168 / 0.9679 | −0.000 / 0.299 / 0.164 / 0.9690 |
| North Sea | −0.347 / 0.557 / 0.150 / 0.9787 | −0.126 / 0.373 / 0.121 / 0.9805 |
| Mediterranean | −0.225 / 0.405 / 0.154 / 0.9689 | −0.054 / 0.330 / 0.148 / 0.9697 |

Grid v2 differs from v1 only by the triangle-quality step (282 fewer
nodes), so the change is the effect of `BETAMAX`: the bias is removed at
every nside, the RMSE drops by 0.08 m (17 %) and the scatter index and
correlation improve slightly. The ten missions now lie between −0.050 m
(CFOSAT) and +0.038 m (HY-2C) in bias. The remaining North Sea bias
(−0.13 m) is a candidate for the finer v3 grid. Statistics:
`results/natl_v2_jan2026_cur_val_stats.json`.

```{figure} _static/fig/natl_jan2026_scatter.png
:width: 100%

WW3 against altimeter $H_s$, and bias / RMSE by nside of the matched node.
```

```{figure} _static/fig/natl_jan2026_maps.png
:width: 100%

Mean and maximum $H_s$ in January 2026, and WW3 − altimeter along tracks.
```
