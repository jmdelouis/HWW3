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

### Grid v4: alternated diagonals and explicit time step

Grid v4 (`configs/natl_v4_datarmor.json`, PBS job 7935472, 8 cores,
149 s, 7.3 GB) is grid v3 with the two new connectivity steps (see
[Algorithm](algorithm.md)): the same 794,613 nodes, 160,814 alternated
blocks and 55,985 CFL flips. The mesh stays valid (same 1,531,542
triangles, Euler characteristic −787, no non-manifold or inconsistent edge,
angles 20.0°–130.0°, 2,550 triangles above 120° instead of 970). In the
equatorial band the share of north–south edges falls from 31 % to 17 % and
that of east–west edges rises from 1 % to 15 %. Explicit WW3 time step of
the interior nodes for the lowest frequency (0.0373 Hz), v3 / v4:

| nside | nodes | min Δt (s) | min L/Δx | median L/Δx |
|---|---|---|---|---|
| 128 | 8,862 | 1303 / 1458 | 0.535 / 0.599 | 0.761 / 0.846 |
| 256 | 15,018 | 614 / 682 | 0.535 / 0.592 | 0.774 / 0.826 |
| 512 | 40,899 | 274 / 304 | 0.522 / 0.557 | 0.771 / 0.812 |
| 1024 | 119,384 | 135 / 147 | 0.502 / 0.537 | 0.764 / 0.813 |
| 2048 | 551,192 | 59.6 / 64.7 | 0.469 / 0.502 | 0.760 / 0.837 |

The global explicit step rises from 59.6 s to 64.7 s (11 to 10 sub-cycles
for `dtxy` = 600 s); it is set by a nside-2048 node at 160–175 m depth,
where the group speed of the lowest frequency is largest. Alternation
alone (threshold 1.2) raises the medians but not the minima; with a
threshold of 1.5 the medians gain less; the CFL flips alone give the same
minima as both steps together with less isotropic diagonals.

```{figure} _static/fig/natl_v4_alternation.png
:width: 100%

Grids v3 and v4 at 2°–10°N, 40°–28°W (nside 128): cells (blue),
triangles (red), nodes (black).
```

January 2026 run on grid v4 with open-boundary spectra (implicit scheme,
same forcing, `BETAMAX` and spectra as the v3 run, PBS job 7935604):
`ww3_shel` 70,120 s instead of 68,800 s (+1.9 %).

| | v3 | v4 |
|---|---|---|
| altimeters (328,902 blocks): bias / RMSE (m) | +0.041 / 0.348 | +0.041 / 0.348 |
| altimeters: SI / r | 0.118 / 0.979 | 0.118 / 0.979 |
| buoys $H_s$ (268 buoys, 167,298 h): RMSE (m) / SI | 0.374 / 0.267 | 0.373 / 0.266 |
| buoys $T_{m02}$ (113,744 h): RMSE (s) / r | 1.325 / 0.780 | 1.316 / 0.783 |
| $H_s$ RMSE, Irish and Celtic seas (m) | 0.627 | 0.616 |
| $T_{m02}$ RMSE, Iberia and Canaries (s) | 0.854 | 0.811 |

Altimeter scores agree to within 1 mm at every level and in every region;
the buoy scores improve slightly, mostly near the coasts. Files:
`results/natl_v4_jan2026_bnd_{val,buoys}_stats.json`.

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

Run `natl_v3_jan2026_cur` (grid v3, same forcing and `BETAMAX` as the v2
run, PBS job 7901053: 68,832 s on 112 ranks, 1.64 times the v2 run for 1.58
times the nodes; `ww3_ounf` 835 s; validation job 7905050) matches 328,902
blocks, since the matching distance scales with the pixel size of the
nearest node; the subsets by nside hold different samples and are not
compared.

| subset | N v2 / v3 | bias v2 / v3 (m) | RMSE v2 / v3 (m) | SI v2 / v3 | r v2 / v3 |
|---|---|---|---|---|---|
| all | 331,154 / 328,902 | +0.005 / +0.005 | 0.384 / 0.376 | 0.131 / 0.129 | 0.9748 / 0.9758 |
| North Sea | 5,567 / 5,629 | −0.126 / −0.139 | 0.373 / 0.380 | 0.121 / 0.123 | 0.9805 / 0.9800 |
| Mediterranean | 14,539 / 14,454 | −0.054 / −0.054 | 0.330 / 0.331 | 0.148 / 0.149 | 0.9697 / 0.9695 |

At the 50 km scale of the altimeter blocks the grid hardly matters: the
whole-domain RMSE improves by 0.008 m and the shelf seas do not change
within 0.01 m. Coastal buoys are needed to see the effect of the finer
shelf resolution. Statistics: `results/natl_v3_jan2026_cur_val_stats.json`.

## January 2026 against moored buoys

`scripts/validate_buoys.py` (`qsub scripts/natl_buoys.pbs`, PBS job
7920689, 5 min on one core) compares the v2 and v3 runs with the moored
buoys of the Copernicus Marine in situ TAC: monthly files `MO/202601` of
`INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030` on the Datarmor mirror
(`/home/ref-copernicus-insitu`). Of 1,067 files, 449 carry `VHM0`. Samples
with flag 1 after the 3-day cold start are averaged per hour; a platform
distributed by several networks is kept once (same position to 0.01°); a
buoy is kept with at least 100 hours and a wet node within 10 km in every
run, so both runs use the same 267 buoys and 166,626 hours. The model is
taken at the nearest node, linear in time. 178 buoys are at nside 2048 in
both grids, 44 move from nside 1024 (v2) to 2048 (v3).

| subset | buoys | bias v2 / v3 (m) | RMSE v2 / v3 (m) | SI v2 / v3 | r v2 / v3 |
|---|---|---|---|---|---|
| all | 267 | −0.015 / −0.004 | 0.368 / 0.375 | 0.263 / 0.268 | 0.9538 / 0.9522 |
| v3 node at nside 2048 | 229 | −0.014 / +0.001 | 0.374 / 0.381 | 0.288 / 0.294 | 0.9475 / 0.9456 |
| v3 node at nside 1024 | 34 | −0.033 / −0.038 | 0.333 / 0.335 | 0.169 / 0.170 | 0.9671 / 0.9667 |
| depth < 50 m | 186 | −0.012 / +0.004 | 0.330 / 0.341 | 0.305 / 0.315 | 0.9261 / 0.9218 |
| depth 50–200 m | 58 | +0.002 / +0.002 | 0.459 / 0.457 | 0.225 / 0.224 | 0.9493 / 0.9500 |
| depth ≥ 200 m | 23 | −0.079 / −0.081 | 0.403 / 0.405 | 0.170 / 0.171 | 0.9747 / 0.9745 |
| North Sea | 95 | −0.050 / −0.052 | 0.288 / 0.286 | 0.285 / 0.283 | 0.9396 / 0.9419 |
| Channel and Biscay | 36 | +0.101 / +0.177 | 0.522 / 0.556 | 0.277 / 0.285 | 0.9452 / 0.9410 |
| Mediterranean | 35 | −0.108 / −0.111 | 0.440 / 0.441 | 0.240 / 0.240 | 0.9528 / 0.9528 |
| US and Canada coasts | 68 | −0.001 / −0.007 | 0.233 / 0.233 | 0.215 / 0.215 | 0.9476 / 0.9475 |

The buoy scatter index is twice the altimeter one: most buoys are in less
than 50 m of water (mean $H_s$ 1.1 m), where local wind, coastal sheltering
and nearest-node sampling of a 3 km cell dominate. Grid v3 brings the bias
closer to zero but not the RMSE, and every region but one is unchanged
within 0.01 m. On the south coast of England (Channel Coastal Observatory
buoys 6201007 Lyme Bay, 6201002 Torbay, 6201000 Mount's Bay, 6201008 Poole
Bay) v2 underestimates and v3 overestimates the westerly storms, by up to
1.5 m at 6201007: the shelter of these bays by the headlands is not
resolved by 3 km cells, and depends on the upstream cells (nside 1024
within 40 km in v2, none in v3). In Liverpool Bay (6201029) the RMSE falls
from 0.52 to 0.32 m. $T_{m02}$ (`VTM02`, 180 buoys) is 0.10 s too short in
both runs, RMSE 1.37 s (v2) and 1.33 s (v3). A coastal gain needs cells
finer than nside 2048 near the coast. Statistics per subset and per buoy:
`results/natl_v3_jan2026_cur_buoys_stats.json`.

```{figure} _static/fig/natl_v3_jan2026_cur_buoys_scatter.png
:width: 100%

Run v3 against 267 moored buoys (hourly $H_s$), and RMSE change per buoy
from v2 to v3.
```

```{figure} _static/fig/natl_v3_jan2026_cur_buoys_series.png
:width: 100%

Hourly $H_s$ at the four buoys where the RMSE changes most from v2 to v3.
```

## Open-boundary spectra

The January 2026 runs impose no spectra on the open boundaries (equator,
Barents Sea), hence the negative band along the equator. On Datarmor the
Ifremer global run `GLOBMULTI` (`/home/datawork-WW3/FORECAST/GLOBMULTI/
GLOB-30M/FIELD_NC/best_estimate/2026`, 0.5°, 3-hourly) writes the height,
peak period, direction and spread of six partitions (`phs0`–`phs5`,
`ptp*`, `pdir*`, `pspr*`), from which the boundary spectra can be rebuilt
for `ww3_bounc`. Its point spectra (`POINTS/SPEC_NC`) are given at buoy
positions only.

`scripts/boundary_spectra.py` (`scripts/natl_bnd_setup.pbs`, PBS job
7920771) takes, for each of the 1,341 open-boundary nodes of grid v3, the
nearest global cell with data (318 cells, at most 1.16° away), and sums
over the six partitions a JONSWAP spectrum (γ = 3.3, peak 1/Tp, scaled to
the partition height on the model frequencies) times a
$\cos^{2s}(\Delta\theta/2)$ spreading with $s = 2/\sigma^2 - 1$. The
rebuilt spectra give 0.99 of the global $H_s$ (mean 1.32 m on the boundary
cells, maximum 4.58 m). `ww3_bounc` (linear interpolation between the two
nearest cells) writes `nest.ww3` (249 times every 3 h, 366 MB) in 9 s. Run
`natl_v3_jan2026_bnd` (grid v3, same forcing as `natl_v3_jan2026_cur` plus
these spectra, PBS job 7920772, 68,799 s on 112 cores) is validated on the
same 328,902 altimeter blocks (PBS job 7920776) and 268 buoys (PBS job
7920777); without / with boundary spectra:

| subset | N | bias (m) | RMSE (m) | SI |
|---|---|---|---|---|
| all | 328,902 | +0.005 / +0.041 | 0.376 / 0.348 | 0.129 / 0.118 |
| nside 128 | 155,126 | +0.013 / +0.055 | 0.384 / 0.350 | 0.123 / 0.110 |
| nside 256 | 75,413 | +0.018 / +0.050 | 0.389 / 0.367 | 0.123 / 0.115 |
| nside 512 | 47,048 | −0.015 / +0.025 | 0.377 / 0.347 | 0.141 / 0.129 |
| nside 1024 | 30,714 | −0.019 / +0.002 | 0.350 / 0.333 | 0.147 / 0.140 |
| nside 2048 | 20,601 | −0.026 / −0.005 | 0.299 / 0.283 | 0.165 / 0.156 |
| North Sea | 5,629 | −0.139 / −0.139 | 0.381 / 0.381 | 0.123 / 0.123 |
| Mediterranean | 14,454 | −0.054 / −0.054 | 0.331 / 0.331 | 0.149 / 0.149 |
| buoys, hourly $H_s$ | 167,298 h | −0.004 / −0.004 | 0.375 / 0.375 | – |

The correlation rises from 0.9758 to 0.9794 and the regression slope is
1.000. All ten missions improve (RMSE lower by 0.015 to 0.038 m). The gain
is largest on day 4, when the swell entering through the equator is
missing from the cold start (RMSE 0.437 → 0.335 m, bias −0.19 → −0.13 m),
and stays 0.02–0.04 m during the storms of 26–31 January. The negative band
along the equator is gone. The North Sea, the Mediterranean and the buoys,
sheltered from the boundaries, do not change ($T_{m02}$ RMSE 1.325 s at
180 buoys in both runs). Statistics: `results/natl_v3_jan2026_bnd_val_stats.json`
and `results/natl_v3_jan2026_bnd_buoys_stats.json`.

```{figure} _static/fig/natl_v3_jan2026_bnd_maps.png
:width: 100%

Run with open-boundary spectra: mean and maximum $H_s$ in January 2026, and
WW3 − altimeter along tracks.
```

```{figure} _static/fig/natl_jan2026_scatter.png
:width: 100%

WW3 against altimeter $H_s$, and bias / RMSE by nside of the matched node.
```

```{figure} _static/fig/natl_jan2026_maps.png
:width: 100%

Mean and maximum $H_s$ in January 2026, and WW3 − altimeter along tracks.
```
