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
