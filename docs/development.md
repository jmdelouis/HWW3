# Tests and continuous integration

Every push to `main` and every pull request runs two GitHub Actions
workflows.

| Workflow | Job | What it runs |
|---|---|---|
| `CI` (`.github/workflows/ci.yml`) | `rust` | `hww3-grid`: build, unit and end-to-end tests, documentation with warnings as errors |
| | `wave` | `hww3-wave`: unit tests of the CPU modules, type check of the GPU backends, documentation with warnings as errors |
| | `python` | byte-compilation of every script in `scripts/` |
| `Documentation` (`.github/workflows/docs.yml`) | `build-docs` | this site with `sphinx-build -W` (warnings are errors), and the Rust API references of both crates |
| | `deploy` | publication on GitHub Pages (pushes to `main` only) |

## Running the checks locally

```bash
# grid generator
cd hww3-grid
cargo test --release
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

# GPU wave model, without a GPU
cd ../hww3-wave
cargo test --no-default-features

# this site
pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
```

## `hww3-wave` without a GPU

The GitHub runners have no GPU, no CUDA toolkit and no LLVM bundle, so the
`wave` job runs what does not need them:

- **Unit tests** of the modules without a GPU (`--no-default-features`).
- **Type check** of the CubeCL and CUDA C backends
  (`cargo check --all-targets --features cuda-c`) and their documentation.
  Two environment variables let the build scripts run without the GPU
  toolchain:
  - `CUDARC_CUDA_VERSION=13000` gives `cudarc` the CUDA version that
    `nvcc` would give;
  - `DOCS_RS=1` makes `cubecl-llvm` skip compiling its LLVM shims, which
    only the final link needs.

  Nothing is linked, so no GPU binary is built in the CI.

The GPU kernels themselves are checked on the A100s of the hpc cluster by
the M0 and M1 drivers: dot-product tests, comparison with the CPU loop of
WW3, `check-ktp3`, `check-geo`, and the runs against WW3 (see
[GPU model results](wave_results.md)).

### Unit tests of `hww3-wave`

They run on a small synthetic grid (`testgrid.rs`): a 12 × 10 lattice of
0.05° cells at 45° N, cut into triangles along alternating diagonals, some
listed clockwise, with the depth increasing eastwards from 20 m and an open
western boundary.

| Test | What it checks |
|---|---|
| `geom::group_speed_limits` | the group speed tends to $g/(2\omega)$ in deep water and $\sqrt{g h}$ in shallow water |
| `geom::triangles_areas_and_normals` | triangles reoriented counter-clockwise; $\sum$ `PDLIB_SI` and $\sum$ areas equal the lattice area; the three edge normals of a triangle sum to zero and point inwards; the node → triangle table lists each corner once |
| `geom::boundary_flags` | `IOBP` = 2 on the open boundary, 0 on the coast, 1 inside; `IOBPA`, `IOBDP` and the masks |
| `geom::coast_node_takes_waves_from_the_sea_only` | `IOBPD` on a southern coast node keeps the directions with $\sin\theta \le 0$, on an eastern one those with $\cos\theta \ge 0$ |
| `geom::dry_node_is_masked` | a node shallower than `DMIN` has `IOBDP` = 0, an empty mask and zero velocities |
| `geom::sub_cycling_keeps_the_cfl_of_every_wet_node_below_one` | with $\Delta t / \mathrm{ITER}$, the CFL number of every wet node (coast and open boundary included, as WW3 with `FSBCCFL = T`) and component is at most 1 |
| `harness::harness_files_round_trip` | the harness parameter file (lower bounds, 2-D tables, kinds) and output file (record layout of every call) built in memory |
| `geom::host_arrays_pack_the_masks` | the mask words and time steps handed to the kernels |
| `spec::spectral_grid` | frequencies, bandwidths and direction cosines of `W3GRID` |
| `spec::wavnu3_follows_the_dispersion_relation` | `WAVNU3` within 0.1 % of the dispersion relation and of the exact group speed, 1 m to 5,000 m, 0.037 to 0.7 Hz |
| `spec::node_wavenumbers_keep_deep_nodes_at_the_bathymetric_depth` | the deep-water rule of `W3INIT` and `W3ULEV` |
| `reference::sub_steps_reproduce_the_reference_step` | a CPU backend of the `Sweeper` trait, sub-step by sub-step, reproduces `PDLIB_W3XYPFSN2` to 1e-13 |
| `reference::transpose_passes_the_dot_product_test` | the transpose formula of the kernels, sub-step and 600 s step, to 1e-12 |
| `reference::uniform_state_stays_uniform_with_constant_velocity` | consistency of the N scheme: $\mathrm{NM}\sum\mathrm{FLALL} = 1$ |
| `reference::step_is_linear_without_the_clip` | linearity of the step without `max(0, ·)` |
| `cpu::ktp3_leaves_deep_still_water_unchanged` | `W3KTP3` without depth gradient or current changes nothing |
| `cpu::refraction_conserves_the_energy_of_each_frequency` | refraction in 15 m of water turns the spectrum and keeps the sum over directions to 1e-5 |
| `cpu::geographic_step_matches_the_reference` | the transcription of `PDLIB_W3XYPUG` (single-precision state) against `PDLIB_W3XYPFSN2` in double precision, `ITER` identical |
| `grid::boundary_of_a_lattice` | boundary nodes from the edges used once |
| `grid::read_raw_arrays_and_namelist` | `raw/*` arrays and `ww3_grid.nml` read back |
| `grid::cut_box_opens_the_cut_edges` | a box cut opens the cut edges and keeps the open boundary of the full grid |
| `ww3io::sequential_records` | Fortran records with big-endian markers; differing markers are an error |
| `ww3io::forcing_file` | a `ww3_prnc` current file with two times |
| `ww3io::cold_and_full_restarts` | `COLD` and `FULL` restarts in the `W3IORS` layout, `MAPSTA` and `MAPST2` decoded |
| `ww3io::boundary_file` | a `ww3_bounc` file: points, column-major `IPBPI` and `RDBPI`, spectra |
| `ww3io::time_parse_and_display`, `ww3io::time_roundtrip` | WW3 times |

The tests of `hww3-grid` are listed in [Validation](validation.md).

## Documentation

The site is written in Markdown (MyST) and built by Sphinx with the PyData
theme. The figures live once, in `tex/hww3/fig`, shared with the LaTeX
note, and are copied next to the pages at build time. The Rust API
references are generated by `rustdoc` and published with the site:
<a href="rust/hww3_grid/index.html">hww3-grid</a> and
<a href="rust-wave/hww3_wave/index.html">hww3-wave</a>. Both crates warn
on any public item without documentation (`#![warn(missing_docs)]`), and
the CI builds their documentation with warnings as errors.
