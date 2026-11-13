# HWW3: HEALPix unstructured grids for WAVEWATCH III

HWW3 builds unstructured (UNST) WAVEWATCH III grids whose nodes are the centres
of multi-resolution HEALPix cells. Each node is a NESTED HEALPix pixel, so the
model output is already a HEALPix map: you can write it as a GRID4EARTH
Zarr V3 dataset and view it in gridlook without any interpolation.

The WAVEWATCH III code is not modified. The grid is a standard Gmsh triangle
mesh read by `ww3_grid` (`GRID%TYPE = 'UNST'`) and run with the implicit
PDLIB scheme.

## How the grid is built (`hww3-grid`, Rust)

1. **Refinement.** Start from HEALPix cells at `nside_min` covering the domain.
   A cell is split into its four NESTED children when it is closer to the coast
   than a rule's `max_dist_km` and is coarser than that rule's `nside`. The
   coastline comes from a shapefile (`"source": "shapefile"`, e.g. GSHHS
   polylines), from a native GSHHG binary file (`"gshhg"`), or from the zero
   contour of the bathymetry (`"bathy"`).
   There are at most 4 levels between `nside_min` and `nside_max`.
2. **2:1 balance.** Leaves are split until any two leaves sharing an edge or a
   corner differ by at most one level.
3. **Bathymetry.** Each leaf gets the mean of the GEBCO samples inside it,
   plus the fraction of those samples that are wet. A leaf is a wet node if
   that fraction is ≥ `wet_fraction` and its mean wet depth is ≥ `min_depth_m`.
   Only the largest connected wet region is kept.
4. **Triangulation from the HEALPix adjacency.** Every leaf registers its
   4 corners and 4 edge midpoints. Around each such point, the wet leaves
   touching it, sorted by angle, form a polygon of cell centres. That polygon
   is triangulated by ear clipping. Two cells that share a boundary segment
   are joined by an edge that belongs only to the polygons at the two ends of
   that segment, so the mesh is conforming, including at coarse/fine
   transitions. Pinch (bow-tie) nodes, which WW3 rejects, are removed.
5. **Triangle quality.** The nodes are fixed, so angles are improved by the
   connectivity only: Delaunay (max-min angle) edge flips inside the mesh,
   then removal of boundary triangles whose angles are outside
   `quality.min_angle_deg`–`quality.max_angle_deg` (20°–130° by default).
   With `quality.enforce`, the program fails if an angle is still out of range.
6. **Ellipsoid.** HEALPix is defined on the authalic sphere. With
   `"ellipsoid": "wgs84"`, the default, input latitudes are converted to
   authalic latitudes before hashing, and node latitudes are converted back to
   geodetic latitudes for WW3. Use `"sphere"` to skip the conversion.
7. **Outputs** (written to `output_dir`):
   - `<name>.msh`: Gmsh 2.2 file with depth positive down;
   - `<name>.obc`: open-boundary nodes;
   - `ww3_grid.nml` and `namelists_<name>.nml`;
   - `raw/*`: little-endian arrays (`uniq`, `level`, `lon`, `lat`, `depth`,
     `tri`, `obc`);
   - `meta.json`: configuration plus mesh report.

The mesh is checked automatically: no edge shared by more than two triangles,
consistent counter-clockwise orientation, no pinch node, Euler
characteristic, angle statistics and level jump across edges.

## Documentation

- Sphinx site (`docs/`, built and published by the CI on GitHub Pages):
  installation, configuration, algorithm, outputs, validation, WW3 setup and
  a step-by-step [reproduction guide](docs/reproducing.md).
- Rust API: `cd hww3-grid && cargo doc --no-deps --open`.
- Internal note (LaTeX): `tex/hww3/main.tex`, mirrored on Overleaf
  (https://www.overleaf.com/project/6af6a96c4ed66a0cad905cb6) with
  `scripts/sync_overleaf.sh`.

## Usage

```bash
cd hww3-grid
cargo test --release
cargo run --release -- ../configs/natl.json
```

To prepare the bathymetry:

```bash
python scripts/prep_gebco.py GEBCO_2024.nc data/gebco/natl -100 45 0 80
```

## Repository layout

| Path | Content |
|---|---|
| `hww3-grid/` | Rust crate: grid generator |
| `configs/` | grid configurations (`natl.json`: North Atlantic + Mediterranean + North Sea) |
| `scripts/` | data preparation and plotting |
| `tex/hww3/` | internal note (LaTeX) |

## License

Apache-2.0
