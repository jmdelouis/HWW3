//! # hww3-grid
//!
//! Build multi-resolution HEALPix unstructured grids for WAVEWATCH III.
//!
//! Every node of the grid is the centre of a NESTED HEALPix cell. Cells are
//! refined from the bathymetry, as in the classical wave-model grid
//! generators: the cell size follows `sqrt(depth)` and the topographic length
//! scale `depth/|grad depth|` (`nside_min` in the deep ocean up to
//! `nside_max` in shallow water and over steep slopes), so the WW3 output on the nodes is already a
//! multi-resolution HEALPix map that can be written as a GRID4EARTH Zarr
//! dataset without interpolation. WAVEWATCH III itself is not modified: the
//! grid is a standard Gmsh triangle mesh read by `ww3_grid` with
//! `GRID%TYPE = 'UNST'`.
//!
//! ## Pipeline
//!
//! [`pipeline::build`] runs the following steps:
//!
//! 1. [`refine`]: the bathymetry, averaged on blocks, gives the cell size
//!    required by the depth criterion `T·sqrt(g h)` and the slope criterion
//!    `2π h/(N |∇h|)`, hence the cells to split;
//! 2. [`tree`]: adaptive refinement of NESTED HEALPix cells driven by these
//!    splits (and optionally by the distance to a [`coast`]line from a
//!    [`shapefile`], a [`gshhg`] file or the bathymetry zero contour),
//!    followed by a 2:1 balance between neighbours;
//! 3. [`bathy`]: cell-averaged depth and wet fraction of every leaf;
//! 4. [`mesh`]: conforming triangulation of the wet cell centres built from
//!    the HEALPix adjacency around every cell corner and edge midpoint, then
//!    removal of pinch (bow-tie) nodes that WW3 rejects;
//! 5. [`validate`]: topological and geometric checks of the mesh;
//! 6. [`ww3`]: Gmsh mesh, open-boundary list, namelists and raw arrays
//!    (written by [`pipeline::write`]).
//!
//! [`geo`] holds the ellipsoid handling: HEALPix lives on the authalic
//! sphere, and geodetic latitudes are converted to and from authalic
//! latitudes with [`geo::Earth`].
//!
//! ## Example
//!
//! ```no_run
//! use hww3_grid::{config::Config, pipeline};
//! use std::path::Path;
//!
//! let cfg = Config::load(Path::new("configs/natl.json"))?;
//! let (grid, summary) = pipeline::build(&cfg)?;
//! assert!(summary.mesh.is_valid());
//! pipeline::write(&cfg, &grid, &summary)?;
//! # Ok::<(), anyhow::Error>(())
//! ```

#![warn(missing_docs)]

pub mod bathy;
pub mod coast;
pub mod config;
pub mod geo;
pub mod gshhg;
pub mod mesh;
pub mod pipeline;
pub mod refine;
pub mod shapefile;
pub mod tree;
pub mod validate;
pub mod ww3;
