//! hww3-grid: build multi-resolution HEALPix unstructured grids for WAVEWATCH III.
//!
//! Pipeline:
//! 1. `tree`     – adaptive refinement of NESTED HEALPix cells driven by the
//!                 distance to the coast, followed by a 2:1 balance.
//! 2. `bathy`    – cell-averaged bathymetry and wet/dry mask.
//! 3. `mesh`     – triangulation whose nodes are the HEALPix cell centres,
//!                 built from the cell adjacency around every cell corner.
//! 4. `validate` – topological and geometric checks of the mesh.
//! 5. `ww3`      – Gmsh / namelist / raw-array writers.

pub mod bathy;
pub mod coast;
pub mod config;
pub mod geo;
pub mod gshhg;
pub mod mesh;
pub mod pipeline;
pub mod tree;
pub mod validate;
pub mod ww3;
