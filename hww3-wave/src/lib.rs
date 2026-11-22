//! Differentiable WAVEWATCH III on GPU. Milestone M0: the PDLIB explicit
//! N scheme of the geographic propagation and its transpose on HWW3 grids.
//! Milestone M1: the propagation-only model (geographic and spectral
//! propagation, currents, water levels, ice, boundary spectra, WW3 restart).
//!
//! Two backends share the geometry, the driver and the CPU reference:
//! - `cubecl` (default): CubeCL kernels on the client of Burn's CUDA device,
//!   with the sub-step as an operation of Burn's autodiff tape;
//! - `cuda-c`: the same kernels in CUDA C through `cudarc` and NVRTC.

pub mod backend;
pub mod cpu;
pub mod geom;
pub mod grid;
pub mod reference;
pub mod spec;
pub mod ww3io;

#[cfg(feature = "cubecl")]
pub mod burn_ext;
#[cfg(feature = "cubecl")]
pub mod device;
#[cfg(feature = "cubecl")]
pub mod kernels;
#[cfg(feature = "cubecl")]
pub mod model;
#[cfg(feature = "cubecl")]
pub mod prop;

#[cfg(feature = "cuda-c")]
pub mod cuda_c;
