//! Propagation model of the differentiable WAVEWATCH III (WW3), on one GPU,
//! for the HEALPix unstructured grids of HWW3.
//!
//! The crate rewrites in Rust the parts of WW3 that the HWW3 explicit
//! configuration runs, so that the model can sit inside a learning chain and
//! return gradients:
//!
//! - **Milestone M0**: the PDLIB explicit N scheme of the geographic
//!   propagation (`PDLIB_W3XYPFSN2`) and its transpose, as GPU kernels, with
//!   dot-product tests and a sub-step that is an operation of Burn's autodiff
//!   tape;
//! - **Milestone M1**: the propagation-only model (`hww3-prop`): geographic
//!   and spectral propagation (`W3KTP3`), currents, water levels, ice and
//!   open-boundary spectra, started from a WW3 restart;
//! - **Milestone M2a** (in progress): the source terms with the simplified
//!   ST4, checked node by node against the WW3 source-term harness
//!   (`harness/src_harness.F90`, read by [`harness`]): CPU transcriptions in
//!   [`sources`], the GPU kernel of the whole `W3SRCE` integration and the
//!   wind of `W3UWND` in `srce`, run after the propagation in the time loop
//!   of `model`.
//!
//! # Backends
//!
//! - `cubecl` (default feature): CubeCL kernels on the client of Burn's CUDA
//!   device (modules `kernels`, `prop`, `device`, `burn_ext`, `model`).
//!   Building it needs glibc 2.34 or later (`cubecl-llvm`).
//! - `cuda-c`: the M0 kernels in CUDA C, compiled by NVRTC and launched
//!   through `cudarc` (module `cuda_c`), for GPUs whose driver predates
//!   CUDA 12.
//!
//! The modules without a GPU ([`grid`], [`geom`], [`spec`], [`ww3io`],
//! [`mod@reference`], [`cpu`], [`harness`], [`sources`], [`backend`]) build with
//! `--no-default-features` and hold the CPU references the kernels are
//! checked against; their unit tests run without a GPU.
//!
//! The design, the cost and the results are described in the HWW3
//! documentation (pages *GPU wave model* and *Differentiable WW3*).

#![warn(missing_docs)]

pub mod backend;
pub mod bounc;
pub mod cpu;
pub mod geom;
pub mod grid;
pub mod harness;
pub mod harness2;
pub mod ice;
pub mod prnc;
pub mod ref1;
pub mod reference;
pub mod sources;
pub mod spec;
pub mod structgrid;
pub mod ww3grid;
pub mod ww3io;

#[cfg(feature = "cubecl")]
pub mod adjoint;
#[cfg(feature = "cubecl")]
pub mod burn_ext;
#[cfg(feature = "cubecl")]
pub mod burn_m4;
#[cfg(feature = "cubecl")]
pub mod device;
#[cfg(feature = "cubecl")]
pub mod kernels;
#[cfg(feature = "cubecl")]
pub mod model;
#[cfg(feature = "cubecl")]
pub mod prop;
#[cfg(feature = "cubecl")]
pub mod prop_tl;
#[cfg(feature = "cubecl")]
pub mod srce;
#[cfg(feature = "cubecl")]
pub mod srce_ice;
#[cfg(feature = "cubecl")]
pub mod srce_tl;

#[cfg(feature = "cuda-c")]
pub mod cuda_c;
#[cfg(all(feature = "cubecl", feature = "cuda-c"))]
pub mod cuda_m4;

#[cfg(test)]
mod testgrid;
