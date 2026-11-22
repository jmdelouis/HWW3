//! CubeCL backend: device copy of a [`Geometry`] and launchers of the
//! kernels of [`crate::kernels`].

use crate::backend::{Dims, Scalar, Sweeper};
use crate::geom::Geometry;
use crate::kernels;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;
use std::marker::PhantomData;

/// Float type of the CubeCL kernels: `f32` or `f64`.
pub trait Real: Scalar + Float + CubeElement {}
impl Real for f32 {}
impl Real for f64 {}

/// A [`Geometry`] on the GPU, with the launchers of the M0 kernels; the
/// CubeCL implementation of [`Sweeper`].
pub struct DeviceGrid<F: Real> {
    /// CubeCL client of the device (from Burn's CUDA device).
    pub client: Client,
    /// Number of nodes.
    pub np: usize,
    /// Number of frequencies.
    pub nk: usize,
    /// Number of directions.
    pub nth: usize,
    /// Number of triangles.
    pub ne: usize,
    /// Units per cube (1D launches).
    pub block: u32,
    /// Kernel variant: 0 fused node gather, 1 two-phase, 2 two-phase with
    /// four directions per unit.
    pub variant: usize,
    /// `ITER(ik, ith)`.
    pub iter: Vec<u32>,
    /// Time of the twelve coefficients per triangle and frequency (s).
    pub coef_seconds: f64,
    cx: Handle,
    cy: Handle,
    ecos: Handle,
    esin: Handle,
    dtk: Handle,
    iter_d: Handle,
    si_inv: Handle,
    mlo: Handle,
    mhi: Handle,
    start: Handle,
    slot: Handle,
    ine: Handle,
    ien: Handle,
    /// `nk ne 12` coefficients of the two-phase kernels.
    coef: Handle,
    /// `ne nth` values per triangle and direction (`Ũ_T`, or its transposed counterpart).
    scratch: Handle,
    _f: PhantomData<F>,
}

fn bf(h: &Handle, n: usize) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(h.clone(), n) }
}

impl<F: Real> DeviceGrid<F> {
    /// Copy a geometry to the device and compute the two-phase coefficients.
    pub fn new(client: Client, g: &Geometry, block: u32) -> Self {
        let upf = |v: &[f64]| {
            let w: Vec<F> = v.iter().map(|&x| <F as Scalar>::of_f64(x)).collect();
            client.create_from_slice(F::as_bytes(&w))
        };
        let upu = |v: &[u32]| client.create_from_slice(u32::as_bytes(v));
        let (si_inv, dtk, mlo, mhi) = g.host_arrays();
        let fb = std::mem::size_of::<F>();
        let mut d = DeviceGrid {
            np: g.np,
            nk: g.nk,
            nth: g.nth,
            ne: g.ne,
            block,
            variant: 0,
            iter: g.iter.clone(),
            coef_seconds: 0.0,
            cx: upf(&g.cx),
            cy: upf(&g.cy),
            ecos: upf(&g.ecos),
            esin: upf(&g.esin),
            dtk: upf(&dtk),
            iter_d: upu(&g.iter),
            si_inv: upf(&si_inv),
            mlo: upu(&mlo),
            mhi: upu(&mhi),
            start: upu(&g.start),
            slot: upu(&g.slot),
            ine: upu(&g.ine),
            ien: upf(&g.ien),
            coef: client.empty(g.nk * g.ne * 12 * fb),
            scratch: client.empty(g.ne * g.nth * fb),
            client,
            _f: PhantomData,
        };
        d.sync();
        let t = std::time::Instant::now();
        for ik in 0..d.nk {
            d.compute_coef(ik);
        }
        d.sync();
        d.coef_seconds = t.elapsed().as_secs_f64();
        d
    }

    fn dims_for(&self, n: usize) -> (CubeCount, CubeDim) {
        (CubeCount::Static((n as u32).div_ceil(self.block), 1, 1), CubeDim::new_1d(self.block))
    }

    /// Coefficients of frequency `ik` (velocities fixed during a step).
    pub fn compute_coef(&self, ik: usize) {
        let (count, dim) = self.dims_for(self.ne);
        let (np, ne, nk) = (self.np, self.ne, self.nk);
        unsafe {
            kernels::nscheme_coef::launch_unchecked::<F>(
                &self.client,
                count,
                dim,
                bf(&self.cx, nk * np),
                bf(&self.cy, nk * np),
                bf(&self.ine, 3 * ne),
                bf(&self.ien, 6 * ne),
                bf(&self.coef, nk * ne * 12),
                np,
                ne,
                ik,
            );
        }
    }

    fn forward_fused(&self, u: &Handle, out: &Handle, ik: usize, sub: u32, clip: bool) {
        let (np, ne, nk, nth) = (self.np, self.ne, self.nk, self.nth);
        let n = np * nth;
        let (count, dim) = self.dims_for(n);
        unsafe {
            kernels::nscheme_forward::launch_unchecked::<F>(
                &self.client,
                count,
                dim,
                bf(u, n),
                bf(out, n),
                bf(&self.cx, nk * np),
                bf(&self.cy, nk * np),
                bf(&self.ecos, nth),
                bf(&self.esin, nth),
                bf(&self.dtk, nk * nth),
                bf(&self.iter_d, nk * nth),
                bf(&self.si_inv, np),
                bf(&self.mlo, np),
                bf(&self.mhi, np),
                bf(&self.start, np + 1),
                bf(&self.slot, 3 * ne),
                bf(&self.ine, 3 * ne),
                bf(&self.ien, 6 * ne),
                np,
                ik,
                sub,
                nth,
                clip,
            );
        }
    }

    fn transpose_fused(&self, y: &Handle, out: &Handle, ik: usize, sub: u32) {
        let (np, ne, nk, nth) = (self.np, self.ne, self.nk, self.nth);
        let n = np * nth;
        let (count, dim) = self.dims_for(n);
        unsafe {
            kernels::nscheme_transpose::launch_unchecked::<F>(
                &self.client,
                count,
                dim,
                bf(y, n),
                bf(out, n),
                bf(&self.cx, nk * np),
                bf(&self.cy, nk * np),
                bf(&self.ecos, nth),
                bf(&self.esin, nth),
                bf(&self.dtk, nk * nth),
                bf(&self.iter_d, nk * nth),
                bf(&self.si_inv, np),
                bf(&self.mlo, np),
                bf(&self.mhi, np),
                bf(&self.start, np + 1),
                bf(&self.slot, 3 * ne),
                bf(&self.ine, 3 * ne),
                bf(&self.ien, 6 * ne),
                np,
                ik,
                sub,
                nth,
            );
        }
    }

    fn forward_two_phase(&self, u: &Handle, out: &Handle, ik: usize, sub: u32, clip: bool) {
        let (np, ne, nk, nth) = (self.np, self.ne, self.nk, self.nth);
        let (n, m) = (np * nth, ne * nth);
        let (count, dim) = self.dims_for(m);
        unsafe {
            kernels::nscheme_tri_forward::launch_unchecked::<F>(
                &self.client,
                count,
                dim,
                bf(u, n),
                bf(&self.scratch, m),
                bf(&self.coef, nk * ne * 12),
                bf(&self.ecos, nth),
                bf(&self.esin, nth),
                bf(&self.iter_d, nk * nth),
                bf(&self.ine, 3 * ne),
                ne,
                ik,
                sub,
                nth,
            );
        }
        let (count, dim) = self.dims_for(n);
        unsafe {
            kernels::nscheme_node_forward::launch_unchecked::<F>(
                &self.client,
                count,
                dim,
                bf(u, n),
                bf(out, n),
                bf(&self.scratch, m),
                bf(&self.coef, nk * ne * 12),
                bf(&self.ecos, nth),
                bf(&self.esin, nth),
                bf(&self.dtk, nk * nth),
                bf(&self.iter_d, nk * nth),
                bf(&self.si_inv, np),
                bf(&self.mlo, np),
                bf(&self.mhi, np),
                bf(&self.start, np + 1),
                bf(&self.slot, 3 * ne),
                np,
                ne,
                ik,
                sub,
                nth,
                clip,
            );
        }
    }

    fn transpose_two_phase(&self, y: &Handle, out: &Handle, ik: usize, sub: u32) {
        let (np, ne, nk, nth) = (self.np, self.ne, self.nk, self.nth);
        let (n, m) = (np * nth, ne * nth);
        let (count, dim) = self.dims_for(m);
        unsafe {
            kernels::nscheme_tri_transpose::launch_unchecked::<F>(
                &self.client,
                count,
                dim,
                bf(y, n),
                bf(&self.scratch, m),
                bf(&self.coef, nk * ne * 12),
                bf(&self.ecos, nth),
                bf(&self.esin, nth),
                bf(&self.dtk, nk * nth),
                bf(&self.iter_d, nk * nth),
                bf(&self.si_inv, np),
                bf(&self.mlo, np),
                bf(&self.mhi, np),
                bf(&self.ine, 3 * ne),
                ne,
                ik,
                sub,
                nth,
            );
        }
        let (count, dim) = self.dims_for(n);
        unsafe {
            kernels::nscheme_node_transpose::launch_unchecked::<F>(
                &self.client,
                count,
                dim,
                bf(y, n),
                bf(out, n),
                bf(&self.scratch, m),
                bf(&self.coef, nk * ne * 12),
                bf(&self.ecos, nth),
                bf(&self.esin, nth),
                bf(&self.dtk, nk * nth),
                bf(&self.iter_d, nk * nth),
                bf(&self.si_inv, np),
                bf(&self.mlo, np),
                bf(&self.mhi, np),
                bf(&self.start, np + 1),
                bf(&self.slot, 3 * ne),
                np,
                ne,
                ik,
                sub,
                nth,
            );
        }
    }

    fn forward_x4(&self, u: &Handle, out: &Handle, ik: usize, sub: u32, clip: bool) {
        let (np, ne, nk, nth) = (self.np, self.ne, self.nk, self.nth);
        let (n, m) = (np * nth, ne * nth);
        let (count, dim) = self.dims_for(m / 4);
        unsafe {
            kernels::nscheme_tri_forward_x4::launch_unchecked::<F>(
                &self.client,
                count,
                dim,
                bf(u, n),
                bf(&self.scratch, m),
                bf(&self.coef, nk * ne * 12),
                bf(&self.ecos, nth),
                bf(&self.esin, nth),
                bf(&self.iter_d, nk * nth),
                bf(&self.ine, 3 * ne),
                ne,
                ik,
                sub,
                nth,
            );
        }
        let (count, dim) = self.dims_for(n / 4);
        unsafe {
            kernels::nscheme_node_forward_x4::launch_unchecked::<F>(
                &self.client,
                count,
                dim,
                bf(u, n),
                bf(out, n),
                bf(&self.scratch, m),
                bf(&self.coef, nk * ne * 12),
                bf(&self.ecos, nth),
                bf(&self.esin, nth),
                bf(&self.dtk, nk * nth),
                bf(&self.iter_d, nk * nth),
                bf(&self.si_inv, np),
                bf(&self.mlo, np),
                bf(&self.mhi, np),
                bf(&self.start, np + 1),
                bf(&self.slot, 3 * ne),
                np,
                ne,
                ik,
                sub,
                nth,
                clip,
            );
        }
    }

    fn transpose_x4(&self, y: &Handle, out: &Handle, ik: usize, sub: u32) {
        let (np, ne, nk, nth) = (self.np, self.ne, self.nk, self.nth);
        let (n, m) = (np * nth, ne * nth);
        let (count, dim) = self.dims_for(m / 4);
        unsafe {
            kernels::nscheme_tri_transpose_x4::launch_unchecked::<F>(
                &self.client,
                count,
                dim,
                bf(y, n),
                bf(&self.scratch, m),
                bf(&self.coef, nk * ne * 12),
                bf(&self.ecos, nth),
                bf(&self.esin, nth),
                bf(&self.dtk, nk * nth),
                bf(&self.iter_d, nk * nth),
                bf(&self.si_inv, np),
                bf(&self.mlo, np),
                bf(&self.mhi, np),
                bf(&self.ine, 3 * ne),
                ne,
                ik,
                sub,
                nth,
            );
        }
        let (count, dim) = self.dims_for(n / 4);
        unsafe {
            kernels::nscheme_node_transpose_x4::launch_unchecked::<F>(
                &self.client,
                count,
                dim,
                bf(y, n),
                bf(out, n),
                bf(&self.scratch, m),
                bf(&self.coef, nk * ne * 12),
                bf(&self.ecos, nth),
                bf(&self.esin, nth),
                bf(&self.dtk, nk * nth),
                bf(&self.iter_d, nk * nth),
                bf(&self.si_inv, np),
                bf(&self.mlo, np),
                bf(&self.mhi, np),
                bf(&self.start, np + 1),
                bf(&self.slot, 3 * ne),
                np,
                ne,
                ik,
                sub,
                nth,
            );
        }
    }
}

impl<F: Real> Sweeper for DeviceGrid<F> {
    type Buf = Handle;

    fn label(&self) -> String {
        format!("cubecl {} {}", <F as Scalar>::NAME, self.variants()[self.variant])
    }
    fn float_bytes(&self) -> usize {
        std::mem::size_of::<F>()
    }
    fn dims(&self) -> Dims {
        Dims { np: self.np, ne: self.ne, nk: self.nk, nth: self.nth }
    }
    fn iter(&self) -> &[u32] {
        &self.iter
    }
    fn block(&self) -> u32 {
        self.block
    }
    fn set_block(&mut self, b: u32) {
        self.block = b;
    }
    fn variants(&self) -> &'static [&'static str] {
        if self.nth % 4 == 0 {
            &["fused", "two_phase", "two_phase_x4"]
        } else {
            &["fused", "two_phase"]
        }
    }
    fn variant(&self) -> usize {
        self.variant
    }
    fn set_variant(&mut self, v: usize) {
        self.variant = v;
    }
    fn coef_seconds(&self) -> f64 {
        self.coef_seconds
    }

    fn upload(&self, v: &[f64]) -> Handle {
        let w: Vec<F> = v.iter().map(|&x| <F as Scalar>::of_f64(x)).collect();
        self.client.create_from_slice(F::as_bytes(&w))
    }

    fn download(&self, h: &Handle) -> Vec<f64> {
        let b = self.client.read_one(h.clone()).expect("device read");
        F::from_bytes(&b).iter().map(|&x| <F as Scalar>::as_f64(x)).collect()
    }

    fn empty(&self) -> Handle {
        self.client.empty(self.np * self.nth * std::mem::size_of::<F>())
    }

    fn sync(&self) {
        futures_lite::future::block_on(self.client.sync()).expect("device sync");
    }

    fn forward(&self, u: &Handle, out: &mut Handle, ik: usize, sub: u32, clip: bool) {
        match self.variant {
            0 => self.forward_fused(u, out, ik, sub, clip),
            1 => self.forward_two_phase(u, out, ik, sub, clip),
            _ => self.forward_x4(u, out, ik, sub, clip),
        }
    }

    fn transpose(&self, y: &Handle, out: &mut Handle, ik: usize, sub: u32) {
        match self.variant {
            0 => self.transpose_fused(y, out, ik, sub),
            1 => self.transpose_two_phase(y, out, ik, sub),
            _ => self.transpose_x4(y, out, ik, sub),
        }
    }
}
