//! Milestone M4: the adjoint of the propagation in CUDA C. The kernels of
//! `src/cuda/w3prop.cu` (the values of the refraction, the frequency shift
//! and the geographic sub-steps, recomputed inside a step, and their
//! adjoints, the transposes of the tangent kernels of `hww3_wave::prop_tl`)
//! are launched on raw device pointers, so that they run on the buffers of
//! the CubeCL model (CubeCL and `cudarc` share the primary context of the
//! device) as well as on `cudarc` allocations.

use anyhow::{anyhow, Result};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

/// The CUDA C propagation kernels and their adjoints.
pub const W3PROP: &str = include_str!("cuda/w3prop.cu");

/// Threads per block of the propagation kernels.
const GB: u32 = 128;

/// The device arrays of the refraction and of the frequency shift (raw
/// pointers, layouts of the CubeCL kernels of `hww3_wave::prop`) and their
/// scalars.
#[derive(Clone, Copy, Debug)]
pub struct Ktp3Ptrs {
    /// Wavenumbers `[j np + i]`, `j = 0 .. nk + 1`.
    pub wn: u64,
    /// Group velocities, as `wn`.
    pub cg: u64,
    /// Radian frequencies (`nk + 2`).
    pub sig: u64,
    /// Widths of the frequency bins (`nk + 2`).
    pub dsip: u64,
    /// Depths.
    pub dw: u64,
    /// Depth gradients.
    pub dddx: u64,
    /// See [`Ktp3Ptrs::dddx`].
    pub dddy: u64,
    /// Currents.
    pub cx: u64,
    /// See [`Ktp3Ptrs::cx`].
    pub cy: u64,
    /// Current gradients.
    pub dcxdx: u64,
    /// See [`Ktp3Ptrs::dcxdx`].
    pub dcxdy: u64,
    /// See [`Ktp3Ptrs::dcxdx`].
    pub dcydx: u64,
    /// See [`Ktp3Ptrs::dcxdx`].
    pub dcydy: u64,
    /// The great-circle term of the refraction.
    pub cthg0: u64,
    /// Node flags.
    pub nflag: u64,
    /// `cos`, `sin`, `sin^2`, `cos^2`, `sin cos` of the directions.
    pub ecos: u64,
    /// See [`Ktp3Ptrs::ecos`].
    pub esin: u64,
    /// See [`Ktp3Ptrs::ecos`].
    pub es2: u64,
    /// See [`Ktp3Ptrs::ecos`].
    pub ec2: u64,
    /// See [`Ktp3Ptrs::ecos`].
    pub esc: u64,
    /// `CTMAX`.
    pub ctmax: f64,
    /// 1 with currents, 0 without.
    pub flcur: f64,
    /// `FACHFA`.
    pub fachfa: f64,
    /// `DMIN`.
    pub dmin: f64,
    /// Nodes.
    pub np: u32,
}

/// The device arrays of the geographic sub-steps (raw pointers, layouts of
/// the CubeCL kernels `geo_tri_x4` and `geo_node_x4`) and their scalars.
#[derive(Clone, Copy, Debug)]
pub struct GeoPtrs {
    /// Triangle coefficients `[(ik ne + t) 12 + c]`.
    pub coef: u64,
    /// Current coefficients `[6 t + c]`.
    pub ccoef: u64,
    /// `cos`, `sin` of the directions.
    pub ecos: u64,
    /// See [`GeoPtrs::ecos`].
    pub esin: u64,
    /// Sub-step lengths `[ik nth + th]`.
    pub dtk: u64,
    /// Sub-step counts `[ik nth + th]`.
    pub iter: u64,
    /// Inverse of the node areas.
    pub si_inv: u64,
    /// Masks of the directions (two words per node).
    pub mlo: u64,
    /// See [`GeoPtrs::mlo`].
    pub mhi: u64,
    /// The triangles around each node (start, slot).
    pub start: u64,
    /// See [`GeoPtrs::start`].
    pub slot: u64,
    /// Open-boundary index of each node (`u32::MAX`: none).
    pub bidx: u64,
    /// Triangle nodes.
    pub ine: u64,
    /// Boundary spectra at the start and end of the step.
    pub bb0: u64,
    /// See [`GeoPtrs::bb0`].
    pub bbn: u64,
    /// `CLATS`.
    pub clats: u64,
    /// Boundary interpolation weights and the global step.
    pub rd10: f64,
    /// See [`GeoPtrs::rd10`].
    pub rd20: f64,
    /// See [`GeoPtrs::rd10`].
    pub dtg: f64,
    /// Nodes.
    pub np: u32,
    /// Triangles.
    pub ne: u32,
}

/// The propagation kernels on a `cudarc` context.
pub struct PropCuda {
    /// The context (the primary context of device 0).
    pub ctx: Arc<CudaContext>,
    /// Its stream.
    pub st: Arc<CudaStream>,
    f: Vec<(&'static str, CudaFunction)>,
    /// Frequencies and directions.
    pub nk: usize,
    /// See [`PropCuda::nk`].
    pub nth: usize,
}

impl PropCuda {
    /// Compile the kernels for `nk` frequencies and `nth` directions.
    pub fn new(nk: usize, nth: usize) -> Result<PropCuda> {
        if nth % 4 != 0 || nk < 3 {
            return Err(anyhow!("the geographic kernels need NTH a multiple of 4 and NK >= 3"));
        }
        let defs = vec![format!("-DNK={nk}u"), format!("-DNTH={nth}u"), format!("-DGB={GB}")];
        let ctx = CudaContext::new(0)?;
        let opts = CompileOptions { arch: Some("compute_80"), options: defs, ..Default::default() };
        let ptx = compile_ptx_with_opts(W3PROP, opts).map_err(|e| anyhow!("NVRTC: {e:?}"))?;
        let module = ctx.load_module(ptx)?;
        let names = [
            "ktp3_theta_fwd",
            "ktp3_theta_adj",
            "ktp3_k_fwd",
            "ktp3_k_adj",
            "va_to_ac",
            "ac_to_va_adj",
            "va_to_ac_adj",
            "geo_tri",
            "geo_node",
            "geo_adj_node1",
            "geo_adj_tri",
            "geo_adj_node2",
            "zero_nodes_adj",
            "outputs_adj",
            "ulev_node_adj",
            "ulev_remap_adj",
        ];
        let mut f = vec![];
        for n in names {
            f.push((n, module.load_function(n)?));
        }
        let st = ctx.default_stream();
        Ok(PropCuda { ctx, st, f, nk, nth })
    }

    fn func(&self, name: &str) -> &CudaFunction {
        &self.f.iter().find(|(n, _)| *n == name).expect("kernel").1
    }

    fn cfg(n: usize) -> LaunchConfig {
        LaunchConfig { grid_dim: ((n as u32).div_ceil(GB).max(1), 1, 1), block_dim: (GB, 1, 1), shared_mem_bytes: 0 }
    }

    /// The refraction of `W3KTP3` applied to the spectra at `va` (in place)
    /// for the factor `facth`; with `ab`, its adjoint instead: `va` the
    /// spectra before it, `ab` the cotangent after it, replaced by the
    /// cotangent before it.
    pub fn theta(&self, va: u64, ab: Option<u64>, p: &Ktp3Ptrs, facth: f64) -> Result<()> {
        let mut b = self.st.launch_builder(self.func(if ab.is_some() { "ktp3_theta_adj" } else { "ktp3_theta_fwd" }));
        b.arg(&va);
        if let Some(x) = ab.as_ref() {
            b.arg(x);
        }
        b.arg(&p.wn).arg(&p.cg).arg(&p.sig).arg(&p.dw).arg(&p.dddx).arg(&p.dddy).arg(&p.dcxdx).arg(&p.dcxdy).arg(&p.dcydx).arg(&p.dcydy);
        b.arg(&p.cthg0).arg(&p.nflag).arg(&p.ecos).arg(&p.esin).arg(&p.es2).arg(&p.ec2).arg(&p.esc);
        b.arg(&facth).arg(&p.ctmax).arg(&p.flcur).arg(&p.dmin).arg(&p.np);
        unsafe { b.launch(Self::cfg(p.np as usize * self.nk)) }?;
        Ok(())
    }

    /// The frequency shift of `W3KTP3` for the factor `fack`, as
    /// [`PropCuda::theta`].
    pub fn kshift(&self, va: u64, ab: Option<u64>, p: &Ktp3Ptrs, fack: f64) -> Result<()> {
        let mut b = self.st.launch_builder(self.func(if ab.is_some() { "ktp3_k_adj" } else { "ktp3_k_fwd" }));
        b.arg(&va);
        if let Some(x) = ab.as_ref() {
            b.arg(x);
        }
        b.arg(&p.wn).arg(&p.cg).arg(&p.sig).arg(&p.dsip).arg(&p.dw).arg(&p.dddx).arg(&p.dddy).arg(&p.cx).arg(&p.cy);
        b.arg(&p.dcxdx).arg(&p.dcxdy).arg(&p.dcydx).arg(&p.dcydy).arg(&p.nflag).arg(&p.es2).arg(&p.ec2).arg(&p.esc);
        b.arg(&fack).arg(&p.ctmax).arg(&p.fachfa).arg(&p.dmin).arg(&p.np);
        unsafe { b.launch(Self::cfg(p.np as usize * self.nth)) }?;
        Ok(())
    }

    /// The adjoint of the geographic propagation of frequency `ik` over
    /// `nsub` sub-steps: `va` the spectra before it (all frequencies; only
    /// `ik` is read), `vab` the cotangent of the spectra after it, whose
    /// frequency `ik` is replaced by the cotangent before it. The sub-steps
    /// are recomputed from `va` and kept on the device (`nsub` arrays of
    /// `np nth` values) for the reverse pass.
    pub fn geographic_adj(&self, ik: usize, nsub: usize, va: u64, vab: u64, cg: u64, g: &GeoPtrs) -> Result<()> {
        let st = &self.st;
        let (np, ne, nth) = (g.np as usize, g.ne as usize, self.nth);
        let n = np * nth;
        let (ik32, np32, ne32) = (ik as u32, g.np, g.ne);
        let mut states: Vec<CudaSlice<f64>> = Vec::with_capacity(nsub + 1);
        let mut ut = st.alloc_zeros::<f64>(ne * nth)?;
        let mut a0 = st.alloc_zeros::<f64>(n)?;
        {
            let mut b = st.launch_builder(self.func("va_to_ac"));
            b.arg(&va).arg(&mut a0).arg(&cg).arg(&g.clats).arg(&np32).arg(&ik32);
            unsafe { b.launch(Self::cfg(n)) }?;
        }
        states.push(a0);
        for sub in 0..nsub as u32 {
            let mut next = st.alloc_zeros::<f64>(n)?;
            let u = states.last().expect("state");
            let mut b = st.launch_builder(self.func("geo_tri"));
            b.arg(u).arg(&mut ut).arg(&g.coef).arg(&g.ccoef).arg(&g.ecos).arg(&g.esin).arg(&g.iter).arg(&g.ine).arg(&ne32).arg(&ik32).arg(&sub);
            unsafe { b.launch(Self::cfg(ne * nth / 4)) }?;
            let mut b = st.launch_builder(self.func("geo_node"));
            b.arg(u).arg(&mut next).arg(&ut);
            b.arg(&g.coef).arg(&g.ccoef).arg(&g.ecos).arg(&g.esin).arg(&g.dtk).arg(&g.iter).arg(&g.si_inv).arg(&g.mlo).arg(&g.mhi);
            b.arg(&g.start).arg(&g.slot).arg(&g.bidx).arg(&np32).arg(&ne32).arg(&ik32).arg(&sub);
            b.arg(&g.bb0).arg(&g.bbn).arg(&g.clats).arg(&g.rd10).arg(&g.rd20).arg(&g.dtg);
            unsafe { b.launch(Self::cfg(np * nth / 4)) }?;
            states.push(next);
        }
        // the reverse pass
        let mut acb = st.alloc_zeros::<f64>(n)?;
        let mut ub = st.alloc_zeros::<f64>(n)?;
        let mut stb = st.alloc_zeros::<f64>(n)?;
        let mut utb = st.alloc_zeros::<f64>(ne * nth)?;
        {
            let mut b = st.launch_builder(self.func("ac_to_va_adj"));
            b.arg(&states[nsub]).arg(&vab).arg(&mut acb).arg(&cg).arg(&g.clats).arg(&np32).arg(&ik32);
            unsafe { b.launch(Self::cfg(n)) }?;
        }
        for sub in (0..nsub as u32).rev() {
            let u = &states[sub as usize];
            let mut b = st.launch_builder(self.func("geo_tri"));
            b.arg(u).arg(&mut ut).arg(&g.coef).arg(&g.ccoef).arg(&g.ecos).arg(&g.esin).arg(&g.iter).arg(&g.ine).arg(&ne32).arg(&ik32).arg(&sub);
            unsafe { b.launch(Self::cfg(ne * nth / 4)) }?;
            let mut b = st.launch_builder(self.func("geo_adj_node1"));
            b.arg(u).arg(&ut).arg(&acb).arg(&mut ub).arg(&mut stb);
            b.arg(&g.coef).arg(&g.ccoef).arg(&g.ecos).arg(&g.esin).arg(&g.dtk).arg(&g.iter).arg(&g.si_inv).arg(&g.mlo).arg(&g.mhi);
            b.arg(&g.start).arg(&g.slot).arg(&g.bidx).arg(&np32).arg(&ne32).arg(&ik32).arg(&sub);
            unsafe { b.launch(Self::cfg(np * nth / 4)) }?;
            let mut b = st.launch_builder(self.func("geo_adj_tri"));
            b.arg(&stb).arg(&mut utb).arg(&g.coef).arg(&g.ccoef).arg(&g.ecos).arg(&g.esin).arg(&g.ine).arg(&ne32).arg(&ik32);
            unsafe { b.launch(Self::cfg(ne * nth / 4)) }?;
            let mut b = st.launch_builder(self.func("geo_adj_node2"));
            b.arg(&ub).arg(&utb).arg(&mut acb).arg(&g.coef).arg(&g.ccoef).arg(&g.ecos).arg(&g.esin).arg(&g.iter).arg(&g.start).arg(&g.slot);
            b.arg(&np32).arg(&ne32).arg(&ik32).arg(&sub);
            unsafe { b.launch(Self::cfg(np * nth / 4)) }?;
        }
        let mut b = st.launch_builder(self.func("va_to_ac_adj"));
        b.arg(&acb).arg(&vab).arg(&cg).arg(&g.clats).arg(&np32).arg(&ik32);
        unsafe { b.launch(Self::cfg(n)) }?;
        st.synchronize()?;
        Ok(())
    }

    /// The adjoint of the outputs `H_s` and `T_m02`
    /// (`hww3_wave::prop::outputs`): from their cotangents `hsb`, `t02b` per
    /// node, adds the cotangent of the spectra `va` to `ab` (`dden`, `sig` as
    /// the model; `fte` and `dth` its scalars).
    #[allow(clippy::too_many_arguments)]
    pub fn outputs_adj(&self, va: u64, cg: u64, sig: u64, dden: u64, nflag: u64, hsb: u64, t02b: u64, ab: u64, fte: f64, dth: f64, np: u32) -> Result<()> {
        let mut b = self.st.launch_builder(self.func("outputs_adj"));
        b.arg(&va).arg(&cg).arg(&sig).arg(&dden).arg(&nflag).arg(&hsb).arg(&t02b).arg(&ab).arg(&fte).arg(&dth).arg(&np);
        unsafe { b.launch(Self::cfg(np as usize)) }?;
        Ok(())
    }

    /// The adjoint of the level remap of `W3ULEV` on the device
    /// (`hww3_wave::prop::ulev_node`): `ab` the cotangent of the spectra after
    /// it, replaced by that before it, for the wavenumbers and group speeds
    /// `wno`, `cgo` before the level change and `wnn`, `cgn` after it, and
    /// the node modes `mode` (2: remapped when the first bin moved by 5 % of
    /// its width).
    #[allow(clippy::too_many_arguments)]
    pub fn ulev_node_adj(&self, ab: u64, wno: u64, cgo: u64, wnn: u64, cgn: u64, mode: u64, dsip: u64, fachfa: f64, np: u32) -> Result<()> {
        let mut b = self.st.launch_builder(self.func("ulev_node_adj"));
        b.arg(&ab).arg(&wno).arg(&cgo).arg(&wnn).arg(&cgn).arg(&mode).arg(&dsip).arg(&fachfa).arg(&np);
        unsafe { b.launch(Self::cfg(np as usize)) }?;
        Ok(())
    }

    /// Zero the cotangent `ab` at the `nlist` nodes listed at `nodes` (the
    /// nodes whose spectra the forcing or the open boundary overwrite).
    pub fn zero_nodes(&self, ab: u64, nodes: u64, np: u32, nlist: usize) -> Result<()> {
        if nlist == 0 {
            return Ok(());
        }
        let nl = nlist as u32;
        let mut b = self.st.launch_builder(self.func("zero_nodes_adj"));
        b.arg(&ab).arg(&nodes).arg(&np).arg(&nl);
        unsafe { b.launch(Self::cfg(nlist * self.nk * self.nth)) }?;
        Ok(())
    }
}
