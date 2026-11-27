//! Milestone M4: the adjoint of the model in CUDA C (Jean-Marc's choice of
//! 27 November 2026). The CUDA C sources (`src/cuda/*.cu*`) are compiled with
//! NVRTC and launched through `cudarc`; the parameter slots `P_*` of
//! [`crate::sources`] are passed as macros, so that the kernels follow the
//! layout of the parameter file.
//!
//! The source terms: the value kernel ([`SrceCuda::forward`], checked against
//! the CubeCL kernel) writes, when asked, the spectrum and the scalars at the
//! start of each dynamic sub-step of each node (the tape); the adjoint kernel
//! ([`SrceCuda::adjoint`]) takes the sub-steps in reverse from the tape,
//! recomputes each one's forward pass and runs the transposes of its parts:
//! the transpose of the tangent kernel of `hww3_wave::srce_tl` (milestone
//! M3), checked by dot-product tests against it (`hww3-src adj-src`).

use crate::dia_adj::DiaAdjTables;
use crate::sources::*;
use anyhow::{anyhow, bail, Result};
use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchArgs, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

/// The CUDA C value of the source terms (device functions).
pub const W3SRCE: &str = include_str!("cuda/w3srce.cuh");
/// The CUDA C value kernel of the source terms.
pub const W3SRCE_FWD: &str = include_str!("cuda/w3srce_fwd.cu");
/// The adjoints of `CALC_USTAR`, `W3SPR4`, `W3SLN1`, `W3SDB1`, `W3SBT4`.
pub const W3SRCE_ADJ1: &str = include_str!("cuda/w3srce_adj1.cuh");
/// The adjoints of `W3SDS4` (`SDSBCHOICE = 1`) and of the DIA.
pub const W3SRCE_ADJ1B: &str = include_str!("cuda/w3srce_adj1b.cuh");
/// The adjoint of `W3SIN4`.
pub const W3SRCE_ADJ2: &str = include_str!("cuda/w3srce_adj2.cuh");
/// The adjoint of `W3SDS4` with `SDSBCHOICE = 3` (Romero).
pub const W3SRCE_ADJ3: &str = include_str!("cuda/w3srce_adj3.cuh");
/// The adjoint kernel of the source terms.
pub const W3SRCE_ADJ: &str = include_str!("cuda/w3srce_adj.cu");

/// The scalars of a record of the tape after the spectrum (`NSC` of the
/// CUDA C sources).
pub const NSC: usize = 40;
/// The parameter cotangents per node of [`SrceCuda::adjoint`]: `BETAMAX`,
/// `SDSC2`, `NLPROP`, `BJGAM`, then the coefficients of `BETAMAX(U)`.
pub const NPB: usize = 4 + BU_MAXK;

macro_rules! pdefs {
    ($($n:ident),* $(,)?) => { vec![$(format!("-D{}={}u", stringify!($n), $n)),*] };
}

/// The macros of the parameter slots used by the CUDA C sources.
fn par_defs() -> Vec<String> {
    pdefs![
        P_GRAV, P_KAPPA, P_AALPHA, P_CAPCHNK, P_ITAUMAX, P_JUMAX, P_TAUWMAX, P_DELTAUW, P_DELU, P_ZZWND, P_WWNMEANPTAIL,
        P_WWNMEANP, P_FTE, P_FTF, P_SSTXFTFTAIL, P_SSTXFTWN, P_BU_KNOT, P_BU_NK, P_BU_COEF, P_NU_AIR, P_DRAT,
        P_TTAUWSHELTER, P_SSWELLF, P_ZZ0RAT, P_ABMIN, P_DELAB, P_NFW, P_ZZALP, P_SSINTHP, P_DTH, P_IUSTAR, P_IALPHA,
        P_DELUST, P_BT_N, P_BT_STRIDE, P_BT_DB, P_BT_B0, P_ILEVTAIL, P_DELTAIL, P_DELALP, P_SLNC1, P_FSPM, P_FSHF,
        P_KDCON, P_KDMN, P_SNLS1, P_SNLS2, P_SNLS3, P_SNLC1, P_DAL1, P_DAL2, P_DAL3, P_AWG, P_SWG, P_FACHFE, P_SSDSBR,
        P_SDSC3, P_SDSC5, P_SDSC21, P_SDSC2, P_SDSC6, P_DIKCUMUL, P_SSDSBM, P_SSDSP, P_RO, P_DSIP, P_SDBC2, P_FDONLY,
        P_SDBC1, P_SED_SG, P_SBTCX, P_XERFMAX, P_DELXERF, P_NERF, P_FACP, P_DMIN, P_ICE_FORCING, P_ICESCALES, P_FFXFM,
        P_FFXPM, P_FACTI1, P_FACTI2, P_XFLT, P_DTMAX, P_XREL, P_DTMIN, P_FFXFA, P_FHMAX, P_FACHFA, P_TLFREEZE, P_BBETA,
        P_SINTABLE, BU_MAXK,
    ]
}

/// The nodes per SM asked of the adjoint kernel of the source terms
/// (`__launch_bounds__`, which caps its registers): 3, or the value of the
/// environment variable `HWW3_ADJ_MINB` (1 to 4), for timing comparisons.
fn adj_minb() -> u32 {
    std::env::var("HWW3_ADJ_MINB").ok().and_then(|v| v.parse().ok()).filter(|v| (1..=4).contains(v)).unwrap_or(3)
}

/// The source tables of [`SrceTables`] on the device, and the kernels.
pub struct SrceCuda {
    /// The context.
    pub ctx: Arc<CudaContext>,
    /// Its stream.
    pub st: Arc<CudaStream>,
    f_fwd: CudaFunction,
    f_adj: CudaFunction,
    par: CudaSlice<f64>,
    sig: CudaSlice<f64>,
    dden: CudaSlice<f64>,
    ecos: CudaSlice<f64>,
    esin: CudaSlice<f64>,
    sig2k: CudaSlice<f64>,
    dden2k: CudaSlice<f64>,
    fwtab: CudaSlice<f64>,
    erftab: CudaSlice<f64>,
    satw: CudaSlice<f64>,
    sati: CudaSlice<u32>,
    dia_ip: CudaSlice<u32>,
    dia_ic: CudaSlice<u32>,
    af11: CudaSlice<f64>,
    taut: CudaSlice<f64>,
    tauhf: CudaSlice<f64>,
    cumw: CudaSlice<f64>,
    dtauhf: CudaSlice<f64>,
    tb: [CudaSlice<u32>; 6],
    sat_t: [CudaSlice<u32>; 2],
    /// Frequencies and directions.
    pub nk: usize,
    /// See [`SrceCuda::nk`].
    pub nth: usize,
    /// Shared memory of the value kernel (bytes).
    pub smem_fwd: u32,
    /// Registers per thread of the value kernel.
    pub regs_fwd: i32,
    /// Shared memory of the adjoint kernel (bytes).
    pub smem_adj: u32,
    /// Registers per thread of the adjoint kernel.
    pub regs_adj: i32,
}

/// The per-node inputs of the source terms on the host (layouts of
/// [`crate::srce::SrceArgs`]: `va[(ik np + i) nth + th]`, `wn`, `cg` as
/// `[ik np + i]` for `ik = 0 .. nk + 1`).
pub struct SrceHost<'a> {
    /// Spectra.
    pub va: &'a [f64],
    /// Wavenumbers.
    pub wn: &'a [f64],
    /// Group velocities.
    pub cg: &'a [f64],
    /// Depths.
    pub dw: &'a [f64],
    /// Wind speeds.
    pub u10: &'a [f64],
    /// Wind directions.
    pub u10d: &'a [f64],
    /// Ice concentrations.
    pub ice: &'a [f64],
    /// Median grain sizes.
    pub d50: &'a [f64],
    /// Critical Shields numbers.
    pub psic: &'a [f64],
    /// Node flags.
    pub nflag: &'a [u32],
}

/// The per-node inputs of the source terms on the device, as raw pointers
/// (the layouts of [`SrceHost`]): `cudarc` allocations or the buffers of the
/// CubeCL model, which share the primary context of the device.
#[derive(Clone, Copy, Debug)]
pub struct SrceDev {
    /// Wavenumbers.
    pub wn: u64,
    /// Group velocities.
    pub cg: u64,
    /// Depths.
    pub dw: u64,
    /// Wind speeds.
    pub u10: u64,
    /// Wind directions.
    pub u10d: u64,
    /// Ice concentrations.
    pub ice: u64,
    /// Median grain sizes.
    pub d50: u64,
    /// Critical Shields numbers.
    pub psic: u64,
    /// Node flags.
    pub nflag: u64,
}

/// The raw pointer of a `cudarc` buffer (used on the one stream of
/// [`SrceCuda`], which is synchronised before any other use).
fn raw<T>(s: &CudaSlice<T>, st: &CudaStream) -> u64 {
    let (p, _g) = s.device_ptr(st);
    p
}

/// What [`SrceCuda::adjoint_dev`] reports.
#[derive(Clone, Debug, Default)]
pub struct AdjInfo {
    /// The number of dynamic sub-steps of each node.
    pub nsub: Vec<u32>,
    /// Largest tape of a batch (bytes).
    pub tape_bytes: usize,
    /// Batches of nodes.
    pub batches: usize,
    /// Time of the value passes (the count, then those writing the tape, s).
    pub secs_fwd: f64,
    /// Time of the adjoint kernel (s).
    pub secs_adj: f64,
}

/// The cotangents of one global step of the source terms
/// ([`SrceCuda::adjoint`]), laid out as the inputs.
pub struct SrceBar {
    /// Cotangent of the spectra at the start of the step.
    pub va: Vec<f64>,
    /// Cotangent of the wind speeds.
    pub u10: Vec<f64>,
    /// Cotangent of the wind directions.
    pub u10d: Vec<f64>,
    /// Cotangents of the parameters per node, [`NPB`] values per node:
    /// `BETAMAX` (with the derivative of the tail-stress table), `SDSC2`,
    /// `NLPROP`, `BJGAM`, then the coefficients of `BETAMAX(U)`.
    pub par: Vec<f64>,
    /// The spectra at the end of the step.
    pub vout: Vec<f64>,
    /// The number of dynamic sub-steps of each node.
    pub nsub: Vec<u32>,
    /// Size of the tape (bytes).
    pub tape_bytes: usize,
    /// Time of the value kernel writing the tape (s).
    pub secs_fwd: f64,
    /// Time of the adjoint kernel (s).
    pub secs_adj: f64,
}

impl SrceCuda {
    /// Compile the kernels for `nk` frequencies and `nth` directions and copy
    /// the tables to the device. The derivative of the tail-stress table in
    /// `BETAMAX` is zero until [`SrceCuda::set_dtauhf`].
    pub fn new(t: &SrceTables, nk: usize, nth: usize) -> Result<SrceCuda> {
        if t.par[P_SINTABLE] <= 0.0 {
            return Err(anyhow!("the CUDA C source terms need SINTABLE = 1 (the tail stress from the tables)"));
        }
        let nu = 128usize;
        let romero = t.par[P_RO] != 0.0;
        let nspec = nk * nth;
        let nue = t.nspecy + nth;
        let nb1 = t.nspecx + nth;
        if nue > 2 * nspec || nue < 10 * nk {
            bail!("the adjoint kernel's scratch needs 10 NK <= NSPECY + NTH <= 2 NSPEC (NSPECY {}, NSPEC {nspec})", t.nspecy);
        }
        let mut defs = par_defs();
        defs.extend([
            format!("-DNK={nk}u"),
            format!("-DNTH={nth}u"),
            format!("-DNSPECX={}u", t.nspecx),
            format!("-DNSPECY={}u", t.nspecy),
            format!("-DNSAT={}u", t.nsat.max(1)),
            format!("-DNU={nu}u"),
            format!("-DROMERO={}", romero as u32),
            // source lines for the profilers (no cost at run time)
            "-lineinfo".to_string(),
            // nodes per SM asked of the adjoint and value kernels (their register caps)
            format!("-DMINB_ADJ={}", adj_minb()),
            "-DMINB_FWD=3".to_string(),
        ]);
        let src = [W3SRCE, W3SRCE_FWD, W3SRCE_ADJ1, W3SRCE_ADJ1B, W3SRCE_ADJ2, W3SRCE_ADJ3, W3SRCE_ADJ].join("\n");
        let ctx = CudaContext::new(0)?;
        let opts = CompileOptions { arch: Some("compute_80"), options: defs, ..Default::default() };
        let ptx = compile_ptx_with_opts(src, opts).map_err(|e| anyhow!("NVRTC: {e:?}"))?;
        let module = ctx.load_module(ptx)?;
        let f_fwd = module.load_function("srce_fwd")?;
        let f_adj = module.load_function("srce_adj")?;
        let doubles = nspec + nue + 2 * nb1 + 13 * nk + NSC + 3 * nth + nu + 2 * nk;
        let smem_fwd = (doubles * 8 + nspec) as u32;
        f_fwd.set_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_fwd as i32)?;
        let regs_fwd = f_fwd.num_regs()?;
        let doubles = nspec + nue + 2 * nb1 + 13 * nk + NSC + (NSC + 10) + 3 * nth + nu + 1 + 4 * nk;
        let smem_adj = (doubles * 8 + nspec) as u32;
        f_adj
            .set_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_adj as i32)
            .map_err(|e| anyhow!("adjoint kernel with {smem_adj} B of shared memory: {e:?}"))?;
        let regs_adj = f_adj.num_regs()?;
        let st = ctx.default_stream();
        let one = |v: &[f64]| if v.is_empty() { vec![0.0] } else { v.to_vec() };
        let oneu = |v: &[u32]| if v.is_empty() { vec![0u32] } else { v.to_vec() };
        let tab = DiaAdjTables::new(&t.dia_ip, &t.dia_ic, nspec, t.nspecx, t.nspecy, nth);
        let tb = [
            st.clone_htod(&tab.tb1s)?,
            st.clone_htod(&tab.tb1e)?,
            st.clone_htod(&tab.tb2s)?,
            st.clone_htod(&tab.tb2e)?,
            st.clone_htod(&tab.tps)?,
            st.clone_htod(&tab.tpe)?,
        ];
        // SATI transposed: for each direction, the entries th NSAT + j that
        // read it, in the order of th then j
        let nsat = t.nsat.max(1);
        let mut sats = vec![0u32; nth + 1];
        let mut sate = vec![];
        for th2 in 0..nth {
            for th in 0..nth {
                for j in 0..nsat {
                    if t.sati.get(j + th * nsat).copied() == Some(th2 as u32) {
                        sate.push((th * nsat + j) as u32);
                    }
                }
            }
            sats[th2 + 1] = sate.len() as u32;
        }
        if sate.is_empty() {
            sate.push(0);
        }
        let sat_t = [st.clone_htod(&sats)?, st.clone_htod(&sate)?];
        Ok(SrceCuda {
            sat_t,
            par: st.clone_htod(&t.par)?,
            sig: st.clone_htod(&t.sig)?,
            dden: st.clone_htod(&t.dden)?,
            ecos: st.clone_htod(&t.ecos)?,
            esin: st.clone_htod(&t.esin)?,
            sig2k: st.clone_htod(&t.sig2k)?,
            dden2k: st.clone_htod(&t.dden2k)?,
            fwtab: st.clone_htod(&one(&t.fwtable))?,
            erftab: st.clone_htod(&one(&t.erftable))?,
            satw: st.clone_htod(&one(&t.satw))?,
            sati: st.clone_htod(&oneu(&t.sati))?,
            dia_ip: st.clone_htod(&t.dia_ip)?,
            dia_ic: st.clone_htod(&t.dia_ic)?,
            af11: st.clone_htod(&t.af11)?,
            taut: st.clone_htod(&one(&t.taut))?,
            tauhf: st.clone_htod(&one(&t.tauhf))?,
            cumw: st.clone_htod(&one(&t.cumw))?,
            dtauhf: st.clone_htod(&vec![0.0; t.tauhf.len().max(1)])?,
            tb,
            ctx,
            st,
            f_fwd,
            f_adj,
            nk,
            nth,
            smem_fwd,
            regs_fwd,
            smem_adj,
            regs_adj,
        })
    }

    /// The derivative of the tail-stress table in `BETAMAX` (laid out as the
    /// table; used when the tables are not in `BETAMAX`, [`P_BT_N`] <= 1).
    pub fn set_dtauhf(&mut self, d: &[f64]) -> Result<()> {
        self.dtauhf = self.st.clone_htod(if d.is_empty() { &[0.0][..] } else { d })?;
        Ok(())
    }

    /// Replace the parameters (same layout and configuration: the freeze mask
    /// [`P_TLFREEZE`] of the adjoint, the values of `BETAMAX`, `SDSC2`...).
    pub fn set_par(&mut self, par: &[f64]) -> Result<()> {
        self.par = self.st.clone_htod(par)?;
        Ok(())
    }

    fn upload(&self, h: &SrceHost) -> Result<(Vec<CudaSlice<f64>>, CudaSlice<u32>, SrceDev)> {
        let st = &self.st;
        let mut v = Vec::with_capacity(8);
        for x in [h.wn, h.cg, h.dw, h.u10, h.u10d, h.ice, h.d50, h.psic] {
            v.push(st.clone_htod(x)?);
        }
        let nf = st.clone_htod(h.nflag)?;
        let d = SrceDev {
            wn: raw(&v[0], st),
            cg: raw(&v[1], st),
            dw: raw(&v[2], st),
            u10: raw(&v[3], st),
            u10d: raw(&v[4], st),
            ice: raw(&v[5], st),
            d50: raw(&v[6], st),
            psic: raw(&v[7], st),
            nflag: raw(&nf, st),
        };
        Ok((v, nf, d))
    }

    /// The arguments `SRCE_IN_ARGS` of the kernels.
    fn push_in<'a>(&'a self, b: &mut LaunchArgs<'a>, d: &'a SrceDev) {
        b.arg(&d.wn).arg(&d.cg).arg(&d.dw).arg(&d.u10).arg(&d.u10d).arg(&d.ice).arg(&d.d50).arg(&d.psic).arg(&d.nflag);
        b.arg(&self.par).arg(&self.sig).arg(&self.dden).arg(&self.ecos).arg(&self.esin).arg(&self.sig2k).arg(&self.dden2k);
        b.arg(&self.fwtab).arg(&self.erftab).arg(&self.satw).arg(&self.sati).arg(&self.dia_ip).arg(&self.dia_ic).arg(&self.af11);
        b.arg(&self.taut).arg(&self.tauhf).arg(&self.cumw);
    }

    /// The value kernel on nodes `i0 .. i0 + n` of `np`: spectra at `va`
    /// updated in place; with `tape` non-zero, the records of the sub-steps
    /// written (`[tape, tape_off, tape_cap]`); `nsub` (or 0) the number of
    /// sub-steps of each node; `scr` the scratch of the `n` nodes
    /// (`3 nspec` values each).
    #[allow(clippy::too_many_arguments)]
    fn launch_fwd(&self, d: &SrceDev, va: u64, ust: u64, dtdyn: u64, dtg: f64, np: usize, i0: usize, n: usize, tape: [u64; 3], nsub: u64, scr: u64) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: self.smem_fwd };
        let (np32, i032) = (np as u32, i0 as u32);
        let mut b = self.st.launch_builder(&self.f_fwd);
        b.arg(&va);
        self.push_in(&mut b, d);
        b.arg(&ust).arg(&dtdyn).arg(&dtg).arg(&np32).arg(&i032).arg(&tape[0]).arg(&tape[1]).arg(&tape[2]).arg(&nsub).arg(&scr);
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// One global step `dtg` of the source terms on the `np` nodes of `h`:
    /// the spectra after the step, u* and the mean dynamic step, and the
    /// time of the launch (s).
    pub fn forward(&self, h: &SrceHost, dtg: f64, np: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, f64)> {
        let st = &self.st;
        let (_keep, _nf, d) = self.upload(h)?;
        let d_va = st.clone_htod(h.va)?;
        let d_ust = st.alloc_zeros::<f64>(np)?;
        let d_dtdyn = st.alloc_zeros::<f64>(np)?;
        let d_scr = st.alloc_zeros::<f64>(3 * self.nk * self.nth * np)?;
        st.synchronize()?;
        let t0 = std::time::Instant::now();
        self.launch_fwd(&d, raw(&d_va, st), raw(&d_ust, st), raw(&d_dtdyn, st), dtg, np, 0, np, [0; 3], 0, raw(&d_scr, st))?;
        st.synchronize()?;
        let secs = t0.elapsed().as_secs_f64();
        Ok((st.clone_dtoh(&d_va)?, st.clone_dtoh(&d_ust)?, st.clone_dtoh(&d_dtdyn)?, secs))
    }

    /// The adjoint of one global step `dtg` of the source terms on the `np`
    /// nodes of `d`, on device buffers: `va` the spectra at the start of the
    /// step (not changed), `ybar` the cotangent of the spectra at its end.
    /// Writes the cotangent of `va` to `abar`, those of the wind speed and
    /// direction to `ubar`, `udbar` (`np` values each), those of the
    /// parameters to `pbar` ([`NPB`] per node) and, with `vout`, the spectra
    /// at the end of the step. A value pass counts the sub-steps of each node;
    /// then the nodes go in batches whose tape fits in `tape_max` bytes: a
    /// value pass writes the tape of the batch, and the adjoint kernel reads it.
    #[allow(clippy::too_many_arguments)]
    pub fn adjoint_dev(
        &self,
        d: &SrceDev,
        va: u64,
        ybar: u64,
        abar: u64,
        ubar: u64,
        udbar: u64,
        pbar: u64,
        vout: Option<u64>,
        dtg: f64,
        np: usize,
        tape_max: usize,
    ) -> Result<AdjInfo> {
        let st = &self.st;
        let nspec = self.nk * self.nth;
        let n = nspec * np;
        let mut info = AdjInfo::default();
        let d_vend = st.alloc_zeros::<f64>(n)?;
        let d_vtape = st.alloc_zeros::<f64>(n)?;
        let d_ust = st.alloc_zeros::<f64>(np)?;
        let d_dtdyn = st.alloc_zeros::<f64>(np)?;
        let d_nsub = st.alloc_zeros::<u32>(np)?;
        let (pend, ptape, pust, pdt, pnsub) = (raw(&d_vend, st), raw(&d_vtape, st), raw(&d_ust, st), raw(&d_dtdyn, st), raw(&d_nsub, st));
        st.synchronize()?;
        let t0 = std::time::Instant::now();
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(pend, va, n * 8, st.cu_stream())?;
            cudarc::driver::result::memcpy_dtod_async(ptape, va, n * 8, st.cu_stream())?;
        }
        // the scratch of the kernels (8 nspec values per node of a batch),
        // the count pass in batches of nodes
        let max_nodes = (tape_max / (64 * nspec)).clamp(1, 1 << 16);
        let d_scr = st.alloc_zeros::<f64>(8 * nspec * max_nodes.min(np))?;
        let pscr = raw(&d_scr, st);
        let mut b0 = 0usize;
        while b0 < np {
            let nb = max_nodes.min(np - b0);
            self.launch_fwd(d, pend, pust, pdt, dtg, np, b0, nb, [0; 3], pnsub, pscr)?;
            b0 += nb;
        }
        info.nsub = st.clone_dtoh(&d_nsub)?;
        if let Some(v) = vout {
            unsafe { cudarc::driver::result::memcpy_dtod_async(v, pend, n * 8, st.cu_stream())? };
        }
        st.synchronize()?;
        info.secs_fwd += t0.elapsed().as_secs_f64();
        // batches of nodes whose tape fits
        let rec = ((nspec + NSC) * 8) as u64;
        let mut off = vec![0u64; np];
        let mut i0 = 0usize;
        while i0 < np {
            let mut i1 = i0;
            let mut tot = 0u64;
            while i1 < np && i1 - i0 < max_nodes && (i1 == i0 || tot + info.nsub[i1] as u64 * rec <= tape_max as u64) {
                off[i1] = tot / 8;
                tot += info.nsub[i1] as u64 * rec;
                i1 += 1;
            }
            let d_off = st.clone_htod(&off)?;
            let d_tape = st.alloc_zeros::<f64>((tot / 8).max(1) as usize)?;
            let (ptp, poff) = (raw(&d_tape, st), raw(&d_off, st));
            st.synchronize()?;
            let t1 = std::time::Instant::now();
            self.launch_fwd(d, ptape, pust, pdt, dtg, np, i0, i1 - i0, [ptp, poff, pnsub], 0, pscr)?;
            st.synchronize()?;
            info.secs_fwd += t1.elapsed().as_secs_f64();
            let t2 = std::time::Instant::now();
            let cfg = LaunchConfig { grid_dim: ((i1 - i0) as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: self.smem_adj };
            let (np32, i032) = (np as u32, i0 as u32);
            let mut b = st.launch_builder(&self.f_adj);
            b.arg(&va).arg(&pend).arg(&ybar).arg(&abar);
            self.push_in(&mut b, d);
            b.arg(&self.dtauhf);
            for x in &self.tb {
                b.arg(x);
            }
            b.arg(&self.sat_t[0]).arg(&self.sat_t[1]);
            b.arg(&dtg).arg(&np32).arg(&i032).arg(&ptp).arg(&poff).arg(&pnsub).arg(&ubar).arg(&udbar).arg(&pbar).arg(&pscr);
            unsafe { b.launch(cfg) }?;
            st.synchronize()?;
            info.secs_adj += t2.elapsed().as_secs_f64();
            info.tape_bytes = info.tape_bytes.max(tot as usize);
            info.batches += 1;
            i0 = i1;
        }
        Ok(info)
    }

    /// The adjoint of one global step `dtg` of the source terms on the `np`
    /// nodes of `h` (host arrays), for the cotangent `ybar` of the spectra at
    /// the end of the step (laid out as `h.va`), in one batch of nodes
    /// ([`SrceCuda::adjoint_dev`]).
    pub fn adjoint(&self, h: &SrceHost, dtg: f64, np: usize, ybar: &[f64]) -> Result<SrceBar> {
        let st = &self.st;
        let nspec = self.nk * self.nth;
        let (_keep, _nf, d) = self.upload(h)?;
        let d_va = st.clone_htod(h.va)?;
        let d_ybar = st.clone_htod(ybar)?;
        let d_abar = st.alloc_zeros::<f64>(nspec * np)?;
        let d_vout = st.alloc_zeros::<f64>(nspec * np)?;
        let d_ub = st.alloc_zeros::<f64>(np)?;
        let d_udb = st.alloc_zeros::<f64>(np)?;
        let d_pb = st.alloc_zeros::<f64>(np * NPB)?;
        let info = self.adjoint_dev(
            &d,
            raw(&d_va, st),
            raw(&d_ybar, st),
            raw(&d_abar, st),
            raw(&d_ub, st),
            raw(&d_udb, st),
            raw(&d_pb, st),
            Some(raw(&d_vout, st)),
            dtg,
            np,
            usize::MAX,
        )?;
        st.synchronize()?;
        Ok(SrceBar {
            va: st.clone_dtoh(&d_abar)?,
            u10: st.clone_dtoh(&d_ub)?,
            u10d: st.clone_dtoh(&d_udb)?,
            par: st.clone_dtoh(&d_pb)?,
            vout: st.clone_dtoh(&d_vout)?,
            nsub: info.nsub,
            tape_bytes: info.tape_bytes,
            secs_fwd: info.secs_fwd,
            secs_adj: info.secs_adj,
        })
    }
}

/// The seeds of the adjoint runs (`Model::adjoint_run`) from the files
/// `hsb_TAG.f64` and `t02b_TAG.f64` (`TAG` = `YYYYMMDDHH`, one value per
/// node, as written by `scripts/cal_obs.py seed`) in `dir`: the cotangents
/// of `H_s` and `T_m02` at the time `t`, none at the times without files.
pub fn seeds_from_dir(dir: &std::path::Path, np: usize, t: crate::ww3io::Time) -> Result<Option<(Vec<f64>, Vec<f64>)>> {
    if t.1 % 10000 != 0 {
        return Ok(None);
    }
    let tag = format!("{:08}{:02}", t.0, t.1 / 10000);
    let (ph, pt) = (dir.join(format!("hsb_{tag}.f64")), dir.join(format!("t02b_{tag}.f64")));
    if !ph.exists() {
        return Ok(None);
    }
    let rd = |p: &std::path::Path| -> Result<Vec<f64>> {
        let b = std::fs::read(p)?;
        if b.len() != 8 * np {
            bail!("{}: {} values against {np} nodes", p.display(), b.len() / 8);
        }
        Ok(b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes"))).collect())
    };
    let h = rd(&ph)?;
    let tt = if pt.exists() { rd(&pt)? } else { vec![0.0; np] };
    Ok(Some((h, tt)))
}
