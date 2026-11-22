//! Propagation-only WW3 model on one GPU (milestone M1): the time loop of
//! `W3WAVE` for the HWW3 explicit configuration with the source terms off
//! (`RUN%FLSOU = F`).
//!
//! One global step `DTG` (600 s) ending at `TIME`:
//! 1. currents interpolated in magnitude and direction (`W3UCUR`) and their
//!    gradients (`UG_GRADIENTS`);
//! 2. boundary spectra read when `TIME` passes the next record (`W3IOBC`,
//!    `W3UBPT`) and imposed (`APPLY_BOUNDARY_CONDITION_VA`);
//! 3. ice update (`W3UICE`) and water level update (`W3ULEV`) when their next
//!    field is within half a field interval;
//! 4. sub-cycling count of each component (`PDLIB_W3XYPFSN2`, `LCALC`);
//! 5. `NTLOC - ITLOCH` ... : `W3KTP3` `ITLOCH` times, the geographic
//!    propagation of every component (`PDLIB_W3XYPUG`), `W3KTP3` the
//!    remaining `NTLOC - ITLOCH` times.
//!
//! Forcing fields are read as `ww3_shel` does at the start of each `W3WAVE`
//! call: currents are kept between the two records around the time, levels
//! and ice keep the first record after the call start (the Copernicus Marine
//! hourly fields of the HWW3 runs are stamped at hh:30).

use crate::device::Real;
use crate::geom::Geometry;
use crate::grid::Grid;
use crate::prop;
use crate::spec::{node_wavenumbers, Spec, DERA, RADIUS};
use crate::ww3io::{FieldFile, NestFile, Restart, Time};
use anyhow::{anyhow, bail, Result};
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::path::PathBuf;

pub const UNDEF: f32 = -999.9;

pub struct Inputs {
    pub restart: PathBuf,
    pub current: Option<PathBuf>,
    pub level: Option<PathBuf>,
    pub ice: Option<PathBuf>,
    pub nest: Option<PathBuf>,
    /// Start time, required for a COLD restart (`ww3_strt`), which has none.
    pub start: Option<Time>,
}

#[derive(Clone)]
pub struct Config {
    /// Global step `DTMAX` (s).
    pub dtg: f32,
    /// Spectral propagation step `DTKTH` (s).
    pub dtkth: f32,
    /// `CFLTM`, the CFL limit of the spectral propagation.
    pub ctmax: f32,
    pub flcx: bool,
    pub flcth: bool,
    pub flck: bool,
    /// `ITSTEP` before the first step (parity of the k/theta order of `W3KTP3`).
    pub itstep0: u64,
    pub block: u32,
    /// Units per (frequency, direction group) for the sub-cycling count.
    pub lanes: usize,
    /// Wavenumbers, group speeds and spectrum remapping of `W3ULEV` on the
    /// host (threads) instead of the device.
    pub host_level: bool,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            dtg: 600.0,
            dtkth: 300.0,
            ctmax: 0.7,
            flcx: true,
            flcth: true,
            flck: true,
            itstep0: 1,
            block: 256,
            lanes: 2048,
            host_level: false,
        }
    }
}

struct Fields {
    file: FieldFile,
    /// Last record read: its time and values.
    next: Option<(Time, Vec<Vec<f32>>)>,
    eof: bool,
}

impl Fields {
    fn open(p: &PathBuf) -> Result<Fields> {
        Ok(Fields { file: FieldFile::open(p)?, next: None, eof: false })
    }
    fn advance(&mut self) -> Result<bool> {
        match self.file.next()? {
            Some(r) => {
                self.next = Some(r);
                Ok(true)
            }
            None => {
                self.eof = true;
                Ok(false)
            }
        }
    }
    fn tnext(&self) -> Option<Time> {
        self.next.as_ref().map(|r| r.0)
    }
}

/// Timers of the parts of a step (s, after device synchronisation when profiling).
#[derive(Default, Clone, Debug)]
pub struct Timers {
    pub forcing: f64,
    /// Host part of the level updates (included in `forcing`).
    pub level_host: f64,
    pub iter: f64,
    pub ktp3: f64,
    /// Theta part of `ktp3` (profiling only).
    pub ktp3_theta: f64,
    pub geographic: f64,
    pub outputs: f64,
    pub sweeps: u64,
}

pub struct Model<F: Real> {
    pub client: Client,
    pub np: usize,
    pub ne: usize,
    pub nk: usize,
    pub nth: usize,
    pub sp: Spec,
    pub cfg: Config,
    pub time: Time,
    pub itstep: u64,
    pub dmin: f32,
    pub timers: Timers,
    pub profile: bool,
    pub iter_host: Vec<u32>,
    // host copies
    pub lat: Vec<f32>,
    zb: Vec<f32>,
    pub dw: Vec<f32>,
    pub wlv: Vec<f32>,
    pub mapsta: Vec<i32>,
    pub mapst2: Vec<i32>,
    iobp: Vec<u8>,
    iobpd: Vec<u64>,
    iobpa: Vec<u8>,
    deep: Vec<bool>,
    /// `WN`, `CG` `[ik][node]` for `ik = 0 .. NK+1`; with the device level
    /// update (default) they stay those of the restart, the device holds the
    /// current ones.
    pub wn: Vec<f32>,
    pub cg: Vec<f32>,
    // forcing
    cur: Option<Fields>,
    tc0: Option<Time>,
    tcn: Option<Time>,
    c0: (Vec<f32>, Vec<f32>),
    cn: (Vec<f32>, Vec<f32>),
    lev: Option<Fields>,
    tlev: Time,
    ice: Option<Fields>,
    tice: Time,
    nest: Option<NestFile>,
    tbpi0: Option<Time>,
    tbpin: Option<Time>,
    abpin: Vec<Vec<f32>>,
    bbpin: Vec<f32>,
    bbpi0: Vec<f32>,
    /// Boundary point -> node, and its RDBPI/IPBPI.
    pub bnode: Vec<u32>,
    ipbpi: Vec<i32>,
    rdbpi: Vec<f32>,
    pub nbnd: usize,
    // device: static
    d_ine: Handle,
    d_ien: Handle,
    d_tria: Handle,
    d_start: Handle,
    d_slot: Handle,
    d_si: Handle,
    d_si_inv: Handle,
    d_clats: Handle,
    d_clatis: Handle,
    d_cthg0: Handle,
    d_ecos: Handle,
    d_esin: Handle,
    d_es2: Handle,
    d_ec2: Handle,
    d_esc: Handle,
    d_sig: Handle,
    d_dsip: Handle,
    d_dden: Handle,
    d_bidx: Handle,
    d_bnode: Handle,
    // device: hourly
    d_dw: Handle,
    d_wn: Handle,
    d_cg: Handle,
    d_gx: Handle,
    d_gy: Handle,
    d_coef: Handle,
    d_nflag: Handle,
    d_mlo: Handle,
    d_mhi: Handle,
    pub d_dddx: Handle,
    pub d_dddy: Handle,
    d_bb0: Handle,
    d_bbn: Handle,
    // device: per step
    d_ca0: Handle,
    d_cai: Handle,
    d_cd0: Handle,
    d_cdi: Handle,
    pub d_cx: Handle,
    pub d_cy: Handle,
    d_ux: Handle,
    d_uy: Handle,
    pub d_dcxdx: Handle,
    pub d_dcxdy: Handle,
    pub d_dcydx: Handle,
    pub d_dcydy: Handle,
    d_ccoef: Handle,
    d_partial: Handle,
    d_iter: Handle,
    d_dtk: Handle,
    // state
    pub d_va: Handle,
    d_ac: Handle,
    d_ac2: Handle,
    d_ut: Handle,
    d_hs: Handle,
    d_t02: Handle,
    d_en: Handle,
    _f: PhantomData<F>,
}

fn bf(h: &Handle, n: usize) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(h.clone(), n) }
}

fn sca<F: Real>(x: f32) -> F {
    <F as crate::backend::Scalar>::of_f64(x as f64)
}

impl<F: Real> Model<F> {
    pub fn new(client: Client, grid: &Grid, geo: &Geometry, inp: &Inputs, cfg: Config) -> Result<Model<F>> {
        let (np, ne, nk, nth) = (geo.np, geo.ne, geo.nk, geo.nth);
        let nspec = nk * nth;
        let s = &grid.spec;
        anyhow::ensure!(nk >= 3, "the W3KTP3 kernels need NK >= 3");
        let sp = Spec::new(nk, nth, s.freq1 as f32, s.xfr as f32, s.thoff as f32);
        let dmin = s.dmin as f32;
        let mut rs = Restart::read(&inp.restart)?;
        if rs.nsea != np || rs.nspec != nspec {
            bail!("restart NSEA {} NSPEC {} against grid {} nodes, {} components", rs.nsea, rs.nspec, np, nspec);
        }
        // MAPSTA of the grid (ww3_grid, unstructured): 2 on the open
        // boundary, 1 elsewhere
        let grid_mapsta: Vec<i32> = (0..np).map(|i| if geo.iobp[i] == 2 { 2 } else { 1 }).collect();
        if rs.cold {
            // ww3_strt: spectra only, no time, levels, ice or masks
            let t = inp.start.ok_or_else(|| anyhow!("{}: COLD restart without time, give --start", inp.restart.display()))?;
            rs.time = t;
            rs.tlev = t;
            rs.tice = t;
            rs.mapsta = grid_mapsta.clone();
        } else if let Some(t) = inp.start {
            if t != rs.time {
                bail!("--start {t} differs from the restart time {}", rs.time);
            }
        }
        eprintln!(
            "restart {}: time {}, TLEV {}, TICE {}, ice nodes {}, MAPSTA<0 {}, MAPSTA unlike the grid's away from ice and dry nodes {}",
            inp.restart.display(),
            rs.time,
            rs.tlev,
            rs.tice,
            rs.mapst2.iter().filter(|&&m| m & 1 == 1).count(),
            rs.mapsta.iter().filter(|&&m| m < 0).count(),
            (0..np).filter(|&i| rs.mapst2[i] == 0 && rs.mapsta[i] != grid_mapsta[i]).count()
        );
        let lat: Vec<f32> = grid.lat.iter().map(|&v| v as f32).collect();
        let zb: Vec<f32> = grid.depth.iter().map(|&d| -(d as f32)).collect();
        let wlv = rs.wlv.clone();
        let dw: Vec<f32> = (0..np).map(|i| (wlv[i] - zb[i]).max(0.0)).collect();
        let mut wn = vec![0f32; (nk + 2) * np];
        let mut cg = vec![0f32; (nk + 2) * np];
        let mut deep = vec![false; np];
        {
            let mut wnc = vec![0f32; nk + 2];
            let mut cgc = vec![0f32; nk + 2];
            for i in 0..np {
                deep[i] = node_wavenumbers(&sp, dw[i], zb[i], dmin, &mut wnc, &mut cgc);
                for ik in 0..nk + 2 {
                    wn[ik * np + i] = wnc[ik];
                    cg[ik * np + i] = cgc[ik];
                }
            }
        }
        // spectra: restart VA(isp, isea), isp = ith + ik nth -> va[(ik np + i) nth + ith]
        let mut va = vec![0f32; nspec * np];
        va.par_chunks_mut(np * nth).enumerate().for_each(|(ik, blk)| {
            for i in 0..np {
                let src = &rs.va[i * nspec + ik * nth..i * nspec + (ik + 1) * nth];
                blk[i * nth..(i + 1) * nth].copy_from_slice(src);
            }
        });
        let fb = std::mem::size_of::<F>();
        let upf = |v: &[f32]| {
            let w: Vec<F> = v.iter().map(|&x| sca::<F>(x)).collect();
            client.create_from_slice(F::as_bytes(&w))
        };
        let upd = |v: &[f64]| {
            let w: Vec<F> = v.iter().map(|&x| <F as crate::backend::Scalar>::of_f64(x)).collect();
            client.create_from_slice(F::as_bytes(&w))
        };
        let upu = |v: &[u32]| client.create_from_slice(u32::as_bytes(v));
        let clats: Vec<f32> = lat.iter().map(|&y| (DERA * y).cos()).collect();
        let clatis: Vec<f32> = clats.iter().map(|c| 1.0 / c).collect();
        let cthg0: Vec<f32> = lat.iter().map(|&y| -(DERA * y).tan() / RADIUS).collect();
        let si_inv: Vec<f64> = geo.si.iter().map(|s| 1.0 / s).collect();
        // boundary points
        let (mut nest, mut bnode, mut ipbpi, mut rdbpi) = (None, vec![], vec![], vec![]);
        if let Some(p) = &inp.nest {
            let nf = NestFile::open(p)?;
            if nf.nk != nk || nf.nth != nth {
                bail!("nest file spectra {}x{} against {}x{}", nf.nk, nf.nth, nk, nth);
            }
            let obc: Vec<usize> = (0..np).filter(|&i| geo.iobp[i] == 2).collect();
            for b in 0..nf.nbi {
                let (x, y) = (nf.xbpi[b] as f64, nf.ybpi[b] as f64);
                let (best, d2) = obc
                    .iter()
                    .map(|&i| (i, (grid.lon[i] - x).powi(2) + (grid.lat[i] - y).powi(2)))
                    .fold((usize::MAX, f64::MAX), |a, c| if c.1 < a.1 { c } else { a });
                if d2 > 1e-6 {
                    bail!("boundary point {b} ({x}, {y}) is {:.4} deg from the nearest open-boundary node", d2.sqrt());
                }
                bnode.push(best as u32);
            }
            ipbpi = nf.ipbpi.clone();
            rdbpi = nf.rdbpi.clone();
            eprintln!("nest {}: {} boundary points, {} open-boundary nodes", p.display(), nf.nbi, obc.len());
            nest = Some(nf);
        }
        let nbnd = bnode.len();
        let mut bidx = vec![u32::MAX; np];
        for (b, &i) in bnode.iter().enumerate() {
            bidx[i as usize] = b as u32;
        }
        let nb1 = nbnd.max(1);
        let mut m = Model {
            np,
            ne,
            nk,
            nth,
            cfg: cfg.clone(),
            time: rs.time,
            itstep: cfg.itstep0,
            dmin,
            timers: Timers::default(),
            profile: false,
            iter_host: vec![1; nspec],
            lat,
            zb,
            dw: dw.clone(),
            wlv,
            mapsta: rs.mapsta.clone(),
            mapst2: rs.mapst2.clone(),
            iobp: geo.iobp.clone(),
            iobpd: geo.iobpd.clone(),
            iobpa: geo.iobpa.clone(),
            deep,
            wn: wn.clone(),
            cg: cg.clone(),
            cur: None,
            tc0: None,
            tcn: None,
            c0: (vec![0.0; np], vec![0.0; np]),
            cn: (vec![0.0; np], vec![0.0; np]),
            lev: None,
            tlev: rs.tlev,
            ice: None,
            tice: rs.tice,
            nest,
            tbpi0: None,
            tbpin: None,
            abpin: vec![],
            bbpin: vec![0.0; nb1 * nspec],
            bbpi0: vec![0.0; nb1 * nspec],
            bnode: bnode.clone(),
            ipbpi,
            rdbpi,
            nbnd,
            d_ine: upu(&geo.ine),
            d_ien: upd(&geo.ien),
            d_tria: upd(&geo.tria),
            d_start: upu(&geo.start),
            d_slot: upu(&geo.slot),
            d_si: upd(&geo.si),
            d_si_inv: upd(&si_inv),
            d_clats: upf(&clats),
            d_clatis: upf(&clatis),
            d_cthg0: upf(&cthg0),
            d_ecos: upf(&sp.ecos),
            d_esin: upf(&sp.esin),
            d_es2: upf(&sp.es2),
            d_ec2: upf(&sp.ec2),
            d_esc: upf(&sp.esc),
            d_sig: upf(&sp.sig),
            d_dsip: upf(&sp.dsip),
            d_dden: upf(&sp.dden),
            d_bidx: upu(&bidx),
            d_bnode: upu(if nbnd > 0 { &bnode } else { &[0u32] }),
            d_dw: upf(&dw),
            d_wn: upf(&wn),
            d_cg: upf(&cg),
            d_gx: client.empty(nk * np * fb),
            d_gy: client.empty(nk * np * fb),
            d_coef: client.empty(nk * ne * 12 * fb),
            d_nflag: client.empty(np * 4),
            d_mlo: client.empty(np * 4),
            d_mhi: client.empty(np * 4),
            d_dddx: client.empty(np * fb),
            d_dddy: client.empty(np * fb),
            d_bb0: upf(&vec![0.0; nb1 * nspec]),
            d_bbn: upf(&vec![0.0; nb1 * nspec]),
            d_ca0: upf(&vec![0.0; np]),
            d_cai: upf(&vec![0.0; np]),
            d_cd0: upf(&vec![0.0; np]),
            d_cdi: upf(&vec![0.0; np]),
            d_cx: upf(&vec![0.0; np]),
            d_cy: upf(&vec![0.0; np]),
            d_ux: upf(&vec![0.0; np]),
            d_uy: upf(&vec![0.0; np]),
            d_dcxdx: upf(&vec![0.0; np]),
            d_dcxdy: upf(&vec![0.0; np]),
            d_dcydx: upf(&vec![0.0; np]),
            d_dcydy: upf(&vec![0.0; np]),
            d_ccoef: upf(&vec![0.0; ne * 6]),
            d_partial: client.empty(nspec * cfg.lanes * fb),
            d_iter: upu(&vec![1u32; nspec]),
            d_dtk: upf(&vec![cfg.dtg; nspec]),
            d_va: upf(&va),
            d_ac: client.empty(np * nth * fb),
            d_ac2: client.empty(np * nth * fb),
            d_ut: client.empty(ne * nth * fb),
            d_hs: client.empty(np * fb),
            d_t02: client.empty(np * fb),
            d_en: client.empty(np * fb),
            sp,
            client,
            _f: PhantomData,
        };
        drop(va);
        if let Some(p) = &inp.current {
            m.cur = Some(Fields::open(p)?);
        }
        if let Some(p) = &inp.level {
            let mut f = Fields::open(p)?;
            // first call of W3FLDG: first record at or after the start
            while f.advance()? {
                if f.tnext().unwrap() >= m.time {
                    break;
                }
            }
            m.lev = Some(f);
        }
        if let Some(p) = &inp.ice {
            let mut f = Fields::open(p)?;
            while f.advance()? {
                if f.tnext().unwrap() >= m.time {
                    break;
                }
            }
            m.ice = Some(f);
        }
        m.upload_flags();
        m.update_depth_dependent();
        Ok(m)
    }

    fn dims(&self, n: usize) -> (CubeCount, CubeDim) {
        let b = self.cfg.block;
        (CubeCount::Static((n as u32).div_ceil(b).max(1), 1, 1), CubeDim::new_1d(b))
    }

    pub fn sync(&self) {
        futures_lite::future::block_on(self.client.sync()).expect("device sync");
    }

    fn upf(&self, v: &[f32]) -> Handle {
        let w: Vec<F> = v.iter().map(|&x| sca::<F>(x)).collect();
        self.client.create_from_slice(F::as_bytes(&w))
    }

    pub fn download(&self, h: &Handle) -> Vec<f32> {
        let b = self.client.read_one(h.clone()).expect("device read");
        F::from_bytes(&b).iter().map(|&x| <F as crate::backend::Scalar>::as_f64(x) as f32).collect()
    }

    pub fn download_u32(&self, h: &Handle) -> Vec<u32> {
        let b = self.client.read_one(h.clone()).expect("device read");
        u32::from_bytes(&b).to_vec()
    }

    /// Node flags and mask words from IOBP, DW, MAPSTA (host copies).
    fn upload_flags(&mut self) {
        let np = self.np;
        let nth = self.nth;
        let full: u64 = (1u64 << nth) - 1;
        let mut nflag = vec![0u32; np];
        let mut mlo = vec![0u32; np];
        let mut mhi = vec![0u32; np];
        for i in 0..np {
            let wet = self.dw[i] >= self.dmin + 1e-5;
            let mut f = 0u32;
            if self.iobp[i] == 1 {
                f |= prop::FLAG_INTERIOR;
            }
            if wet {
                f |= prop::FLAG_WET;
            }
            if self.mapsta[i] == 1 {
                f |= prop::FLAG_SEA;
            }
            if self.iobp[i] > 0 {
                f |= prop::FLAG_NOTCOAST;
            }
            if self.mapsta[i] > 0 {
                f |= prop::FLAG_OUTPUT;
            }
            nflag[i] = f;
            let mask = if !wet { 0 } else if self.iobpa[i] == 1 { full } else { self.iobpd[i] };
            mlo[i] = mask as u32;
            mhi[i] = ((mask >> 32) as u32 & 0x7fff_ffff) | ((self.iobpa[i] as u32) << 31);
        }
        self.d_nflag = self.client.create_from_slice(u32::as_bytes(&nflag));
        self.d_mlo = self.client.create_from_slice(u32::as_bytes(&mlo));
        self.d_mhi = self.client.create_from_slice(u32::as_bytes(&mhi));
    }

    /// After a depth change on the host: upload depths, wavenumbers and
    /// group speeds, then [`Model::depth_kernels`].
    fn update_depth_dependent(&mut self) {
        self.d_dw = self.upf(&self.dw);
        self.d_wn = self.upf(&self.wn);
        self.d_cg = self.upf(&self.cg);
        self.depth_kernels();
    }

    /// After a depth change: group velocities, per-frequency coefficients, depth gradients.
    fn depth_kernels(&mut self) {
        let (np, ne, nk) = (self.np, self.ne, self.nk);
        let facx = 1.0 / (DERA * RADIUS);
        let (c, d) = self.dims(np * nk);
        unsafe {
            prop::group_velocity::launch_unchecked::<F>(
                &self.client,
                c,
                d,
                bf(&self.d_cg, (nk + 2) * np),
                bf(&self.d_clats, np),
                bf(&self.d_nflag, np),
                bf(&self.d_gx, nk * np),
                bf(&self.d_gy, nk * np),
                facx,
                np,
                nk,
            );
        }
        let (c, d) = self.dims(ne);
        for ik in 0..nk {
            unsafe {
                prop::tri_coef::launch_unchecked::<F>(
                    &self.client,
                    c.clone(),
                    d,
                    bf(&self.d_gx, nk * np),
                    bf(&self.d_gy, nk * np),
                    bf(&self.d_ine, 3 * ne),
                    bf(&self.d_ien, 6 * ne),
                    bf(&self.d_coef, nk * ne * 12),
                    np,
                    ne,
                    ik,
                );
            }
        }
        let dw = self.d_dw.clone();
        let (dx, dy) = (self.d_dddx.clone(), self.d_dddy.clone());
        self.gradient(&dw, &dx, &dy);
    }

    fn gradient(&self, p: &Handle, dx: &Handle, dy: &Handle) {
        let (np, ne) = (self.np, self.ne);
        let (c, d) = self.dims(np);
        unsafe {
            prop::gradient::launch_unchecked::<F>(
                &self.client,
                c,
                d,
                bf(p, np),
                bf(dx, np),
                bf(dy, np),
                bf(&self.d_ine, 3 * ne),
                bf(&self.d_ien, 6 * ne),
                bf(&self.d_tria, ne),
                bf(&self.d_clatis, np),
                bf(&self.d_start, np + 1),
                bf(&self.d_slot, 3 * ne),
                1.0 / (DERA * RADIUS),
                np,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Forcing
    // -----------------------------------------------------------------------

    /// At the start of an hourly `W3WAVE` call (`t0`): read currents, levels
    /// and ice as `ww3_shel` does.
    fn read_fields(&mut self, t0: Time) -> Result<()> {
        // currents: keep TC0 <= t0 < TCN
        if let Some(f) = self.cur.as_mut() {
            let mut changed = false;
            loop {
                let need = match self.tcn {
                    None => true,
                    Some(t) => t <= t0,
                };
                if !need {
                    break;
                }
                if !f.advance()? {
                    bail!("current file ends before {t0}");
                }
                let (t, flds) = f.next.take().unwrap();
                if self.tcn.is_some() {
                    self.tc0 = self.tcn;
                    std::mem::swap(&mut self.c0, &mut self.cn);
                } else {
                    self.tc0 = Some(t);
                    self.c0 = (flds[0].clone(), flds[1].clone());
                }
                self.tcn = Some(t);
                self.cn = (flds[0].clone(), flds[1].clone());
                changed = true;
            }
            if changed {
                self.current_coefficients();
            }
        }
        // levels and ice: read while the last record read is not after t0
        for which in 0..2 {
            let f = if which == 0 { self.lev.as_mut() } else { self.ice.as_mut() };
            if let Some(f) = f {
                while !f.eof && f.tnext().map_or(true, |t| t <= t0) {
                    if !f.advance()? {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    /// `W3UCUR` with `FLFRST`: magnitude and direction of the two current records.
    fn current_coefficients(&mut self) {
        let np = self.np;
        let tpi = 2.0 * std::f32::consts::PI;
        let pi = std::f32::consts::PI;
        let (mut ca0, mut cai, mut cd0, mut cdi) = (vec![0f32; np], vec![0f32; np], vec![0f32; np], vec![0f32; np]);
        for i in 0..np {
            let (x0, y0) = (self.c0.0[i], self.c0.1[i]);
            let (xn, yn) = (self.cn.0[i], self.cn.1[i]);
            let a0 = (x0 * x0 + y0 * y0).sqrt();
            let an = (xn * xn + yn * yn).sqrt();
            let d0 = if a0 > 1e-7 { (tpi + y0.atan2(x0)) % tpi } else { 0.0 };
            let dn = if an > 1e-7 { (tpi + yn.atan2(xn)) % tpi } else { d0 };
            let c0 = if a0 > 1e-7 { d0 } else { dn };
            let mut dd = dn - c0;
            if dd.abs() > pi {
                dd -= tpi * dd.signum();
            }
            ca0[i] = a0;
            cai[i] = an - a0;
            cd0[i] = c0;
            cdi[i] = dd;
        }
        self.d_ca0 = self.upf(&ca0);
        self.d_cai = self.upf(&cai);
        self.d_cd0 = self.upf(&cd0);
        self.d_cdi = self.upf(&cdi);
    }

    fn update_currents(&mut self) {
        let (np, ne) = (self.np, self.ne);
        let rd = match (self.tc0, self.tcn) {
            (Some(a), Some(b)) => (self.time.secs() - a.secs()) as f32 / ((b.secs() - a.secs()) as f32).max(1e-7),
            _ => return,
        };
        let facx = 1.0 / (DERA * RADIUS);
        let (c, d) = self.dims(np);
        unsafe {
            prop::current_interp::launch_unchecked::<F>(
                &self.client,
                c,
                d,
                bf(&self.d_ca0, np),
                bf(&self.d_cai, np),
                bf(&self.d_cd0, np),
                bf(&self.d_cdi, np),
                bf(&self.d_clats, np),
                bf(&self.d_nflag, np),
                bf(&self.d_cx, np),
                bf(&self.d_cy, np),
                bf(&self.d_ux, np),
                bf(&self.d_uy, np),
                rd,
                facx,
                np,
            );
        }
        let (cx, cy) = (self.d_cx.clone(), self.d_cy.clone());
        let (a, b, c2, d2) = (self.d_dcxdx.clone(), self.d_dcxdy.clone(), self.d_dcydx.clone(), self.d_dcydy.clone());
        self.gradient(&cx, &a, &b);
        self.gradient(&cy, &c2, &d2);
        let (c, d) = self.dims(ne);
        unsafe {
            prop::cur_coef::launch_unchecked::<F>(
                &self.client,
                c,
                d,
                bf(&self.d_ux, np),
                bf(&self.d_uy, np),
                bf(&self.d_ine, 3 * ne),
                bf(&self.d_ien, 6 * ne),
                bf(&self.d_ccoef, ne * 6),
                ne,
            );
        }
    }

    /// `W3IOBC` / `W3UBPT`: boundary spectra around `TIME`, interpolated to the points.
    fn update_boundary(&mut self) -> Result<()> {
        let nspec = self.nk * self.nth;
        let Some(nest) = self.nest.as_mut() else { return Ok(()) };
        let mut changed = false;
        if self.tbpin.is_none() {
            let (t, sp) = nest.next()?.ok_or_else(|| anyhow!("empty nest file"))?;
            self.abpin = sp;
            self.tbpi0 = Some(t);
            self.tbpin = Some(t);
            self.bbpin = interp_boundary(&self.abpin, &self.ipbpi, &self.rdbpi, self.nbnd, nspec, self.nth, &self.sp.sig);
            self.bbpi0 = self.bbpin.clone();
            changed = true;
        }
        while self.time > self.tbpin.unwrap() {
            match nest.next()? {
                Some((t, sp)) => {
                    self.tbpi0 = self.tbpin;
                    self.tbpin = Some(t);
                    self.abpin = sp;
                    self.bbpi0 = std::mem::take(&mut self.bbpin);
                    self.bbpin = interp_boundary(&self.abpin, &self.ipbpi, &self.rdbpi, self.nbnd, nspec, self.nth, &self.sp.sig);
                    changed = true;
                }
                None => bail!("nest file ends before {}", self.time),
            }
        }
        if changed {
            self.d_bb0 = self.upf(&self.bbpi0);
            self.d_bbn = self.upf(&self.bbpin);
        }
        Ok(())
    }

    /// `RD10 = DSEC21(TBPI0, TIME)`, `RD20 = DSEC21(TBPI0, TBPIN)`.
    fn boundary_rd(&self) -> (f32, f32) {
        match (self.tbpi0, self.tbpin) {
            (Some(a), Some(b)) => ((self.time.secs() - a.secs()) as f32, (b.secs() - a.secs()) as f32),
            _ => (1.0, 0.0),
        }
    }

    fn apply_boundary(&self) {
        if self.nbnd == 0 {
            return;
        }
        let (rd10, rd20) = self.boundary_rd();
        let (rd1, rd2) = if rd20 > 0.001 {
            let r2 = (rd10 / rd20).clamp(0.0, 1.0);
            (1.0 - r2, r2)
        } else {
            (0.0, 1.0)
        };
        let (np, nk, nth) = (self.np, self.nk, self.nth);
        let n = self.nbnd * nk * nth;
        let (c, d) = self.dims(n);
        unsafe {
            prop::apply_boundary::launch_unchecked::<F>(
                &self.client,
                c,
                d,
                bf(&self.d_va, nk * np * nth),
                bf(&self.d_bnode, self.nbnd),
                bf(&self.d_bb0, n),
                bf(&self.d_bbn, n),
                bf(&self.d_cg, (nk + 2) * np),
                rd1,
                rd2,
                np,
                self.nbnd,
                nk,
                nth,
            );
        }
    }

    /// `W3UICE` when the next ice field is within half an interval.
    fn update_ice(&mut self) -> Result<()> {
        let Some(f) = self.ice.as_ref() else { return Ok(()) };
        let Some((tin, flds)) = f.next.as_ref() else { return Ok(()) };
        let dti0 = (tin.secs() - self.tice.secs()) as f64;
        if dti0 == 0.0 {
            return Ok(());
        }
        if ((tin.secs() - self.time.secs()) as f64) > 0.5 * dti0 {
            return Ok(());
        }
        let (tin, ice) = (*tin, flds[0].clone());
        self.tice = tin;
        let ficen = 0.5f32;
        let mut zero = vec![];
        for i in 0..self.np {
            let was = self.mapst2[i] & 1 == 1;
            if ice[i] >= ficen && !was {
                self.mapsta[i] = -self.mapsta[i].abs();
                self.mapst2[i] |= 1;
                zero.push(i as u32);
            } else if ice[i] < ficen && was {
                self.mapst2[i] &= !1;
                if self.mapst2[i] == 0 {
                    self.mapsta[i] = self.mapsta[i].abs();
                    zero.push(i as u32);
                }
            }
        }
        eprintln!("{}: ice update from {tin}, {} nodes reset", self.time, zero.len());
        self.zero_nodes(&zero);
        self.upload_flags();
        Ok(())
    }

    fn zero_nodes(&self, nodes: &[u32]) {
        if nodes.is_empty() {
            return;
        }
        let (np, nk, nth) = (self.np, self.nk, self.nth);
        let d_nodes = self.client.create_from_slice(u32::as_bytes(nodes));
        let n = nodes.len() * nk * nth;
        let (c, d) = self.dims(n);
        unsafe {
            prop::zero_nodes::launch_unchecked::<F>(
                &self.client,
                c,
                d,
                bf(&self.d_va, nk * np * nth),
                bf(&d_nodes, nodes.len()),
                np,
                nodes.len(),
                nk,
                nth,
            );
        }
    }

    /// `W3ULEV` when the next level field is within half an interval. Returns
    /// whether the depth changed.
    fn update_level(&mut self) -> Result<bool> {
        let Some(f) = self.lev.as_ref() else { return Ok(false) };
        let Some((tln, flds)) = f.next.as_ref() else { return Ok(false) };
        let dtl0 = (tln.secs() - self.tlev.secs()) as f64;
        if dtl0 == 0.0 || ((tln.secs() - self.time.secs()) as f64) > 0.5 * dtl0 {
            return Ok(false);
        }
        let (tln, wlev) = (*tln, flds[0].clone());
        self.tlev = tln;
        if !self.cfg.host_level {
            self.update_level_device(tln, &wlev);
            return Ok(true);
        }
        let t_host = std::time::Instant::now();
        let np = self.np;
        for i in 0..np {
            self.wlv[i] = wlev[i];
            self.dw[i] = (self.wlv[i] - self.zb[i]).max(0.0);
        }
        // Nodes are independent: split them in contiguous chunks over the
        // available cores and concatenate the per-chunk lists in node order,
        // so the result does not depend on the number of threads.
        let nthr = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 32);
        let chunk = np.div_ceil(nthr).max(1);
        let nchunk = np.div_ceil(chunk);
        let mut wn_parts: Vec<Vec<&mut [f32]>> = (0..nchunk).map(|_| vec![]).collect();
        let mut cg_parts: Vec<Vec<&mut [f32]>> = (0..nchunk).map(|_| vec![]).collect();
        for row in self.wn.chunks_mut(np) {
            for (t, piece) in row.chunks_mut(chunk).enumerate() {
                wn_parts[t].push(piece);
            }
        }
        for row in self.cg.chunks_mut(np) {
            for (t, piece) in row.chunks_mut(chunk).enumerate() {
                cg_parts[t].push(piece);
            }
        }
        let (sp, dmin) = (&self.sp, self.dmin);
        let (wlv, zb, dw, deep) = (&self.wlv, &self.zb, &self.dw, &self.deep);
        let outs: Vec<LevelChunk> = std::thread::scope(|scope| {
            let handles: Vec<_> = wn_parts
                .into_iter()
                .zip(cg_parts)
                .zip(self.mapsta.chunks_mut(chunk).zip(self.mapst2.chunks_mut(chunk)))
                .enumerate()
                .map(|(t, ((wnp, cgp), (ms, ms2)))| {
                    scope.spawn(move || level_chunk(sp, dmin, t * chunk, wlv, zb, dw, deep, wnp, cgp, ms, ms2))
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("level thread")).collect()
        });
        let mut zero = vec![];
        let mut rnodes: Vec<u32> = vec![];
        let (mut own, mut dwn, mut i1v, mut r1v, mut tail) = (vec![], vec![], vec![], vec![], vec![]);
        for o in outs {
            zero.extend(o.zero);
            rnodes.extend(o.rnodes);
            own.extend(o.own);
            dwn.extend(o.dwn);
            i1v.extend(o.i1v);
            r1v.extend(o.r1v);
            tail.extend(o.tail);
        }
        self.timers.level_host += t_host.elapsed().as_secs_f64();
        let nk = self.nk;
        self.zero_nodes(&zero);
        if !rnodes.is_empty() {
            let nl = rnodes.len();
            let d_nodes = self.client.create_from_slice(u32::as_bytes(&rnodes));
            let d_own = self.upf(&own);
            let d_dwn = self.upf(&dwn);
            let d_i1 = self.client.create_from_slice(u32::as_bytes(&i1v));
            let d_r1 = self.upf(&r1v);
            let d_tail = self.client.create_from_slice(u32::as_bytes(&tail));
            let (c, d) = self.dims(nl * self.nth);
            unsafe {
                prop::ulev_remap::launch_unchecked::<F>(
                    &self.client,
                    c,
                    d,
                    bf(&self.d_va, nk * np * self.nth),
                    bf(&d_nodes, nl),
                    bf(&d_own, nl * nk),
                    bf(&d_dwn, nl * nk),
                    bf(&d_i1, nl * nk),
                    bf(&d_r1, nl * nk),
                    bf(&d_tail, nl),
                    self.sp.fachfa,
                    np,
                    nl,
                    nk,
                    self.nth,
                );
            }
        }
        if self.profile {
            eprintln!("{}: level from {tln}, {} nodes remapped, {} reset", self.time, rnodes.len(), zero.len());
        }
        self.upload_flags();
        self.update_depth_dependent();
        Ok(true)
    }

    /// `W3ULEV` with the wavenumbers, group speeds and remapping on the
    /// device (kernel `ulev_node`); the host keeps the depths and the
    /// drying and wetting of `MAPSTA` / `MAPST2`, and its copies of `wn`
    /// and `cg` stay those of the restart.
    fn update_level_device(&mut self, tln: Time, wlev: &[f32]) {
        let t_host = std::time::Instant::now();
        let (np, nk, nth) = (self.np, self.nk, self.nth);
        let mut mode = vec![0u32; np];
        let mut zero = vec![];
        for i in 0..np {
            self.wlv[i] = wlev[i];
            self.dw[i] = (self.wlv[i] - self.zb[i]).max(0.0);
            if self.deep[i] {
                continue;
            }
            let dry = self.mapst2[i] & 2 == 2;
            if self.wlv[i] - self.zb[i] <= 0.0 {
                mode[i] = 1;
                if !dry {
                    zero.push(i as u32);
                    self.mapst2[i] |= 2;
                    self.mapsta[i] = -self.mapsta[i].abs();
                }
                continue;
            }
            if dry {
                mode[i] = 1;
                self.mapst2[i] &= !2;
                zero.push(i as u32);
                if self.mapst2[i] == 0 {
                    self.mapsta[i] = self.mapsta[i].abs();
                }
                continue;
            }
            mode[i] = if self.mapsta[i] < 0 { 1 } else { 2 };
        }
        self.timers.level_host += t_host.elapsed().as_secs_f64();
        self.zero_nodes(&zero);
        self.d_dw = self.upf(&self.dw);
        let d_mode = self.client.create_from_slice(u32::as_bytes(&mode));
        let (c, d) = self.dims(np);
        unsafe {
            prop::ulev_node::launch_unchecked::<F>(
                &self.client,
                c,
                d,
                bf(&self.d_va, nk * np * nth),
                bf(&self.d_wn, (nk + 2) * np),
                bf(&self.d_cg, (nk + 2) * np),
                bf(&self.d_dw, np),
                bf(&d_mode, np),
                bf(&self.d_sig, nk + 2),
                bf(&self.d_dsip, nk + 2),
                self.dmin,
                self.sp.fachfa,
                np,
                nk,
                nth,
            );
        }
        if self.profile {
            let remapped = self.download_u32(&d_mode).iter().filter(|&&m| m == 3).count();
            eprintln!("{}: level from {tln}, {remapped} nodes remapped, {} reset", self.time, zero.len());
        }
        self.upload_flags();
        self.depth_kernels();
    }

    // -----------------------------------------------------------------------
    // Propagation
    // -----------------------------------------------------------------------

    /// `ITER(ik, ith)` of every component (device), and its copy on the host.
    pub fn compute_iter(&mut self) {
        let (np, ne, nk, nth) = (self.np, self.ne, self.nk, self.nth);
        let lanes = self.cfg.lanes;
        let n = nk * (nth / 4) * lanes;
        let (c, d) = self.dims(n);
        {
            unsafe {
                prop::iter_partial::launch_unchecked::<F>(
                    &self.client,
                    c,
                    d,
                    bf(&self.d_coef, nk * ne * 12),
                    bf(&self.d_ccoef, ne * 6),
                    bf(&self.d_ecos, nth),
                    bf(&self.d_esin, nth),
                    bf(&self.d_si, np),
                    bf(&self.d_nflag, np),
                    bf(&self.d_start, np + 1),
                    bf(&self.d_slot, 3 * ne),
                    bf(&self.d_partial, nk * nth * lanes),
                    np,
                    ne,
                    nk,
                    lanes,
                    nth,
                );
            }
        }
        let (c, d) = self.dims(nk * nth);
        unsafe {
            prop::iter_finish::launch_unchecked::<F>(
                &self.client,
                c,
                d,
                bf(&self.d_partial, nk * nth * lanes),
                bf(&self.d_iter, nk * nth),
                bf(&self.d_dtk, nk * nth),
                self.cfg.dtg,
                lanes,
                nk * nth,
            );
        }
        self.iter_host = self.download_u32(&self.d_iter);
    }

    pub fn max_iter(&self, ik: usize) -> u32 {
        *self.iter_host[ik * self.nth..(ik + 1) * self.nth].iter().max().unwrap()
    }

    /// One call of `W3KTP3` on every active node, in the order of `ITSTEP`.
    pub fn ktp3(&mut self, facth: f32, fack: f32) {
        let k_first = self.itstep % 2 == 0;
        if k_first {
            self.ktp3_k(fack);
            self.ktp3_theta_timed(facth);
        } else {
            self.ktp3_theta_timed(facth);
            self.ktp3_k(fack);
        }
    }

    /// [`Model::ktp3_theta`], timed apart from the k part when profiling.
    fn ktp3_theta_timed(&mut self, facth: f32) {
        if !self.profile {
            self.ktp3_theta(facth);
            return;
        }
        self.sync();
        let t = std::time::Instant::now();
        self.ktp3_theta(facth);
        self.sync();
        self.timers.ktp3_theta += t.elapsed().as_secs_f64();
    }

    pub fn ktp3_theta(&self, facth: f32) {
        if !self.cfg.flcth {
            return;
        }
        let (np, nk, nth) = (self.np, self.nk, self.nth);
        // rows staged in shared memory: keep block (nth + 1) values within 48 KiB
        let mut tb = self.cfg.block as usize;
        while tb > 32 && tb * (nth + 1) * std::mem::size_of::<F>() > 48 * 1024 {
            tb /= 2;
        }
        let c = CubeCount::Static((np * nk).div_ceil(tb) as u32, 1, 1);
        let d = CubeDim::new_1d(tb as u32);
        let flcur = if self.cur.is_some() { 1.0 } else { 0.0 };
        unsafe {
            prop::ktp3_theta::launch_unchecked::<F>(
                &self.client,
                c,
                d,
                bf(&self.d_va, nk * np * nth),
                bf(&self.d_wn, (nk + 2) * np),
                bf(&self.d_cg, (nk + 2) * np),
                bf(&self.d_sig, nk + 2),
                bf(&self.d_dw, np),
                bf(&self.d_dddx, np),
                bf(&self.d_dddy, np),
                bf(&self.d_dcxdx, np),
                bf(&self.d_dcxdy, np),
                bf(&self.d_dcydx, np),
                bf(&self.d_dcydy, np),
                bf(&self.d_cthg0, np),
                bf(&self.d_nflag, np),
                bf(&self.d_ecos, nth),
                bf(&self.d_esin, nth),
                bf(&self.d_es2, nth),
                bf(&self.d_ec2, nth),
                bf(&self.d_esc, nth),
                facth,
                self.cfg.ctmax,
                flcur,
                self.dmin,
                np,
                nk,
                nth,
                tb,
            );
        }
    }

    pub fn ktp3_k(&self, fack: f32) {
        if !self.cfg.flck {
            return;
        }
        let (np, nk, nth) = (self.np, self.nk, self.nth);
        let (c, d) = self.dims(np * nth);
        unsafe {
            prop::ktp3_k::launch_unchecked::<F>(
                &self.client,
                c,
                d,
                bf(&self.d_va, nk * np * nth),
                bf(&self.d_wn, (nk + 2) * np),
                bf(&self.d_cg, (nk + 2) * np),
                bf(&self.d_sig, nk + 2),
                bf(&self.d_dsip, nk + 2),
                bf(&self.d_dw, np),
                bf(&self.d_dddx, np),
                bf(&self.d_dddy, np),
                bf(&self.d_cx, np),
                bf(&self.d_cy, np),
                bf(&self.d_dcxdx, np),
                bf(&self.d_dcxdy, np),
                bf(&self.d_dcydx, np),
                bf(&self.d_dcydy, np),
                bf(&self.d_nflag, np),
                bf(&self.d_es2, nth),
                bf(&self.d_ec2, nth),
                bf(&self.d_esc, nth),
                fack,
                self.cfg.ctmax,
                self.sp.fachfa,
                self.dmin,
                np,
                nk,
                nth,
            );
        }
    }

    /// `PDLIB_W3XYPUG` for every direction of frequency `ik`.
    pub fn geographic(&mut self, ik: usize) {
        let (np, ne, nk, nth) = (self.np, self.ne, self.nk, self.nth);
        let nspec = nk * nth;
        let n = np * nth;
        let (c, d) = self.dims(n);
        unsafe {
            prop::va_to_ac::launch_unchecked::<F>(
                &self.client,
                c.clone(),
                d,
                bf(&self.d_va, nk * n),
                bf(&self.d_ac, n),
                bf(&self.d_cg, (nk + 2) * np),
                bf(&self.d_clats, np),
                np,
                ik,
                nth,
            );
        }
        let (rd10, rd20) = self.boundary_rd();
        let nb1 = self.nbnd.max(1);
        let (ct, dt) = self.dims(ne * (nth / 4));
        let (cn, dn) = self.dims(np * (nth / 4));
        let nsub = self.max_iter(ik);
        for sub in 0..nsub {
            unsafe {
                prop::geo_tri_x4::launch_unchecked::<F>(
                    &self.client,
                    ct.clone(),
                    dt,
                    bf(&self.d_ac, n),
                    bf(&self.d_ut, ne * nth),
                    bf(&self.d_coef, nk * ne * 12),
                    bf(&self.d_ccoef, ne * 6),
                    bf(&self.d_ecos, nth),
                    bf(&self.d_esin, nth),
                    bf(&self.d_iter, nspec),
                    bf(&self.d_ine, 3 * ne),
                    ne,
                    ik,
                    sub,
                    nth,
                );
                prop::geo_node_x4::launch_unchecked::<F>(
                    &self.client,
                    cn.clone(),
                    dn,
                    bf(&self.d_ac, n),
                    bf(&self.d_ac2, n),
                    bf(&self.d_ut, ne * nth),
                    bf(&self.d_coef, nk * ne * 12),
                    bf(&self.d_ccoef, ne * 6),
                    bf(&self.d_ecos, nth),
                    bf(&self.d_esin, nth),
                    bf(&self.d_dtk, nspec),
                    bf(&self.d_iter, nspec),
                    bf(&self.d_si_inv, np),
                    bf(&self.d_mlo, np),
                    bf(&self.d_mhi, np),
                    bf(&self.d_start, np + 1),
                    bf(&self.d_slot, 3 * ne),
                    bf(&self.d_bidx, np),
                    bf(&self.d_bb0, nb1 * nspec),
                    bf(&self.d_bbn, nb1 * nspec),
                    bf(&self.d_clats, np),
                    rd10,
                    rd20,
                    self.cfg.dtg,
                    np,
                    ne,
                    nspec,
                    ik,
                    sub,
                    nth,
                );
            }
            std::mem::swap(&mut self.d_ac, &mut self.d_ac2);
            self.timers.sweeps += 1;
        }
        unsafe {
            prop::ac_to_va::launch_unchecked::<F>(
                &self.client,
                c,
                d,
                bf(&self.d_ac, n),
                bf(&self.d_va, nk * n),
                bf(&self.d_cg, (nk + 2) * np),
                bf(&self.d_clats, np),
                np,
                ik,
                nth,
            );
        }
    }

    fn tick(&mut self, what: usize, t: &mut std::time::Instant) {
        if self.profile {
            self.sync();
        }
        let e = t.elapsed().as_secs_f64();
        match what {
            0 => self.timers.forcing += e,
            1 => self.timers.iter += e,
            2 => self.timers.ktp3 += e,
            3 => self.timers.geographic += e,
            _ => self.timers.outputs += e,
        }
        *t = std::time::Instant::now();
    }

    /// First part of a global step of `W3WAVE`: time, forcing, boundary,
    /// ice, levels and the sub-cycling count. Returns `(NTLOC, ITLOCH, FACTH, FACK)`.
    pub fn prepare(&mut self) -> Result<(usize, usize, f32, f32)> {
        let mut t = std::time::Instant::now();
        let dtg = self.cfg.dtg;
        let t_start = self.time;
        // ww3_shel reads fields at the start of each W3WAVE call; calls start
        // at every field or output time, and at other step starts no record
        // is due, so reading at every step start is equivalent.
        self.read_fields(t_start)?;
        self.time = Time::from_secs(t_start.secs() + dtg as i64);
        self.itstep += 1;
        self.update_currents();
        self.update_boundary()?;
        self.apply_boundary();
        self.update_ice()?;
        self.update_level()?;
        self.tick(0, &mut t);
        self.compute_iter();
        self.tick(1, &mut t);
        let ntloc = 1 + ((dtg / self.cfg.dtkth - 0.001) as i64).max(0) as usize;
        let facth = dtg / (self.sp.dth * ntloc as f32);
        let fack = dtg / ntloc as f32;
        let day0 = Time(self.time.0, 0).secs();
        let istep = ((self.time.secs() - day0) as f64 / dtg as f64).round() as i64;
        let itloch = ((ntloc as i64 + 1 - istep.rem_euclid(2)) / 2) as usize;
        Ok((ntloc, itloch, facth, fack))
    }

    /// Second part: spectral, geographic, spectral propagation.
    pub fn propagate(&mut self, ntloc: usize, itloch: usize, facth: f32, fack: f32) {
        let mut t = std::time::Instant::now();
        for _ in 0..itloch {
            self.ktp3(facth, fack);
        }
        self.tick(2, &mut t);
        if self.cfg.flcx {
            for ik in 0..self.nk {
                self.geographic(ik);
            }
        }
        self.tick(3, &mut t);
        for _ in itloch..ntloc {
            self.ktp3(facth, fack);
        }
        self.tick(2, &mut t);
    }

    /// One global step of `W3WAVE`.
    pub fn step(&mut self) -> Result<()> {
        let (ntloc, itloch, facth, fack) = self.prepare()?;
        self.propagate(ntloc, itloch, facth, fack);
        Ok(())
    }

    /// Geographic step of frequency `ik` on the GPU against the CPU
    /// transcription of `PDLIB_W3XYPUG` (call after [`Model::prepare`]).
    /// Returns (max |gpu - cpu| / max |cpu|, rms ratio, directions whose ITER differs).
    pub fn check_geo(&mut self, geo: &Geometry, ik: usize) -> (f64, f64, usize) {
        let (np, nk, nth) = (self.np, self.nk, self.nth);
        let nspec = nk * nth;
        let all = self.download(&self.d_va);
        let mut va: Vec<f32> = all[ik * np * nth..(ik + 1) * np * nth].to_vec();
        drop(all);
        let cx = self.download(&self.d_cx);
        let cy = self.download(&self.d_cy);
        let clats: Vec<f32> = self.lat.iter().map(|&y| (DERA * y).cos()).collect();
        let iobdp: Vec<u8> = self.dw.iter().map(|&d| (d >= self.dmin + 1e-5) as u8).collect();
        let mut bb0 = vec![0f32; self.nbnd * nth];
        let mut bbn = vec![0f32; self.nbnd * nth];
        for b in 0..self.nbnd {
            for th in 0..nth {
                bb0[b * nth + th] = self.bbpi0[b * nspec + ik * nth + th];
                bbn[b * nth + th] = self.bbpin[b * nspec + ik * nth + th];
            }
        }
        let (rd10, rd20) = self.boundary_rd();
        let cg = self.cg[(ik + 1) * np..(ik + 2) * np].to_vec();
        let g = crate::cpu::GeoFreq {
            ine: &geo.ine,
            ien: &geo.ien,
            si: &geo.si,
            iobp: &self.iobp,
            iobpa: &self.iobpa,
            iobpd: &self.iobpd,
            iobdp: &iobdp,
            cg: &cg,
            clats: &clats,
            cx: &cx,
            cy: &cy,
            ecos: &self.sp.ecos,
            esin: &self.sp.esin,
            facx: 1.0 / (DERA * RADIUS),
            dt: self.cfg.dtg,
            bnode: &self.bnode,
            bb0: &bb0,
            bbn: &bbn,
            rd10,
            rd20,
        };
        let iters = crate::cpu::geo_freq(&g, &mut va, nth);
        self.geographic(ik);
        let all = self.download(&self.d_va);
        let gpu = &all[ik * np * nth..(ik + 1) * np * nth];
        let vmax = va.iter().cloned().fold(0f32, f32::max).max(1e-30) as f64;
        let (mut dmax, mut d2, mut v2) = (0f64, 0f64, 0f64);
        for j in 0..np * nth {
            let d = (gpu[j] - va[j]) as f64;
            dmax = dmax.max(d.abs());
            d2 += d * d;
            v2 += (va[j] as f64).powi(2);
        }
        let bad = (0..nth).filter(|&th| iters[th] != self.iter_host[ik * nth + th]).count();
        (dmax / vmax, (d2 / v2.max(1e-300)).sqrt(), bad)
    }

    /// H_s and T_m02 of every node (`UNDEF` where `MAPSTA <= 0`).
    /// `W3OUTG` on the device: H_s, T_m02 and the energy of every node.
    fn launch_outputs(&self) {
        let (np, nk, nth) = (self.np, self.nk, self.nth);
        let (c, d) = self.dims(np);
        unsafe {
            prop::outputs::launch_unchecked::<F>(
                &self.client,
                c,
                d,
                bf(&self.d_va, nk * np * nth),
                bf(&self.d_cg, (nk + 2) * np),
                bf(&self.d_sig, nk + 2),
                bf(&self.d_dden, nk),
                bf(&self.d_nflag, np),
                bf(&self.d_hs, np),
                bf(&self.d_t02, np),
                bf(&self.d_en, np),
                self.sp.fte,
                self.sp.dth,
                UNDEF,
                np,
                nk,
                nth,
            );
        }
    }

    pub fn outputs(&mut self) -> (Vec<f32>, Vec<f32>) {
        let mut t = std::time::Instant::now();
        self.launch_outputs();
        let hs = self.download(&self.d_hs);
        let t02 = self.download(&self.d_t02);
        self.tick(4, &mut t);
        (hs, t02)
    }

    /// Total energy `sum_i sum_k DDEN(k) / CG(k, i) sum_th VA(k, th, i)` over
    /// the nodes with `MAPSTA > 0` (per-node sums on the device, total in double).
    pub fn total_energy(&self) -> f64 {
        self.launch_outputs();
        self.download(&self.d_en).iter().map(|&x| x as f64).sum()
    }
}

/// Lists produced by [`level_chunk`] for the nodes of one chunk.
struct LevelChunk {
    zero: Vec<u32>,
    rnodes: Vec<u32>,
    own: Vec<f32>,
    dwn: Vec<f32>,
    i1v: Vec<u32>,
    r1v: Vec<f32>,
    tail: Vec<u32>,
}

/// `W3ULEV` for the nodes `lo .. lo + mapsta.len()`: new wavenumbers and group
/// speeds (`wn[ik][i - lo]`, `cg[ik][i - lo]`, updated in place), drying and
/// wetting (`MAPSTA`, `MAPST2` in place, nodes to zero) and the weights of
/// the spectral remapping of the nodes whose wavenumbers moved by 5 % of a bin.
#[allow(clippy::too_many_arguments)]
fn level_chunk(
    sp: &Spec,
    dmin: f32,
    lo: usize,
    wlv: &[f32],
    zb: &[f32],
    dw: &[f32],
    deep: &[bool],
    mut wn: Vec<&mut [f32]>,
    mut cg: Vec<&mut [f32]>,
    mapsta: &mut [i32],
    mapst2: &mut [i32],
) -> LevelChunk {
    let nk = sp.nk;
    let mut o = LevelChunk { zero: vec![], rnodes: vec![], own: vec![], dwn: vec![], i1v: vec![], r1v: vec![], tail: vec![] };
    let mut wno = vec![0f32; nk + 2];
    let mut cgo = vec![0f32; nk + 2];
    let mut wnn = vec![0f32; nk + 2];
    let mut cgn = vec![0f32; nk + 2];
    for j in 0..mapsta.len() {
        let i = lo + j;
        if deep[i] {
            continue;
        }
        for ik in 0..nk + 2 {
            wno[ik] = wn[ik][j];
            cgo[ik] = cg[ik][j];
        }
        let depth = dmin.max(dw[i]);
        for ik in 0..nk + 2 {
            let (k, c) = crate::spec::wavnu3(sp.sig[ik], depth);
            wnn[ik] = k;
            cgn[ik] = c;
            wn[ik][j] = k;
            cg[ik][j] = c;
        }
        let dry = mapst2[j] & 2 == 2;
        if wlv[i] - zb[i] <= 0.0 {
            if !dry {
                o.zero.push(i as u32);
                mapst2[j] |= 2;
                mapsta[j] = -mapsta[j].abs();
            }
            continue;
        }
        if dry {
            mapst2[j] &= !2;
            o.zero.push(i as u32);
            if mapst2[j] == 0 {
                mapsta[j] = mapsta[j].abs();
            }
            continue;
        }
        // DSIP(1..NK) / CG at index ik (1-based ik -> array index ik)
        let dwn1 = sp.dsip[1] / cgn[1];
        let rdk = (wno[1] - wnn[1]).abs() / dwn1;
        if rdk < 0.05 || mapsta[j] < 0 {
            continue;
        }
        // remapping weights (W3ULEV), 1-based bins
        let mut wi1 = vec![nk as u32 + 1; nk];
        let mut wr1 = vec![0f32; nk];
        let (mut ik0, mut ii1, mut ii2): (usize, usize, usize);
        let mut cond = true;
        if wno[1] < wnn[1] {
            ik0 = 1;
            ii1 = 0;
            ii2 = 1;
            loop {
                ik0 += 1;
                if ik0 > nk + 1 {
                    cond = false;
                    break;
                }
                if wno[ik0] >= wnn[1] {
                    ik0 -= 1;
                    break;
                }
            }
        } else {
            ik0 = 1;
            ii1 = 1;
            ii2 = 2;
        }
        let mut rd2 = 1.0f32;
        if cond {
            for ik in ik0..=nk {
                while wno[ik] > wnn[ii2] {
                    ii1 += 1;
                    if ii1 > nk {
                        break;
                    }
                    ii2 = ii1 + 1;
                }
                if ii1 > nk {
                    continue;
                }
                let rd1 = if ii1 == 0 { (wnn[1] - wno[ik]) / dwn1 } else { (wnn[ii2] - wno[ik]) / (wnn[ii2] - wnn[ii1]) };
                rd2 = 1.0 - rd1;
                wi1[ik - 1] = ii1 as u32;
                wr1[ik - 1] = rd1;
            }
        } else {
            ii2 = nk + 1;
        }
        let t0 = if ii2 <= nk && rd2 <= 0.95 { ii2.max(2) as u32 } else { nk as u32 + 1 };
        o.rnodes.push(i as u32);
        for ik in 1..=nk {
            o.own.push(sp.dsip[ik] / cgo[ik]);
            o.dwn.push(sp.dsip[ik] / cgn[ik]);
        }
        o.i1v.extend_from_slice(&wi1);
        o.r1v.extend_from_slice(&wr1);
        o.tail.push(t0);
    }
    o
}

/// `W3UBPT`: `BBPI(isp, ibi) = Σ_j RDBPI(ibi, j) ABPI(isp, IPBPI(ibi, j)) / SIG2(isp)`.
fn interp_boundary(abpi: &[Vec<f32>], ip: &[i32], rd: &[f32], nbnd: usize, nspec: usize, nth: usize, sig: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; nbnd.max(1) * nspec];
    for b in 0..nbnd {
        for isp in 0..nspec {
            let mut v = 0f32;
            for j in 0..4 {
                let p = ip[4 * b + j];
                if p >= 1 {
                    v += rd[4 * b + j] * abpi[(p - 1) as usize][isp];
                }
            }
            out[b * nspec + isp] = v / sig[1 + isp / nth];
        }
    }
    out
}
