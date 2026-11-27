//! Milestone M4: the CUDA C adjoint of the propagation ([`crate::cuda_prop`])
//! on the buffers of the model. CubeCL and `cudarc` share the primary context
//! of the device, so the CUDA C kernels run on the raw pointers of the
//! model's buffers (`get_resource`), with a synchronisation of each side
//! before the other reads.

use super::*;
use crate::adjoint::Lcg;
use crate::cuda_prop::{GeoPtrs, Ktp3Ptrs, PropCuda};

// The CUDA server of CubeCL, for `get_resource` (CubeCL 0.11 exports the
// `Runtime` trait to its macros only).
type CudaSrv = <cubecl::cuda::CudaRuntime as cubecl::__private::Runtime>::Server;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl<F: Real> Model<F> {
    /// The raw device pointer of a buffer of the model.
    pub(crate) fn dptr(&self, h: &Handle) -> u64 {
        self.client.get_resource::<CudaSrv>(h.clone()).expect("a CUDA buffer").resource().ptr
    }

    /// The arrays of `W3KTP3` for [`PropCuda::theta`] and [`PropCuda::kshift`].
    pub(crate) fn ktp3_ptrs(&self) -> Ktp3Ptrs {
        Ktp3Ptrs {
            wn: self.dptr(&self.d_wn),
            cg: self.dptr(&self.d_cg),
            sig: self.dptr(&self.d_sig),
            dsip: self.dptr(&self.d_dsip),
            dw: self.dptr(&self.d_dw),
            dddx: self.dptr(&self.d_dddx),
            dddy: self.dptr(&self.d_dddy),
            cx: self.dptr(&self.d_cx),
            cy: self.dptr(&self.d_cy),
            dcxdx: self.dptr(&self.d_dcxdx),
            dcxdy: self.dptr(&self.d_dcxdy),
            dcydx: self.dptr(&self.d_dcydx),
            dcydy: self.dptr(&self.d_dcydy),
            cthg0: self.dptr(&self.d_cthg0),
            nflag: self.dptr(&self.d_nflag),
            ecos: self.dptr(&self.d_ecos),
            esin: self.dptr(&self.d_esin),
            es2: self.dptr(&self.d_es2),
            ec2: self.dptr(&self.d_ec2),
            esc: self.dptr(&self.d_esc),
            ctmax: self.cfg.ctmax as f64,
            flcur: if self.cur.is_some() { 1.0 } else { 0.0 },
            fachfa: self.sp.fachfa as f64,
            dmin: self.dmin as f64,
            np: self.np as u32,
        }
    }

    /// The arrays of the geographic sub-steps for [`PropCuda::geographic_adj`].
    pub(crate) fn geo_ptrs(&self) -> GeoPtrs {
        let (rd10, rd20) = self.boundary_rd();
        GeoPtrs {
            coef: self.dptr(&self.d_coef),
            ccoef: self.dptr(&self.d_ccoef),
            ecos: self.dptr(&self.d_ecos),
            esin: self.dptr(&self.d_esin),
            dtk: self.dptr(&self.d_dtk),
            iter: self.dptr(&self.d_iter),
            si_inv: self.dptr(&self.d_si_inv),
            mlo: self.dptr(&self.d_mlo),
            mhi: self.dptr(&self.d_mhi),
            start: self.dptr(&self.d_start),
            slot: self.dptr(&self.d_slot),
            bidx: self.dptr(&self.d_bidx),
            ine: self.dptr(&self.d_ine),
            bb0: self.dptr(&self.d_bb0),
            bbn: self.dptr(&self.d_bbn),
            clats: self.dptr(&self.d_clats),
            rd10: rd10 as f64,
            rd20: rd20 as f64,
            dtg: self.cfg.dtg as f64,
            np: self.np as u32,
            ne: self.ne as u32,
        }
    }

    /// The adjoint of one call of `W3KTP3` ([`Model::ktp3`]): `va` the
    /// spectra before the call (the CUDA C value of the first part is
    /// recomputed from it into `work`), `ab` the cotangent after the call,
    /// replaced by the cotangent before it.
    pub(crate) fn ktp3_adj(&self, pc: &PropCuda, va: &Handle, work: &Handle, ab: &Handle, facth: f32, fack: f32) -> Result<()> {
        let kp = self.ktp3_ptrs();
        let (pva, pw, pab) = (self.dptr(va), self.dptr(work), self.dptr(ab));
        let n = self.nk * self.np * self.nth;
        let k_first = self.itstep % 2 == 0;
        let (th, kk) = (self.cfg.flcth, self.cfg.flck);
        self.sync();
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(pw, pva, n * 8, pc.st.cu_stream())?;
        }
        if k_first {
            if kk {
                pc.kshift(pw, None, &kp, fack as f64)?;
            }
            if th {
                pc.theta(pw, Some(pab), &kp, facth as f64)?;
            }
            if kk {
                pc.kshift(pva, Some(pab), &kp, fack as f64)?;
            }
        } else {
            if th {
                pc.theta(pw, None, &kp, facth as f64)?;
            }
            if kk {
                pc.kshift(pw, Some(pab), &kp, fack as f64)?;
            }
            if th {
                pc.theta(pva, Some(pab), &kp, facth as f64)?;
            }
        }
        pc.st.synchronize()?;
        Ok(())
    }

    /// Milestone M4 (`hww3-prop adj-prop`): the dot-product tests of the CUDA
    /// C adjoint of the propagation against the tangent kernels of
    /// [`crate::prop_tl`] on the state after [`Model::prepare`], for the
    /// refraction, the frequency shift, the geographic propagation of three
    /// frequencies and the whole propagation of the step
    /// ([`Model::propagate`]): `<adjoint(y), x>` against `<y, tangent(x)>`
    /// for random `x` (relative to the spectra) and `y`, with the times of
    /// the tangent and of the adjoint.
    pub fn adj_prop_check(&mut self) -> Result<serde_json::Value> {
        if std::mem::size_of::<F>() != 8 {
            bail!("adj-prop needs the model in double precision (--f64)");
        }
        let (np, ne, nk, nth) = (self.np, self.ne, self.nk, self.nth);
        let n = np * nth;
        let ntot = nk * n;
        let (ntloc, itloch, facth, fack) = self.prepare()?;
        let pc = PropCuda::new(nk, nth)?;
        let va0 = self.download_f64(&self.d_va);
        let mut rng = Lcg(2026);
        let dva: Vec<f64> = rng.vec(ntot, -1.0, 1.0).iter().zip(&va0).map(|(r, x)| r * x).collect();
        let ybar = rng.vec(ntot, -1.0, 1.0);
        let fb = std::mem::size_of::<F>();
        let e = |k: usize| self.client.empty(k.max(1) * fb);
        let tl = Tangent {
            wind: false,
            d_vad: self.upd(&dva),
            d_dac: e(n),
            d_dac2: e(n),
            d_dut: e(ne * nth),
            d_du10: e(np),
            d_du10d: e(np),
            d_dpar: e(1),
            d_dtauhf: e(1),
            d_dhs: e(np),
            d_dt02: e(np),
            d_den: e(np),
        };
        let saved = self.tl.replace(tl);
        let mut rows = vec![];
        let geo_ik = [0usize, nk / 2, nk - 1];
        let mut parts: Vec<(String, Option<usize>)> = vec![("theta".into(), None), ("k".into(), None)];
        if self.cfg.flcx {
            for ik in geo_ik {
                parts.push((format!("geographic ik={ik}"), Some(ik)));
            }
        }
        parts.push(("propagate".into(), None));
        println!(
            "adj-prop at {}: NTLOC {ntloc}, ITLOCH {itloch}, FACTH {facth}, FACK {fack}, refraction {}, frequency shift {}, geographic {}",
            self.time, self.cfg.flcth, self.cfg.flck, self.cfg.flcx
        );
        println!("{:<22} {:>14} {:>14} {:>10} {:>10} {:>10}", "part", "<adj(y),x>", "<y,tl(x)>", "rel", "tl (s)", "adj (s)");
        for (name, gik) in parts {
            // the tangent from va0 along dva
            self.d_va = self.upd(&va0);
            let h = self.upd(&dva);
            if let Some(t) = self.tl.as_mut() {
                t.d_vad = h;
            }
            self.sync();
            let t0 = std::time::Instant::now();
            match (name.as_str(), gik) {
                ("theta", _) => self.ktp3_theta(facth),
                ("k", _) => self.ktp3_k(fack),
                (_, Some(ik)) => self.geographic(ik),
                _ => self.propagate(ntloc, itloch, facth, fack),
            }
            self.sync();
            let t_tl = t0.elapsed().as_secs_f64();
            let tlout = self.download_f64(&self.tl.as_ref().expect("tangent").d_vad);
            let rhs = dot(&ybar, &tlout);
            // the adjoint of ybar at va0
            let d_va0 = self.upd(&va0);
            let d_ab = self.upd(&ybar);
            let d_w = self.upd(&va0);
            self.sync();
            let t1 = std::time::Instant::now();
            let kp = self.ktp3_ptrs();
            match (name.as_str(), gik) {
                ("theta", _) => {
                    pc.theta(self.dptr(&d_va0), Some(self.dptr(&d_ab)), &kp, facth as f64)?;
                    pc.st.synchronize()?;
                }
                ("k", _) => {
                    pc.kshift(self.dptr(&d_va0), Some(self.dptr(&d_ab)), &kp, fack as f64)?;
                    pc.st.synchronize()?;
                }
                (_, Some(ik)) => {
                    let g = self.geo_ptrs();
                    pc.geographic_adj(ik, self.max_iter(ik) as usize, self.dptr(&d_va0), self.dptr(&d_ab), self.dptr(&self.d_cg), &g)?;
                }
                _ => {
                    // the values of the step from va0 (the model without its
                    // tangent), the spectra kept before each W3KTP3 call and
                    // before the geographic propagation, then the transposes
                    let tl = self.tl.take();
                    self.d_va = self.upd(&va0);
                    let mut before = vec![];
                    for _ in 0..itloch {
                        before.push(self.upd(&self.download_f64(&self.d_va)));
                        self.ktp3(facth, fack);
                    }
                    let d_g = self.upd(&self.download_f64(&self.d_va));
                    if self.cfg.flcx {
                        for ik in 0..nk {
                            self.geographic(ik);
                        }
                    }
                    for _ in itloch..ntloc {
                        before.push(self.upd(&self.download_f64(&self.d_va)));
                        self.ktp3(facth, fack);
                    }
                    self.tl = tl;
                    self.sync();
                    for c in (itloch..ntloc).rev() {
                        self.ktp3_adj(&pc, &before[c], &d_w, &d_ab, facth, fack)?;
                    }
                    if self.cfg.flcx {
                        let g = self.geo_ptrs();
                        for ik in (0..nk).rev() {
                            pc.geographic_adj(ik, self.max_iter(ik) as usize, self.dptr(&d_g), self.dptr(&d_ab), self.dptr(&self.d_cg), &g)?;
                        }
                    }
                    for c in (0..itloch).rev() {
                        self.ktp3_adj(&pc, &before[c], &d_w, &d_ab, facth, fack)?;
                    }
                }
            }
            let t_adj = t1.elapsed().as_secs_f64();
            let abar = self.download_f64(&d_ab);
            let lhs = dot(&abar, &dva);
            let rel = (lhs - rhs).abs() / rhs.abs().max(1e-300);
            println!("{name:<22} {lhs:>14.6e} {rhs:>14.6e} {rel:>10.2e} {t_tl:>10.4} {t_adj:>10.4}");
            rows.push(serde_json::json!({"part": name, "adj": lhs, "tl": rhs, "rel": rel, "t_tl": t_tl, "t_adj": t_adj}));
        }
        self.tl = saved;
        self.d_va = self.upd(&va0);
        Ok(serde_json::json!({"time": self.time.to_string(), "ntloc": ntloc, "itloch": itloch, "rows": rows}))
    }
}

// ---------------------------------------------------------------------------
// Checkpoints of the model and the adjoint of a whole step
// ---------------------------------------------------------------------------

/// The state of the model at the start of a step, all that a step reads from
/// the steps before it: a checkpoint of the adjoint. The spectra, wavenumbers
/// and group speeds are kept exactly (double precision); the forcing readers
/// are kept as the times of their records and reopened on restore.
pub struct Snap {
    time: Time,
    itstep: u64,
    va: Vec<f64>,
    wn: Vec<f64>,
    cg: Vec<f64>,
    dw: Vec<f32>,
    wlv: Vec<f32>,
    mapsta: Vec<i32>,
    mapst2: Vec<i32>,
    ice_c: Vec<f32>,
    tlev: Time,
    tice: Time,
    lev: Option<(Option<Time>, bool)>,
    ice: Option<(Option<Time>, bool)>,
    tc0: Option<Time>,
    tcn: Option<Time>,
    c0: (Vec<f32>, Vec<f32>),
    cn: (Vec<f32>, Vec<f32>),
    tw0: Option<Time>,
    twn: Option<Time>,
    w0: (Vec<f32>, Vec<f32>),
    wnx: (Vec<f32>, Vec<f32>),
    tbpi0: Option<Time>,
    tbpin: Option<Time>,
    bbpi0: Vec<f32>,
    bbpin: Vec<f32>,
}

impl Snap {
    /// Host memory of the checkpoint (bytes).
    pub fn bytes(&self) -> usize {
        8 * (self.va.len() + self.wn.len() + self.cg.len())
            + 4 * (self.dw.len() + self.wlv.len() + self.mapsta.len() + self.mapst2.len() + self.ice_c.len())
            + 4 * (self.c0.0.len() * 4 + self.w0.0.len() * 4 + self.bbpi0.len() + self.bbpin.len())
    }

    /// Model time at the start of the step.
    pub fn time(&self) -> Time {
        self.time
    }
}

/// A forcing file reopened and read up to the record of time `until` (none:
/// nothing read), then to its end when `eof`.
fn reopen(p: &PathBuf, until: Option<Time>, eof: bool) -> Result<Fields> {
    let mut f = Fields::open(p)?;
    if let Some(t) = until {
        loop {
            if !f.advance()? {
                bail!("{}: no record at {t}", p.display());
            }
            if f.tnext() == Some(t) {
                break;
            }
        }
    }
    if eof {
        while f.advance()? {}
    }
    Ok(f)
}

/// The cotangents gathered over the steps by the adjoint.
#[derive(Clone, Debug)]
pub struct StepBar {
    /// Cotangents of `BETAMAX`, `SDSC2`, `NLPROP`, `BJGAM` and of the
    /// coefficients of `BETAMAX(U)` ([`crate::cuda_adj::NPB`] values, summed
    /// over the nodes and the steps).
    pub par: Vec<f64>,
    /// Time of the adjoint of the source terms, of the propagation and of the
    /// recomputed steps (s).
    pub secs: [f64; 3],
}

/// The CUDA C kernels of the adjoint of a step.
pub struct StepAdj {
    /// The propagation.
    pub prop: PropCuda,
    /// The source terms.
    pub srce: crate::cuda_adj::SrceCuda,
    /// Size of the tape of the source terms per batch of nodes (bytes).
    pub tape_max: usize,
}

impl<F: Real> Model<F> {
    /// The checkpoint of the state at the start of the next step.
    pub fn snapshot(&self) -> Snap {
        let rd = |f: &Option<Fields>| f.as_ref().map(|x| (x.tnext(), x.eof));
        Snap {
            time: self.time,
            itstep: self.itstep,
            va: self.download_f64(&self.d_va),
            wn: self.download_f64(&self.d_wn),
            cg: self.download_f64(&self.d_cg),
            dw: self.dw.clone(),
            wlv: self.wlv.clone(),
            mapsta: self.mapsta.clone(),
            mapst2: self.mapst2.clone(),
            ice_c: self.ice_c.clone(),
            tlev: self.tlev,
            tice: self.tice,
            lev: rd(&self.lev),
            ice: rd(&self.ice),
            tc0: self.tc0,
            tcn: self.tcn,
            c0: self.c0.clone(),
            cn: self.cn.clone(),
            tw0: self.tw0,
            twn: self.twn,
            w0: self.w0.clone(),
            wnx: self.wnx.clone(),
            tbpi0: self.tbpi0,
            tbpin: self.tbpin,
            bbpi0: self.bbpi0.clone(),
            bbpin: self.bbpin.clone(),
        }
    }

    /// Back to the checkpoint `s`, the forcing files of `inp` reopened and
    /// read up to the records the model held then; the next step then gives
    /// the same values bit for bit as the run the checkpoint was taken from.
    pub fn restore(&mut self, s: &Snap, inp: &Inputs) -> Result<()> {
        if self.cfg.host_level {
            bail!("the adjoint restores the device level update only (not --host-level)");
        }
        self.time = s.time;
        self.itstep = s.itstep;
        self.d_va = self.upd(&s.va);
        self.d_wn = self.upd(&s.wn);
        self.d_cg = self.upd(&s.cg);
        self.dw = s.dw.clone();
        self.wlv = s.wlv.clone();
        self.mapsta = s.mapsta.clone();
        self.mapst2 = s.mapst2.clone();
        self.ice_c = s.ice_c.clone();
        self.tlev = s.tlev;
        self.tice = s.tice;
        if let (Some(p), Some((t, e))) = (&inp.level, s.lev) {
            self.lev = Some(reopen(p, t, e)?);
        }
        if let (Some(p), Some((t, e))) = (&inp.ice, s.ice) {
            self.ice = Some(reopen(p, t, e)?);
        }
        if let Some(p) = &inp.current {
            let mut f = reopen(p, s.tcn, false)?;
            f.next = None;
            self.cur = Some(f);
            self.tc0 = s.tc0;
            self.tcn = s.tcn;
            self.c0 = s.c0.clone();
            self.cn = s.cn.clone();
            if s.tcn.is_some() {
                self.current_coefficients();
            }
        }
        if let Some(p) = &inp.wind {
            let mut f = reopen(p, s.twn, false)?;
            f.next = None;
            self.wnd = Some(f);
            self.tw0 = s.tw0;
            self.twn = s.twn;
            self.w0 = s.w0.clone();
            self.wnx = s.wnx.clone();
            if s.twn.is_some() {
                self.wind_coefficients();
            }
        }
        if let Some(p) = &inp.nest {
            let mut nf = NestFile::open(p)?;
            if let Some(t) = s.tbpin {
                loop {
                    match nf.next()? {
                        Some((tt, _)) if tt == t => break,
                        Some(_) => {}
                        None => bail!("{}: no boundary record at {t}", p.display()),
                    }
                }
            }
            self.nest = Some(nf);
            self.tbpi0 = s.tbpi0;
            self.tbpin = s.tbpin;
            self.bbpi0 = s.bbpi0.clone();
            self.bbpin = s.bbpin.clone();
            if s.tbpin.is_some() {
                self.d_bb0 = self.upf(&self.bbpi0);
                self.d_bbn = self.upf(&self.bbpin);
            }
        }
        self.d_dw = self.upf(&self.dw);
        self.upload_flags();
        self.depth_kernels();
        self.d_ice = self.upf(&self.ice_c);
        Ok(())
    }

    /// The derivative of the tail-stress table of the source kernel in
    /// `BETAMAX` (`TABU_TAUHF2` in double precision at the `BETAMAX` of the
    /// parameter file of `inp`), laid out as the table; zeros with tables in
    /// `BETAMAX` ([`crate::sources::P_BT_N`] > 1), whose derivative the kernels
    /// interpolate themselves.
    pub fn dtauhf_unit(&self, inp: &Inputs) -> Result<Vec<f64>> {
        use crate::sources::{tabu_tauhf2, P_BT_N};
        let t = self.src_tables.as_ref().ok_or_else(|| anyhow!("the adjoint needs the source terms (--sources)"))?;
        if t.par[P_BT_N] > 1.0 {
            return Ok(vec![0.0; t.tauhf.len()]);
        }
        let p = inp.sources.as_ref().ok_or_else(|| anyhow!("no parameter file of the sources"))?;
        let q = SrcePar::new(&Params::read(p)?)?;
        let stt = q.st.tables.as_ref().ok_or_else(|| anyhow!("no stress tables (SINTABLE = 0)"))?;
        let (_, d1, _, d2) = tabu_tauhf2(&q.st, stt, q.st.sig[q.st.nk], q.st.bbeta);
        let d = if q.st.ttauwshelter > 0.0 { d2 } else { d1 };
        if d.len() != t.tauhf.len() {
            bail!("tail-stress table of {} values against {}", d.len(), t.tauhf.len());
        }
        Ok(d)
    }

    /// The CUDA C kernels of the adjoint of a step for this model: the
    /// propagation and the source terms (with the tables of the source
    /// kernel and `dtauhf`, the derivative of the tail-stress table in
    /// `BETAMAX`, laid out as the table).
    pub fn step_adj_kernels(&self, dtauhf: &[f64], tape_max: usize) -> Result<StepAdj> {
        if std::mem::size_of::<F>() != 8 {
            bail!("the adjoint needs the model in double precision (--f64)");
        }
        if self.iceref.is_some() {
            bail!("the adjoint of the ice and reflection step is not written yet");
        }
        let t = self.src_tables.as_ref().ok_or_else(|| anyhow!("the adjoint needs the source terms (--sources)"))?;
        let mut srce = crate::cuda_adj::SrceCuda::new(t, self.nk, self.nth)?;
        srce.set_dtauhf(dtauhf)?;
        Ok(StepAdj { prop: PropCuda::new(self.nk, self.nth)?, srce, tape_max })
    }

    /// A device copy (`cudarc`) of the model's spectra.
    fn copy_va(&self, st: &std::sync::Arc<cudarc::driver::CudaStream>) -> Result<cudarc::driver::CudaSlice<f64>> {
        let n = self.nk * self.np * self.nth;
        let c = st.alloc_zeros::<f64>(n)?;
        let pc = raw_ptr(&c, st);
        self.sync();
        unsafe { cudarc::driver::result::memcpy_dtod_async(pc, self.dptr(&self.d_va), n * 8, st.cu_stream())? };
        st.synchronize()?;
        Ok(c)
    }

    /// The adjoint of one step from the checkpoint `s` at its start: the
    /// step is recomputed from `s` (the spectra kept before each part), then
    /// `ab`, the cotangent of the spectra at the end of the step, becomes the
    /// cotangent at its start, through the transposes of the source terms,
    /// the propagation and the parts of the forcing that change the spectra
    /// (the open boundary, the ice and drying transitions, the level remap).
    /// The cotangents of the parameters are added to `bar`. With `seed`, the
    /// cotangents of `H_s` and `T_m02` at the end of the step (per node) are
    /// first added to `ab` through the transpose of the outputs.
    pub fn step_adj(&mut self, k: &StepAdj, s: &Snap, inp: &Inputs, ab: &mut cudarc::driver::CudaSlice<f64>, bar: &mut StepBar, seed: Option<(&[f64], &[f64])>) -> Result<()> {
        use crate::cuda_adj::{SrceDev, NPB};
        let (np, nk, nth) = (self.np, self.nk, self.nth);
        let n = nk * np * nth;
        let pc = &k.prop;
        let st = pc.st.clone();
        let t0 = std::time::Instant::now();
        self.restore(s, inp)?;
        let (ntloc, itloch, facth, fack) = self.prepare()?;
        // what the forcing did to the spectra: the open boundary (every
        // step), the ice transitions, then the level update (drying and
        // wetting nodes zeroed, then the remap of the other nodes)
        let mut zero_ice = vec![];
        let mut zero_lev = vec![];
        let mut mode = vec![0u32; np];
        let level_done = self.tlev != s.tlev;
        for i in 0..np {
            let (b0, b1) = (s.mapst2[i] & 1, self.mapst2[i] & 1);
            let mid2 = if b0 != b1 { s.mapst2[i] ^ 1 } else { s.mapst2[i] };
            if b0 != b1 && (b1 == 1 || mid2 == 0) {
                zero_ice.push(i as u32);
            }
            if level_done && !self.deep[i] {
                let mut msa = s.mapsta[i];
                if b0 != b1 {
                    msa = if b1 == 1 { -msa.abs() } else if mid2 == 0 { msa.abs() } else { msa };
                }
                let dry = mid2 & 2 == 2;
                if self.wlv[i] - self.zb[i] <= 0.0 {
                    mode[i] = 1;
                    if !dry {
                        zero_lev.push(i as u32);
                    }
                } else if dry {
                    mode[i] = 1;
                    zero_lev.push(i as u32);
                } else {
                    mode[i] = if msa < 0 { 1 } else { 2 };
                }
            }
        }
        // the propagation, the spectra kept before each W3KTP3 call and
        // before the geographic part, then the source terms
        let mut before = vec![];
        for _ in 0..itloch {
            before.push(self.copy_va(&st)?);
            self.ktp3(facth, fack);
        }
        let d_g = self.copy_va(&st)?;
        if self.cfg.flcx {
            for ik in 0..nk {
                self.geographic(ik);
            }
        }
        for _ in itloch..ntloc {
            before.push(self.copy_va(&st)?);
            self.ktp3(facth, fack);
        }
        let d_src = self.copy_va(&st)?;
        self.sync();
        if let Some((hsb, t02b)) = seed {
            // the spectra at the end of the step, for the outputs
            self.sources();
            self.sync();
            let (d_h, d_t) = (st.clone_htod(hsb)?, st.clone_htod(t02b)?);
            pc.outputs_adj(
                self.dptr(&self.d_va),
                self.dptr(&self.d_cg),
                self.dptr(&self.d_sig),
                self.dptr(&self.d_dden),
                self.dptr(&self.d_nflag),
                raw_ptr(&d_h, &st),
                raw_ptr(&d_t, &st),
                raw_ptr(ab, &st),
                self.sp.fte as f64,
                self.sp.dth as f64,
                np as u32,
            )?;
            st.synchronize()?;
        }
        let t1 = std::time::Instant::now();
        bar.secs[2] += (t1 - t0).as_secs_f64();
        // the source terms
        let ab2 = st.alloc_zeros::<f64>(n)?;
        let d_ub = st.alloc_zeros::<f64>(np)?;
        let d_udb = st.alloc_zeros::<f64>(np)?;
        let d_pb = st.alloc_zeros::<f64>(np * NPB)?;
        if self.src.is_some() {
            let d = SrceDev {
                wn: self.dptr(&self.d_wn),
                cg: self.dptr(&self.d_cg),
                dw: self.dptr(&self.d_dw),
                u10: self.dptr(&self.d_u10),
                u10d: self.dptr(&self.d_u10d),
                ice: self.dptr(&self.d_ice),
                d50: self.dptr(&self.d_d50),
                psic: self.dptr(&self.d_psic),
                nflag: self.dptr(&self.d_nflag),
            };
            self.sync();
            k.srce.adjoint_dev(
                &d,
                raw_ptr(&d_src, &st),
                raw_ptr(ab, &st),
                raw_ptr(&ab2, &st),
                raw_ptr(&d_ub, &st),
                raw_ptr(&d_udb, &st),
                raw_ptr(&d_pb, &st),
                None,
                self.cfg.dtg as f64,
                np,
                k.tape_max,
            )?;
            let pb = st.clone_dtoh(&d_pb)?;
            for i in 0..np {
                for j in 0..NPB {
                    bar.par[j] += pb[i * NPB + j];
                }
            }
            unsafe { cudarc::driver::result::memcpy_dtod_async(raw_ptr(ab, &st), raw_ptr(&ab2, &st), n * 8, st.cu_stream())? };
        }
        st.synchronize()?;
        let t2 = std::time::Instant::now();
        bar.secs[0] += (t2 - t1).as_secs_f64();
        // the propagation
        let w = st.alloc_zeros::<f64>(n)?;
        let pab = raw_ptr(ab, &st);
        for c in (itloch..ntloc).rev() {
            self.ktp3_adj_raw(pc, raw_ptr(&before[c], &st), raw_ptr(&w, &st), pab, facth, fack)?;
        }
        if self.cfg.flcx {
            let g = self.geo_ptrs();
            let pcg = self.dptr(&self.d_cg);
            for ik in (0..nk).rev() {
                pc.geographic_adj(ik, self.max_iter(ik) as usize, raw_ptr(&d_g, &st), pab, pcg, &g)?;
            }
        }
        for c in (0..itloch).rev() {
            self.ktp3_adj_raw(pc, raw_ptr(&before[c], &st), raw_ptr(&w, &st), pab, facth, fack)?;
        }
        // the forcing: level remap, drying and wetting, ice, open boundary
        if level_done {
            let d_wno = st.clone_htod(&s.wn)?;
            let d_cgo = st.clone_htod(&s.cg)?;
            let d_mode = st.clone_htod(&mode)?;
            pc.ulev_node_adj(pab, raw_ptr(&d_wno, &st), raw_ptr(&d_cgo, &st), self.dptr(&self.d_wn), self.dptr(&self.d_cg), raw_ptr(&d_mode, &st), self.dptr(&self.d_dsip), self.sp.fachfa as f64, np as u32)?;
            if !zero_lev.is_empty() {
                let d_z = st.clone_htod(&zero_lev)?;
                pc.zero_nodes(pab, raw_ptr(&d_z, &st), np as u32, zero_lev.len())?;
            }
        }
        if !zero_ice.is_empty() {
            let d_z = st.clone_htod(&zero_ice)?;
            pc.zero_nodes(pab, raw_ptr(&d_z, &st), np as u32, zero_ice.len())?;
        }
        if self.nbnd > 0 {
            pc.zero_nodes(pab, self.dptr(&self.d_bnode), np as u32, self.nbnd)?;
        }
        st.synchronize()?;
        bar.secs[1] += t2.elapsed().as_secs_f64();
        Ok(())
    }

    /// [`Model::ktp3_adj`] on raw pointers (`va` the spectra before the call,
    /// `work` scratch, `ab` the cotangent).
    fn ktp3_adj_raw(&self, pc: &PropCuda, pva: u64, pw: u64, pab: u64, facth: f32, fack: f32) -> Result<()> {
        let kp = self.ktp3_ptrs();
        let n = self.nk * self.np * self.nth;
        let k_first = self.itstep % 2 == 0;
        let (th, kk) = (self.cfg.flcth, self.cfg.flck);
        unsafe { cudarc::driver::result::memcpy_dtod_async(pw, pva, n * 8, pc.st.cu_stream())? };
        if k_first {
            if kk {
                pc.kshift(pw, None, &kp, fack as f64)?;
            }
            if th {
                pc.theta(pw, Some(pab), &kp, facth as f64)?;
            }
            if kk {
                pc.kshift(pva, Some(pab), &kp, fack as f64)?;
            }
        } else {
            if th {
                pc.theta(pw, None, &kp, facth as f64)?;
            }
            if kk {
                pc.kshift(pw, Some(pab), &kp, fack as f64)?;
            }
            if th {
                pc.theta(pva, Some(pab), &kp, facth as f64)?;
            }
        }
        pc.st.synchronize()?;
        Ok(())
    }

    /// The adjoint of the `nsteps` steps from the present state, with two
    /// levels of checkpoints in host memory: a value run keeps the state
    /// every `k1` steps; then, from the last segment back, each segment is
    /// run again from its checkpoint keeping the state at the start of each
    /// of its steps, and its steps are taken in reverse ([`Model::step_adj`]).
    /// `ab` holds the cotangent of the spectra at the end on entry and at the
    /// start on return; `seeds` gives the cotangents of `H_s` and `T_m02` at
    /// the end of a step (none: no observation then). Returns the cotangents
    /// of the parameters and the largest number of checkpoints held at once.
    pub fn adjoint_run(
        &mut self,
        k: &StepAdj,
        inp: &Inputs,
        nsteps: usize,
        k1: usize,
        ab: &mut cudarc::driver::CudaSlice<f64>,
        seeds: &dyn Fn(Time) -> Result<Option<(Vec<f64>, Vec<f64>)>>,
    ) -> Result<(StepBar, usize)> {
        let k1 = k1.max(1);
        let mut bar = StepBar { par: vec![0.0; crate::cuda_adj::NPB], secs: [0.0; 3] };
        let mut l1 = vec![];
        for n in 0..nsteps {
            if n % k1 == 0 {
                l1.push(self.snapshot());
            }
            self.step()?;
        }
        let held = l1.len() + k1.min(nsteps);
        let dtg = self.cfg.dtg as i64;
        for (j, c) in l1.iter().enumerate().rev() {
            let (n0, n1) = (j * k1, ((j + 1) * k1).min(nsteps));
            self.restore(c, inp)?;
            let mut l2 = vec![];
            for _ in n0..n1 {
                l2.push(self.snapshot());
                self.step()?;
            }
            for sn in l2.iter().rev() {
                let t_end = Time::from_secs(sn.time.secs() + dtg);
                let sd = seeds(t_end)?;
                self.step_adj(k, sn, inp, ab, &mut bar, sd.as_ref().map(|(h, t)| (h.as_slice(), t.as_slice())))?;
            }
        }
        Ok((bar, held))
    }

    /// Milestone M4 (`hww3-prop adj-step`): the dot-product tests of the
    /// adjoint of `nsteps` whole steps from the state of the model, against
    /// the tangent of the same steps ([`crate::prop_tl`], [`crate::srce_tl`]):
    /// `<adjoint(y), x>` against `<y, tangent(x)>` for `x` a random change of
    /// the spectra at the start, then a relative change of `BETAMAX`, `SDSC2`,
    /// `NLPROP` and `BJGAM` each, and `y` a random cotangent of the spectra
    /// at the end. `dtauhf` is the derivative of the tail-stress table in
    /// `BETAMAX` (for the tangent of `BETAMAX`, laid out as the table).
    pub fn adj_step_check(&mut self, inp: &Inputs, nsteps: usize, k1: usize, dtauhf: &[f64]) -> Result<serde_json::Value> {
        use crate::sources::{NPAR, P_BBETA, P_BT_N, P_SDBC2, P_SDSC2, P_SNLC1};
        let (np, ne, nk, nth) = (self.np, self.ne, self.nk, self.nth);
        let n = nk * np * nth;
        let k = self.step_adj_kernels(dtauhf, 8 << 30)?;
        let st = k.prop.st.clone();
        let s0 = self.snapshot();
        let mut rng = Lcg(2027);
        let dva: Vec<f64> = rng.vec(n, -1.0, 1.0).iter().zip(&s0.va).map(|(r, x)| r * x).collect();
        let ybar = rng.vec(n, -1.0, 1.0);
        let par = self.src_tables.as_ref().expect("sources").par.clone();
        let fb = std::mem::size_of::<F>();
        let client = self.client.clone();
        let e = |c: usize| client.empty(c.max(1) * fb);
        let classes = ["spectrum", "BETAMAX", "SDSC2", "NLPROP", "BJGAM"];
        let mut rhs = vec![];
        let mut t_tl = 0.0;
        for name in classes {
            let mut dpar = vec![0f64; NPAR];
            let mut dtab = vec![0f64; dtauhf.len().max(1)];
            let dv = if name == "spectrum" { dva.clone() } else { vec![0f64; n] };
            match name {
                "BETAMAX" => {
                    dpar[P_BBETA] = par[P_BBETA];
                    if par[P_BT_N] <= 1.0 {
                        dtab = dtauhf.iter().map(|x| x * par[P_BBETA]).collect();
                    }
                }
                "SDSC2" => dpar[P_SDSC2] = par[P_SDSC2],
                "NLPROP" => dpar[P_SNLC1] = par[P_SNLC1],
                "BJGAM" => dpar[P_SDBC2] = par[P_SDBC2],
                _ => {}
            }
            self.restore(&s0, inp)?;
            let tl = Tangent {
                wind: false,
                d_vad: self.upd(&dv),
                d_dac: e(np * nth),
                d_dac2: e(np * nth),
                d_dut: e(ne * nth),
                d_du10: self.upd(&vec![0.0; np]),
                d_du10d: self.upd(&vec![0.0; np]),
                d_dpar: self.upd(&dpar),
                d_dtauhf: self.upd(&dtab),
                d_dhs: e(np),
                d_dt02: e(np),
                d_den: e(np),
            };
            let saved = self.tl.replace(tl);
            self.sync();
            let t = std::time::Instant::now();
            for _ in 0..nsteps {
                self.step()?;
            }
            self.sync();
            t_tl += t.elapsed().as_secs_f64();
            let out = self.download_f64(&self.tl.as_ref().expect("tangent").d_vad);
            self.tl = saved;
            rhs.push((name, dot(&ybar, &out), dpar));
        }
        // the adjoint through the driver with two levels of checkpoints
        let saved = self.tl.take();
        self.restore(&s0, inp)?;
        let mut ab = st.clone_htod(&ybar)?;
        let t = std::time::Instant::now();
        let (bar, held) = self.adjoint_run(&k, inp, nsteps, k1, &mut ab, &|_| Ok(None))?;
        let t_adj = t.elapsed().as_secs_f64();
        let t_fwd = t_tl / classes.len() as f64;
        self.tl = saved;
        let abar = st.clone_dtoh(&ab)?;
        println!(
            "adj-step from {}: {nsteps} steps, checkpoints every {k1} steps ({held} held at most); tangent {t_fwd:.2} s per direction, adjoint with its value runs {t_adj:.2} s (sources {:.2}, propagation and forcing {:.2}, recomputed steps {:.2}); checkpoints of {:.2} GB",
            s0.time,
            bar.secs[0],
            bar.secs[1],
            bar.secs[2],
            s0.bytes() as f64 / 1e9
        );
        println!("{:<10} {:>14} {:>14} {:>10}", "direction", "<adj(y),x>", "<y,tl(x)>", "rel");
        let mut rows = vec![];
        for (name, r, dpar) in rhs {
            let l = match name {
                "spectrum" => dot(&abar, &dva),
                "BETAMAX" => bar.par[0] * dpar[P_BBETA],
                "SDSC2" => bar.par[1] * dpar[P_SDSC2],
                "NLPROP" => bar.par[2] * dpar[P_SNLC1],
                _ => bar.par[3] * dpar[P_SDBC2],
            };
            let rel = (l - r).abs() / r.abs().max(1e-300);
            println!("{name:<10} {l:>14.6e} {r:>14.6e} {rel:>10.2e}");
            rows.push(serde_json::json!({"direction": name, "adj": l, "tl": r, "rel": rel}));
        }
        Ok(serde_json::json!({"start": s0.time.to_string(), "steps": nsteps, "k1": k1, "held": held, "t_tl": t_fwd, "t_adj": t_adj, "secs": bar.secs, "snap_gb": s0.bytes() as f64 / 1e9, "rows": rows}))
    }
}

/// The raw pointer of a `cudarc` buffer.
fn raw_ptr<T>(s: &cudarc::driver::CudaSlice<T>, st: &cudarc::driver::CudaStream) -> u64 {
    use cudarc::driver::DevicePtr;
    let (p, _g) = s.device_ptr(st);
    p
}
