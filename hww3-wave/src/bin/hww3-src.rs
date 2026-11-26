//! Milestone M2a: the CPU transcriptions of the WW3 source terms
//! ([`hww3_wave::sources`]) against the WW3 source-term harness
//! (`harness/src_harness.F90`), node by node.
//!
//! ```text
//! hww3-src check HARNESS        (reads HARNESS and HARNESS.params)
//! hww3-src check-ice HARNESS2 --params ST4.params --nml NAMELISTS.nml --grid-nml WW3_GRID.nml [--data DIR]
//! hww3-src check-gpu HARNESS [--f64] [--units N] [--replicate R] [--par-red]
//! hww3-src check-tl HARNESS [--units N] [--h H] [--tl-nlcut C [--tl-nlnode]]
//! hww3-src adj-dia HARNESS [--units N] [--replicate R] [--burn-replicate RB] [--reps K] [--only cubecl|burn]
//! hww3-src tables BASE.params [OTHER.params ...]
//! hww3-src params BASE.params OUT.params [--betamax B] [--scale-par NAME:F+NAME:F]
//! ```
//!
//! `check-gpu` (CubeCL build) runs the GPU kernel of the whole `W3SRCE`
//! integration (`hww3_wave::srce`) on the harness nodes and compares the
//! spectra after the step with WW3 and with the CPU transcription; with
//! `--replicate R` each node is repeated `R` times for the timing.
//! `adj-dia` (milestone M4 prototype) runs the DIA alone on the harness
//! spectra repeated `R` times in double precision: its value, its tangent
//! and its hand-written adjoint (`hww3_wave::adjoint`) with the dot-product
//! test and the times, then the same DIA as Burn tensor operations on the
//! first `RB` copies, its gradient through Burn's tape against the adjoint
//! and the tangent (`hww3_wave::burn_m4`); `--only` runs one of the two.
//! `check-tl` (CubeCL build, milestone M3) checks the tangent kernel
//! `hww3_wave::srce_tl` against finite differences of that kernel in double
//! precision, `tables` the stress tables computed here against those of
//! WW3, and `params` writes the parameter file of other values of
//! `BETAMAX`, `SDSC2`, `NLPROP` and `BJGAM` (stress tables computed here),
//! so that no `ww3_grid` and harness run is needed for them.
//!
//! Each call of the transcription gets the inputs WW3 got in the harness,
//! so that each routine is checked on its own: `W3SPR4` (twice), `W3SIN4`
//! (twice), `W3SLN1`, `W3SNL1`, `W3SDS4`, `W3SDB1`, `W3SBT4`, then the whole
//! `W3SRCE` integration over the global step. For every output the
//! largest difference over the nodes is given, relative to the largest
//! magnitude of that output at the node.

use anyhow::{bail, Result};
use hww3_wave::harness::{Harness, Params, Spr4, Srce};
use hww3_wave::sources::{sbt4, sdb1, sds4, sin4, sln1, snl1, spr4, srce, SrceIn, SrcePar};
use std::path::PathBuf;

#[derive(Default)]
struct Stat {
    worst: f64,
    node: usize,
    sum: f64,
    n: usize,
}

impl Stat {
    fn add(&mut self, err: f64, isea: usize) {
        if err > self.worst || self.n == 0 {
            self.worst = err;
            self.node = isea;
        }
        self.sum += err;
        self.n += 1;
    }
}

struct Report {
    rows: Vec<(String, Stat)>,
}

impl Report {
    fn stat(&mut self, name: &str) -> &mut Stat {
        if let Some(i) = self.rows.iter().position(|(n, _)| n == name) {
            return &mut self.rows[i].1;
        }
        self.rows.push((name.to_string(), Stat::default()));
        &mut self.rows.last_mut().unwrap().1
    }

    /// Scalar: difference relative to |WW3| (absolute below 1e-12).
    fn scalar(&mut self, name: &str, ours: f64, ww3: f32, isea: usize) {
        let w = ww3 as f64;
        let err = (ours - w).abs() / w.abs().max(1e-12);
        self.stat(name).add(err, isea);
    }

    /// Vector: largest difference relative to the largest |WW3| value of the node.
    fn vector(&mut self, name: &str, ours: &[f64], ww3: &[f32], isea: usize) {
        let scale = ww3.iter().fold(0f64, |m, &x| m.max((x as f64).abs())).max(1e-30);
        let err = ours.iter().zip(ww3).fold(0f64, |m, (&o, &w)| m.max((o - w as f64).abs())) / scale;
        self.stat(name).add(err, isea);
    }

    fn print(&self) {
        println!("{:<24} {:>12} {:>12} {:>10}", "output", "max error", "mean error", "worst ISEA");
        for (n, s) in &self.rows {
            println!("{:<24} {:>12.3e} {:>12.3e} {:>10}", n, s.worst, s.sum / s.n.max(1) as f64, s.node);
        }
    }
}

fn spr4_rows(r: &mut Report, tag: &str, o: [f64; 14], h: &Spr4, isea: usize) {
    for (k, name) in Spr4::NAMES.iter().enumerate() {
        if *name == "TAUWX" || *name == "TAUWY" || *name == "USTDIR" {
            continue;
        }
        r.scalar(&format!("{tag} {name}"), o[k], h.v[k], isea);
    }
}

fn check(path: PathBuf) -> Result<()> {
    let h = Harness::read(&path)?;
    let mut pp = path.clone().into_os_string();
    pp.push(".params");
    let p = Params::read(&PathBuf::from(pp))?;
    let q = SrcePar::new(&p)?;
    let (st, dia) = (&q.st, &q.dia);
    let drat = p.r1("DAIR")? as f64 / p.r1("DWAT")? as f64;
    println!(
        "harness {}: {} nodes, NK {}, NTH {}, restart {:?}, wind {:?}",
        path.display(),
        h.nodes.len(),
        h.nk,
        h.nth,
        h.restart_time,
        h.wind_time
    );
    let (nk, nth) = (h.nk, h.nth);
    let mut r = Report { rows: vec![] };
    let mut llws_bad = 0usize;
    let mut lbreak_bad = 0usize;
    let mut shave_bad = 0usize;
    let mut nsteps = (usize::MAX, 0usize, 0usize);
    for n in &h.nodes {
        let inp = &n.inputs;
        let a: Vec<f64> = inp.spec.iter().map(|&x| x as f64).collect();
        let cg: Vec<f64> = inp.cg.iter().map(|&x| x as f64).collect();
        let wn: Vec<f64> = inp.wn.iter().map(|&x| x as f64).collect();
        let wn2: Vec<f64> = (0..nk * nth).map(|is| wn[is / nth]).collect();
        let (u, udir, ust0) = (inp.v[3] as f64, inp.v[4] as f64, inp.v[5] as f64);
        // W3SPR4, first call (TAUW = 0, LLWS as WW3 had them)
        let o = spr4(st, &a, &cg, &wn, u, udir, ust0, 0.0, 0.0, &n.spr4a.llws);
        spr4_rows(&mut r, "spr4a", o.as_array(0.0, 0.0), &n.spr4a, n.isea);
        // W3SIN4 with the u* and z0 of that call
        let ustar = n.spr4a.v[5] as f64;
        let z0 = n.spr4a.v[10] as f64;
        let s = sin4(st, &a, &cg, &wn2, u, ustar, drat, udir, z0);
        r.vector("sin4a S", &s.s, &n.sin4a.s, n.isea);
        r.vector("sin4a D", &s.d, &n.sin4a.d, n.isea);
        r.vector("sin4a TAUW", &[s.tauwx, s.tauwy], &n.sin4a.tau[..2], n.isea);
        r.vector("sin4a TAUWN", &[s.tauwnx, s.tauwny], &n.sin4a.tau[2..], n.isea);
        llws_bad += s.llws.iter().zip(&n.spr4b.llws).filter(|(a, b)| a != b).count();
        // W3SPR4, second call, with the stress of W3SIN4
        let tx = n.sin4a.tau[0] as f64;
        let ty = n.sin4a.tau[1] as f64;
        let o2 = spr4(st, &a, &cg, &wn, u, udir, ustar, tx, ty, &n.spr4b.llws);
        spr4_rows(&mut r, "spr4b", o2.as_array(tx, ty), &n.spr4b, n.isea);
        // W3SIN4 of the integration loop
        let ustar2 = n.spr4b.v[5] as f64;
        let z02 = n.spr4b.v[10] as f64;
        let s2 = sin4(st, &a, &cg, &wn2, u, ustar2, drat, udir, z02);
        r.vector("sin4 S", &s2.s, &n.sin4.s, n.isea);
        r.vector("sin4 D", &s2.d, &n.sin4.d, n.isea);
        r.vector("sin4 TAUW", &[s2.tauwx, s2.tauwy], &n.sin4.tau[..2], n.isea);
        // W3SLN1 with the FHIGH of W3SRCE
        let depth = inp.v[1] as f64;
        let ln = sln1(st, &wn, n.fhigh as f64, ustar2, udir);
        r.vector("sln1 S", &ln, &n.sln1, n.isea);
        // W3SNL1 with KDMEAN = WNMEAN DEPTH of the second W3SPR4
        let (nls, nld) = snl1(dia, &a, &cg, n.spr4b.v[3] as f64 * depth);
        r.vector("snl1 S", &nls, &n.snl1.s, n.isea);
        r.vector("snl1 D", &nld, &n.snl1.d, n.isea);
        // W3SDS4
        let ds = sds4(st, &a, &wn, &cg, depth, ustar2, udir, drat, n.spr4b.v[13] as f64)?;
        r.vector("sds4 S", &ds.s, &n.sds4.s, n.isea);
        r.vector("sds4 D", &ds.d, &n.sds4.d, n.isea);
        // W3SDB1 with the mean parameters of the second W3SPR4
        let (emean, fmean, wnmean) = (n.spr4b.v[0] as f64, n.spr4b.v[1] as f64, n.spr4b.v[3] as f64);
        let (dbs, dbd, lbreak) = sdb1(st, &a, depth, emean, fmean, wnmean);
        r.vector("sdb1 S", &dbs, &n.sdb1.s, n.isea);
        r.vector("sdb1 D", &dbd, &n.sdb1.d, n.isea);
        lbreak_bad += (lbreak != n.lbreak) as usize;
        // W3SBT4
        let bt = sbt4(st, &a, &cg, &wn, depth, inp.v[10] as f64, inp.v[11] as f64);
        r.vector("sbt4 S", &bt.s, &n.sbt4.s, n.isea);
        r.vector("sbt4 D", &bt.d, &n.sbt4.d, n.isea);
        r.vector("sbt4 TAUBBL", &bt.taubbl, &n.sbt4_extra[..2], n.isea);
        r.scalar("sbt4 KSUBN", bt.bedform[0], n.sbt4_extra[2], n.isea);
        // the full W3SRCE over DTG from the same state
        let si = SrceIn {
            start: false,
            dtg: h.dtg as f64,
            depth: inp.v[0] as f64,
            u10: u,
            u10dir: udir,
            ice: inp.v[7] as f64,
            d50: inp.v[10] as f64,
            psic: inp.v[11] as f64,
            coef: inp.v[8] as f64,
        };
        let mut sp = a.clone();
        let so = srce(&q, &si, &mut sp, &wn, &cg, ust0);
        r.vector("srce SPEC", &sp, &n.srce.spec, n.isea);
        for (k, (name, v)) in Srce::NAMES.iter().zip(so.as_array(0.0)).enumerate() {
            if matches!(*name, "FPI" | "SHAVE" | "WHITECAP1" | "WHITECAP2" | "WHITECAP4" | "TAUICE1" | "TAUICE2" | "PHICE") {
                continue;
            }
            r.scalar(&format!("srce {name}"), v, n.srce.v[k], n.isea);
        }
        shave_bad += (so.shave != (n.srce.v[29] > 0.5)) as usize;
        nsteps = (nsteps.0.min(so.nsteps), nsteps.1.max(so.nsteps), nsteps.2 + so.nsteps);
    }
    r.print();
    println!("LLWS of W3SIN4 different from WW3 in {llws_bad} of {} bins", h.nodes.len() * nk * nth);
    println!("LBREAK of W3SDB1 different from WW3 at {lbreak_bad} of {} nodes", h.nodes.len());
    println!("SHAVE of W3SRCE different from WW3 at {shave_bad} nodes");
    println!(
        "dynamic steps of W3SRCE: {} to {}, mean {:.2}",
        nsteps.0,
        nsteps.1,
        nsteps.2 as f64 / h.nodes.len().max(1) as f64
    );
    Ok(())
}

/// The GPU kernel `hww3_wave::srce::srce` on the harness nodes (each node
/// `replicate` times, for the timing), against the full `W3SRCE` of WW3 and
/// the CPU transcription.
#[cfg(feature = "cubecl")]
fn check_gpu<F: hww3_wave::device::Real>(path: PathBuf, nu: usize, replicate: usize, parred: bool) -> Result<()> {
    use burn::tensor::Device;
    use cubecl::prelude::CubeElement;
    use hww3_wave::sources::SrceTables;
    use hww3_wave::srce::{SourceKernel, SrceArgs};
    let h = Harness::read(&path)?;
    let mut pp = path.clone().into_os_string();
    pp.push(".params");
    let p = Params::read(&PathBuf::from(pp))?;
    let q = SrcePar::new(&p)?;
    let (nk, nth) = (h.nk, h.nth);
    let nspec = nk * nth;
    let n0 = h.nodes.len();
    let t = SrceTables::new(&q)?;
    let device = Device::cuda(0);
    let client = hww3_wave::burn_ext::client(&device)?;
    let mut k = SourceKernel::<F>::new(&client, &t, nu);
    k.parred = parred;
    let n = n0 * replicate;
    let mut va = vec![0f64; nspec * n];
    let mut wn = vec![0f64; (nk + 2) * n];
    let mut cg = vec![0f64; (nk + 2) * n];
    let (mut dw, mut u10, mut u10d, mut ice) = (vec![0f64; n], vec![0f64; n], vec![0f64; n], vec![0f64; n]);
    let (mut d50, mut psic) = (vec![0f64; n], vec![0f64; n]);
    for j in 0..n {
        let inp = &h.nodes[j % n0].inputs;
        for ik in 0..nk {
            for th in 0..nth {
                va[(ik * n + j) * nth + th] = inp.spec[ik * nth + th] as f64;
            }
            wn[(ik + 1) * n + j] = inp.wn[ik] as f64;
            cg[(ik + 1) * n + j] = inp.cg[ik] as f64;
        }
        dw[j] = inp.v[0] as f64;
        u10[j] = inp.v[3] as f64;
        u10d[j] = inp.v[4] as f64;
        ice[j] = inp.v[7] as f64;
        d50[j] = inp.v[10] as f64;
        psic[j] = inp.v[11] as f64;
    }
    let up = |v: &[f64]| {
        let w: Vec<F> = v.iter().map(|&x| F::of_f64(x)).collect();
        client.create_from_slice(F::as_bytes(&w))
    };
    let (d_va, d_wn, d_cg, d_dw, d_u10, d_u10d, d_ice) = (up(&va), up(&wn), up(&cg), up(&dw), up(&u10), up(&u10d), up(&ice));
    let (d_d50, d_psic) = (up(&d50), up(&psic));
    let d_nflag = client.create_from_slice(u32::as_bytes(&vec![4u32; n]));
    let (d_ust, d_dtdyn) = (up(&vec![0.0; n]), up(&vec![0.0; n]));
    let args = SrceArgs {
        va: &d_va,
        wn: &d_wn,
        cg: &d_cg,
        dw: &d_dw,
        u10: &d_u10,
        u10d: &d_u10d,
        ice: &d_ice,
        d50: &d_d50,
        psic: &d_psic,
        nflag: &d_nflag,
        ust: &d_ust,
        dtdyn: &d_dtdyn,
    };
    let sync = || futures_lite::future::block_on(client.sync()).expect("device sync");
    let down = |hd: &cubecl::server::Handle| -> Vec<f64> {
        let b = client.read_one(hd.clone()).expect("device read");
        F::from_bytes(&b).iter().map(|&x| x.as_f64()).collect()
    };
    // the first launch compiles the kernel
    let t0 = std::time::Instant::now();
    k.launch(&client, &args, h.dtg, false, n, nk, nth);
    sync();
    let first = t0.elapsed().as_secs_f64();
    let gva = down(&d_va);
    let gust = down(&d_ust);
    let gdt = down(&d_dtdyn);
    // timing on a fresh copy of the spectra
    let d_va2 = up(&va);
    let args2 = SrceArgs { va: &d_va2, ..args };
    let t1 = std::time::Instant::now();
    k.launch(&client, &args2, h.dtg, false, n, nk, nth);
    sync();
    let second = t1.elapsed().as_secs_f64();
    let hs = |a: &dyn Fn(usize) -> f64, cgk: &[f32]| -> f64 {
        let mut e = 0.0;
        for ik in 0..nk {
            let s: f64 = (0..nth).map(|th| a(ik * nth + th)).sum();
            e += s * q.st.dden[ik] / cgk[ik] as f64;
        }
        4.0 * e.max(0.0).sqrt()
    };
    let (mut w_spec, mut w_spec_mean, mut c_spec, mut w_dhs, mut c_dhs, mut w_dt, mut w_ust) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    let mut worst = 0usize;
    let mut bad_dt = 0usize;
    for (j, node) in h.nodes.iter().enumerate() {
        let g = |isp: usize| gva[((isp / nth) * n + j) * nth + isp % nth];
        let w = |isp: usize| node.srce.spec[isp] as f64;
        let inp = &node.inputs;
        let a: Vec<f64> = inp.spec.iter().map(|&x| x as f64).collect();
        let wnk: Vec<f64> = inp.wn.iter().map(|&x| x as f64).collect();
        let cgk: Vec<f64> = inp.cg.iter().map(|&x| x as f64).collect();
        let si = SrceIn {
            start: false,
            dtg: h.dtg as f64,
            depth: inp.v[0] as f64,
            u10: inp.v[3] as f64,
            u10dir: inp.v[4] as f64,
            ice: inp.v[7] as f64,
            d50: inp.v[10] as f64,
            psic: inp.v[11] as f64,
            coef: inp.v[8] as f64,
        };
        let mut cpu = a.clone();
        srce(&q, &si, &mut cpu, &wnk, &cgk, inp.v[5] as f64);
        let vmax = (0..nspec).map(w).fold(0f64, f64::max).max(1e-30);
        let cmax = cpu.iter().cloned().fold(0f64, f64::max).max(1e-30);
        let dw_ = (0..nspec).map(|i| (g(i) - w(i)).abs()).fold(0f64, f64::max) / vmax;
        let dc_ = (0..nspec).map(|i| (g(i) - cpu[i]).abs()).fold(0f64, f64::max) / cmax;
        if dw_ > w_spec {
            w_spec = dw_;
            worst = node.isea;
        }
        w_spec_mean += dw_ / n0 as f64;
        c_spec = c_spec.max(dc_);
        let hg = hs(&g, &inp.cg);
        w_dhs = w_dhs.max((hg - hs(&w, &inp.cg)).abs());
        c_dhs = c_dhs.max((hg - hs(&|i| cpu[i], &inp.cg)).abs());
        let rdt = (gdt[j] - node.srce.v[0] as f64).abs() / (node.srce.v[0] as f64).abs().max(1e-12);
        w_dt = w_dt.max(rdt);
        bad_dt += (rdt > 1e-3) as usize;
        w_ust = w_ust.max((gust[j] - node.srce.v[2] as f64).abs() / (node.srce.v[2] as f64).abs().max(1e-12));
    }
    println!(
        "GPU {} ({} units per cube), {} nodes ({} harness nodes x {}): first launch {:.3} s (with compilation), second {:.4} s",
        F::NAME,
        nu,
        n,
        n0,
        replicate,
        first,
        second
    );
    println!("  spectrum after the step, max |gpu - ww3| / max ww3 per node: max {w_spec:.3e} (ISEA {worst}), mean {w_spec_mean:.3e}; |dHs| max {w_dhs:.3e} m");
    println!("  spectrum after the step, max |gpu - cpu f64| / max cpu per node: max {c_spec:.3e}; |dHs| max {c_dhs:.3e} m");
    println!("  mean dynamic step against WW3: max relative difference {w_dt:.3e}, {bad_dt} nodes above 1e-3; u* max relative difference {w_ust:.3e}");
    Ok(())
}

/// `adj-dia`: the DIA prototypes of milestone M4 (see the header).
#[cfg(feature = "cubecl")]
fn adj_dia(path: PathBuf, nu: usize, replicate: usize, rb: usize, reps: usize, only: &str) -> Result<()> {
    use burn::tensor::Device;
    use hww3_wave::sources::SrceTables;
    use hww3_wave::srce::SourceKernel;
    let h = Harness::read(&path)?;
    let mut pp = path.clone().into_os_string();
    pp.push(".params");
    let p = Params::read(&PathBuf::from(pp))?;
    let q = SrcePar::new(&p)?;
    let (nk, nth) = (h.nk, h.nth);
    let nspec = nk * nth;
    let n0 = h.nodes.len();
    let t = SrceTables::new(&q)?;
    let n = n0 * replicate;
    let mut va = vec![0f64; nspec * n];
    let mut cg = vec![0f64; nk * n];
    let mut kdm = vec![0f64; n];
    for j in 0..n {
        let inp = &h.nodes[j % n0].inputs;
        let (mut e, mut ek) = (0f64, 0f64);
        for ik in 0..nk {
            let mut eb = 0f64;
            for th in 0..nth {
                let v = inp.spec[ik * nth + th] as f64;
                va[j * nspec + ik * nth + th] = v;
                eb += v;
            }
            cg[j * nk + ik] = inp.cg[ik] as f64;
            e += eb;
            ek += eb * inp.wn[ik] as f64;
        }
        // k_mean depth from the energy-weighted wavenumber (for the test)
        kdm[j] = if e > 0.0 { ek / e * inp.v[0] as f64 } else { 1.0 };
    }
    let device = Device::cuda(0);
    let client = hww3_wave::burn_ext::client(&device)?;
    let k = SourceKernel::<f64>::new(&client, &t, nu);
    let mut out = serde_json::json!({});
    let (rep, refs) = hww3_wave::adjoint::dia_check(&client, &k, &t, &va, &cg, &kdm, n, nk, nth, reps)?;
    out["cubecl"] = rep;
    if only != "cubecl" && rb > 0 {
        let nb = n0 * rb.min(replicate);
        out["burn"] = hww3_wave::burn_m4::dia_burn(&t, &va, &cg, &kdm, nb, nk, nth, &refs, reps)?;
    }
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}

/// The tangent kernel `hww3_wave::srce_tl` on the harness nodes against
/// central differences of the GPU source kernel, both in double precision,
/// for six directions: a random relative change of the spectrum, a relative
/// change of the wind speed and of `BETAMAX` (with the tail-stress table of
/// `TABU_TAUHF2` recomputed here), `SDSC2`, `NLPROP` and `BJGAM`. The
/// differences are taken with relative steps `h` and `h / 10`; a node where
/// they differ by more than `1e-6` of the largest derivative is counted as
/// not smooth (a discrete choice of the integration or a cell edge of an
/// interpolation table is crossed within the step) and the tangent is
/// checked against the step `h / 10` elsewhere. Derivatives below `1e-9` of
/// the largest spectral density of the node are below what the differences
/// resolve and count as zero.
#[cfg(feature = "cubecl")]
fn check_tl(path: PathBuf, nu: usize, h1: f64, nlcut: f64, nlnode: bool) -> Result<()> {
    use burn::tensor::Device;
    use cubecl::prelude::CubeElement;
    use hww3_wave::sources::{par_index, tabu_tauhf2, SrceTables, NPAR, P_BBETA};
    use hww3_wave::srce::{SourceKernel, SrceArgs};
    use hww3_wave::srce_tl::TangentArgs;
    let h = Harness::read(&path)?;
    let mut pp = path.clone().into_os_string();
    pp.push(".params");
    let q = SrcePar::new(&Params::read(&PathBuf::from(pp))?)?;
    let (nk, nth) = (h.nk, h.nth);
    let nspec = nk * nth;
    let n = h.nodes.len();
    let st = q.st.tables.as_ref().ok_or_else(|| anyhow::anyhow!("no stress tables (SINTABLE = 0)"))?;
    let shelter = q.st.ttauwshelter > 0.0;
    let bb = q.st.bbeta;
    let table = |b: f64| {
        let (t1, d1, t2, d2) = tabu_tauhf2(&q.st, st, q.st.sig[nk], b);
        if shelter {
            (t2, d2)
        } else {
            (t1, d1)
        }
    };
    let t0 = std::time::Instant::now();
    let (tauhf0, dtauhf0) = table(bb);
    let mut base = SrceTables::new(&q)?;
    base.tauhf = tauhf0;
    // the stiff interactions of the tangent (tangent kernel only)
    base.par[hww3_wave::sources::P_TLNLCUT] = nlcut;
    base.par[hww3_wave::sources::P_TLNLNODE] = if nlnode { 1.0 } else { 0.0 };
    println!(
        "harness {}: {} nodes, NK {nk}, NTH {nth}; tail-stress table in f64 for BETAMAX {bb} in {:.1} s",
        path.display(),
        n,
        t0.elapsed().as_secs_f64()
    );
    let device = Device::cuda(0);
    let client = hww3_wave::burn_ext::client(&device)?;
    let mut va = vec![0f64; nspec * n];
    let mut wn = vec![0f64; (nk + 2) * n];
    let mut cg = vec![0f64; (nk + 2) * n];
    let (mut dw, mut u10, mut u10d, mut ice) = (vec![0f64; n], vec![0f64; n], vec![0f64; n], vec![0f64; n]);
    let (mut d50, mut psic) = (vec![0f64; n], vec![0f64; n]);
    for (j, node) in h.nodes.iter().enumerate() {
        let inp = &node.inputs;
        for ik in 0..nk {
            for th in 0..nth {
                va[(ik * n + j) * nth + th] = inp.spec[ik * nth + th] as f64;
            }
            wn[(ik + 1) * n + j] = inp.wn[ik] as f64;
            cg[(ik + 1) * n + j] = inp.cg[ik] as f64;
        }
        dw[j] = inp.v[0] as f64;
        u10[j] = inp.v[3] as f64;
        u10d[j] = inp.v[4] as f64;
        ice[j] = inp.v[7] as f64;
        d50[j] = inp.v[10] as f64;
        psic[j] = inp.v[11] as f64;
    }
    let up = |v: &[f64]| client.create_from_slice(f64::as_bytes(v));
    let sync = || futures_lite::future::block_on(client.sync()).expect("device sync");
    let down = |hd: &cubecl::server::Handle| -> Vec<f64> { f64::from_bytes(&client.read_one(hd.clone()).expect("device read")).to_vec() };
    let (d_wn, d_cg, d_dw, d_u10d, d_ice, d_d50, d_psic) = (up(&wn), up(&cg), up(&dw), up(&u10d), up(&ice), up(&d50), up(&psic));
    let d_nflag = client.create_from_slice(u32::as_bytes(&vec![4u32; n]));
    let zeros = vec![0f64; n];
    // one step of the value kernel with tables `tb`, spectra `v`, wind speeds `u`
    let run = |tb: &SrceTables, v: &[f64], u: &[f64]| -> (Vec<f64>, f64) {
        let k = SourceKernel::<f64>::new(&client, tb, nu);
        let (d_va, d_u10, d_ust, d_dt) = (up(v), up(u), up(&zeros), up(&zeros));
        let args = SrceArgs {
            va: &d_va,
            wn: &d_wn,
            cg: &d_cg,
            dw: &d_dw,
            u10: &d_u10,
            u10d: &d_u10d,
            ice: &d_ice,
            d50: &d_d50,
            psic: &d_psic,
            nflag: &d_nflag,
            ust: &d_ust,
            dtdyn: &d_dt,
        };
        sync();
        let t = std::time::Instant::now();
        k.launch(&client, &args, h.dtg, false, n, nk, nth);
        sync();
        let s = t.elapsed().as_secs_f64();
        (down(&d_va), s)
    };
    // the first launch compiles the kernel
    let (v_first, _) = run(&base, &va, &u10);
    let (v_base, t_val) = run(&base, &va, &u10);
    if v_first != v_base {
        println!("  two launches of the value kernel differ");
    }
    let rnd: Vec<f64> = (0..nspec * n).map(|i| ((i * 7919 + 13) % 2001) as f64 / 1000.0 - 1.0).collect();
    let dva: Vec<f64> = va.iter().zip(&rnd).map(|(a, r)| a * r).collect();
    let at = |v: &[f64], j: usize, isp: usize| v[((isp / nth) * n + j) * nth + isp % nth];
    let energy = |v: &[f64], j: usize| -> f64 {
        let cgk = &h.nodes[j].inputs.cg;
        (0..nk).map(|ik| (0..nth).map(|th| at(v, j, ik * nth + th)).sum::<f64>() * q.st.dden[ik] / cgk[ik] as f64).sum()
    };
    let pct = |v: &mut Vec<f64>, p: f64| -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f64 * p).round() as usize]
    };
    println!("value kernel {:.4} s per launch; relative steps h = {h1:e} and {:e}", t_val, h1 / 10.0);
    println!(
        "{:<9} {:>9} {:>9} {:>7} {:>10} {:>10} {:>10} {:>6} {:>10} {:>10} {:>10}",
        "direction", "value", "sensitive", "smooth", "tl median", "tl p99", "tl max", ">1e-4", "dE median", "dE max", "rough med"
    );
    for name in ["spectrum", "wind", "BETAMAX", "SDSC2", "NLPROP", "BJGAM"] {
        let pert = |s: f64| -> Vec<f64> {
            let mut tb = base.clone();
            let mut v = va.clone();
            let mut u = u10.clone();
            match name {
                "spectrum" => v.iter_mut().zip(&dva).for_each(|(x, d)| *x += s * d),
                "wind" => u.iter_mut().for_each(|x| *x *= 1.0 + s),
                "BETAMAX" => {
                    tb.par[P_BBETA] = bb * (1.0 + s);
                    tb.tauhf = table(bb * (1.0 + s)).0;
                }
                _ => {
                    let i = par_index(name).unwrap();
                    tb.par[i] *= 1.0 + s;
                }
            }
            run(&tb, &v, &u).0
        };
        let fd = |hh: f64| -> Vec<f64> {
            let (p, m) = (pert(hh), pert(-hh));
            p.iter().zip(&m).map(|(a, b)| (a - b) / (2.0 * hh)).collect()
        };
        let (fd1, fd2) = (fd(h1), fd(h1 / 10.0));
        let mut dpar = vec![0f64; NPAR];
        let mut dtab = vec![0f64; base.tauhf.len()];
        let mut du = vec![0f64; n];
        let mut dv = vec![0f64; nspec * n];
        match name {
            "spectrum" => dv.copy_from_slice(&dva),
            "wind" => du.copy_from_slice(&u10),
            "BETAMAX" => {
                dpar[P_BBETA] = bb;
                dtab = dtauhf0.iter().map(|x| x * bb).collect();
            }
            _ => {
                let i = par_index(name)?;
                dpar[i] = base.par[i];
            }
        }
        let k = SourceKernel::<f64>::new(&client, &base, nu);
        let (d_va, d_vad, d_u10, d_ust, d_dt) = (up(&va), up(&dv), up(&u10), up(&zeros), up(&zeros));
        let (d_du, d_dud, d_dpar, d_dtab) = (up(&du), up(&zeros), up(&dpar), up(&dtab));
        let args = SrceArgs {
            va: &d_va,
            wn: &d_wn,
            cg: &d_cg,
            dw: &d_dw,
            u10: &d_u10,
            u10d: &d_u10d,
            ice: &d_ice,
            d50: &d_d50,
            psic: &d_psic,
            nflag: &d_nflag,
            ust: &d_ust,
            dtdyn: &d_dt,
        };
        let targs = TangentArgs { vad: &d_vad, du10: &d_du, du10d: &d_dud, dpar: &d_dpar, dtauhf: &d_dtab };
        sync();
        let t = std::time::Instant::now();
        k.launch_tl(&client, &args, &targs, h.dtg, n, nk, nth);
        sync();
        let t_tl = t.elapsed().as_secs_f64();
        let (val, tl) = (down(&d_va), down(&d_vad));
        let (mut vdiff, mut nsens) = (0f64, 0usize);
        let (mut e_s, mut e_r, mut e_de) = (vec![], vec![], vec![]);
        let mut worst = (0f64, 0usize);
        for j in 0..n {
            let vmax = (0..nspec).map(|i| at(&v_base, j, i).abs()).fold(0f64, f64::max).max(1e-300);
            vdiff = vdiff.max((0..nspec).map(|i| (at(&val, j, i) - at(&v_base, j, i)).abs()).fold(0f64, f64::max) / vmax);
            let fmax = (0..nspec).map(|i| at(&fd2, j, i).abs()).fold(0f64, f64::max);
            let tmax = (0..nspec).map(|i| at(&tl, j, i).abs()).fold(0f64, f64::max);
            if fmax < 1e-9 * vmax && tmax < 1e-9 * vmax {
                continue;
            }
            nsens += 1;
            let fm = fmax.max(1e-9 * vmax);
            let e = (0..nspec).map(|i| (at(&tl, j, i) - at(&fd2, j, i)).abs()).fold(0f64, f64::max) / fm;
            let rough = (0..nspec).map(|i| (at(&fd1, j, i) - at(&fd2, j, i)).abs()).fold(0f64, f64::max) / fm;
            if rough <= 1e-6 {
                e_s.push(e);
                if e > worst.0 {
                    worst = (e, h.nodes[j].isea);
                }
                // relative to the energy norm of the derivative (no cancellation)
                let (a, b) = (energy(&tl, j), energy(&fd2, j));
                let fa: Vec<f64> = fd2.iter().map(|x| x.abs()).collect();
                e_de.push((a - b).abs() / energy(&fa, j).max(1e-300));
            } else {
                e_r.push(e);
            }
        }
        let above = e_s.iter().filter(|&&e| e > 1e-4).count();
        println!(
            "{:<9} {:>9.1e} {:>9} {:>7} {:>10.2e} {:>10.2e} {:>10.2e} {:>6} {:>10.2e} {:>10.2e} {:>10.2e}",
            name,
            vdiff,
            nsens,
            e_s.len(),
            pct(&mut e_s.clone(), 0.5),
            pct(&mut e_s.clone(), 0.99),
            worst.0,
            above,
            pct(&mut e_de.clone(), 0.5),
            pct(&mut e_de, 1.0),
            pct(&mut e_r, 0.5)
        );
        println!("          worst smooth node ISEA {}; tangent kernel {:.4} s ({:.2} times the value kernel)", worst.1, t_tl, t_tl / t_val);
    }
    Ok(())
}

/// The stress tables of `INSIN4` computed here in double precision
/// (`TABU_STRESS`, `TABU_TAUHF`, `TABU_TAUHF2`) against those that WW3 wrote
/// in each parameter file, at the `BETAMAX` of that file, with the time to
/// build them; with four or more files, the derivative of `TAUHFT` and
/// `TAUHFT2` with respect to `BETAMAX` against the central differences of
/// the WW3 tables of the 2nd and 3rd files, then of the 4th and 5th.
fn tables(files: &[PathBuf]) -> Result<()> {
    use hww3_wave::sources::{tabu_stress, tabu_tauhf2};
    let rel = |x: &[f64], y: &[f64]| -> f64 {
        let m = y.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-300);
        x.iter().zip(y).fold(0f64, |e, (a, b)| e.max((a - b).abs())) / m
    };
    let mut qs = vec![];
    for f in files {
        let q = SrcePar::new(&Params::read(f)?)?;
        let t = q.st.tables.as_ref().ok_or_else(|| anyhow::anyhow!("{}: no stress tables (SINTABLE = 0)", f.display()))?;
        let t0 = std::time::Instant::now();
        let taut = tabu_stress(&q.st, t);
        let t1 = t0.elapsed().as_secs_f64();
        let (h1, dh1, h2, dh2) = tabu_tauhf2(&q.st, t, q.st.sig[q.st.nk], q.st.bbeta);
        let t2 = t0.elapsed().as_secs_f64() - t1;
        println!(
            "{} (BETAMAX {}): TAUT {:.3e}, TAUHFT {:.3e}, TAUHFT2 {:.3e} (max |diff| / max against WW3); TABU_STRESS {:.3} s, TABU_TAUHF and TABU_TAUHF2 with their derivative {:.2} s",
            f.display(),
            q.st.bbeta,
            rel(&taut, &t.taut),
            rel(&h1, &t.tauhft),
            rel(&h2, &t.tauhft2),
            t1,
            t2
        );
        if std::env::var("TAUT_WORST").is_ok() {
            // the same iterations in double precision: the cells with the largest
            // differences from WW3, relative to the largest value
            let taut = tabu_stress_f64(&q.st, t);
            let n1 = t.itaumax + 1;
            let m = t.taut.iter().fold(0f64, |m, v| m.max(v.abs()));
            let mut d: Vec<(f64, usize)> = taut.iter().zip(&t.taut).enumerate().map(|(k, (a, b))| ((a - b).abs() / m, k)).collect();
            d.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
            for &(e, k) in d.iter().take(8) {
                println!("    TAUT({}, {}): ours {:.7e}, WW3 {:.7e}, {:.2e}", k % n1, k / n1, taut[k], t.taut[k], e);
            }
            let big = d.iter().filter(|x| x.0 > 1e-5).count();
            println!("    {big} of {} cells above 1e-5", d.len());

        }
        qs.push((q, dh1, dh2));
    }
    for k in [1usize, 3] {
        if qs.len() < k + 2 {
            break;
        }
        let (qp, qm) = (&qs[k].0, &qs[k + 1].0);
        let (tp, tm) = (qp.st.tables.as_ref().unwrap(), qm.st.tables.as_ref().unwrap());
        let db = qp.st.bbeta - qm.st.bbeta;
        let fd1: Vec<f64> = tp.tauhft.iter().zip(&tm.tauhft).map(|(a, b)| (a - b) / db).collect();
        let fd2: Vec<f64> = tp.tauhft2.iter().zip(&tm.tauhft2).map(|(a, b)| (a - b) / db).collect();
        println!(
            "  d/dBETAMAX at {} against the central difference of the WW3 tables (BETAMAX {} and {}): TAUHFT {:.3e}, TAUHFT2 {:.3e}",
            qs[0].0.st.bbeta,
            qp.st.bbeta,
            qm.st.bbeta,
            rel(&qs[0].1, &fd1),
            rel(&qs[0].2, &fd2)
        );
    }
    Ok(())
}

/// `TABU_STRESS` in double precision (diagnostic of [`tables`]: WW3 iterates
/// in single precision, see `hww3_wave::sources::tabu_stress`).
fn tabu_stress_f64(p: &hww3_wave::sources::St4, t: &hww3_wave::sources::StressTables) -> Vec<f64> {
    let (ni, nj) = (t.itaumax, t.jumax);
    let (xm, eps1) = (0.5, 0.00001);
    let wcd = 0.0012875f64.sqrt();
    let mut out = vec![0f64; (ni + 1) * (nj + 1)];
    for j in 1..=nj {
        let utop = j as f64 * t.delu;
        for i in 0..=ni {
            let ztauw = (i as f64 * t.deltauw).powi(2);
            let mut tauold = (utop * wcd).powi(2).max(ztauw + eps1);
            for _ in 0..10 {
                let x = ztauw / tauold;
                let mut ust = tauold.sqrt();
                let mut zz00 = p.aalpha * tauold / p.grav;
                if p.zz0max != 0.0 {
                    zz00 = zz00.min(p.zz0max);
                }
                let zz0 = zz00 / (1.0 - x).powf(xm);
                let l = (p.zzwnd / zz0).ln();
                let f = ust - p.kappa * utop / l;
                let delf = 1.0 - p.kappa * utop / (l * l) * 2.0 / ust * (1.0 - (xm + 1.0) * x) / (1.0 - x);
                ust -= f / delf;
                tauold = (ust * ust).max(ztauw + eps1);
            }
            out[i + j * (ni + 1)] = tauold.sqrt();
        }
    }
    out
}

/// A parameter file for other values of the calibrated parameters: `BASE`
/// with `BETAMAX` set to `betamax` and its stress tables (`TAUT`, `TAUHFT`,
/// `TAUHFT2`) computed here, and `SDSC2`, `NLPROP`, `BJGAM` multiplied by the
/// factors of `scale`; everything else is copied. No other entry of the
/// file depends on these parameters (`INSIN4`, `INSDS4`, `INSNL1`).
fn params(base: PathBuf, out: PathBuf, betamax: Option<f64>, scale: &[(String, f64)]) -> Result<()> {
    use hww3_wave::sources::rebuild_stress_tables;
    let mut p = Params::read(&base)?;
    let mut q = SrcePar::new(&p)?;
    if let Some(b) = betamax {
        let t0 = std::time::Instant::now();
        q.st.bbeta = b;
        rebuild_stress_tables(&mut q.st)?;
        let t = q.st.tables.as_ref().unwrap();
        p.set_real("BBETA", &[b])?;
        p.set_real("TAUT", &t.taut)?;
        p.set_real("TAUHFT", &t.tauhft)?;
        if p.has("TAUHFT2") {
            p.set_real("TAUHFT2", &t.tauhft2)?;
        }
        println!("BETAMAX {b}: stress tables in {:.2} s", t0.elapsed().as_secs_f64());
    }
    for (name, f) in scale {
        let (rec, idx) = match name.as_str() {
            "SDSC2" => ("SSDSC", 1usize),
            "NLPROP" => ("SNLC1", 0),
            "BJGAM" => ("SDBC2", 0),
            _ => bail!("{name}: the factors are for SDSC2, NLPROP and BJGAM (BETAMAX takes --betamax)"),
        };
        let mut v: Vec<f64> = p.r(rec)?.iter().map(|&x| x as f64).collect();
        println!("{name} ({rec}[{idx}]): {} -> {}", v[idx], v[idx] * f);
        v[idx] *= f;
        p.set_real(rec, &v)?;
    }
    p.write(&out)?;
    // the file reads back with the new values
    let q2 = SrcePar::new(&Params::read(&out)?)?;
    println!("{}: BETAMAX {}, SDSC2 {}, NLPROP {}, BJGAM {}", out.display(), q2.st.bbeta, q2.st.ssdsc[1], q2.dia.snlc1, q2.st.sdbc2);
    Ok(())
}

/// The parameter file of the source terms built in Rust from the grid and
/// the namelists (`hww3_wave::ww3grid`), compared record by record with a
/// parameter file of the harness when `check` is given.
fn build_params(grid_dir: PathBuf, nml: PathBuf, out: PathBuf, grid_nml: Option<PathBuf>, check: Option<PathBuf>) -> Result<()> {
    use hww3_wave::ww3grid::{compare, source_params, GridSetup, Namelists};
    let g = hww3_wave::grid::Grid::read(&grid_dir)?;
    let gn = Namelists::read(&grid_nml.unwrap_or_else(|| grid_dir.join("ww3_grid.nml")))?;
    let s = &g.spec;
    let setup = GridSetup {
        nk: s.nk,
        nth: s.nth,
        fr1: s.freq1,
        xfr: s.xfr,
        thoff: s.thoff,
        dmin: s.dmin,
        dtmax: gn.f("TIMESTEPS_NML", "TIMESTEPS%DTMAX", 600.0)?,
        dtmin: gn.f("TIMESTEPS_NML", "TIMESTEPS%DTMIN", 10.0)?,
        nsea: g.lon.len(),
    };
    let t0 = std::time::Instant::now();
    let p = source_params(&setup, &Namelists::read(&nml)?)?;
    println!("{:?}: {} records in {:.2} s", setup, p.map.len(), t0.elapsed().as_secs_f64());
    p.write(&out)?;
    if let Some(c) = check {
        let r = Params::read(&c)?;
        for (name, line) in compare(&p, &r) {
            println!("  {name:<14} {line}");
        }
    }
    Ok(())
}

/// The ice terms (`hww3_wave::ice`) against the second harness
/// (`harness/src_harness2.F90`): `W3SIS2`, `W3SIC2` on the restart
/// spectra, then the full `W3SRCE` without ice and reflection (the ST4 of
/// `params`, built by `build-params` from the same namelists) and with ice
/// (`srce` then the ice step after the loop), on the nodes without a
/// reflection coefficient. With `data` (the directory of the grid files),
/// also the reflection (`hww3_wave::ref1`): the geometry of `W3SETREF`
/// against the inputs WW3 passed, `W3SREF`, and the full `W3SRCE` with ice
/// and reflection on every node.
fn check_ice(path: PathBuf, params: PathBuf, nml: PathBuf, grid_nml: PathBuf, data: Option<PathBuf>) -> Result<()> {
    use hww3_wave::harness2::Harness2;
    use hww3_wave::ice::{ice_after_loop, sic2, sis2, IceIn, IceParams};
    use hww3_wave::ww3grid::Namelists;
    let h = Harness2::read(&path)?;
    let q = SrcePar::new(&Params::read(&params)?)?;
    let gs = hww3_wave::structgrid::spectrum(&Namelists::read(&grid_nml)?)?;
    let sp = hww3_wave::spec::Spec::new(gs.nk, gs.nth, gs.freq1 as f32, gs.xfr as f32, gs.thoff as f32);
    let ip = IceParams::from_namelists(&Namelists::read(&nml)?, &sp)?;
    anyhow::ensure!(h.nk == gs.nk && h.nth == gs.nth, "harness {}x{}, grid {}x{}", h.nk, h.nth, gs.nk, gs.nth);
    let nice = h.nodes.iter().filter(|n| n.ice > 0.0).count();
    let nref = h.nodes.iter().filter(|n| n.reflec[0] > 0.0 || n.reflec[1] > 0.0).count();
    println!(
        "harness {}: {} nodes ({} with ice, {} with reflection), NK {}, NTH {}, restart {:?}, wind {:?}",
        path.display(),
        h.nodes.len(),
        nice,
        nref,
        h.nk,
        h.nth,
        h.restart_time,
        h.wind_time
    );
    let f = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| x as f64).collect() };
    let mut r = Report { rows: vec![] };
    // reflection: parameters and the geometry of the structured grid
    let refl = match &data {
        Some(d) => {
            use hww3_wave::ref1::{flagtr, read_obstructions, reflection_geometry_trn, Ref1Params};
            let rn = Namelists::read(&nml)?;
            let gn = Namelists::read(&grid_nml)?;
            let g = hww3_wave::structgrid::StructGrid::read(&grid_nml, d)?;
            let rp = Ref1Params::from_namelists(&rn, &gs)?;
            let trn = read_obstructions(&gn, flagtr(&rn)?, d, g.nx, g.ny)?;
            let geo = reflection_geometry_trn(&g, &rp, trn.as_ref())?;
            println!("reflection geometry of {}: {} sea points", g.name, geo.ij.len());
            Some((rp, geo))
        }
        None => None,
    };
    let mut geo_bad = 0usize;
    for n in &h.nodes {
        let isea = n.isea as usize;
        let (a, wn, cg) = (f(&n.spec0), f(&n.wn), f(&n.cg));
        let (depth, ice) = (n.depth as f64, n.ice as f64);
        if ice > 0.0 {
            let o = sis2(&ip, &a, depth, ice, n.iceh as f64, n.icef as f64, n.icedmax as f64, &wn, &cg, &wn, &cg);
            r.vector("sis2 S", &o.s, &n.sis2_s, isea);
            r.vector("sis2 D", &o.d, &n.sis2_d, isea);
            r.vector("sis2 DISSIP", &o.dissip, &n.sis2_dissip, isea);
            r.vector("sis2 R", &o.r, &n.sis2_r, isea);
            r.scalar("sis2 ICEF", o.icef, n.sis2_icef, isea);
            let (s, d) = sic2(&ip, &a, depth, n.iceh as f64, n.icef as f64, &cg, &wn, &wn, &cg, &o.r, Some(ice), false)?;
            r.vector("sic2 S", &s, &n.sic2_s, isea);
            r.vector("sic2 D", &d, &n.sic2_d, isea);
        }
        let si = hww3_wave::sources::SrceIn {
            start: false,
            dtg: h.dtg as f64,
            depth: n.dw as f64,
            u10: n.u10 as f64,
            u10dir: n.u10dir as f64,
            ice: 0.0,
            d50: n.d50 as f64,
            psic: n.psic as f64,
            coef: n.asf as f64,
        };
        let mut s0 = a.clone();
        let o0 = srce(&q, &si, &mut s0, &wn, &cg, n.ust as f64);
        r.vector("srce0 SPEC", &s0, &n.srce0_spec, isea);
        r.scalar("srce0 DTDYN", o0.dtdyn, n.srce0_dtdyn, isea);
        let has_ref = n.reflec[0] > 0.0 || n.reflec[1] > 0.0;
        // the reflection inputs and source of this node
        let mut vref: Option<Vec<f64>> = None;
        if let Some((rp, geo)) = &refl {
            let k = isea - 1;
            let ww3_lc = [n.reflec[0] as f64, n.reflec[1] as f64, n.reflec[2] as f64, n.reflec[3] as f64];
            let ours_lc = hww3_wave::ref1::reflec(&geo.reflc[k], n.berg as f64);
            if ours_lc.iter().zip(&ww3_lc).any(|(a, b)| (a - b).abs() > 1e-6 * b.abs().max(1e-6)) || geo.refld[k] != n.refled {
                geo_bad += 1;
                if geo_bad <= 5 {
                    println!("  ISEA {isea}: REFLEC {:?} REFLED {:?} against WW3 {:?} {:?}", ours_lc, geo.refld[k], ww3_lc, n.refled);
                }
            }
            r.scalar("ref1 DELX", geo.delx[k], n.delx, isea);
            r.scalar("ref1 DELY", geo.dely[k], n.dely, isea);
            r.scalar("ref1 TRNX", geo.trnx[k], n.trnx, isea);
            r.scalar("ref1 TRNY", geo.trny[k], n.trny, isea);
            if has_ref {
                let v = hww3_wave::ref1::sref(
                    rp,
                    &a,
                    &cg,
                    &ours_lc,
                    &geo.refld[k],
                    geo.delx[k],
                    geo.dely[k],
                    geo.trnx[k],
                    geo.trny[k],
                    n.berg as f64,
                );
                r.vector("sref S", &v, &n.sref, isea);
                vref = Some(v);
            }
        }
        if has_ref && vref.is_none() {
            continue;
        }
        let si = hww3_wave::sources::SrceIn { ice, ..si };
        let mut s1 = a.clone();
        let o1 = srce(&q, &si, &mut s1, &wn, &cg, n.ust as f64);
        let mut icef = n.icef as f64;
        let inp = IceIn {
            ice,
            iceh: n.iceh as f64,
            icedmax: n.icedmax as f64,
            depth,
            ice_read: true,
            icei: ice,
            southern_rlg: false,
        };
        let fl = ice_after_loop(&ip, &mut s1, &inp, &mut icef, &wn, &cg, h.dtg as f64)?;
        if let Some((rp, geo)) = &refl {
            // W3SREF on the spectrum after the ice terms, SPEC += DTG VREF
            let k = isea - 1;
            let lc = hww3_wave::ref1::reflec(&geo.reflc[k], n.berg as f64);
            if hww3_wave::ref1::is_called(&lc, n.berg as f64) {
                let v = hww3_wave::ref1::sref(rp, &s1, &cg, &lc, &geo.refld[k], geo.delx[k], geo.dely[k], geo.trnx[k], geo.trny[k], n.berg as f64);
                for (x, d) in s1.iter_mut().zip(&v) {
                    *x += h.dtg as f64 * d;
                }
            }
        }
        let tag = match (ice > 0.0, has_ref) {
            (true, true) => "srce ice+ref",
            (true, false) => "srce ice",
            (false, true) => "srce ref",
            (false, false) => "srce open",
        };
        r.vector(&format!("{tag} SPEC"), &s1, &n.srce_spec, isea);
        r.scalar(&format!("{tag} DTDYN"), o1.dtdyn, n.srce_dtdyn, isea);
        if ice > 0.0 {
            r.scalar("srce ice ICEF", icef, n.srce_icef, isea);
            r.scalar("srce ice PHICE", fl.phice, n.srce_phice, isea);
            r.vector("srce ice TAUICE", &fl.tauice, &n.srce_tauice, isea);
        }
    }
    r.print();
    if refl.is_some() {
        println!("reflection inputs (REFLEC, REFLED) unlike WW3's at {geo_bad} of {} nodes", h.nodes.len());
    }
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let opt = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let _ = &opt;
    match (a.get(1).map(|s| s.as_str()), a.get(2)) {
        (Some("check"), Some(f)) => check(PathBuf::from(f)),
        (Some("build-params"), Some(gd)) => {
            let usage = "build-params GRID_DIR NAMELISTS.nml OUT.params [--grid-nml WW3_GRID.nml] [--check BASE.params]";
            let nml = a.get(3).ok_or_else(|| anyhow::anyhow!(usage))?;
            let out = a.get(4).ok_or_else(|| anyhow::anyhow!(usage))?;
            build_params(PathBuf::from(gd), PathBuf::from(nml), PathBuf::from(out), opt("--grid-nml").map(PathBuf::from), opt("--check").map(PathBuf::from))
        }
        (Some("check-ice"), Some(f)) => {
            let usage = "check-ice HARNESS2 --params ST4.params --nml NAMELISTS.nml --grid-nml WW3_GRID.nml [--data DIR]";
            let need = |k: &str| opt(k).map(PathBuf::from).ok_or_else(|| anyhow::anyhow!(usage));
            check_ice(PathBuf::from(f), need("--params")?, need("--nml")?, need("--grid-nml")?, opt("--data").map(PathBuf::from))
        }
        (Some("tables"), Some(_)) => tables(&a[2..].iter().map(PathBuf::from).collect::<Vec<_>>()),
        (Some("params"), Some(f)) => {
            let out = a.get(3).ok_or_else(|| anyhow::anyhow!("params BASE.params OUT.params [--betamax B] [--scale-par NAME:F+NAME:F]"))?;
            let b: Option<f64> = opt("--betamax").map(|s| s.parse()).transpose()?;
            let mut sc = vec![];
            if let Some(v) = opt("--scale-par") {
                for item in v.split('+') {
                    let (n, x) = item.split_once(':').ok_or_else(|| anyhow::anyhow!("--scale-par wants NAME:FACTOR, got {item}"))?;
                    sc.push((n.to_string(), x.parse::<f64>()?));
                }
            }
            params(PathBuf::from(f), PathBuf::from(out), b, &sc)
        }
        #[cfg(feature = "cubecl")]
        (Some("check-gpu"), Some(f)) => {
            let nu: usize = opt("--units").map(|s| s.parse()).transpose()?.unwrap_or(128);
            let rep: usize = opt("--replicate").map(|s| s.parse()).transpose()?.unwrap_or(1);
            let pr = a.iter().any(|x| x == "--par-red");
            if a.iter().any(|x| x == "--f64") {
                check_gpu::<f64>(PathBuf::from(f), nu, rep, pr)
            } else {
                check_gpu::<f32>(PathBuf::from(f), nu, rep, pr)
            }
        }
        #[cfg(feature = "cubecl")]
        (Some("adj-dia"), Some(f)) => {
            let nu: usize = opt("--units").map(|s| s.parse()).transpose()?.unwrap_or(128);
            let rep: usize = opt("--replicate").map(|s| s.parse()).transpose()?.unwrap_or(200);
            let rb: usize = opt("--burn-replicate").map(|s| s.parse()).transpose()?.unwrap_or(4);
            let reps: usize = opt("--reps").map(|s| s.parse()).transpose()?.unwrap_or(3);
            let only = opt("--only").unwrap_or_default();
            adj_dia(PathBuf::from(f), nu, rep, rb, reps, &only)
        }
        #[cfg(feature = "cubecl")]
        (Some("check-tl"), Some(f)) => {
            let nu: usize = opt("--units").map(|s| s.parse()).transpose()?.unwrap_or(128);
            let h: f64 = opt("--h").map(|s| s.parse()).transpose()?.unwrap_or(1e-4);
            let cut: f64 = opt("--tl-nlcut").map(|s| s.parse()).transpose()?.unwrap_or(0.0);
            check_tl(PathBuf::from(f), nu, h, cut, a.iter().any(|x| x == "--tl-nlnode"))
        }
        _ => bail!(
            "usage: hww3-src check HARNESS | tables BASE.params [OTHER.params ...] | params BASE.params OUT.params [--betamax B] [--scale-par NAME:F+...] | build-params GRID_DIR NAMELISTS.nml OUT.params [--grid-nml F] [--check BASE.params] | check-gpu HARNESS [--f64] [--units N] [--replicate R] [--par-red] | check-tl HARNESS [--units N] [--h H] [--tl-nlcut C [--tl-nlnode]] | adj-dia HARNESS [--units N] [--replicate R] [--burn-replicate RB] [--reps K] [--only cubecl|burn]"
        ),
    }
}
