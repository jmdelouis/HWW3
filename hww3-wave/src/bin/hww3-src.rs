//! Milestone M2a: the CPU transcriptions of the WW3 source terms
//! ([`hww3_wave::sources`]) against the WW3 source-term harness
//! (`harness/src_harness.F90`), node by node.
//!
//! ```text
//! hww3-src check HARNESS        (reads HARNESS and HARNESS.params)
//! hww3-src check-gpu HARNESS [--f64] [--units N] [--replicate R]
//! ```
//!
//! `check-gpu` (CubeCL build) runs the GPU kernel of the whole `W3SRCE`
//! integration (`hww3_wave::srce`) on the harness nodes and compares the
//! spectra after the step with WW3 and with the CPU transcription; with
//! `--replicate R` each node is repeated `R` times for the timing.
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
        let ds = sds4(st, &a, &wn, depth)?;
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
fn check_gpu<F: hww3_wave::device::Real>(path: PathBuf, nu: usize, replicate: usize) -> Result<()> {
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
    let k = SourceKernel::<F>::new(&client, &t, nu);
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

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let opt = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let _ = &opt;
    match (a.get(1).map(|s| s.as_str()), a.get(2)) {
        (Some("check"), Some(f)) => check(PathBuf::from(f)),
        #[cfg(feature = "cubecl")]
        (Some("check-gpu"), Some(f)) => {
            let nu: usize = opt("--units").map(|s| s.parse()).transpose()?.unwrap_or(128);
            let rep: usize = opt("--replicate").map(|s| s.parse()).transpose()?.unwrap_or(1);
            if a.iter().any(|x| x == "--f64") {
                check_gpu::<f64>(PathBuf::from(f), nu, rep)
            } else {
                check_gpu::<f32>(PathBuf::from(f), nu, rep)
            }
        }
        _ => bail!("usage: hww3-src check HARNESS | check-gpu HARNESS [--f64] [--units N] [--replicate R]"),
    }
}
