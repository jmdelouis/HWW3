//! Milestone M2a: the CPU transcriptions of the WW3 source terms
//! ([`hww3_wave::sources`]) against the WW3 source-term harness
//! (`harness/src_harness.F90`), node by node.
//!
//! ```text
//! hww3-src check HARNESS        (reads HARNESS and HARNESS.params)
//! ```
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

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    match (a.get(1).map(|s| s.as_str()), a.get(2)) {
        (Some("check"), Some(f)) => check(PathBuf::from(f)),
        _ => bail!("usage: hww3-src check HARNESS"),
    }
}
