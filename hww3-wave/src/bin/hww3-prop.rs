//! M1 driver: the propagation-only WW3 model on one GPU, started from a WW3
//! restart, with the currents, levels, ice and boundary spectra of a WW3 run.
//!
//! ```text
//! hww3-prop restart-hs <grid dir> --restart FILE --out FILE.f32
//!     H_s and T_m02 of the restart spectra (CPU), to check the reader
//!     against the WW3 output at the restart time.
//! hww3-prop run <grid dir> --restart FILE [--current FILE] [--level FILE]
//!     [--ice FILE] [--nest FILE] [--start YYYYMMDDHHMMSS] --hours H --out DIR [--f64] [--block B]
//!     [--itstep0 N] [--no-cx] [--no-cth] [--no-ck] [--host-level] [--profile]
//!     Hourly H_s and T_m02 (little-endian f32, one value per node,
//!     UNDEF = -999.9 where MAPSTA <= 0) in DIR/hs_YYYYMMDDHH.f32 and
//!     DIR/t02_YYYYMMDDHH.f32, and DIR/summary.json.
//! hww3-prop check-ktp3 <grid dir> --restart FILE [forcing as for run]
//!     One step's forcing, then one W3KTP3 call on the GPU compared with
//!     the CPU transcription of W3KTP3 (hww3_wave::cpu) on a sample of nodes.
//! hww3-prop check-geo <grid dir> --restart FILE [forcing as for run]
//!     One step's forcing, then the geographic step of frequencies 1, 17
//!     and 32 on the GPU against the CPU transcription of PDLIB_W3XYPUG.
//! ```

use anyhow::{bail, Result};
use burn::tensor::Device;
use hww3_wave::device::Real;
use hww3_wave::geom::Geometry;
use hww3_wave::grid::Grid;
use hww3_wave::model::{Config, Inputs, Model, UNDEF};
use hww3_wave::spec::{node_wavenumbers, Spec};
use hww3_wave::ww3io::{Restart, Time};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    cmd: String,
    grid: PathBuf,
    opts: std::collections::HashMap<String, String>,
    flags: std::collections::HashSet<String>,
}

fn parse() -> Result<Args> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 2 {
        bail!("usage: hww3-prop restart-hs|run <grid dir> [options] (see the source header)");
    }
    let mut opts = std::collections::HashMap::new();
    let mut flags = std::collections::HashSet::new();
    let valued = ["--restart", "--current", "--level", "--ice", "--nest", "--hours", "--out", "--block", "--itstep0", "--start"];
    let mut i = 2;
    while i < a.len() {
        if valued.contains(&a[i].as_str()) {
            if i + 1 >= a.len() {
                bail!("{} needs a value", a[i]);
            }
            opts.insert(a[i].clone(), a[i + 1].clone());
            i += 2;
        } else if a[i].starts_with("--") {
            flags.insert(a[i].clone());
            i += 1;
        } else {
            bail!("unexpected argument {}", a[i]);
        }
    }
    Ok(Args { cmd: a[0].clone(), grid: PathBuf::from(&a[1]), opts, flags })
}

fn write_f32(p: &Path, v: &[f32]) -> Result<()> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(p, b)?;
    Ok(())
}

/// H_s and T_m02 of the restart spectra, as `W3OUTG` computes them.
fn restart_hs(grid: &Grid, a: &Args) -> Result<()> {
    let rs = Restart::read(Path::new(&a.opts["--restart"]))?;
    let s = &grid.spec;
    let sp = Spec::new(s.nk, s.nth, s.freq1 as f32, s.xfr as f32, s.thoff as f32);
    let (nk, nth, np) = (s.nk, s.nth, rs.nsea);
    let dmin = s.dmin as f32;
    let mut hs = vec![UNDEF; np];
    let mut t02 = vec![UNDEF; np];
    let mut wn = vec![0f32; nk + 2];
    let mut cg = vec![0f32; nk + 2];
    for i in 0..np {
        let zb = -(grid.depth[i] as f32);
        let dw = (rs.wlv[i] - zb).max(0.0);
        node_wavenumbers(&sp, dw, zb, dmin, &mut wn, &mut cg);
        let (mut et, mut et02, mut ab) = (0f32, 0f32, 0f32);
        for ik in 0..nk {
            ab = rs.va[i * nk * nth + ik * nth..i * nk * nth + (ik + 1) * nth].iter().sum();
            let ebd = ab * sp.dden[ik] / cg[ik + 1];
            et += ebd;
            et02 += ebd * sp.sig[ik + 1] * sp.sig[ik + 1];
        }
        let eband = ab / cg[nk];
        et += sp.fte * eband;
        et02 += eband * 0.5 * sp.sig[nk].powi(4) * sp.dth;
        if rs.mapsta[i] > 0 {
            hs[i] = 4.0 * et.sqrt();
            t02[i] = if et02 > 1e-7 && et > 0.0 { 2.0 * std::f32::consts::PI * (et / et02).sqrt() } else { 2.0 * std::f32::consts::PI / sp.sig[nk] };
        }
    }
    let out = PathBuf::from(&a.opts["--out"]);
    write_f32(&out, &hs)?;
    write_f32(&out.with_extension("t02.f32"), &t02)?;
    let valid: Vec<f32> = hs.iter().copied().filter(|&h| h > UNDEF).collect();
    println!(
        "{}: time {}, {} nodes with H_s, mean {:.4} m, max {:.3} m",
        out.display(),
        rs.time,
        valid.len(),
        valid.iter().map(|&h| h as f64).sum::<f64>() / valid.len().max(1) as f64,
        valid.iter().cloned().fold(0f32, f32::max)
    );
    Ok(())
}

fn run<F: Real>(grid: &Grid, geo: &Geometry, a: &Args) -> Result<()> {
    let device = Device::cuda(0);
    let client = hww3_wave::burn_ext::client(&device)?;
    let opt = |k: &str| a.opts.get(k).map(PathBuf::from);
    let inp = Inputs {
        restart: PathBuf::from(&a.opts["--restart"]),
        current: opt("--current"),
        level: opt("--level"),
        ice: opt("--ice"),
        nest: opt("--nest"),
        start: a.opts.get("--start").map(|s| Time::parse(s)).transpose()?,
    };
    let mut cfg = Config::default();
    if let Some(b) = a.opts.get("--block") {
        cfg.block = b.parse()?;
    }
    if let Some(s) = a.opts.get("--itstep0") {
        cfg.itstep0 = s.parse()?;
    }
    cfg.flcx = !a.flags.contains("--no-cx");
    cfg.flcth = !a.flags.contains("--no-cth");
    cfg.flck = !a.flags.contains("--no-ck");
    cfg.host_level = a.flags.contains("--host-level");
    let hours: i64 = a.opts.get("--hours").map(|h| h.parse()).transpose()?.unwrap_or(48);
    let out = PathBuf::from(a.opts.get("--out").cloned().unwrap_or_else(|| "prop_out".into()));
    std::fs::create_dir_all(&out)?;
    let t0 = Instant::now();
    let mut m = Model::<F>::new(client, grid, geo, &inp, cfg.clone())?;
    m.profile = a.flags.contains("--profile");
    m.sync();
    let setup = t0.elapsed().as_secs_f64();
    let start = m.time;
    let stop = Time::from_secs(start.secs() + 3600 * hours);
    println!("start {start}, stop {stop}, {} nodes, {} boundary points, setup {setup:.1} s", m.np, m.nbnd);
    let tag = |t: Time| format!("{:08}{:02}", t.0, t.1 / 10000);
    let mut energy = vec![];
    let (hs, t02) = m.outputs();
    write_f32(&out.join(format!("hs_{}.f32", tag(start))), &hs)?;
    write_f32(&out.join(format!("t02_{}.f32", tag(start))), &t02)?;
    energy.push(json!({"time": start.to_string(), "energy": m.total_energy()}));
    let t1 = Instant::now();
    let mut steps = 0u64;
    let mut iters = vec![];
    while m.time < stop {
        m.step()?;
        steps += 1;
        if steps == 1 || m.time.secs() % 21600 == 0 {
            let s: u32 = (0..m.nk).map(|ik| m.max_iter(ik)).sum();
            iters.push(json!({"time": m.time.to_string(), "sweeps_per_step": s}));
        }
        if m.time.secs() % 3600 == 0 {
            let (hs, t02) = m.outputs();
            write_f32(&out.join(format!("hs_{}.f32", tag(m.time))), &hs)?;
            write_f32(&out.join(format!("t02_{}.f32", tag(m.time))), &t02)?;
            if m.time.secs() % 21600 == 0 {
                let e = m.total_energy();
                energy.push(json!({"time": m.time.to_string(), "energy": e}));
                let valid: Vec<f32> = hs.iter().copied().filter(|&h| h > UNDEF).collect();
                println!(
                    "{}: {} steps, {:.1} s, mean H_s {:.4} m, energy {:.6e}",
                    m.time,
                    steps,
                    t1.elapsed().as_secs_f64(),
                    valid.iter().map(|&h| h as f64).sum::<f64>() / valid.len().max(1) as f64,
                    e
                );
            }
        }
    }
    m.sync();
    let wall = t1.elapsed().as_secs_f64();
    let per_day = wall / (hours as f64 / 24.0);
    let t = &m.timers;
    println!(
        "{steps} steps in {wall:.1} s: {per_day:.1} s per model day; forcing {:.1} s (level update on the host {:.1} s, {} threads), iter {:.1} s, ktp3 {:.1} s (theta {:.1} s), geographic {:.1} s ({} sweeps), outputs {:.1} s",
        t.forcing,
        t.level_host,
        std::thread::available_parallelism().map_or(1, |n| n.get()),
        t.iter,
        t.ktp3,
        t.ktp3_theta,
        t.geographic,
        t.sweeps,
        t.outputs
    );
    let summary = json!({
        "grid": a.grid.display().to_string(),
        "precision": if std::mem::size_of::<F>() == 8 { "f64" } else { "f32" },
        "start": start.to_string(),
        "stop": stop.to_string(),
        "steps": steps,
        "setup_seconds": setup,
        "wall_seconds": wall,
        "seconds_per_model_day": per_day,
        "profiled": m.profile,
        "timers": {"forcing": t.forcing, "level_host": t.level_host, "iter": t.iter, "ktp3": t.ktp3, "ktp3_theta": t.ktp3_theta, "geographic": t.geographic,
                   "outputs": t.outputs, "sweeps": t.sweeps},
        "sweeps": iters,
        "energy": energy,
        "config": {"dtg": cfg.dtg, "dtkth": cfg.dtkth, "ctmax": cfg.ctmax, "flcx": cfg.flcx, "flcth": cfg.flcth,
                   "flck": cfg.flck, "itstep0": cfg.itstep0, "block": cfg.block,
                   "host_level": cfg.host_level},
    });
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
    Ok(())
}

fn check_ktp3<F: Real>(grid: &Grid, geo: &Geometry, a: &Args) -> Result<()> {
    use hww3_wave::cpu::{ktp3, Ktp3Node};
    use hww3_wave::spec::{DERA, RADIUS};
    let device = Device::cuda(0);
    let client = hww3_wave::burn_ext::client(&device)?;
    let opt = |k: &str| a.opts.get(k).map(PathBuf::from);
    let inp = Inputs {
        restart: PathBuf::from(&a.opts["--restart"]),
        current: opt("--current"),
        level: opt("--level"),
        ice: opt("--ice"),
        nest: opt("--nest"),
        start: a.opts.get("--start").map(|s| Time::parse(s)).transpose()?,
    };
    let mut m = Model::<F>::new(client, grid, geo, &inp, Config::default())?;
    let (_ntloc, _itloch, facth, fack) = m.prepare()?;
    let (np, nk, nth) = (m.np, m.nk, m.nth);
    let get = |h| m.download(h);
    let (cx, cy, dddx, dddy) = (get(&m.d_cx), get(&m.d_cy), get(&m.d_dddx), get(&m.d_dddy));
    let (a1, a2, a3, a4) = (get(&m.d_dcxdx), get(&m.d_dcxdy), get(&m.d_dcydx), get(&m.d_dcydy));
    let va0 = get(&m.d_va);
    let t = Instant::now();
    m.ktp3(facth, fack);
    m.sync();
    let gpu_s = t.elapsed().as_secs_f64();
    let va1 = m.download(&m.d_va);
    let active: Vec<usize> = (0..np).filter(|&i| geo.iobp[i] == 1 && m.mapsta[i] == 1).collect();
    let mut worst = 0f64;
    let mut worst_node = 0usize;
    let mut sum_d2 = 0f64;
    let mut sum_v2 = 0f64;
    let mut changed = 0f64;
    let flcur = inp.current.is_some();
    let mut wn = vec![0f32; nk + 2];
    let mut cg = vec![0f32; nk + 2];
    let step = (active.len() / 20000).max(1);
    let mut nchk = 0;
    for &i in active.iter().step_by(step) {
        for ik in 0..nk + 2 {
            wn[ik] = m.wn[ik * np + i];
            cg[ik] = m.cg[ik * np + i];
        }
        let node = Ktp3Node {
            cg: &cg,
            wn: &wn,
            depth: m.dmin.max(m.dw[i]),
            cthg0: -(DERA * m.lat[i]).tan() / RADIUS,
            dddx: dddx[i],
            dddy: dddy[i],
            cx: cx[i],
            cy: cy[i],
            dcxdx: a1[i],
            dcxdy: a2[i],
            dcydx: a3[i],
            dcydy: a4[i],
        };
        let mut v: Vec<f32> = (0..nk * nth).map(|isp| va0[((isp / nth) * np + i) * nth + isp % nth]).collect();
        let before = v.clone();
        ktp3(&m.sp, &node, &mut v, facth, fack, m.cfg.ctmax, flcur, true, true, m.itstep);
        let vmax = v.iter().cloned().fold(0f32, f32::max).max(1e-30) as f64;
        let mut dmax = 0f64;
        for isp in 0..nk * nth {
            let g = va1[((isp / nth) * np + i) * nth + isp % nth] as f64;
            let d = (g - v[isp] as f64).abs();
            dmax = dmax.max(d);
            sum_d2 += d * d;
            sum_v2 += (v[isp] as f64).powi(2);
            changed += ((v[isp] - before[isp]) as f64).powi(2);
        }
        if dmax / vmax > worst {
            worst = dmax / vmax;
            worst_node = i;
        }
        nchk += 1;
    }
    println!(
        "check-ktp3 at {} (ITSTEP {}): {} of {} active nodes, GPU call {:.4} s; max |gpu - cpu| / max |cpu| per node {:.3e} (node {}),          global rms ratio {:.3e}, rms change by the call / rms value {:.3e}",
        m.time, m.itstep, nchk, active.len(), gpu_s, worst, worst_node, (sum_d2 / sum_v2).sqrt(), (changed / sum_v2).sqrt()
    );
    Ok(())
}

fn main() -> Result<()> {
    let a = parse()?;
    let grid = Grid::read(&a.grid)?;
    match a.cmd.as_str() {
        "restart-hs" => restart_hs(&grid, &a),
        "run" => {
            let t = Instant::now();
            let geo = Geometry::new(&grid);
            println!("geometry: {} nodes, {} triangles, {:.1} s", geo.np, geo.ne, t.elapsed().as_secs_f64());
            if a.flags.contains("--f64") {
                run::<f64>(&grid, &geo, &a)
            } else {
                run::<f32>(&grid, &geo, &a)
            }
        }
        "check-geo" => {
            let geo = Geometry::new(&grid);
            let device = Device::cuda(0);
            let client = hww3_wave::burn_ext::client(&device)?;
            let opt = |k: &str| a.opts.get(k).map(PathBuf::from);
            let inp = Inputs {
                restart: PathBuf::from(&a.opts["--restart"]),
                current: opt("--current"),
                level: opt("--level"),
                ice: opt("--ice"),
                nest: opt("--nest"),
                start: a.opts.get("--start").map(|s| Time::parse(s)).transpose()?,
            };
            let mut m = Model::<f32>::new(client, &grid, &geo, &inp, Config::default())?;
            m.prepare()?;
            for ik in [0usize, m.nk / 2, m.nk - 1] {
                let t = Instant::now();
                let (rel, rms, bad) = m.check_geo(&geo, ik);
                println!(
                    "check-geo at {} ik={} (ITER max {}): max |gpu - cpu| / max |cpu| {:.3e}, rms ratio {:.3e}, directions with another ITER {}, {:.1} s",
                    m.time, ik, m.max_iter(ik), rel, rms, bad, t.elapsed().as_secs_f64()
                );
            }
            Ok(())
        }
        "check-ktp3" => {
            let geo = Geometry::new(&grid);
            check_ktp3::<f32>(&grid, &geo, &a)
        }
        other => bail!("unknown command {other}"),
    }
}
