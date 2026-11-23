//! M1 driver: the propagation-only WW3 model on one GPU, started from a WW3
//! restart, with the currents, levels, ice and boundary spectra of a WW3 run.
//!
//! ```text
//! hww3-prop restart-hs <grid dir> --restart FILE --out FILE.f32
//!     H_s and T_m02 of the restart spectra (CPU), to check the reader
//!     against the WW3 output at the restart time.
//! hww3-prop run <grid dir> --restart FILE [--current FILE] [--level FILE]
//!     [--ice FILE] [--nest FILE] [--wind FILE --sources PARAMS] [--start YYYYMMDDHHMMSS]
//!     --hours H --out DIR [--f64] [--block B] [--units U]
//!     [--itstep0 N] [--no-cx] [--no-cth] [--no-ck] [--host-level] [--no-bccfl] [--profile]
//!     Hourly H_s and T_m02 (little-endian f32, one value per node,
//!     UNDEF = -999.9 where MAPSTA <= 0) in DIR/hs_YYYYMMDDHH.f32 and
//!     DIR/t02_YYYYMMDDHH.f32, and DIR/summary.json. The sub-cycling counts
//!     every wet node as WW3 with UGBCCFL = T (its default); --no-bccfl
//!     counts the interior nodes only. With --sources (the parameter file
//!     of the source-term harness, from the mod_def.ww3 of the run) and
//!     --wind, the source terms of the simplified ST4 run after the
//!     propagation (milestone M2a); --wind-out also writes the hourly wind
//!     (uwnd_, vwnd_YYYYMMDDHH.f32, as WW3 outputs it). --scale-par
//!     NAME:F+NAME:F multiplies source parameters (SDSC2, NLPROP, BJGAM),
//!     --wind-factor F the wind and --init-factor F the spectra of the
//!     restart, for finite differences. --tangent NAME (milestone M3) also
//!     runs the tangent of the model for a relative change of the wind
//!     (WIND), of a source parameter (BETAMAX, SDSC2, NLPROP, BJGAM) or of
//!     the spectra of the restart (INIT): hourly tangent of H_s in
//!     DIR/dhs_YYYYMMDDHH.f32 and tangent of the total energy ("denergy") in
//!     DIR/summary.json. --out64 also writes H_s and its tangent in double
//!     precision (hs_, dhs_YYYYMMDDHH.f64).
//! hww3-prop check-ktp3 <grid dir> --restart FILE [forcing as for run]
//!     One step's forcing, then one W3KTP3 call on the GPU compared with
//!     the CPU transcription of W3KTP3 (hww3_wave::cpu) on a sample of nodes.
//! hww3-prop check-geo <grid dir> --restart FILE [forcing as for run]
//!     One step's forcing, then the geographic step of frequencies 1, 17
//!     and 32 on the GPU against the CPU transcription of PDLIB_W3XYPUG.
//! hww3-prop check-steps <grid dir> --restart FILE [forcing as for run]
//!     --ref RESTART1,RESTART2,... [--nodes N1,N2,...] [--no-cth] [--no-ck] [--dump DIR]
//!     Steps from the restart and compares the spectra with the WW3 restarts
//!     written at the end of some steps: H_s differences by node class, the
//!     worst nodes, and for the listed nodes the frequencies and directions
//!     that differ, with IOBPD, ITER and the local CFL number; --dump writes
//!     the depth and currents of those steps (dw_, cx_, cy_YYYYMMDDHHMM.f32).
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
    let valued = [
        "--restart", "--current", "--level", "--ice", "--nest", "--hours", "--out", "--block", "--itstep0", "--start", "--ref", "--nodes",
        "--dump", "--wind", "--sources", "--units", "--scale-par", "--wind-factor", "--tangent", "--init-factor",
    ];
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

fn write_f64(p: &Path, v: &[f64]) -> Result<()> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(p, b)?;
    Ok(())
}

/// The hourly outputs at `tag`: H_s and T_m02 (f32), with `out64` H_s in
/// double precision too, and in a tangent run the tangent of H_s. Returns
/// H_s and the tangent outputs.
fn write_hourly<F: Real>(m: &mut Model<F>, out: &Path, tag: &str, out64: bool) -> Result<(Vec<f32>, Option<(Vec<f64>, f64)>)> {
    let (hs, t02) = m.outputs_f64();
    let hs32: Vec<f32> = hs.iter().map(|&x| x as f32).collect();
    let t32: Vec<f32> = t02.iter().map(|&x| x as f32).collect();
    write_f32(&out.join(format!("hs_{tag}.f32")), &hs32)?;
    write_f32(&out.join(format!("t02_{tag}.f32")), &t32)?;
    if out64 {
        write_f64(&out.join(format!("hs_{tag}.f64")), &hs)?;
    }
    let tan = m.tangent_outputs_f64();
    if let Some((dhs, _)) = &tan {
        let d32: Vec<f32> = dhs.iter().map(|&x| x as f32).collect();
        write_f32(&out.join(format!("dhs_{tag}.f32")), &d32)?;
        if out64 {
            write_f64(&out.join(format!("dhs_{tag}.f64")), dhs)?;
        }
    }
    Ok((hs32, tan))
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
        wind: opt("--wind"),
        sources: opt("--sources"),
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
    cfg.bccfl = !a.flags.contains("--no-bccfl");
    perturbations(a, &mut cfg)?;
    if let Some(u) = a.opts.get("--units") {
        cfg.units = u.parse()?;
    }
    let hours: i64 = a.opts.get("--hours").map(|h| h.parse()).transpose()?.unwrap_or(48);
    let out64 = a.flags.contains("--out64");
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
    if start.secs() % 3600 == 0 {
        // the outputs are hourly: a start between two hours (a restart
        // written after the first step) has none
        write_hourly(&mut m, &out, &tag(start), out64)?;
    }
    let mut e0 = json!({"time": start.to_string(), "energy": m.total_energy()});
    if let Some((_, de)) = m.tangent_outputs() {
        e0["denergy"] = json!(de);
    }
    energy.push(e0);
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
            let tg = tag(m.time);
            let (hs, tan) = write_hourly(&mut m, &out, &tg, out64)?;
            if a.flags.contains("--wind-out") {
                // the wind WW3 writes (UA cos UD, UA sin UD): the wind of the
                // step before the current is removed
                let (u, d) = (m.download(&m.d_u10), m.download(&m.d_u10d));
                let (cx, cy) = (m.download(&m.d_cx), m.download(&m.d_cy));
                let rw = cfg.rwindc;
                let uw: Vec<f32> = (0..m.np).map(|i| u[i] * d[i].cos() + rw * cx[i]).collect();
                let vw: Vec<f32> = (0..m.np).map(|i| u[i] * d[i].sin() + rw * cy[i]).collect();
                write_f32(&out.join(format!("uwnd_{}.f32", tag(m.time))), &uw)?;
                write_f32(&out.join(format!("vwnd_{}.f32", tag(m.time))), &vw)?;
            }
            if m.time.secs() % 21600 == 0 {
                let e = m.total_energy();
                let mut ej = json!({"time": m.time.to_string(), "energy": e});
                if let Some((_, de)) = &tan {
                    ej["denergy"] = json!(de);
                }
                energy.push(ej);
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
    if m.time.secs() % 21600 != 0 {
        // energy at the end of a run that does not stop at a 6-hour mark
        let mut ej = json!({"time": m.time.to_string(), "energy": m.total_energy()});
        if let Some((_, de)) = m.tangent_outputs() {
            ej["denergy"] = json!(de);
        }
        energy.push(ej);
    }
    let per_day = wall / (hours as f64 / 24.0);
    let t = &m.timers;
    println!(
        "{steps} steps in {wall:.1} s: {per_day:.1} s per model day; forcing {:.1} s (level update on the host {:.1} s, {} threads), iter {:.1} s, ktp3 {:.1} s (theta {:.1} s), geographic {:.1} s ({} sweeps), sources {:.1} s, outputs {:.1} s",
        t.forcing,
        t.level_host,
        std::thread::available_parallelism().map_or(1, |n| n.get()),
        t.iter,
        t.ktp3,
        t.ktp3_theta,
        t.geographic,
        t.sweeps,
        t.sources,
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
                   "outputs": t.outputs, "sweeps": t.sweeps, "sources": t.sources},
        "sweeps": iters,
        "energy": energy,
        "config": {"dtg": cfg.dtg, "dtkth": cfg.dtkth, "ctmax": cfg.ctmax, "flcx": cfg.flcx, "flcth": cfg.flcth,
                   "flck": cfg.flck, "itstep0": cfg.itstep0, "block": cfg.block,
                   "host_level": cfg.host_level, "sources": m.has_sources(), "units": cfg.units,
                   "tangent": cfg.tangent, "wind_factor": cfg.wind_factor, "init_factor": cfg.init_factor, "out64": out64,
                   "scale_par": cfg.scale_par.iter().map(|(n, f)| json!([n, f])).collect::<Vec<_>>()},
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
        wind: opt("--wind"),
        sources: opt("--sources"),
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

/// `--scale-par NAME:F+NAME:F` (source parameters multiplied by F) and
/// `--wind-factor F` (wind speed multiplied by F): finite differences.
fn perturbations(a: &Args, cfg: &mut Config) -> Result<()> {
    if let Some(s) = a.opts.get("--scale-par") {
        for item in s.split('+').filter(|x| !x.is_empty()) {
            let (n, f) = item.split_once(':').ok_or_else(|| anyhow::anyhow!("--scale-par wants NAME:FACTOR, got {item}"))?;
            cfg.scale_par.push((n.to_string(), f.parse()?));
        }
    }
    if let Some(n) = a.opts.get("--tangent") {
        cfg.tangent = Some(n.to_string());
    }
    if let Some(f) = a.opts.get("--init-factor") {
        cfg.init_factor = f.parse()?;
    }
    if let Some(f) = a.opts.get("--wind-factor") {
        cfg.wind_factor = f.parse()?;
    }
    Ok(())
}

/// Time of a WW3 restart file (its second record), without reading the spectra.
fn restart_time(p: &Path) -> Result<Time> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(p)?;
    let mut h = [0u8; 78];
    f.read_exact(&mut h)?;
    let nspec = i32::from_be_bytes(h[74..78].try_into()?) as u64;
    f.seek(SeekFrom::Start(4 * nspec))?;
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    Ok(Time(i32::from_be_bytes(b[0..4].try_into()?), i32::from_be_bytes(b[4..8].try_into()?)))
}

/// Inputs of a run from the command line.
fn inputs(a: &Args) -> Result<Inputs> {
    let opt = |k: &str| a.opts.get(k).map(PathBuf::from);
    Ok(Inputs {
        restart: PathBuf::from(&a.opts["--restart"]),
        current: opt("--current"),
        level: opt("--level"),
        ice: opt("--ice"),
        nest: opt("--nest"),
        wind: opt("--wind"),
        sources: opt("--sources"),
        start: a.opts.get("--start").map(|s| Time::parse(s)).transpose()?,
    })
}

/// Step from the restart and, at the time of each WW3 restart of `--ref`,
/// compare the spectra node by node: H_s differences by node class (coast,
/// interior, open boundary), the worst nodes, and for the nodes of
/// `--nodes` the frequencies and directions that differ, with the mask
/// `IOBPD`, `ITER` and the local CFL number `DT / ITER * KKSUM / SI`.
fn check_steps(grid: &Grid, geo: &Geometry, a: &Args) -> Result<()> {
    use hww3_wave::spec::{DERA, RADIUS};
    let device = Device::cuda(0);
    let client = hww3_wave::burn_ext::client(&device)?;
    let inp = inputs(a)?;
    let mut cfg = Config::default();
    cfg.flcth = !a.flags.contains("--no-cth");
    cfg.flck = !a.flags.contains("--no-ck");
    cfg.bccfl = !a.flags.contains("--no-bccfl");
    if let Some(s) = a.opts.get("--itstep0") {
        // ITSTEP of the restart (its parity sets the order of W3KTP3)
        cfg.itstep0 = s.parse()?;
    }
    perturbations(a, &mut cfg)?;
    if let Some(u) = a.opts.get("--units") {
        cfg.units = u.parse()?;
    }
    let mut m = Model::<f32>::new(client, grid, geo, &inp, cfg)?;
    let mut refs: Vec<(Time, PathBuf)> = Vec::new();
    for s in a.opts.get("--ref").map(|s| s.as_str()).unwrap_or("").split(',').filter(|s| !s.is_empty()) {
        let p = PathBuf::from(s);
        refs.push((restart_time(&p)?, p));
    }
    refs.sort();
    let nodes: Vec<usize> = match a.opts.get("--nodes") {
        Some(s) => s.split(',').map(|x| x.parse::<usize>()).collect::<Result<_, _>>()?,
        None => vec![],
    };
    let (np, nk, nth) = (m.np, m.nk, m.nth);
    let nspec = nk * nth;
    let facx = 1.0 / (DERA * RADIUS);
    let clats: Vec<f32> = m.lat.iter().map(|&y| (DERA * y).cos()).collect();
    let ntri = |i: usize| (geo.start[i + 1] - geo.start[i]) as usize;
    // graph distance (edges) of every node to the nearest coast node
    let mut hops = vec![u32::MAX; np];
    let mut front: Vec<usize> = (0..np).filter(|&i| geo.iobp[i] == 0).collect();
    for &i in &front {
        hops[i] = 0;
    }
    let mut d = 0u32;
    while !front.is_empty() {
        d += 1;
        let mut next = vec![];
        for &i in &front {
            for q in geo.start[i] as usize..geo.start[i + 1] as usize {
                let t = geo.slot[q] as usize / 3;
                for r in 0..3 {
                    let j = geo.ine[3 * t + r] as usize;
                    if hops[j] == u32::MAX {
                        hops[j] = d;
                        next.push(j);
                    }
                }
            }
        }
        front = next;
    }
    for (tref, pref) in &refs {
        while m.time < *tref {
            m.step()?;
        }
        if m.time != *tref {
            bail!("no step ends at {tref} (model at {})", m.time);
        }
        if let Some(dir) = a.opts.get("--dump") {
            // depth and currents of this step, to compare with WW3's DPT and CUR fields
            let tag = format!("{:08}{:04}", m.time.0, m.time.1 / 100);
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir)?;
            write_f32(&dir.join(format!("dw_{tag}.f32")), &m.dw)?;
            write_f32(&dir.join(format!("cx_{tag}.f32")), &m.download(&m.d_cx))?;
            write_f32(&dir.join(format!("cy_{tag}.f32")), &m.download(&m.d_cy))?;
        }
        let rs = Restart::read(pref)?;
        let va = m.download(&m.d_va);
        let cg = m.device_cg();
        let band = |src: &dyn Fn(usize, usize, usize) -> f32, i: usize, k: usize| -> f64 {
            let s: f64 = (0..nth).map(|th| src(i, k, th) as f64).sum();
            s * (m.sp.dden[k] / cg[(k + 1) * np + i]) as f64
        };
        let gpu = |i: usize, k: usize, th: usize| va[(k * np + i) * nth + th];
        let ww3 = |i: usize, k: usize, th: usize| rs.va[i * nspec + k * nth + th];
        let mut dh = vec![0f64; np];
        let mut hh = vec![(0f64, 0f64); np];
        let mut relmax = [(0f64, 0usize); 3];
        for i in 0..np {
            if m.mapsta[i] <= 0 {
                continue;
            }
            let (mut eg, mut ew, mut dmax, mut vmax) = (0f64, 0f64, 0f64, 0f64);
            for k in 0..nk {
                eg += band(&gpu, i, k);
                ew += band(&ww3, i, k);
                for th in 0..nth {
                    let (g, w) = (gpu(i, k, th) as f64, ww3(i, k, th) as f64);
                    dmax = dmax.max((g - w).abs());
                    vmax = vmax.max(w.abs());
                }
            }
            let (hg, hw) = (4.0 * eg.max(0.0).sqrt(), 4.0 * ew.max(0.0).sqrt());
            hh[i] = (hg, hw);
            dh[i] = hg - hw;
            let c = geo.iobp[i].min(2) as usize;
            let rel = dmax / vmax.max(1e-30);
            if vmax > 1e-12 && rel > relmax[c].0 {
                relmax[c] = (rel, i);
            }
        }
        println!("== {} (ITSTEP {}) against {}", m.time, m.itstep, pref.display());
        let names = ["coast", "interior", "open boundary"];
        for (c, name) in names.iter().enumerate() {
            let sel: Vec<usize> = (0..np).filter(|&i| m.mapsta[i] > 0 && geo.iobp[i].min(2) as usize == c).collect();
            let worst = sel.iter().copied().max_by(|&x, &y| dh[x].abs().total_cmp(&dh[y].abs()));
            let single = sel.iter().filter(|&&i| ntri(i) == 1).count();
            println!(
                "  {name}: {} nodes ({} in one triangle), |dHs| > 1 mm {}, > 1 cm {}, > 5 cm {}; max |dHs| {:.4} m at {:?}; max spectral |gpu - ww3| / max ww3 {:.2e} at node {}",
                sel.len(),
                single,
                sel.iter().filter(|&&i| dh[i].abs() > 1e-3).count(),
                sel.iter().filter(|&&i| dh[i].abs() > 1e-2).count(),
                sel.iter().filter(|&&i| dh[i].abs() > 5e-2).count(),
                worst.map_or(0.0, |w| dh[w].abs()),
                worst,
                relmax[c].0,
                relmax[c].1
            );
        }
        println!("  by distance to the coast (edges: nodes, |dHs| > 1 mm, > 1 cm, max |dHs|, max spectral relative difference):");
        for dd in 0..=8u32 {
            let sel: Vec<usize> = (0..np)
                .filter(|&i| m.mapsta[i] > 0 && if dd < 8 { hops[i] == dd } else { hops[i] >= 8 })
                .collect();
            let mut rmax = 0f64;
            for &i in &sel {
                let mut dmax = 0f64;
                let mut vmax = 0f64;
                for k in 0..nk {
                    for th in 0..nth {
                        let (g, w) = (gpu(i, k, th) as f64, ww3(i, k, th) as f64);
                        dmax = dmax.max((g - w).abs());
                        vmax = vmax.max(w.abs());
                    }
                }
                if vmax > 1e-12 {
                    rmax = rmax.max(dmax / vmax);
                }
            }
            println!(
                "    {}{dd}: {} nodes, {}, {}, {:.2e} m, {:.2e}",
                if dd < 8 { "" } else { ">=" },
                sel.len(),
                sel.iter().filter(|&&i| dh[i].abs() > 1e-3).count(),
                sel.iter().filter(|&&i| dh[i].abs() > 1e-2).count(),
                sel.iter().map(|&i| dh[i].abs()).fold(0f64, f64::max),
                rmax
            );
        }
        let one: Vec<usize> = (0..np).filter(|&i| m.mapsta[i] > 0 && geo.iobp[i] == 0 && ntri(i) == 1).collect();
        println!(
            "  coast nodes in one triangle: |dHs| > 1 cm {} of {}; coast nodes in 2 or more: |dHs| > 1 cm {}",
            one.iter().filter(|&&i| dh[i].abs() > 1e-2).count(),
            one.len(),
            (0..np).filter(|&i| m.mapsta[i] > 0 && geo.iobp[i] == 0 && ntri(i) > 1 && dh[i].abs() > 1e-2).count()
        );
        let mut order: Vec<usize> = (0..np).filter(|&i| m.mapsta[i] > 0).collect();
        order.sort_by(|&x, &y| dh[y].abs().total_cmp(&dh[x].abs()));
        println!("  worst nodes (node iobp triangles hops depth: Hs gpu / ww3):");
        let far: Vec<usize> = order.iter().copied().filter(|&i| hops[i] >= 8).take(4).collect();
        for &i in order.iter().take(12).chain(far.iter()) {
            println!(
                "    {i} {} {} {} {:.1}: {:.4} / {:.4} (lon {:.3}, lat {:.3})",
                geo.iobp[i],
                ntri(i),
                hops[i],
                m.dw[i],
                hh[i].0,
                hh[i].1,
                grid.lon[i],
                grid.lat[i]
            );
        }
        let nodes: Vec<usize> = nodes.iter().copied().chain(far.iter().copied().take(2)).collect();
        if nodes.is_empty() {
            continue;
        }
        let cx = m.download(&m.d_cx);
        let cy = m.download(&m.d_cy);
        let dt = m.cfg.dtg as f64;
        // local CFL number of node i for component (k, th)
        let cfl = |i: usize, k: usize, th: usize| -> f64 {
            let (ec, es) = (m.sp.ecos[th] as f64, m.sp.esin[th] as f64);
            let vel = |j: usize| -> (f64, f64) {
                let wet = (m.dw[j] >= m.dmin + 1e-5) as u8 as f64;
                let c = cg[(k + 1) * np + j] as f64;
                let cur = (geo.iobp[j] > 0) as u8 as f64;
                let cl = clats[j] as f64;
                (
                    (facx as f64 * ec * c / cl + cur * facx as f64 * cx[j] as f64 / cl) * wet,
                    (facx as f64 * es * c + cur * facx as f64 * cy[j] as f64) * wet,
                )
            };
            let mut kk = 0f64;
            for q in geo.start[i] as usize..geo.start[i + 1] as usize {
                let code = geo.slot[q] as usize;
                let (t, p) = (code / 3, code % 3);
                let v: Vec<(f64, f64)> = (0..3).map(|r| vel(geo.ine[3 * t + r] as usize)).collect();
                let lx = (v[0].0 + v[1].0 + v[2].0) / 6.0;
                let ly = (v[0].1 + v[1].1 + v[2].1) / 6.0;
                kk += (lx * geo.ien[6 * t + 2 * p] + ly * geo.ien[6 * t + 2 * p + 1]).max(0.0);
            }
            let it = m.iter_host[k * nth + th].max(1) as f64;
            dt / it * kk / geo.si[i]
        };
        for &i in &nodes {
            let (hg, hw) = hh[i];
            let mut bands: Vec<(usize, f64, f64)> = (0..nk).map(|k| (k, band(&gpu, i, k), band(&ww3, i, k))).collect();
            bands.sort_by(|x, y| (y.1 - y.2).abs().total_cmp(&(x.1 - x.2).abs()));
            let (mut cmax, mut kmax, mut tmax) = (0f64, 0, 0);
            for k in 0..nk {
                for th in 0..nth {
                    let c = cfl(i, k, th);
                    if c > cmax {
                        (cmax, kmax, tmax) = (c, k, th);
                    }
                }
            }
            println!(
                "  node {i} (iobp {}, {} triangles, depth {:.1} m): Hs {:.4} / {:.4}; largest local CFL {:.2} at k {} th {} (ITER {})",
                geo.iobp[i],
                ntri(i),
                m.dw[i],
                hg,
                hw,
                cmax,
                kmax,
                tmax,
                m.iter_host[kmax * nth + tmax]
            );
            print!("    bands with the largest energy differences (k: gpu / ww3):");
            for &(k, g, w) in bands.iter().take(4) {
                print!(" {k}: {g:.4e} / {w:.4e};");
            }
            println!();
            let k = bands[0].0;
            let mask = geo.iobpd[i];
            println!("    k {k}, per direction (th mask ITER CFL: gpu / ww3):");
            let mut line = String::new();
            for th in 0..nth {
                let (g, w) = (gpu(i, k, th), ww3(i, k, th));
                if g.abs().max(w.abs()) == 0.0 {
                    continue;
                }
                line.push_str(&format!(
                    " [{th} {} {} {:.2}: {:.3e}/{:.3e}]",
                    (mask >> th) & 1,
                    m.iter_host[k * nth + th],
                    cfl(i, k, th),
                    g,
                    w
                ));
            }
            println!("   {line}");
        }
    }
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
                wind: opt("--wind"),
                sources: opt("--sources"),
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
        "check-steps" => {
            let geo = Geometry::new(&grid);
            check_steps(&grid, &geo, &a)
        }
        other => bail!("unknown command {other}"),
    }
}
