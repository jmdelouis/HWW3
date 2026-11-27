//! M1 driver: the propagation-only WW3 model on one GPU, started from a WW3
//! restart, with the currents, levels, ice and boundary spectra of a WW3 run.
//!
//! ```text
//! hww3-prop restart-hs <grid dir> --restart FILE --out FILE.f32
//!     H_s and T_m02 of the restart spectra (CPU), to check the reader
//!     against the WW3 output at the restart time.
//! hww3-prop restart-copy <restart> --out FILE
//!     Read a FULL restart and write it again (CPU), to check the writer:
//!     the copy is identical byte for byte.
//! hww3-prop run <grid dir> --restart FILE|calm [--current FILE] [--level FILE] [--ice-ref NAMELISTS]
//!     [--ice FILE] [--nest FILE] [--wind FILE --sources PARAMS] [--start YYYYMMDDHHMMSS]
//!     --hours H|--stop YYYYMMDDHHMMSS --out DIR [--restart-out FILE] [--f64] [--block B] [--units U]
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
//!     DIR/summary.json, and of T_m02 in DIR/dt02_YYYYMMDDHH.f32. --out64
//!     also writes H_s, T_m02 and their tangents in double precision (hs_,
//!     t02_, dhs_, dt02_YYYYMMDDHH.f64). --betamax B sets BETAMAX with the
//!     tail-stress tables computed for it in double precision. --betau
//!     KNOTS:COEFS (comma-separated, m/s and coefficients of ln BETAMAX)
//!     makes BETAMAX depend on the local wind speed, BETAMAX exp(sum_k c_k
//!     phi_k(U)) with hat functions phi_k on the knots, the tail-stress
//!     tables computed at --betau-grid B0:DB:N (0.75:0.25:13) and
//!     interpolated in BETAMAX; --tangent BETAU<k> is then the tangent for
//!     c_k, and BETAU for all of them at once. --restart
//!     calm starts from calm conditions (zero spectra, WW3's start without
//!     a restart file) at --start; --restart-out writes the state at the
//!     end as a FULL WW3 restart, from which a run continues with
//!     --itstep0 set to the ITSTEP it prints. --par-red computes the sums
//!     of W3SIN4 over the directions and of W3SPR4 over the frequencies,
//!     and the minimum of the dynamic step, by reductions over the warps in
//!     the value source kernel: another order of the additions, so results
//!     that differ at round-off (off by default; the tangent kernel keeps
//!     the order of WW3).
//!     On a grid directory of the UQ scheme of a structured grid
//!     (hww3-pre struct --scheme uq, hww3_wave::uqgrid), the geographic
//!     propagation is WW3's W3XYP3 (PR3 + UQ, hww3_wave::prop_uq) with
//!     the global steps of W3WAVE (DTMAX, DTKTH of the grid), the restart
//!     holds one spectrum per sea point (hww3-pre restart-uq), and H_s,
//!     T_m02 (and their tangents) are written on the whole NX x NY grid
//!     (iy NX + ix, UNDEF off the sea points). The global steps are those
//!     of ww3_shel (each hour in equal steps of at most DTMAX), or with
//!     --steps multi those of ww3_multi (DTMAX, then the rest of the hour).
//! hww3-prop check-uq <uq grid dir> --restart FILE [--f64] [--ik K1,K2,..] [--tangent INIT] [--h H] [--bnd-test]
//!     The UQ scheme after the first step's forcing: the geographic step
//!     of the frequencies K (default 1, NK/2 and NK, 1-based) on the GPU
//!     against the CPU transcription of W3XYP3 (hww3_wave::pro3::xyp3) on
//!     the same inputs, every direction (with --bnd-test, with synthetic
//!     boundary spectra at the points of MAPSTA = 2 on both sides); with
//!     --tangent, the tangent of the
//!     step along the field itself (equal to the stepped field: the scheme
//!     is positively homogeneous of degree 1) and along a random relative
//!     perturbation against central differences of steps H (default 1e-7)
//!     and H / 10 at the sea points with no kink within H.
//! hww3-prop adj-geo <grid dir> --restart FILE [forcing as for run] [--f64] [--ik K] [--reps N] [--only cubecl]
//!     Milestone M4 prototype: one geographic sub-step of frequency K after
//!     the first step's forcing, its value, tangent and hand-written adjoint
//!     with the dot-product test and the times, then the same sub-step as
//!     Burn tensor operations with its gradient through Burn's tape
//!     (unless --only cubecl).
//! hww3-prop adj-grad <grid dir> --restart FILE [forcing and parameters as for run] --hours H --seeds DIR [--k1 K]   (feature cuda-c)
//!     Milestone M4: the gradient of a cost of the hourly H_s and T_m02 over
//!     H hours by the adjoint, from the derivatives of the cost with respect
//!     to those fields (hsb_, t02b_YYYYMMDDHH.f64 in DIR, written by
//!     scripts/cal_obs.py seed), with two levels of checkpoints in host
//!     memory (every K steps, default the square root of the steps): dJ/dln p
//!     of BETAMAX, SDSC2, NLPROP, BJGAM and dJ/dc_k of BETAMAX(U).
//! hww3-prop adj-step <grid dir> --restart FILE [forcing as for run] [--steps N] [--k1 K]   (feature cuda-c)
//!     Milestone M4: the dot-product tests of the adjoint of N whole steps
//!     (default 6) against their tangent, for the spectra at the start and
//!     BETAMAX, SDSC2, NLPROP and BJGAM, through the driver with two levels
//!     of checkpoints (every K steps, default 3), with the times and the size
//!     of the checkpoints.
//! hww3-prop adj-prop <grid dir> --restart FILE [forcing as for run]   (feature cuda-c)
//!     Milestone M4: after the first step's forcing, the dot-product tests
//!     of the CUDA C adjoint of the propagation (hww3_wave::cuda_prop)
//!     against the tangent kernels in double precision: the refraction, the
//!     frequency shift, the geographic propagation of three frequencies and
//!     the whole propagation of the step, with the times.
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
        "--dump", "--wind", "--sources", "--units", "--scale-par", "--wind-factor", "--tangent", "--init-factor", "--betamax",
        "--stop", "--restart-out", "--ice-ref", "--betau", "--betau-grid", "--probe", "--probe-check", "--tl-nlcut", "--ik", "--reps", "--only", "--steps", "--k1", "--seeds",
        "--h",
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

/// The hourly outputs at `tag`: H_s and T_m02 (f32), with `out64` in
/// double precision too, and in a tangent run their tangents. Returns H_s
/// and the tangent outputs.
#[allow(clippy::type_complexity)]
fn write_hourly<F: Real>(m: &mut Model<F>, out: &Path, tag: &str, out64: bool) -> Result<(Vec<f32>, Option<(Vec<f64>, Vec<f64>, f64)>)> {
    let (hs, t02) = m.outputs_f64();
    let hs32: Vec<f32> = hs.iter().map(|&x| x as f32).collect();
    let t32: Vec<f32> = t02.iter().map(|&x| x as f32).collect();
    // with the UQ scheme, the whole NX x NY grid
    let f32s = |v: &[f32]| -> Vec<f32> {
        match m.uq() {
            Some(u) => u.expand(v, UNDEF),
            None => v.to_vec(),
        }
    };
    let f64s = |v: &[f64]| -> Vec<f64> {
        match m.uq() {
            Some(u) => u.expand(v, UNDEF as f64),
            None => v.to_vec(),
        }
    };
    write_f32(&out.join(format!("hs_{tag}.f32")), &f32s(&hs32))?;
    write_f32(&out.join(format!("t02_{tag}.f32")), &f32s(&t32))?;
    if out64 {
        write_f64(&out.join(format!("hs_{tag}.f64")), &f64s(&hs))?;
        write_f64(&out.join(format!("t02_{tag}.f64")), &f64s(&t02))?;
    }
    let tan = m.tangent_outputs_f64();
    if let Some((dhs, dt02, _)) = &tan {
        for (name, v) in [("dhs", dhs), ("dt02", dt02)] {
            let d32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
            write_f32(&out.join(format!("{name}_{tag}.f32")), &f32s(&d32))?;
            if out64 {
                write_f64(&out.join(format!("{name}_{tag}.f64")), &f64s(v))?;
            }
        }
    }
    Ok((hs32, tan))
}

/// `run`: on the triangles of `geo`, or with the UQ scheme of the grid
/// directory when `geo` is `None`.
fn run<F: Real>(grid: &Grid, geo: Option<&Geometry>, a: &Args) -> Result<()> {
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
        ice_ref: match a.opts.get("--ice-ref") {
            Some(n) => Some(hww3_wave::srce_ice::IceRefInputs::load(&a.grid, Path::new(n), grid)?),
            None => None,
        },
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
    // UQ scheme: the global steps of ww3_multi (--steps multi) or ww3_shel
    cfg.steps_multi = match a.opts.get("--steps").map(String::as_str) {
        None | Some("shel") => false,
        Some("multi") => true,
        Some(x) => bail!("--steps {x}: shel or multi"),
    };
    perturbations(a, &mut cfg)?;
    if let Some(u) = a.opts.get("--units") {
        cfg.units = u.parse()?;
    }
    let hours: i64 = a.opts.get("--hours").map(|h| h.parse()).transpose()?.unwrap_or(48);
    let out64 = a.flags.contains("--out64");
    let out = PathBuf::from(a.opts.get("--out").cloned().unwrap_or_else(|| "prop_out".into()));
    std::fs::create_dir_all(&out)?;
    let t0 = Instant::now();
    let mut m = match geo {
        Some(geo) => Model::<F>::new(client, grid, geo, &inp, cfg.clone())?,
        None => Model::<F>::new_uq(client, grid, &hww3_wave::uqgrid::UqSetup::read(&a.grid)?, &inp, cfg.clone())?,
    };
    m.profile = a.flags.contains("--profile");
    m.sync();
    let setup = t0.elapsed().as_secs_f64();
    let start = m.time;
    let stop = match a.opts.get("--stop") {
        Some(s) => Time::parse(s)?,
        None => Time::from_secs(start.secs() + 3600 * hours),
    };
    anyhow::ensure!(stop > start, "stop {stop} is not after the start {start}");
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
    // --probe N1,N2,..: the spectra and tangents of these nodes after each
    // part of every step (probe_spec.f64, rows of probe.csv) and the node
    // scalars after the source terms
    let probe: Vec<u32> = match a.opts.get("--probe") {
        Some(s) => s.split(',').map(|x| x.parse::<u32>()).collect::<Result<_, _>>()?,
        None => vec![],
    };
    let mut probe_csv = None;
    let mut probe_bin = None;
    // --probe-check H: before each source call, the tangent of that step at
    // the probed nodes against central differences with relative steps H
    // and H / 10 (probe_check.csv, probe_check.f64)
    let check: Option<f64> = a.opts.get("--probe-check").map(|h| h.parse()).transpose()?;
    let mut check_csv = None;
    let mut check_bin = None;
    if !probe.is_empty() {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(out.join("probe.csv"))?);
        writeln!(f, "time,phase,node,row,sum_va,max_va,sum_abs_vad,l2_vad,max_abs_vad,ik_max,ith_max,dw,u10,u10d,ust,dtdyn,ice,mapsta")?;
        probe_csv = Some(f);
        probe_bin = Some(std::io::BufWriter::new(std::fs::File::create(out.join("probe_spec.f64"))?));
        if check.is_some() {
            let mut f = std::io::BufWriter::new(std::fs::File::create(out.join("probe_check.csv"))?);
            writeln!(
                f,
                "time,node,row,l2_in,l2_tl,l2_fd1,l2_fd2,amp_tl,amp_fd,err_tl_fd2,rough_fd1_fd2,ik_err,ith_err,amp_no_sln,amp_no_sin,amp_no_snl,amp_no_sds,amp_no_sbt,amp_no_ust,amp_no_den,amp_no_sdb,amp_no_sin_ust,amp_no_snl_sds,amp_nlcut8,amp_nlcut4,amp_nlcut2,amp_nlcut1,amp_nlcut0.5,amp_nlabs2,amp_nlabs1,amp_nlcut0.25,amp_node1,amp_node0.5,amp_node0.25"
            )?;
            check_csv = Some(f);
            check_bin = Some(std::io::BufWriter::new(std::fs::File::create(out.join("probe_check.f64"))?));
        }
    }
    let mut check_row = 0u64;
    let mut probe_row = 0u64;
    while m.time < stop {
        if probe.is_empty() {
            m.step()?;
        } else {
            use std::io::Write;
            let (parts, checks) = m.step_probed(&probe, check)?;
            if let (Some(cs), Some(f), Some(b)) = (&checks, check_csv.as_mut(), check_bin.as_mut()) {
                let l2 = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
                let mx = |v: &[f64]| v.iter().fold(0f64, |m, x| m.max(x.abs()));
                for (j, c) in cs.iter().enumerate() {
                    let [_, din, tl, f1, f2, amps] = c;
                    let fm = mx(f2).max(1e-300);
                    let (mut qe, mut de) = (0usize, 0f64);
                    for q in 0..tl.len() {
                        if (tl[q] - f2[q]).abs() > de {
                            de = (tl[q] - f2[q]).abs();
                            qe = q;
                        }
                    }
                    let rough = f1.iter().zip(f2).fold(0f64, |m, (x, y)| m.max((x - y).abs())) / fm;
                    writeln!(
                        f,
                        "{},{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{},{},{}",
                        m.time,
                        probe[j],
                        check_row,
                        l2(din),
                        l2(tl),
                        l2(f1),
                        l2(f2),
                        l2(tl) / l2(din).max(1e-300),
                        l2(f2) / l2(din).max(1e-300),
                        de / fm,
                        rough,
                        qe / m.nth,
                        qe % m.nth,
                        amps.iter().map(|x| format!("{x:.6e}")).collect::<Vec<_>>().join(",")
                    )?;
                    for v in c[..5].iter() {
                        for x in v {
                            b.write_all(&x.to_le_bytes())?;
                        }
                    }
                    check_row += 1;
                }
            }
            let sc = m.node_scalars(&probe);
            let nspec = m.nk * m.nth;
            let (f, b) = (probe_csv.as_mut().unwrap(), probe_bin.as_mut().unwrap());
            for (phase, va, vad) in &parts {
                for (j, &node) in probe.iter().enumerate() {
                    let v = &va[j * nspec..(j + 1) * nspec];
                    let zero = vec![0.0; nspec];
                    let d = vad.as_ref().map(|x| &x[j * nspec..(j + 1) * nspec]).unwrap_or(&zero);
                    let (mut im, mut dm) = (0usize, 0f64);
                    for (q, x) in d.iter().enumerate() {
                        if x.abs() > dm {
                            dm = x.abs();
                            im = q;
                        }
                    }
                    let s = sc[j];
                    writeln!(
                        f,
                        "{},{},{},{},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{},{},{:.6},{:.4},{:.5},{:.6},{:.3},{:.4},{}",
                        m.time,
                        phase,
                        node,
                        probe_row,
                        v.iter().sum::<f64>(),
                        v.iter().cloned().fold(0.0, f64::max),
                        d.iter().map(|x| x.abs()).sum::<f64>(),
                        d.iter().map(|x| x * x).sum::<f64>().sqrt(),
                        dm,
                        im / m.nth,
                        im % m.nth,
                        s[0],
                        s[1],
                        s[2],
                        s[3],
                        s[4],
                        s[5],
                        s[6]
                    )?;
                    for x in v.iter().chain(d.iter()) {
                        b.write_all(&x.to_le_bytes())?;
                    }
                    probe_row += 1;
                }
            }
        }
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
                if let Some((_, _, de)) = &tan {
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
    if let (Some(mut f), Some(mut b)) = (probe_csv, probe_bin) {
        use std::io::Write;
        f.flush()?;
        b.flush()?;
    }
    if let (Some(mut f), Some(mut b)) = (check_csv, check_bin) {
        use std::io::Write;
        f.flush()?;
        b.flush()?;
    }
    let wall = t1.elapsed().as_secs_f64();
    if m.time.secs() % 21600 != 0 {
        // energy at the end of a run that does not stop at a 6-hour mark
        let mut ej = json!({"time": m.time.to_string(), "energy": m.total_energy()});
        if let Some((_, de)) = m.tangent_outputs() {
            ej["denergy"] = json!(de);
        }
        energy.push(ej);
    }
    let per_day = wall / ((stop.secs() - start.secs()) as f64 / 86400.0);
    if let Some(p) = a.opts.get("--restart-out") {
        let t = Instant::now();
        m.restart_state().write(Path::new(p))?;
        println!("restart {p} at {}, ITSTEP {} (continue with --itstep0 {}), {:.1} s", m.time, m.itstep, m.itstep, t.elapsed().as_secs_f64());
    }
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
        "itstep": m.itstep,
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
                   "betamax": cfg.betamax,
                   "tl_nlcut": cfg.tl_nlcut,
                   "tl_nlnode": cfg.tl_nlnode,
                   "par_red": cfg.par_red,
                   "betau": cfg.betau.as_ref().map(|b| json!({"knots": b.knots, "coefs": b.coefs, "grid": [b.b0, b.db, b.n]})),
                   "scale_par": cfg.scale_par.iter().map(|(n, f)| json!([n, f])).collect::<Vec<_>>()},
    });
    let mut summary = summary;
    if let Some(u) = m.uq() {
        summary["uq"] = json!({"nx": u.tables.nx, "ny": u.tables.ny, "nsea": u.tables.nsea, "dtmax": u.steps.dtmax, "dtkth": u.dtkth, "cfltm": u.cfltm, "steps_multi": u.steps.multi,
                               "last_dtg": m.cfg.dtg});
    }
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
    Ok(())
}

/// `check-uq`: the GPU geographic step of the UQ scheme against the CPU
/// transcription of `W3XYP3`, and its tangent against central differences.
fn check_uq<F: Real>(grid: &Grid, a: &Args) -> Result<()> {
    use hww3_wave::pro3::xyp3;
    use hww3_wave::prop_uq::{read_range, write_range};
    use hww3_wave::ww3io::Time;
    use rayon::prelude::*;
    let device = Device::cuda(0);
    let client = hww3_wave::burn_ext::client(&device)?;
    let inp = inputs(a)?;
    let mut cfg = Config::default();
    perturbations(a, &mut cfg)?;
    let setup = hww3_wave::uqgrid::UqSetup::read(&a.grid)?;
    let mut m = Model::<F>::new_uq(client, grid, &setup, &inp, cfg)?;
    m.prepare()?;
    let (np, nk, nth) = (m.np, m.nk, m.nth);
    let iks: Vec<usize> = match a.opts.get("--ik") {
        Some(s) => s.split(',').map(|x| x.parse::<usize>().map(|k| k - 1)).collect::<Result<_, _>>()?,
        None => vec![0, nk / 2 - 1, nk - 1],
    };
    let pg = m.uq().expect("UQ scheme").pg.clone();
    let secs = (m.time.secs() - Time(m.time.0, 0).secs()) as f64;
    let dtg = m.cfg.dtg as f64;
    let cgall = m.device_cg();
    let len = nk * np * nth;
    let n = np * nth;
    let prec = <F as hww3_wave::backend::Scalar>::NAME;
    println!("check-uq at {} ({prec}): DTG {dtg} s, {} sea points, {} directions", m.time, np, nth);
    let mut saved = vec![];
    // --bnd-test: synthetic boundary spectra at the points of MAPSTA = 2,
    // between records 2,400 s before and 1,200 s after the end of the step
    let nspec = nk * nth;
    let bnd = if a.flags.contains("--bnd-test") {
        let nb = (0..np).filter(|&i| grid.obc.binary_search(&(i as u32)).is_ok()).count();
        let syn = |b: usize, isp: usize, ph: f64| (1e-3 * (1.2 + (0.7 * b as f64 + 0.3 * isp as f64 + ph).sin())) as f32;
        let b0: Vec<f32> = (0..nb * nspec).map(|q| syn(q / nspec, q % nspec, 0.0)).collect();
        let bn: Vec<f32> = (0..nb * nspec).map(|q| syn(q / nspec, q % nspec, 1.3)).collect();
        let (t0, tn) = (Time::from_secs(m.time.secs() - 2400), Time::from_secs(m.time.secs() + 1200));
        let bnode = m.set_test_boundary(&b0, &bn, t0, tn)?;
        println!("  synthetic boundary spectra at {} points of MAPSTA = 2, records at {t0} and {tn}", bnode.len());
        Some((bnode, b0, bn))
    } else {
        None
    };
    for &ik in &iks {
        let cg: Vec<f64> = cgall[(ik + 1) * np..(ik + 2) * np].iter().map(|&x| x as f64).collect();
        let va0 = read_range::<F>(&m.client, &m.d_va, len, ik * n, n);
        let t = Instant::now();
        let cpu: Vec<Vec<f64>> = (0..nth)
            .into_par_iter()
            .map(|th| {
                let mut f = vec![0.0; pg.nx * pg.ny];
                for (s, &i) in pg.sea_ij.iter().enumerate() {
                    f[i] = va0[s * nth + th];
                }
                let isp = ik * nth + th;
                let bv = bnd.as_ref().map(|(bnode, b0, bn)| {
                    let isb: Vec<usize> = bnode.iter().map(|&i| i as usize).collect();
                    let v0: Vec<f64> = (0..isb.len()).map(|b| b0[b * nspec + isp] as f64 * cg[isb[b]]).collect();
                    let vn: Vec<f64> = (0..isb.len()).map(|b| bn[b * nspec + isp] as f64 * cg[isb[b]]).collect();
                    (isb, v0, vn)
                });
                let bi = bv.as_ref().map(|(isb, v0, vn)| hww3_wave::pro3::BoundaryInput { dt0: 2400.0, dtn: 3600.0, isbpi: isb, bbpi0: v0, bbpin: vn });
                xyp3(&pg, isp, dtg, secs, &mut f, &cg, None, bi.as_ref());
                pg.sea_ij.iter().map(|&i| f[i]).collect()
            })
            .collect();
        let tcpu = t.elapsed().as_secs_f64();
        m.sync();
        let t = Instant::now();
        m.geographic(ik);
        m.sync();
        let tgpu = t.elapsed().as_secs_f64();
        let gpu = read_range::<F>(&m.client, &m.d_va, len, ik * n, n);
        let vmax = cpu.iter().flatten().fold(0f64, |x, &y| x.max(y.abs())).max(1e-300);
        let (mut dmax, mut d2, mut v2, mut worst) = (0f64, 0f64, 0f64, (0usize, 0usize));
        let mut above = [0usize; 3];
        for th in 0..nth {
            for s in 0..np {
                let d = (gpu[s * nth + th] - cpu[th][s]).abs();
                if d > dmax {
                    dmax = d;
                    worst = (s, th);
                }
                d2 += d * d;
                v2 += cpu[th][s] * cpu[th][s];
                for (j, e) in [1e-12, 1e-9, 1e-6].iter().enumerate() {
                    if d > e * vmax {
                        above[j] += 1;
                    }
                }
            }
        }
        let i = pg.sea_ij[worst.0];
        println!(
            "  frequency {} (max NTLOC {}): max |gpu - cpu| / max |cpu| {:.3e}, RMS ratio {:.3e}; values above 1e-12, 1e-9, 1e-6 of the max {:?}; worst at sea point {} (ix {} iy {}, direction {}): {:.6e} vs {:.6e}; CPU {:.1} s, GPU {:.4} s",
            ik + 1,
            m.uq().unwrap().max_ntloc(ik),
            dmax / vmax,
            (d2 / v2.max(1e-300)).sqrt(),
            above,
            worst.0,
            i % pg.nx + 1,
            i / pg.nx + 1,
            worst.1 + 1,
            gpu[worst.0 * nth + worst.1],
            cpu[worst.1][worst.0],
            tcpu,
            tgpu
        );
        // back to the state before the step
        write_range::<F>(&m.client, &m.d_va, len, ik * n, &va0);
        saved.push(va0);
    }
    let Some(dh) = m.tangent_handle().cloned() else { return Ok(()) };
    let h: f64 = a.opts.get("--h").map(|s| s.parse()).transpose()?.unwrap_or(1e-7);
    for (&ik, va0) in iks.iter().zip(&saved) {
        // 1. homogeneity: the scheme is positively homogeneous of degree 1,
        // so that its tangent along the field itself is the stepped field
        write_range::<F>(&m.client, &m.d_va, len, ik * n, va0);
        write_range::<F>(&m.client, &dh, len, ik * n, va0);
        m.geographic(ik);
        let tl = read_range::<F>(&m.client, &dh, len, ik * n, n);
        let v1 = read_range::<F>(&m.client, &m.d_va, len, ik * n, n);
        let vmax = v1.iter().fold(0f64, |x, &y| x.max(y.abs())).max(1e-300);
        let dmax = tl.iter().zip(&v1).fold(0f64, |x, (p, q)| x.max((p - q).abs()));
        println!("  tangent, frequency {}: along the field itself (homogeneity), max |TL - value| / max value {:.3e}", ik + 1, dmax / vmax);
        // 2. a random relative perturbation of the frequency against
        // central differences of steps h and h / 10, the smooth sea points
        // being those where these agree and the one-sided differences of
        // step h / 10 agree (no kink of the limiters or clamps within h)
        let mut seed = 0x2545F4914F6CDD1Du64 ^ (ik as u64 * 7919);
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let dva: Vec<f64> = va0.iter().map(|&v| rnd() * v).collect();
        write_range::<F>(&m.client, &m.d_va, len, ik * n, va0);
        write_range::<F>(&m.client, &dh, len, ik * n, &dva);
        m.geographic(ik);
        let tl = read_range::<F>(&m.client, &dh, len, ik * n, n);
        let f0 = read_range::<F>(&m.client, &m.d_va, len, ik * n, n);
        let run = |m: &mut Model<F>, step: f64| -> Vec<f64> {
            let v: Vec<f64> = va0.iter().zip(&dva).map(|(x, d)| x + step * d).collect();
            write_range::<F>(&m.client, &m.d_va, len, ik * n, &v);
            m.geographic(ik);
            read_range::<F>(&m.client, &m.d_va, len, ik * n, n)
        };
        let (p1, m1) = (run(&mut m, h), run(&mut m, -h));
        let (p2, m2) = (run(&mut m, h / 10.0), run(&mut m, -h / 10.0));
        write_range::<F>(&m.client, &m.d_va, len, ik * n, va0);
        let (mut nsm, mut nrough, mut nzero, mut emax, mut wnode) = (0usize, 0usize, 0usize, 0f64, 0usize);
        let mut errs = vec![];
        let mut big = vec![];
        let h2 = h / 10.0;
        let nfmax = (0..np)
            .map(|s| (s * nth..(s + 1) * nth).map(|q| ((p2[q] - m2[q]) / (2.0 * h2)).powi(2)).sum::<f64>().sqrt())
            .fold(0f64, f64::max);
        for s in 0..np {
            let r = s * nth..(s + 1) * nth;
            let f1: Vec<f64> = r.clone().map(|q| (p1[q] - m1[q]) / (2.0 * h)).collect();
            let f2: Vec<f64> = r.clone().map(|q| (p2[q] - m2[q]) / (2.0 * h2)).collect();
            let fp: Vec<f64> = r.clone().map(|q| (p2[q] - f0[q]) / h2).collect();
            let fm: Vec<f64> = r.clone().map(|q| (f0[q] - m2[q]) / h2).collect();
            let l2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
            let nf: f64 = f2.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nf == 0.0 {
                nzero += 1;
                continue;
            }
            if l2(&f1, &f2) > 1e-6 * nf || l2(&fp, &fm) > 1e-6 * nf {
                nrough += 1;
                continue;
            }
            let e = l2(&tl[r.clone()], &f2) / nf;
            nsm += 1;
            errs.push(e);
            if e > 1e-4 {
                // the size of the point's derivative and spectrum against the largest
                let vn = f0[r.clone()].iter().map(|x| x * x).sum::<f64>().sqrt();
                let vm = f0.iter().fold(0f64, |x, &y| x.max(y.abs()));
                big.push(format!("sea point {s}: |TL - FD| / |FD| {e:.3e}, |FD| / max |FD| {:.3e}, |VA| / max VA {:.3e}, |TL - FD| / max |FD| {:.3e}", nf / nfmax, vn / vm.max(1e-300), e * nf / nfmax));
            }
            if e > emax {
                emax = e;
                wnode = s;
            }
        }
        errs.sort_by(|x, y| x.total_cmp(y));
        let med = errs.get(errs.len() / 2).copied().unwrap_or(0.0);
        let p99 = errs.get(errs.len() * 99 / 100).copied().unwrap_or(0.0);
        let above = errs.iter().filter(|&&e| e > 1e-4).count();
        println!(
            "  tangent, frequency {} (h {h:e}): {nsm} smooth sea points, {nrough} not smooth (a kink within h), {nzero} with zero FD; |TL - FD| / |FD| at the smooth points: median {med:.3e}, p99 {p99:.3e}, max {emax:.3e} (sea point {wnode}), above 1e-4: {above}",
            ik + 1
        );
        for b in big.iter().take(5) {
            println!("    {b}");
        }
    }
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
        ice_ref: None,
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
    if let Some(b) = a.opts.get("--betamax") {
        cfg.betamax = Some(b.parse()?);
    }
    if let Some(c) = a.opts.get("--tl-nlcut") {
        cfg.tl_nlcut = c.parse()?;
    }
    cfg.tl_nlnode = a.flags.contains("--tl-nlnode");
    cfg.par_red = a.flags.contains("--par-red");
    if let Some(b) = a.opts.get("--betau") {
        let grid = a.opts.get("--betau-grid").map(String::as_str).unwrap_or("0.75:0.25:13");
        cfg.betau = Some(hww3_wave::sources::BetaU::parse(b, grid)?);
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
        ice_ref: None,
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
    if a.cmd == "restart-copy" {
        // the second argument is the restart, not a grid
        let t = Instant::now();
        let r = Restart::read(&a.grid)?;
        let out = a.opts.get("--out").ok_or_else(|| anyhow::anyhow!("restart-copy needs --out"))?;
        r.write(Path::new(out))?;
        println!(
            "restart {} ({}, {}, NSEA {}, NSPEC {}, time {}, TLEV {}, TICE {}, bytes after FPIS {}) written to {out} in {:.1} s",
            a.grid.display(),
            r.version,
            r.gname.trim_end(),
            r.nsea,
            r.nspec,
            r.time,
            r.tlev,
            r.tice,
            r.tail.len(),
            t.elapsed().as_secs_f64()
        );
        return Ok(());
    }
    let grid = Grid::read(&a.grid)?;
    match a.cmd.as_str() {
        "restart-hs" => restart_hs(&grid, &a),
        "run" => {
            if hww3_wave::uqgrid::UqSetup::is_uq_dir(&a.grid) {
                println!("{}: UQ scheme of a structured grid, {} sea points", a.grid.display(), grid.lon.len());
                return if a.flags.contains("--f64") { run::<f64>(&grid, None, &a) } else { run::<f32>(&grid, None, &a) };
            }
            let t = Instant::now();
            let geo = Geometry::new(&grid);
            println!("geometry: {} nodes, {} triangles, {:.1} s", geo.np, geo.ne, t.elapsed().as_secs_f64());
            if a.flags.contains("--f64") {
                run::<f64>(&grid, Some(&geo), &a)
            } else {
                run::<f32>(&grid, Some(&geo), &a)
            }
        }
        "check-uq" => {
            if a.flags.contains("--f64") {
                check_uq::<f64>(&grid, &a)
            } else {
                check_uq::<f32>(&grid, &a)
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
                ice_ref: None,
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
        "adj-geo" => {
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
                ice_ref: None,
            };
            let ik: usize = a.opts.get("--ik").map(|s| s.parse()).transpose()?.unwrap_or(12);
            let reps: usize = a.opts.get("--reps").map(|s| s.parse()).transpose()?.unwrap_or(20);
            let only = a.opts.get("--only").cloned().unwrap_or_default();
            let rep = if a.flags.contains("--f64") {
                let mut m = Model::<f64>::new(client, &grid, &geo, &inp, Config::default())?;
                m.prepare()?;
                m.adj_geo_check(ik, reps, &only)?
            } else {
                let mut m = Model::<f32>::new(client, &grid, &geo, &inp, Config::default())?;
                m.prepare()?;
                m.adj_geo_check(ik, reps, &only)?
            };
            println!("{}", serde_json::to_string(&rep)?);
            Ok(())
        }
        #[cfg(feature = "cuda-c")]
        "adj-grad" => {
            use hww3_wave::sources::{NPAR, P_BBETA, P_BU_NK, P_SDBC2, P_SDSC2, P_SNLC1};
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
                ice_ref: None,
            };
            let mut cfg = Config::default();
            perturbations(&a, &mut cfg)?;
            let hours: f64 = a.opts.get("--hours").map(|s| s.parse()).transpose()?.unwrap_or(1.0);
            let seeds = PathBuf::from(a.opts.get("--seeds").ok_or_else(|| anyhow::anyhow!("adj-grad needs --seeds DIR"))?);
            let mut m = Model::<f64>::new(client, &grid, &geo, &inp, cfg)?;
            let nsteps = (hours * 3600.0 / m.cfg.dtg as f64).round() as usize;
            let k1: usize = a.opts.get("--k1").map(|s| s.parse()).transpose()?.unwrap_or(((nsteps as f64).sqrt().ceil() as usize).max(1));
            let d = m.dtauhf_unit(&inp)?;
            let k = m.step_adj_kernels(&d, 8 << 30)?;
            let np = m.np;
            let n = m.nk * np * m.nth;
            let start = m.time;
            let mut ab = k.prop.st.alloc_zeros::<f64>(n)?;
            let t0 = Instant::now();
            let (bar, held) = m.adjoint_run(&k, &inp, nsteps, k1, &mut ab, &|t| hww3_wave::cuda_adj::seeds_from_dir(&seeds, np, t))?;
            let secs = t0.elapsed().as_secs_f64();
            let par = m.source_par().unwrap_or_else(|| vec![0.0; NPAR]);
            println!(
                "adj-grad from {start}: {nsteps} steps ({hours} h), checkpoints every {k1} steps ({held} held at most), {secs:.0} s (value run {:.0}, segments again {:.0}, of which checkpoints {:.0}; adjoint steps: sources {:.0}, propagation and forcing {:.0}, recomputed steps {:.0})",
                bar.secs_runs[0], bar.secs_runs[1], bar.secs_runs[2], bar.secs[0], bar.secs[1], bar.secs[2]
            );
            let mut out = serde_json::Map::new();
            let names = ["BETAMAX", "SDSC2", "NLPROP", "BJGAM"];
            let slots = [P_BBETA, P_SDSC2, P_SNLC1, P_SDBC2];
            for j in 0..4 {
                let v = bar.par[j] * par[slots[j]];
                println!("dJ/dln {:<10} {v:.6e}", names[j]);
                out.insert(names[j].to_string(), json!(v));
            }
            for kk in 0..par[P_BU_NK] as usize {
                let v = bar.par[4 + kk];
                println!("dJ/dc{kk:<13} {v:.6e}");
                out.insert(format!("BETAU{kk}"), json!(v));
            }
            out.insert("secs".into(), json!(secs));
            out.insert("parts".into(), json!(bar.secs));
            out.insert("runs".into(), json!(bar.secs_runs));
            out.insert("steps".into(), json!(nsteps));
            out.insert("k1".into(), json!(k1));
            println!("{}", serde_json::to_string(&serde_json::Value::Object(out))?);
            Ok(())
        }
        #[cfg(feature = "cuda-c")]
        "adj-step" => {
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
                ice_ref: None,
            };
            let steps: usize = a.opts.get("--steps").map(|s| s.parse()).transpose()?.unwrap_or(6);
            let k1: usize = a.opts.get("--k1").map(|s| s.parse()).transpose()?.unwrap_or(3);
            let mut m = Model::<f64>::new(client, &grid, &geo, &inp, Config::default())?;
            let d = m.dtauhf_unit(&inp)?;
            let rep = m.adj_step_check(&inp, steps, k1, &d)?;
            println!("{}", serde_json::to_string(&rep)?);
            Ok(())
        }
        #[cfg(feature = "cuda-c")]
        "adj-prop" => {
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
                ice_ref: None,
            };
            let mut m = Model::<f64>::new(client, &grid, &geo, &inp, Config::default())?;
            let rep = m.adj_prop_check()?;
            println!("{}", serde_json::to_string(&rep)?);
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
