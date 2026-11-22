//! M0 driver: geometry and sub-cycling of an HWW3 grid, timing of the
//! N-scheme sweeps on the GPU, dot-product tests of the transpose, check
//! against the CPU reference of the WW3 loop and, with the CubeCL backend,
//! the same dot-product test through Burn's autodiff tape.
//!
//! ```text
//! hww3-wave info  <grid dir> [--box lon0 lon1 lat0 lat1]
//! hww3-wave bench <grid dir> [--box lon0 lon1 lat0 lat1] [--backend cubecl|cuda-c]
//!                 [--reps N] [--block B] [--f64] [--reference] [--burn] [--out results.json]
//! ```

// Built without a GPU backend, the driver only has `info`, and the timing
// and test helpers are unused.
#![cfg_attr(
    not(any(feature = "cubecl", feature = "cuda-c")),
    allow(dead_code, unused_imports, unused_variables, unused_mut)
)]

use anyhow::{bail, Result};
use hww3_wave::backend::{Scalar, Sweeper};
use hww3_wave::geom::Geometry;
use hww3_wave::grid::Grid;
use hww3_wave::reference;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Instant;

/// splitmix64, uniform in [0, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
    fn vec(&mut self, n: usize, a: f64, b: f64) -> Vec<f64> {
        (0..n).map(|_| a + (b - a) * self.next()).collect()
    }
}

/// Values as the backend stores them (rounded to f32 for an f32 backend).
fn rounded<S: Sweeper>(s: &S, v: &[f64]) -> Vec<f64> {
    if s.float_bytes() == 4 {
        v.iter().map(|&x| f32::of_f64(x).as_f64()).collect()
    } else {
        v.to_vec()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

struct Args {
    cmd: String,
    grid: PathBuf,
    bbox: Option<[f64; 4]>,
    backend: String,
    reps: usize,
    block: Option<u32>,
    f64: bool,
    reference: bool,
    burn: bool,
    out: Option<PathBuf>,
}

fn parse() -> Result<Args> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 2 {
        bail!("usage: hww3-wave info|bench <grid dir> [--box lon0 lon1 lat0 lat1] [--backend cubecl|cuda-c] [--reps N] [--block B] [--f64] [--reference] [--burn] [--out file]");
    }
    let mut r = Args {
        cmd: a[0].clone(),
        grid: PathBuf::from(&a[1]),
        bbox: None,
        backend: if cfg!(feature = "cubecl") { "cubecl".into() } else { "cuda-c".into() },
        reps: 50,
        block: None,
        f64: false,
        reference: false,
        burn: false,
        out: None,
    };
    let mut i = 2;
    while i < a.len() {
        match a[i].as_str() {
            "--box" => {
                let v: Vec<f64> = a[i + 1..i + 5].iter().map(|s| s.parse().unwrap()).collect();
                r.bbox = Some([v[0], v[1], v[2], v[3]]);
                i += 4;
            }
            "--backend" => {
                r.backend = a[i + 1].clone();
                i += 1;
            }
            "--reps" => {
                r.reps = a[i + 1].parse()?;
                i += 1;
            }
            "--block" => {
                r.block = Some(a[i + 1].parse()?);
                i += 1;
            }
            "--out" => {
                r.out = Some(PathBuf::from(&a[i + 1]));
                i += 1;
            }
            "--f64" => r.f64 = true,
            "--reference" => r.reference = true,
            "--burn" => r.burn = true,
            s => bail!("unknown option {s}"),
        }
        i += 1;
    }
    Ok(r)
}

fn summary(geo: &Geometry, seconds: f64) -> Value {
    let count = |v: &[u8], x: u8| v.iter().filter(|&&b| b == x).count();
    let s: u32 = geo.iter.iter().sum();
    let sweeps: u32 = (0..geo.nk).map(|ik| geo.max_iter(ik)).sum();
    let f1 = &geo.iter[..geo.nth];
    let coast_dirs: Vec<u32> =
        (0..geo.np).filter(|&i| geo.iobp[i] == 0).map(|i| geo.mask[i].count_ones()).collect();
    json!({
        "nodes": geo.np, "triangles": geo.ne, "nk": geo.nk, "nth": geo.nth,
        "interior": count(&geo.iobp, 1), "coastal_boundary": count(&geo.iobp, 0),
        "open_boundary": count(&geo.iobp, 2), "dry": count(&geo.iobdp, 0),
        "coastal_mean_open_directions": coast_dirs.iter().sum::<u32>() as f64 / coast_dirs.len().max(1) as f64,
        "S_component_substeps": s,
        "frequency_sweeps_per_step": sweeps,
        "iter_f1": [f1.iter().min(), f1.iter().max()],
        "iter_max_by_freq": (0..geo.nk).map(|ik| geo.max_iter(ik)).collect::<Vec<_>>(),
        "geometry_seconds": seconds,
    })
}

/// Bytes a sweep must move at least: the state read and written, the nodal
/// and triangle arrays read once.
fn compulsory_bytes<S: Sweeper>(s: &S) -> f64 {
    let f = s.float_bytes() as f64;
    let d = s.dims();
    let (np, ne) = (d.np as f64, d.ne as f64);
    2.0 * s.len() as f64 * f + np * (3.0 * f + 12.0) + ne * (12.0 + 6.0 * f + 12.0)
}

fn time_sweep<S: Sweeper>(s: &S, reps: usize, transpose: bool) -> f64 {
    let mut rng = Rng(7);
    let mut a = s.upload(&rng.vec(s.len(), 0.0, 1.0));
    let mut b = s.empty();
    let run = |n: usize, a: &mut S::Buf, b: &mut S::Buf| {
        for _ in 0..n {
            if transpose {
                s.transpose(a, b, 0, 0);
            } else {
                s.forward(a, b, 0, 0, true);
            }
            std::mem::swap(a, b);
        }
    };
    run(5, &mut a, &mut b);
    s.sync();
    let t = Instant::now();
    run(reps, &mut a, &mut b);
    s.sync();
    t.elapsed().as_secs_f64() / reps as f64
}

/// One 600 s step of all frequencies. The buffer of one frequency is reused
/// for every frequency: the timing does not depend on the values.
fn time_step<S: Sweeper>(s: &S, reps: usize, transpose: bool) -> f64 {
    let mut rng = Rng(8);
    let mut a = s.upload(&rng.vec(s.len(), 0.0, 1.0));
    let mut b = s.empty();
    let run = |a: S::Buf, b: S::Buf| {
        let (mut a, mut b) = (a, b);
        for ik in 0..s.dims().nk {
            (a, b) = if transpose { s.step_transpose(a, b, ik) } else { s.step(a, b, ik, true) };
        }
        (a, b)
    };
    (a, b) = run(a, b);
    s.sync();
    let t = Instant::now();
    for _ in 0..reps {
        (a, b) = run(a, b);
    }
    s.sync();
    t.elapsed().as_secs_f64() / reps as f64
}

fn timings<S: Sweeper>(s: &mut S, reps: usize, block: Option<u32>) -> Value {
    let label = s.label();
    let bytes = compulsory_bytes(s);
    let blocks: Vec<u32> = match block {
        Some(b) => vec![b],
        None => vec![128, 256, 512],
    };
    let mut per_block = vec![];
    let mut best = (f64::INFINITY, 256);
    for &b in &blocks {
        s.set_block(b);
        let t = time_sweep(s, reps, false);
        println!("  {label} block {b}: forward sweep {:.4} ms", t * 1e3);
        per_block.push(json!({"block": b, "forward_sweep_ms": t * 1e3}));
        if t < best.0 {
            best = (t, b);
        }
    }
    s.set_block(best.1);
    let tf = time_sweep(s, reps, false);
    let tt = time_sweep(s, reps, true);
    let sreps = (reps / 10).max(2);
    let sf = time_step(s, sreps, false);
    let st = time_step(s, sreps, true);
    let sweeps: u32 = (0..s.dims().nk).map(|ik| s.max_iter(ik)).sum();
    println!(
        "  {label}: sweep forward {:.4} ms ({:.0} GB/s), transpose {:.4} ms ({:.0} GB/s); 600 s step ({sweeps} sweeps) forward {:.4} s, transpose {:.4} s",
        tf * 1e3,
        bytes / tf / 1e9,
        tt * 1e3,
        bytes / tt / 1e9,
        sf,
        st
    );
    json!({
        "backend": label, "block": best.1, "per_block": per_block,
        "forward_sweep_ms": tf * 1e3, "transpose_sweep_ms": tt * 1e3,
        "compulsory_bytes_per_sweep": bytes,
        "forward_GBps": bytes / tf / 1e9, "transpose_GBps": bytes / tt / 1e9,
        "sweeps_per_step": sweeps,
        "forward_step_s": sf, "transpose_step_s": st,
        "forward_s_per_model_day": sf * 144.0, "transpose_s_per_model_day": st * 144.0,
    })
}

fn dot_entry(case: &str, a: f64, b: f64, scale: f64) -> Value {
    let rel = (a - b).abs() / a.abs().max(b.abs());
    println!("  {case}: <Lx,y> = {a:.15e}, <x,LTy> = {b:.15e}, rel {rel:.3e}, normwise {:.3e}", (a - b).abs() / scale);
    json!({"case": case, "lhs": a, "rhs": b, "rel": rel, "normwise": (a - b).abs() / scale})
}

fn dot_tests<S: Sweeper>(s: &S) -> Value {
    let label = s.label();
    let n = s.len();
    let mut rng = Rng(12345);
    let x = rounded(s, &rng.vec(n, 0.0, 1.0));
    let y = rounded(s, &rng.vec(n, -1.0, 1.0));
    let mut out = vec![];
    let (hx, hy) = (s.upload(&x), s.upload(&y));
    let (mut o1, mut o2) = (s.empty(), s.empty());
    s.forward(&hx, &mut o1, 0, 0, false);
    s.transpose(&hy, &mut o2, 0, 0);
    let (lx, lty) = (s.download(&o1), s.download(&o2));
    out.push(dot_entry(&format!("{label} one sub-step ik=0"), dot(&lx, &y), dot(&x, &lty), norm(&lx) * norm(&y)));
    let nk = s.dims().nk;
    for ik in [0, nk / 2, nk - 1] {
        let (r1, _) = s.step(s.upload(&x), s.empty(), ik, false);
        let (r2, _) = s.step_transpose(s.upload(&y), s.empty(), ik);
        let (lx, lty) = (s.download(&r1), s.download(&r2));
        out.push(dot_entry(
            &format!("{label} 600 s step ik={ik} ({} sub-steps)", s.max_iter(ik)),
            dot(&lx, &y),
            dot(&x, &lty),
            norm(&lx) * norm(&y),
        ));
    }
    Value::Array(out)
}

/// GPU gather kernel against the CPU WW3 loop, with the clip, from a positive state.
fn reference_check<S: Sweeper>(s: &S, geo: &Geometry) -> Value {
    let label = s.label();
    let mut rng = Rng(4242);
    let x = rounded(s, &rng.vec(s.len(), 0.0, 1.0));
    let mut out = vec![];
    for ik in [0, geo.nk - 1] {
        let (r, _) = s.step(s.upload(&x), s.empty(), ik, true);
        let gpu = s.download(&r);
        let t = Instant::now();
        let cpu = reference::step(geo, &x, ik, true);
        let tcpu = t.elapsed().as_secs_f64();
        let scale = cpu.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let maxd = gpu.iter().zip(&cpu).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        let rms = (gpu.iter().zip(&cpu).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / cpu.len() as f64).sqrt();
        println!(
            "  {label} reference ik={ik} ({} sub-steps): max |gpu - cpu| / max |cpu| = {:.3e}, rms {:.3e} (cpu {:.1} s)",
            geo.max_iter(ik),
            maxd / scale,
            rms / scale,
            tcpu
        );
        out.push(json!({"backend": label, "ik": ik, "substeps": geo.max_iter(ik),
                        "max_rel": maxd / scale, "rms_rel": rms / scale, "cpu_seconds": tcpu}));
    }
    Value::Array(out)
}

/// Both kernel variants: timings, dot-product tests, reference check.
fn bench_one<S: Sweeper>(s: &mut S, geo: &Geometry, a: &Args, block: Option<u32>) -> Value {
    println!("  {}: per-triangle coefficients of {} frequencies in {:.4} s", s.label(), s.dims().nk, s.coef_seconds());
    let mut res = json!({"coef_seconds": s.coef_seconds()});
    for (v, name) in s.variants().iter().enumerate() {
        s.set_variant(v);
        let t = timings(s, a.reps, block);
        let d = dot_tests(s);
        let r = if a.reference { reference_check(s, geo) } else { Value::Null };
        res[name] = json!({"timings": t, "dot": d, "reference": r});
    }
    s.set_variant(0);
    res
}

#[cfg(feature = "cubecl")]
mod with_cubecl {
    use super::*;
    use anyhow::anyhow;
    use burn::tensor::{Device, Tensor, TensorData};
    use hww3_wave::burn_ext::{self, SubStep};
    use hww3_wave::device::DeviceGrid;
    use std::sync::Arc;

    /// `<L x, ȳ> = <x, ∂<L x, ȳ>/∂x>` through Burn's tape, f32.
    fn burn_dot(geo: &Geometry, block: u32) -> Result<Value> {
        let device = Device::cuda(0).autodiff();
        let client = burn_ext::client(&device)?;
        let dg = Arc::new(DeviceGrid::<f32>::new(client, geo, block));
        let (np, nth, n) = (dg.np, dg.nth, dg.len());
        let mut rng = Rng(999);
        let xv: Vec<f32> = rng.vec(n, 0.0, 1.0).iter().map(|&v| v as f32).collect();
        let yv: Vec<f32> = rng.vec(n, -1.0, 1.0).iter().map(|&v| v as f32).collect();
        let x = Tensor::<2>::from_data(TensorData::new(xv.clone(), [np, nth]), &device).require_grad();
        let yb = Tensor::<2>::from_data(TensorData::new(yv.clone(), [np, nth]), &device);
        let t = Instant::now();
        let lx = burn_ext::substep(x.clone(), SubStep { dg: dg.clone(), ik: 0, sub: 0 });
        let loss = (lx.clone() * yb).sum();
        let grads = loss.backward();
        let gx = x.grad(&grads).ok_or_else(|| anyhow!("no gradient for x"))?;
        let lxv: Vec<f32> = lx.into_data().try_to_vec::<f32>().map_err(|e| anyhow!("{e:?}"))?;
        let gxv: Vec<f32> = gx.into_data().try_to_vec::<f32>().map_err(|e| anyhow!("{e:?}"))?;
        let secs = t.elapsed().as_secs_f64();
        let f = |v: &[f32]| v.iter().map(|&a| a as f64).collect::<Vec<f64>>();
        let (xd, yd, lxd, gxd) = (f(&xv), f(&yv), f(&lxv), f(&gxv));
        let hy = dg.upload(&yd);
        let mut o = dg.empty();
        dg.transpose(&hy, &mut o, 0, 0);
        let direct = dg.download(&o);
        let diff = gxd.iter().zip(&direct).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        let e = dot_entry("cubecl f32 Burn autodiff one sub-step ik=0", dot(&lxd, &yd), dot(&xd, &gxd), norm(&lxd) * norm(&yd));
        println!("  Burn gradient vs transpose kernel: max |diff| = {diff:.3e} ({secs:.2} s)");
        Ok(json!({"dot": e, "max_abs_diff_gradient_vs_transpose_kernel": diff, "seconds": secs}))
    }

    pub fn run(geo: &Geometry, a: &Args, res: &mut Value) -> Result<()> {
        let device = Device::cuda(0);
        let client = burn_ext::client(&device)?;
        println!("CubeCL client from Burn's CUDA device: {}", client.name());
        let mut s = DeviceGrid::<f32>::new(client.clone(), geo, a.block.unwrap_or(256));
        res["f32"] = bench_one(&mut s, geo, a, a.block);
        if a.burn {
            res["burn_autodiff"] = burn_dot(geo, s.block).unwrap_or_else(|e| {
                println!("  Burn autodiff test failed: {e}");
                json!({"error": e.to_string()})
            });
        }
        let block = s.block;
        drop(s);
        if a.f64 {
            let mut s = DeviceGrid::<f64>::new(client, geo, block);
            res["f64"] = bench_one(&mut s, geo, a, Some(block));
        }
        Ok(())
    }
}

#[cfg(feature = "cuda-c")]
mod with_cuda_c {
    use super::*;
    use cudarc::driver::CudaContext;
    use hww3_wave::cuda_c::CudaGrid;

    pub fn run(geo: &Geometry, a: &Args, res: &mut Value) -> Result<()> {
        let ctx = CudaContext::new(0)?;
        println!("cudarc context on {}", ctx.name()?);
        let mut s = CudaGrid::<f32>::new(&ctx, geo, a.block.unwrap_or(256))?;
        println!("NVRTC compile and load: {:.2} s", s.compile_seconds);
        res["f32"] = bench_one(&mut s, geo, a, a.block);
        let block = s.block();
        drop(s);
        if a.f64 {
            let mut s = CudaGrid::<f64>::new(&ctx, geo, block)?;
            res["f64"] = bench_one(&mut s, geo, a, Some(block));
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let args = parse()?;
    let t = Instant::now();
    let mut grid = Grid::read(&args.grid)?;
    if let Some(b) = args.bbox {
        grid = grid.cut_box(b[0], b[1], b[2], b[3]);
    }
    let geo = Geometry::new(&grid);
    let sum = summary(&geo, t.elapsed().as_secs_f64());
    println!("{}", serde_json::to_string(&sum)?);
    let mut res = json!({"grid": args.grid.display().to_string(), "box": args.bbox,
                         "backend": args.backend, "geometry": sum});
    if args.cmd == "bench" {
        match args.backend.as_str() {
            #[cfg(feature = "cubecl")]
            "cubecl" => with_cubecl::run(&geo, &args, &mut res)?,
            #[cfg(feature = "cuda-c")]
            "cuda-c" => with_cuda_c::run(&geo, &args, &mut res)?,
            b => bail!("backend {b} not compiled in"),
        }
    }
    if let Some(p) = &args.out {
        std::fs::write(p, serde_json::to_string_pretty(&res)?)?;
    }
    Ok(())
}
