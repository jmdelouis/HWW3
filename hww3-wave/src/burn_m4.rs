//! Milestone M4 prototypes with Burn's autodiff: the DIA (`W3SNL1`) and one
//! geographic sub-step written as Burn tensor operations, whose gradients
//! come from Burn's tape, to compare with the hand-written adjoints of
//! [`crate::adjoint`] (cost, memory, agreement with the tangent kernels).
//!
//! The tensors hold all the nodes at once: a spectrum is `[n, nspec]` (the
//! DIA) or `[np, nth]` (one frequency of the propagation), and every
//! intermediate of the forward pass that its backward needs stays on the
//! tape until the backward runs.

use crate::adjoint::dot;
use crate::sources::*;
use burn::tensor::{DType, Device, Int, Tensor, TensorData};
use std::time::Instant;

/// An f64 tensor (the dtype must be given: without it the data is
/// converted to the default float type of the device, f32).
fn t2(v: Vec<f64>, s: [usize; 2], dev: &Device) -> Tensor<2> {
    Tensor::<2>::from_data(TensorData::new(v, s), (dev, DType::F64))
}

fn i1(v: Vec<i32>, dev: &Device) -> Tensor<1, Int> {
    let n = v.len();
    Tensor::<1, Int>::from_data(TensorData::new(v, [n]), dev)
}

fn vec2(t: Tensor<2>) -> Vec<f64> {
    t.into_data().as_slice::<f64>().expect("f64 data").to_vec()
}

/// The DIA of `n` nodes as Burn tensor operations: `S`, `D` from the
/// spectrum `a` (`[n, nspec]`), with `CONX` (`[n, nspec]`) and `CAF`
/// (`[n, nspecx]`, `CONS AF11`) given.
#[allow(clippy::too_many_arguments)]
fn dia_forward(
    a: Tensor<2>,
    conx: &Tensor<2>,
    caf: &Tensor<2>,
    ip: &[Tensor<1, Int>],
    ic: &[Tensor<1, Int>],
    t: &SrceTables,
    n: usize,
    nk: usize,
    nth: usize,
    dev: &Device,
) -> (Tensor<2>, Tensor<2>) {
    let par = &t.par;
    let (nspec, nspecx, nspecy) = (nk * nth, t.nspecx, t.nspecy);
    let zeros = || t2(vec![0.0; n * nth], [n, nth], dev);
    let core = a.div(conx.clone());
    let mut parts = vec![zeros(), core.clone()];
    let mut last = core.slice([0..n, (nk - 1) * nth..nk * nth]);
    for _ in nk..nspecy / nth {
        last = last.mul_scalar(par[P_FACHFE]);
        parts.push(last.clone());
    }
    let ue = Tensor::cat(parts, 1);
    let w: Vec<f64> = (0..8).map(|j| par[P_AWG + j]).collect();
    let g = |k: usize| ue.clone().select(1, ip[k].clone());
    let sum4 = |k0: usize, w0: usize| g(k0).mul_scalar(w[w0]) + g(k0 + 1).mul_scalar(w[w0 + 1]) + g(k0 + 2).mul_scalar(w[w0 + 2]) + g(k0 + 3).mul_scalar(w[w0 + 3]);
    let e00 = ue.clone().slice([0..n, nth..nth + nspecx]);
    let ep1 = sum4(0, 0);
    let em1 = sum4(4, 4);
    let ep2 = sum4(8, 0);
    let em2 = sum4(12, 4);
    let (dal1, dal2, dal3) = (par[P_DAL1], par[P_DAL2], par[P_DAL3]);
    let factor = caf.clone().mul(e00.clone());
    let sa1a = e00.clone().mul(ep1.clone().mul_scalar(dal1) + em1.clone().mul_scalar(dal2));
    let sa1b = sa1a.clone() - ep1.clone().mul(em1.clone()).mul_scalar(dal3);
    let sa2a = e00.clone().mul(ep2.clone().mul_scalar(dal1) + em2.clone().mul_scalar(dal2));
    let sa2b = sa2a.clone() - ep2.clone().mul(em2.clone()).mul_scalar(dal3);
    let pad = |b: Tensor<2>| Tensor::cat(vec![zeros(), b], 1);
    let gat = |b: &Tensor<2>, k: usize| b.clone().select(1, ic[k].clone());
    // pass 0: S
    let b1 = pad(factor.clone().mul(sa1b.clone()));
    let b2 = pad(factor.clone().mul(sa2b.clone()));
    let mut ss = (b1.clone().slice([0..n, nth..nth + nspec]) + b2.clone().slice([0..n, nth..nth + nspec])).mul_scalar(-2.0);
    for j in 0..8 {
        ss = ss + (gat(&b1, 2 * j) + gat(&b2, 2 * j + 1)).mul_scalar(w[j]);
    }
    let s = ss.mul(conx.clone());
    // pass 1: DN and the first four terms of D
    let b1 = pad(factor.clone().mul(e00.clone().mul_scalar(dal1) - em1.clone().mul_scalar(dal3)));
    let b2 = pad(factor.clone().mul(e00.clone().mul_scalar(dal1) - em2.clone().mul_scalar(dal3)));
    let dn = (caf.clone().mul(sa1a + sa1b) + caf.clone().mul(sa2a + sa2b)).mul_scalar(-2.0);
    let mut d = dn.slice([0..n, 0..nspec]);
    for j in 0..4 {
        d = d + (gat(&b1, 2 * j) + gat(&b2, 2 * j + 1)).mul_scalar(par[P_SWG + j]);
    }
    // pass 2: the last four terms of D
    let b1 = pad(factor.clone().mul(e00.clone().mul_scalar(dal2) - ep1.mul_scalar(dal3)));
    let b2 = pad(factor.mul(e00.mul_scalar(dal2) - ep2.mul_scalar(dal3)));
    for j in 4..8 {
        d = d + (gat(&b1, 2 * j) + gat(&b2, 2 * j + 1)).mul_scalar(par[P_SWG + j]);
    }
    (s, d)
}

/// The DIA of the first `nb` nodes through Burn's tape, f64: `S`, `D`
/// against the CubeCL value kernel (`s_ref`, `d_ref`, the first `nb` nodes
/// of [`crate::adjoint::dia_check`]), the gradient of `<S, sbar> + <D, dbar>`
/// against the hand-written adjoint (`ab_ref`) and in the dot-product test
/// with the tangent kernel (`ls`, `ld`: `L dva`, all in `refs`), and the
/// times.
#[allow(clippy::too_many_arguments)]
pub fn dia_burn(
    t: &SrceTables,
    va: &[f64],
    cg: &[f64],
    kdm: &[f64],
    nb: usize,
    nk: usize,
    nth: usize,
    refs: &[Vec<f64>; 8],
    reps: usize,
) -> anyhow::Result<serde_json::Value> {
    let dev = Device::cuda(0).autodiff();
    let par = &t.par;
    let (nspec, nspecx) = (nk * nth, t.nspecx);
    let m = nb * nspec;
    let tpiinv = 0.15915494f32 as f64;
    let mut conx = vec![0.0; m];
    let mut caf = vec![0.0; nb * nspecx];
    for i in 0..nb {
        for is in 0..nspec {
            let ik = is / nth;
            conx[i * nspec + is] = tpiinv / t.sig[ik + 1] * cg[i * nk + ik];
        }
        let x = (par[P_KDCON] * kdm[i]).max(par[P_KDMN]);
        let x2 = (par[P_SNLS3] * x).max(-1.0e15);
        let cons = par[P_SNLC1] * (1.0 + par[P_SNLS1] / x * (1.0 - par[P_SNLS2] * x) * x2.exp());
        for isp in 0..nspecx {
            caf[i * nspecx + isp] = cons * t.af11[isp];
        }
    }
    let conx_t = t2(conx, [nb, nspec], &dev);
    let caf_t = t2(caf, [nb, nspecx], &dev);
    let ip: Vec<Tensor<1, Int>> = (0..16).map(|k| i1(t.dia_ip[k * nspecx..(k + 1) * nspecx].iter().map(|&v| v as i32).collect(), &dev)).collect();
    let ic: Vec<Tensor<1, Int>> = (0..16).map(|k| i1(t.dia_ic[k * nspec..(k + 1) * nspec].iter().map(|&v| v as i32).collect(), &dev)).collect();
    let [s_ref, d_ref, dva, sbar, dbar, ab_ref, ls, ld] = refs;
    let sbar_t = t2(sbar[..m].to_vec(), [nb, nspec], &dev);
    let dbar_t = t2(dbar[..m].to_vec(), [nb, nspec], &dev);
    let run = || {
        let a = t2(va[..m].to_vec(), [nb, nspec], &dev).require_grad();
        let (s, d) = dia_forward(a.clone(), &conx_t, &caf_t, &ip, &ic, t, nb, nk, nth, &dev);
        let loss = (s.clone().mul(sbar_t.clone()) + d.clone().mul(dbar_t.clone())).sum();
        let grads = loss.backward();
        let g = a.grad(&grads).expect("gradient of the spectrum");
        (s, d, g)
    };
    // the first run compiles the kernels
    let t0 = Instant::now();
    let (s, d, g) = run();
    let (s, d, g) = (vec2(s), vec2(d), vec2(g));
    let first = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    for _ in 0..reps {
        let (_, _, g) = run();
        let _ = vec2(g);
    }
    let per = t1.elapsed().as_secs_f64() / reps as f64;
    // forward alone
    let t2_ = Instant::now();
    for _ in 0..reps {
        let a = t2(va[..m].to_vec(), [nb, nspec], &dev);
        let (s, _) = dia_forward(a, &conx_t, &caf_t, &ip, &ic, t, nb, nk, nth, &dev);
        let _ = vec2(s);
    }
    let fwd = t2_.elapsed().as_secs_f64() / reps as f64;
    let rel = |a: &[f64], b: &[f64]| {
        let mx = b.iter().fold(0f64, |m, x| m.max(x.abs())).max(1e-300);
        a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs())) / mx
    };
    let lhs = dot(&ls[..m], &sbar[..m]) + dot(&ld[..m], &dbar[..m]);
    let rhs = dot(&dva[..m], &g);
    println!("DIA with Burn autodiff, f64, {nb} nodes (tensors [nodes, bins], all nodes at once):");
    println!("  forward + backward {per:.4} s, forward alone {fwd:.4} s (means of {reps}); first run {first:.2} s");
    println!(
        "  S, D against the CubeCL value kernel: {:.3e}, {:.3e}; gradient against the hand-written adjoint: {:.3e} (relative to the largest)",
        rel(&s, &s_ref[..m]),
        rel(&d, &d_ref[..m]),
        rel(&g, &ab_ref[..m])
    );
    println!("  dot-product test with the tangent kernel: <L x, y> = {lhs:.15e}, <x, grad> = {rhs:.15e}, relative difference {:.3e}", (lhs - rhs).abs() / lhs.abs());
    Ok(serde_json::json!({
        "nodes": nb, "reps": reps, "seconds": {"forward_backward": per, "forward": fwd, "first": first},
        "rel": {"s": rel(&s, &s_ref[..m]), "d": rel(&d, &d_ref[..m]), "grad_vs_adjoint": rel(&g, &ab_ref[..m])},
        "dot": {"lhs": lhs, "rhs": rhs, "rel": (lhs - rhs).abs() / lhs.abs()},
    }))
}

/// Host copies of one frequency of a geographic sub-step (`sub = 0`) for
/// [`geo_burn`]: the input `u` (`[np, nth]`), the coefficients of the
/// frequency (`coef`, `12 ne`), the current coefficients, the directions,
/// `DT / ITER` and `ITER` of the frequency, the node arrays of
/// [`crate::prop::geo_node_x4()`] and the CubeCL output `out`.
pub struct GeoHost {
    /// Nodes.
    pub np: usize,
    /// Triangles.
    pub ne: usize,
    /// Directions.
    pub nth: usize,
    /// Input `AC` of the frequency, `[np, nth]`.
    pub u: Vec<f64>,
    /// Twelve coefficients per triangle of the frequency.
    pub coef: Vec<f64>,
    /// Six current coefficients per triangle.
    pub ccoef: Vec<f64>,
    /// Cosines of the directions.
    pub ecos: Vec<f64>,
    /// Sines of the directions.
    pub esin: Vec<f64>,
    /// `DT / ITER` of the frequency, per direction.
    pub dtk: Vec<f64>,
    /// `ITER` of the frequency, per direction.
    pub iter: Vec<u32>,
    /// `1 / SI` per node.
    pub si_inv: Vec<f64>,
    /// Direction mask, directions 0-31.
    pub mlo: Vec<u32>,
    /// Direction mask, directions 32-63, and bit 31: no update.
    pub mhi: Vec<u32>,
    /// Start of the slots of each node (`np + 1`).
    pub start: Vec<u32>,
    /// Slots `3 t + p` (triangle, vertex) of the nodes.
    pub slot: Vec<u32>,
    /// The three nodes of each triangle.
    pub ine: Vec<u32>,
    /// Open-boundary index of each node (`u32::MAX`: none).
    pub bidx: Vec<u32>,
    /// The CubeCL output of the sub-step.
    pub out: Vec<f64>,
}

/// One geographic sub-step as Burn tensor operations (all nodes at once):
/// the values at the three nodes of each triangle by `select`, `Ũ_T`, the
/// contributions `K+ (U_i - Ũ_T)` of each (node, triangle) pair, their sum
/// per node through a padded table (deterministic, no scatter), then the
/// update with its clip at 0, the mask and the open-boundary values (taken
/// from the CubeCL output, as they do not depend on `u`). The gradient of
/// `<out, ybar>` against the hand-written adjoint `ab` and in the dot-product
/// test with the tangent kernel (`dout = L du`), the times of `reps` runs.
pub fn geo_burn(h: &GeoHost, ybar: &[f64], du: &[f64], dout: &[f64], ab: &[f64], reps: usize) -> anyhow::Result<serde_json::Value> {
    use burn::tensor::Bool;
    let dev = Device::cuda(0).autodiff();
    let (np, ne, nth) = (h.np, h.ne, h.nth);
    let thr = 1.1754944e-38f32 as f64;
    let mut nm = vec![0.0; ne * nth];
    let mut fa = [vec![0.0; ne * nth], vec![0.0; ne * nth], vec![0.0; ne * nth]];
    for t in 0..ne {
        let b = t * 12;
        for k in 0..nth {
            let (c, s) = (h.ecos[k], h.esin[k]);
            let mut kn = 0.0;
            for (j, faj) in fa.iter_mut().enumerate() {
                let kj = h.coef[b + j] * c + h.coef[b + 3 + j] * s + h.ccoef[t * 6 + j];
                kn += kj.min(0.0);
                faj[t * nth + k] = h.coef[b + 6 + j] * c + h.coef[b + 9 + j] * s + h.ccoef[t * 6 + 3 + j] + kj.max(0.0);
            }
            nm[t * nth + k] = -1.0 / (-thr).min(kn);
        }
    }
    let nslot = h.start[np] as usize;
    let mut nq = vec![0i32; nslot];
    let mut tq = vec![0i32; nslot];
    let mut kp = vec![0.0; nslot * nth];
    let mut kmax = 0usize;
    for i in 0..np {
        let (q0, q1) = (h.start[i] as usize, h.start[i + 1] as usize);
        kmax = kmax.max(q1 - q0);
        for q in q0..q1 {
            let code = h.slot[q] as usize;
            let (t, p) = (code / 3, code % 3);
            nq[q] = i as i32;
            tq[q] = t as i32;
            for k in 0..nth {
                kp[q * nth + k] = (h.coef[t * 12 + p] * h.ecos[k] + h.coef[t * 12 + 3 + p] * h.esin[k] + h.ccoef[t * 6 + p]).max(0.0);
            }
        }
    }
    let mut pad = vec![nslot as i32; np * kmax];
    for i in 0..np {
        let (q0, q1) = (h.start[i] as usize, h.start[i + 1] as usize);
        for (j, q) in (q0..q1).enumerate() {
            pad[i * kmax + j] = q as i32;
        }
    }
    let mut coefn = vec![0.0; np * nth];
    let mut m = vec![0.0; np * nth];
    let mut bnd = vec![false; np * nth];
    let mut inact = vec![false; np * nth];
    for i in 0..np {
        let sa = h.si_inv[i] * (1.0 - ((h.mhi[i] >> 31) & 1) as f64);
        for k in 0..nth {
            let w = if k < 32 { h.mlo[i] } else { h.mhi[i] };
            coefn[i * nth + k] = h.dtk[k] * sa;
            m[i * nth + k] = ((w >> (k % 32)) & 1) as f64;
            bnd[i * nth + k] = h.bidx[i] != 0xffff_ffff;
            inact[i * nth + k] = h.iter[k] == 0;
        }
    }
    let b2 = |v: Vec<bool>| Tensor::<2, Bool>::from_data(TensorData::new(v, [np, nth]), &dev);
    let i1t = |v: Vec<i32>| i1(v, &dev);
    let tri = |j: usize| i1t((0..ne).map(|t| h.ine[3 * t + j] as i32).collect());
    let (it1, it2, it3) = (tri(0), tri(1), tri(2));
    let nm_t = t2(nm, [ne, nth], &dev);
    let [fa1, fa2, fa3] = fa;
    let (fa1_t, fa2_t, fa3_t) = (t2(fa1, [ne, nth], &dev), t2(fa2, [ne, nth], &dev), t2(fa3, [ne, nth], &dev));
    let kp_t = t2(kp, [nslot, nth], &dev);
    let (nq_t, tq_t, pad_t) = (i1t(nq), i1t(tq), i1t(pad));
    let coefn_t = t2(coefn, [np, nth], &dev);
    let m_t = t2(m, [np, nth], &dev);
    let bv_t = t2(h.out.clone(), [np, nth], &dev);
    let (bnd_t, inact_t) = (b2(bnd), b2(inact));
    let ybar_t = t2(ybar.to_vec(), [np, nth], &dev);
    let zrow = t2(vec![0.0; nth], [1, nth], &dev);
    let forward = |u: Tensor<2>| -> Tensor<2> {
        let ut = nm_t.clone().mul(
            fa1_t.clone().mul(u.clone().select(0, it1.clone()))
                + fa2_t.clone().mul(u.clone().select(0, it2.clone()))
                + fa3_t.clone().mul(u.clone().select(0, it3.clone())),
        );
        let contrib = kp_t.clone().mul(u.clone().select(0, nq_t.clone()) - ut.select(0, tq_t.clone()));
        let ext = Tensor::cat(vec![contrib, zrow.clone()], 0);
        let st: Tensor<2> = ext.select(0, pad_t.clone()).reshape([np, kmax, nth]).sum_dim(1).reshape([np, nth]);
        let x = u.clone() - st.mul(coefn_t.clone());
        // max(x, 0) with a zero derivative at x = 0, as the tangent kernel
        let v = x.clone().mask_fill(x.lower_equal_elem(0.0), 0.0).mul(m_t.clone());
        v.mask_where(bnd_t.clone(), bv_t.clone()).mask_where(inact_t.clone(), u)
    };
    let run = || {
        let u = t2(h.u.clone(), [np, nth], &dev).require_grad();
        let out = forward(u.clone());
        let loss = out.clone().mul(ybar_t.clone()).sum();
        let grads = loss.backward();
        let g = u.grad(&grads).expect("gradient of u");
        (out, g)
    };
    let t0 = Instant::now();
    let (out, g) = run();
    let (out, g) = (vec2(out), vec2(g));
    let first = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    for _ in 0..reps {
        let (_, g) = run();
        let _ = vec2(g);
    }
    let per = t1.elapsed().as_secs_f64() / reps as f64;
    let t2_ = Instant::now();
    for _ in 0..reps {
        let o = forward(t2(h.u.clone(), [np, nth], &dev));
        let _ = vec2(o);
    }
    let fwd = t2_.elapsed().as_secs_f64() / reps as f64;
    let rel = |a: &[f64], b: &[f64]| {
        let mx = b.iter().fold(0f64, |m, x| m.max(x.abs())).max(1e-300);
        a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs())) / mx
    };
    let lhs = dot(dout, ybar);
    let rhs = dot(du, &g);
    // tensors of the forward pass: [ne, nth] x (3 gathers, 3 products, UT),
    // [slots, nth] x (2 gathers, the difference, the contributions),
    // [np kmax, nth] (padded), [np, nth] x (ST, X, V, OUT)
    let bytes = 8 * nth * (7 * ne + 4 * nslot + np * kmax + 4 * np);
    println!("one geographic sub-step with Burn autodiff (f64, all nodes at once, padded table of {kmax} triangles per node):");
    println!("  forward + backward {:.3} ms, forward alone {:.3} ms (means of {reps}); first run {first:.2} s", per * 1e3, fwd * 1e3);
    println!(
        "  output against the CubeCL value kernel {:.3e}; gradient against the hand-written adjoint {:.3e} (relative to the largest)",
        rel(&out, &h.out),
        rel(&g, ab)
    );
    println!(
        "  dot-product test with the tangent kernel: <L x, y> = {lhs:.15e}, <x, grad> = {rhs:.15e}, relative difference {:.3e}",
        (lhs - rhs).abs() / lhs.abs()
    );
    println!("  intermediate tensors of the forward pass: about {:.2} GB ({:.0} B per node)", bytes as f64 / 1e9, bytes as f64 / np as f64);
    Ok(serde_json::json!({
        "ms": {"forward_backward": per * 1e3, "forward": fwd * 1e3}, "first": first, "kmax": kmax,
        "rel": {"out": rel(&out, &h.out), "grad_vs_adjoint": rel(&g, ab)},
        "dot": {"lhs": lhs, "rhs": rhs, "rel": (lhs - rhs).abs() / lhs.abs()},
        "forward_tensor_bytes": bytes,
    }))
}
