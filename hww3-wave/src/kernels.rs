//! CubeCL kernels of one sub-step of the WW3 PDLIB explicit N scheme
//! (`PDLIB_W3XYPFSN2`) and of its transpose, for all directions of one
//! frequency.
//!
//! WW3 loops over the triangles and scatters into the nodes:
//!
//! ```text
//! ST(i) += K+_iT (U_i - Ũ_T),  Ũ_T = NM_T Σ_q FLALL_qT U_q
//! U_i  <- max(0, U_i - DTSI_i ST_i (1 - IOBPA_i)) IOBPD(θ, i) IOBDP_i
//! ```
//!
//! Here one unit computes one (node, direction) value and gathers over the
//! triangles around its node, recomputing `K`, `NM` and `FLALL` of each
//! triangle from the nodal velocities and `PDLIB_IEN`, so there are no
//! atomics and nothing per triangle is stored. The state is laid out as
//! `u[i nth + k]`, so the 36 directions of a node are contiguous.
//!
//! For fixed velocities one sub-step is `y = m (x - d Σ_T K+ (x_i - Σ_q W_q x_q))`
//! with `d = (1 - IOBPA) DTSI`, `W_q = NM FLALL_q` (without the `max(0, .)`
//! clip, which is the identity on positive states). Its transpose is
//!
//! ```text
//! z_j = m_j ȳ_j - Σ_{T∋j} [ K+_jT v_j - W_jT Σ_{i∈T} K+_iT v_i ],  v_i = m_i d_i ȳ_i,
//! ```
//!
//! again a gather over the triangles around `j`.
//!
//! A direction whose `ITER(ik, k)` is not larger than `sub` is copied, so
//! that launching sub-steps `0 .. max_k ITER` reproduces WW3's sub-cycling
//! of each component with its own time step `DT / ITER`.
//!
//! Per-node masks: bit `k` of `mlo` (k < 32) or `mhi` (k >= 32) is
//! `IOBPD(k, i) IOBDP_i`; bit 31 of `mhi` is `IOBPA_i`.

// The `#[cube]` macro generates, for every kernel, a launch struct and
// functions that carry no documentation; the kernels themselves are
// documented below.
#![allow(missing_docs)]

use cubecl::prelude::*;

/// Fused forward sub-step: one unit per (node, direction), gathering over
/// the triangles around the node and recomputing `K`, `NM` and `FLALL` of
/// each of them.
#[cube(launch_unchecked)]
pub fn nscheme_forward<F: Float>(
    u: &[F],
    out: &mut [F],
    cx: &[F],
    cy: &[F],
    ecos: &[F],
    esin: &[F],
    dtk: &[F],
    iter: &[u32],
    si_inv: &[F],
    mlo: &[u32],
    mhi: &[u32],
    start: &[u32],
    slot: &[u32],
    ine: &[u32],
    ien: &[F],
    np: usize,
    ik: usize,
    sub: u32,
    #[comptime] nth: usize,
    #[comptime] clip: bool,
) {
    let g = ABSOLUTE_POS;
    if g < np * nth {
        let i = g / nth;
        let k = g - i * nth;
        let ui = u[g];
        let kk = ik * nth + k;
        if sub < iter[kk] {
            let zero = F::new(0.0f32);
            let one = F::new(1.0f32);
            let two = F::new(2.0f32);
            let sixth = one / F::new(6.0f32);
            let thr = F::new(1.1754944e-38f32);
            let co = ecos[k];
            let sn = esin[k];
            let off = ik * np;
            let s0 = start[i] as usize;
            let s1 = start[i + 1] as usize;
            let mut st = zero;
            for s in s0..s1 {
                let code = slot[s] as usize;
                let t = code / 3;
                let p = code - 3 * t;
                let n1 = ine[3 * t] as usize;
                let n2 = ine[3 * t + 1] as usize;
                let n3 = ine[3 * t + 2] as usize;
                let c1x = cx[off + n1] * co;
                let c1y = cy[off + n1] * sn;
                let c2x = cx[off + n2] * co;
                let c2y = cy[off + n2] * sn;
                let c3x = cx[off + n3] * co;
                let c3y = cy[off + n3] * sn;
                let e1x = ien[6 * t];
                let e1y = ien[6 * t + 1];
                let e2x = ien[6 * t + 2];
                let e2y = ien[6 * t + 3];
                let e3x = ien[6 * t + 4];
                let e3y = ien[6 * t + 5];
                let lx = (c1x + c2x + c3x) * sixth;
                let ly = (c1y + c2y + c3y) * sixth;
                let k1 = lx * e1x + ly * e1y;
                let k2 = lx * e2x + ly * e2y;
                let k3 = lx * e3x + ly * e3y;
                let kp1 = max(k1, zero);
                let kp2 = max(k2, zero);
                let kp3 = max(k3, zero);
                let kn = min(k1, zero) + min(k2, zero) + min(k3, zero);
                let nm = -one / min(-thr, kn);
                let fl11 = c2x * e1x + c2y * e1y;
                let fl12 = c3x * e1x + c3y * e1y;
                let fl21 = c3x * e2x + c3y * e2y;
                let fl22 = c1x * e2x + c1y * e2y;
                let fl31 = c1x * e3x + c1y * e3y;
                let fl32 = c2x * e3x + c2y * e3y;
                let fa1 = (two * fl31 + fl32 + two * fl22 + fl21) * sixth + kp1;
                let fa2 = (two * fl11 + fl12 + two * fl32 + fl31) * sixth + kp2;
                let fa3 = (two * fl21 + fl22 + two * fl12 + fl11) * sixth + kp3;
                let ut = nm * (fa1 * u[n1 * nth + k] + fa2 * u[n2 * nth + k] + fa3 * u[n3 * nth + k]);
                let kpi = select(p == 0, kp1, select(p == 1, kp2, kp3));
                st += kpi * (ui - ut);
            }
            let w = select(k < 32, mlo[i], mhi[i]);
            let m = F::cast_from((w >> ((k % 32) as u32)) & 1u32);
            let a = one - F::cast_from((mhi[i] >> 31u32) & 1u32);
            let mut v = ui - dtk[kk] * si_inv[i] * st * a;
            if clip {
                v = max(v, zero);
            }
            out[g] = v * m;
        } else {
            out[g] = ui;
        }
    }
}

/// Fused transposed sub-step: one unit per (node, direction), gathering
/// over the triangles around the node (formula of the module documentation).
#[cube(launch_unchecked)]
pub fn nscheme_transpose<F: Float>(
    y: &[F],
    out: &mut [F],
    cx: &[F],
    cy: &[F],
    ecos: &[F],
    esin: &[F],
    dtk: &[F],
    iter: &[u32],
    si_inv: &[F],
    mlo: &[u32],
    mhi: &[u32],
    start: &[u32],
    slot: &[u32],
    ine: &[u32],
    ien: &[F],
    np: usize,
    ik: usize,
    sub: u32,
    #[comptime] nth: usize,
) {
    let g = ABSOLUTE_POS;
    if g < np * nth {
        let j = g / nth;
        let k = g - j * nth;
        let yj = y[g];
        let kk = ik * nth + k;
        if sub < iter[kk] {
            let zero = F::new(0.0f32);
            let one = F::new(1.0f32);
            let two = F::new(2.0f32);
            let sixth = one / F::new(6.0f32);
            let thr = F::new(1.1754944e-38f32);
            let co = ecos[k];
            let sn = esin[k];
            let dt = dtk[kk];
            let off = ik * np;
            let lo = k < 32;
            let sh = (k % 32) as u32;
            let wj = select(lo, mlo[j], mhi[j]);
            let mj = F::cast_from((wj >> sh) & 1u32);
            let aj = one - F::cast_from((mhi[j] >> 31u32) & 1u32);
            let vj = mj * aj * dt * si_inv[j] * yj;
            let s0 = start[j] as usize;
            let s1 = start[j + 1] as usize;
            let mut acc = zero;
            for s in s0..s1 {
                let code = slot[s] as usize;
                let t = code / 3;
                let p = code - 3 * t;
                let n1 = ine[3 * t] as usize;
                let n2 = ine[3 * t + 1] as usize;
                let n3 = ine[3 * t + 2] as usize;
                let c1x = cx[off + n1] * co;
                let c1y = cy[off + n1] * sn;
                let c2x = cx[off + n2] * co;
                let c2y = cy[off + n2] * sn;
                let c3x = cx[off + n3] * co;
                let c3y = cy[off + n3] * sn;
                let e1x = ien[6 * t];
                let e1y = ien[6 * t + 1];
                let e2x = ien[6 * t + 2];
                let e2y = ien[6 * t + 3];
                let e3x = ien[6 * t + 4];
                let e3y = ien[6 * t + 5];
                let lx = (c1x + c2x + c3x) * sixth;
                let ly = (c1y + c2y + c3y) * sixth;
                let k1 = lx * e1x + ly * e1y;
                let k2 = lx * e2x + ly * e2y;
                let k3 = lx * e3x + ly * e3y;
                let kp1 = max(k1, zero);
                let kp2 = max(k2, zero);
                let kp3 = max(k3, zero);
                let kn = min(k1, zero) + min(k2, zero) + min(k3, zero);
                let nm = -one / min(-thr, kn);
                let fl11 = c2x * e1x + c2y * e1y;
                let fl12 = c3x * e1x + c3y * e1y;
                let fl21 = c3x * e2x + c3y * e2y;
                let fl22 = c1x * e2x + c1y * e2y;
                let fl31 = c1x * e3x + c1y * e3y;
                let fl32 = c2x * e3x + c2y * e3y;
                let fa1 = (two * fl31 + fl32 + two * fl22 + fl21) * sixth + kp1;
                let fa2 = (two * fl11 + fl12 + two * fl32 + fl31) * sixth + kp2;
                let fa3 = (two * fl21 + fl22 + two * fl12 + fl11) * sixth + kp3;
                let w1 = select(lo, mlo[n1], mhi[n1]);
                let w2 = select(lo, mlo[n2], mhi[n2]);
                let w3 = select(lo, mlo[n3], mhi[n3]);
                let v1 = F::cast_from((w1 >> sh) & 1u32)
                    * (one - F::cast_from((mhi[n1] >> 31u32) & 1u32))
                    * si_inv[n1]
                    * y[n1 * nth + k];
                let v2 = F::cast_from((w2 >> sh) & 1u32)
                    * (one - F::cast_from((mhi[n2] >> 31u32) & 1u32))
                    * si_inv[n2]
                    * y[n2 * nth + k];
                let v3 = F::cast_from((w3 >> sh) & 1u32)
                    * (one - F::cast_from((mhi[n3] >> 31u32) & 1u32))
                    * si_inv[n3]
                    * y[n3 * nth + k];
                let kv = dt * (kp1 * v1 + kp2 * v2 + kp3 * v3);
                let kpj = select(p == 0, kp1, select(p == 1, kp2, kp3));
                let faj = select(p == 0, fa1, select(p == 1, fa2, fa3));
                acc += kpj * vj - nm * faj * kv;
            }
            out[g] = mj * yj - acc;
        } else {
            out[g] = yj;
        }
    }
}

// ---------------------------------------------------------------------------
// Two-phase variant.
//
// For fixed velocities, the quantities of a triangle are linear in
// (cos θ, sin θ): `K_q = a_q cos θ + b_q sin θ` and the Simpson part of
// `FLALL_q` is `A_q cos θ + B_q sin θ`. `nscheme_coef` stores the twelve
// coefficients (a1..3, b1..3, A1..3, B1..3) of every triangle and frequency.
// A sub-step is then one pass over (triangle, direction), which computes
// `Ũ_T` once instead of once per vertex, and one gather over
// (node, direction), still without atomics.
// ---------------------------------------------------------------------------

/// Twelve coefficients per triangle and frequency, `(a, b, A, B)` of its
/// three vertices, from which `K` and `FLALL` of any direction follow.
#[cube(launch_unchecked)]
pub fn nscheme_coef<F: Float>(
    cx: &[F],
    cy: &[F],
    ine: &[u32],
    ien: &[F],
    coef: &mut [F],
    np: usize,
    ne: usize,
    ik: usize,
) {
    let t = ABSOLUTE_POS;
    if t < ne {
        let two = F::new(2.0f32);
        let sixth = F::new(1.0f32) / F::new(6.0f32);
        let off = ik * np;
        let n1 = ine[3 * t] as usize;
        let n2 = ine[3 * t + 1] as usize;
        let n3 = ine[3 * t + 2] as usize;
        let x1 = cx[off + n1];
        let x2 = cx[off + n2];
        let x3 = cx[off + n3];
        let y1 = cy[off + n1];
        let y2 = cy[off + n2];
        let y3 = cy[off + n3];
        let e1x = ien[6 * t];
        let e1y = ien[6 * t + 1];
        let e2x = ien[6 * t + 2];
        let e2y = ien[6 * t + 3];
        let e3x = ien[6 * t + 4];
        let e3y = ien[6 * t + 5];
        let sx = (x1 + x2 + x3) * sixth;
        let sy = (y1 + y2 + y3) * sixth;
        let b = (ik * ne + t) * 12;
        coef[b] = sx * e1x;
        coef[b + 1] = sx * e2x;
        coef[b + 2] = sx * e3x;
        coef[b + 3] = sy * e1y;
        coef[b + 4] = sy * e2y;
        coef[b + 5] = sy * e3y;
        coef[b + 6] = (two * x1 * e3x + x2 * e3x + two * x1 * e2x + x3 * e2x) * sixth;
        coef[b + 7] = (two * x2 * e1x + x3 * e1x + two * x2 * e3x + x1 * e3x) * sixth;
        coef[b + 8] = (two * x3 * e2x + x1 * e2x + two * x3 * e1x + x2 * e1x) * sixth;
        coef[b + 9] = (two * y1 * e3y + y2 * e3y + two * y1 * e2y + y3 * e2y) * sixth;
        coef[b + 10] = (two * y2 * e1y + y3 * e1y + two * y2 * e3y + y1 * e3y) * sixth;
        coef[b + 11] = (two * y3 * e2y + y1 * e2y + two * y3 * e1y + y2 * e1y) * sixth;
    }
}

/// Phase 1 of the forward sub-step: `ut[t nth + k] = Ũ_T` for direction k.
#[cube(launch_unchecked)]
pub fn nscheme_tri_forward<F: Float>(
    u: &[F],
    ut: &mut [F],
    coef: &[F],
    ecos: &[F],
    esin: &[F],
    iter: &[u32],
    ine: &[u32],
    ne: usize,
    ik: usize,
    sub: u32,
    #[comptime] nth: usize,
) {
    let g = ABSOLUTE_POS;
    if g < ne * nth {
        let t = g / nth;
        let k = g - t * nth;
        if sub < iter[ik * nth + k] {
            let zero = F::new(0.0f32);
            let one = F::new(1.0f32);
            let thr = F::new(1.1754944e-38f32);
            let c = ecos[k];
            let s = esin[k];
            let b = (ik * ne + t) * 12;
            let k1 = coef[b] * c + coef[b + 3] * s;
            let k2 = coef[b + 1] * c + coef[b + 4] * s;
            let k3 = coef[b + 2] * c + coef[b + 5] * s;
            let kn = min(k1, zero) + min(k2, zero) + min(k3, zero);
            let nm = -one / min(-thr, kn);
            let fa1 = coef[b + 6] * c + coef[b + 9] * s + max(k1, zero);
            let fa2 = coef[b + 7] * c + coef[b + 10] * s + max(k2, zero);
            let fa3 = coef[b + 8] * c + coef[b + 11] * s + max(k3, zero);
            let n1 = ine[3 * t] as usize;
            let n2 = ine[3 * t + 1] as usize;
            let n3 = ine[3 * t + 2] as usize;
            ut[g] = nm * (fa1 * u[n1 * nth + k] + fa2 * u[n2 * nth + k] + fa3 * u[n3 * nth + k]);
        }
    }
}

/// Phase 2 of the forward sub-step: gather `Σ_T K+_iT (U_i - Ũ_T)` and update.
#[cube(launch_unchecked)]
pub fn nscheme_node_forward<F: Float>(
    u: &[F],
    out: &mut [F],
    ut: &[F],
    coef: &[F],
    ecos: &[F],
    esin: &[F],
    dtk: &[F],
    iter: &[u32],
    si_inv: &[F],
    mlo: &[u32],
    mhi: &[u32],
    start: &[u32],
    slot: &[u32],
    np: usize,
    ne: usize,
    ik: usize,
    sub: u32,
    #[comptime] nth: usize,
    #[comptime] clip: bool,
) {
    let g = ABSOLUTE_POS;
    if g < np * nth {
        let i = g / nth;
        let k = g - i * nth;
        let ui = u[g];
        let kk = ik * nth + k;
        if sub < iter[kk] {
            let zero = F::new(0.0f32);
            let one = F::new(1.0f32);
            let c = ecos[k];
            let s = esin[k];
            let cb = ik * ne * 12;
            let s0 = start[i] as usize;
            let s1 = start[i + 1] as usize;
            let mut st = zero;
            for q in s0..s1 {
                let code = slot[q] as usize;
                let t = code / 3;
                let p = code - 3 * t;
                let b = cb + t * 12 + p;
                let kp = max(coef[b] * c + coef[b + 3] * s, zero);
                st += kp * (ui - ut[t * nth + k]);
            }
            let w = select(k < 32, mlo[i], mhi[i]);
            let m = F::cast_from((w >> ((k % 32) as u32)) & 1u32);
            let a = one - F::cast_from((mhi[i] >> 31u32) & 1u32);
            let mut v = ui - dtk[kk] * si_inv[i] * st * a;
            if clip {
                v = max(v, zero);
            }
            out[g] = v * m;
        } else {
            out[g] = ui;
        }
    }
}

/// Phase 1 of the transposed sub-step: `gt[t nth + k] = NM_T DT/ITER Σ_{i∈T} K+_iT v_i`.
#[cube(launch_unchecked)]
pub fn nscheme_tri_transpose<F: Float>(
    y: &[F],
    gt: &mut [F],
    coef: &[F],
    ecos: &[F],
    esin: &[F],
    dtk: &[F],
    iter: &[u32],
    si_inv: &[F],
    mlo: &[u32],
    mhi: &[u32],
    ine: &[u32],
    ne: usize,
    ik: usize,
    sub: u32,
    #[comptime] nth: usize,
) {
    let g = ABSOLUTE_POS;
    if g < ne * nth {
        let t = g / nth;
        let k = g - t * nth;
        let kk = ik * nth + k;
        if sub < iter[kk] {
            let zero = F::new(0.0f32);
            let one = F::new(1.0f32);
            let thr = F::new(1.1754944e-38f32);
            let c = ecos[k];
            let s = esin[k];
            let b = (ik * ne + t) * 12;
            let k1 = coef[b] * c + coef[b + 3] * s;
            let k2 = coef[b + 1] * c + coef[b + 4] * s;
            let k3 = coef[b + 2] * c + coef[b + 5] * s;
            let kn = min(k1, zero) + min(k2, zero) + min(k3, zero);
            let nm = -one / min(-thr, kn);
            let lo = k < 32;
            let sh = (k % 32) as u32;
            let n1 = ine[3 * t] as usize;
            let n2 = ine[3 * t + 1] as usize;
            let n3 = ine[3 * t + 2] as usize;
            let w1 = select(lo, mlo[n1], mhi[n1]);
            let w2 = select(lo, mlo[n2], mhi[n2]);
            let w3 = select(lo, mlo[n3], mhi[n3]);
            let v1 = F::cast_from((w1 >> sh) & 1u32)
                * (one - F::cast_from((mhi[n1] >> 31u32) & 1u32))
                * si_inv[n1]
                * y[n1 * nth + k];
            let v2 = F::cast_from((w2 >> sh) & 1u32)
                * (one - F::cast_from((mhi[n2] >> 31u32) & 1u32))
                * si_inv[n2]
                * y[n2 * nth + k];
            let v3 = F::cast_from((w3 >> sh) & 1u32)
                * (one - F::cast_from((mhi[n3] >> 31u32) & 1u32))
                * si_inv[n3]
                * y[n3 * nth + k];
            gt[g] = nm * dtk[kk] * (max(k1, zero) * v1 + max(k2, zero) * v2 + max(k3, zero) * v3);
        }
    }
}

/// Phase 2 of the transposed sub-step:
/// `z_j = m_j ȳ_j - Σ_{T∋j} [K+_jT v_j - FLALL_jT gt_T]`.
#[cube(launch_unchecked)]
pub fn nscheme_node_transpose<F: Float>(
    y: &[F],
    out: &mut [F],
    gt: &[F],
    coef: &[F],
    ecos: &[F],
    esin: &[F],
    dtk: &[F],
    iter: &[u32],
    si_inv: &[F],
    mlo: &[u32],
    mhi: &[u32],
    start: &[u32],
    slot: &[u32],
    np: usize,
    ne: usize,
    ik: usize,
    sub: u32,
    #[comptime] nth: usize,
) {
    let g = ABSOLUTE_POS;
    if g < np * nth {
        let j = g / nth;
        let k = g - j * nth;
        let yj = y[g];
        let kk = ik * nth + k;
        if sub < iter[kk] {
            let zero = F::new(0.0f32);
            let one = F::new(1.0f32);
            let c = ecos[k];
            let s = esin[k];
            let w = select(k < 32, mlo[j], mhi[j]);
            let mj = F::cast_from((w >> ((k % 32) as u32)) & 1u32);
            let aj = one - F::cast_from((mhi[j] >> 31u32) & 1u32);
            let vj = mj * aj * dtk[kk] * si_inv[j] * yj;
            let cb = ik * ne * 12;
            let s0 = start[j] as usize;
            let s1 = start[j + 1] as usize;
            let mut acc = zero;
            for q in s0..s1 {
                let code = slot[q] as usize;
                let t = code / 3;
                let p = code - 3 * t;
                let b = cb + t * 12 + p;
                let kp = max(coef[b] * c + coef[b + 3] * s, zero);
                let fa = coef[b + 6] * c + coef[b + 9] * s + kp;
                acc += kp * vj - fa * gt[t * nth + k];
            }
            out[g] = mj * yj - acc;
        } else {
            out[g] = yj;
        }
    }
}

// ---------------------------------------------------------------------------
// Two-phase variant with four directions per unit (nth a multiple of 4): the
// triangle decoding, the vertex indices and the coefficients are loaded once
// for four directions.
// ---------------------------------------------------------------------------

#[cube]
fn mask_bit<F: Float>(wlo: u32, whi: u32, k: usize) -> F {
    let w = select(k < 32, wlo, whi);
    F::cast_from((w >> ((k % 32) as u32)) & 1u32)
}

/// Phase 1 of the forward sub-step, four directions per unit:
/// `ut[t nth + k] = Ũ_T`.
#[cube(launch_unchecked)]
pub fn nscheme_tri_forward_x4<F: Float>(
    u: &[F],
    ut: &mut [F],
    coef: &[F],
    ecos: &[F],
    esin: &[F],
    iter: &[u32],
    ine: &[u32],
    ne: usize,
    ik: usize,
    sub: u32,
    #[comptime] nth: usize,
) {
    let g = ABSOLUTE_POS;
    let nw = nth / 4;
    if g < ne * nw {
        let t = g / nw;
        let k0 = (g - t * nw) * 4;
        let zero = F::new(0.0f32);
        let one = F::new(1.0f32);
        let thr = F::new(1.1754944e-38f32);
        let b = (ik * ne + t) * 12;
        let a1 = coef[b];
        let a2 = coef[b + 1];
        let a3 = coef[b + 2];
        let b1 = coef[b + 3];
        let b2 = coef[b + 4];
        let b3 = coef[b + 5];
        let ca1 = coef[b + 6];
        let ca2 = coef[b + 7];
        let ca3 = coef[b + 8];
        let cb1 = coef[b + 9];
        let cb2 = coef[b + 10];
        let cb3 = coef[b + 11];
        let r1 = ine[3 * t] as usize * nth;
        let r2 = ine[3 * t + 1] as usize * nth;
        let r3 = ine[3 * t + 2] as usize * nth;
        for d in 0..4usize {
            let k = k0 + d;
            if sub < iter[ik * nth + k] {
                let c = ecos[k];
                let s = esin[k];
                let k1 = a1 * c + b1 * s;
                let k2 = a2 * c + b2 * s;
                let k3 = a3 * c + b3 * s;
                let kn = min(k1, zero) + min(k2, zero) + min(k3, zero);
                let nm = -one / min(-thr, kn);
                let fa1 = ca1 * c + cb1 * s + max(k1, zero);
                let fa2 = ca2 * c + cb2 * s + max(k2, zero);
                let fa3 = ca3 * c + cb3 * s + max(k3, zero);
                ut[t * nth + k] = nm * (fa1 * u[r1 + k] + fa2 * u[r2 + k] + fa3 * u[r3 + k]);
            }
        }
    }
}

#[cube]
fn finish_forward<F: Float>(ui: F, st: F, dt: F, sa: F, m: F, active: bool, #[comptime] clip: bool) -> F {
    let mut v = ui - dt * sa * st;
    if clip {
        v = max(v, F::new(0.0f32));
    }
    select(active, v * m, ui)
}

/// Phase 2 of the forward sub-step, four directions per unit: gather and
/// update.
#[cube(launch_unchecked)]
pub fn nscheme_node_forward_x4<F: Float>(
    u: &[F],
    out: &mut [F],
    ut: &[F],
    coef: &[F],
    ecos: &[F],
    esin: &[F],
    dtk: &[F],
    iter: &[u32],
    si_inv: &[F],
    mlo: &[u32],
    mhi: &[u32],
    start: &[u32],
    slot: &[u32],
    np: usize,
    ne: usize,
    ik: usize,
    sub: u32,
    #[comptime] nth: usize,
    #[comptime] clip: bool,
) {
    let g = ABSOLUTE_POS;
    let nw = nth / 4;
    if g < np * nw {
        let i = g / nw;
        let k0 = (g - i * nw) * 4;
        let base = i * nth + k0;
        let zero = F::new(0.0f32);
        let one = F::new(1.0f32);
        let c0 = ecos[k0];
        let c1 = ecos[k0 + 1];
        let c2 = ecos[k0 + 2];
        let c3 = ecos[k0 + 3];
        let s0 = esin[k0];
        let s1 = esin[k0 + 1];
        let s2 = esin[k0 + 2];
        let s3 = esin[k0 + 3];
        let u0 = u[base];
        let u1 = u[base + 1];
        let u2 = u[base + 2];
        let u3 = u[base + 3];
        let mut st0 = zero;
        let mut st1 = zero;
        let mut st2 = zero;
        let mut st3 = zero;
        let cb = ik * ne * 12;
        let q0 = start[i] as usize;
        let q1 = start[i + 1] as usize;
        for q in q0..q1 {
            let code = slot[q] as usize;
            let t = code / 3;
            let p = code - 3 * t;
            let b = cb + t * 12 + p;
            let a = coef[b];
            let bb = coef[b + 3];
            let r = t * nth + k0;
            st0 += max(a * c0 + bb * s0, zero) * (u0 - ut[r]);
            st1 += max(a * c1 + bb * s1, zero) * (u1 - ut[r + 1]);
            st2 += max(a * c2 + bb * s2, zero) * (u2 - ut[r + 2]);
            st3 += max(a * c3 + bb * s3, zero) * (u3 - ut[r + 3]);
        }
        let wlo = mlo[i];
        let whi = mhi[i];
        let sa = si_inv[i] * (one - F::cast_from((whi >> 31u32) & 1u32));
        let kk = ik * nth + k0;
        out[base] = finish_forward::<F>(u0, st0, dtk[kk], sa, mask_bit::<F>(wlo, whi, k0), sub < iter[kk], clip);
        out[base + 1] =
            finish_forward::<F>(u1, st1, dtk[kk + 1], sa, mask_bit::<F>(wlo, whi, k0 + 1), sub < iter[kk + 1], clip);
        out[base + 2] =
            finish_forward::<F>(u2, st2, dtk[kk + 2], sa, mask_bit::<F>(wlo, whi, k0 + 2), sub < iter[kk + 2], clip);
        out[base + 3] =
            finish_forward::<F>(u3, st3, dtk[kk + 3], sa, mask_bit::<F>(wlo, whi, k0 + 3), sub < iter[kk + 3], clip);
    }
}

/// Phase 1 of the transposed sub-step, four directions per unit.
#[cube(launch_unchecked)]
pub fn nscheme_tri_transpose_x4<F: Float>(
    y: &[F],
    gt: &mut [F],
    coef: &[F],
    ecos: &[F],
    esin: &[F],
    dtk: &[F],
    iter: &[u32],
    si_inv: &[F],
    mlo: &[u32],
    mhi: &[u32],
    ine: &[u32],
    ne: usize,
    ik: usize,
    sub: u32,
    #[comptime] nth: usize,
) {
    let g = ABSOLUTE_POS;
    let nw = nth / 4;
    if g < ne * nw {
        let t = g / nw;
        let k0 = (g - t * nw) * 4;
        let zero = F::new(0.0f32);
        let one = F::new(1.0f32);
        let thr = F::new(1.1754944e-38f32);
        let b = (ik * ne + t) * 12;
        let a1 = coef[b];
        let a2 = coef[b + 1];
        let a3 = coef[b + 2];
        let b1 = coef[b + 3];
        let b2 = coef[b + 4];
        let b3 = coef[b + 5];
        let n1 = ine[3 * t] as usize;
        let n2 = ine[3 * t + 1] as usize;
        let n3 = ine[3 * t + 2] as usize;
        let l1 = mlo[n1];
        let h1 = mhi[n1];
        let l2 = mlo[n2];
        let h2 = mhi[n2];
        let l3 = mlo[n3];
        let h3 = mhi[n3];
        let w1 = (one - F::cast_from((h1 >> 31u32) & 1u32)) * si_inv[n1];
        let w2 = (one - F::cast_from((h2 >> 31u32) & 1u32)) * si_inv[n2];
        let w3 = (one - F::cast_from((h3 >> 31u32) & 1u32)) * si_inv[n3];
        for d in 0..4usize {
            let k = k0 + d;
            let kk = ik * nth + k;
            if sub < iter[kk] {
                let c = ecos[k];
                let s = esin[k];
                let k1 = a1 * c + b1 * s;
                let k2 = a2 * c + b2 * s;
                let k3 = a3 * c + b3 * s;
                let kn = min(k1, zero) + min(k2, zero) + min(k3, zero);
                let nm = -one / min(-thr, kn);
                let v1 = mask_bit::<F>(l1, h1, k) * w1 * y[n1 * nth + k];
                let v2 = mask_bit::<F>(l2, h2, k) * w2 * y[n2 * nth + k];
                let v3 = mask_bit::<F>(l3, h3, k) * w3 * y[n3 * nth + k];
                gt[t * nth + k] = nm * dtk[kk] * (max(k1, zero) * v1 + max(k2, zero) * v2 + max(k3, zero) * v3);
            }
        }
    }
}

/// Phase 2 of the transposed sub-step, four directions per unit.
#[cube(launch_unchecked)]
pub fn nscheme_node_transpose_x4<F: Float>(
    y: &[F],
    out: &mut [F],
    gt: &[F],
    coef: &[F],
    ecos: &[F],
    esin: &[F],
    dtk: &[F],
    iter: &[u32],
    si_inv: &[F],
    mlo: &[u32],
    mhi: &[u32],
    start: &[u32],
    slot: &[u32],
    np: usize,
    ne: usize,
    ik: usize,
    sub: u32,
    #[comptime] nth: usize,
) {
    let g = ABSOLUTE_POS;
    let nw = nth / 4;
    if g < np * nw {
        let j = g / nw;
        let k0 = (g - j * nw) * 4;
        let base = j * nth + k0;
        let zero = F::new(0.0f32);
        let one = F::new(1.0f32);
        let c0 = ecos[k0];
        let c1 = ecos[k0 + 1];
        let c2 = ecos[k0 + 2];
        let c3 = ecos[k0 + 3];
        let s0 = esin[k0];
        let s1 = esin[k0 + 1];
        let s2 = esin[k0 + 2];
        let s3 = esin[k0 + 3];
        let y0 = y[base];
        let y1 = y[base + 1];
        let y2 = y[base + 2];
        let y3 = y[base + 3];
        let wlo = mlo[j];
        let whi = mhi[j];
        let sj = si_inv[j] * (one - F::cast_from((whi >> 31u32) & 1u32));
        let kk = ik * nth + k0;
        let m0 = mask_bit::<F>(wlo, whi, k0);
        let m1 = mask_bit::<F>(wlo, whi, k0 + 1);
        let m2 = mask_bit::<F>(wlo, whi, k0 + 2);
        let m3 = mask_bit::<F>(wlo, whi, k0 + 3);
        let v0 = m0 * sj * dtk[kk] * y0;
        let v1 = m1 * sj * dtk[kk + 1] * y1;
        let v2 = m2 * sj * dtk[kk + 2] * y2;
        let v3 = m3 * sj * dtk[kk + 3] * y3;
        let mut acc0 = zero;
        let mut acc1 = zero;
        let mut acc2 = zero;
        let mut acc3 = zero;
        let cb = ik * ne * 12;
        let q0 = start[j] as usize;
        let q1 = start[j + 1] as usize;
        for q in q0..q1 {
            let code = slot[q] as usize;
            let t = code / 3;
            let p = code - 3 * t;
            let b = cb + t * 12 + p;
            let a = coef[b];
            let bb = coef[b + 3];
            let ca = coef[b + 6];
            let cbb = coef[b + 9];
            let r = t * nth + k0;
            let kp0 = max(a * c0 + bb * s0, zero);
            let kp1 = max(a * c1 + bb * s1, zero);
            let kp2 = max(a * c2 + bb * s2, zero);
            let kp3 = max(a * c3 + bb * s3, zero);
            acc0 += kp0 * v0 - (ca * c0 + cbb * s0 + kp0) * gt[r];
            acc1 += kp1 * v1 - (ca * c1 + cbb * s1 + kp1) * gt[r + 1];
            acc2 += kp2 * v2 - (ca * c2 + cbb * s2 + kp2) * gt[r + 2];
            acc3 += kp3 * v3 - (ca * c3 + cbb * s3 + kp3) * gt[r + 3];
        }
        out[base] = select(sub < iter[kk], m0 * y0 - acc0, y0);
        out[base + 1] = select(sub < iter[kk + 1], m1 * y1 - acc1, y1);
        out[base + 2] = select(sub < iter[kk + 2], m2 * y2 - acc2, y2);
        out[base + 3] = select(sub < iter[kk + 3], m3 * y3 - acc3, y3);
    }
}
