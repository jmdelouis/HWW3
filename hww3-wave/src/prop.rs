//! CubeCL kernels of the propagation-only model (milestone M1): the WW3
//! PDLIB explicit geographic propagation with currents and open-boundary
//! spectra (`PDLIB_W3XYPUG`, `PDLIB_W3XYPFSN2`), its sub-cycling count, the
//! spectral propagation of `W3KTP3` (refraction and frequency shift with the
//! ULTIMATE QUICKEST schemes `W3QCK1` and `W3QCK2`), the triangle gradients
//! of `UG_GRADIENTS`, the time interpolation of the currents (`W3UCUR`), the
//! boundary values (`APPLY_BOUNDARY_CONDITION_VA`) and the outputs H_s and
//! T_m02 of `W3OUTG`.
//!
//! Layouts:
//! - action spectra `va[(ik np + i) nth + ith]` (frequency-major, so one
//!   frequency is the contiguous `[np, nth]` state of the M0 kernels);
//! - per-node, per-frequency arrays `wn`, `cg` as `[(ik) np + i]` for
//!   `ik = 0 .. NK+1` (WW3 index, ghosts included);
//! - node flags `nflag`: bit 0 `IOBP = 1`, bit 1 `IOBDP = 1`, bit 2
//!   `MAPSTA = 1` (sea point, neither ice nor dry), bit 3 `IOBP > 0`, bit 4
//!   `MAPSTA > 0` (H_s is output).
//!
//! The geographic advection velocity of component (f, θ) at node i is
//! `C = (gx cos θ + ux, gy sin θ + uy)` with `gx = FACX cg / CLATS IOBDP`,
//! `gy = FACX cg IOBDP` per frequency and the current part
//! `ux = FACX CX / CLATS [IOBP > 0] IOBDP`, `uy = FACX CY [IOBP > 0] IOBDP`.
//! `K_q` and the Simpson part of `FLALL_q` are then affine in (cos θ, sin θ):
//! twelve coefficients per triangle and frequency (`coef`, as in M0) plus six
//! per triangle for the current (`ccoef`: `d_q` of `K_q`, then `D_q` of
//! `FLALL_q`).

// The `#[cube]` macro generates, for every kernel, a launch struct and
// functions that carry no documentation; the kernels themselves are
// documented below.
#![allow(missing_docs)]

use cubecl::prelude::*;

/// Node flag: `IOBP = 1` (interior node).
pub const FLAG_INTERIOR: u32 = 1;
/// Node flag: `IOBDP = 1` (wet).
pub const FLAG_WET: u32 = 2;
/// Node flag: `MAPSTA = 1` (sea point, neither ice nor dry).
pub const FLAG_SEA: u32 = 4;
/// Node flag: `IOBP > 0` (not on the coast; currents are applied).
pub const FLAG_NOTCOAST: u32 = 8;
/// Node flag: `MAPSTA > 0` (H_s is output).
pub const FLAG_OUTPUT: u32 = 16;

// ---------------------------------------------------------------------------
// Velocities and coefficients
// ---------------------------------------------------------------------------

/// `gx`, `gy` for all frequencies: `gx[ik np + i] = facx cg(ik+1) / clats IOBDP`.
#[cube(launch_unchecked)]
pub fn group_velocity<F: Float>(cg: &[F], clats: &[F], nflag: &[u32], gx: &mut [F], gy: &mut [F], facx_: f32, np: usize, nk: usize) {
    let facx = F::cast_from(facx_);
    let g = ABSOLUTE_POS;
    if g < np * nk {
        let ik = g / np;
        let i = g - ik * np;
        let wet = F::cast_from((nflag[i] >> 1u32) & 1u32);
        let c = cg[(ik + 1) * np + i] * facx * wet;
        gx[g] = c / clats[i];
        gy[g] = c;
    }
}

/// Twelve coefficients per triangle for frequency `ik` (as M0 `nscheme_coef`).
#[cube(launch_unchecked)]
pub fn tri_coef<F: Float>(gx: &[F], gy: &[F], ine: &[u32], ien: &[F], coef: &mut [F], np: usize, ne: usize, ik: usize) {
    let t = ABSOLUTE_POS;
    if t < ne {
        let two = F::new(2.0f32);
        let sixth = F::new(1.0f32) / F::new(6.0f32);
        let off = ik * np;
        let n1 = ine[3 * t] as usize;
        let n2 = ine[3 * t + 1] as usize;
        let n3 = ine[3 * t + 2] as usize;
        let x1 = gx[off + n1];
        let x2 = gx[off + n2];
        let x3 = gx[off + n3];
        let y1 = gy[off + n1];
        let y2 = gy[off + n2];
        let y3 = gy[off + n3];
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

/// Currents of the step (`W3UCUR`: magnitude and direction interpolated in
/// time), and their part `ux`, `uy` of the advection velocity.
#[cube(launch_unchecked)]
pub fn current_interp<F: Float>(
    ca0: &[F],
    cai: &[F],
    cd0: &[F],
    cdi: &[F],
    clats: &[F],
    nflag: &[u32],
    cx: &mut [F],
    cy: &mut [F],
    ux: &mut [F],
    uy: &mut [F],
    rd_: f32,
    facx_: f32,
    np: usize,
) {
    let rd = F::cast_from(rd_);
    let facx = F::cast_from(facx_);
    let i = ABSOLUTE_POS;
    if i < np {
        let cabs = ca0[i] + rd * cai[i];
        let cdir = cd0[i] + rd * cdi[i];
        let x = cabs * cdir.cos();
        let y = cabs * cdir.sin();
        cx[i] = x;
        cy[i] = y;
        let f = nflag[i];
        let on = F::cast_from(((f >> 3u32) & 1u32) * ((f >> 1u32) & 1u32));
        ux[i] = facx * x / clats[i] * on;
        uy[i] = facx * y * on;
    }
}

/// Six current coefficients per triangle: `d_q = n_q · Σ u / 6` and the
/// Simpson part `D_q` of `FLALL_q`.
#[cube(launch_unchecked)]
pub fn cur_coef<F: Float>(ux: &[F], uy: &[F], ine: &[u32], ien: &[F], ccoef: &mut [F], ne: usize) {
    let t = ABSOLUTE_POS;
    if t < ne {
        let two = F::new(2.0f32);
        let sixth = F::new(1.0f32) / F::new(6.0f32);
        let n1 = ine[3 * t] as usize;
        let n2 = ine[3 * t + 1] as usize;
        let n3 = ine[3 * t + 2] as usize;
        let e1x = ien[6 * t];
        let e1y = ien[6 * t + 1];
        let e2x = ien[6 * t + 2];
        let e2y = ien[6 * t + 3];
        let e3x = ien[6 * t + 4];
        let e3y = ien[6 * t + 5];
        let sx = (ux[n1] + ux[n2] + ux[n3]) * sixth;
        let sy = (uy[n1] + uy[n2] + uy[n3]) * sixth;
        // C_j · n_q
        let c1n2 = ux[n1] * e2x + uy[n1] * e2y;
        let c1n3 = ux[n1] * e3x + uy[n1] * e3y;
        let c2n1 = ux[n2] * e1x + uy[n2] * e1y;
        let c2n3 = ux[n2] * e3x + uy[n2] * e3y;
        let c3n1 = ux[n3] * e1x + uy[n3] * e1y;
        let c3n2 = ux[n3] * e2x + uy[n3] * e2y;
        let b = t * 6;
        ccoef[b] = sx * e1x + sy * e1y;
        ccoef[b + 1] = sx * e2x + sy * e2y;
        ccoef[b + 2] = sx * e3x + sy * e3y;
        // FLALL1 = (2 FL31 + FL32 + 2 FL22 + FL21)/6: FL31 = C1.n3, FL32 = C2.n3, FL22 = C1.n2, FL21 = C3.n2
        ccoef[b + 3] = (two * c1n3 + c2n3 + two * c1n2 + c3n2) * sixth;
        // FLALL2 = (2 FL11 + FL12 + 2 FL32 + FL31)/6: FL11 = C2.n1, FL12 = C3.n1
        ccoef[b + 4] = (two * c2n1 + c3n1 + two * c2n3 + c1n3) * sixth;
        // FLALL3 = (2 FL21 + FL22 + 2 FL12 + FL11)/6
        ccoef[b + 5] = (two * c3n2 + c1n2 + two * c3n1 + c2n1) * sixth;
    }
}

// ---------------------------------------------------------------------------
// Sub-cycling count of PDLIB_W3XYPFSN2 (LCALC)
// ---------------------------------------------------------------------------

/// Partial minima of `SI / max(1e-9, KKSUM IOBDP)` over the nodes whose flags
/// contain `need` (`FLAG_WET`: every wet node, as WW3 with `UGBCCFL = T`, its
/// default; `FLAG_INTERIOR | FLAG_WET`: the interior nodes only), for
/// every frequency in one launch: unit `g = (ik nth/4 + grp) lanes + lane`
/// handles frequency `ik`, directions `4 grp .. 4 grp + 3` and nodes
/// `lane, lane + lanes, ...`.
#[cube(launch_unchecked)]
pub fn iter_partial<F: Float>(
    coef: &[F],
    ccoef: &[F],
    ecos: &[F],
    esin: &[F],
    si: &[F],
    nflag: &[u32],
    start: &[u32],
    slot: &[u32],
    partial: &mut [F],
    need: u32,
    np: usize,
    ne: usize,
    nk: usize,
    lanes: usize,
    #[comptime] nth: usize,
) {
    let g = ABSOLUTE_POS;
    let nw = nth / 4;
    if g < nk * nw * lanes {
        let q = g / lanes;
        let lane = g - q * lanes;
        let ik = q / nw;
        let grp = q - ik * nw;
        let k0 = grp * 4;
        let zero = F::new(0.0f32);
        let tiny = F::new(1.0e-9f32);
        let big = F::new(1.0e11f32);
        let c0 = ecos[k0];
        let c1 = ecos[k0 + 1];
        let c2 = ecos[k0 + 2];
        let c3 = ecos[k0 + 3];
        let s0 = esin[k0];
        let s1 = esin[k0 + 1];
        let s2 = esin[k0 + 2];
        let s3 = esin[k0 + 3];
        let mut m0 = big;
        let mut m1 = big;
        let mut m2 = big;
        let mut m3 = big;
        let cb = ik * ne * 12;
        let mut i = lane;
        while i < np {
            if (nflag[i] & need) == need {
                let mut k0s = zero;
                let mut k1s = zero;
                let mut k2s = zero;
                let mut k3s = zero;
                let q0 = start[i] as usize;
                let q1 = start[i + 1] as usize;
                for q in q0..q1 {
                    let code = slot[q] as usize;
                    let t = code / 3;
                    let p = code - 3 * t;
                    let a = coef[cb + t * 12 + p];
                    let bb = coef[cb + t * 12 + 3 + p];
                    let d = ccoef[t * 6 + p];
                    k0s += max(a * c0 + bb * s0 + d, zero);
                    k1s += max(a * c1 + bb * s1 + d, zero);
                    k2s += max(a * c2 + bb * s2 + d, zero);
                    k3s += max(a * c3 + bb * s3 + d, zero);
                }
                let s = si[i];
                m0 = min(m0, s / max(tiny, k0s));
                m1 = min(m1, s / max(tiny, k1s));
                m2 = min(m2, s / max(tiny, k2s));
                m3 = min(m3, s / max(tiny, k3s));
            }
            i += lanes;
        }
        let base = (ik * nth + k0) * lanes + lane;
        partial[base] = m0;
        partial[base + lanes] = m1;
        partial[base + 2 * lanes] = m2;
        partial[base + 3 * lanes] = m3;
    }
}

/// `ITER(ik, ith)` and `DT / ITER` from the partial minima (WW3's rounding of `DT / DTMAX`).
#[cube(launch_unchecked)]
pub fn iter_finish<F: Float>(partial: &[F], iter: &mut [u32], dtk: &mut [F], dt_: f32, lanes: usize, ncomp: usize) {
    let dt = F::cast_from(dt_);
    let c = ABSOLUTE_POS;
    if c < ncomp {
        let mut m = F::new(1.0e11f32);
        for l in 0..lanes {
            m = min(m, partial[c * lanes + l]);
        }
        let cfl = dt / m;
        let fl = cfl.floor();
        let rest = cfl - fl;
        let thr = F::new(1.1754944e-38f32);
        let half = F::new(0.5f32);
        // NINT(cfl) + 1 if 0 < rest < 0.5, NINT(cfl) otherwise: ceil, except exact integers
        let nint = select(rest >= half, fl + F::new(1.0f32), fl);
        let it = select(rest < thr, nint, select(rest < half, nint + F::new(1.0f32), nint));
        let itu = max(u32::cast_from(it), 1u32);
        iter[c] = itu;
        dtk[c] = dt / F::cast_from(itu);
    }
}

// ---------------------------------------------------------------------------
// Geographic sub-step with currents and open boundary
// ---------------------------------------------------------------------------

#[cube]
pub(crate) fn mask_bit<F: Float>(wlo: u32, whi: u32, k: usize) -> F {
    let w = select(k < 32, wlo, whi);
    F::cast_from((w >> ((k % 32) as u32)) & 1u32)
}

/// Phase 1: `ut[t nth + k] = Ũ_T` for four directions per unit.
#[cube(launch_unchecked)]
pub fn geo_tri_x4<F: Float>(
    u: &[F],
    ut: &mut [F],
    coef: &[F],
    ccoef: &[F],
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
        let d1 = ccoef[t * 6];
        let d2 = ccoef[t * 6 + 1];
        let d3 = ccoef[t * 6 + 2];
        let e1 = ccoef[t * 6 + 3];
        let e2 = ccoef[t * 6 + 4];
        let e3 = ccoef[t * 6 + 5];
        let r1 = ine[3 * t] as usize * nth;
        let r2 = ine[3 * t + 1] as usize * nth;
        let r3 = ine[3 * t + 2] as usize * nth;
        for d in 0..4usize {
            let k = k0 + d;
            if sub < iter[ik * nth + k] {
                let c = ecos[k];
                let s = esin[k];
                let k1 = a1 * c + b1 * s + d1;
                let k2 = a2 * c + b2 * s + d2;
                let k3 = a3 * c + b3 * s + d3;
                let kn = min(k1, zero) + min(k2, zero) + min(k3, zero);
                let nm = -one / min(-thr, kn);
                let fa1 = ca1 * c + cb1 * s + e1 + max(k1, zero);
                let fa2 = ca2 * c + cb2 * s + e2 + max(k2, zero);
                let fa3 = ca3 * c + cb3 * s + e3 + max(k3, zero);
                ut[t * nth + k] = nm * (fa1 * u[r1 + k] + fa2 * u[r2 + k] + fa3 * u[r3 + k]);
            }
        }
    }
}

#[cube]
pub(crate) fn node_value<F: Float>(
    ui: F,
    st: F,
    dt: F,
    sa: F,
    m: F,
    active: bool,
    bnd: bool,
    b0: F,
    bn: F,
    clat: F,
    it: u32,
    sub: u32,
    rd10: F,
    rd20: F,
    dtg: F,
) -> F {
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let v = max(ui - dt * sa * st, zero) * m;
    // boundary value at the end of this sub-step (PDLIB_W3XYPFSN2, FLBPI)
    let itf = F::cast_from(it);
    let rd1 = rd10 - dtg * (itf - F::cast_from(sub + 1u32)) / itf;
    let r = min(one, max(zero, rd1 / rd20));
    let rd2 = select(rd20 > F::new(0.001f32), r, one);
    let rd1b = one - rd2;
    let bv = (rd1b * b0 + rd2 * bn) * clat;
    select(active, select(bnd, bv, v), ui)
}

/// Phase 2: gather `Σ_T K+_iT (U_i - Ũ_T)` and update; open-boundary nodes
/// take the boundary spectrum interpolated at the end of the sub-step.
#[cube(launch_unchecked)]
pub fn geo_node_x4<F: Float>(
    u: &[F],
    out: &mut [F],
    ut: &[F],
    coef: &[F],
    ccoef: &[F],
    ecos: &[F],
    esin: &[F],
    dtk: &[F],
    iter: &[u32],
    si_inv: &[F],
    mlo: &[u32],
    mhi: &[u32],
    start: &[u32],
    slot: &[u32],
    bidx: &[u32],
    bb0: &[F],
    bbn: &[F],
    clats: &[F],
    rd10_: f32,
    rd20_: f32,
    dtg_: f32,
    np: usize,
    ne: usize,
    nspec: usize,
    ik: usize,
    sub: u32,
    #[comptime] nth: usize,
) {
    let rd10 = F::cast_from(rd10_);
    let rd20 = F::cast_from(rd20_);
    let dtg = F::cast_from(dtg_);
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
            let d = ccoef[t * 6 + p];
            let r = t * nth + k0;
            st0 += max(a * c0 + bb * s0 + d, zero) * (u0 - ut[r]);
            st1 += max(a * c1 + bb * s1 + d, zero) * (u1 - ut[r + 1]);
            st2 += max(a * c2 + bb * s2 + d, zero) * (u2 - ut[r + 2]);
            st3 += max(a * c3 + bb * s3 + d, zero) * (u3 - ut[r + 3]);
        }
        let wlo = mlo[i];
        let whi = mhi[i];
        let sa = si_inv[i] * (one - F::cast_from((whi >> 31u32) & 1u32));
        let kk = ik * nth + k0;
        let bi = bidx[i];
        let bnd = bi != 0xffff_ffffu32;
        let bsafe = select(bnd, bi as usize, 0usize);
        let boff = bsafe * nspec + kk;
        let clat = clats[i];
        out[base] = node_value::<F>(
            u0, st0, dtk[kk], sa, mask_bit::<F>(wlo, whi, k0), sub < iter[kk], bnd,
            bb0[boff], bbn[boff], clat, iter[kk], sub, rd10, rd20, dtg,
        );
        out[base + 1] = node_value::<F>(
            u1, st1, dtk[kk + 1], sa, mask_bit::<F>(wlo, whi, k0 + 1), sub < iter[kk + 1], bnd,
            bb0[boff + 1], bbn[boff + 1], clat, iter[kk + 1], sub, rd10, rd20, dtg,
        );
        out[base + 2] = node_value::<F>(
            u2, st2, dtk[kk + 2], sa, mask_bit::<F>(wlo, whi, k0 + 2), sub < iter[kk + 2], bnd,
            bb0[boff + 2], bbn[boff + 2], clat, iter[kk + 2], sub, rd10, rd20, dtg,
        );
        out[base + 3] = node_value::<F>(
            u3, st3, dtk[kk + 3], sa, mask_bit::<F>(wlo, whi, k0 + 3), sub < iter[kk + 3], bnd,
            bb0[boff + 3], bbn[boff + 3], clat, iter[kk + 3], sub, rd10, rd20, dtg,
        );
    }
}

// ---------------------------------------------------------------------------
// Conversions between VA and the propagated AC = VA / CG CLATS
// ---------------------------------------------------------------------------

/// `AC = VA CLATS / CG` for frequency `ik` (the quantity `PDLIB_W3XYPUG`
/// propagates).
#[cube(launch_unchecked)]
pub fn va_to_ac<F: Float>(va: &[F], ac: &mut [F], cg: &[F], clats: &[F], np: usize, ik: usize, #[comptime] nth: usize) {
    let g = ABSOLUTE_POS;
    if g < np * nth {
        let i = g / nth;
        ac[g] = va[ik * np * nth + g] / cg[(ik + 1) * np + i] * clats[i];
    }
}

/// `VA = max(0, CG / CLATS AC)` for frequency `ik`.
#[cube(launch_unchecked)]
pub fn ac_to_va<F: Float>(ac: &[F], va: &mut [F], cg: &[F], clats: &[F], np: usize, ik: usize, #[comptime] nth: usize) {
    let g = ABSOLUTE_POS;
    if g < np * nth {
        let i = g / nth;
        va[ik * np * nth + g] = max(F::new(0.0f32), cg[(ik + 1) * np + i] / clats[i] * ac[g]);
    }
}

/// `APPLY_BOUNDARY_CONDITION_VA`: `VA = max(0, CG (RD1 BBPI0 + RD2 BBPIN))`
/// at the boundary points, one unit per (point, component).
#[cube(launch_unchecked)]
pub fn apply_boundary<F: Float>(
    va: &mut [F],
    bnode: &[u32],
    bb0: &[F],
    bbn: &[F],
    cg: &[F],
    rd1_: f32,
    rd2_: f32,
    np: usize,
    nbnd: usize,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
) {
    let rd1 = F::cast_from(rd1_);
    let rd2 = F::cast_from(rd2_);
    let g = ABSOLUTE_POS;
    let nspec = nk * nth;
    if g < nbnd * nspec {
        let b = g / nspec;
        let isp = g - b * nspec;
        let ik = isp / nth;
        let ith = isp - ik * nth;
        let i = bnode[b] as usize;
        let v = cg[(ik + 1) * np + i] * (rd1 * bb0[g] + rd2 * bbn[g]);
        va[(ik * np + i) * nth + ith] = max(F::new(0.0f32), v);
    }
}

/// `UG_GRADIENTS` as a gather: area-weighted mean over the triangles around
/// each node of the gradient of the linear interpolant, in units per metre
/// (`fact = 1 / (DERA RADIUS)`, x scaled by the triangle mean of `1 / CLATS`).
#[cube(launch_unchecked)]
pub fn gradient<F: Float>(
    param: &[F],
    dx: &mut [F],
    dy: &mut [F],
    ine: &[u32],
    ien: &[F],
    tria: &[F],
    clatis: &[F],
    start: &[u32],
    slot: &[u32],
    fact_: f32,
    np: usize,
) {
    let fact = F::cast_from(fact_);
    let i = ABSOLUTE_POS;
    if i < np {
        let zero = F::new(0.0f32);
        let two = F::new(2.0f32);
        let third = F::new(1.0f32) / F::new(3.0f32);
        let mut sx = zero;
        let mut sy = zero;
        let mut w = zero;
        let q0 = start[i] as usize;
        let q1 = start[i + 1] as usize;
        for q in q0..q1 {
            let t = slot[q] as usize / 3;
            let n1 = ine[3 * t] as usize;
            let n2 = ine[3 * t + 1] as usize;
            let n3 = ine[3 * t + 2] as usize;
            let v1 = param[n1] * fact;
            let v2 = param[n2] * fact;
            let v3 = param[n3] * fact;
            let lat = third * (clatis[n1] + clatis[n2] + clatis[n3]);
            let gx = v1 * ien[6 * t] + v2 * ien[6 * t + 2] + v3 * ien[6 * t + 4];
            let gy = v1 * ien[6 * t + 1] + v2 * ien[6 * t + 3] + v3 * ien[6 * t + 5];
            sx += gx * lat;
            sy += gy;
            w += two * tria[t];
        }
        dx[i] = sx / w;
        dy[i] = sy / w;
    }
}

// ---------------------------------------------------------------------------
// W3KTP3: refraction (theta) and frequency shift (k)
// ---------------------------------------------------------------------------

/// `DSDD(IK)` of `W3KTP3`.
#[cube]
pub(crate) fn dsdd<F: Float>(wn: F, cg: F, sig: F, depth: F) -> F {
    let v = max(F::new(0.0f32), cg * wn - F::new(0.5f32) * sig) / depth;
    select(depth * wn < F::new(5.0f32), v, F::new(0.0f32))
}

/// Limited face value of ULTIMATE QUICKEST (`W3QCK1` / `W3QCK2`, after `QB`):
/// `qc`, `qu`, `qd` the central, upstream and downstream values.
#[cube]
pub(crate) fn uq_limit<F: Float>(qb: F, qc: F, qu: F, qd: F, cfl: F) -> F {
    let one = F::new(1.0f32);
    let dq = qd - qu;
    let m = max(F::new(1.0e-15f32), dq.abs());
    let dqnz = select(dq >= F::new(0.0f32), m, -m);
    let qcn = min(F::new(1.1f32), max(F::new(-0.1f32), (qc - qu) / dqnz));
    let qbn0 = max((qb - qu) / dqnz, qcn);
    let qbn = min(min(qbn0, one), qcn / max(F::new(1.0e-10f32), cfl.abs()));
    let qbr = qu + qbn * dq;
    let cfac = select(F::new(2.0f32) * (qcn - F::new(0.5f32)).abs() >= one, one, F::new(0.0f32));
    (one - cfac) * qbr + cfac * qc
}

/// Clipped refraction CFL number of one direction (`W3KTP3`, `CTMAX`).
#[cube]
pub(crate) fn theta_cfl<F: Float>(
    es2: F,
    esc: F,
    ec2: F,
    ec: F,
    es: F,
    dcyx: F,
    dcxxyy: F,
    dcxy: F,
    frg: F,
    frk: F,
    gx: F,
    gy: F,
    ctmax: F,
) -> F {
    let vc = es2 * dcyx + esc * dcxxyy - ec2 * dcxy;
    let v = vc + frg * ec + frk * (es * gx - ec * gy);
    let lim = min(v.abs(), ctmax);
    select(v >= F::new(0.0f32), lim, -lim)
}

/// ULTIMATE QUICKEST flux through the interface between directions `t` and
/// `t + 1` (`W3QCK1`), from the CFL numbers of both cells and the values of
/// cells `t - 1 .. t + 2`.
#[cube]
pub(crate) fn theta_flux<F: Float>(ct: F, ctp: F, qtm: F, qt: F, qtp: F, qtpp: F) -> F {
    let half = F::new(0.5f32);
    let one = F::new(1.0f32);
    let sixth = one / F::new(6.0f32);
    let cf = half * (ct + ctp);
    // central cell: t for cf >= 0, t + 1 otherwise
    let up = cf < F::new(0.0f32);
    let qc = select(up, qtp, qt);
    let qcm = select(up, qt, qtm);
    let qcp = select(up, qtpp, qtp);
    let qb = half * ((one - cf) * qtp + (one + cf) * qt) - (one - cf * cf) * sixth * (qcm - F::new(2.0f32) * qc + qcp);
    let qu = select(up, qcp, qcm);
    let qd = select(up, qcm, qcp);
    cf * uq_limit::<F>(qb, qc, qu, qd, cf)
}

//// Refraction: one unit per (node, frequency), periodic ULTIMATE QUICKEST in
/// theta (`W3QCK1`, CFL numbers of `W3KTP3` limited to `CTMAX`). The rows
/// of the cube (`block` consecutive (frequency, node) rows of `nth` values,
/// contiguous in `va`) are staged in shared memory with coalesced loads and
/// stores, with rows padded to `nth + 1` values (no bank conflicts). Each
/// unit then sweeps its row once with a window of four values and two CFL
/// numbers in registers (no local arrays), keeping the first two values for
/// the periodic wrap. Launch with `block` units per cube.
#[cube(launch_unchecked)]
pub fn ktp3_theta<F: Float>(
    va: &mut [F],
    wn: &[F],
    cg: &[F],
    sig: &[F],
    dw: &[F],
    dddx: &[F],
    dddy: &[F],
    dcxdx: &[F],
    dcxdy: &[F],
    dcydx: &[F],
    dcydy: &[F],
    cthg0: &[F],
    nflag: &[u32],
    ecos: &[F],
    esin: &[F],
    es2: &[F],
    ec2: &[F],
    esc: &[F],
    facth_: f32,
    ctmax_: f32,
    flcur_: f32,
    dmin_: f32,
    np: usize,
    nk: usize,
    #[comptime] nth: usize,
    #[comptime] block: usize,
) {
    let facth = F::cast_from(facth_);
    let ctmax = F::cast_from(ctmax_);
    let flcur = F::cast_from(flcur_);
    let dmin = F::cast_from(dmin_);
    let rs = nth + 1;
    let mut sm = Shared::<[F]>::new_slice(block * (nth + 1));
    let t = UNIT_POS as usize;
    let g0 = CUBE_POS as usize * block;
    let total = np * nk * nth;
    let e0 = g0 * nth;
    for r in 0..nth {
        let e = r * block + t;
        let row = e / nth;
        let th = e - row * nth;
        if e0 + e < total {
            sm[row * rs + th] = va[e0 + e];
        }
    }
    sync_cube();
    let g = g0 + t;
    if g < np * nk {
        let ik = g / np;
        let i = g - ik * np;
        if (nflag[i] & 5u32) == 5u32 {
            let zero = F::new(0.0f32);
            let depth = max(dmin, dw[i]);
            let w = wn[(ik + 1) * np + i];
            let c = cg[(ik + 1) * np + i];
            let frk = facth * dsdd::<F>(w, c, sig[ik + 1], depth) / w;
            let frg = facth * cthg0[i] * c;
            let dcyx = facth * dcydx[i] * flcur;
            let dcxxyy = facth * (dcxdx[i] - dcydy[i]) * flcur;
            let dcxy = facth * dcxdy[i] * flcur;
            let gx = dddx[i];
            let gy = dddy[i];
            let base = t * rs;
            let l = nth - 1;
            let q0 = sm[base];
            let q1 = sm[base + 1];
            let c0 = theta_cfl::<F>(es2[0], esc[0], ec2[0], ecos[0], esin[0], dcyx, dcxxyy, dcxy, frg, frk, gx, gy, ctmax);
            let cl = theta_cfl::<F>(es2[l], esc[l], ec2[l], ecos[l], esin[l], dcyx, dcxxyy, dcxy, frg, frk, gx, gy, ctmax);
            let qlm = sm[base + l - 1];
            let ql = sm[base + l];
            // flux through the interface nth-1 | 0, needed by cell 0
            let mut fprev = theta_flux::<F>(cl, c0, qlm, ql, q0, q1);
            let mut qm = ql;
            let mut qc = q0;
            let mut qp = q1;
            let mut cc = c0;
            for th in 0..nth {
                let t2 = th + 2;
                let mut qpp = q1;
                if t2 < nth {
                    qpp = sm[base + t2];
                } else if t2 == nth {
                    qpp = q0;
                }
                let tp = (th + 1) % nth;
                let cp = theta_cfl::<F>(es2[tp], esc[tp], ec2[tp], ecos[tp], esin[tp], dcyx, dcxxyy, dcxy, frg, frk, gx, gy, ctmax);
                let fl = theta_flux::<F>(cc, cp, qm, qc, qp, qpp);
                sm[base + th] = max(zero, qc + fprev - fl);
                fprev = fl;
                qm = qc;
                qc = qp;
                qp = qpp;
                cc = cp;
            }
        }
    }
    sync_cube();
    for r in 0..nth {
        let e = r * block + t;
        let row = e / nth;
        let th = e - row * nth;
        if e0 + e < total {
            va[e0 + e] = sm[row * rs + th];
        }
    }
}

// Velocity of one cell of the frequency advection, clipped to `CTMAX`
/// (`W3KTP3`: `VELFAC = FACK / DB`), for `db = DSIP(j) / CG(j)`.
#[cube]
pub(crate) fn k_vel<F: Float>(w: F, c: F, s: F, db: F, depth: F, fkd: F, fkc: F, fack: F, ctmax: F) -> F {
    let d = dsdd::<F>(w, c, s, depth);
    let velfac = fack / db;
    let v = (fkd / c * d + w * fkc) * velfac;
    let lim = min(v.abs(), ctmax);
    select(v >= F::new(0.0f32), lim, -lim) / velfac
}

/// Frequency shift: one unit per (node, direction), ULTIMATE QUICKEST in k
/// on the variable wavenumber grid (`W3QCK2`), with the ghost bins `IK = 0`
/// (zero) and `NK+1` (`FACHFA` times bin `NK`). The frequencies are swept
/// once with a window of four values, two velocities and three wavenumber
/// steps held in registers (no local arrays); needs `nk >= 3`.
#[cube(launch_unchecked)]
pub fn ktp3_k<F: Float>(
    va: &mut [F],
    wn: &[F],
    cg: &[F],
    sig: &[F],
    dsip: &[F],
    dw: &[F],
    dddx: &[F],
    dddy: &[F],
    cx: &[F],
    cy: &[F],
    dcxdx: &[F],
    dcxdy: &[F],
    dcydx: &[F],
    dcydy: &[F],
    nflag: &[u32],
    es2: &[F],
    ec2: &[F],
    esc: &[F],
    fack_: f32,
    ctmax_: f32,
    fachfa_: f32,
    dmin_: f32,
    np: usize,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
) {
    let fack = F::cast_from(fack_);
    let ctmax = F::cast_from(ctmax_);
    let fachfa = F::cast_from(fachfa_);
    let dmin = F::cast_from(dmin_);
    let g = ABSOLUTE_POS;
    if g < np * nth {
        let i = g / nth;
        let th = g - i * nth;
        if (nflag[i] & 5u32) == 5u32 {
            let zero = F::new(0.0f32);
            let half = F::new(0.5f32);
            let one = F::new(1.0f32);
            let sixth = one / F::new(6.0f32);
            let depth = max(dmin, dw[i]);
            let fkd = cx[i] * dddx[i] + cy[i] * dddy[i];
            let fkc = ec2[th] * (-dcxdx[i]) + esc[th] * (-(dcxdy[i] + dcydx[i])) + es2[th] * (-dcydy[i]);
            // cell j of the spectrum (1 .. nk) is va[((j - 1) np + i) nth + th]
            let stride = np * nth;
            let c0 = i * nth + th;
            // window at interface m (cells m-1 .. m+2), starting at m = 1
            let w0 = wn[i];
            let w1 = wn[np + i];
            let w2 = wn[2 * np + i];
            let mut w3 = wn[3 * np + i];
            let cg1 = cg[np + i];
            let cg2 = cg[2 * np + i];
            let mut db_c = dsip[1] / cg1;
            let mut db_h = dsip[2] / cg2;
            let mut vel_c = k_vel::<F>(w1, cg1, sig[1], db_c, depth, fkd, fkc, fack, ctmax);
            let mut vel_h = k_vel::<F>(w2, cg2, sig[2], db_h, depth, fkd, fkc, fack, ctmax);
            let mut dm_l = w1 - w0;
            let mut dm_c = w2 - w1;
            let mut dm_h = w3 - w2;
            let mut q_l = zero;
            let mut q_c = va[c0];
            let mut q_h = va[c0 + stride];
            let mut q_hh = va[c0 + 2 * stride];
            // interface 0|1: first order, velocity of cell 1
            let mut fprev = vel_c * select(vel_c >= zero, zero, q_c);
            for m in 1..nk {
                // interface m|m+1: ULTIMATE QUICKEST
                let v = half * (vel_c + vel_h);
                let cf = fack * v / dm_c;
                let up = cf < zero;
                let qc = select(up, q_h, q_c);
                let qcm = select(up, q_c, q_l);
                let qcp = select(up, q_hh, q_h);
                let dbi = select(up, db_h, db_c);
                let dmi = select(up, dm_h, dm_c);
                let dmim = select(up, dm_c, dm_l);
                let qb = half * ((one - cf) * q_h + (one + cf) * q_c)
                    - dm_c * dm_c / dbi * (one - cf * cf) * sixth * ((qcp - qc) / dmi - (qc - qcm) / dmim);
                let qu = select(up, qcp, qcm);
                let qd = select(up, qcm, qcp);
                let fl = v * uq_limit::<F>(qb, qc, qu, qd, cf);
                va[c0 + (m - 1) * stride] = max(zero, q_c + fack / db_c * (fprev - fl));
                fprev = fl;
                // shift the window to interface m + 1
                let j3 = m + 3;
                let mut qn = zero;
                if j3 <= nk {
                    qn = va[c0 + (j3 - 1) * stride];
                } else if j3 == nk + 1 {
                    qn = fachfa * q_hh;
                }
                q_l = q_c;
                q_c = q_h;
                q_h = q_hh;
                q_hh = qn;
                let j2 = m + 2;
                let w2n = w3;
                let cgn = cg[j2 * np + i];
                let dbn = dsip[j2] / cgn;
                let veln = k_vel::<F>(w2n, cgn, sig[j2], dbn, depth, fkd, fkc, fack, ctmax);
                vel_c = vel_h;
                vel_h = veln;
                db_c = db_h;
                db_h = dbn;
                dm_l = dm_c;
                dm_c = dm_h;
                if j2 <= nk {
                    w3 = wn[(j2 + 1) * np + i];
                    dm_h = w3 - w2n;
                } else {
                    dm_h = zero;
                }
            }
            // interface nk|nk+1: first order, velocity of cell nk
            let fl = vel_c * select(vel_c >= zero, q_c, q_h);
            va[c0 + (nk - 1) * stride] = max(zero, q_c + fack / db_c * (fprev - fl));
        }
    }
}

/// Spectrum remapping of `W3ULEV` for the listed nodes, one unit per (node,
/// direction): `TA(k) = A(k) DSIP(k) / CG_old(k)` redistributed with the
/// weights computed on the host (`i1[k]` the lower target bin, 0 = below
/// bin 1, nk+1 = none; `r1[k]` its weight, `1 - r1[k]` to `i1[k] + 1`), then
/// divided by `DSIP / CG_new` and completed by the tail from bin `tail`
/// (1-based, `nk + 1` = no tail).
#[cube(launch_unchecked)]
pub fn ulev_remap<F: Float>(
    va: &mut [F],
    nodes: &[u32],
    own: &[F],
    dwn: &[F],
    i1: &[u32],
    r1: &[F],
    tail: &[u32],
    fachfa_: f32,
    np: usize,
    nlist: usize,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
) {
    let fachfa = F::cast_from(fachfa_);
    let g = ABSOLUTE_POS;
    if g < nlist * nth {
        let l = g / nth;
        let th = g - l * nth;
        let i = nodes[l] as usize;
        let zero = F::new(0.0f32);
        let one = F::new(1.0f32);
        let mut ta = Array::<F>::new(nk);
        let mut a = Array::<F>::new(nk);
        for k in 0..nk {
            ta[k] = va[(k * np + i) * nth + th] * own[l * nk + k];
            a[k] = zero;
        }
        for k in 0..nk {
            let j1 = i1[l * nk + k] as usize;
            if j1 <= nk {
                let w1 = r1[l * nk + k];
                if j1 >= 1 {
                    a[j1 - 1] += w1 * ta[k];
                }
                if j1 < nk {
                    a[j1] += (one - w1) * ta[k];
                }
            }
        }
        for k in 0..nk {
            a[k] = a[k] / dwn[l * nk + k];
        }
        let t0 = tail[l] as usize;
        for k in 1..nk {
            if k + 1 >= t0 {
                a[k] = fachfa * a[k - 1];
            }
        }
        for k in 0..nk {
            va[(k * np + i) * nth + th] = a[k];
        }
    }
}

//// `WAVNU3` on the device: `k d` of radian frequency `si` at depth `h`.
#[cube]
pub(crate) fn wavnu3_kh<F: Float>(si: F, h: F) -> F {
    let kdmax = F::new(20.0f32);
    let one = F::new(1.0f32);
    let zpi = F::new(6.2831855f32);
    let grav = F::new(9.806f32);
    let tp = si / zpi;
    let kh0 = zpi * zpi * h / grav * tp * tp;
    let tmp = F::new(1.55f32) + F::new(1.3f32) * kh0 + F::new(0.216f32) * kh0 * kh0;
    kh0 * (one + kh0.powf(F::new(1.09f32)) * one / min(tmp, kdmax).exp()) / min(kh0, kdmax).tanh().sqrt()
}

/// `W3ULEV` on the device, one unit per node, after the host has set the
/// new depths `dw` and the node modes: 0 nothing (deep node), 1 new
/// wavenumbers and group speeds only (dry, drying or wetting node, or
/// `MAPSTA < 0`), 2 the same and the remapping of the spectrum to the new
/// wavenumbers when the first bin moved by 5 % of its width. A remapped
/// node gets mode 3.
#[cube(launch_unchecked)]
pub fn ulev_node<F: Float>(
    va: &mut [F],
    wn: &mut [F],
    cg: &mut [F],
    dw: &[F],
    mode: &mut [u32],
    sig: &[F],
    dsip: &[F],
    dmin_: f32,
    fachfa_: f32,
    np: usize,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
) {
    let dmin = F::cast_from(dmin_);
    let fachfa = F::cast_from(fachfa_);
    let i = ABSOLUTE_POS;
    if i < np {
        let md = mode[i];
        if md != 0u32 {
            let zero = F::new(0.0f32);
            let one = F::new(1.0f32);
            let half = F::new(0.5f32);
            let two = F::new(2.0f32);
            let kdmax = F::new(20.0f32);
            let depth = max(dmin, dw[i]);
            let mut wno = Array::<F>::new(nk + 2);
            let mut cgo = Array::<F>::new(nk + 2);
            let mut wnn = Array::<F>::new(nk + 2);
            let mut cgn = Array::<F>::new(nk + 2);
            for j in 0..nk + 2 {
                wno[j] = wn[j * np + i];
                cgo[j] = cg[j * np + i];
                let s = sig[j];
                let kh = wavnu3_kh::<F>(s, depth);
                let k = kh / depth;
                let c = half * (one + (two * kh / min(two * kh, kdmax).sinh())) * s / k;
                wnn[j] = k;
                cgn[j] = c;
                wn[j * np + i] = k;
                cg[j * np + i] = c;
            }
            if md == 2u32 {
                let dwn1 = dsip[1] / cgn[1];
                let rdk = (wno[1] - wnn[1]).abs() / dwn1;
                if rdk >= F::new(0.05f32) {
                    // remapping weights of W3ULEV, 1-based bins
                    let mut wi1 = Array::<u32>::new(nk);
                    let mut wr1 = Array::<F>::new(nk);
                    for k in 0..nk {
                        wi1[k] = (nk + 1) as u32;
                        wr1[k] = zero;
                    }
                    let mut ik0 = 1usize;
                    let mut ii1 = 1usize;
                    let mut ii2 = 2usize;
                    let mut cond = true;
                    if wno[1] < wnn[1] {
                        ii1 = 0;
                        ii2 = 1;
                        let mut found = false;
                        for j in 2..nk + 2 {
                            if !found && wno[j] >= wnn[1] {
                                found = true;
                                ik0 = j - 1;
                            }
                        }
                        cond = found;
                    }
                    let mut rd2 = one;
                    if cond {
                        for ik in ik0..nk + 1 {
                            while ii1 <= nk && wno[ik] > wnn[ii2] {
                                ii1 += 1;
                                if ii1 <= nk {
                                    ii2 = ii1 + 1;
                                }
                            }
                            if ii1 <= nk {
                                let rd1 = select(
                                    ii1 == 0,
                                    (wnn[1] - wno[ik]) / dwn1,
                                    (wnn[ii2] - wno[ik]) / (wnn[ii2] - wnn[ii1]),
                                );
                                rd2 = one - rd1;
                                wi1[ik - 1] = ii1 as u32;
                                wr1[ik - 1] = rd1;
                            }
                        }
                    } else {
                        ii2 = nk + 1;
                    }
                    // nk + 1 (ii2 <= nk + 1), as a runtime value
                    let mut t0 = max(ii2, nk + 1);
                    if ii2 <= nk && rd2 <= F::new(0.95f32) {
                        t0 = max(ii2, 2);
                    }
                    let mut ta = Array::<F>::new(nk);
                    let mut a = Array::<F>::new(nk);
                    for th in 0..nth {
                        for k in 0..nk {
                            ta[k] = va[(k * np + i) * nth + th] * (dsip[k + 1] / cgo[k + 1]);
                            a[k] = zero;
                        }
                        for k in 0..nk {
                            let j1 = wi1[k] as usize;
                            if j1 <= nk {
                                let w1 = wr1[k];
                                if j1 >= 1 {
                                    a[j1 - 1] += w1 * ta[k];
                                }
                                if j1 < nk {
                                    a[j1] += (one - w1) * ta[k];
                                }
                            }
                        }
                        for k in 0..nk {
                            a[k] = a[k] / (dsip[k + 1] / cgn[k + 1]);
                        }
                        for k in 1..nk {
                            if k + 1 >= t0 {
                                a[k] = fachfa * a[k - 1];
                            }
                        }
                        for k in 0..nk {
                            va[(k * np + i) * nth + th] = a[k];
                        }
                    }
                    mode[i] = 3u32;
                }
            }
        }
    }
}

/// Zero the spectra of the listed nodes (ice or drying transitions).
#[cube(launch_unchecked)]
pub fn zero_nodes<F: Float>(va: &mut [F], nodes: &[u32], np: usize, nlist: usize, #[comptime] nk: usize, #[comptime] nth: usize) {
    let g = ABSOLUTE_POS;
    if g < nlist * nk * nth {
        let l = g / (nk * nth);
        let r = g - l * nk * nth;
        let k = r / nth;
        let th = r - k * nth;
        let i = nodes[l] as usize;
        va[(k * np + i) * nth + th] = F::new(0.0f32);
    }
}

// ---------------------------------------------------------------------------
// Outputs (W3OUTG): H_s and T_m02 with the tail
// ---------------------------------------------------------------------------

/// `W3OUTG`: H_s, T_m02 and the energy of every node, with the
/// high-frequency tail; `UNDEF` where `MAPSTA <= 0`.
#[cube(launch_unchecked)]
pub fn outputs<F: Float>(
    va: &[F],
    cg: &[F],
    sig: &[F],
    dden: &[F],
    nflag: &[u32],
    hs: &mut [F],
    t02: &mut [F],
    en: &mut [F],
    fte_: f32,
    dth_: f32,
    undef_: f32,
    np: usize,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
) {
    let fte = F::cast_from(fte_);
    let dth = F::cast_from(dth_);
    let undef = F::cast_from(undef_);
    let i = ABSOLUTE_POS;
    if i < np {
        let zero = F::new(0.0f32);
        let mut et = zero;
        let mut et02 = zero;
        let mut ab = zero;
        let mut c = F::new(1.0f32);
        for k in 0..nk {
            ab = zero;
            for th in 0..nth {
                ab += va[(k * np + i) * nth + th];
            }
            c = cg[(k + 1) * np + i];
            let ebd = ab * dden[k] / c;
            et += ebd;
            let s = sig[k + 1];
            et02 += ebd * s * s;
        }
        let eband = ab / c;
        let sn = sig[nk];
        // energy of the discrete spectrum, without the tail, for the budget
        let eraw = et;
        et += fte * eband;
        et02 += eband * F::new(0.5f32) * sn * sn * sn * sn * dth;
        let tpi = F::new(6.2831855f32);
        let out = (nflag[i] & 16u32) != 0u32;
        let ok = et02 > F::new(1.0e-7f32) && et > zero;
        let tm = select(ok, tpi * (et / max(et02, F::new(1.0e-30f32))).sqrt(), tpi / sn);
        hs[i] = select(out, F::new(4.0f32) * et.sqrt(), undef);
        t02[i] = select(out, tm, undef);
        en[i] = select(out, eraw, zero);
    }
}
