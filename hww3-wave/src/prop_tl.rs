//! Tangent-linear CubeCL kernels of the propagation and forcing (milestone
//! M3): with [`crate::srce_tl`], the model step of [`crate::model`]
//! differentiated in forward mode, for a perturbation of the source
//! parameters or of the wind.
//!
//! The forcing (depths, currents, ice, levels, boundary spectra) does not
//! depend on the parameters, so the propagation velocities, CFL numbers and
//! sub-cycling counts carry no tangent. The geographic N-scheme and the
//! remappings are linear in the spectrum and are applied to its tangent
//! with zero boundary spectra; the ULTIMATE QUICKEST face values of
//! `W3KTP3` are linear in the cell values once the limiter branch is chosen,
//! and their tangent follows the branch of the values. The clamps at zero
//! (`MAX(0, ...)`) give a zero tangent where they act. Each tangent kernel
//! updates the values exactly as the value kernel of [`crate::prop`] does.

// The `#[cube]` macro generates, for every kernel, a launch struct and
// functions that carry no documentation; the kernels themselves are
// documented below.
#![allow(missing_docs)]

use crate::prop::{dsdd, k_vel, mask_bit, node_value, theta_cfl, theta_flux, uq_limit, wavnu3_kh};
use cubecl::prelude::*;

/// Tangent of [`uq_limit`] along `(dqb, dqc, dqu, dqd)` (the CFL number
/// carries none).
#[cube]
#[allow(clippy::too_many_arguments)]
fn uq_limit_d<F: Float>(qb: F, qc: F, qu: F, qd: F, cfl: F, dqb: F, dqc: F, dqu: F, dqd: F) -> F {
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let eps = F::new(1.0e-15f32);
    let dq = qd - qu;
    let ddq = dqd - dqu;
    let m = max(eps, dq.abs());
    let dqnz = select(dq >= zero, m, -m);
    let ddqnz = select(dq.abs() > eps, ddq, zero);
    let r = (qc - qu) / dqnz;
    let dr = ((dqc - dqu) - r * ddqnz) / dqnz;
    let qcn = min(F::new(1.1f32), max(F::new(-0.1f32), r));
    let dqcn = select(r > F::new(-0.1f32) && r < F::new(1.1f32), dr, zero);
    let s = (qb - qu) / dqnz;
    let ds = ((dqb - dqu) - s * ddqnz) / dqnz;
    let qbn0 = max(s, qcn);
    let dqbn0 = select(s > qcn, ds, dqcn);
    let cl = max(F::new(1.0e-10f32), cfl.abs());
    let l2 = qcn / cl;
    let a = min(qbn0, one);
    let da = select(qbn0 < one, dqbn0, zero);
    let qbn = min(a, l2);
    let dqbn = select(a < l2, da, dqcn / cl);
    let dqbr = dqu + dqbn * dq + qbn * ddq;
    let cfac = select(F::new(2.0f32) * (qcn - F::new(0.5f32)).abs() >= one, one, zero);
    (one - cfac) * dqbr + cfac * dqc
}

/// Tangent of [`theta_flux`] along the tangents of the four cell values.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn theta_flux_d<F: Float>(ct: F, ctp: F, qtm: F, qt: F, qtp: F, qtpp: F, dqtm: F, dqt: F, dqtp: F, dqtpp: F) -> F {
    let half = F::new(0.5f32);
    let one = F::new(1.0f32);
    let two = F::new(2.0f32);
    let sixth = one / F::new(6.0f32);
    let cf = half * (ct + ctp);
    let up = cf < F::new(0.0f32);
    let qc = select(up, qtp, qt);
    let qcm = select(up, qt, qtm);
    let qcp = select(up, qtpp, qtp);
    let dqc = select(up, dqtp, dqt);
    let dqcm = select(up, dqt, dqtm);
    let dqcp = select(up, dqtpp, dqtp);
    let qb = half * ((one - cf) * qtp + (one + cf) * qt) - (one - cf * cf) * sixth * (qcm - two * qc + qcp);
    let dqb = half * ((one - cf) * dqtp + (one + cf) * dqt) - (one - cf * cf) * sixth * (dqcm - two * dqc + dqcp);
    let qu = select(up, qcp, qcm);
    let qd = select(up, qcm, qcp);
    let dqu = select(up, dqcp, dqcm);
    let dqd = select(up, dqcm, dqcp);
    cf * uq_limit_d::<F>(qb, qc, qu, qd, cf, dqb, dqc, dqu, dqd)
}

/// [`crate::prop::ktp3_theta()`] with its tangent `vad`: both rows of each
/// unit are staged in shared memory (launch with `block` units such that
/// `2 block (nth + 1)` values fit).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ktp3_theta_tl<F: Float>(
    va: &mut [F],
    vad: &mut [F],
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
    let mut sd = Shared::<[F]>::new_slice(block * (nth + 1));
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
            sd[row * rs + th] = vad[e0 + e];
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
            let dq0 = sd[base];
            let dq1 = sd[base + 1];
            let c0 = theta_cfl::<F>(es2[0], esc[0], ec2[0], ecos[0], esin[0], dcyx, dcxxyy, dcxy, frg, frk, gx, gy, ctmax);
            let cl = theta_cfl::<F>(es2[l], esc[l], ec2[l], ecos[l], esin[l], dcyx, dcxxyy, dcxy, frg, frk, gx, gy, ctmax);
            let qlm = sm[base + l - 1];
            let ql = sm[base + l];
            let dqlm = sd[base + l - 1];
            let dql = sd[base + l];
            let mut fprev = theta_flux::<F>(cl, c0, qlm, ql, q0, q1);
            let mut dfprev = theta_flux_d::<F>(cl, c0, qlm, ql, q0, q1, dqlm, dql, dq0, dq1);
            let mut qm = ql;
            let mut qc = q0;
            let mut qp = q1;
            let mut dqm = dql;
            let mut dqc = dq0;
            let mut dqp = dq1;
            let mut cc = c0;
            for th in 0..nth {
                let t2 = th + 2;
                let mut qpp = q1;
                let mut dqpp = dq1;
                if t2 < nth {
                    qpp = sm[base + t2];
                    dqpp = sd[base + t2];
                } else if t2 == nth {
                    qpp = q0;
                    dqpp = dq0;
                }
                let tp = (th + 1) % nth;
                let cp = theta_cfl::<F>(es2[tp], esc[tp], ec2[tp], ecos[tp], esin[tp], dcyx, dcxxyy, dcxy, frg, frk, gx, gy, ctmax);
                let fl = theta_flux::<F>(cc, cp, qm, qc, qp, qpp);
                let dfl = theta_flux_d::<F>(cc, cp, qm, qc, qp, qpp, dqm, dqc, dqp, dqpp);
                let v = qc + fprev - fl;
                sm[base + th] = max(zero, v);
                sd[base + th] = select(v > zero, dqc + dfprev - dfl, zero);
                fprev = fl;
                dfprev = dfl;
                qm = qc;
                qc = qp;
                qp = qpp;
                dqm = dqc;
                dqc = dqp;
                dqp = dqpp;
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
            vad[e0 + e] = sd[row * rs + th];
        }
    }
}

/// [`crate::prop::ktp3_k()`] with its tangent `vad`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ktp3_k_tl<F: Float>(
    va: &mut [F],
    vad: &mut [F],
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
            let stride = np * nth;
            let c0 = i * nth + th;
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
            let mut d_l = zero;
            let mut d_c = vad[c0];
            let mut d_h = vad[c0 + stride];
            let mut d_hh = vad[c0 + 2 * stride];
            let mut fprev = vel_c * select(vel_c >= zero, zero, q_c);
            let mut dfprev = vel_c * select(vel_c >= zero, zero, d_c);
            for m in 1..nk {
                let v = half * (vel_c + vel_h);
                let cf = fack * v / dm_c;
                let up = cf < zero;
                let qc = select(up, q_h, q_c);
                let qcm = select(up, q_c, q_l);
                let qcp = select(up, q_hh, q_h);
                let dqc = select(up, d_h, d_c);
                let dqcm = select(up, d_c, d_l);
                let dqcp = select(up, d_hh, d_h);
                let dbi = select(up, db_h, db_c);
                let dmi = select(up, dm_h, dm_c);
                let dmim = select(up, dm_c, dm_l);
                let curv = dm_c * dm_c / dbi * (one - cf * cf) * sixth;
                let qb = half * ((one - cf) * q_h + (one + cf) * q_c) - curv * ((qcp - qc) / dmi - (qc - qcm) / dmim);
                let dqb = half * ((one - cf) * d_h + (one + cf) * d_c) - curv * ((dqcp - dqc) / dmi - (dqc - dqcm) / dmim);
                let qu = select(up, qcp, qcm);
                let qd = select(up, qcm, qcp);
                let dqu = select(up, dqcp, dqcm);
                let dqd = select(up, dqcm, dqcp);
                let fl = v * uq_limit::<F>(qb, qc, qu, qd, cf);
                let dfl = v * uq_limit_d::<F>(qb, qc, qu, qd, cf, dqb, dqc, dqu, dqd);
                let nv = q_c + fack / db_c * (fprev - fl);
                va[c0 + (m - 1) * stride] = max(zero, nv);
                vad[c0 + (m - 1) * stride] = select(nv > zero, d_c + fack / db_c * (dfprev - dfl), zero);
                fprev = fl;
                dfprev = dfl;
                let j3 = m + 3;
                let mut qn = zero;
                let mut dn = zero;
                if j3 <= nk {
                    qn = va[c0 + (j3 - 1) * stride];
                    dn = vad[c0 + (j3 - 1) * stride];
                } else if j3 == nk + 1 {
                    qn = fachfa * q_hh;
                    dn = fachfa * d_hh;
                }
                q_l = q_c;
                q_c = q_h;
                q_h = q_hh;
                q_hh = qn;
                d_l = d_c;
                d_c = d_h;
                d_h = d_hh;
                d_hh = dn;
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
            let fl = vel_c * select(vel_c >= zero, q_c, q_h);
            let dfl = vel_c * select(vel_c >= zero, d_c, d_h);
            let nv = q_c + fack / db_c * (fprev - fl);
            va[c0 + (nk - 1) * stride] = max(zero, nv);
            vad[c0 + (nk - 1) * stride] = select(nv > zero, d_c + fack / db_c * (dfprev - dfl), zero);
        }
    }
}

/// Tangent of [`node_value`]: the boundary nodes keep a zero tangent.
#[cube]
#[allow(clippy::too_many_arguments)]
fn node_value_d<F: Float>(ui: F, st: F, dt: F, sa: F, m: F, active: bool, bnd: bool, dui: F, dst: F) -> F {
    let zero = F::new(0.0f32);
    let x = ui - dt * sa * st;
    let v = select(x > zero, dui - dt * sa * dst, zero) * m;
    select(active, select(bnd, zero, v), dui)
}

/// [`crate::prop::geo_node_x4()`] with its tangent: `u`, `ut` the values and
/// `du`, `dut` their tangents (from [`crate::prop::geo_tri_x4()`] applied to
/// each), written to `out` and `dout`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn geo_node_x4_tl<F: Float>(
    u: &[F],
    du: &[F],
    out: &mut [F],
    dout: &mut [F],
    ut: &[F],
    dut: &[F],
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
        let du0 = du[base];
        let du1 = du[base + 1];
        let du2 = du[base + 2];
        let du3 = du[base + 3];
        let mut st0 = zero;
        let mut st1 = zero;
        let mut st2 = zero;
        let mut st3 = zero;
        let mut ds0 = zero;
        let mut ds1 = zero;
        let mut ds2 = zero;
        let mut ds3 = zero;
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
            let k0p = max(a * c0 + bb * s0 + d, zero);
            let k1p = max(a * c1 + bb * s1 + d, zero);
            let k2p = max(a * c2 + bb * s2 + d, zero);
            let k3p = max(a * c3 + bb * s3 + d, zero);
            st0 += k0p * (u0 - ut[r]);
            st1 += k1p * (u1 - ut[r + 1]);
            st2 += k2p * (u2 - ut[r + 2]);
            st3 += k3p * (u3 - ut[r + 3]);
            ds0 += k0p * (du0 - dut[r]);
            ds1 += k1p * (du1 - dut[r + 1]);
            ds2 += k2p * (du2 - dut[r + 2]);
            ds3 += k3p * (du3 - dut[r + 3]);
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
        dout[base] = node_value_d::<F>(u0, st0, dtk[kk], sa, mask_bit::<F>(wlo, whi, k0), sub < iter[kk], bnd, du0, ds0);
        dout[base + 1] = node_value_d::<F>(u1, st1, dtk[kk + 1], sa, mask_bit::<F>(wlo, whi, k0 + 1), sub < iter[kk + 1], bnd, du1, ds1);
        dout[base + 2] = node_value_d::<F>(u2, st2, dtk[kk + 2], sa, mask_bit::<F>(wlo, whi, k0 + 2), sub < iter[kk + 2], bnd, du2, ds2);
        dout[base + 3] = node_value_d::<F>(u3, st3, dtk[kk + 3], sa, mask_bit::<F>(wlo, whi, k0 + 3), sub < iter[kk + 3], bnd, du3, ds3);
    }
}

/// Tangent of [`crate::prop::ac_to_va()`]: `VAD = CG / CLATS DAC` where the
/// value is positive, 0 where the clamp acts (call before `ac_to_va`
/// overwrites nothing it reads: both read `ac`).
#[cube(launch_unchecked)]
pub fn ac_to_va_tl<F: Float>(ac: &[F], dac: &[F], vad: &mut [F], cg: &[F], clats: &[F], np: usize, ik: usize, #[comptime] nth: usize) {
    let g = ABSOLUTE_POS;
    if g < np * nth {
        let i = g / nth;
        let f = cg[(ik + 1) * np + i] / clats[i];
        vad[ik * np * nth + g] = select(f * ac[g] > F::new(0.0f32), f * dac[g], F::new(0.0f32));
    }
}

/// [`crate::prop::ulev_node()`] remapping the tangent `vad` with the weights
/// of the values.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ulev_node_tl<F: Float>(
    va: &mut [F],
    vad: &mut [F],
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
                    let mut t0 = max(ii2, nk + 1);
                    if ii2 <= nk && rd2 <= F::new(0.95f32) {
                        t0 = max(ii2, 2);
                    }
                    let mut ta = Array::<F>::new(nk);
                    let mut a = Array::<F>::new(nk);
                    let mut tb = Array::<F>::new(nk);
                    let mut b = Array::<F>::new(nk);
                    for th in 0..nth {
                        for k in 0..nk {
                            let f = dsip[k + 1] / cgo[k + 1];
                            ta[k] = va[(k * np + i) * nth + th] * f;
                            tb[k] = vad[(k * np + i) * nth + th] * f;
                            a[k] = zero;
                            b[k] = zero;
                        }
                        for k in 0..nk {
                            let j1 = wi1[k] as usize;
                            if j1 <= nk {
                                let w1 = wr1[k];
                                if j1 >= 1 {
                                    a[j1 - 1] += w1 * ta[k];
                                    b[j1 - 1] += w1 * tb[k];
                                }
                                if j1 < nk {
                                    a[j1] += (one - w1) * ta[k];
                                    b[j1] += (one - w1) * tb[k];
                                }
                            }
                        }
                        for k in 0..nk {
                            let f = dsip[k + 1] / cgn[k + 1];
                            a[k] = a[k] / f;
                            b[k] = b[k] / f;
                        }
                        for k in 1..nk {
                            if k + 1 >= t0 {
                                a[k] = fachfa * a[k - 1];
                                b[k] = fachfa * b[k - 1];
                            }
                        }
                        for k in 0..nk {
                            va[(k * np + i) * nth + th] = a[k];
                            vad[(k * np + i) * nth + th] = b[k];
                        }
                    }
                    mode[i] = 3u32;
                }
            }
        }
    }
}

/// Tangent of the wind of the step ([`crate::srce::wind_interp()`]) for a
/// relative change of the wind factor: the interpolated speed scales with
/// the factor (the `WNT2` correction is homogeneous of degree 0), and the
/// tangent goes through the wind relative to the current.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn wind_interp_tl<F: Float>(
    ua0: &[F],
    uai: &[F],
    ud0: &[F],
    udi: &[F],
    cx: &[F],
    cy: &[F],
    du10: &mut [F],
    du10d: &mut [F],
    rd_: f32,
    rwindc_: f32,
    flcur_: u32,
    np: usize,
) {
    let i = ABSOLUTE_POS;
    if i < np {
        let rd = F::cast_from(rd_);
        let zero = F::new(0.0f32);
        let one = F::new(1.0f32);
        let rd2 = one - rd;
        let a0 = ua0[i];
        let ai = uai[i];
        let mut ua = a0 + rd * ai;
        let an = a0 + ai;
        let ui2 = (rd2 * a0 * a0 + rd * an * an).sqrt();
        ua = ua * min(F::new(1.25f32), ui2 / max(F::new(1.0e-7f32), ua));
        let ud = ud0[i] + rd * udi[i];
        if flcur_ == 1u32 {
            let rw = F::cast_from(rwindc_);
            let cu = ud.cos();
            let su = ud.sin();
            let uxr = ua * cu - rw * cx[i];
            let uyr = ua * su - rw * cy[i];
            let n2 = uxr * uxr + uyr * uyr;
            let n = n2.sqrt();
            du10[i] = select(n > F::new(0.001f32), (uxr * cu + uyr * su) * ua / n, zero);
            du10d[i] = select(n2 > zero, (uxr * su - uyr * cu) * ua / n2, zero);
        } else {
            du10[i] = select(ua > F::new(0.001f32), ua, zero);
            du10d[i] = zero;
        }
    }
}

/// Tangent of [`crate::prop::outputs()`]: `dhs` and `dt02` the tangents of
/// H_s and T_m02 (`UNDEF` where `MAPSTA <= 0`) and `den` that of the energy
/// without the tail.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn outputs_tl<F: Float>(
    va: &[F],
    vad: &[F],
    cg: &[F],
    sig: &[F],
    dden: &[F],
    nflag: &[u32],
    dhs: &mut [F],
    dt02: &mut [F],
    den: &mut [F],
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
        let mut det = zero;
        let mut et02 = zero;
        let mut det02 = zero;
        let mut ab = zero;
        let mut dab = zero;
        let mut c = F::new(1.0f32);
        for k in 0..nk {
            ab = zero;
            dab = zero;
            for th in 0..nth {
                ab += va[(k * np + i) * nth + th];
                dab += vad[(k * np + i) * nth + th];
            }
            c = cg[(k + 1) * np + i];
            let s = sig[k + 1];
            et += ab * dden[k] / c;
            det += dab * dden[k] / c;
            et02 += ab * dden[k] / c * s * s;
            det02 += dab * dden[k] / c * s * s;
        }
        let draw = det;
        et += fte * ab / c;
        det += fte * dab / c;
        let sn = sig[nk];
        let tail = F::new(0.5f32) * sn * sn * sn * sn * dth;
        et02 += ab / c * tail;
        det02 += dab / c * tail;
        let out = (nflag[i] & 16u32) != 0u32;
        let d = select(et > zero, F::new(2.0f32) * det / et.sqrt(), zero);
        // T_m02 = 2 pi sqrt(et / et02) where et02 > 1e-7 and et > 0, else 2 pi / sig(nk)
        let ok = et02 > F::new(1.0e-7f32) && et > zero;
        let tm = F::new(6.2831855f32) * (et / max(et02, F::new(1.0e-30f32))).sqrt();
        let dt = select(ok, F::new(0.5f32) * tm * (det / et - det02 / et02), zero);
        dhs[i] = select(out, d, undef);
        dt02[i] = select(out, dt, undef);
        den[i] = select(out, draw, zero);
    }
}
