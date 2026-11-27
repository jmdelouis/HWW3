//! CubeCL kernels of the third-order geographic propagation of WW3's
//! structured grids (`W3XYP3` with the switches `PR3` + `UQ`: the ULTIMATE
//! QUICKEST scheme `W3QCK3` with the transparencies of the sub-grid
//! obstructions and of the ice, and the averaging of Booij and Holthuijsen
//! against the Garden-Sprinkler Effect), and the structured gradients of
//! `W3DZXY` that `W3KTP3` takes on these grids: the second propagation path
//! of [`crate::model`], beside the N scheme on triangles of [`crate::prop`].
//!
//! The kernels follow the CPU transcription [`crate::pro3`] (itself checked
//! against WW3) through the tables of [`crate::uqgrid`]: the whole grid in
//! WW3's layout `k = ix NY + iy`, the `NTH` directions of one frequency at
//! once (`q[th NX NY + k]`, one unit per point and direction). One global
//! step of frequency `ik`:
//! 1. [`uq_gather()`]: `Q = VA CLATS / CG` at the sea points, zero elsewhere;
//! 2. for each sub-step (`NTLOC` of the direction, [`crate::uqgrid::ptab`];
//!    directions with fewer sub-steps are copied through):
//!    [`uq_average()`], the averaging gathered at each point from its eight
//!    neighbours (the point of `MAPTRN` keeping what it had and what the
//!    points after it in `MAPCXY` send, the points of `MAPSTA = 2` keeping
//!    their value, what crosses the closure in longitude always added), times
//!    `GSQRT`; then the two sweeps of [`uq_sweep()`] in the order of `YFIRST`,
//!    each point's two face fluxes of `W3QCK3` from the values before the
//!    sweep (third order with the ULTIMATE limiter between two points of
//!    `MAPSTA = 1`, first order on the side of the only one), the
//!    transparencies on the incoming fluxes, divided by `GSQRT` after the
//!    second;
//! 3. [`uq_scatter()`]: `VA = max(0, CG / CLATS Q)` where `MAPSTA > 0`, `Q`
//!    where it is negative.
//!
//! The tangent ([`uq_sweep_tl()`], [`uq_scatter_tl()`]; the gathering and
//! the averaging are linear and their kernels serve the tangent as they
//! are): the scheme is linear in the field once the branches are fixed,
//! and the tangent takes the derivative of the branch the values take. Its
//! non-differentiable points are those of the values: the limiter (the
//! clipping of the normalised value `QCN` to [-0.1, 1.1], the minimum and
//! maximum of `QBN`, the switch `CFAC` to the central value when `QCN`
//! leaves [0, 1]), `|DQ| < 1e-15` (constant denominator), the sign of each
//! face flux where the transparency of that side is not 1, and the clamps
//! `MAX(0, ...)` (zero tangent where they act).
//!
//! Not in the kernels: the boundary spectra (`FLBPI`), the moving grid
//! (`VGX`, `VGY`), the tripolar closure, Cartesian grids.

// The `#[cube]` macro generates, for every kernel, a launch struct and
// functions that carry no documentation; the kernels themselves are
// documented below.
#![allow(missing_docs)]

use crate::device::Real;
use crate::pro3::Pro3Grid;
use crate::prop::theta_flux;
use crate::prop_tl::theta_flux_d;
use crate::uqgrid::{ptab, yfirst, UqSetup, UqTables, W3Steps, PT};
use anyhow::{ensure, Result};
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// `Q = VA / CG CLATS` of frequency `ik` at the sea points (zero elsewhere),
/// one unit per point and direction.
#[cube(launch_unchecked)]
pub fn uq_gather<F: Float>(va: &[F], q: &mut [F], cg: &[F], clats: &[F], gsea: &[u32], np: usize, ik: usize, ng: usize, #[comptime] nth: usize) {
    let g = ABSOLUTE_POS;
    if g < nth * ng {
        let th = g / ng;
        let k = g - th * ng;
        let s = gsea[k];
        let mut v = F::new(0.0f32);
        if s != 4294967295u32 {
            let i = s as usize;
            v = va[(ik * np + i) * nth + th] / cg[(ik + 1) * np + i] * clats[i];
        }
        q[g] = v;
    }
}

/// The share of a source point's action that one corner of its averaging
/// sends to the neighbour `(tx, ty)`, or with `selfmode` what stays at the
/// source minus `RD / 3` ([`crate::uqgrid::corner_w`]).
#[cube]
#[allow(clippy::too_many_arguments)]
fn corner_w<F: Float>(dxc: F, dyc: F, cgs: F, tx: u32, ty: u32, wet: u32, central: u32, selfmode: u32) -> F {
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let six = F::new(6.0f32);
    let ox = select(dxc < zero, 0u32, 2u32);
    let oy = select(dyc < zero, 0u32, 2u32);
    let adx = dxc.abs();
    let ady = dyc.abs();
    let xmajor = adx > ady;
    let ra = select(xmajor, (dyc / dxc).abs(), select(ady > F::new(1.0e-5f32), (dxc / dyc).abs(), one));
    let rb = select(xmajor, adx, ady);
    let rdb = min(rb * cgs, one);
    let wd = rdb * ra / six;
    let wa = (one - ra) * rdb / six;
    let axx = select(xmajor, ox, 1u32);
    let axy = select(xmajor, 1u32, oy);
    let mut w = zero;
    for e in 0..4u32 {
        let neg = e % 2u32 == 1u32;
        let diag = e < 2u32;
        let bx = select(diag, ox, axx);
        let by = select(diag, oy, axy);
        let px = select(neg, 2u32 - bx, bx);
        let py = select(neg, 2u32 - by, by);
        let we = select(diag, wd, wa);
        let wt = (wet >> (3u32 * py + px)) & 1u32;
        let wx = (wet >> (3u32 + px)) & 1u32;
        let wy = (wet >> (3u32 * py + 1u32)) & 1u32;
        let test = wt == 1u32 && (!diag || wx == 1u32 || wy == 1u32);
        let pass = central == 1u32 || test;
        if selfmode == 1u32 {
            if !pass {
                w += we;
            }
        } else if pass && px == tx && py == ty {
            w += we;
        }
    }
    if selfmode == 1u32 {
        w -= rdb / F::new(3.0f32);
    }
    w
}

/// The share of a source point's action sent to `(tx, ty)`, or kept (plus
/// 1) with `selfmode`: the two corners of the averaging from the widths
/// `FTHX`, `FTHY`, `FCGX`, `FCGY` of the direction and the metric of the
/// point.
#[cube]
#[allow(clippy::too_many_arguments)]
fn source_w<F: Float>(cl: F, dpdx: F, dpdy: F, dqdx: F, dqdy: F, fthx: F, fthy: F, fcgx: F, fcgy: F, cgs: F, tx: u32, ty: u32, fl: u32, selfmode: u32) -> F {
    let tmpx = fthx / cl;
    let dxn = dpdx * tmpx + dpdy * fthy;
    let dyn_ = dqdx * tmpx + dqdy * fthy;
    let tmpx2 = fcgx / cl;
    let dxs = dpdx * tmpx2 + dpdy * fcgy;
    let dys = dqdx * tmpx2 + dqdy * fcgy;
    let wet = fl & 511u32;
    let cen = (fl >> 13u32) & 1u32;
    let w = corner_w::<F>(dxn + dxs, dyn_ + dys, cgs, tx, ty, wet, cen, selfmode) + corner_w::<F>(dxn - dxs, dyn_ - dys, cgs, tx, ty, wet, cen, selfmode);
    select(selfmode == 1u32, F::new(1.0f32) + w, w)
}

/// The averaging of sub-step `sub` of `W3XYP3` (section 3.a), gathered at
/// each point from its neighbours, times `GSQRT` (section 3.b); a copy for
/// the directions whose sub-steps are over.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn uq_average<F: Float>(
    qin: &[F],
    qout: &mut [F],
    gfl: &[u32],
    gpos: &[u32],
    gsea: &[u32],
    cg: &[F],
    clats: &[F],
    dpdx: &[F],
    dpdy: &[F],
    dqdx: &[F],
    dqdy: &[F],
    gsqrt: &[F],
    ptab: &[F],
    np: usize,
    ik: usize,
    nx: usize,
    ny: usize,
    closed: u32,
    sub: u32,
    #[comptime] nth: usize,
) {
    let ng = nx * ny;
    let g = ABSOLUTE_POS;
    if g < nth * ng {
        let th = g / ng;
        let k = g - th * ng;
        let pt = (ik * nth + th) * 8;
        let zero = F::new(0.0f32);
        let mut out = qin[g];
        if F::cast_from(sub) < ptab[pt] {
            out = zero;
            let fk = gfl[k];
            if ((fk >> 10u32) & 1u32) == 1u32 && ((fk >> 11u32) & 1u32) == 1u32 {
                let sk = gsea[k] as usize;
                let cgo = (ik + 1) * np;
                let fthx = ptab[pt + 2];
                let fthy = ptab[pt + 3];
                let fcgx = ptab[pt + 4];
                let fcgy = ptab[pt + 5];
                let ak = qin[g];
                let trn = (fk >> 12u32) & 1u32;
                let mut sum = ak;
                if trn == 0u32 {
                    sum = ak * source_w::<F>(clats[sk], dpdx[sk], dpdy[sk], dqdx[sk], dqdy[sk], fthx, fthy, fcgx, fcgy, cg[cgo + sk], 1u32, 1u32, fk, 1u32);
                }
                let mut seam = zero;
                let ix = k / ny;
                let iy = k - ix * ny;
                let posk = gpos[k];
                for dyi in 0..3u32 {
                    for dxi in 0..3u32 {
                        let center = dxi == 1u32 && dyi == 1u32;
                        let rowok = !(iy == 0 && dyi == 0u32) && !(iy + 1 == ny && dyi == 2u32);
                        let edge = (ix == 0 && dxi == 0u32) || (ix + 1 == nx && dxi == 2u32);
                        if !center && rowok && (!edge || closed == 1u32) {
                            let jx = (ix + nx + dxi as usize - 1) % nx;
                            let s = jx * ny + iy + dyi as usize - 1;
                            let fs = gfl[s];
                            let spread = ((fs >> 10u32) & 1u32) == 1u32 && ((fs >> 11u32) & 1u32) == 1u32 && ((fs >> 12u32) & 1u32) == 0u32;
                            let order = edge || trn == 0u32 || gpos[s] > posk;
                            if spread && order {
                                let ss = gsea[s] as usize;
                                let w = qin[th * ng + s]
                                    * source_w::<F>(clats[ss], dpdx[ss], dpdy[ss], dqdx[ss], dqdy[ss], fthx, fthy, fcgx, fcgy, cg[cgo + ss], 2u32 - dxi, 2u32 - dyi, fs, 0u32);
                                if edge {
                                    seam += w;
                                } else {
                                    sum += w;
                                }
                            }
                        }
                    }
                }
                if ((fk >> 14u32) & 1u32) == 1u32 {
                    sum = ak;
                }
                out = (sum + seam) * gsqrt[sk];
            }
        }
        qout[g] = out;
    }
}

/// The next (`fwd = 1`) or previous point of `k` along the sweep: `y`
/// (`ydir = 1`), or `x` with the wrap in longitude.
#[cube]
fn step_idx(k: usize, ydir: u32, fwd: u32, nx: usize, ny: usize) -> usize {
    let ix = k / ny;
    let iy = k - ix * ny;
    let mut r = ((ix + nx - 1) % nx) * ny + iy;
    if ydir == 1u32 {
        if fwd == 1u32 {
            r = k + 1;
        } else {
            r = k - 1;
        }
    } else if fwd == 1u32 {
        r = ((ix + 1) % nx) * ny + iy;
    }
    r
}

/// The Courant number of sea point `s` along the sweep (`m1`, `m2`: `DPDX`,
/// `DPDY` along `x`, `DQDX`, `DQDY` along `y`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn cfl_at<F: Float>(s: usize, cgo: usize, cg: &[F], clats: &[F], m1: &[F], m2: &[F], cx: &[F], cy: &[F], ec: F, es: F, dtrad: F, flcur: u32) -> F {
    let cl = clats[s];
    let c = cg[cgo + s];
    let mut cxtot = ec * c / cl;
    let mut cytot = es * c;
    if flcur == 1u32 {
        cxtot += cx[s] / cl;
        cytot += cy[s];
    }
    (cxtot * m1[s] + cytot * m2[s]) * dtrad
}

/// The flux of `W3QCK3` through the face between `a` and the next point
/// along the sweep (`q` of the direction at offset `qo`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn face_flux<F: Float>(
    a: usize,
    q: &[F],
    qo: usize,
    gfl: &[u32],
    gsea: &[u32],
    cg: &[F],
    clats: &[F],
    m1: &[F],
    m2: &[F],
    cx: &[F],
    cy: &[F],
    cgo: usize,
    ec: F,
    es: F,
    dtrad: F,
    flcur: u32,
    ydir: u32,
    nx: usize,
    ny: usize,
) -> F {
    let zero = F::new(0.0f32);
    let b = step_idx(a, ydir, 1u32, nx, ny);
    let ta = (gfl[a] >> 9u32) & 1u32;
    let tb = (gfl[b] >> 9u32) & 1u32;
    let mut f = zero;
    if ta == 1u32 && tb == 1u32 {
        let ca = cfl_at::<F>(gsea[a] as usize, cgo, cg, clats, m1, m2, cx, cy, ec, es, dtrad, flcur);
        let cb = cfl_at::<F>(gsea[b] as usize, cgo, cg, clats, m1, m2, cx, cy, ec, es, dtrad, flcur);
        let am = step_idx(a, ydir, 0u32, nx, ny);
        let bp = step_idx(b, ydir, 1u32, nx, ny);
        f = theta_flux::<F>(ca, cb, q[qo + am], q[qo + a], q[qo + b], q[qo + bp]);
    } else if ta == 1u32 || tb == 1u32 {
        let mut p = b;
        if ta == 1u32 {
            p = a;
        }
        let c = cfl_at::<F>(gsea[p] as usize, cgo, cg, clats, m1, m2, cx, cy, ec, es, dtrad, flcur);
        f = c * select(c < zero, q[qo + b], q[qo + a]);
    }
    f
}

/// The tangent of [`face_flux()`] along `dq`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn face_flux_d<F: Float>(
    a: usize,
    q: &[F],
    dq: &[F],
    qo: usize,
    gfl: &[u32],
    gsea: &[u32],
    cg: &[F],
    clats: &[F],
    m1: &[F],
    m2: &[F],
    cx: &[F],
    cy: &[F],
    cgo: usize,
    ec: F,
    es: F,
    dtrad: F,
    flcur: u32,
    ydir: u32,
    nx: usize,
    ny: usize,
) -> F {
    let zero = F::new(0.0f32);
    let b = step_idx(a, ydir, 1u32, nx, ny);
    let ta = (gfl[a] >> 9u32) & 1u32;
    let tb = (gfl[b] >> 9u32) & 1u32;
    let mut f = zero;
    if ta == 1u32 && tb == 1u32 {
        let ca = cfl_at::<F>(gsea[a] as usize, cgo, cg, clats, m1, m2, cx, cy, ec, es, dtrad, flcur);
        let cb = cfl_at::<F>(gsea[b] as usize, cgo, cg, clats, m1, m2, cx, cy, ec, es, dtrad, flcur);
        let am = step_idx(a, ydir, 0u32, nx, ny);
        let bp = step_idx(b, ydir, 1u32, nx, ny);
        f = theta_flux_d::<F>(ca, cb, q[qo + am], q[qo + a], q[qo + b], q[qo + bp], dq[qo + am], dq[qo + a], dq[qo + b], dq[qo + bp]);
    } else if ta == 1u32 || tb == 1u32 {
        let mut p = b;
        if ta == 1u32 {
            p = a;
        }
        let c = cfl_at::<F>(gsea[p] as usize, cgo, cg, clats, m1, m2, cx, cy, ec, es, dtrad, flcur);
        f = c * select(c < zero, dq[qo + b], dq[qo + a]);
    }
    f
}

/// One sweep of `W3QCK3` along `x` (`ydir = 0`: `m1`, `m2` = `DPDX`, `DPDY`,
/// transparencies `ATRNX`) or `y` (`DQDX`, `DQDY`, `ATRNY`), with `on`
/// (`FLCX` or `FLCY`), at the active points; with `fin`, the division by
/// `GSQRT` at the sea points that ends the sub-step. A copy for the
/// directions whose sub-steps are over.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn uq_sweep<F: Float>(
    qin: &[F],
    qout: &mut [F],
    gfl: &[u32],
    gsea: &[u32],
    cg: &[F],
    clats: &[F],
    m1: &[F],
    m2: &[F],
    gsqrt: &[F],
    atrn: &[F],
    cx: &[F],
    cy: &[F],
    ptab: &[F],
    np: usize,
    ik: usize,
    nx: usize,
    ny: usize,
    sub: u32,
    ydir: u32,
    on: u32,
    fin: u32,
    flcur: u32,
    #[comptime] nth: usize,
) {
    let ng = nx * ny;
    let g = ABSOLUTE_POS;
    if g < nth * ng {
        let th = g / ng;
        let k = g - th * ng;
        let pt = (ik * nth + th) * 8;
        let zero = F::new(0.0f32);
        let one = F::new(1.0f32);
        let mut v = qin[g];
        if F::cast_from(sub) < ptab[pt] {
            let fk = gfl[k];
            if on == 1u32 && ((fk >> 15u32) & 1u32) == 1u32 {
                let qo = th * ng;
                let cgo = (ik + 1) * np;
                let dtrad = ptab[pt + 1];
                let ec = ptab[pt + 6];
                let es = ptab[pt + 7];
                let km = step_idx(k, ydir, 0u32, nx, ny);
                let fl = face_flux::<F>(km, qin, qo, gfl, gsea, cg, clats, m1, m2, cx, cy, cgo, ec, es, dtrad, flcur, ydir, nx, ny);
                let fr = face_flux::<F>(k, qin, qo, gfl, gsea, cg, clats, m1, m2, cx, cy, cgo, ec, es, dtrad, flcur, ydir, nx, ny);
                let s = gsea[k] as usize;
                let o = 4 * s + 2 * ydir as usize;
                let tl = select(fl > zero, atrn[o], one);
                let tr = select(fr < zero, atrn[o + 1], one);
                let x = v + tl * fl - tr * fr;
                v = max(zero, x);
            }
            if fin == 1u32 && ((fk >> 11u32) & 1u32) == 1u32 {
                v = v / gsqrt[gsea[k] as usize];
            }
        }
        qout[g] = v;
    }
}

/// [`uq_sweep()`] with its tangent: the values exactly as [`uq_sweep()`],
/// the tangent `dqout` along `dqin`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn uq_sweep_tl<F: Float>(
    qin: &[F],
    dqin: &[F],
    qout: &mut [F],
    dqout: &mut [F],
    gfl: &[u32],
    gsea: &[u32],
    cg: &[F],
    clats: &[F],
    m1: &[F],
    m2: &[F],
    gsqrt: &[F],
    atrn: &[F],
    cx: &[F],
    cy: &[F],
    ptab: &[F],
    np: usize,
    ik: usize,
    nx: usize,
    ny: usize,
    sub: u32,
    ydir: u32,
    on: u32,
    fin: u32,
    flcur: u32,
    #[comptime] nth: usize,
) {
    let ng = nx * ny;
    let g = ABSOLUTE_POS;
    if g < nth * ng {
        let th = g / ng;
        let k = g - th * ng;
        let pt = (ik * nth + th) * 8;
        let zero = F::new(0.0f32);
        let one = F::new(1.0f32);
        let mut v = qin[g];
        let mut dv = dqin[g];
        if F::cast_from(sub) < ptab[pt] {
            let fk = gfl[k];
            if on == 1u32 && ((fk >> 15u32) & 1u32) == 1u32 {
                let qo = th * ng;
                let cgo = (ik + 1) * np;
                let dtrad = ptab[pt + 1];
                let ec = ptab[pt + 6];
                let es = ptab[pt + 7];
                let km = step_idx(k, ydir, 0u32, nx, ny);
                let fl = face_flux::<F>(km, qin, qo, gfl, gsea, cg, clats, m1, m2, cx, cy, cgo, ec, es, dtrad, flcur, ydir, nx, ny);
                let fr = face_flux::<F>(k, qin, qo, gfl, gsea, cg, clats, m1, m2, cx, cy, cgo, ec, es, dtrad, flcur, ydir, nx, ny);
                let dfl = face_flux_d::<F>(km, qin, dqin, qo, gfl, gsea, cg, clats, m1, m2, cx, cy, cgo, ec, es, dtrad, flcur, ydir, nx, ny);
                let dfr = face_flux_d::<F>(k, qin, dqin, qo, gfl, gsea, cg, clats, m1, m2, cx, cy, cgo, ec, es, dtrad, flcur, ydir, nx, ny);
                let s = gsea[k] as usize;
                let o = 4 * s + 2 * ydir as usize;
                let tl = select(fl > zero, atrn[o], one);
                let tr = select(fr < zero, atrn[o + 1], one);
                let x = v + tl * fl - tr * fr;
                let dx = dv + tl * dfl - tr * dfr;
                v = max(zero, x);
                dv = select(x > zero, dx, zero);
            }
            if fin == 1u32 && ((fk >> 11u32) & 1u32) == 1u32 {
                let gs = gsqrt[gsea[k] as usize];
                v = v / gs;
                dv = dv / gs;
            }
        }
        qout[g] = v;
        dqout[g] = dv;
    }
}

/// `VA = max(0, CG / CLATS Q)` of frequency `ik` at the sea points of
/// `MAPSTA > 0`, `VA = Q` at those of `MAPSTA < 0`, one unit per sea point
/// and direction.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn uq_scatter<F: Float>(q: &[F], va: &mut [F], cg: &[F], clats: &[F], gfl: &[u32], ksea: &[u32], np: usize, ik: usize, ng: usize, #[comptime] nth: usize) {
    let g = ABSOLUTE_POS;
    if g < nth * np {
        let th = g / np;
        let i = g - th * np;
        let k = ksea[i] as usize;
        let x = q[th * ng + k];
        let c = cg[(ik + 1) * np + i] / clats[i];
        va[(ik * np + i) * nth + th] = select(((gfl[k] >> 10u32) & 1u32) == 1u32, max(F::new(0.0f32), c * x), x);
    }
}

/// [`uq_scatter()`] with its tangent.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn uq_scatter_tl<F: Float>(q: &[F], dq: &[F], va: &mut [F], vad: &mut [F], cg: &[F], clats: &[F], gfl: &[u32], ksea: &[u32], np: usize, ik: usize, ng: usize, #[comptime] nth: usize) {
    let g = ABSOLUTE_POS;
    if g < nth * np {
        let zero = F::new(0.0f32);
        let th = g / np;
        let i = g - th * np;
        let k = ksea[i] as usize;
        let x = q[th * ng + k];
        let dx = dq[th * ng + k];
        let c = cg[(ik + 1) * np + i] / clats[i];
        let pos = ((gfl[k] >> 10u32) & 1u32) == 1u32;
        let o = (ik * np + i) * nth + th;
        vad[o] = select(pos, select(c * x > zero, c * dx, zero), dx);
        va[o] = select(pos, max(zero, c * x), x);
    }
}

/// `W3DZXY`: the gradients of `z` (one value per sea point) with the
/// stencils of [`crate::uqgrid::UqTables`].
#[cube(launch_unchecked)]
pub fn uq_dzxy<F: Float>(z: &[F], dzdx: &mut [F], dzdy: &mut [F], dzi: &[u32], dzc: &[F], nsea: usize) {
    let i = ABSOLUTE_POS;
    if i < nsea {
        let a = z[dzi[4 * i] as usize] - z[dzi[4 * i + 1] as usize];
        let b = z[dzi[4 * i + 2] as usize] - z[dzi[4 * i + 3] as usize];
        dzdx[i] = a * dzc[4 * i] + b * dzc[4 * i + 1];
        dzdy[i] = a * dzc[4 * i + 2] + b * dzc[4 * i + 3];
    }
}

/// `dst[dof + i] = src[so + i]` for `i < n`.
#[cube(launch_unchecked)]
pub fn uq_copy<F: Float>(src: &[F], dst: &mut [F], so: usize, dof: usize, n: usize) {
    let i = ABSOLUTE_POS;
    if i < n {
        dst[dof + i] = src[so + i];
    }
}

// ---------------------------------------------------------------------------
// Host state
// ---------------------------------------------------------------------------

fn bf(h: &Handle, n: usize) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(h.clone(), n) }
}

#[allow(clippy::too_many_arguments)]
fn copy_launch<F: Real>(client: &Client, src: &Handle, slen: usize, dst: &Handle, dlen: usize, so: usize, dof: usize, n: usize) {
    unsafe {
        uq_copy::launch_unchecked::<F>(client, CubeCount::Static((n as u32).div_ceil(256).max(1), 1, 1), CubeDim::new_1d(256), bf(src, slen), bf(dst, dlen), so, dof, n);
    }
}

/// The values `off .. off + n` of the device buffer `h` of `len` values,
/// in double precision (for the checks: a part of a large buffer).
pub fn read_range<F: Real>(client: &Client, h: &Handle, len: usize, off: usize, n: usize) -> Vec<f64> {
    let tmp = client.empty(n * std::mem::size_of::<F>());
    copy_launch::<F>(client, h, len, &tmp, n, off, 0, n);
    let b = client.read_one(tmp).expect("device read");
    F::from_bytes(&b).iter().map(|&x| <F as crate::backend::Scalar>::as_f64(x)).collect()
}

/// Write `v` at `off ..` in the device buffer `h` of `len` values.
pub fn write_range<F: Real>(client: &Client, h: &Handle, len: usize, off: usize, v: &[f64]) {
    let w: Vec<F> = v.iter().map(|&x| <F as crate::backend::Scalar>::of_f64(x)).collect();
    let tmp = client.create_from_slice(F::as_bytes(&w));
    copy_launch::<F>(client, &tmp, v.len(), h, len, 0, off, v.len());
}

/// The UQ scheme of one structured grid on the device: the maps of
/// `W3MAP3`, `W3UTRN`, `W3MAPT` on the host ([`Pro3Grid`]) and their tables
/// on the device ([`UqTables`]), the parameters of the step, the global
/// steps of `W3WAVE`, and the work buffers of one frequency (and of its
/// tangent).
pub struct UqState<F: Real> {
    /// The maps of the grid.
    pub pg: Pro3Grid,
    /// The tables of the kernels.
    pub tables: UqTables,
    /// The global steps of `W3WAVE` (from `DTMAX`).
    pub steps: W3Steps,
    /// `DTKTH` (s).
    pub dtkth: f64,
    /// `CFLTM`.
    pub cfltm: f64,
    /// `NTLOC` of every component in the last step ([`UqState::set_step`]).
    pub ntloc: Vec<usize>,
    nth: usize,
    d_gsea: Handle,
    d_ksea: Handle,
    d_gfl: Handle,
    d_gpos: Handle,
    d_atrn: Handle,
    d_met: [Handle; 6],
    d_dzi: Handle,
    d_dzc: Handle,
    d_ptab: Handle,
    d_a: Handle,
    d_b: Handle,
    d_da: Option<Handle>,
    d_db: Option<Handle>,
    _f: PhantomData<F>,
}

impl<F: Real> UqState<F> {
    /// The state of the grid of `setup` with `MAPSTA` and the ice of the sea
    /// points `mapsta`, `ice` (`W3UTRN`), the buffers of the tangent with
    /// `tangent`.
    pub fn new(client: &Client, setup: &UqSetup, mapsta: &[i32], ice: &[f32], tangent: bool) -> Result<UqState<F>> {
        let pg = setup.pro3()?;
        let tables = UqTables::new(&pg)?;
        ensure!(mapsta.len() == tables.nsea, "MAPSTA of {} points for {} sea points", mapsta.len(), tables.nsea);
        let nth = pg.nth;
        let ng = tables.ng;
        let fb = std::mem::size_of::<F>();
        let upd = |v: &[f64]| {
            let w: Vec<F> = v.iter().map(|&x| <F as crate::backend::Scalar>::of_f64(x)).collect();
            client.create_from_slice(F::as_bytes(&w))
        };
        let upu = |v: &[u32]| client.create_from_slice(u32::as_bytes(v));
        let d_met = [0, 1, 2, 3, 4, 5].map(|j| upd(&tables.metric[j]));
        let mut st = UqState {
            d_gsea: upu(&tables.gsea),
            d_ksea: upu(&tables.ksea),
            d_gfl: upu(&tables.gfl),
            d_gpos: upu(&tables.gpos),
            d_atrn: upd(&tables.atrn),
            d_met,
            d_dzi: upu(&tables.dzi),
            d_dzc: upd(&tables.dzc),
            d_ptab: upd(&vec![0.0; PT * pg.nk * nth]),
            d_a: client.empty(nth * ng * fb),
            d_b: client.empty(nth * ng * fb),
            d_da: if tangent { Some(client.empty(nth * ng * fb)) } else { None },
            d_db: if tangent { Some(client.empty(nth * ng * fb)) } else { None },
            steps: W3Steps::new(setup.dtmax),
            dtkth: setup.dtkth,
            cfltm: setup.cfltm,
            ntloc: vec![1; pg.nk * nth],
            nth,
            pg,
            tables,
            _f: PhantomData,
        };
        st.update_maps(client, mapsta, ice)?;
        Ok(st)
    }

    /// The update of the maps of `W3WAVE` (`FLMAP`: `W3MAP3`, `W3UTRN`,
    /// `W3MAPT`) for `MAPSTA` and the ice of the sea points, and of their
    /// tables on the device.
    pub fn update_maps(&mut self, client: &Client, mapsta: &[i32], ice: &[f32]) -> Result<()> {
        let t = &self.tables;
        let mut full = vec![0i32; t.ng];
        for (s, &i) in self.pg.sea_ij.iter().enumerate() {
            full[i] = mapsta[s];
        }
        let icef: Vec<f64> = ice.iter().map(|&c| c as f64).collect();
        self.pg.update_maps(Some(&full), Some(&icef), None)?;
        self.tables.update(&self.pg);
        self.d_gfl = client.create_from_slice(u32::as_bytes(&self.tables.gfl));
        self.d_gpos = client.create_from_slice(u32::as_bytes(&self.tables.gpos));
        let w: Vec<F> = self.tables.atrn.iter().map(|&x| <F as crate::backend::Scalar>::of_f64(x)).collect();
        self.d_atrn = client.create_from_slice(F::as_bytes(&w));
        Ok(())
    }

    /// The parameters of the sub-steps for the global step `dtg` with the
    /// currents `cur` of the sea points ([`ptab`]).
    pub fn set_step(&mut self, client: &Client, dtg: f64, cur: Option<(&[f64], &[f64])>) {
        let (t, nt) = ptab(&self.pg, dtg, cur);
        let w: Vec<F> = t.iter().map(|&x| <F as crate::backend::Scalar>::of_f64(x)).collect();
        self.d_ptab = client.create_from_slice(F::as_bytes(&w));
        self.ntloc = nt;
    }

    /// Largest `NTLOC` over the directions of frequency `ik`.
    pub fn max_ntloc(&self, ik: usize) -> usize {
        *self.ntloc[ik * self.nth..(ik + 1) * self.nth].iter().max().unwrap()
    }

    /// `W3DZXY` of `z` (one value per sea point) into `dx`, `dy`.
    pub fn dzxy(&self, client: &Client, z: &Handle, dx: &Handle, dy: &Handle, block: u32) {
        let n = self.tables.nsea;
        unsafe {
            uq_dzxy::launch_unchecked::<F>(
                client,
                CubeCount::Static((n as u32).div_ceil(block).max(1), 1, 1),
                CubeDim::new_1d(block),
                bf(z, n),
                bf(dx, n),
                bf(dy, n),
                bf(&self.d_dzi, 4 * n),
                bf(&self.d_dzc, 4 * n),
                n,
            );
        }
    }

    /// `W3XYP3` for every direction of frequency `ik` over the global step
    /// `dtg` ending `secs_of_day` after 00:00, in place in `va` (`[(ik np + i)
    /// nth + ith]`, `np` the sea points) with the group velocities `cg`
    /// (`[ik np + i]`, `ik = 0 .. NK+1`) and, with `flcur`, the currents
    /// `cx`, `cy`; with `vad`, the tangent too. Returns the number of
    /// sub-steps launched.
    #[allow(clippy::too_many_arguments)]
    pub fn geographic(
        &self,
        client: &Client,
        va: &Handle,
        vad: Option<&Handle>,
        cg: &Handle,
        cx: &Handle,
        cy: &Handle,
        flcur: bool,
        np: usize,
        ik: usize,
        secs_of_day: f64,
        dtg: f64,
        flcx: bool,
        flcy: bool,
        block: u32,
    ) -> usize {
        let t = &self.tables;
        let (nx, ny, ng, nth) = (t.nx, t.ny, t.ng, self.nth);
        let nk = self.pg.nk;
        let nspec = nk * nth;
        let n = nth * ng;
        let dims = |m: usize| (CubeCount::Static((m as u32).div_ceil(block).max(1), 1, 1), CubeDim::new_1d(block));
        let (c, d) = dims(n);
        let (cs, ds) = dims(nth * np);
        let [d_cl, d_px, d_py, d_qx, d_qy, d_gs] = &self.d_met;
        let ns = t.nsea;
        let ncg = (nk + 2) * np;
        let fl = flcur as u32;
        let gather = |src: &Handle, dst: &Handle| unsafe {
            uq_gather::launch_unchecked::<F>(client, c.clone(), d, bf(src, nk * np * nth), bf(dst, n), bf(cg, ncg), bf(d_cl, ns), bf(&self.d_gsea, ng), np, ik, ng, nth);
        };
        let average = |src: &Handle, dst: &Handle, sub: u32| unsafe {
            uq_average::launch_unchecked::<F>(
                client,
                c.clone(),
                d,
                bf(src, n),
                bf(dst, n),
                bf(&self.d_gfl, ng),
                bf(&self.d_gpos, ng),
                bf(&self.d_gsea, ng),
                bf(cg, ncg),
                bf(d_cl, ns),
                bf(d_px, ns),
                bf(d_py, ns),
                bf(d_qx, ns),
                bf(d_qy, ns),
                bf(d_gs, ns),
                bf(&self.d_ptab, PT * nspec),
                np,
                ik,
                nx,
                ny,
                t.closed as u32,
                sub,
                nth,
            );
        };
        // (m1, m2) of the sweep
        let metric = |ydir: u32| if ydir == 1 { (d_qx, d_qy) } else { (d_px, d_py) };
        let sweep = |src: &Handle, dst: &Handle, sub: u32, ydir: u32, on: bool, fin: bool| unsafe {
            let (m1, m2) = metric(ydir);
            uq_sweep::launch_unchecked::<F>(
                client,
                c.clone(),
                d,
                bf(src, n),
                bf(dst, n),
                bf(&self.d_gfl, ng),
                bf(&self.d_gsea, ng),
                bf(cg, ncg),
                bf(d_cl, ns),
                bf(m1, ns),
                bf(m2, ns),
                bf(d_gs, ns),
                bf(&self.d_atrn, 4 * ns),
                bf(cx, np),
                bf(cy, np),
                bf(&self.d_ptab, PT * nspec),
                np,
                ik,
                nx,
                ny,
                sub,
                ydir,
                on as u32,
                fin as u32,
                fl,
                nth,
            );
        };
        #[allow(clippy::too_many_arguments)]
        let sweep_tl = |src: &Handle, dsrc: &Handle, dst: &Handle, ddst: &Handle, sub: u32, ydir: u32, on: bool, fin: bool| unsafe {
            let (m1, m2) = metric(ydir);
            uq_sweep_tl::launch_unchecked::<F>(
                client,
                c.clone(),
                d,
                bf(src, n),
                bf(dsrc, n),
                bf(dst, n),
                bf(ddst, n),
                bf(&self.d_gfl, ng),
                bf(&self.d_gsea, ng),
                bf(cg, ncg),
                bf(d_cl, ns),
                bf(m1, ns),
                bf(m2, ns),
                bf(d_gs, ns),
                bf(&self.d_atrn, 4 * ns),
                bf(cx, np),
                bf(cy, np),
                bf(&self.d_ptab, PT * nspec),
                np,
                ik,
                nx,
                ny,
                sub,
                ydir,
                on as u32,
                fin as u32,
                fl,
                nth,
            );
        };
        let nsub = self.max_ntloc(ik);
        let mut yf = yfirst(secs_of_day, dtg);
        let on = |dir: u32| if dir == 1 { flcy } else { flcx };
        // the state in a, the other buffer b; each sub-step leaves its
        // result in b and the two swap
        let (mut a, mut b) = (&self.d_a, &self.d_b);
        match (vad, &self.d_da, &self.d_db) {
            (Some(vad), Some(da0), Some(db0)) => {
                let (mut da, mut db) = (da0, db0);
                gather(va, a);
                gather(vad, da);
                for sub in 0..nsub as u32 {
                    average(a, b, sub);
                    average(da, db, sub);
                    let (d1, d2) = if yf { (1u32, 0u32) } else { (0, 1) };
                    sweep_tl(b, db, a, da, sub, d1, on(d1), false);
                    sweep_tl(a, da, b, db, sub, d2, on(d2), true);
                    std::mem::swap(&mut a, &mut b);
                    std::mem::swap(&mut da, &mut db);
                    yf = !yf;
                }
                unsafe {
                    uq_scatter_tl::launch_unchecked::<F>(
                        client,
                        cs,
                        ds,
                        bf(a, n),
                        bf(da, n),
                        bf(va, nk * np * nth),
                        bf(vad, nk * np * nth),
                        bf(cg, ncg),
                        bf(d_cl, ns),
                        bf(&self.d_gfl, ng),
                        bf(&self.d_ksea, ns),
                        np,
                        ik,
                        ng,
                        nth,
                    );
                }
            }
            _ => {
                gather(va, a);
                for sub in 0..nsub as u32 {
                    average(a, b, sub);
                    let (d1, d2) = if yf { (1u32, 0u32) } else { (0, 1) };
                    sweep(b, a, sub, d1, on(d1), false);
                    sweep(a, b, sub, d2, on(d2), true);
                    std::mem::swap(&mut a, &mut b);
                    yf = !yf;
                }
                unsafe {
                    uq_scatter::launch_unchecked::<F>(
                        client,
                        cs,
                        ds,
                        bf(a, n),
                        bf(va, nk * np * nth),
                        bf(cg, ncg),
                        bf(d_cl, ns),
                        bf(&self.d_gfl, ng),
                        bf(&self.d_ksea, ns),
                        np,
                        ik,
                        ng,
                        nth,
                    );
                }
            }
        }
        nsub
    }

    /// A field of the sea points on the whole `NX x NY` grid (`iy NX + ix`),
    /// `undef` elsewhere.
    pub fn expand<T: Copy>(&self, v: &[T], undef: T) -> Vec<T> {
        let mut o = vec![undef; self.tables.nx * self.tables.ny];
        for (s, &i) in self.pg.sea_ij.iter().enumerate() {
            o[i] = v[s];
        }
        o
    }
}
