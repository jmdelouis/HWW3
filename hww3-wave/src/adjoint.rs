//! Milestone M4 prototypes: hand-written adjoints of the DIA (`W3SNL1`) and
//! of one geographic sub-step, checked against the tangent kernels of M3 by
//! the dot-product test `<L x, y> = <x, Lᵀ y>`, and timed against the value
//! kernels (`hww3-src adj-dia`, `hww3-prop adj-geo`). The Burn autodiff
//! versions of the same two pieces are in [`crate::burn_m4`].
//!
//! The DIA adjoint needs no intermediate of the forward pass: from the
//! spectrum it rebuilds the extended spectrum `UE`, then each bin of the
//! interaction grid gathers the cotangents of its three passes from the
//! cotangents of `S` and `D` through the transposed index tables of
//! `DIA_IC` ([`DiaAdjTables`], built once on the host, so no atomics), and
//! each bin of `UE` gathers its cotangent through the transposed tables of
//! `DIA_IP`.

// The `#[cube]` macro generates, for every kernel, a launch struct and
// functions that carry no documentation; the kernels themselves are
// documented below.
#![allow(missing_docs)]

use crate::sources::*;
use crate::srce::*;
use crate::srce_tl::snl1_tl;
use cubecl::prelude::*;

/// `S` and `D` of `W3SNL1` alone at each node (spectrum `va[i nspec + is]`,
/// group velocities `cg[i nk + ik]`, `k_mean depth` in `kdm`), scale 1.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn dia_value<F: Float>(
    va: &[F],
    cg: &[F],
    kdm: &[F],
    par: &[F],
    sig: &[F],
    dia_ip: &[u32],
    dia_ic: &[u32],
    af11: &[F],
    s_out: &mut [F],
    d_out: &mut [F],
    np: usize,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
    #[comptime] nspecx: usize,
    #[comptime] nspecy: usize,
    #[comptime] nu: usize,
) {
    let i = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let nspec = nk * nth;
    if i < np {
        let zero = F::new(0.0f32);
        let mut a = Shared::<[F]>::new_slice(nspec);
        let mut vs = Shared::<[F]>::new_slice(nspec);
        let mut vd = Shared::<[F]>::new_slice(nspec);
        let mut dn = Shared::<[F]>::new_slice(nspec);
        let mut ue = Shared::<[F]>::new_slice(nspecy + nth);
        let mut b1 = Shared::<[F]>::new_slice(nspecx + nth);
        let mut b2 = Shared::<[F]>::new_slice(nspecx + nth);
        let mut kb = Shared::<[F]>::new_slice(NKB * nk);
        for r in 0..(nspec + nu - 1) / nu {
            let is = r * nu + t;
            if is < nspec {
                a[is] = va[i * nspec + is];
                vs[is] = zero;
                vd[is] = zero;
            }
        }
        if t < nk {
            kb[K_CG * nk + t] = cg[i * nk + t];
        }
        sync_cube();
        crate::srce::snl1::<F>(
            par, sig, dia_ip, dia_ic, af11, &a, &kb, &mut ue, &mut b1, &mut b2, &mut dn, &mut vs, &mut vd, kdm[i], F::new(1.0f32), nk, nth, nspecx,
            nspecy, nu,
        );
        for r in 0..(nspec + nu - 1) / nu {
            let is = r * nu + t;
            if is < nspec {
                s_out[i * nspec + is] = vs[is];
                d_out[i * nspec + is] = vd[is];
            }
        }
    }
}

/// The tangent of [`dia_value()`] along `dva` (the DIA of the tangent kernel,
/// `snl1_tl`, with `k_mean depth` and `NLPROP` fixed): `S`, `D` and their
/// tangents.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn dia_tangent<F: Float>(
    va: &[F],
    dva: &[F],
    cg: &[F],
    kdm: &[F],
    par: &[F],
    sig: &[F],
    dia_ip: &[u32],
    dia_ic: &[u32],
    af11: &[F],
    s_out: &mut [F],
    d_out: &mut [F],
    ds_out: &mut [F],
    dd_out: &mut [F],
    np: usize,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
    #[comptime] nspecx: usize,
    #[comptime] nspecy: usize,
    #[comptime] nu: usize,
) {
    let i = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let nspec = nk * nth;
    if i < np {
        let zero = F::new(0.0f32);
        let mut a = Shared::<[F]>::new_slice(nspecy + 1);
        let mut ad = Shared::<[F]>::new_slice(nspecy + 1);
        let mut b1 = Shared::<[F]>::new_slice(nspecx + nth);
        let mut b2 = Shared::<[F]>::new_slice(nspecx + nth);
        let mut kb = Shared::<[F]>::new_slice(NKB * nk);
        let mut sa = Array::<F>::new((nspec + nu - 1) / nu);
        let mut sad = Array::<F>::new((nspec + nu - 1) / nu);
        let mut vs = Array::<F>::new((nspec + nu - 1) / nu);
        let mut vsd = Array::<F>::new((nspec + nu - 1) / nu);
        let mut vd = Array::<F>::new((nspec + nu - 1) / nu);
        let mut vdd = Array::<F>::new((nspec + nu - 1) / nu);
        for r in 0..(nspec + nu - 1) / nu {
            let is = r * nu + t;
            if is < nspec {
                a[is] = va[i * nspec + is];
                ad[is] = dva[i * nspec + is];
                vs[r] = zero;
                vsd[r] = zero;
                vd[r] = zero;
                vdd[r] = zero;
            }
        }
        if t < nk {
            kb[K_CG * nk + t] = cg[i * nk + t];
        }
        sync_cube();
        snl1_tl::<F>(
            par, sig, dia_ip, dia_ic, af11, &mut a, &mut ad, &kb, &mut sa, &mut sad, &mut b1, &mut b2, &mut vs, &mut vsd, &mut vd, &mut vdd, kdm[i], zero,
            zero, F::new(1.0f32), nk, nth, nspecx, nspecy, nu,
        );
        for r in 0..(nspec + nu - 1) / nu {
            let is = r * nu + t;
            if is < nspec {
                s_out[i * nspec + is] = vs[r];
                d_out[i * nspec + is] = vd[r];
                ds_out[i * nspec + is] = vsd[r];
                dd_out[i * nspec + is] = vdd[r];
            }
        }
    }
}

/// The adjoint of [`dia_value()`]: `abar = (∂S/∂A)ᵀ sbar + (∂D/∂A)ᵀ dbar` at
/// each node. `tb1s`, `tb1e` (and `tb2s`, `tb2e`) are the transposed index
/// tables of the gathers of `B1` (and `B2`) by `DIA_IC`: for each position
/// `x` of `B1`, the entries `is 8 + j` with `DIA_IC(2 j, is) = x`; `tps`,
/// `tpe` those of the gathers of `UE` by `DIA_IP`, by group of four
/// interaction points `g` (`ep1`, `em1`, `ep2`, `em2`): for each position `u`
/// of `UE`, the entries `isp 16 + k` with `DIA_IP(k, isp) = u` and
/// `k / 4 = g`, the start of group `g` and position `u` at
/// `tps[g (nspecy + nth + 1) + u]`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn dia_adjoint<F: Float>(
    va: &[F],
    cg: &[F],
    kdm: &[F],
    par: &[F],
    sig: &[F],
    dia_ip: &[u32],
    af11: &[F],
    sbar: &[F],
    dbar: &[F],
    tb1s: &[u32],
    tb1e: &[u32],
    tb2s: &[u32],
    tb2e: &[u32],
    tps: &[u32],
    tpe: &[u32],
    abar: &mut [F],
    np: usize,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
    #[comptime] nspecx: usize,
    #[comptime] nspecy: usize,
    #[comptime] nu: usize,
) {
    let i = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let nspec = nk * nth;
    if i < np {
        let zero = F::new(0.0f32);
        let one = F::new(1.0f32);
        let two = F::new(2.0f32);
        let nue = nspecy + nth;
        let mut ue = Shared::<[F]>::new_slice(nspecy + nth);
        let mut ueb = Shared::<[F]>::new_slice(nspecy + nth);
        let mut gs = Shared::<[F]>::new_slice(nspec);
        let mut gd = Shared::<[F]>::new_slice(nspec);
        let mut eb = Shared::<[F]>::new_slice(nspecx);
        let mut conx = Shared::<[F]>::new_slice(nk);
        let mut e00b = Array::<F>::new((nspecx + nu - 1) / nu);
        let mut ep1b = Array::<F>::new((nspecx + nu - 1) / nu);
        let mut em1b = Array::<F>::new((nspecx + nu - 1) / nu);
        let mut ep2b = Array::<F>::new((nspecx + nu - 1) / nu);
        let mut em2b = Array::<F>::new((nspecx + nu - 1) / nu);
        let mut acc = Array::<F>::new((nspecy + nth + nu - 1) / nu);
        if t < nk {
            conx[t] = tpiinv::<F>() / sig[t + 1] * cg[i * nk + t];
        }
        sync_cube();
        // the extended spectrum as in W3SNL1, and the cotangents of S (times
        // CON, scale 1) and of D
        for r in 0..(nue + nu - 1) / nu {
            let u = r * nu + t;
            if u < nue {
                let mut v = zero;
                if u >= nth && u < nspec + nth {
                    let is = u - nth;
                    v = va[i * nspec + is] / conx[is / nth];
                }
                ue[u] = v;
            }
        }
        for r in 0..(nspec + nu - 1) / nu {
            let is = r * nu + t;
            if is < nspec {
                gs[is] = conx[is / nth] * sbar[i * nspec + is];
                gd[is] = dbar[i * nspec + is];
            }
        }
        sync_cube();
        if t < nth {
            for ifr in nk..nspecy / nth {
                ue[nth + ifr * nth + t] = ue[ifr * nth + t] * par[P_FACHFE];
            }
        }
        sync_cube();
        let xv = par[P_KDCON] * kdm[i];
        let x = max(xv, par[P_KDMN]);
        let x2 = max(F::new(-1.0e15f32), par[P_SNLS3] * x);
        let cons = par[P_SNLC1] * (one + par[P_SNLS1] / x * (one - par[P_SNLS2] * x) * x2.exp());
        let dal1 = par[P_DAL1];
        let dal2 = par[P_DAL2];
        let dal3 = par[P_DAL3];
        // per bin of the interaction grid: the cotangents of B1, B2 in the
        // three passes and of DN, then those of E00, EP1, EM1, EP2, EM2
        for r in 0..(nspecx + nu - 1) / nu {
            let isp = r * nu + t;
            e00b[r] = zero;
            ep1b[r] = zero;
            em1b[r] = zero;
            ep2b[r] = zero;
            em2b[r] = zero;
            if isp < nspecx {
                let xb = isp + nth;
                let mut b10 = zero;
                let mut b11 = zero;
                let mut b12 = zero;
                let mut b20 = zero;
                let mut b21 = zero;
                let mut b22 = zero;
                if isp < nspec {
                    b10 = -two * gs[isp];
                    b20 = -two * gs[isp];
                }
                for q in tb1s[xb] as usize..tb1s[xb + 1] as usize {
                    let e = tb1e[q] as usize;
                    let is = e / 8;
                    let j = e - 8 * is;
                    b10 += par[P_AWG + j] * gs[is];
                    if j < 4 {
                        b11 += par[P_SWG + j] * gd[is];
                    } else {
                        b12 += par[P_SWG + j] * gd[is];
                    }
                }
                for q in tb2s[xb] as usize..tb2s[xb + 1] as usize {
                    let e = tb2e[q] as usize;
                    let is = e / 8;
                    let j = e - 8 * is;
                    b20 += par[P_AWG + j] * gs[is];
                    if j < 4 {
                        b21 += par[P_SWG + j] * gd[is];
                    } else {
                        b22 += par[P_SWG + j] * gd[is];
                    }
                }
                let e00 = ue[isp + nth];
                let w0 = par[P_AWG];
                let w1 = par[P_AWG + 1];
                let w2 = par[P_AWG + 2];
                let w3 = par[P_AWG + 3];
                let w4 = par[P_AWG + 4];
                let w5 = par[P_AWG + 5];
                let w6 = par[P_AWG + 6];
                let w7 = par[P_AWG + 7];
                let ep1 = w0 * ue[dia_ip[isp] as usize]
                    + w1 * ue[dia_ip[nspecx + isp] as usize]
                    + w2 * ue[dia_ip[2 * nspecx + isp] as usize]
                    + w3 * ue[dia_ip[3 * nspecx + isp] as usize];
                let em1 = w4 * ue[dia_ip[4 * nspecx + isp] as usize]
                    + w5 * ue[dia_ip[5 * nspecx + isp] as usize]
                    + w6 * ue[dia_ip[6 * nspecx + isp] as usize]
                    + w7 * ue[dia_ip[7 * nspecx + isp] as usize];
                let ep2 = w0 * ue[dia_ip[8 * nspecx + isp] as usize]
                    + w1 * ue[dia_ip[9 * nspecx + isp] as usize]
                    + w2 * ue[dia_ip[10 * nspecx + isp] as usize]
                    + w3 * ue[dia_ip[11 * nspecx + isp] as usize];
                let em2 = w4 * ue[dia_ip[12 * nspecx + isp] as usize]
                    + w5 * ue[dia_ip[13 * nspecx + isp] as usize]
                    + w6 * ue[dia_ip[14 * nspecx + isp] as usize]
                    + w7 * ue[dia_ip[15 * nspecx + isp] as usize];
                let caf = cons * af11[isp];
                let factor = caf * e00;
                let sa1a = e00 * (ep1 * dal1 + em1 * dal2);
                let sa1b = sa1a - ep1 * em1 * dal3;
                let sa2a = e00 * (ep2 * dal1 + em2 * dal2);
                let sa2b = sa2a - ep2 * em2 * dal3;
                let p1 = dal1 * e00 - dal3 * em1;
                let p2 = dal1 * e00 - dal3 * em2;
                let m1 = dal2 * e00 - dal3 * ep1;
                let m2 = dal2 * e00 - dal3 * ep2;
                // pass 0: B1 = FACTOR SA1B, B2 = FACTOR SA2B
                let mut ge = b10 * (caf * sa1b + factor * (ep1 * dal1 + em1 * dal2)) + b20 * (caf * sa2b + factor * (ep2 * dal1 + em2 * dal2));
                let mut gp1 = b10 * factor * (e00 * dal1 - em1 * dal3);
                let mut gm1 = b10 * factor * (e00 * dal2 - ep1 * dal3);
                let mut gp2 = b20 * factor * (e00 * dal1 - em2 * dal3);
                let mut gm2 = b20 * factor * (e00 * dal2 - ep2 * dal3);
                // pass 1: B1 = FACTOR P1, B2 = FACTOR P2, and DN
                ge += b11 * (caf * p1 + factor * dal1) + b21 * (caf * p2 + factor * dal1);
                gm1 += -b11 * factor * dal3;
                gm2 += -b21 * factor * dal3;
                if isp < nspec {
                    let gdn = -two * caf * gd[isp];
                    ge += gdn * two * (ep1 * dal1 + em1 * dal2 + ep2 * dal1 + em2 * dal2);
                    gp1 += gdn * (two * e00 * dal1 - em1 * dal3);
                    gm1 += gdn * (two * e00 * dal2 - ep1 * dal3);
                    gp2 += gdn * (two * e00 * dal1 - em2 * dal3);
                    gm2 += gdn * (two * e00 * dal2 - ep2 * dal3);
                }
                // pass 2: B1 = FACTOR M1, B2 = FACTOR M2
                ge += b12 * (caf * m1 + factor * dal2) + b22 * (caf * m2 + factor * dal2);
                gp1 += -b12 * factor * dal3;
                gp2 += -b22 * factor * dal3;
                e00b[r] = ge;
                ep1b[r] = gp1;
                em1b[r] = gm1;
                ep2b[r] = gp2;
                em2b[r] = gm2;
            }
        }
        // the cotangent of UE, by gathers: E00 at UE(ISP), then the four
        // groups of interaction points through the transposed DIA_IP
        for r in 0..(nue + nu - 1) / nu {
            acc[r] = zero;
        }
        for qg in 0..5u32 {
            for r in 0..(nspecx + nu - 1) / nu {
                let isp = r * nu + t;
                if isp < nspecx {
                    let mut v = e00b[r];
                    if qg == 1u32 {
                        v = ep1b[r];
                    } else if qg == 2u32 {
                        v = em1b[r];
                    } else if qg == 3u32 {
                        v = ep2b[r];
                    } else if qg == 4u32 {
                        v = em2b[r];
                    }
                    eb[isp] = v;
                }
            }
            sync_cube();
            for r in 0..(nue + nu - 1) / nu {
                let u = r * nu + t;
                if u < nue {
                    if qg == 0u32 {
                        if u >= nth && u - nth < nspecx {
                            acc[r] += eb[u - nth];
                        }
                    } else {
                        let g = (qg - 1u32) as usize;
                        let s0 = tps[g * (nue + 1) + u] as usize;
                        let s1 = tps[g * (nue + 1) + u + 1] as usize;
                        for q in s0..s1 {
                            let e = tpe[q] as usize;
                            let isp = e / 16;
                            let k = e - 16 * isp;
                            acc[r] += par[P_AWG + (k % 8)] * eb[isp];
                        }
                    }
                }
            }
            sync_cube();
        }
        for r in 0..(nue + nu - 1) / nu {
            let u = r * nu + t;
            if u < nue {
                ueb[u] = acc[r];
            }
        }
        sync_cube();
        // the extension of UE above the grid, in reverse
        if t < nth {
            for jj in 0..(nspecy / nth - nk) {
                let ifr = nspecy / nth - 1 - jj;
                let up = ueb[nth + ifr * nth + t];
                ueb[ifr * nth + t] += par[P_FACHFE] * up;
            }
        }
        sync_cube();
        for r in 0..(nspec + nu - 1) / nu {
            let is = r * nu + t;
            if is < nspec {
                abar[i * nspec + is] = ueb[is + nth] / conx[is / nth];
            }
        }
    }
}

pub use crate::dia_adj::DiaAdjTables;

/// A small generator of reproducible random numbers in `[lo, hi)`.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn vec(&mut self, n: usize, lo: f64, hi: f64) -> Vec<f64> {
        (0..n).map(|_| lo + (hi - lo) * self.next()).collect()
    }
}

/// Inner product.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The DIA prototype on `n` nodes (spectra `va[i nspec + is]`, `cg[i nk + ik]`,
/// `kdm[i]`, all f64): value, tangent along a random spectrum, hand-written
/// adjoint of random cotangents of `S` and `D`, the dot-product test and the
/// times per launch (the mean of `reps` launches after one that compiles).
/// Returns the JSON report and, for [`crate::burn_m4`], the value `(S, D)`,
/// the tangent direction, the cotangents, the adjoint and the tangent
/// `(dS, dD)`.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn dia_check(
    client: &cubecl::client::Client,
    k: &SourceKernel<f64>,
    t: &SrceTables,
    va: &[f64],
    cg: &[f64],
    kdm: &[f64],
    n: usize,
    nk: usize,
    nth: usize,
    reps: usize,
) -> anyhow::Result<(serde_json::Value, [Vec<f64>; 8])> {
    use cubecl::prelude::CubeElement;
    let nspec = nk * nth;
    let (nspecx, nspecy, nu) = (k.nspecx, k.nspecy, k.nu);
    let up = |v: &[f64]| client.create_from_slice(f64::as_bytes(v));
    let upu = |v: &[u32]| client.create_from_slice(u32::as_bytes(v));
    let down = |h: &cubecl::server::Handle| -> Vec<f64> { f64::from_bytes(&client.read_one(h.clone()).expect("device read")).to_vec() };
    let sync = || futures_lite::future::block_on(client.sync()).expect("device sync");
    let tab = DiaAdjTables::new(&t.dia_ip, &t.dia_ic, nspec, nspecx, nspecy, nth);
    let mut rng = Lcg(20261126);
    let dva: Vec<f64> = rng.vec(n * nspec, -1.0, 1.0).iter().zip(va).map(|(r, a)| r * a).collect();
    let sbar = rng.vec(n * nspec, -1.0, 1.0);
    let dbar = rng.vec(n * nspec, -1.0, 1.0);
    let (d_va, d_dva, d_cg, d_kdm, d_sbar, d_dbar) = (up(va), up(&dva), up(cg), up(kdm), up(&sbar), up(&dbar));
    let z = || client.empty(n * nspec * 8);
    let (d_s, d_d, d_s2, d_d2, d_ds, d_dd, d_ab) = (z(), z(), z(), z(), z(), z(), z());
    let (d_tb1s, d_tb1e, d_tb2s, d_tb2e, d_tps, d_tpe) =
        (upu(&tab.tb1s), upu(&tab.tb1e), upu(&tab.tb2s), upu(&tab.tb2e), upu(&tab.tps), upu(&tab.tpe));
    let ns = n * nspec;
    let cc = || CubeCount::Static(n as u32, 1, 1);
    let cd = CubeDim::new_1d(nu as u32);
    let value = || unsafe {
        dia_value::launch_unchecked::<f64>(
            client,
            cc(),
            cd,
            bf(&d_va, ns),
            bf(&d_cg, n * nk),
            bf(&d_kdm, n),
            bf(&k.par, NPAR),
            bf(&k.sig, nk + 2),
            bf(&k.dia_ip, 16 * nspecx),
            bf(&k.dia_ic, 16 * nspec),
            bf(&k.af11, nspecx),
            bf(&d_s, ns),
            bf(&d_d, ns),
            n,
            nk,
            nth,
            nspecx,
            nspecy,
            nu,
        );
    };
    let tangent = || unsafe {
        dia_tangent::launch_unchecked::<f64>(
            client,
            cc(),
            cd,
            bf(&d_va, ns),
            bf(&d_dva, ns),
            bf(&d_cg, n * nk),
            bf(&d_kdm, n),
            bf(&k.par, NPAR),
            bf(&k.sig, nk + 2),
            bf(&k.dia_ip, 16 * nspecx),
            bf(&k.dia_ic, 16 * nspec),
            bf(&k.af11, nspecx),
            bf(&d_s2, ns),
            bf(&d_d2, ns),
            bf(&d_ds, ns),
            bf(&d_dd, ns),
            n,
            nk,
            nth,
            nspecx,
            nspecy,
            nu,
        );
    };
    let adjoint = || unsafe {
        dia_adjoint::launch_unchecked::<f64>(
            client,
            cc(),
            cd,
            bf(&d_va, ns),
            bf(&d_cg, n * nk),
            bf(&d_kdm, n),
            bf(&k.par, NPAR),
            bf(&k.sig, nk + 2),
            bf(&k.dia_ip, 16 * nspecx),
            bf(&k.af11, nspecx),
            bf(&d_sbar, ns),
            bf(&d_dbar, ns),
            bf(&d_tb1s, tab.tb1s.len()),
            bf(&d_tb1e, tab.tb1e.len()),
            bf(&d_tb2s, tab.tb2s.len()),
            bf(&d_tb2e, tab.tb2e.len()),
            bf(&d_tps, tab.tps.len()),
            bf(&d_tpe, tab.tpe.len()),
            bf(&d_ab, ns),
            n,
            nk,
            nth,
            nspecx,
            nspecy,
            nu,
        );
    };
    let time = |f: &dyn Fn()| -> (f64, f64) {
        let t0 = std::time::Instant::now();
        f();
        sync();
        let first = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            f();
        }
        sync();
        (first, t1.elapsed().as_secs_f64() / reps as f64)
    };
    let tv = time(&value);
    let tt = time(&tangent);
    let ta = time(&adjoint);
    let (s, d, s2, d2, ds, dd, ab) = (down(&d_s), down(&d_d), down(&d_s2), down(&d_d2), down(&d_ds), down(&d_dd), down(&d_ab));
    let rel = |a: &[f64], b: &[f64]| {
        let m = b.iter().fold(0f64, |m, x| m.max(x.abs())).max(1e-300);
        a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs())) / m
    };
    let lhs = dot(&ds, &sbar) + dot(&dd, &dbar);
    let rhs = dot(&dva, &ab);
    let scale = (dot(&ds, &ds) + dot(&dd, &dd)).sqrt() * (dot(&sbar, &sbar) + dot(&dbar, &dbar)).sqrt();
    // shared memory per node of the three kernels (f64)
    let f = 8usize;
    let smem_value = (4 * nspec + (nspecy + nth) + 2 * (nspecx + nth) + NKB * nk) * f;
    let smem_tangent = (2 * (nspecy + 1) + 2 * (nspecx + nth) + NKB * nk) * f;
    let smem_adjoint = (2 * (nspecy + nth) + 2 * nspec + nspecx + nk) * f;
    println!("DIA (W3SNL1 alone), f64, {n} nodes, {nu} units per cube:");
    println!(
        "  time per launch (mean of {reps}; first launch with compilation): value {:.4} s ({:.2}), tangent {:.4} s ({:.2}), hand-written adjoint {:.4} s ({:.2})",
        tv.1, tv.0, tt.1, tt.0, ta.1, ta.0
    );
    println!("  adjoint / value {:.2}, tangent / value {:.2}", ta.1 / tv.1, tt.1 / tv.1);
    println!("  shared memory per node: value {smem_value} B, tangent {smem_tangent} B, adjoint {smem_adjoint} B; transposed tables {} entries", tab.tb1e.len() + tab.tb2e.len() + tab.tpe.len());
    println!("  value of the tangent kernel against the value kernel: S {:.3e}, D {:.3e} (relative to the largest)", rel(&s2, &s), rel(&d2, &d));
    println!("  dot-product test: <L x, y> = {lhs:.15e}, <x, Lt y> = {rhs:.15e}, relative difference {:.3e} (of |Lx| |y|: {:.3e})", (lhs - rhs).abs() / lhs.abs().max(1e-300), (lhs - rhs).abs() / scale.max(1e-300));
    #[allow(unused_mut)]
    let mut rep = serde_json::json!({
        "nodes": n, "units": nu, "reps": reps,
        "seconds": {"value": tv.1, "tangent": tt.1, "adjoint": ta.1, "first": [tv.0, tt.0, ta.0]},
        "shared_bytes_per_node": {"value": smem_value, "tangent": smem_tangent, "adjoint": smem_adjoint},
        "dot": {"lhs": lhs, "rhs": rhs, "rel": (lhs - rhs).abs() / lhs.abs().max(1e-300)},
        "tangent_value_vs_value": [rel(&s2, &s), rel(&d2, &d)],
    });
    #[cfg(feature = "cuda-c")]
    {
        let mut cu = vec![];
        for minb in [1u32, 2] {
            let (abc, tc, regs) = crate::cuda_m4::dia_adjoint_cuda(t, &tab, va, cg, kdm, &sbar, &dbar, n, nk, nth, minb, reps)?;
            let rc = dot(&dva, &abc);
            println!(
                "  CUDA C adjoint (__launch_bounds__(128, {minb}), {regs} registers): {:.4} s per launch (adjoint / CubeCL value {:.2}); against the CubeCL adjoint {:.3e}; dot-product test {:.3e}",
                tc,
                tc / tv.1,
                rel(&abc, &ab),
                (lhs - rc).abs() / lhs.abs().max(1e-300)
            );
            cu.push(serde_json::json!({"minb": minb, "regs": regs, "seconds": tc, "rel_vs_cubecl": rel(&abc, &ab), "dot_rel": (lhs - rc).abs() / lhs.abs().max(1e-300)}));
        }
        rep["cuda_c"] = serde_json::Value::Array(cu);
    }
    Ok((rep, [s, d, dva, sbar, dbar, ab, ds, dd]))
}

// ---------------------------------------------------------------------------
// One geographic sub-step (`geo_tri_x4` then `geo_node_x4`) and its adjoint
// ---------------------------------------------------------------------------

/// Adjoint of a sub-step, first node pass (four directions per unit, as
/// [`crate::prop::geo_node_x4()`]): with `ut` the triangle values of the
/// forward input `u` (from [`crate::prop::geo_tri_x4()`]) and `ybar` the
/// cotangent of the output, `stb` the cotangent of the upwind sum `ST` of the
/// node and `ub` that of `u` at the node through the node's own value (its
/// value and `ST`); the triangle values are taken by
/// [`geo_adj_tri()`] and [`geo_adj_node2()`].
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn geo_adj_node1<F: Float>(
    u: &[F],
    ut: &[F],
    ybar: &[F],
    ub: &mut [F],
    stb: &mut [F],
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
    np: usize,
    ne: usize,
    ik: usize,
    sub: u32,
    #[comptime] nth: usize,
) {
    let g = ABSOLUTE_POS;
    let nw = nth / 4;
    if g < np * nw {
        let i = g / nw;
        let k0 = (g - i * nw) * 4;
        let base = i * nth + k0;
        let zero = F::new(0.0f32);
        let one = F::new(1.0f32);
        let cb = ik * ne * 12;
        let q0 = start[i] as usize;
        let q1 = start[i + 1] as usize;
        let wlo = mlo[i];
        let whi = mhi[i];
        let sa = si_inv[i] * (one - F::cast_from((whi >> 31u32) & 1u32));
        let bnd = bidx[i] != 0xffff_ffffu32;
        let kk = ik * nth + k0;
        for d in 0..4usize {
            let k = k0 + d;
            let c = ecos[k];
            let s = esin[k];
            let ui = u[base + d];
            let mut st = zero;
            let mut ks = zero;
            for q in q0..q1 {
                let code = slot[q] as usize;
                let t = code / 3;
                let p = code - 3 * t;
                let b = cb + t * 12 + p;
                let kp = max(coef[b] * c + coef[b + 3] * s + ccoef[t * 6 + p], zero);
                st += kp * (ui - ut[t * nth + k]);
                ks += kp;
            }
            let dt = dtk[kk + d];
            let m = crate::prop::mask_bit::<F>(wlo, whi, k);
            let active = sub < iter[kk + d];
            let x = ui - dt * sa * st;
            let y = ybar[base + d];
            // V = max(0, U - DT SA ST) M on the active interior directions
            let w = select(active && !bnd && x > zero, y * m, zero);
            let direct = select(active, select(bnd, zero, w), y);
            let sb = -dt * sa * w;
            stb[base + d] = sb;
            ub[base + d] = direct + sb * ks;
        }
    }
}

/// Adjoint of a sub-step, triangle pass: the cotangent of `Ũ_T` from the
/// cotangents of `ST` of its three nodes, `utb[t nth + k] = -Σ_p K+_p stb`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn geo_adj_tri<F: Float>(
    stb: &[F],
    utb: &mut [F],
    coef: &[F],
    ccoef: &[F],
    ecos: &[F],
    esin: &[F],
    ine: &[u32],
    ne: usize,
    ik: usize,
    #[comptime] nth: usize,
) {
    let g = ABSOLUTE_POS;
    let nw = nth / 4;
    if g < ne * nw {
        let t = g / nw;
        let k0 = (g - t * nw) * 4;
        let zero = F::new(0.0f32);
        let b = (ik * ne + t) * 12;
        let r1 = ine[3 * t] as usize * nth;
        let r2 = ine[3 * t + 1] as usize * nth;
        let r3 = ine[3 * t + 2] as usize * nth;
        for d in 0..4usize {
            let k = k0 + d;
            let c = ecos[k];
            let s = esin[k];
            let k1 = max(coef[b] * c + coef[b + 3] * s + ccoef[t * 6], zero);
            let k2 = max(coef[b + 1] * c + coef[b + 4] * s + ccoef[t * 6 + 1], zero);
            let k3 = max(coef[b + 2] * c + coef[b + 5] * s + ccoef[t * 6 + 2], zero);
            utb[t * nth + k] = -(k1 * stb[r1 + k] + k2 * stb[r2 + k] + k3 * stb[r3 + k]);
        }
    }
}

/// Adjoint of a sub-step, second node pass: `abar = ub + Σ_T NM_T FA_p(T)
/// utb_T` over the triangles `T` of the node (vertex `p` of `T`), the
/// coefficients of `Ũ_T` as in [`crate::prop::geo_tri_x4()`].
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn geo_adj_node2<F: Float>(
    ub: &[F],
    utb: &[F],
    abar: &mut [F],
    coef: &[F],
    ccoef: &[F],
    ecos: &[F],
    esin: &[F],
    start: &[u32],
    slot: &[u32],
    np: usize,
    ne: usize,
    ik: usize,
    #[comptime] nth: usize,
) {
    let g = ABSOLUTE_POS;
    let nw = nth / 4;
    if g < np * nw {
        let i = g / nw;
        let k0 = (g - i * nw) * 4;
        let base = i * nth + k0;
        let zero = F::new(0.0f32);
        let one = F::new(1.0f32);
        let thr = F::new(1.1754944e-38f32);
        let cb = ik * ne * 12;
        let q0 = start[i] as usize;
        let q1 = start[i + 1] as usize;
        for d in 0..4usize {
            let k = k0 + d;
            let c = ecos[k];
            let s = esin[k];
            let mut acc = ub[base + d];
            for q in q0..q1 {
                let code = slot[q] as usize;
                let t = code / 3;
                let p = code - 3 * t;
                let b = cb + t * 12;
                let k1 = coef[b] * c + coef[b + 3] * s + ccoef[t * 6];
                let k2 = coef[b + 1] * c + coef[b + 4] * s + ccoef[t * 6 + 1];
                let k3 = coef[b + 2] * c + coef[b + 5] * s + ccoef[t * 6 + 2];
                let kn = min(k1, zero) + min(k2, zero) + min(k3, zero);
                let nm = -one / min(-thr, kn);
                let kpp = max(coef[b + p] * c + coef[b + 3 + p] * s + ccoef[t * 6 + p], zero);
                let fa = coef[b + 6 + p] * c + coef[b + 9 + p] * s + ccoef[t * 6 + 3 + p] + kpp;
                acc += nm * fa * utb[t * nth + k];
            }
            abar[base + d] = acc;
        }
    }
}
