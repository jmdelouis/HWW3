//! CubeCL kernel of the source terms (milestone M2a): `W3SRCE` called with
//! `srce_direct` for the HWW3 configuration with the simplified ST4
//! (`W3SPR4`, `CALC_USTAR` by Newton iterations, `W3SIN4` with the tail
//! stress in line, `W3SLN1`, `W3SNL1`, `W3SDS4` without the cumulative
//! term, `W3SDB1`, `W3SBT4`), transcribed from the CPU references of
//! [`crate::sources`].
//!
//! One cube per node: the spectrum and the work arrays sit in shared memory,
//! the units of the cube share the spectral loops, and the scalar parts
//! (mean parameters, u*, the tail stress, the dynamic step) run on unit 0
//! with the sums in the order of WW3. The whole integration of a global step
//! (all dynamic sub-steps) runs in one launch; nodes whose `MAPSTA` is not 1
//! are skipped, as in `W3WAVE`.
//!
//! Shared arrays of a cube: the spectrum `a`, the sums `vs`, `vd` of the
//! source terms and their diagonals, the wind-sea flags `ws`, the extended
//! spectrum `ue` and the interaction arrays `b1`, `b2`, `dn` of the DIA
//! (`b1` also holds the saturation of `W3SDS4`), per-frequency values `kb`,
//! per-direction partial sums `dr`, scalars `sc` and a reduction buffer.

// The `#[cube]` macro generates, for every kernel, a launch struct and
// functions that carry no documentation; the kernels themselves are
// documented below.
#![allow(missing_docs)]

use crate::sources::*;
use cubecl::prelude::*;

// Scalar slots of `sc`.
const S_EMEAN: usize = 0;
const S_FMEAN: usize = 1;
const S_FMEAN1: usize = 2;
const S_WNMEAN: usize = 3;
const S_AMAX: usize = 4;
const S_USTAR: usize = 5;
const S_Z0: usize = 6;
const S_CHARN: usize = 7;
const S_FMEANWS: usize = 8;
const S_TAUWX: usize = 9;
const S_TAUWY: usize = 10;
const S_FHIGH: usize = 11;
const S_DTTOT: usize = 12;
const S_DT: usize = 13;
const S_SHAVE: usize = 14;
const S_COSU: usize = 17;
const S_SINU: usize = 18;
const S_USTP: usize = 19;
const S_UORB: usize = 20;
const S_FW: usize = 21;
const S_FU: usize = 22;
const S_FUD: usize = 23;
const S_PTURB: usize = 24;
const S_PVISC: usize = 25;
const S_SX: usize = 26;
const S_SY: usize = 27;
const S_SNX: usize = 28;
const S_SNY: usize = 29;
const S_CONS: usize = 30;
const S_NKH: usize = 31;
const S_FBT: usize = 32;
const S_DTDYN: usize = 33;
const S_NSTEPS: usize = 34;
const S_DONE: usize = 35;
const S_TAUX: usize = 36;
const S_TAUY: usize = 37;
const S_SCALE: usize = 38;
const NSC: usize = 40;

// Per-frequency slots of `kb` (`kb[slot nk + ik]`).
const K_WN: usize = 0;
const K_CG: usize = 1;
const K_DAM: usize = 2;
const K_EB: usize = 3;
const K_EB2: usize = 4;
const K_EBX: usize = 5;
const K_EBY: usize = 6;
const K_CBETA: usize = 7;
const K_BTH0: usize = 8;
const K_AMAX: usize = 9;
const NKB: usize = 10;

/// `1 / (2 pi)` as WW3's single-precision `TPIINV`.
#[cube]
fn tpiinv<F: Float>() -> F {
    F::new(0.15915494f32)
}

/// `2 pi` as WW3's `TPI`.
#[cube]
fn tpi<F: Float>() -> F {
    F::new(6.2831855f32)
}

/// `LOG10(x)`.
#[cube]
fn log10<F: Float>(x: F) -> F {
    x.ln() * F::new(0.4342945f32)
}

/// Interpolation in a table on a uniform axis as WW3 does it: index
/// `MIN(SIZE - 1, INT(x))`, weight clipped at 1 (`x >= 0`).
#[cube]
fn table<F: Float>(tab: &[F], n: usize, x: F) -> F {
    let size = n - 1;
    let xi = u32::cast_from(max(x.trunc(), F::new(0.0f32))) as usize;
    let ind = select(xi < size - 1, xi, size - 1);
    let deli1 = min(F::new(1.0f32), x - F::cast_from(ind as u32));
    tab[ind] * (F::new(1.0f32) - deli1) + tab[ind + 1] * deli1
}

/// `CALC_USTAR` with `SINTABLE = 0`: u* by 10 Newton iterations on the
/// Charnock relation with the wave-supported stress `tauw`; writes u*, z0
/// and the Charnock coefficient into `sc`.
#[cube]
fn calc_ustar<F: Float>(par: &[F], taut: &[F], sc: &mut Shared<[F]>, windspeed: F, tauw: F) {
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let two = F::new(2.0f32);
    let grav = par[P_GRAV];
    let kappa = par[P_KAPPA];
    let mut chath = par[P_AALPHA];
    let capped = par[P_CAPCHNK] == one;
    let mut ust = zero;
    if par[P_SINTABLE] > zero {
        // SINTABLE = 1: bilinear interpolation in TAUT
        let itm = u32::cast_from(par[P_ITAUMAX]) as usize;
        let jum = u32::cast_from(par[P_JUMAX]) as usize;
        let tl = max(min(tauw, par[P_TAUWMAX]), zero);
        let xi = tl.sqrt() / par[P_DELTAUW];
        let xif = u32::cast_from(xi.trunc()) as usize;
        let ind = select(xif < itm - 1, xif, itm - 1);
        let deli1 = min(one, xi - F::cast_from(ind as u32));
        let deli2 = one - deli1;
        let xj = windspeed / par[P_DELU];
        let xjf = u32::cast_from(xj.trunc()) as usize;
        let j = select(xjf < jum - 1, xjf, jum - 1);
        let delj1 = min(one, xj - F::cast_from(j as u32));
        let delj2 = one - delj1;
        let n = itm + 1;
        ust = (taut[ind + j * n] * deli2 + taut[ind + 1 + j * n] * deli1) * delj2
            + (taut[ind + (j + 1) * n] * deli2 + taut[ind + 1 + (j + 1) * n] * deli1) * delj1;
    } else {
        let mut xmin = zero;
        if capped {
            let c1 = par[P_CAPCHNK + 1];
            let c2 = par[P_CAPCHNK + 2];
            chath = c1 + F::new(0.5f32) * (c2 - c1) * (one - ((windspeed - par[P_CAPCHNK + 3]) / par[P_CAPCHNK + 4]).tanh());
            xmin = F::new(0.15f32) * (c2 - chath);
        }
        let eps1 = F::new(0.00001f32);
        let xm = F::new(0.5f32);
        let wcd = F::new(0.0012875f32).sqrt();
        let ustold = windspeed * wcd;
        let mut tauold = max(ustold * ustold, tauw + eps1);
        let zz0max = par[P_ZZ0MAX];
        for _ in 0..10u32 {
            let x = max(tauw / tauold, xmin);
            ust = tauold.sqrt();
            let mut zz00 = chath * tauold / grav;
            if zz0max != zero {
                zz00 = min(zz00, zz0max);
            }
            let mut zz0 = zz00 / (one - x).powf(xm);
            let znu = F::new(0.11f32) * par[P_NU_AIR] / max(ust, F::new(1.0e-6f32));
            zz0 += par[P_SINTAILPAR + 4] * znu;
            let l = (par[P_ZZWND] / zz0).ln();
            let f = ust - kappa * windspeed / l;
            let delf = one - kappa * windspeed / (l * l) * two / ust * (one - (xm + one) * x) / (one - x);
            ust -= f / delf;
            tauold = max(ust * ust, tauw + eps1);
        }
    }
    let mut z0 = chath * F::new(0.000001f32) / grav;
    let mut charn = chath;
    if ust > F::new(0.001f32) {
        let sqrtcdm1 = min(windspeed / ust, F::new(100.0f32));
        z0 = par[P_ZZWND] * (-kappa * sqrtcdm1).exp();
        charn = grav * z0 / (ust * ust);
    } else if ust > zero {
        z0 = par[P_ZZWND] * (-kappa * min(windspeed / ust, F::new(100.0f32))).exp();
    }
    if capped {
        charn = max(min(charn, F::new(0.09f32)), chath);
    }
    sc[S_USTAR] = ust;
    sc[S_Z0] = z0;
    sc[S_CHARN] = charn;
}

/// `W3SPR4`: mean parameters of the spectrum `a` with the wind-sea flags
/// `ws`, then u*, z0 and Charnock from the wind `u` and the wave-supported
/// stress (`tauwx`, `tauwy`). Ends with a cube synchronisation.
#[cube]
fn spr4<F: Float>(
    par: &[F],
    taut: &[F],
    sig: &[F],
    dden: &[F],
    a: &Shared<[F]>,
    ws: &Shared<[F]>,
    kb: &mut Shared<[F]>,
    sc: &mut Shared<[F]>,
    u: F,
    tauwx: F,
    tauwy: F,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
) {
    let t = UNIT_POS as usize;
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    if t < nk {
        let mut eb = zero;
        let mut eb2 = zero;
        let mut amax = zero;
        for th in 0..nth {
            let x = a[t * nth + th];
            eb += x;
            eb2 += x * ws[t * nth + th];
            amax = max(amax, x);
        }
        kb[K_EB * nk + t] = eb;
        kb[K_EB2 * nk + t] = eb2;
        kb[K_AMAX * nk + t] = amax;
    }
    sync_cube();
    if t == 0 {
        let tw = par[P_WWNMEANPTAIL];
        let wp = par[P_WWNMEANP];
        let mut emean = zero;
        let mut fmean = zero;
        let mut fmean1 = zero;
        let mut wnmean = zero;
        let mut emeanws = zero;
        let mut fmeanws = zero;
        let mut amax = zero;
        for ik in 0..nk {
            let f = dden[ik] / kb[K_CG * nk + ik];
            let eb = kb[K_EB * nk + ik] * f;
            let eb2 = kb[K_EB2 * nk + ik] * f;
            let s = sig[ik + 1];
            emean += eb;
            fmean += eb / s;
            fmean1 += eb * s.powf(F::new(2.0f32) * tw);
            wnmean += eb * kb[K_WN * nk + ik].powf(wp);
            emeanws += eb2;
            fmeanws += eb2 * s.powf(F::new(2.0f32) * tw);
            amax = max(amax, kb[K_AMAX * nk + ik]);
        }
        let ddl = dden[nk - 1];
        let cgl = kb[K_CG * nk + nk - 1];
        let eband = kb[K_EB * nk + nk - 1] * (ddl / cgl) / ddl;
        emean += eband * par[P_FTE];
        fmean += eband * par[P_FTF];
        fmean1 += eband * par[P_SSTXFTFTAIL];
        wnmean += eband * par[P_SSTXFTWN];
        let eband2 = kb[K_EB2 * nk + nk - 1] * (ddl / cgl) / ddl;
        emeanws += eband2 * par[P_FTE];
        fmeanws += eband2 * par[P_SSTXFTFTAIL];
        let e7 = F::new(1.0e-7f32);
        fmean = tpiinv::<F>() * emean / max(e7, fmean);
        if fmean1 < e7 {
            fmean1 = tpiinv::<F>() * sig[nk];
        } else {
            fmean1 = tpiinv::<F>() * (max(e7, fmean1) / max(e7, emean)).powf(one / (F::new(2.0f32) * tw));
        }
        wnmean = (max(e7, wnmean) / max(e7, emean)).powf(one / wp);
        if fmeanws < e7 || emeanws < e7 {
            fmeanws = tpiinv::<F>() * sig[nk];
        } else {
            fmeanws = tpiinv::<F>() * (max(e7, fmeanws) / max(e7, emeanws)).powf(one / (F::new(2.0f32) * tw));
        }
        sc[S_EMEAN] = emean;
        sc[S_FMEAN] = fmean;
        sc[S_FMEAN1] = fmean1;
        sc[S_WNMEAN] = wnmean;
        sc[S_AMAX] = amax;
        sc[S_FMEANWS] = fmeanws;
        let tauw = (tauwx * tauwx + tauwy * tauwy).sqrt();
        calc_ustar::<F>(par, taut, sc, u, tauw);
    }
    sync_cube();
}

/// `W3SIN4` with `SINTABLE = 0`: wind input with sheltering and swell
/// damping, the stress of the high-frequency tail integrated in line. Sets
/// the wind-sea flags `ws` and the wave-supported stress in `sc`; with
/// `accumulate`, adds `scale S` and `scale D` to `vs` and `vd`. Ends with a
/// cube synchronisation.
#[cube]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn sin4<F: Float>(
    par: &[F],
    tauhf: &[F],
    sig: &[F],
    dden: &[F],
    ecos: &[F],
    esin: &[F],
    sig2k: &[F],
    dden2k: &[F],
    fwtab: &[F],
    a: &Shared<[F]>,
    vs: &mut Shared<[F]>,
    vd: &mut Shared<[F]>,
    ws: &mut Shared<[F]>,
    kb: &mut Shared<[F]>,
    sc: &mut Shared<[F]>,
    dr: &mut Shared<[F]>,
    ustar: F,
    z0: F,
    usdir: F,
    scale: F,
    accumulate: bool,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
) {
    let t = UNIT_POS as usize;
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let half = F::new(0.5f32);
    let grav = par[P_GRAV];
    let kappa = par[P_KAPPA];
    let nu_air = par[P_NU_AIR];
    let drat = par[P_DRAT];
    let const1 = par[P_BBETA] / (kappa * kappa);
    let const0 = const1 * drat;
    let tsh = par[P_TTAUWSHELTER].abs();
    if t < nk {
        let mut eb = zero;
        for th in 0..nth {
            eb += a[t * nth + th];
        }
        kb[K_EB * nk + t] = eb;
    }
    sync_cube();
    if t == 0 {
        let mut uorb = zero;
        let mut aorb = zero;
        for ik in 0..nk {
            let eb = kb[K_EB * nk + ik];
            let s = sig[ik + 1];
            let c = kb[K_CG * nk + ik];
            uorb += eb * s * s * dden[ik] / c;
            aorb += eb * dden[ik] / c;
        }
        uorb = F::new(2.0f32) * uorb.sqrt();
        let aorb1 = F::new(2.0f32) * aorb.powf(one - half * par[P_SSWELLF + 5]);
        let re = F::new(4.0f32) * uorb * aorb1 / nu_air;
        let mut pturb = one;
        let mut pvisc = one;
        if par[P_SSWELLF + 3] > zero {
            if par[P_SSWELLF + 6] > zero {
                let smooth = half * ((re - par[P_SSWELLF + 3]) / par[P_SSWELLF + 6]).tanh();
                pturb = half + smooth;
                pvisc = half - smooth;
            } else if re <= par[P_SSWELLF + 3] {
                pturb = zero;
                pvisc = one;
            } else {
                pturb = one;
                pvisc = zero;
            }
        }
        let mut fw = max(par[P_SSWELLF + 2].abs(), zero);
        let mut fu = zero;
        let mut fud = zero;
        if par[P_SSWELLF + 1] != zero {
            fu = par[P_SSWELLF + 2].abs();
            fud = par[P_SSWELLF + 1];
            let z0visc = F::new(0.1f32) * nu_air / max(ustar, F::new(0.0001f32));
            let z0noz = max(z0visc, par[P_ZZ0RAT] * z0);
            let aorb2 = F::new(2.0f32) * aorb.sqrt();
            let xi = (log10::<F>(max(aorb2 / z0noz, F::new(3.0f32))) - par[P_ABMIN]) / par[P_DELAB];
            let nfw = u32::cast_from(par[P_NFW]) as usize;
            fw = table::<F>(fwtab, nfw, xi);
        }
        sc[S_UORB] = uorb;
        sc[S_FW] = fw;
        sc[S_FU] = fu;
        sc[S_FUD] = fud;
        sc[S_PTURB] = pturb;
        sc[S_PVISC] = pvisc;
        sc[S_TAUX] = ustar * ustar * usdir.cos();
        sc[S_TAUY] = ustar * ustar * usdir.sin();
        sc[S_SX] = zero;
        sc[S_SY] = zero;
        sc[S_SNX] = zero;
        sc[S_SNY] = zero;
    }
    sync_cube();
    let uorb = sc[S_UORB];
    let fw = sc[S_FW];
    let fu = sc[S_FU];
    let fud = sc[S_FUD];
    let pturb = sc[S_PTURB];
    let pvisc = sc[S_PVISC];
    let sw0 = par[P_SSWELLF];
    let sw2 = par[P_SSWELLF + 2];
    let sw4 = par[P_SSWELLF + 4];
    for ik in 0..nk {
        if t == 0 {
            let taupx = sc[S_TAUX] - tsh * sc[S_SX];
            let taupy = sc[S_TAUY] - tsh * sc[S_SY];
            let ustp = min((taupx * taupx + taupy * taupy).powf(F::new(0.25f32)), max(ustar, F::new(0.3f32)));
            let usdirp = taupy.atan2(taupx);
            sc[S_USTP] = ustp;
            sc[S_COSU] = usdirp.cos();
            sc[S_SINU] = usdirp.sin();
        }
        sync_cube();
        if t < nth {
            let ustp = sc[S_USTP];
            let cosu = sc[S_COSU];
            let sinu = sc[S_SINU];
            let k = kb[K_WN * nk + ik];
            let c = kb[K_CG * nk + ik];
            let s2 = sig2k[ik];
            let cm = k / s2;
            let ucn = ustp * cm + par[P_ZZALP];
            let s = sig[ik + 1];
            let const2 = dden2k[ik] / c * grav / (s / k * drat);
            let cnst = s2 * const0;
            let zcn = (k * z0).ln();
            let swellcoefv = -sw4 * drat * F::new(2.0f32) * k * (F::new(2.0f32) * nu_air * s2).sqrt();
            let swellcoeft = -drat * sw0 * F::new(16.0f32) * s2 * s2 / grav;
            let is = ik * nth + t;
            let coswind = ecos[t] * cosu + esin[t] * sinu;
            let mut dst = zero;
            let mut llws = zero;
            if coswind > F::new(0.01f32) {
                let x = coswind * ucn;
                let zarg = kappa / x;
                let zlog = zcn + zarg;
                if zlog < zero {
                    dst = cnst * zlog.exp() * zlog.powi(4) * ucn * ucn * coswind.powf(par[P_SSINTHP]);
                    llws = one;
                }
                if F::new(28.0f32) * cm * ustar * coswind >= one {
                    llws = one;
                }
            }
            if (sw0 != zero && dst < F::new(1.0e-7f32) * s2) || sw2 > zero {
                let dvisc = swellcoefv;
                let dturb = swellcoeft * (fw * uorb + (fu + fud * coswind) * ustp);
                dst += pturb * dturb + pvisc * dvisc;
            }
            let x = a[is];
            let temp2 = const2 * dst * x;
            dr[t] = temp2 * ecos[t];
            dr[nth + t] = temp2 * esin[t];
            dr[2 * nth + t] = select(dst < zero, one, zero);
            ws[is] = llws;
            if accumulate {
                vs[is] += scale * (dst * x);
                vd[is] += scale * dst;
            }
        }
        sync_cube();
        if t == 0 {
            let mut sx = sc[S_SX];
            let mut sy = sc[S_SY];
            let mut snx = sc[S_SNX];
            let mut sny = sc[S_SNY];
            for th in 0..nth {
                if dr[2 * nth + th] > zero {
                    snx += dr[th];
                    sny += dr[nth + th];
                } else {
                    sx += dr[th];
                    sy += dr[nth + th];
                }
            }
            sc[S_SX] = sx;
            sc[S_SY] = sy;
            sc[S_SNX] = snx;
            sc[S_SNY] = sny;
        }
        sync_cube();
    }
    if t == 0 {
        let sx = sc[S_SX];
        let sy = sc[S_SY];
        let taupx = sc[S_TAUX] - tsh * sx;
        let taupy = sc[S_TAUY] - tsh * sy;
        let ust = (taupx * taupx + taupy * taupy).powf(F::new(0.25f32));
        let usdirp = taupy.atan2(taupx);
        // tail level from the last frequency, with the direction of its loop
        let cosu = sc[S_COSU];
        let sinu = sc[S_SINU];
        let snk = sig[nk];
        let c0 = par[P_DTH] * snk.powi(5) / (grav * grav * tpi::<F>()) * tpi::<F>() * snk / kb[K_CG * nk + nk - 1];
        let mut temp = zero;
        for th in 0..nth {
            let cw = max(ecos[th] * cosu + esin[th] * sinu, zero);
            temp += a[(nk - 1) * nth + th] * cw * cw * cw;
        }
        let levtail0 = c0 * temp;
        let mut tauhf_v = zero;
        if par[P_SINTABLE] > zero {
            // SINTABLE = 1: tail stress interpolated in TAUHFT2 (sheltering) or TAUHFT
            let ius = u32::cast_from(par[P_IUSTAR]) as usize;
            let ial = u32::cast_from(par[P_IALPHA]) as usize;
            let xi = ust / par[P_DELUST];
            let xif = u32::cast_from(xi.trunc()) as usize;
            let ind = select(xif < 1, 1, select(xif < ius - 1, xif, ius - 1));
            let deli1 = max(min(one, xi - F::cast_from(ind as u32)), zero);
            let deli2 = one - deli1;
            let um = max(ust, F::new(0.00001f32));
            let xj = max(zero, (grav * z0 / (um * um) - par[P_AALPHA]) / par[P_DELALP]);
            let xjf = u32::cast_from(xj.trunc()) as usize;
            let j = select(xjf < 1, 1, select(xjf < ial - 1, xjf, ial - 1));
            let delj1 = max(zero, min(one, xj - F::cast_from(j as u32)));
            let delj2 = one - delj1;
            let n1 = ius + 1;
            let n2 = ial + 1;
            let mut tau1 = zero;
            if par[P_TTAUWSHELTER] > zero {
                let ilv = u32::cast_from(par[P_ILEVTAIL]) as usize;
                let xk = levtail0 / par[P_DELTAIL];
                let xkf = u32::cast_from(xk.trunc()) as usize;
                let i = select(xkf < ilv - 1, xkf, ilv - 1);
                let delk1 = min(one, xk - F::cast_from(i as u32));
                let delk2 = one - delk1;
                let o0 = i * n1 * n2;
                let o1 = (i + 1) * n1 * n2;
                tau1 = ((tauhf[ind + j * n1 + o0] * deli2 + tauhf[ind + 1 + j * n1 + o0] * deli1) * delj2
                    + (tauhf[ind + (j + 1) * n1 + o0] * deli2 + tauhf[ind + 1 + (j + 1) * n1 + o0] * deli1) * delj1)
                    * delk2
                    + ((tauhf[ind + j * n1 + o1] * deli2 + tauhf[ind + 1 + j * n1 + o1] * deli1) * delj2
                        + (tauhf[ind + (j + 1) * n1 + o1] * deli2 + tauhf[ind + 1 + (j + 1) * n1 + o1] * deli1) * delj1)
                        * delk1;
            } else {
                tau1 = (tauhf[ind + j * n1] * deli2 + tauhf[ind + 1 + j * n1] * deli1) * delj2
                    + (tauhf[ind + (j + 1) * n1] * deli2 + tauhf[ind + 1 + (j + 1) * n1] * deli1) * delj1;
            }
            tauhf_v = levtail0 * ust * ust * tau1;
        } else {
        let km = F::new(363.0f32);
        let mut ustr = ust;
        let zinf = (snk * snk / grav).ln();
        let dely = ((tpi::<F>() / F::new(0.005f32)).ln() - zinf) / F::new(50.0f32);
        let mut taur = ust * ust;
        let mut tau1 = zero;
        let tp2 = par[P_SINTAILPAR + 1];
        let tp3 = par[P_SINTAILPAR + 2];
        for j in 1..51u32 {
            let w = select(j == 1u32 || j == 50u32, half, one);
            let y = (zinf + F::cast_from(j - 1u32) * dely).exp();
            let yk = y / km;
            let tensk = one + yk * yk;
            let cm = (grav * tensk / y).sqrt();
            let cgtail = half * (F::new(3.0f32) * yk * yk + one) * (grav / (y * tensk)).sqrt();
            let zx = ustr / cm + par[P_ZZALP];
            let zarg = min(kappa / zx, F::new(20.0f32));
            let zmu = min(grav * z0 / (cm * cm) * zarg.exp(), one);
            let zlog = min(zmu.ln(), zero);
            let zbeta = const1 * zmu * zlog.powi(4);
            let mut levtail = levtail0;
            if tp3 > zero {
                let cmm = F::new(0.2325f32);
                let mut alpham = F::new(0.01f32) * (one + F::new(3.0f32) * (ustr / cm).ln());
                if ustr < cm {
                    alpham = max(F::new(0.01f32) * (one + (ustr / cm).ln()), zero);
                }
                let ym = yk - one;
                let fm = (F::new(-0.25f32) * ym * ym).exp();
                let alphat = alpham * (cmm / cm) * fm;
                let th20 = ((y - F::new(20.0f32)) / F::new(5.0f32)).tanh();
                levtail = levtail0 * half * (one - th20) + tp3 * half * (one + th20) * alphat;
            }
            tau1 += w * zbeta * ustr * ustr * levtail * dely * cgtail / cm;
            taur -= w * tp2 * ustr * ustr * zbeta * levtail * dely * cgtail / cm;
            ustr = max(taur, zero).sqrt();
        }
        tauhf_v = tau1;
        }
        let mut tauwx = sx + tauhf_v * usdirp.cos();
        let mut tauwy = sy + tauhf_v * usdirp.sin();
        let tauw = (tauwx * tauwx + tauwy * tauwy).sqrt();
        let eps2 = F::new(0.000001f32);
        let um = max(ustar, eps2);
        let ust2 = um * um;
        let tauwb = min(tauw, max(ust2 - F::new(0.00001f32), eps2 * eps2));
        if tauwb < tauw {
            tauwx *= tauwb / tauw;
            tauwy *= tauwb / tauw;
        }
        sc[S_TAUWX] = tauwx;
        sc[S_TAUWY] = tauwy;
    }
    sync_cube();
}

/// `W3SLN1`: linear wind input, added as `scale S` to `vs`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn sln1<F: Float>(
    par: &[F],
    sig: &[F],
    ecos: &[F],
    esin: &[F],
    kb: &Shared<[F]>,
    vs: &mut Shared<[F]>,
    fhigh: F,
    ustar: F,
    usdir: F,
    scale: F,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
    #[comptime] nu: usize,
) {
    let t = UNIT_POS as usize;
    let zero = F::new(0.0f32);
    let nspec = nk * nth;
    let cosu = usdir.cos();
    let sinu = usdir.sin();
    let u2 = ustar * ustar;
    let fac = par[P_SLNC1] * (u2 * u2);
    let ff1 = par[P_FSPM] * par[P_GRAV] / (F::new(28.0f32) * ustar);
    let ff2 = par[P_FSHF] * min(sig[nk], fhigh);
    let ffilt = min(max(ff1, ff2), F::new(2.0f32) * sig[nk]);
    for r in 0..(nspec + nu - 1) / nu {
        let is = r * nu + t;
        if is < nspec {
            let ik = is / nth;
            let th = is - ik * nth;
            let rfr = sig[ik + 1] / ffilt;
            let mut wnf = zero;
            if rfr >= F::new(0.5f32) {
                let r2 = rfr * rfr;
                wnf = fac / kb[K_WN * nk + ik] * (-(F::new(1.0f32) / (r2 * r2))).exp();
            }
            let d = max(ecos[th] * cosu + esin[th] * sinu, zero);
            let d2 = d * d;
            vs[is] += scale * (wnf * (d2 * d2));
        }
    }
}

/// `W3SNL1`: nonlinear interactions by the DIA for the mean relative depth
/// `kdmean`, added as `scale S` and `scale D` to `vs` and `vd`. Ends with a
/// cube synchronisation.
#[cube]
#[allow(clippy::too_many_arguments)]
fn snl1<F: Float>(
    par: &[F],
    sig: &[F],
    dia_ip: &[u32],
    dia_ic: &[u32],
    af11: &[F],
    a: &Shared<[F]>,
    kb: &Shared<[F]>,
    ue: &mut Shared<[F]>,
    b1: &mut Shared<[F]>,
    b2: &mut Shared<[F]>,
    dn: &mut Shared<[F]>,
    vs: &mut Shared<[F]>,
    vd: &mut Shared<[F]>,
    kdmean: F,
    scale: F,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
    #[comptime] nspecx: usize,
    #[comptime] nspecy: usize,
    #[comptime] nu: usize,
) {
    let t = UNIT_POS as usize;
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let two = F::new(2.0f32);
    let nspec = nk * nth;
    let x = max(par[P_KDCON] * kdmean, par[P_KDMN]);
    let x2 = max(F::new(-1.0e15f32), par[P_SNLS3] * x);
    let cons = par[P_SNLC1] * (one + par[P_SNLS1] / x * (one - par[P_SNLS2] * x) * x2.exp());
    let dal1 = par[P_DAL1];
    let dal2 = par[P_DAL2];
    let dal3 = par[P_DAL3];
    // extended spectrum UE(1-NTH..NSPECY) at ue[ISP + NTH - 1]
    for r in 0..(nspecy + nth + nu - 1) / nu {
        let u = r * nu + t;
        if u < nspecy + nth {
            let mut v = zero;
            if u >= nth && u < nspec + nth {
                let is = u - nth;
                let ik = is / nth;
                let conx = tpiinv::<F>() / sig[ik + 1] * kb[K_CG * nk + ik];
                v = a[is] / conx;
            }
            ue[u] = v;
        }
    }
    for r in 0..(nth + nu - 1) / nu {
        let u = r * nu + t;
        if u < nth {
            b1[u] = zero;
            b2[u] = zero;
        }
    }
    sync_cube();
    if t < nth {
        for ifr in nk..nspecy / nth {
            ue[nth + ifr * nth + t] = ue[ifr * nth + t] * par[P_FACHFE];
        }
    }
    sync_cube();
    // three passes over the interaction arrays: SA1/SA2, DA1P/DA2P, DA1M/DA2M
    for pass in 0..3u32 {
        for r in 0..(nspecx + nu - 1) / nu {
            let isp = r * nu + t;
            if isp < nspecx {
                let e00 = ue[isp + nth];
                let ep1 = par[P_AWG] * ue[dia_ip[isp] as usize]
                    + par[P_AWG + 1] * ue[dia_ip[nspecx + isp] as usize]
                    + par[P_AWG + 2] * ue[dia_ip[2 * nspecx + isp] as usize]
                    + par[P_AWG + 3] * ue[dia_ip[3 * nspecx + isp] as usize];
                let em1 = par[P_AWG + 4] * ue[dia_ip[4 * nspecx + isp] as usize]
                    + par[P_AWG + 5] * ue[dia_ip[5 * nspecx + isp] as usize]
                    + par[P_AWG + 6] * ue[dia_ip[6 * nspecx + isp] as usize]
                    + par[P_AWG + 7] * ue[dia_ip[7 * nspecx + isp] as usize];
                let ep2 = par[P_AWG] * ue[dia_ip[8 * nspecx + isp] as usize]
                    + par[P_AWG + 1] * ue[dia_ip[9 * nspecx + isp] as usize]
                    + par[P_AWG + 2] * ue[dia_ip[10 * nspecx + isp] as usize]
                    + par[P_AWG + 3] * ue[dia_ip[11 * nspecx + isp] as usize];
                let em2 = par[P_AWG + 4] * ue[dia_ip[12 * nspecx + isp] as usize]
                    + par[P_AWG + 5] * ue[dia_ip[13 * nspecx + isp] as usize]
                    + par[P_AWG + 6] * ue[dia_ip[14 * nspecx + isp] as usize]
                    + par[P_AWG + 7] * ue[dia_ip[15 * nspecx + isp] as usize];
                let caf = cons * af11[isp];
                let factor = caf * e00;
                let sa1a = e00 * (ep1 * dal1 + em1 * dal2);
                let sa1b = sa1a - ep1 * em1 * dal3;
                let sa2a = e00 * (ep2 * dal1 + em2 * dal2);
                let sa2b = sa2a - ep2 * em2 * dal3;
                if pass == 0u32 {
                    b1[isp + nth] = factor * sa1b;
                    b2[isp + nth] = factor * sa2b;
                } else if pass == 1u32 {
                    b1[isp + nth] = factor * (dal1 * e00 - dal3 * em1);
                    b2[isp + nth] = factor * (dal1 * e00 - dal3 * em2);
                    if isp < nspec {
                        dn[isp] = -two * (caf * (sa1a + sa1b) + caf * (sa2a + sa2b));
                    }
                } else {
                    b1[isp + nth] = factor * (dal2 * e00 - dal3 * ep1);
                    b2[isp + nth] = factor * (dal2 * e00 - dal3 * ep2);
                }
            }
        }
        sync_cube();
        for r in 0..(nspec + nu - 1) / nu {
            let is = r * nu + t;
            if is < nspec {
                if pass == 0u32 {
                    let ik = is / nth;
                    let con = tpiinv::<F>() / sig[ik + 1] * kb[K_CG * nk + ik];
                    let mut ss = -two * (b1[is + nth] + b2[is + nth]);
                    for j in 0..8u32 {
                        let jj = j as usize;
                        ss += par[P_AWG + jj] * (b1[dia_ic[2 * jj * nspec + is] as usize] + b2[dia_ic[(2 * jj + 1) * nspec + is] as usize]);
                    }
                    vs[is] += scale * (con * ss);
                } else if pass == 1u32 {
                    let mut d = dn[is];
                    for j in 0..4u32 {
                        let jj = j as usize;
                        d += par[P_SWG + jj] * (b1[dia_ic[2 * jj * nspec + is] as usize] + b2[dia_ic[(2 * jj + 1) * nspec + is] as usize]);
                    }
                    dn[is] = d;
                } else {
                    let mut d = dn[is];
                    for j in 4..8u32 {
                        let jj = j as usize;
                        d += par[P_SWG + jj] * (b1[dia_ic[2 * jj * nspec + is] as usize] + b2[dia_ic[(2 * jj + 1) * nspec + is] as usize]);
                    }
                    vd[is] += scale * d;
                }
            }
        }
        sync_cube();
    }
}

/// `W3SDS4` with `SDSBCHOICE = 1` and the saturation tables, without the
/// cumulative term, added as `scale S`, `scale D`. `b1` holds the
/// saturation. Ends with a cube synchronisation.
#[cube]
#[allow(clippy::too_many_arguments)]
fn sds4<F: Float>(
    par: &[F],
    sig: &[F],
    ecos: &[F],
    esin: &[F],
    satw: &[F],
    sati: &[u32],
    cumw: &[F],
    a: &Shared<[F]>,
    kb: &mut Shared<[F]>,
    b1: &mut Shared<[F]>,
    dn: &mut Shared<[F]>,
    vs: &mut Shared<[F]>,
    vd: &mut Shared<[F]>,
    depth: F,
    ustar: F,
    usdir: F,
    scale: F,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
    #[comptime] nsat: usize,
    #[comptime] nu: usize,
) {
    let t = UNIT_POS as usize;
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let nspec = nk * nth;
    for r in 0..(nspec + nu - 1) / nu {
        let is = r * nu + t;
        if is < nspec {
            let ik = is / nth;
            let th = is - ik * nth;
            let k = kb[K_WN * nk + ik];
            let facsat = sig[ik + 1] * (k * k * k) * par[P_DTH];
            let mut dot = zero;
            for j in 0..nsat {
                dot += satw[j + th * nsat] * a[ik * nth + sati[j + th * nsat] as usize];
            }
            b1[is] = dot * facsat;
        }
    }
    sync_cube();
    if t < nk {
        let mut m = b1[t * nth];
        for th in 1..nth {
            m = max(m, b1[t * nth + th]);
        }
        kb[K_BTH0 * nk + t] = m;
    }
    let br = par[P_SSDSBR];
    let c3 = par[P_SDSC3];
    let c5 = par[P_SDSC5];
    let c21 = par[P_SDSC21];
    let extra = c3 != zero || c5 != zero || c21 != zero;
    if c3 != zero {
        // breaking-crest lengths (BRLAMBDA) for the cumulative term
        let epsr = br.sqrt();
        let pi = F::new(3.1415927f32);
        for r in 0..(nspec + nu - 1) / nu {
            let is = r * nu + t;
            if is < nspec {
                let q = max(b1[is].sqrt() - epsr, zero);
                dn[is] = q * q * F::new(28.16f32) / (F::new(2.0f32) * (pi * pi));
            }
        }
    }
    sync_cube();
    let c2 = par[P_SDSC2];
    let c6 = par[P_SDSC6];
    let dik = u32::cast_from(par[P_DIKCUMUL]) as usize;
    let facturb = c5 * ustar * ustar / par[P_GRAV] * par[P_DRAT];
    let cu = usdir.cos();
    let su = usdir.sin();
    for r in 0..(nspec + nu - 1) / nu {
        let is = r * nu + t;
        if is < nspec {
            let ik = is / nth;
            let mut miche = one;
            if par[P_SSDSBM] != one {
                let x = min(kb[K_WN * nk + ik] * depth, F::new(10.0f32)).tanh();
                let p = x * (par[P_SSDSBM + 1] + x * (par[P_SSDSBM + 2] + x * (par[P_SSDSBM + 3] + x * par[P_SSDSBM + 4])));
                miche = p * p;
            }
            let coef1 = br * miche;
            let s = sig[ik + 1];
            let q = max(kb[K_BTH0 * nk + ik] / coef1 - one, zero);
            let sdiagiso = c2 * s * c6 * (q * q);
            let coef2 = c2 * s * (one - c6) / (coef1 * coef1);
            let d = sdiagiso + coef2 * max(b1[is] - coef1, zero).powf(par[P_SSDSP]);
            let mut sv = d * a[is];
            let mut dv = d;
            if extra {
                let th = is - ik * nth;
                let k = kb[K_WN * nk + ik];
                let mut renewal = zero;
                if c3 < zero && ik + 1 > dik {
                    for ik2 in 0..(ik + 1 - dik) {
                        if kb[K_BTH0 * nk + ik2] > br {
                            let mut dot = zero;
                            for th2 in 0..nth {
                                let dth = select(th2 >= th, th2 - th, th - th2);
                                dot += cumw[(ik2 * nk + ik) * nth + dth] * dn[ik2 * nth + th2];
                            }
                            renewal += dot;
                        }
                    }
                }
                let facturb2 = F::new(-2.0f32) * s * k * facturb;
                let dvisc = F::new(-4.0f32) * c21 * k * k;
                let coswind = ecos[th] * cu + esin[th] * su;
                let dturb = facturb2 * max(zero, coswind);
                let diag2 = c3 * renewal + dturb + dvisc;
                dv += diag2;
                sv += a[is] * diag2;
            }
            vs[is] += scale * sv;
            vd[is] += scale * dv;
        }
    }
    sync_cube();
}

/// `W3SDB1`: the coefficient `D` of depth-induced breaking (`S = D A`),
/// from the mean parameters of `W3SPR4`.
#[cube]
fn sdb1<F: Float>(par: &[F], emean: F, fmean: F, wnmean: F, depth: F) -> F {
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let two = F::new(2.0f32);
    let thr = F::new(1.1754944e-38f32);
    let mut d = zero;
    if emean >= thr {
        let mut hm = par[P_SDBC2] * depth;
        if par[P_FDONLY] == zero {
            hm = par[P_SDBC2] / wnmean * (wnmean * max(depth, zero)).tanh();
        }
        let hrms = (F::new(8.0f32) * emean).sqrt();
        let mut bb = zero;
        let mut b = zero;
        if hm > thr {
            bb = hrms * hrms / (hm * hm);
            b = bb.sqrt();
        }
        let mut q0 = zero;
        if b > F::new(0.5f32) && b <= one {
            let y = two * b - one;
            q0 = y * y;
        }
        let mut qb = zero;
        if b > F::new(0.2f32) && b < one {
            let arg = ((q0 - one) / bb).exp();
            qb = q0 - bb * (q0 - arg) / (bb - arg);
            for _ in 0..3u32 {
                qb = ((qb - one) / bb).exp();
            }
        } else if b >= one {
            qb = one - thr;
        }
        let mut cbj = zero;
        if bb > thr && (bb - qb).abs() > thr {
            if bb < one {
                cbj = two * par[P_SDBC1] * qb * fmean / bb;
            } else {
                cbj = two * par[P_SDBC1] * fmean * bb;
            }
        }
        d = -cbj;
    }
    d
}

/// `W3SBT4`: bottom friction with movable-bed roughness, added as `S`, `D`
/// (no ice scaling). Ends with a cube synchronisation.
#[cube]
#[allow(clippy::too_many_arguments)]
fn sbt4<F: Float>(
    par: &[F],
    sig: &[F],
    dden: &[F],
    ecos: &[F],
    esin: &[F],
    fwtab: &[F],
    erftab: &[F],
    a: &Shared<[F]>,
    kb: &mut Shared<[F]>,
    sc: &mut Shared<[F]>,
    vs: &mut Shared<[F]>,
    vd: &mut Shared<[F]>,
    depth: F,
    d50_: F,
    psic: F,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
    #[comptime] nu: usize,
) {
    let t = UNIT_POS as usize;
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let nspec = nk * nth;
    if t < nk {
        let mut eb = zero;
        let mut ebx = zero;
        let mut eby = zero;
        for th in 0..nth {
            let x = a[t * nth + th];
            eb += x;
            ebx += x * ecos[th];
            eby += x * esin[th];
        }
        kb[K_EB * nk + t] = eb;
        kb[K_EBX * nk + t] = ebx;
        kb[K_EBY * nk + t] = eby;
    }
    sync_cube();
    if t == 0 {
        let grav = par[P_GRAV];
        let sg1 = par[P_SED_SG] - one;
        let nfw = u32::cast_from(par[P_NFW]) as usize;
        let d50 = min(max(d50_, F::new(0.00001f32)), one);
        let background = max(par[P_SBTCX + 5], par[P_SBTCX + 6] * d50);
        let mut sh0 = zero;
        let mut sh1 = zero;
        let mut sh2 = zero;
        let mut uorb = zero;
        let mut uorb2 = zero;
        let mut aorb = zero;
        let mut lx = zero;
        let mut ly = zero;
        for isub in 0..3u32 {
            let mut xsub = zero;
            if isub == 0u32 {
                xsub = F::new(-0.001f32);
            } else if isub == 1u32 {
                xsub = F::new(0.001f32);
            }
            let dsub = depth * (one + xsub);
            let mut u = zero;
            let mut ao = zero;
            let mut ax = zero;
            let mut ay = zero;
            for ik in 0..nk {
                let k = kb[K_WN * nk + ik];
                let mut cb = zero;
                if k * dsub < F::new(6.0f32) {
                    let s = sig[ik + 1];
                    let sh = (k * dsub).sinh();
                    cb = F::new(0.5f32) * s * s / (grav * (sh * sh));
                    let factor = (dden[ik] / kb[K_CG * nk + ik]) * F::new(2.0f32) * cb * grav;
                    let varu = kb[K_EB * nk + ik] * factor;
                    u += varu;
                    ao += varu / (s * s);
                    ax += kb[K_EBX * nk + ik] * factor;
                    ay += kb[K_EBY * nk + ik] * factor;
                }
                kb[K_CBETA * nk + ik] = cb;
            }
            uorb2 = F::new(2.0f32) * u;
            uorb = max(F::new(1.0e-7f32), uorb2).sqrt();
            aorb = max(F::new(1.0e-7f32), F::new(2.0f32) * ao).sqrt();
            let n = (ax * ax + ay * ay + F::new(1.0e-12f32)).sqrt();
            lx = aorb * F::new(1.7f32) * ax / n;
            ly = aorb * F::new(1.7f32) * ay / n;
            let xi = max((log10::<F>(max(aorb / d50, F::new(0.3f32))) - par[P_ABMIN]) / par[P_DELAB], one);
            let fw = table::<F>(fwtab, nfw, xi);
            let psi = fw * uorb2 / (F::new(2.0f32) * grav * sg1 * d50);
            let shields = psi / psic;
            if isub == 0u32 {
                sh0 = shields;
            } else if isub == 1u32 {
                sh1 = shields;
            } else {
                sh2 = shields;
            }
        }
        let _ = lx;
        let _ = ly;
        let sb0 = par[P_SBTCX];
        let sb1 = par[P_SBTCX + 1];
        let sb2 = par[P_SBTCX + 2];
        let sb3 = par[P_SBTCX + 3];
        let dpsi = (sh1 - sh0) / (F::new(0.001f32) - F::new(-0.001f32)) * par[P_SBTCX + 4];
        let ripple = F::new(0.0655f32) * (uorb2 / (sg1 * grav * aorb)).powf(F::new(1.4f32));
        let mut ksubn = max(background, aorb * sb3);
        if dpsi.abs() < F::new(0.0001f32) * sh2 || dpsi.abs() < F::new(1.0e-8f32) {
            if sh2 > sb2 {
                ksubn = aorb * sb0 * sh2.powf(sb1) + aorb * ripple;
            }
        } else {
            let psix = (sb2 - sh2) / dpsi;
            let psixt = max((psix + par[P_XERFMAX]) / par[P_DELXERF], zero);
            let nerf = u32::cast_from(par[P_NERF]) as usize;
            let proba2 = min(max(table::<F>(erftab, nerf, psixt), zero), one);
            let proba1 = one - proba2;
            let psin2 = max(sh2 + (-(F::new(0.5f32) * psix * psix)).exp() / tpi::<F>().sqrt() * dpsi / (proba2 + F::new(0.0001f32)), sb2);
            ksubn = proba1 * max(background, aorb * sb3) + proba2 * aorb * (sb0 * psin2.powf(sb1) + ripple);
        }
        let xi = max((log10::<F>(max(aorb / ksubn, F::new(0.3f32))) - par[P_ABMIN]) / par[P_DELAB], one);
        let fw = table::<F>(fwtab, nfw, xi);
        sc[S_FBT] = fw * uorb;
    }
    sync_cube();
    let fbt = sc[S_FBT];
    for r in 0..(nspec + nu - 1) / nu {
        let is = r * nu + t;
        if is < nspec {
            let ik = is / nth;
            let dsum = -fbt * kb[K_CBETA * nk + ik];
            vs[is] += dsum * a[is];
            vd[is] += dsum;
        }
    }
    sync_cube();
}

/// `W3SRCE` (`srce_direct`) for every node with `MAPSTA = 1`: one cube per
/// node, `nu` units per cube. With `start_ = 1` it is the `IT = 0` call of
/// `W3WAVE` (`dtg_ = 1`): u* is computed and the spectrum left unchanged.
/// Writes u* and the mean dynamic step of each node to `ust` and `dtdyn`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn srce<F: Float>(
    va: &mut [F],
    wn: &[F],
    cg: &[F],
    dw: &[F],
    u10: &[F],
    u10d: &[F],
    ice: &[F],
    d50: &[F],
    psic: &[F],
    nflag: &[u32],
    ust: &mut [F],
    dtdyn: &mut [F],
    par: &[F],
    sig: &[F],
    dden: &[F],
    ecos: &[F],
    esin: &[F],
    sig2k: &[F],
    dden2k: &[F],
    fwtab: &[F],
    erftab: &[F],
    satw: &[F],
    sati: &[u32],
    dia_ip: &[u32],
    dia_ic: &[u32],
    af11: &[F],
    taut: &[F],
    tauhf: &[F],
    cumw: &[F],
    dtg_: f32,
    start_: u32,
    np: usize,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
    #[comptime] nspecx: usize,
    #[comptime] nspecy: usize,
    #[comptime] nsat: usize,
    #[comptime] nu: usize,
) {
    let i = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let nspec = nk * nth;
    if i < np {
        if (nflag[i] & 4u32) != 0u32 {
            let zero = F::new(0.0f32);
            let one = F::new(1.0f32);
            let start = start_ == 1u32;
            let dtg = F::cast_from(dtg_);
            let mut a = Shared::<[F]>::new_slice(nk * nth);
            let mut vs = Shared::<[F]>::new_slice(nk * nth);
            let mut vd = Shared::<[F]>::new_slice(nk * nth);
            let mut ws = Shared::<[F]>::new_slice(nk * nth);
            let mut dn = Shared::<[F]>::new_slice(nk * nth);
            let mut ue = Shared::<[F]>::new_slice(nspecy + nth);
            let mut b1 = Shared::<[F]>::new_slice(nspecx + nth);
            let mut b2 = Shared::<[F]>::new_slice(nspecx + nth);
            let mut kb = Shared::<[F]>::new_slice(NKB * nk);
            let mut sc = Shared::<[F]>::new_slice(NSC);
            let mut dr = Shared::<[F]>::new_slice(3 * nth);
            let mut red = Shared::<[F]>::new_slice(nu);
            for r in 0..(nspec + nu - 1) / nu {
                let is = r * nu + t;
                if is < nspec {
                    let ik = is / nth;
                    let th = is - ik * nth;
                    a[is] = va[(ik * np + i) * nth + th];
                    ws[is] = one;
                }
            }
            if t < nk {
                let w = wn[(t + 1) * np + i];
                kb[K_WN * nk + t] = w;
                kb[K_CG * nk + t] = cg[(t + 1) * np + i];
                kb[K_DAM * nk + t] = par[P_FACP] / (sig[t + 1] * (w * w * w));
            }
            sync_cube();
            let d_inp = dw[i];
            let depth = max(par[P_DMIN], d_inp);
            let u = u10[i];
            let udir = u10d[i];
            let c = ice[i];
            let sd50 = d50[i];
            let spsic = psic[i];
            let mut isc_ln = one;
            let mut isc_in = one;
            let mut isc_nl = one;
            let mut isc_ds = one;
            if par[P_ICE_FORCING] > zero {
                isc_ln = max(zero, min(one, one - c * par[P_ICESCALES]));
                isc_in = max(zero, min(one, one - c * par[P_ICESCALES + 1]));
                isc_nl = max(zero, min(one, one - c * par[P_ICESCALES + 2]));
                isc_ds = max(zero, min(one, one - c * par[P_ICESCALES + 3]));
            }
            // before the loop: W3SPR4, W3SIN4 (TAUWBUG), W3SPR4
            spr4::<F>(par, taut, sig, dden, &a, &ws, &mut kb, &mut sc, u, zero, zero, nk, nth);
            if !start {
                let ust0 = sc[S_USTAR];
                let z00 = sc[S_Z0];
                sin4::<F>(
                    par, tauhf, sig, dden, ecos, esin, sig2k, dden2k, fwtab, &a, &mut vs, &mut vd, &mut ws, &mut kb, &mut sc, &mut dr, ust0, z00, udir, one, false, nk, nth,
                );
                let tauwx = sc[S_TAUWX];
                let tauwy = sc[S_TAUWY];
                spr4::<F>(par, taut, sig, dden, &a, &ws, &mut kb, &mut sc, u, tauwx, tauwy, nk, nth);
            }
            if t == 0 {
                let fh = max(par[P_FFXFM] * max(sc[S_FMEAN1], sc[S_FMEANWS]), par[P_FFXPM] / sc[S_USTAR]);
                sc[S_FHIGH] = fh;
                sc[S_DTTOT] = zero;
                sc[S_DTDYN] = zero;
                sc[S_NSTEPS] = zero;
                sc[S_DONE] = zero;
            }
            sync_cube();
            loop {
                let ustar = sc[S_USTAR];
                let z0 = sc[S_Z0];
                let fhigh = sc[S_FHIGH];
                let wnmean = sc[S_WNMEAN];
                let emean = sc[S_EMEAN];
                let fmean = sc[S_FMEAN];
                for r in 0..(nspec + nu - 1) / nu {
                    let is = r * nu + t;
                    if is < nspec {
                        vs[is] = zero;
                        vd[is] = zero;
                    }
                }
                sync_cube();
                sln1::<F>(par, sig, ecos, esin, &kb, &mut vs, fhigh, ustar, udir, isc_ln, nk, nth, nu);
                sync_cube();
                sin4::<F>(
                    par, tauhf, sig, dden, ecos, esin, sig2k, dden2k, fwtab, &a, &mut vs, &mut vd, &mut ws, &mut kb, &mut sc, &mut dr, ustar, z0, udir, isc_in, true, nk,
                    nth,
                );
                let stauwx = sc[S_TAUWX];
                let stauwy = sc[S_TAUWY];
                snl1::<F>(
                    par, sig, dia_ip, dia_ic, af11, &a, &kb, &mut ue, &mut b1, &mut b2, &mut dn, &mut vs, &mut vd, wnmean * depth, isc_nl, nk, nth, nspecx, nspecy, nu,
                );
                sds4::<F>(
                    par, sig, ecos, esin, satw, sati, cumw, &a, &mut kb, &mut b1, &mut dn, &mut vs, &mut vd, depth, ustar, udir, isc_ds, nk, nth, nsat, nu,
                );
                let dbd = sdb1::<F>(par, emean, fmean, wnmean, depth);
                sbt4::<F>(par, sig, dden, ecos, esin, fwtab, erftab, &a, &mut kb, &mut sc, &mut vs, &mut vd, depth, sd50, spsic, nk, nth, nu);
                // dynamic step: NKH from the FHIGH of the previous step
                let nkh = min(F::cast_from(nk as u32), (par[P_FACTI2] + par[P_FACTI1] * max(F::new(1.0e-7f32), fhigh).ln()).trunc());
                let nkh1 = min(F::cast_from(nk as u32), nkh + one);
                let nspech = u32::cast_from(max(nkh1, zero)) as usize * nth;
                let dttot = sc[S_DTTOT];
                let afilt = max(kb[K_DAM * nk + nk - 1], par[P_XFLT] * sc[S_AMAX]);
                let mut dtl = min(dtg - dttot, par[P_DTMAX]);
                for r in 0..(nspec + nu - 1) / nu {
                    let is = r * nu + t;
                    if is < nspech {
                        let ik = is / nth;
                        let th = is - ik * nth;
                        let ainit = va[(ik * np + i) * nth + th];
                        let damax = min(kb[K_DAM * nk + ik], max(par[P_XREL] * ainit, afilt));
                        let afac = one / max(F::new(1.0e-10f32), (vs[is] / damax).abs());
                        dtl = min(dtl, afac / max(F::new(1.0e-10f32), one + afac * min(zero, vd[is])));
                    }
                }
                red[t] = dtl;
                sync_cube();
                if t == 0 {
                    let mut dt = red[0];
                    for j in 1..nu {
                        dt = min(dt, red[j]);
                    }
                    dt = max(F::new(0.5f32), dt);
                    sc[S_DTDYN] = sc[S_DTDYN] + dt;
                    sc[S_NSTEPS] = sc[S_NSTEPS] + one;
                    let rem = dtg - dttot;
                    let idt = one + (F::new(0.99f32) * rem / dt).trunc();
                    dt = rem / idt;
                    let shave = dt < par[P_DTMIN] && dt < rem;
                    dt = max(dt, min(par[P_DTMIN], rem));
                    sc[S_DT] = dt;
                    sc[S_SHAVE] = select(shave, one, zero);
                    sc[S_DTTOT] = dttot + dt;
                }
                sync_cube();
                let dt = sc[S_DT];
                let shave = sc[S_SHAVE] > zero;
                let hdt = dt;
                for r in 0..(nspec + nu - 1) / nu {
                    let is = r * nu + t;
                    if is < nspech {
                        let ik = is / nth;
                        let aold = a[is];
                        let mut inc = vs[is] * dt / max(one, one - hdt * vd[is]);
                        if shave {
                            let dam = kb[K_DAM * nk + ik];
                            let m = min(dam, inc.abs());
                            inc = select(inc >= zero, m, -m);
                        }
                        let a1 = max(zero, aold + inc);
                        let incdb = (dbd * aold) * dt / max(one, one - hdt * dbd);
                        a[is] = max(zero, a1 + incdb);
                    }
                }
                sync_cube();
                spr4::<F>(par, taut, sig, dden, &a, &ws, &mut kb, &mut sc, u, stauwx, stauwy, nk, nth);
                if t == 0 {
                    let fage = par[P_FFXFA] * (F::new(0.3f32) * u * sc[S_FMEANWS] * tpi::<F>() / par[P_GRAV]).tanh();
                    let fh1 = (par[P_FFXFM] + fage) * sc[S_FMEAN1];
                    let fh2 = par[P_FFXPM] / sc[S_USTAR];
                    let fh = min(sig[nk], max(fh1, fh2));
                    sc[S_FHIGH] = fh;
                    let nk2 = max(F::new(2.0f32), min(nkh1, (par[P_FACTI2] + par[P_FACTI1] * max(F::new(1.0e-7f32), fh).ln()).trunc()));
                    sc[S_NKH] = nk2;
                    let last = sc[S_DTTOT] >= F::new(0.9999f32) * dtg;
                    sc[S_DONE] = select(last, one, zero);
                    // FHMAX limit at the end of the global step
                    sc[S_SCALE] = zero;
                    if last {
                        let wnm = sc[S_WNMEAN];
                        let hm = par[P_FHMAX] * (wnm * max(zero, d_inp)).tanh() / max(F::new(0.0001f32), wnm);
                        let em = hm * hm / F::new(16.0f32);
                        let e = sc[S_EMEAN];
                        if e > em && e > F::new(1.0e-30f32) {
                            sc[S_SCALE] = one;
                            sc[S_CONS] = em;
                        }
                    }
                }
                sync_cube();
                if sc[S_SCALE] > zero {
                    let fac = sc[S_CONS];
                    let e = sc[S_EMEAN];
                    for r in 0..(nspec + nu - 1) / nu {
                        let is = r * nu + t;
                        if is < nspec {
                            a[is] = a[is] / e * fac;
                        }
                    }
                    sync_cube();
                }
                // high-frequency tail above NKH
                let nkt = u32::cast_from(sc[S_NKH]) as usize;
                if t < nth {
                    for ik in nkt..nk {
                        a[ik * nth + t] = a[(ik - 1) * nth + t] * par[P_FACHFA];
                    }
                }
                sync_cube();
                if sc[S_DONE] > zero {
                    break;
                }
            }
            if t == 0 {
                ust[i] = sc[S_USTAR];
                dtdyn[i] = sc[S_DTDYN] / max(one, sc[S_NSTEPS]);
            }
            if !start {
                for r in 0..(nspec + nu - 1) / nu {
                    let is = r * nu + t;
                    if is < nspec {
                        let ik = is / nth;
                        let th = is - ik * nth;
                        va[(ik * np + i) * nth + th] = max(zero, a[is]);
                    }
                }
            }
        }
    }
}

/// Wind of the step (`W3UWND` with `WNT2` and `RWND`): speed and direction
/// interpolated in time with the speed correction of `WNT2`, then the wind
/// relative to the current (`RWINDC` times the current), as WW3 passes it
/// to `W3SRCE`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn wind_interp<F: Float>(
    ua0: &[F],
    uai: &[F],
    ud0: &[F],
    udi: &[F],
    cx: &[F],
    cy: &[F],
    u10: &mut [F],
    u10d: &mut [F],
    rd_: f32,
    rwindc_: f32,
    flcur_: u32,
    np: usize,
) {
    let i = ABSOLUTE_POS;
    if i < np {
        let rd = F::cast_from(rd_);
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
            let uxr = ua * ud.cos() - rw * cx[i];
            let uyr = ua * ud.sin() - rw * cy[i];
            u10[i] = max(F::new(0.001f32), (uxr * uxr + uyr * uyr).sqrt());
            let tpi = tpi::<F>();
            let v = tpi + uyr.atan2(uxr);
            u10d[i] = v - (v / tpi).trunc() * tpi;
        } else {
            u10[i] = max(ua, F::new(0.001f32));
            u10d[i] = ud;
        }
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

use crate::device::Real;
use cubecl::client::Client;
use cubecl::server::Handle;

fn bf(h: &Handle, n: usize) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(h.clone(), n) }
}

/// The device buffers `srce` reads for one node set: spectra `va`
/// (`[(ik np + i) nth + ith]`), wavenumbers and group speeds `wn`, `cg`
/// (`[(ik + 1) np + i]`, `ik = 0 .. NK - 1`), depth `dw`, wind speed and
/// direction relative to the current `u10`, `u10d`, ice concentration `ice`,
/// sediment diameter `d50` and critical Shields number `psic` (`SED_D50`,
/// `SED_PSIC`), node flags `nflag` (bit 2: `MAPSTA = 1`) and the outputs
/// `ust`, `dtdyn`.
pub struct SrceArgs<'a> {
    /// See [`SrceArgs`].
    pub va: &'a Handle,
    /// See [`SrceArgs`].
    pub wn: &'a Handle,
    /// See [`SrceArgs`].
    pub cg: &'a Handle,
    /// See [`SrceArgs`].
    pub dw: &'a Handle,
    /// See [`SrceArgs`].
    pub u10: &'a Handle,
    /// See [`SrceArgs`].
    pub u10d: &'a Handle,
    /// See [`SrceArgs`].
    pub ice: &'a Handle,
    /// See [`SrceArgs`].
    pub d50: &'a Handle,
    /// See [`SrceArgs`].
    pub psic: &'a Handle,
    /// See [`SrceArgs`].
    pub nflag: &'a Handle,
    /// See [`SrceArgs`].
    pub ust: &'a Handle,
    /// See [`SrceArgs`].
    pub dtdyn: &'a Handle,
}

/// The source-term parameters and tables on the device ([`SrceTables`])
/// and the launcher of `srce`.
pub struct SourceKernel<F: Real> {
    par: Handle,
    sig2k: Handle,
    dden2k: Handle,
    fwtab: Handle,
    erftab: Handle,
    satw: Handle,
    sati: Handle,
    dia_ip: Handle,
    dia_ic: Handle,
    af11: Handle,
    taut: Handle,
    tauhf: Handle,
    cumw: Handle,
    sig: Handle,
    dden: Handle,
    ecos: Handle,
    esin: Handle,
    nfw: usize,
    nerf: usize,
    ntaut: usize,
    ntauhf: usize,
    ncumw: usize,
    /// DIA sizes and the first dimension of the saturation tables.
    pub nspecx: usize,
    /// See [`SourceKernel::nspecx`].
    pub nspecy: usize,
    /// See [`SourceKernel::nspecx`].
    pub nsat: usize,
    /// Units per cube.
    pub nu: usize,
    _f: std::marker::PhantomData<F>,
}

impl<F: Real> SourceKernel<F> {
    /// Copy the tables to the device; `nu` units per cube (a power of two
    /// at least `NK` and `NTH`).
    pub fn new(client: &Client, t: &SrceTables, nu: usize) -> SourceKernel<F> {
        let upd = |v: &[f64]| {
            let w: Vec<F> = v.iter().map(|&x| <F as crate::backend::Scalar>::of_f64(x)).collect();
            client.create_from_slice(F::as_bytes(&w))
        };
        let upu = |v: &[u32]| client.create_from_slice(u32::as_bytes(v));
        SourceKernel {
            par: upd(&t.par),
            sig2k: upd(&t.sig2k),
            dden2k: upd(&t.dden2k),
            fwtab: upd(&t.fwtable),
            erftab: upd(&t.erftable),
            satw: upd(&t.satw),
            sati: upu(&t.sati),
            dia_ip: upu(&t.dia_ip),
            dia_ic: upu(&t.dia_ic),
            af11: upd(&t.af11),
            taut: upd(&t.taut),
            tauhf: upd(&t.tauhf),
            cumw: upd(&t.cumw),
            ntaut: t.taut.len(),
            ntauhf: t.tauhf.len(),
            ncumw: t.cumw.len(),
            sig: upd(&t.sig),
            dden: upd(&t.dden),
            ecos: upd(&t.ecos),
            esin: upd(&t.esin),
            nfw: t.fwtable.len(),
            nerf: t.erftable.len(),
            nspecx: t.nspecx,
            nspecy: t.nspecy,
            nsat: t.nsat,
            nu,
            _f: std::marker::PhantomData,
        }
    }

    /// Launch `srce` on `np` nodes: one global step `dtg`, or the start
    /// call of `W3WAVE` (`start`, `dtg = 1`).
    #[allow(clippy::too_many_arguments)]
    pub fn launch(&self, client: &Client, s: &SrceArgs, dtg: f32, start: bool, np: usize, nk: usize, nth: usize) {
        let nspec = nk * nth;
        unsafe {
            srce::launch_unchecked::<F>(
                client,
                CubeCount::Static(np as u32, 1, 1),
                CubeDim::new_1d(self.nu as u32),
                bf(s.va, nspec * np),
                bf(s.wn, (nk + 2) * np),
                bf(s.cg, (nk + 2) * np),
                bf(s.dw, np),
                bf(s.u10, np),
                bf(s.u10d, np),
                bf(s.ice, np),
                bf(s.d50, np),
                bf(s.psic, np),
                bf(s.nflag, np),
                bf(s.ust, np),
                bf(s.dtdyn, np),
                bf(&self.par, NPAR),
                bf(&self.sig, nk + 2),
                bf(&self.dden, nk),
                bf(&self.ecos, nth),
                bf(&self.esin, nth),
                bf(&self.sig2k, nk),
                bf(&self.dden2k, nk),
                bf(&self.fwtab, self.nfw),
                bf(&self.erftab, self.nerf),
                bf(&self.satw, self.nsat * nth),
                bf(&self.sati, self.nsat * nth),
                bf(&self.dia_ip, 16 * self.nspecx),
                bf(&self.dia_ic, 16 * nspec),
                bf(&self.af11, self.nspecx),
                bf(&self.taut, self.ntaut),
                bf(&self.tauhf, self.ntauhf),
                bf(&self.cumw, self.ncumw),
                dtg,
                start as u32,
                np,
                nk,
                nth,
                self.nspecx,
                self.nspecy,
                self.nsat,
                self.nu,
            );
        }
    }
}
