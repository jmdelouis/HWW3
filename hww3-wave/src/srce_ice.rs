//! CubeCL kernel of the part of `W3SRCE` that follows the dynamic
//! integration loop in the IRI configuration, once per global step `DTG`:
//! the sea-ice step with `IC_NUMERICS = F` (`W3SIS2` then `W3SIC2`, the
//! implicit attenuation per frequency and the isotropic back-scatter,
//! section 10.1 of `W3SRCE`) and the shoreline, sub-grid and iceberg
//! reflection `W3SREF` with `SPEC = SPEC + DTG VREF` (section 12),
//! transcribed from the CPU references [`crate::ice::ice_after_loop`] and
//! [`crate::ref1::sref`].
//!
//! One cube per node, `nu` units: the spectrum sits in shared memory; the
//! units `t < NK` do the work of frequency `t` (band sums, scattering and
//! dissipation coefficients, the attenuation and the isotropic
//! redistribution of band `t`), unit 0 the serial parts (the break-up of the
//! floes and the new `ICEF`, the orbital sums of `W3SIC2`, the mean energy
//! and frequency of `W3SREF`) with the sums in the order of WW3, and every
//! unit the reflection of the spectral bins it owns (bin `r nu + t`), summing
//! over the incident directions in the order of WW3, so that the reflected
//! spectrum is written straight to global memory.
//!
//! Ported options (the host refuses the others, see [`IceRefKernel::new`]):
//! `IS2DISP = F` (in-ice dispersion of the open water, `WN_R = WN`,
//! `CG_ICE = CG`), the isotropic scattering `IS2ISOSCAT = T`,
//! `IC2DISPER = F`, `IC2TURBS = 0`, `ICEDISP = F`, a constant ice thickness
//! (`ICEH = ICEHINIT`) and maximum floe diameter of the forcing
//! (`ICEDMAX = 1000 m`), the structured-grid branch of `W3SREF` without
//! `IG1`. The fluxes `PHICE` and `TAUICE` to the ice are not computed.
//!
//! Shared arrays of a cube: the spectrum `a`, per-frequency values `kb`
//! (`kb[slot nk + ik]`) and scalars `sc`.

// The `#[cube]` macro generates, for every kernel, a launch struct and
// functions that carry no documentation; the kernel itself is documented
// below.
#![allow(missing_docs)]

use cubecl::prelude::*;

// Slots of the parameter buffer `par` of the ice terms.
/// `IS2PARS(1..24)` at `P_IS2 + i - 1`.
pub(crate) const P_IS2: usize = 0;
/// `IC2PARS(1..8)` at `P_IC2 + i - 1`.
pub(crate) const P_IC2: usize = 24;
/// Ice thickness `ICEH` (`ICEHINIT`).
pub(crate) const P_ICEH: usize = 32;
/// Maximum floe diameter of the forcing `ICEDMAX`.
pub(crate) const P_ICEDMAX: usize = 33;
/// `ICEDMIN` of `W3SIS2` (`IS2DMIN` when positive).
pub(crate) const P_ICEDMIN: usize = 34;
/// `IS2BREAKE 4 YOUNG ICEH^3 / (3 DWAT GRAV (1 - POISSON^2))`, the factor of
/// `(PI / WLG)^4` in `R`.
pub(crate) const P_RC: usize = 35;
/// `CURVTOSTRAIN = 0.25 max(ICEH, ICEHMIN)^2`.
pub(crate) const P_CTS: usize = 36;
/// `STRAIN_C = IS2FLEXSTR (1 - POISSON^2) / YOUNG`.
pub(crate) const P_STRC: usize = 37;
/// `D_FLEX_FAIL`: smallest floe that can break.
pub(crate) const P_DFF: usize = 38;
/// `CREEPFAC` of the creep dissipation.
pub(crate) const P_CREEP: usize = 39;
/// `NSUM`: half width of the frequency window of the curvature.
pub(crate) const P_NSUM: usize = 40;
/// `PI`.
pub(crate) const P_PI: usize = 41;
/// `TPI = 2 PI`.
pub(crate) const P_TPI: usize = 42;
/// `VISCM` of `W3SIC2`.
pub(crate) const P_VISCM: usize = 43;
/// `GRAV`.
pub(crate) const P_GRAV: usize = 44;
/// Exponent `GAM` of the floe-size distribution of `W3FSD_DAVE`.
pub(crate) const P_GAM: usize = 45;
/// `DELAB` of the friction factor table.
pub(crate) const P_DELAB: usize = 46;
/// `ABMIN` of the friction factor table.
pub(crate) const P_ABMIN: usize = 47;
/// 1 when `W3SIC2` sees no ice (`ICEH = 0`).
pub(crate) const P_NOICE: usize = 48;
/// `1 / ln 10`.
pub(crate) const P_INVLN10: usize = 49;
/// `REFPARS(1..10)` at `P_REF + i - 1`.
pub(crate) const P_REF: usize = 50;
/// `FAC1 = 1 / (NTH / 2)` of `W3SREF`.
pub(crate) const P_FAC1: usize = 60;
/// `FAC2 = 1.57 / (NTH / 2)`.
pub(crate) const P_FAC2: usize = 61;
/// `FAC3 = 2 / sum |ECOS|^NRS`.
pub(crate) const P_FAC3: usize = 62;
/// `NRS = NINT(REFCOSP_STRAIGHT)`.
pub(crate) const P_NRS: usize = 63;
/// `TPIINV`.
pub(crate) const P_TPIINV: usize = 64;
/// Literal constants of the Fortran kept at full precision in double
/// precision: 0.07, 0.007, 1e-4, 0.005, 0.001, 1e-7, 0.75, 1.01, 0.2, 0.8.
pub(crate) const P_C: usize = 65;
/// Number of slots of `par`.
pub(crate) const NPAR: usize = 80;

// Per-frequency tables of `ktab` (`ktab[slot nk + ik]`).
/// `SIG(IK)` of the ice terms.
pub(crate) const T_SIG: usize = 0;
/// `DDEN(IK)` of the ice terms.
pub(crate) const T_DDEN: usize = 1;
/// `SIS2ALPHAS` interpolated at the thickness `ICEH`.
pub(crate) const T_ALPH: usize = 2;
/// `2 IS2C2 exp(-IS2C3 / SIG)` (scattering of the pack ice).
pub(crate) const T_SQ: usize = 3;
/// The factor of `WN^4 SMOOTHD / R^2` in the anelastic dissipation.
pub(crate) const T_ANEL: usize = 4;
/// `DDEN(IK)` of `W3SREF`.
pub(crate) const T_RDDEN: usize = 5;
/// Number of tables of `ktab`.
pub(crate) const NKT: usize = 6;

// Per-frequency slots of the shared `kb`.
const K_WN: usize = 0;
const K_CG: usize = 1;
const K_WLG: usize = 2;
const K_R: usize = 3;
const K_CS: usize = 4;
const K_CURV: usize = 5;
const K_VDIR: usize = 6;
const K_VDIR2: usize = 7;
const K_EB: usize = 8;
const K_BRK: usize = 9;
const K_SMD: usize = 10;
const K_EBR: usize = 11;
const NKB: usize = 12;

// Scalar slots of the shared `sc`.
const S_DMAX: usize = 0;
const S_ICEF: usize = 1;
const S_PTURB: usize = 2;
const S_PVISC: usize = 3;
const S_DTURB: usize = 4;
const S_EMEANA: usize = 5;
const S_FMEANA: usize = 6;
const S_RAMP0: usize = 7;
const NSC: usize = 8;

/// `W3FSD_DAVE`: mean diameter of the power-law floe-size distribution
/// between `dmin` and `dmax` (see [`crate::ice::fsd_dave`]).
#[cube]
fn fsd_dave<F: Float>(par: &[F], dmin: F, dmax: F) -> F {
    let gam = par[P_GAM];
    let one = F::new(1.0f32);
    let dmaxl = max(dmin * par[P_C + 7], dmax);
    let mut r = dmaxl;
    if dmin != F::new(0.0f32) {
        r = gam / (gam - one) * ((dmaxl.powf(-gam + one) - dmin.powf(-gam + one)) / (dmaxl.powf(-gam) - dmin.powf(-gam)));
    }
    r
}

/// `x^4` as `x x x x`.
#[cube]
fn pow4<F: Float>(x: F) -> F {
    let x2 = x * x;
    x2 * x2
}

/// `|i - j| mod n` for direction indices.
#[cube]
fn dmod(i: u32, j: u32, n: u32) -> u32 {
    // the branch not taken may wrap around, unused
    let d = select(i >= j, i - j, j - i);
    d % n
}

/// The ice step and the reflection of `W3SRCE` after the dynamic loop, for
/// every node whose flag has bit 2 (`MAPSTA = 1`): with `ice`, the terms of
/// `W3SIS2` and `W3SIC2` and the implicit update of the spectrum where the
/// ice concentration is positive (`ICEF` updated, or set to 0 without ice
/// when `IS2DUPDATE = F`); with `refl`, `SPEC + DTG VREF` of `W3SREF` where
/// it is called. With neither, nothing is done.
///
/// Per node: `reflc[4 i + k]` is `REFLEC(k + 1)` (with `BERG` applied),
/// `refld[6 i + k]` is `REFLD(k + 1)`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ice_ref<F: Float>(
    va: &mut [F],
    wn: &[F],
    cg: &[F],
    _depth: &[F],
    icec: &[F],
    icef: &mut [F],
    reflc: &[F],
    refld: &[u32],
    delx: &[F],
    dely: &[F],
    trnx: &[F],
    trny: &[F],
    berg: &[F],
    nflag: &[u32],
    par: &[F],
    ktab: &[F],
    rsig: &[F],
    recos: &[F],
    resin: &[F],
    idave: &[F],
    fwtab: &[F],
    dtg_: f32,
    np: usize,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
    #[comptime] nu: usize,
    #[comptime] nfw: usize,
    #[comptime] ice: bool,
    #[comptime] refl: bool,
) {
    if ice || refl {
        let i = CUBE_POS as usize;
        let t = UNIT_POS as usize;
        let nspec = nk * nth;
        if i < np {
            if (nflag[i] & 4u32) != 0u32 {
                let zero = F::new(0.0f32);
                let one = F::new(1.0f32);
                let two = F::new(2.0f32);
                let half = F::new(0.5f32);
                let dtg = F::cast_from(dtg_);
                let tpi = par[P_TPI];
                let mut a = Shared::<[F]>::new_slice(nk * nth);
                let mut kb = Shared::<[F]>::new_slice(NKB * nk);
                let mut sc = Shared::<[F]>::new_slice(NSC);
                for r in 0..(nspec + nu - 1) / nu {
                    let is = r * nu + t;
                    if is < nspec {
                        let ik = is / nth;
                        let th = is - ik * nth;
                        a[is] = va[(ik * np + i) * nth + th];
                    }
                }
                if t < nk {
                    kb[K_WN * nk + t] = wn[(t + 1) * np + i];
                    kb[K_CG * nk + t] = cg[(t + 1) * np + i];
                }
                sync_cube();
                if ice {
                    let cice = icec[i];
                    if cice > zero {
                        let icef0 = icef[i];
                        // W3SIS2, 1-3: scattering coefficient and R per frequency
                        if t < nk {
                            let w = kb[K_WN * nk + t];
                            let c = kb[K_CG * nk + t];
                            let mut suma = zero;
                            for th in 0..nth {
                                suma += a[t * nth + th];
                            }
                            let dmax = icef0;
                            let mut icedave = dmax;
                            if dmax > F::new(500.0f32) {
                                icedave = dmax;
                            } else if par[P_IS2 + 8] > zero {
                                let ir = u32::cast_from((dmax + half).trunc());
                                let ir = select(ir < 1u32, 1u32, ir);
                                icedave = idave[(ir - 1u32) as usize];
                            } else if par[P_IS2 + 13] * (tpi / w) < dmax {
                                icedave = fsd_dave::<F>(par, (tpi / w) * par[P_IS2 + 13], dmax);
                            }
                            let cg_eff = cice * c + (one - cice) * c;
                            let mut alpha = -one * (ktab[T_ALPH * nk + t] / icedave + zero) * cice;
                            alpha = par[P_IS2 + 0] * alpha - ktab[T_SQ * nk + t];
                            if par[P_IS2 + 10] > zero {
                                let lo = par[P_C + 8];
                                let hi = par[P_C + 9];
                                if cice < lo {
                                    alpha = zero;
                                }
                                if cice > hi {
                                    alpha = zero;
                                }
                                if cice >= lo && cice <= hi {
                                    alpha *= (cice - lo) * (hi - cice);
                                }
                            }
                            kb[K_VDIR * nk + t] = alpha * cice * cg_eff;
                            let wlg = tpi / w;
                            let r = one + par[P_RC] * pow4::<F>(par[P_PI] / wlg);
                            let sume = suma * ktab[T_DDEN * nk + t] / c / (r * one);
                            kb[K_CS * nk + t] = sume * pow4::<F>(two * par[P_PI] / wlg);
                            kb[K_WLG * nk + t] = wlg;
                            kb[K_R * nk + t] = r;
                            kb[K_EB * nk + t] = suma;
                        }
                        sync_cube();
                        // curvature summed over the window of NSUM frequencies
                        let nsum = u32::cast_from(par[P_NSUM]) as usize;
                        if t < nk {
                            let lo = select(t >= nsum, t - nsum, 0usize);
                            let hi = select(t + nsum < nk - 1, t + nsum, nk - 1);
                            let mut s = zero;
                            for j in 0..nk {
                                if j >= lo && j <= hi {
                                    s += kb[K_CS * nk + j];
                                }
                            }
                            kb[K_CURV * nk + t] = s;
                        }
                        sync_cube();
                        // 4. floe size and break-up (serial)
                        if t == 0 {
                            let mut dmax = icef0;
                            let mut nicef = icef0;
                            if par[P_IS2 + 9] < half {
                                dmax = par[P_ICEDMAX];
                                nicef = par[P_ICEDMAX];
                            }
                            if par[P_IS2 + 2] > half {
                                let dff = par[P_DFF];
                                let mut icedmin = par[P_ICEDMIN];
                                let mut ikbreak = 0usize;
                                let mut brk = false;
                                for ik in 0..nk {
                                    let strain = kb[K_CURV * nk + ik] * par[P_CTS];
                                    let mut bw = zero;
                                    if dff < dmax {
                                        let wamp = par[P_IS2 + 16] * strain.sqrt();
                                        if par[P_IS2 + 8] == zero {
                                            icedmin = (tpi / kb[K_WN * nk + ik]) * par[P_IS2 + 13];
                                        }
                                        let hw = kb[K_WLG * nk + ik] / two;
                                        if wamp > par[P_STRC] && hw > icedmin && hw < dmax && hw > dff {
                                            bw = one;
                                            ikbreak = ik;
                                            brk = true;
                                        }
                                    }
                                    kb[K_BRK * nk + ik] = bw;
                                }
                                if brk {
                                    let lo = select(ikbreak >= nsum, ikbreak - nsum, 0usize);
                                    let hi = ikbreak;
                                    for ik in 0..nk {
                                        if ik >= lo && ik <= hi {
                                            if kb[K_CS * nk + ik] >= kb[K_CS * nk + ikbreak] && dmax >= kb[K_WLG * nk + ik] / two {
                                                ikbreak = ik;
                                            }
                                        }
                                    }
                                    dmax = kb[K_WLG * nk + ikbreak] / two;
                                    let ikp1 = select(ikbreak + 1 < nk - 1, ikbreak + 1, nk - 1);
                                    let ikm1 = select(ikbreak >= 1, ikbreak - 1, 0usize);
                                    if kb[K_BRK * nk + ikp1] > zero && kb[K_BRK * nk + ikm1] > zero {
                                        let cs0 = kb[K_CS * nk + ikbreak];
                                        let csp = kb[K_CS * nk + ikp1];
                                        let csm = kb[K_CS * nk + ikm1];
                                        dmax = (kb[K_WLG * nk + ikbreak] * cs0 + kb[K_WLG * nk + ikp1] * csp + kb[K_WLG * nk + ikm1] * csm) / (two * (cs0 + csm + csp));
                                    }
                                    nicef = dmax;
                                }
                            }
                            sc[S_DMAX] = dmax;
                            sc[S_ICEF] = nicef;
                            icef[i] = nicef;
                        }
                        sync_cube();
                        // 5. dissipation (VDIR2) and the floe factor of W3SIC2
                        if t < nk {
                            let dmax = sc[S_DMAX];
                            let wlg = kb[K_WLG * nk + t];
                            let r = kb[K_R * nk + t];
                            let mut v = zero;
                            if par[P_IS2 + 11] > zero {
                                if par[P_IS2 + 13] * wlg < dmax {
                                    let smoothd = ((dmax - par[P_IS2 + 13] * wlg) / (dmax * par[P_IS2 + 12])).tanh();
                                    if par[P_IS2 + 22] <= half {
                                        v = par[P_CREEP] * F::new(4.0f32) * kb[K_CURV * nk + t] * (two * par[P_PI] / wlg).powf(par[P_IS2 + 14] + one) / (one * r * r) * smoothd;
                                    } else {
                                        v = ktab[T_ANEL * nk + t] * pow4::<F>(kb[K_WN * nk + t]) * smoothd / (r * r * one);
                                    }
                                }
                            }
                            kb[K_VDIR2 * nk + t] = v;
                            let mut smd = one;
                            if par[P_IC2 + 7] > zero {
                                let nf = sc[S_ICEF];
                                let wlg_r = tpi / kb[K_WN * nk + t];
                                let x = half * (one + ((nf - par[P_IC2 + 7] * wlg_r) / (nf * half)).tanh());
                                smd = x * x;
                            }
                            kb[K_SMD * nk + t] = smd;
                        }
                        sync_cube();
                        // W3SIC2: orbital velocity and amplitude (serial)
                        if t == 0 {
                            let mut pturb = zero;
                            let mut pvisc = one;
                            let mut dturb = zero;
                            if par[P_IC2 + 1] > zero {
                                let mut uorb = zero;
                                let mut aorb = zero;
                                for ik in 0..nk {
                                    let eb = kb[K_EB * nk + ik];
                                    let smd = kb[K_SMD * nk + ik];
                                    let sg = ktab[T_SIG * nk + ik];
                                    let dd = ktab[T_DDEN * nk + ik];
                                    let c = kb[K_CG * nk + ik];
                                    let r = kb[K_R * nk + ik];
                                    if r > one {
                                        uorb += eb * smd * sg * sg * dd / c / (r * c / c);
                                        aorb += eb * smd * dd / c / (r * c / c);
                                    } else {
                                        uorb += eb * smd * sg * sg * dd / c;
                                        aorb += eb * smd * dd / c;
                                    }
                                }
                                let aorb = two * aorb.sqrt();
                                let uorb = two * uorb.sqrt();
                                let re = uorb * aorb / par[P_VISCM];
                                let smooth = half * ((re - par[P_IC2 + 3]) / par[P_IC2 + 4]).tanh();
                                pturb = half + smooth;
                                pvisc = half - smooth;
                                let xi = (max(aorb / par[P_IC2 + 2], F::new(3.0f32)).ln() * par[P_INVLN10] - par[P_ABMIN]) / par[P_DELAB];
                                let fw = crate::srce::table::<F>(fwtab, nfw, xi);
                                dturb = par[P_IC2 + 1] * fw * uorb / par[P_GRAV];
                            }
                            sc[S_PTURB] = pturb;
                            sc[S_PVISC] = pvisc;
                            sc[S_DTURB] = dturb;
                        }
                        sync_cube();
                        // 10.1 attenuation and isotropic scattering of band t
                        if t < nk {
                            let sg = ktab[T_SIG * nk + t];
                            let mut vdic = zero;
                            if par[P_NOICE] == zero {
                                let dvisc = par[P_IC2 + 5] * kb[K_WN * nk + t] * (par[P_VISCM] * sg / two).sqrt();
                                vdic = -one * (sc[S_PTURB] * max(sc[S_DTURB] * sg * sg, dvisc) + sc[S_PVISC] * dvisc) * kb[K_SMD * nk + t];
                            }
                            let vdir = kb[K_VDIR * nk + t];
                            let mut att = (cice * vdic * dtg).exp();
                            att *= (cice * kb[K_VDIR2 * nk + t] * dtg).exp();
                            if par[P_IS2 + 1] == zero {
                                att *= (cice * vdir * dtg).exp();
                            }
                            for th in 0..nth {
                                a[t * nth + th] = att * a[t * nth + th];
                            }
                            if par[P_IS2 + 1] >= zero {
                                let scf = (vdir * par[P_IS2 + 1] * dtg).exp();
                                let mut s = zero;
                                for th in 0..nth {
                                    s += a[t * nth + th];
                                }
                                let iso = s / F::cast_from(nth as u32);
                                for th in 0..nth {
                                    a[t * nth + th] = iso + (a[t * nth + th] - iso) * scf;
                                }
                            }
                        }
                        sync_cube();
                    } else {
                        if t == 0 && par[P_IS2 + 9] < half {
                            icef[i] = zero;
                        }
                    }
                }
                // W3SREF
                let mut called = false;
                let mut lc0 = zero;
                let mut lc1 = zero;
                let mut lc3 = zero;
                let mut bg = zero;
                if refl {
                    lc0 = reflc[4 * i];
                    lc1 = reflc[4 * i + 1];
                    lc3 = reflc[4 * i + 3];
                    bg = berg[i];
                    called = lc0 > zero || lc1 > zero || (lc3 > zero && bg > zero);
                    if called {
                        if t < nk {
                            let mut eb = zero;
                            for th in 0..nth {
                                eb += a[t * nth + th];
                            }
                            kb[K_EBR * nk + t] = eb * ktab[T_RDDEN * nk + t] / kb[K_CG * nk + t];
                        }
                        sync_cube();
                        if t == 0 {
                            let mut emeana = zero;
                            let mut fmeana = zero;
                            for ik in 0..nk {
                                let eb = kb[K_EBR * nk + ik];
                                emeana += eb;
                                fmeana += eb / rsig[ik + 1];
                            }
                            fmeana = par[P_TPIINV] * (emeana / max(fmeana, par[P_C + 5]));
                            fmeana = max(fmeana, rsig[1]);
                            let mut ramp0 = one;
                            if par[P_REF + 5] > zero {
                                // max() also takes 0.001 for a NaN slope
                                let slope = max(reflc[4 * i + 2], par[P_C + 4]);
                                let g = par[P_GRAV];
                                let fm = max(fmeana, par[P_C + 4]);
                                let michefac = par[P_C + 2] * g * g * (pow4::<F>(slope) * slope) / (max(emeana, par[P_C + 2]) * pow4::<F>(fm));
                                ramp0 = max(par[P_C] * (michefac.ln() * par[P_INVLN10] + F::new(4.5f32)) + F::new(1.5f32) * michefac, par[P_C + 3]);
                            }
                            sc[S_EMEANA] = emeana;
                            sc[S_FMEANA] = fmeana;
                            sc[S_RAMP0] = ramp0;
                        }
                        sync_cube();
                    }
                }
                // write back, with DTG VREF on the bins of this unit
                for r in 0..(nspec + nu - 1) / nu {
                    let is = r * nu + t;
                    if is < nspec {
                        let ik = is / nth;
                        let th = is - ik * nth;
                        let mut v = a[is];
                        if refl {
                            if called {
                                let fmeana = sc[S_FMEANA];
                                let emeana = sc[S_EMEANA];
                                let ramp0 = sc[S_RAMP0];
                                let cgk = kb[K_CG * nk + ik];
                                let dx = delx[i];
                                let dy = dely[i];
                                let nthu = nth as u32;
                                let ith2 = th as u32 + 1u32;
                                let mut s = zero;
                                // 2. shoreline reflection
                                if lc0 > zero {
                                    let mut ramp1 = ramp0 * lc0;
                                    if par[P_REF + 5] > zero {
                                        let ramp = max(par[P_C + 6] * tpi * fmeana / rsig[ik + 1], one).powf(par[P_REF + 9]) * ramp0;
                                        ramp1 = min(par[P_REF + 8], lc0 * ramp);
                                    }
                                    let r1d = refld[6 * i];
                                    let rd2 = refld[6 * i + 1];
                                    let rd3 = F::cast_from(refld[6 * i + 2]);
                                    let rd4 = F::cast_from(refld[6 * i + 3]);
                                    let ith2x = 1u32 + (nthu + ith2 - refld[6 * i + 4] - 1u32) % nthu;
                                    let ith2y = 1u32 + (nthu + ith2 - refld[6 * i + 5] - 1u32) % nthu;
                                    let r3 = recos[dmod(ith2, r1d, nthu) as usize];
                                    if r3 < zero {
                                        let cx = (recos[(ith2x - 1u32) as usize] / dx).abs();
                                        let cy = (resin[(ith2y - 1u32) as usize] / dy).abs();
                                        let nrs = u32::cast_from(par[P_NRS]);
                                        for th1 in 0..nth {
                                            let ith = th1 as u32 + 1u32;
                                            let r1 = recos[dmod(ith, r1d, nthu) as usize];
                                            let r2 = ramp1 * a[ik * nth + th1];
                                            if r1 > zero && r2 > zero {
                                                let ith3 = 1u32 + (nthu / 2u32 + nthu + 2u32 * r1d - ith - 1u32) % nthu;
                                                let r4 = recos[dmod(ith2, ith3, nthu) as usize];
                                                if r4 > zero {
                                                    if rd2 == 0u32 {
                                                        s += rd3 * r2 * cgk * cx * par[P_FAC1] + rd4 * r2 * cgk * cy * par[P_FAC1];
                                                    } else if rd2 == 1u32 {
                                                        s += rd3 * r2 * cgk * cx * r4.abs() * par[P_FAC2] + rd4 * r2 * cgk * cy * r4.abs() * par[P_FAC2];
                                                    } else if rd2 == 2u32 {
                                                        let mut r4n = one;
                                                        let mut k = 0u32;
                                                        loop {
                                                            if k >= nrs {
                                                                break;
                                                            }
                                                            r4n *= r4;
                                                            k += 1u32;
                                                        }
                                                        s += rd3 * r2 * cgk * cx * r4n * par[P_FAC3] + rd4 * r2 * cgk * cy * r4n * par[P_FAC3];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                // diffuse reflection by sub-grid islands and icebergs
                                let tx = trnx[i];
                                let ty = trny[i];
                                if (par[P_REF + 1] > zero && tx + ty < two) || (par[P_REF + 3] > zero && bg > zero) {
                                    let sigk = rsig[nk + 1];
                                    let mut ramp2 = ramp0 * lc1;
                                    let mut ramp4 = ramp0 * lc3;
                                    if par[P_REF + 5] > zero {
                                        let ramp = max(par[P_C + 6] * tpi * fmeana / sigk, one).powf(par[P_REF + 9]) * ramp0;
                                        ramp2 = min(par[P_REF + 8], lc1 * ramp);
                                        let slope = max(lc3, par[P_C + 4]);
                                        let g = par[P_GRAV];
                                        let fm = max(fmeana, par[P_C + 4]);
                                        let michefac = par[P_C + 2] * g * g * (pow4::<F>(slope) * slope) / (max(emeana, par[P_C + 2]) * pow4::<F>(fm));
                                        let ramp0b = max(par[P_C + 1] * (michefac.ln() * par[P_INVLN10] + F::new(4.5f32)) + F::new(1.5f32) * michefac, par[P_C + 3]);
                                        let rampb = max(par[P_C + 6] * tpi * fmeana / sigk, one).powf(par[P_REF + 9]) * ramp0b;
                                        ramp4 = min(par[P_REF + 8], rampb);
                                    }
                                    let ex = max(min(one - (-bg * dx * par[P_C + 2]).exp(), one), zero);
                                    let ey = max(min(one - (-bg * dy * par[P_C + 2]).exp(), one), zero);
                                    let r2x = ramp2 * par[P_REF + 1] * max(min(one - tx, one), zero) + ramp4 * par[P_REF + 3] * ex;
                                    let r2y = ramp2 * par[P_REF + 1] * max(min(one - ty, one), zero) + ramp4 * par[P_REF + 3] * ey;
                                    let cx = (recos[th] / dx).abs();
                                    let cy = (resin[th] / dy).abs();
                                    for th1 in 0..nth {
                                        let r2 = a[ik * nth + th1];
                                        if r2 > zero {
                                            let r3 = recos[((nthu + ith2 - (th1 as u32 + 1u32)) % nthu) as usize];
                                            if r3 < zero {
                                                s += cgk * r2x * r2 * cx * par[P_FAC1] + cgk * r2y * r2 * cy * par[P_FAC1];
                                            }
                                        }
                                    }
                                }
                                v += dtg * s;
                            }
                        }
                        va[(ik * np + i) * nth + th] = v;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

use crate::device::Real;
use crate::ice::IceParams;
use crate::ref1::Ref1Params;
use crate::srce::bf;
use anyhow::{bail, ensure, Result};
use cubecl::client::Client;
use cubecl::server::Handle;

/// The device buffers [`ice_ref`] reads and writes for one node set of `np`
/// nodes: spectra `va` (`[(ik np + i) nth + ith]`, updated), wavenumbers and
/// group speeds `wn`, `cg` (`[(ik + 1) np + i]`, `ik = 0 .. NK - 1`), node
/// flags `nflag` (bit 2: `MAPSTA = 1`); for the ice terms the depth `depth`
/// (`max(DMIN, DW)`, unused by the ported terms), the ice concentration
/// `ice` and the maximum floe diameter `icef` (updated); for the reflection
/// `reflc` (`REFLEC(1:4)` with `BERG` applied, at `4 i + k`), `refld`
/// (`REFLD(1:6)` as `u32`, at `6 i + k`), the cell sizes `delx`, `dely`
/// (m), the transparencies `trnx`, `trny` and the iceberg damage `berg`.
/// The buffers of a part the kernel does not run may be `None`.
pub struct IceRefArgs<'a> {
    /// See [`IceRefArgs`].
    pub va: &'a Handle,
    /// See [`IceRefArgs`].
    pub wn: &'a Handle,
    /// See [`IceRefArgs`].
    pub cg: &'a Handle,
    /// See [`IceRefArgs`].
    pub nflag: &'a Handle,
    /// See [`IceRefArgs`].
    pub depth: Option<&'a Handle>,
    /// See [`IceRefArgs`].
    pub ice: Option<&'a Handle>,
    /// See [`IceRefArgs`].
    pub icef: Option<&'a Handle>,
    /// See [`IceRefArgs`].
    pub reflc: Option<&'a Handle>,
    /// See [`IceRefArgs`].
    pub refld: Option<&'a Handle>,
    /// See [`IceRefArgs`].
    pub delx: Option<&'a Handle>,
    /// See [`IceRefArgs`].
    pub dely: Option<&'a Handle>,
    /// See [`IceRefArgs`].
    pub trnx: Option<&'a Handle>,
    /// See [`IceRefArgs`].
    pub trny: Option<&'a Handle>,
    /// See [`IceRefArgs`].
    pub berg: Option<&'a Handle>,
}

/// The parameters and tables of the ice terms ([`IceParams`]) and of the
/// reflection ([`Ref1Params`]) on the device, and the launcher of
/// [`ice_ref`].
pub struct IceRefKernel<F: Real> {
    par: Handle,
    ktab: Handle,
    rsig: Handle,
    recos: Handle,
    resin: Handle,
    idave: Handle,
    fwtab: Handle,
    dummy: Handle,
    nfw: usize,
    nidave: usize,
    /// Number of frequencies.
    pub nk: usize,
    /// Number of directions.
    pub nth: usize,
    /// Units per cube.
    pub nu: usize,
    /// Whether the ice step runs.
    pub ice: bool,
    /// Whether the reflection runs.
    pub refl: bool,
    _f: std::marker::PhantomData<F>,
}

impl<F: Real> IceRefKernel<F> {
    /// Build the parameter buffers: `ice` the parameters of `IS2` and `IC2`
    /// (the ice step runs when given), `refl` those of `REF1` (the
    /// reflection runs when given), `nu` units per cube (a power of two at
    /// least `NK` and `NTH`). Refuses the options the kernel lacks:
    /// `IC_NUMERICS`, `ICEDISP`, `IS2DISP`, the matrix scattering
    /// (`IS2ISOSCAT = F`), `IC2DISPER`, `IC2TURBS > 0`.
    pub fn new(client: &Client, ice: Option<&IceParams>, refl: Option<&Ref1Params>, nu: usize) -> Result<IceRefKernel<F>> {
        const PI: f64 = std::f64::consts::PI;
        const TPI: f64 = 2.0 * PI;
        const YOUNG: f64 = 5.49e9;
        const POISSON: f64 = 0.3;
        const DENS: f64 = 1025.0;
        const DWAT: f64 = crate::ice::DWAT;
        const GRAV: f64 = crate::ice::GRAV;
        const THICK1: f64 = 0.1;
        const DTHICK: f64 = 0.25;
        let (nk, nth) = match (ice, refl) {
            (Some(p), _) => (p.nk, p.nth),
            (None, Some(r)) => (r.nk, r.nth),
            (None, None) => (1, 1),
        };
        if let (Some(p), Some(r)) = (ice, refl) {
            ensure!(p.nk == r.nk && p.nth == r.nth, "ice terms on {}x{}, reflection on {}x{}", p.nk, p.nth, r.nk, r.nth);
        }
        ensure!(nu.is_power_of_two() && nu >= nk && nu >= nth, "{nu} units per cube: a power of two at least NK = {nk} and NTH = {nth} is needed");
        let mut par = vec![0f64; NPAR];
        let mut ktab = vec![0f64; NKT * nk];
        let mut idave = vec![0f64];
        let mut fwtab = vec![0f64];
        par[P_PI] = PI;
        par[P_TPI] = TPI;
        par[P_TPIINV] = 1.0 / TPI;
        par[P_GRAV] = GRAV;
        par[P_INVLN10] = 1.0 / 10f64.ln();
        par[P_VISCM] = 1.83e-6;
        let consts = [0.07, 0.007, 1e-4, 0.005, 0.001, 1e-7, 0.75, 1.01, 0.2, 0.8];
        par[P_C..P_C + consts.len()].copy_from_slice(&consts);
        if let Some(p) = ice {
            if p.ic_numerics {
                bail!("IC_NUMERICS = T (ice terms in the dynamic loop) is not in the GPU kernel");
            }
            if p.iicedisp {
                bail!("ICEDISP = T is not in the GPU kernel");
            }
            if p.iicesmooth {
                bail!("ICESMOOTH = T is not in the GPU kernel");
            }
            if !p.with_is2 || !p.with_ic2 {
                bail!("the GPU kernel needs both switches IS2 and IC2");
            }
            let is2 = |i: usize| p.is2pars[i - 1];
            let ic2 = |i: usize| p.ic2pars[i - 1];
            if is2(6) > 0.5 {
                bail!("IS2DISP = T (Robinson-Palmer dispersion in W3SIS2) is not in the GPU kernel");
            }
            if is2(20) <= 0.5 {
                bail!("IS2ISOSCAT = F (scattering matrix) is not in the GPU kernel");
            }
            if ic2(1) > 0.5 {
                bail!("IC2DISPER = T is not in the GPU kernel");
            }
            if ic2(7) > 0.0 {
                bail!("IC2TURBS > 0 (southern ocean factor of W3SIC2) is not in the GPU kernel");
            }
            ensure!(p.fwtable.len() >= 2, "empty friction factor table");
            let iceh = p.iicehinit;
            let mp2 = 1.0 - POISSON * POISSON;
            par[P_IS2..P_IS2 + 24].copy_from_slice(&p.is2pars);
            par[P_IC2..P_IC2 + 8].copy_from_slice(&p.ic2pars);
            par[P_ICEH] = iceh;
            par[P_ICEDMAX] = 1000.0;
            par[P_ICEDMIN] = if is2(9) > 0.0 { is2(9) } else { p.icedmin };
            par[P_RC] = is2(16) * 4.0 * YOUNG * iceh.powi(3) / (3.0 * DWAT * GRAV * mp2);
            par[P_CTS] = 0.25 * iceh.max(p.iicehmin).powi(2);
            par[P_STRC] = is2(19) * mp2 / YOUNG;
            par[P_DFF] = 0.5 * ((PI.powi(4) * YOUNG * iceh.powi(3)) / (48.0 * DENS * GRAV * mp2)).powf(0.25);
            par[P_CREEP] = if is2(12) > 0.0 {
                let (b, n) = (is2(12), is2(15));
                -2.0 * (0.25 / (n + 2.0)) * 0.375 * b * iceh.powf(n + 2.0) * (YOUNG / (2.0 * b * mp2)).powf(n + 1.0) / (DWAT * GRAV)
            } else {
                1.0
            };
            par[P_NSUM] = (0.3f32 / (p.xfr as f32 - 1.0)).round().max(0.0) as f64;
            par[P_GAM] = 2.0 + is2(8).ln() / 2f64.ln();
            par[P_DELAB] = p.delab;
            par[P_ABMIN] = p.abmin;
            par[P_NOICE] = if iceh == 0.0 { 1.0 } else { 0.0 };
            // thickness interpolation of the scattering table (W3SIS2, 2.)
            let nthick = crate::ice::NTHICK;
            let xi = (iceh - THICK1) / DTHICK;
            let mut ind1 = 1 + xi.floor() as i64;
            let mut ind2 = ind1 + 1;
            let mut w = xi - (ind1 - 1) as f64;
            if ind1 < 1 {
                ind1 = 1;
                w = 0.0;
            } else if ind2 > nthick as i64 {
                ind2 = nthick as i64;
                ind1 = nthick as i64;
                w = 0.0;
            }
            let i1 = (ind1 - 1) as usize;
            let i2 = (ind2.max(1) - 1) as usize;
            // anelastic dissipation (Cole et al. 1995)
            let alpha_d = 0.54;
            let b_cole = 1.205e-9 * (is2(24) * 1.60218e-19 / (1.38064852e-23 * 268.15)).exp();
            let tau_d = b_cole / 0.07;
            for ik in 0..nk {
                let sg = p.sig[ik];
                let al = &p.sis2alphas;
                ktab[T_SIG * nk + ik] = sg;
                ktab[T_DDEN * nk + ik] = p.dden[ik];
                ktab[T_ALPH * nk + ik] = al[i1 + ik * nthick] * (1.0 - w) + al[i2 + ik * nthick] * w;
                ktab[T_SQ * nk + ik] = 2.0 * is2(4) * (-1.0 * is2(5) / sg).exp();
                let s_d = (sg * tau_d).ln();
                ktab[T_ANEL * nk + ik] = -5.0 * sg * is2(21) * alpha_d * (YOUNG / mp2).powi(2) * (iceh / 2.0).powi(3) / 3.0
                    / ((alpha_d * s_d).exp() + (-alpha_d * s_d).exp())
                    / (DWAT * GRAV)
                    * (1.0 / TPI);
            }
            idave = p.icedavetab.clone();
            fwtab = p.fwtable.clone();
        }
        let (mut rsig, mut recos, mut resin) = (vec![0f64; nk + 2], vec![0f64; nth], vec![0f64; nth]);
        if let Some(r) = refl {
            ensure!(r.refmap == 0.0, "REFMAP > 0 is not in the GPU kernel");
            par[P_REF..P_REF + 10].copy_from_slice(&r.refpars);
            let nrs = r.refpars[7].round() as i32;
            par[P_NRS] = nrs as f64;
            par[P_FAC1] = 1.0 / (0.5 * nth as f64);
            par[P_FAC2] = 1.57 / (0.5 * nth as f64);
            par[P_FAC3] = 2.0 / r.ecos.iter().map(|c| c.abs().powi(nrs)).sum::<f64>();
            ensure!(nrs >= 0, "REFCOSP_STRAIGHT < 0");
            for ik in 0..nk {
                ktab[T_RDDEN * nk + ik] = r.dden[ik];
            }
            rsig.copy_from_slice(&r.sig[..nk + 2]);
            recos.copy_from_slice(&r.ecos);
            resin.copy_from_slice(&r.esin);
        }
        let upd = |v: &[f64]| {
            let w: Vec<F> = v.iter().map(|&x| <F as crate::backend::Scalar>::of_f64(x)).collect();
            client.create_from_slice(F::as_bytes(&w))
        };
        Ok(IceRefKernel {
            par: upd(&par),
            ktab: upd(&ktab),
            rsig: upd(&rsig),
            recos: upd(&recos),
            resin: upd(&resin),
            idave: upd(&idave),
            fwtab: upd(&fwtab),
            dummy: client.create_from_slice(u32::as_bytes(&[0u32, 0u32])),
            nfw: fwtab.len(),
            nidave: idave.len(),
            nk,
            nth,
            nu,
            ice: ice.is_some(),
            refl: refl.is_some(),
            _f: std::marker::PhantomData,
        })
    }

    /// Launch [`ice_ref`] on `np` nodes for the global step `dtg`; nothing
    /// is launched when neither the ice step nor the reflection runs.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(&self, client: &Client, s: &IceRefArgs, np: usize, nk: usize, nth: usize, dtg: f32) -> Result<()> {
        if !self.ice && !self.refl {
            return Ok(());
        }
        ensure!(nk == self.nk && nth == self.nth, "kernel built for {}x{}, launched on {nk}x{nth}", self.nk, self.nth);
        let nspec = nk * nth;
        let need = |h: Option<&Handle>, on: bool, name: &str| -> Result<(Handle, usize)> {
            match (h, on) {
                (Some(h), true) => Ok((h.clone(), np)),
                (None, true) => bail!("ice_ref: buffer {name} missing"),
                (_, false) => Ok((self.dummy.clone(), 1)),
            }
        };
        let (depth, nd) = need(s.depth, self.ice, "depth")?;
        let (ice, ni) = need(s.ice, self.ice, "ice")?;
        let (icef, nf) = need(s.icef, self.ice, "icef")?;
        let (reflc, nr) = need(s.reflc, self.refl, "reflc")?;
        let (refld, nrd) = need(s.refld, self.refl, "refld")?;
        let (delx, ndx) = need(s.delx, self.refl, "delx")?;
        let (dely, ndy) = need(s.dely, self.refl, "dely")?;
        let (trnx, ntx) = need(s.trnx, self.refl, "trnx")?;
        let (trny, nty) = need(s.trny, self.refl, "trny")?;
        let (berg, nb) = need(s.berg, self.refl, "berg")?;
        let r4 = if self.refl { 4 } else { 1 };
        let r6 = if self.refl { 6 } else { 1 };
        unsafe {
            ice_ref::launch_unchecked::<F>(
                client,
                CubeCount::Static(np as u32, 1, 1),
                CubeDim::new_1d(self.nu as u32),
                bf(s.va, nspec * np),
                bf(s.wn, (nk + 2) * np),
                bf(s.cg, (nk + 2) * np),
                bf(&depth, nd),
                bf(&ice, ni),
                bf(&icef, nf),
                bf(&reflc, r4 * nr),
                bf(&refld, r6 * nrd),
                bf(&delx, ndx),
                bf(&dely, ndy),
                bf(&trnx, ntx),
                bf(&trny, nty),
                bf(&berg, nb),
                bf(s.nflag, np),
                bf(&self.par, NPAR),
                bf(&self.ktab, NKT * nk),
                bf(&self.rsig, nk + 2),
                bf(&self.recos, nth),
                bf(&self.resin, nth),
                bf(&self.idave, self.nidave),
                bf(&self.fwtab, self.nfw),
                dtg,
                np,
                nk,
                nth,
                self.nu,
                self.nfw,
                self.ice,
                self.refl,
            );
        }
        Ok(())
    }
}

/// The ice and reflection inputs of a run (`hww3-prop run --ice-ref
/// NAMELISTS`): the ice parameters from the namelists (`&SIS2`, `&SIC2`,
/// `&MISC`) and, when the grid directory holds the reflection geometry
/// written by `hww3-pre struct --nml` and `&REF1` asks for a reflection,
/// the reflection parameters and the per-node coefficients.
pub struct IceRefInputs {
    /// Ice parameters (the ice step runs where the ice concentration is
    /// positive).
    pub ice: Option<IceParams>,
    /// Reflection parameters and per-node inputs.
    pub refl: Option<(Ref1Params, RefNodes)>,
}

/// Per-node inputs of `W3SREF` on a mesh (in node order).
pub struct RefNodes {
    /// `REFLEC(1..4)` with `BERG` applied (no iceberg field: `REFLEC(4) = 0`), `4 np`.
    pub reflc: Vec<f64>,
    /// `REFLD(1..6)`, `6 np`.
    pub refld: Vec<u32>,
    /// Cell sizes (m) and transparencies.
    pub delx: Vec<f64>,
    /// See [`RefNodes::delx`].
    pub dely: Vec<f64>,
    /// See [`RefNodes::delx`].
    pub trnx: Vec<f64>,
    /// See [`RefNodes::delx`].
    pub trny: Vec<f64>,
}

impl IceRefInputs {
    /// Read the namelists `nml` and the reflection files of the grid
    /// directory `dir` (`raw/ref1_*`, absent on HWW3 grids).
    pub fn load(dir: &std::path::Path, nml: &std::path::Path, grid: &crate::grid::Grid) -> Result<IceRefInputs> {
        let n = crate::ww3grid::Namelists::read(nml)?;
        let s = &grid.spec;
        let sp = crate::spec::Spec::new(s.nk, s.nth, s.freq1 as f32, s.xfr as f32, s.thoff as f32);
        let ice = Some(IceParams::from_namelists(&n, &sp)?);
        let raw = dir.join("raw");
        let np = grid.lon.len();
        let refl = if raw.join("ref1_reflc.f64").exists() {
            let rp = Ref1Params::from_namelists(&n, s)?;
            let f64s = |name: &str, len: usize| -> Result<Vec<f64>> {
                let b = std::fs::read(raw.join(name))?;
                ensure!(b.len() == 8 * len, "{name}: {} bytes for {len} values", b.len());
                Ok(b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
            };
            let b = std::fs::read(raw.join("ref1_refld.u32"))?;
            ensure!(b.len() == 4 * 6 * np, "ref1_refld.u32: {} bytes for {np} nodes", b.len());
            let refld: Vec<u32> = b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let mut reflc = f64s("ref1_reflc.f64", 4 * np)?;
            for k in 0..np {
                // REFLEC(4) = BERG REFLC(4), no iceberg field
                reflc[4 * k + 3] = 0.0;
            }
            let any = (0..np).any(|k| crate::ref1::is_called(&[reflc[4 * k], reflc[4 * k + 1], reflc[4 * k + 2], reflc[4 * k + 3]], 0.0));
            if any {
                Some((rp, RefNodes { reflc, refld, delx: f64s("ref1_delx.f64", np)?, dely: f64s("ref1_dely.f64", np)?, trnx: f64s("ref1_trnx.f64", np)?, trny: f64s("ref1_trny.f64", np)? }))
            } else {
                None
            }
        } else {
            None
        };
        Ok(IceRefInputs { ice, refl })
    }
}

/// The ice and reflection step of a run on the device: the kernel, the
/// floe diameter `ICEF` of every node (initial value 1000 m, as WW3 without
/// a floe-size field; not in the restart files, as in WW3) and the
/// reflection inputs.
pub struct IceRefState<F: Real> {
    kernel: IceRefKernel<F>,
    icef: Option<Handle>,
    refl: Option<[Handle; 7]>,
}

impl<F: Real> IceRefState<F> {
    /// Build the kernel and upload the per-node inputs of `inp` for `np`
    /// nodes.
    pub fn new(client: &Client, inp: &IceRefInputs, np: usize, nu: usize) -> Result<IceRefState<F>> {
        let kernel = IceRefKernel::<F>::new(client, inp.ice.as_ref(), inp.refl.as_ref().map(|r| &r.0), nu)?;
        let upd = |v: &[f64]| {
            let w: Vec<F> = v.iter().map(|&x| <F as crate::backend::Scalar>::of_f64(x)).collect();
            client.create_from_slice(F::as_bytes(&w))
        };
        let icef = inp.ice.as_ref().map(|_| upd(&vec![1000.0; np]));
        let refl = inp.refl.as_ref().map(|(_, r)| {
            [
                upd(&r.reflc),
                client.create_from_slice(u32::as_bytes(&r.refld)),
                upd(&r.delx),
                upd(&r.dely),
                upd(&r.trnx),
                upd(&r.trny),
                upd(&vec![0.0; np]),
            ]
        });
        Ok(IceRefState { kernel, icef, refl })
    }

    /// Apply the step to the spectra `va` after the source integration.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(&self, client: &Client, va: &Handle, wn: &Handle, cg: &Handle, nflag: &Handle, dw: &Handle, ice: &Handle, np: usize, nk: usize, nth: usize, dtg: f32) -> Result<()> {
        let r = self.refl.as_ref();
        let args = IceRefArgs {
            va,
            wn,
            cg,
            nflag,
            depth: Some(dw),
            ice: self.icef.as_ref().map(|_| ice),
            icef: self.icef.as_ref(),
            reflc: r.map(|a| &a[0]),
            refld: r.map(|a| &a[1]),
            delx: r.map(|a| &a[2]),
            dely: r.map(|a| &a[3]),
            trnx: r.map(|a| &a[4]),
            trny: r.map(|a| &a[5]),
            berg: r.map(|a| &a[6]),
        };
        self.kernel.launch(client, &args, np, nk, nth, dtg)
    }
}
