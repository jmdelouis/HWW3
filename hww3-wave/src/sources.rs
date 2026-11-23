//! CPU transcriptions of the WW3 source terms of the HWW3 configuration
//! with the simplified ST4 (milestone M2a), in double precision, as
//! references for the GPU kernels.
//!
//! The parameters and tables are those WW3 holds after `W3IOGR`, read from
//! the source-term harness ([`crate::harness::Params`]). Spectra are
//! `A(ISPEC)` with `ISPEC = ITH + (IK-1) NTH` stored at `ith + ik nth`
//! (0-based), as `VA` in WW3.
//!
//! So far: `W3SPR4` (mean parameters and u*), `CALC_USTAR` (u* by Newton
//! iterations, `SINTABLE = 0`), `W3SIN4` (wind input with swell damping and
//! the tail stress integrated in line), `W3SDS4` (saturation-based
//! whitecapping, `SDSBCHOICE = 1`, without the cumulative term), `W3SLN1`,
//! `W3SDB1`, `W3SBT4`, `W3SNL1` (discrete interaction approximation,
//! [`Dia`]) and the explicit integration of `W3SRCE` ([`srce`]).

use crate::harness::Params;
use anyhow::{bail, Result};

/// Parameters of `W3SPR4`, `CALC_USTAR` and `W3SIN4`.
#[derive(Clone, Debug)]
pub struct St4 {
    /// Number of frequencies.
    pub nk: usize,
    /// Number of directions.
    pub nth: usize,
    /// `SIG(0..NK+1)`.
    pub sig: Vec<f64>,
    /// `DDEN(1..NK)` at `ik - 1`.
    pub dden: Vec<f64>,
    /// `SIG2(ISPEC)`.
    pub sig2: Vec<f64>,
    /// `DDEN2(ISPEC)`.
    pub dden2: Vec<f64>,
    /// `ECOS(ISPEC)`, `ESIN(ISPEC)`.
    pub ecos: Vec<f64>,
    /// See [`St4::ecos`].
    pub esin: Vec<f64>,
    /// Direction step.
    pub dth: f64,
    /// Tail factors of `W3SPR4`.
    pub fte: f64,
    /// See [`St4::fte`].
    pub ftf: f64,
    /// See [`St4::fte`].
    pub sstxftwn: f64,
    /// See [`St4::fte`].
    pub sstxftftail: f64,
    /// Exponents of the mean wavenumber and frequency.
    pub wwnmeanp: f64,
    /// See [`St4::wwnmeanp`].
    pub wwnmeanptail: f64,
    /// `SSDSC(12)`: exponent of the mean direction weights.
    pub ssdsc12: f64,
    /// Wind height, Charnock constant, `BETAMAX`, `ZALP`, `TAUWSHELTER`.
    pub zzwnd: f64,
    /// See [`St4::zzwnd`].
    pub aalpha: f64,
    /// See [`St4::zzwnd`].
    pub bbeta: f64,
    /// See [`St4::zzwnd`].
    pub zzalp: f64,
    /// See [`St4::zzwnd`].
    pub ttauwshelter: f64,
    /// `SSWELLF(1..7)` at `i - 1`.
    pub sswellf: [f64; 7],
    /// `SINTHP`, `Z0RAT`, `Z0MAX`.
    pub ssinthp: f64,
    /// See [`St4::ssinthp`].
    pub zz0rat: f64,
    /// See [`St4::ssinthp`].
    pub zz0max: f64,
    /// `SINTAILPAR(1..5)` at `i - 1`: `SINTABLE`, `SINTAIL1`, `SINTAIL2`, `TAUWBUG`, `VISCSTRESS`.
    pub sintailpar: [f64; 5],
    /// `CAPCHNK(1..5)`.
    pub capchnk: [f64; 5],
    /// Constants.
    pub grav: f64,
    /// See [`St4::grav`].
    pub kappa: f64,
    /// See [`St4::grav`].
    pub nu_air: f64,
    /// Friction-factor table `FWTABLE(0..SIZEFWTABLE)`, `DELAB`, `ABMIN`.
    pub fwtable: Vec<f64>,
    /// See [`St4::fwtable`].
    pub delab: f64,
    /// See [`St4::fwtable`].
    pub abmin: f64,
    /// `SSDSC(1..21)` at `i - 1`.
    pub ssdsc: Vec<f64>,
    /// Saturation threshold, exponent and angular width of `W3SDS4`.
    pub ssdsbr: f64,
    /// See [`St4::ssdsbr`].
    pub ssdsp: f64,
    /// See [`St4::ssdsbr`].
    pub ssdsdth: f64,
    /// `SSDSBM(0..4)`.
    pub ssdsbm: [f64; 5],
    /// `SATINDICES(j, ith)` (1-based direction offsets) at `j + ith nsat`.
    pub satindices: Vec<i32>,
    /// `SATWEIGHTS(j, ith)` at `j + ith nsat`.
    pub satweights: Vec<f64>,
    /// First dimension of the saturation tables.
    pub nsat: usize,
    /// `W3SLN1`: `SLNC1`, `FSPM`, `FSHF`.
    pub slnc1: f64,
    /// See [`St4::slnc1`].
    pub fspm: f64,
    /// See [`St4::slnc1`].
    pub fshf: f64,
    /// `W3SDB1`: `SDBC1`, `SDBC2`, `FDONLY`.
    pub sdbc1: f64,
    /// See [`St4::sdbc1`].
    pub sdbc2: f64,
    /// See [`St4::sdbc1`].
    pub fdonly: bool,
    /// `W3SBT4`: `SBTCX(1..10)` at `i - 1`.
    pub sbtcx: [f64; 10],
    /// Specific gravity of the sediment.
    pub sed_sg: f64,
    /// `ERFTABLE(0..SIZEERFTABLE)`, `DELXERF`, `XERFMAX` of `W3SBT4`.
    pub erftable: Vec<f64>,
    /// See [`St4::erftable`].
    pub delxerf: f64,
    /// See [`St4::erftable`].
    pub xerfmax: f64,
    /// Stress tables of `INSIN4` (`SINTABLE = 1`, full ST4), `None` with
    /// `SINTABLE = 0`.
    pub tables: Option<StressTables>,
    /// `CUMULW(IS2, IS)` at `is2 + is NSPEC` (cumulative breaking term,
    /// `SDSCUM < 0`), empty without it.
    pub cumulw: Vec<f64>,
    /// `DIKCUMUL`: frequency offset of the cumulative term.
    pub dikcumul: i32,
}

/// The stress tables of `INSIN4` (`TABU_STRESS`, `TABU_TAUHF`,
/// `TABU_TAUHF2`), Fortran order from index 0.
#[derive(Clone, Debug)]
pub struct StressTables {
    /// `TAUT(0:ITAUMAX, 0:JUMAX)`: u* from the wave stress and the wind.
    pub taut: Vec<f64>,
    /// `TAUHFT(0:IUSTAR, 0:IALPHA)`: tail stress without sheltering.
    pub tauhft: Vec<f64>,
    /// `TAUHFT2(0:IUSTAR, 0:IALPHA, 0:ILEVTAIL)`: tail stress with
    /// sheltering (empty when `TAUWSHELTER = 0`).
    pub tauhft2: Vec<f64>,
    /// `ITAUMAX`, `JUMAX`, `IUSTAR`, `IALPHA`, `ILEVTAIL`.
    pub itaumax: usize,
    /// See [`StressTables::itaumax`].
    pub jumax: usize,
    /// See [`StressTables::itaumax`].
    pub iustar: usize,
    /// See [`StressTables::itaumax`].
    pub ialpha: usize,
    /// See [`StressTables::itaumax`].
    pub ilevtail: usize,
    /// Table steps `DELUST`, `DELALP`, `DELTAUW`, `DELU`, `DELTAIL` and the
    /// largest stress `TAUWMAX`.
    pub delust: f64,
    /// See [`StressTables::delust`].
    pub delalp: f64,
    /// See [`StressTables::delust`].
    pub deltauw: f64,
    /// See [`StressTables::delust`].
    pub delu: f64,
    /// See [`StressTables::delust`].
    pub deltail: f64,
    /// See [`StressTables::delust`].
    pub tauwmax: f64,
}

impl StressTables {
    fn read(p: &Params) -> Result<StressTables> {
        let st = p.shape("TAUT")?.to_vec();
        let sh = p.shape("TAUHFT")?.to_vec();
        if st.len() != 2 || sh.len() != 2 || p.get_lbounds("TAUT")? != [0, 0] || p.get_lbounds("TAUHFT")? != [0, 0] {
            bail!("TAUT and TAUHFT must be 2-D tables from 0");
        }
        let (tauhft2, ilevtail) = if p.has("TAUHFT2") {
            let s2 = p.shape("TAUHFT2")?.to_vec();
            if s2.len() != 3 || s2[0] != sh[0] || s2[1] != sh[1] {
                bail!("TAUHFT2 shape {s2:?} unlike TAUHFT {sh:?}");
            }
            (v64(p.r("TAUHFT2")?), s2[2] - 1)
        } else {
            (vec![], 0)
        };
        let d = p.r("TABDEL")?;
        Ok(StressTables {
            taut: v64(p.r("TAUT")?),
            tauhft: v64(p.r("TAUHFT")?),
            tauhft2,
            itaumax: st[0] - 1,
            jumax: st[1] - 1,
            iustar: sh[0] - 1,
            ialpha: sh[1] - 1,
            ilevtail,
            delust: d[0] as f64,
            delalp: d[1] as f64,
            deltauw: d[2] as f64,
            delu: d[3] as f64,
            deltail: d[4] as f64,
            tauwmax: d[5] as f64,
        })
    }
}

fn v64(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| v as f64).collect()
}

impl St4 {
    /// The parameters of the harness parameter file.
    pub fn new(p: &Params) -> Result<St4> {
        let nk = p.i1("NK")? as usize;
        let nth = p.i1("NTH")? as usize;
        let tables = if p.r1("SINTAILPAR")? > 0.5 { Some(StressTables::read(p)?) } else { None };
        let cumulw = if p.r("SSDSC")?[2] != 0.0 { v64(p.r("CUMULW")?) } else { vec![] };
        if !cumulw.is_empty() && cumulw.len() != (nk * nth) * (nk * nth) {
            bail!("CUMULW holds {} values for NSPEC {}", cumulw.len(), nk * nth);
        }
        let a7 = |n: &str| -> Result<[f64; 7]> {
            let v = p.r(n)?;
            let mut a = [0f64; 7];
            for (k, x) in a.iter_mut().enumerate() {
                *x = v.get(k).copied().unwrap_or(0.0) as f64;
            }
            Ok(a)
        };
        let a5 = |n: &str| -> Result<[f64; 5]> {
            let v = p.r(n)?;
            let mut a = [0f64; 5];
            for (k, x) in a.iter_mut().enumerate() {
                *x = v.get(k).copied().unwrap_or(0.0) as f64;
            }
            Ok(a)
        };
        let sat = (p.r1("SSDSDTH")? as f64) < 180.0;
        Ok(St4 {
            nk,
            nth,
            sig: v64(p.r("SIG")?),
            dden: v64(p.r("DDEN")?),
            sig2: v64(p.r("SIG2")?),
            dden2: v64(p.r("DDEN2")?),
            ecos: v64(p.r("ECOS")?),
            esin: v64(p.r("ESIN")?),
            dth: p.r1("DTH")? as f64,
            fte: p.r1("FTE")? as f64,
            ftf: p.r1("FTF")? as f64,
            sstxftwn: p.r1("SSTXFTWN")? as f64,
            sstxftftail: p.r1("SSTXFTFTAIL")? as f64,
            wwnmeanp: p.r1("WWNMEANP")? as f64,
            wwnmeanptail: p.r1("WWNMEANPTAIL")? as f64,
            ssdsc12: p.r("SSDSC")?[11] as f64,
            zzwnd: p.r1("ZZWND")? as f64,
            aalpha: p.r1("AALPHA")? as f64,
            bbeta: p.r1("BBETA")? as f64,
            zzalp: p.r1("ZZALP")? as f64,
            ttauwshelter: p.r1("TTAUWSHELTER")? as f64,
            sswellf: a7("SSWELLF")?,
            ssinthp: p.r1("SSINTHP")? as f64,
            zz0rat: p.r1("ZZ0RAT")? as f64,
            zz0max: p.r1("ZZ0MAX")? as f64,
            sintailpar: a5("SINTAILPAR")?,
            capchnk: a5("CAPCHNK")?,
            grav: p.r1("GRAV")? as f64,
            kappa: p.r1("KAPPA")? as f64,
            nu_air: p.r1("NU_AIR")? as f64,
            fwtable: v64(p.r("FWTABLE")?),
            delab: p.r1("DELAB")? as f64,
            abmin: p.r1("ABMIN")? as f64,
            ssdsc: v64(p.r("SSDSC")?),
            ssdsbr: p.r1("SSDSBR")? as f64,
            ssdsp: p.r1("SSDSP")? as f64,
            ssdsdth: p.r1("SSDSDTH")? as f64,
            ssdsbm: a5("SSDSBM")?,
            satindices: if sat { p.i("SATINDICES")?.to_vec() } else { vec![] },
            satweights: if sat { v64(p.r("SATWEIGHTS")?) } else { vec![] },
            nsat: if sat { p.shape("SATINDICES")?[0] } else { 0 },
            slnc1: p.r1("SLNC1")? as f64,
            fspm: p.r1("FSPM")? as f64,
            fshf: p.r1("FSHF")? as f64,
            sdbc1: p.r1("SDBC1")? as f64,
            sdbc2: p.r1("SDBC2")? as f64,
            fdonly: p.i1("FDONLY")? != 0,
            sbtcx: {
                let v = p.r("SBTCX")?;
                let mut a = [0f64; 10];
                for (k, x) in a.iter_mut().enumerate() {
                    *x = v.get(k).copied().unwrap_or(0.0) as f64;
                }
                a
            },
            sed_sg: p.r1("SED_SG")? as f64,
            erftable: v64(p.r("ERFTABLE")?),
            delxerf: p.r1("DELXERF")? as f64,
            xerfmax: p.r1("XERFMAX")? as f64,
            tables,
            cumulw,
            dikcumul: if p.has("DIKCUMUL") { p.i1("DIKCUMUL")? } else { 0 },
        })
    }

    /// `SIG(IK)` for `IK = 1..NK` (1-based, as in WW3).
    fn s(&self, ik: usize) -> f64 {
        self.sig[ik]
    }
}

/// Outputs of `W3SPR4` (named as in WW3).
#[derive(Clone, Copy, Debug, Default)]
pub struct Spr4Out {
    /// Total energy with the tail.
    pub emean: f64,
    /// Mean frequency (Hz).
    pub fmean: f64,
    /// Mean frequency with the `WWNMEANPTAIL` weight.
    pub fmean1: f64,
    /// Mean wavenumber.
    pub wnmean: f64,
    /// Largest spectral density.
    pub amax: f64,
    /// u* (inout in WW3).
    pub ustar: f64,
    /// u* direction (= wind direction).
    pub usdir: f64,
    /// Drag coefficient.
    pub cd: f64,
    /// Roughness length.
    pub z0: f64,
    /// Charnock coefficient.
    pub charn: f64,
    /// Mean frequency of the wind sea.
    pub fmeanws: f64,
    /// Mean direction of the long waves.
    pub dlwmean: f64,
}

impl Spr4Out {
    /// The values in the order of [`crate::harness::Spr4::NAMES`] (TAUWX and TAUWY being inputs).
    pub fn as_array(&self, tauwx: f64, tauwy: f64) -> [f64; 14] {
        [
            self.emean,
            self.fmean,
            self.fmean1,
            self.wnmean,
            self.amax,
            self.ustar,
            self.usdir,
            tauwx,
            tauwy,
            self.cd,
            self.z0,
            self.charn,
            self.fmeanws,
            self.dlwmean,
        ]
    }
}

/// `CALC_USTAR`: u* from the wind and the wave-supported stress `tauw`,
/// by bilinear interpolation in `TAUT` (`SINTABLE = 1`) or by 10 Newton
/// iterations on the Charnock relation (`SINTABLE = 0`). Returns
/// `(USTAR, Z0, CHARN)`.
pub fn calc_ustar(p: &St4, windspeed: f64, tauw: f64) -> (f64, f64, f64) {
    if let Some(t) = &p.tables {
        let tl = tauw.min(t.tauwmax).max(0.0);
        let xi = tl.sqrt() / t.deltauw;
        let ind = (t.itaumax - 1).min(xi as usize);
        let deli1 = 1f64.min(xi - ind as f64);
        let deli2 = 1.0 - deli1;
        let xj = windspeed / t.delu;
        let j = (t.jumax - 1).min(xj as usize);
        let delj1 = 1f64.min(xj - j as f64);
        let delj2 = 1.0 - delj1;
        let n = t.itaumax + 1;
        let tt = |i: usize, j: usize| t.taut[i + j * n];
        let ustar = (tt(ind, j) * deli2 + tt(ind + 1, j) * deli1) * delj2 + (tt(ind, j + 1) * deli2 + tt(ind + 1, j + 1) * deli1) * delj1;
        return z0_charn(p, windspeed, ustar, p.aalpha);
    }
    const NITER: usize = 10;
    const XM: f64 = 0.50;
    const EPS1: f64 = 0.00001;
    let mut chath = p.aalpha;
    let xmin = if p.capchnk[0] == 1.0 {
        chath = p.capchnk[1] + 0.5 * (p.capchnk[2] - p.capchnk[1]) * (1.0 - ((windspeed - p.capchnk[3]) / p.capchnk[4]).tanh());
        0.15 * (p.capchnk[2] - chath)
    } else {
        0.0
    };
    let tauw_local = tauw;
    let cdrag = 0.0012875f64;
    let wcd = cdrag.sqrt();
    let ustold = windspeed * wcd;
    let mut tauold = (ustold * ustold).max(tauw_local + EPS1);
    let mut ust = 0.0;
    for _ in 0..NITER {
        let x = (tauw_local / tauold).max(xmin);
        ust = tauold.sqrt();
        let mut zz00 = chath * tauold / p.grav;
        if p.zz0max != 0.0 {
            zz00 = zz00.min(p.zz0max);
        }
        let mut zz0 = zz00 / (1.0 - x).powf(XM);
        let znu = 0.11 * p.nu_air / ust.max(1e-6);
        zz0 += p.sintailpar[4] * znu;
        let l = (p.zzwnd / zz0).ln();
        let f = ust - p.kappa * windspeed / l;
        let delf = 1.0 - p.kappa * windspeed / (l * l) * 2.0 / ust * (1.0 - (XM + 1.0) * x) / (1.0 - x);
        ust -= f / delf;
        tauold = (ust * ust).max(tauw_local + EPS1);
    }
    z0_charn(p, windspeed, ust, chath)
}

/// The end of `CALC_USTAR`: roughness and Charnock coefficient from u*.
fn z0_charn(p: &St4, windspeed: f64, ustar: f64, chath: f64) -> (f64, f64, f64) {
    let (z0, mut charn);
    if ustar > 0.001 {
        let sqrtcdm1 = (windspeed / ustar).min(100.0);
        z0 = p.zzwnd * (-p.kappa * sqrtcdm1).exp();
        charn = p.grav * z0 / (ustar * ustar);
    } else {
        z0 = if ustar > 0.0 { p.zzwnd * (-p.kappa * (windspeed / ustar).min(100.0)).exp() } else { chath * 0.001 * 0.001 / p.grav };
        charn = chath;
    }
    if p.capchnk[0] == 1.0 {
        charn = charn.min(0.09);
        if charn < chath {
            charn = chath;
        }
    }
    (ustar, z0, charn)
}

/// `W3SPR4`: mean parameters of spectrum `a`, then u*, z0 and the Charnock
/// coefficient from the wind `u` and the wave-supported stress
/// `(tauwx, tauwy)`; `llws` flags the wind-sea bins (from `W3SIN4`).
#[allow(clippy::too_many_arguments)]
pub fn spr4(p: &St4, a: &[f64], cg: &[f64], wn: &[f64], u: f64, udir: f64, ustar_in: f64, tauwx: f64, tauwy: f64, llws: &[bool]) -> Spr4Out {
    let (nk, nth) = (p.nk, p.nth);
    let tpiinv = 1.0 / std::f64::consts::TAU;
    let _ustar = ustar_in.max(0.0001);
    let (mut emean, mut emeanws, mut fmeanws, mut fmean, mut fmean1, mut wnmean, mut amax) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0f64);
    let (mut elcs, mut elsn) = (0.0, 0.0);
    let mut eb = vec![0f64; nk];
    let mut eb2 = vec![0f64; nk];
    for ik in 0..nk {
        let sigfac = p.s(ik + 1).powf(p.ssdsc12) * p.dden[ik] / cg[ik];
        for ith in 0..nth {
            let is = ith + ik * nth;
            eb[ik] += a[is];
            elcs += a[is] * p.ecos[is] * sigfac;
            elsn += a[is] * p.esin[is] * sigfac;
            if llws[is] {
                eb2[ik] += a[is];
            }
            amax = amax.max(a[is]);
        }
    }
    let dlwmean = elsn.atan2(elcs);
    for ik in 0..nk {
        eb[ik] *= p.dden[ik] / cg[ik];
        eb2[ik] *= p.dden[ik] / cg[ik];
        let s = p.s(ik + 1);
        emean += eb[ik];
        fmean += eb[ik] / s;
        fmean1 += eb[ik] * s.powf(2.0 * p.wwnmeanptail);
        wnmean += eb[ik] * wn[ik].powf(p.wwnmeanp);
        emeanws += eb2[ik];
        fmeanws += eb2[ik] * s.powf(2.0 * p.wwnmeanptail);
    }
    let mut eband = eb[nk - 1] / p.dden[nk - 1];
    emean += eband * p.fte;
    fmean += eband * p.ftf;
    fmean1 += eband * p.sstxftftail;
    wnmean += eband * p.sstxftwn;
    eband = eb2[nk - 1] / p.dden[nk - 1];
    emeanws += eband * p.fte;
    fmeanws += eband * p.sstxftftail;
    fmean = tpiinv * emean / fmean.max(1e-7);
    fmean1 = if fmean1 < 1e-7 {
        tpiinv * p.s(nk)
    } else {
        tpiinv * (fmean1.max(1e-7) / emean.max(1e-7)).powf(1.0 / (2.0 * p.wwnmeanptail))
    };
    wnmean = (wnmean.max(1e-7) / emean.max(1e-7)).powf(1.0 / p.wwnmeanp);
    fmeanws = if fmeanws < 1e-7 || emeanws < 1e-7 {
        tpiinv * p.s(nk)
    } else {
        tpiinv * (fmeanws.max(1e-7) / emeanws.max(1e-7)).powf(1.0 / (2.0 * p.wwnmeanptail))
    };
    let tauw = (tauwx * tauwx + tauwy * tauwy).sqrt();
    let (ustar, z0, charn) = calc_ustar(p, u, tauw);
    let unz = u.max(0.01);
    let cd = (ustar / unz).powi(2);
    Spr4Out { emean, fmean, fmean1, wnmean, amax, ustar, usdir: udir, cd, z0, charn, fmeanws, dlwmean }
}

/// Outputs of `W3SIN4`.
#[derive(Clone, Debug, Default)]
pub struct Sin4Out {
    /// Source term `S = D A`.
    pub s: Vec<f64>,
    /// Diagonal term.
    pub d: Vec<f64>,
    /// Wave-supported stress (with the tail).
    pub tauwx: f64,
    /// See [`Sin4Out::tauwx`].
    pub tauwy: f64,
    /// Stress given back to the air by swell damping.
    pub tauwnx: f64,
    /// See [`Sin4Out::tauwnx`].
    pub tauwny: f64,
    /// Wind-sea flags.
    pub llws: Vec<bool>,
}

/// `W3SIN4` with `SINTABLE = 0`: Janssen's input with sheltering and the
/// swell damping of Ardhuin et al. (2010), the high-frequency tail stress
/// integrated in line over 50 wavenumbers. `k` is the wavenumber of each
/// bin, `usdir` the direction of u* (the wind direction in `W3SRCE`).
#[allow(clippy::too_many_arguments)]
pub fn sin4(p: &St4, a: &[f64], cg: &[f64], k: &[f64], u: f64, ustar: f64, drat: f64, usdir: f64, z0: f64) -> Sin4Out {
    let _ = u;
    let (nk, nth) = (p.nk, p.nth);
    let nspec = nk * nth;
    let (grav, kappa, nu_air) = (p.grav, p.kappa, p.nu_air);
    let tpi = std::f64::consts::TAU;
    const JTOT: usize = 50;
    const KM: f64 = 363.0;
    let sw = &p.sswellf;
    let mut d = vec![0f64; nspec];
    let mut llws = vec![false; nspec];
    let const1 = p.bbeta / (kappa * kappa);
    let const0 = const1 * drat;
    let z0visc = 0.1 * nu_air / ustar.max(0.0001);
    let z0noz = z0visc.max(p.zz0rat * z0);
    let (mut uorb, mut aorb) = (0.0, 0.0);
    for ik in 0..nk {
        let eb: f64 = a[ik * nth..(ik + 1) * nth].iter().sum();
        let s = p.s(ik + 1);
        uorb += eb * s * s * p.dden[ik] / cg[ik];
        aorb += eb * p.dden[ik] / cg[ik];
    }
    uorb = 2.0 * uorb.sqrt();
    let aorb1 = 2.0 * aorb.powf(1.0 - 0.5 * sw[5]);
    let re = 4.0 * uorb * aorb1 / nu_air;
    let (pturb, pvisc) = if sw[3] > 0.0 {
        if sw[6] > 0.0 {
            let smooth = 0.5 * ((re - sw[3]) / sw[6]).tanh();
            (0.5 + smooth, 0.5 - smooth)
        } else if re <= sw[3] {
            (0.0, 1.0)
        } else {
            (1.0, 0.0)
        }
    } else {
        (1.0, 1.0)
    };
    let (fw, fu, fud);
    if sw[1] == 0.0 {
        fw = sw[2].abs().max(0.0);
        fu = 0.0;
        fud = 0.0;
    } else {
        fu = sw[2].abs();
        fud = sw[1];
        let aorb2 = 2.0 * aorb.sqrt();
        let xi = ((aorb2 / z0noz).max(3.0).log10() - p.abmin) / p.delab;
        let size = p.fwtable.len() - 1;
        let ind = (size - 1).min(xi as usize);
        let deli1 = 1f64.min(xi - ind as f64);
        let deli2 = 1.0 - deli1;
        fw = p.fwtable[ind] * deli2 + p.fwtable[ind + 1] * deli1;
    }
    let ust = ustar;
    let taux = ust * ust * usdir.cos();
    let tauy = ust * ust * usdir.sin();
    let (mut sx, mut sy, mut snx, mut sny) = (0.0, 0.0, 0.0, 0.0);
    let (mut cosu, mut sinu) = (1.0, 0.0);
    for ik in 0..nk {
        let taupx = taux - p.ttauwshelter.abs() * sx;
        let taupy = tauy - p.ttauwshelter.abs() * sy;
        let ustp = (taupx * taupx + taupy * taupy).powf(0.25).min(ust.max(0.3));
        let usdirp = taupy.atan2(taupx);
        cosu = usdirp.cos();
        sinu = usdirp.sin();
        let is0 = ik * nth;
        let cm = k[is0] / p.sig2[is0];
        let ucn = ustp * cm + p.zzalp;
        let s = p.s(ik + 1);
        let const2 = p.dden2[is0] / cg[ik] * grav / (s / k[is0] * drat);
        let cnst = p.sig2[is0] * const0;
        let zcn = (k[is0] * z0).ln();
        let swellcoefv = -sw[4] * drat * 2.0 * k[is0] * (2.0 * nu_air * p.sig2[is0]).sqrt();
        let swellcoeft = -drat * sw[0] * 16.0 * p.sig2[is0] * p.sig2[is0] / grav;
        for ith in 0..nth {
            let is = is0 + ith;
            let coswind = p.ecos[is] * cosu + p.esin[is] * sinu;
            let mut dst;
            if coswind > 0.01 {
                let x = coswind * ucn;
                let zarg = kappa / x;
                let zlog = zcn + zarg;
                if zlog < 0.0 {
                    dst = cnst * zlog.exp() * zlog.powi(4) * ucn * ucn * coswind.powf(p.ssinthp);
                    llws[is] = true;
                } else {
                    dst = 0.0;
                    llws[is] = false;
                }
                if 28.0 * cm * ustar * coswind >= 1.0 {
                    llws[is] = true;
                }
            } else {
                dst = 0.0;
                llws[is] = false;
            }
            if (sw[0] != 0.0 && dst < 1e-7 * p.sig2[is]) || sw[2] > 0.0 {
                let dvisc = swellcoefv;
                let dturb = swellcoeft * (fw * uorb + (fu + fud * coswind) * ustp);
                dst += pturb * dturb + pvisc * dvisc;
            }
            let temp2 = const2 * dst * a[is];
            if dst < 0.0 {
                snx += temp2 * p.ecos[is];
                sny += temp2 * p.esin[is];
            } else {
                sx += temp2 * p.ecos[is];
                sy += temp2 * p.esin[is];
            }
            d[is] = dst;
        }
    }
    let s: Vec<f64> = d.iter().zip(a).map(|(d, a)| d * a).collect();
    let taupx = taux - p.ttauwshelter.abs() * sx;
    let taupy = tauy - p.ttauwshelter.abs() * sy;
    let ustp = (taupx * taupx + taupy * taupy).powf(0.25);
    let usdirp = taupy.atan2(taupx);
    let ust = ustp;
    // tail level from the last frequency, with the direction of its loop
    let snk = p.s(nk);
    let c0 = p.dth * snk.powi(5) / (grav * grav * tpi) * tpi * snk / cg[nk - 1];
    let mut temp = 0.0;
    for ith in 0..nth {
        let is = ith + (nk - 1) * nth;
        let coswind = p.ecos[is] * cosu + p.esin[is] * sinu;
        temp += a[is] * coswind.max(0.0).powi(3);
    }
    let levtail0 = c0 * temp;
    if let Some(t) = &p.tables {
        // tail stress interpolated in TAUHFT or TAUHFT2 (SINTABLE = 1)
        let xi = ust / t.delust;
        let ind = 1.max((t.iustar - 1).min(xi as usize));
        let deli1 = (xi - ind as f64).min(1.0).max(0.0);
        let deli2 = 1.0 - deli1;
        let xj = ((grav * z0 / ust.max(0.00001).powi(2) - p.aalpha) / t.delalp).max(0.0);
        let j = 1.max((t.ialpha - 1).min(xj as usize));
        let delj1 = (xj - j as f64).min(1.0).max(0.0);
        let delj2 = 1.0 - delj1;
        let n1 = t.iustar + 1;
        let n2 = t.ialpha + 1;
        let tau1 = if p.ttauwshelter > 0.0 {
            let xk = levtail0 / t.deltail;
            let i = (t.ilevtail - 1).min(xk as usize);
            let delk1 = (xk - i as f64).min(1.0);
            let delk2 = 1.0 - delk1;
            let h = |a: usize, b: usize, c: usize| t.tauhft2[a + b * n1 + c * n1 * n2];
            ((h(ind, j, i) * deli2 + h(ind + 1, j, i) * deli1) * delj2 + (h(ind, j + 1, i) * deli2 + h(ind + 1, j + 1, i) * deli1) * delj1) * delk2
                + ((h(ind, j, i + 1) * deli2 + h(ind + 1, j, i + 1) * deli1) * delj2 + (h(ind, j + 1, i + 1) * deli2 + h(ind + 1, j + 1, i + 1) * deli1) * delj1) * delk1
        } else {
            let h = |a: usize, b: usize| t.tauhft[a + b * n1];
            (h(ind, j) * deli2 + h(ind + 1, j) * deli1) * delj2 + (h(ind, j + 1) * deli2 + h(ind + 1, j + 1) * deli1) * delj1
        };
        let tauhf = levtail0 * ust * ust * tau1;
        return finish_sin4(s, d, llws, sx, sy, snx, sny, tauhf, usdirp, ustar);
    }
    // tail stress integrated in line (SINTABLE = 0)
    let x0 = 0.05;
    let mut ustr = ust;
    let zz0 = z0;
    let _omegacc = snk.max(x0 * grav / ust);
    let zinf = (snk * snk / grav).ln();
    let dely = ((tpi / 0.005).ln() - zinf) / JTOT as f64;
    let mut taur = ust * ust;
    let mut tau1 = 0.0;
    for j in 1..=JTOT {
        let w = if j == 1 || j == JTOT { 0.5 } else { 1.0 };
        let y = (zinf + (j - 1) as f64 * dely).exp();
        let tensk = 1.0 + (y / KM).powi(2);
        let cm = (grav * tensk / y).sqrt();
        let cgtail = 0.5 * (3.0 * (y / KM).powi(2) + 1.0) * (grav / (y * tensk)).sqrt();
        let zx = ustr / cm + p.zzalp;
        let zarg = (kappa / zx).min(20.0);
        let zmu = (grav * zz0 / (cm * cm) * zarg.exp()).min(1.0);
        let zlog = zmu.ln().min(0.0);
        let zbeta = const1 * zmu * zlog.powi(4);
        let levtail = if p.sintailpar[2] > 0.0 {
            let cmm = 0.2325;
            let alpham = if ustr < cm { (0.01 * (1.0 + (ustr / cm).ln())).max(0.0) } else { 0.01 * (1.0 + 3.0 * (ustr / cm).ln()) };
            let fm = (-0.25 * (y / KM - 1.0).powi(2)).exp();
            let alphat = alpham * (cmm / cm) * fm;
            levtail0 * 0.5 * (1.0 - ((y - 20.0) / 5.0).tanh()) + p.sintailpar[2] * 0.5 * (1.0 + ((y - 20.0) / 5.0).tanh()) * alphat
        } else {
            levtail0
        };
        tau1 += w * zbeta * ustr * ustr * levtail * dely * cgtail / cm;
        taur -= w * p.sintailpar[1] * ustr * ustr * zbeta * levtail * dely * cgtail / cm;
        ustr = taur.max(0.0).sqrt();
    }
    finish_sin4(s, d, llws, sx, sy, snx, sny, tau1, usdirp, ustar)
}

/// The end of `W3SIN4`: wave-supported stress with the tail, limited by u*.
#[allow(clippy::too_many_arguments)]
fn finish_sin4(s: Vec<f64>, d: Vec<f64>, llws: Vec<bool>, sx: f64, sy: f64, snx: f64, sny: f64, tauhf: f64, usdirp: f64, ustar: f64) -> Sin4Out {
    const EPS1: f64 = 0.00001;
    const EPS2: f64 = 0.000001;
    let mut tauwx = sx + tauhf * usdirp.cos();
    let mut tauwy = sy + tauhf * usdirp.sin();
    let tauw = (tauwx * tauwx + tauwy * tauwy).sqrt();
    let ust2 = ustar.max(EPS2).powi(2);
    let tauwb = tauw.min((ust2 - EPS1).max(EPS2 * EPS2));
    if tauwb < tauw {
        tauwx *= tauwb / tauw;
        tauwy *= tauwb / tauw;
    }
    Sin4Out { s, d, tauwx, tauwy, tauwnx: snx, tauwny: sny, llws }
}

/// Linear interpolation in a table on a uniform axis (`FWTABLE`, `ERFTABLE`)
/// as WW3 does it: index `min(size - 1, INT(x))`, weights clipped at 1.
fn table(t: &[f64], x: f64) -> f64 {
    let size = t.len() - 1;
    let ind = (size - 1).min(x as usize);
    let deli1 = 1f64.min(x - ind as f64);
    t[ind] * (1.0 - deli1) + t[ind + 1] * deli1
}

/// `W3SLN1`: linear wind input of Cavaleri and Malanotte-Rizzoli (1981) with
/// the filter of Tolman (1992), for u* `ustar` in direction `usdir` and the
/// cut-off frequency `fhigh` (rad/s).
pub fn sln1(p: &St4, k: &[f64], fhigh: f64, ustar: f64, usdir: f64) -> Vec<f64> {
    let (nk, nth) = (p.nk, p.nth);
    let (cosu, sinu) = (usdir.cos(), usdir.sin());
    let dirf: Vec<f64> = (0..nth).map(|ith| (p.ecos[ith] * cosu + p.esin[ith] * sinu).max(0.0).powi(4)).collect();
    let fac = p.slnc1 * ustar.powi(4);
    let ff1 = p.fspm * p.grav / (28.0 * ustar);
    let ff2 = p.fshf * p.s(nk).min(fhigh);
    let ffilt = ff1.max(ff2).min(2.0 * p.s(nk));
    let mut s = vec![0f64; nk * nth];
    for ik in 0..nk {
        let rfr = p.s(ik + 1) / ffilt;
        let wnf = if rfr < 0.5 { 0.0 } else { fac / k[ik] * (-rfr.powi(-4)).exp() };
        for ith in 0..nth {
            s[ith + ik * nth] = wnf * dirf[ith];
        }
    }
    s
}

/// `W3SDB1`: depth-induced breaking of Battjes and Janssen (1978), with the
/// mean parameters of `W3SPR4`. Returns `(S, D, LBREAK)`.
pub fn sdb1(p: &St4, a: &[f64], depth: f64, emean: f64, fmean: f64, wnmean: f64) -> (Vec<f64>, Vec<f64>, bool) {
    let nspec = p.nk * p.nth;
    let thr = f32::MIN_POSITIVE as f64;
    if emean < f64::MIN_POSITIVE {
        return (vec![0.0; nspec], vec![0.0; nspec], false);
    }
    let hm = if p.fdonly { p.sdbc2 * depth } else { p.sdbc2 / wnmean * (wnmean * depth.max(0.0)).tanh() };
    let hrms = (8.0 * emean).sqrt();
    let (bb, b) = if hm > thr {
        let bb = hrms * hrms / (hm * hm);
        (bb, bb.sqrt())
    } else {
        (0.0, 0.0)
    };
    let q0 = if b <= 0.5 {
        0.0
    } else if b <= 1.0 {
        (2.0 * b - 1.0).powi(2)
    } else {
        0.0
    };
    let qb = if b <= 0.2 {
        0.0
    } else if b < 1.0 {
        let arg = ((q0 - 1.0) / bb).exp();
        let mut qb = q0 - bb * (q0 - arg) / (bb - arg);
        for _ in 0..3 {
            qb = ((qb - 1.0) / bb).exp();
        }
        qb
    } else {
        1.0 - thr
    };
    let cbj = if bb > thr && (bb - qb).abs() > thr {
        if bb < 1.0 {
            2.0 * p.sdbc1 * qb * fmean / bb
        } else {
            2.0 * p.sdbc1 * fmean * bb
        }
    } else {
        0.0
    };
    // WW3 stores the coefficient in single precision
    let dv = (-cbj) as f32 as f64;
    let d = vec![dv; nspec];
    let s = a.iter().map(|&x| dv * x).collect();
    (s, d, cbj > 0.0)
}

/// Outputs of `W3SBT4`.
#[derive(Clone, Debug, Default)]
pub struct Sbt4Out {
    /// Source term.
    pub s: Vec<f64>,
    /// Diagonal term.
    pub d: Vec<f64>,
    /// Bottom stress.
    pub taubbl: [f64; 2],
    /// Roughness and ripple wavelength vector.
    pub bedform: [f64; 3],
}

/// `W3SBT4`: bottom friction of the SHOWEX experiment (Ardhuin et al. 2003)
/// with movable-bed roughness, for sediment diameter `d50` and critical
/// Shields number `psic`.
#[allow(clippy::too_many_arguments)]
pub fn sbt4(p: &St4, a: &[f64], cg: &[f64], wn: &[f64], depth: f64, d50: f64, psic: f64) -> Sbt4Out {
    let (nk, nth) = (p.nk, p.nth);
    let grav = p.grav;
    let tpi = std::f64::consts::TAU;
    let sb = &p.sbtcx;
    let xsub = [-0.001f32 as f64, 0.001f32 as f64, 0.0];
    let dd50 = d50.max(1e-5).min(1.0);
    let background = sb[5].max(sb[6] * dd50);
    let mut cbeta = vec![0f64; nk];
    let mut shields = [0f64; 3];
    let (mut uorb, mut uorb2, mut aorb, mut lx, mut ly) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for isub in 0..3 {
        let dsub = depth * (1.0 + xsub[isub]);
        let (mut u, mut ao, mut ax, mut ay) = (0.0, 0.0, 0.0, 0.0);
        for ik in 0..nk {
            if wn[ik] * dsub < 6.0 {
                let (mut eb, mut ebx, mut eby) = (0.0, 0.0, 0.0);
                for ith in 0..nth {
                    let x = a[ith + ik * nth];
                    eb += x;
                    ebx += x * p.ecos[ith];
                    eby += x * p.esin[ith];
                }
                let s = p.s(ik + 1);
                cbeta[ik] = 0.5 * s * s / (grav * (wn[ik] * dsub).sinh().powi(2));
                let factor = (p.dden[ik] / cg[ik]) * 2.0 * cbeta[ik] * grav;
                let varu = eb * factor;
                u += varu;
                ao += varu / (s * s);
                ax += ebx * factor;
                ay += eby * factor;
            } else {
                cbeta[ik] = 0.0;
            }
        }
        uorb2 = 2.0 * u;
        uorb = 1.0e-7f64.max(uorb2).sqrt();
        aorb = 1.0e-7f64.max(2.0 * ao).sqrt();
        let n = (ax * ax + ay * ay + 1e-12).sqrt();
        lx = aorb * 1.7 * ax / n;
        ly = aorb * 1.7 * ay / n;
        let xi = (((aorb / dd50).max(0.3).log10() - p.abmin) / p.delab).max(1.0);
        let fw = table(&p.fwtable, xi);
        let psi = fw * uorb2 / (2.0 * grav * (p.sed_sg - 1.0) * dd50);
        shields[isub] = psi / psic;
    }
    let dpsi = (shields[1] - shields[0]) / (xsub[1] - xsub[0]) * sb[4];
    let mut bedform = [0f64; 3];
    let ksubn;
    if dpsi.abs() < 0.0001 * shields[2] || dpsi.abs() < 1e-8 {
        if shields[2] > sb[2] {
            let ksubr = aorb * sb[0] * shields[2].powf(sb[1]);
            let ksubs = aorb * 0.0655 * (uorb2 / ((p.sed_sg - 1.0) * grav * aorb)).powf(1.4);
            ksubn = ksubr + ksubs;
            bedform[1] = lx;
            bedform[2] = ly;
        } else {
            ksubn = background.max(aorb * sb[3]);
            bedform[1] = -lx;
            bedform[2] = -ly;
        }
    } else {
        let psix = (sb[2] - shields[2]) / dpsi;
        let psixt = ((psix + p.xerfmax) / p.delxerf).max(0.0);
        let proba2 = table(&p.erftable, psixt).clamp(0.0, 1.0);
        let proba1 = 1.0 - proba2;
        let psin2 = (shields[2] + (-(0.5 * psix * psix)).exp() / tpi.sqrt() * dpsi / (proba2 + 0.0001)).max(sb[2]);
        ksubn = proba1 * background.max(aorb * sb[3])
            + proba2 * aorb * (sb[0] * psin2.powf(sb[1]) + 0.0655 * (uorb2 / ((p.sed_sg - 1.0) * grav * aorb)).powf(1.4));
        if proba2 > 0.5 {
            bedform[1] = lx;
            bedform[2] = ly;
        } else {
            bedform[1] = -lx;
            bedform[2] = -ly;
        }
    }
    bedform[0] = ksubn;
    let xi = (((aorb / ksubn).max(0.3).log10() - p.abmin) / p.delab).max(1.0);
    let fw = table(&p.fwtable, xi);
    let mut s = vec![0f64; nk * nth];
    let mut d = vec![0f64; nk * nth];
    let mut taubbl = [0f64; 2];
    for ik in 0..nk {
        let sg = p.s(ik + 1);
        let const2 = p.dden[ik] / cg[ik] * grav / (sg / wn[ik]);
        let dsum = -fw * uorb * cbeta[ik];
        for ith in 0..nth {
            let is = ith + ik * nth;
            d[is] = dsum;
            let temp2 = const2 * dsum * a[is];
            taubbl[0] -= temp2 * p.ecos[is];
            taubbl[1] -= temp2 * p.esin[is];
            s[is] = dsum * a[is];
        }
    }
    Sbt4Out { s, d, taubbl, bedform }
}

/// Outputs of `W3SDS4`.
#[derive(Clone, Debug, Default)]
pub struct Sds4Out {
    /// Source term.
    pub s: Vec<f64>,
    /// Diagonal term.
    pub d: Vec<f64>,
    /// Breaking-crest length per unit area (Phillips' Lambda).
    pub brlambda: Vec<f64>,
}

/// `W3SDS4` with `SDSBCHOICE = 1` (directional saturation of Ardhuin et
/// al. 2010), with the directional cumulative term through `CUMULW`
/// (`SDSCUM < 0`, full ST4), the turbulent term (`SSDSC(5)`) and the
/// viscous term (`SSDSC(21)`); `ustar`, `usdir` are u* and its direction,
/// `drat` the air/water density ratio. The whitecap outputs are not
/// computed (they are off in the HWW3 runs).
pub fn sds4(p: &St4, a: &[f64], k: &[f64], depth: f64, ustar: f64, usdir: f64, drat: f64) -> Result<Sds4Out> {
    let (nk, nth) = (p.nk, p.nth);
    let nspec = nk * nth;
    let c = &p.ssdsc;
    if c[0].round() as i32 != 1 {
        bail!("W3SDS4: only SDSBCHOICE = 1 is ported");
    }
    if c[2] > 0.0 {
        bail!("W3SDS4: the isotropic cumulative term (SDSCUM > 0) is not ported");
    }
    if c[2] < 0.0 && p.cumulw.is_empty() {
        bail!("W3SDS4: SDSCUM < 0 needs CUMULW in the harness file");
    }
    let epsr = p.ssdsbr.sqrt();
    let mut bth = vec![0f64; nspec];
    let mut bth0 = vec![0f64; nk];
    for ik in 0..nk {
        let facsat = p.s(ik + 1) * k[ik].powi(3) * p.dth;
        let is0 = ik * nth;
        let asum: f64 = a[is0..is0 + nth].iter().sum();
        bth0[ik] = asum * facsat;
        if p.ssdsdth >= 180.0 {
            bth[is0..is0 + nth].iter_mut().for_each(|b| *b = bth0[ik]);
        } else {
            for ith in 0..nth {
                let mut dot = 0.0;
                for j in 0..p.nsat {
                    let o = j + ith * p.nsat;
                    dot += p.satweights[o] * a[is0 + (p.satindices[o] - 1) as usize];
                }
                bth[is0 + ith] = dot * facsat;
            }
            bth0[ik] = bth[is0..is0 + nth].iter().cloned().fold(f64::MIN, f64::max);
        }
    }
    let mut d = vec![0f64; nspec];
    for ik in 0..nk {
        let miche = if p.ssdsbm[0] == 1.0 {
            1.0
        } else {
            let x = (k[ik] * depth).min(10.0).tanh();
            let b = &p.ssdsbm;
            (x * (b[1] + x * (b[2] + x * (b[3] + x * b[4])))).powi(2)
        };
        let coef1 = p.ssdsbr * miche;
        let sg = p.s(ik + 1);
        let sdiagiso = c[1] * sg * c[5] * (bth0[ik] / coef1 - 1.0).max(0.0).powi(2);
        let coef2 = c[1] * sg * (1.0 - c[5]) / (coef1 * coef1);
        for ith in 0..nth {
            let is = ik * nth + ith;
            d[is] = sdiagiso + coef2 * (bth[is] - coef1).max(0.0).powf(p.ssdsp);
        }
    }
    let pi = std::f64::consts::PI;
    let brlambda: Vec<f64> = bth.iter().map(|&b| (b.sqrt() - epsr).max(0.0).powi(2) * 28.16 / (2.0 * pi * pi)).collect();
    let mut s: Vec<f64> = d.iter().zip(a).map(|(d, a)| d * a).collect();
    let sdsc21 = c.get(20).copied().unwrap_or(0.0);
    if c[2] != 0.0 || c[4] != 0.0 || sdsc21 != 0.0 {
        let facturb = c[4] * ustar * ustar / p.grav * drat;
        let dik = p.dikcumul.max(0) as usize;
        let (cu, su) = (usdir.cos(), usdir.sin());
        for ik in 0..nk {
            let facturb2 = -2.0 * p.s(ik + 1) * k[ik] * facturb;
            let dvisc = -4.0 * sdsc21 * k[ik] * k[ik];
            for ith in 0..nth {
                let is = ith + ik * nth;
                // IK > DIKCUMUL in WW3's 1-based count
                let mut renewalfreq = 0.0;
                if c[2] < 0.0 && ik + 1 > dik {
                    for ik2 in 0..(ik + 1 - dik) {
                        if bth0[ik2] > p.ssdsbr {
                            let is2 = ik2 * nth;
                            let col = &p.cumulw[is * nspec..(is + 1) * nspec];
                            let mut dot = 0.0;
                            for ith2 in 0..nth {
                                dot += col[is2 + ith2] * brlambda[is2 + ith2];
                            }
                            renewalfreq += dot;
                        }
                    }
                }
                let coswind = p.ecos[ith] * cu + p.esin[ith] * su;
                let dturb = facturb2 * coswind.max(0.0);
                let diag2 = c[2] * renewalfreq + dturb + dvisc;
                d[is] += diag2;
                s[is] += a[is] * diag2;
            }
        }
    }
    Ok(Sds4Out { s, d, brlambda })
}

/// Tables of the discrete interaction approximation (`W3SNL1`), as `INSNL1`
/// leaves them in `W3ADATMD`.
#[derive(Clone, Debug)]
pub struct Dia {
    /// Number of directions.
    pub nth: usize,
    /// Frequencies of the prognostic spectrum (`NFR = NK`).
    pub nfr: usize,
    /// Last frequency of the extended spectrum (`NFRHGH`).
    pub nfrhgh: usize,
    /// Number of bins where the interactions are computed (`NSPECX`).
    pub nspecx: usize,
    /// Number of bins of the extended spectrum (`NSPECY`).
    pub nspecy: usize,
    /// `KDCON`, `KDMN`: scaling of the mean relative depth.
    pub kdcon: f64,
    /// See [`Dia::kdcon`].
    pub kdmn: f64,
    /// `SNLC1`, `SNLS1..3`: proportionality and shallow-water constants.
    pub snlc1: f64,
    /// See [`Dia::snlc1`].
    pub snls: [f64; 3],
    /// High-frequency tail factor of the energy spectrum (`FACHFE`).
    pub fachfe: f64,
    /// `SIG(1..NK)` at `ik - 1`.
    pub sig: Vec<f64>,
    /// `IP11..IP14, IM11..IM14, IP21..IP24, IM21..IM24` (1-based bins of the
    /// extended spectrum, 0 or below for no energy), `NSPECX` each.
    pub ip: Vec<Vec<i32>>,
    /// `IC11, IC12, IC21, IC22, ..., IC81, IC82` (1-based bins of the
    /// interaction arrays), `NSPEC` each.
    pub ic: Vec<Vec<i32>>,
    /// `DAL1..3`.
    pub dal: [f64; 3],
    /// `AF11(1..NSPECX)` at `isp - 1`.
    pub af11: Vec<f64>,
    /// `AWG1..8`, `SWG1..8`: interpolation weights.
    pub awg: [f64; 8],
    /// See [`Dia::awg`].
    pub swg: [f64; 8],
}

impl Dia {
    /// The tables of the harness parameter file.
    pub fn new(p: &Params) -> Result<Dia> {
        let nth = p.i1("NTH")? as usize;
        let nk = p.i1("NK")? as usize;
        let idx = |n: &str| -> Result<Vec<i32>> {
            if p.lbound(n)? != 1 {
                bail!("{n} does not start at 1");
            }
            Ok(p.i(n)?.to_vec())
        };
        let mut ip = vec![];
        for g in ["IP1", "IM1", "IP2", "IM2"] {
            for j in 1..=4 {
                ip.push(idx(&format!("{g}{j}"))?);
            }
        }
        let mut ic = vec![];
        for j in 1..=8 {
            for k in 1..=2 {
                ic.push(idx(&format!("IC{j}{k}"))?);
            }
        }
        let a8 = |n: &str| -> Result<[f64; 8]> {
            let v = p.r(n)?;
            let mut a = [0f64; 8];
            for (k, x) in a.iter_mut().enumerate() {
                *x = v[k] as f64;
            }
            Ok(a)
        };
        let sig_all = p.r("SIG")?;
        let s0 = p.lbound("SIG")?;
        let sig = (1..=nk as i32).map(|ik| sig_all[(ik - s0) as usize] as f64).collect();
        let d = Dia {
            nth,
            nfr: p.i1("NFR")? as usize,
            nfrhgh: p.i1("NFRHGH")? as usize,
            nspecx: p.i1("NSPECX")? as usize,
            nspecy: p.i1("NSPECY")? as usize,
            kdcon: p.r1("KDCON")? as f64,
            kdmn: p.r1("KDMN")? as f64,
            snlc1: p.r1("SNLC1")? as f64,
            snls: [p.r1("SNLS1")? as f64, p.r1("SNLS2")? as f64, p.r1("SNLS3")? as f64],
            fachfe: p.r1("FACHFE")? as f64,
            sig,
            ip,
            ic,
            dal: [p.r1("DAL1")? as f64, p.r1("DAL2")? as f64, p.r1("DAL3")? as f64],
            af11: v64(p.r("AF11")?),
            awg: a8("AWG")?,
            swg: a8("SWG")?,
        };
        if d.nfr != nk || d.nspecy < d.nfrhgh * nth || d.af11.len() < d.nspecx {
            bail!("unexpected DIA dimensions");
        }
        Ok(d)
    }
}

/// `W3SNL1`: nonlinear interactions by the discrete interaction
/// approximation, source `S` and diagonal `D` for the action spectrum `a`,
/// group velocities `cg(ik)` and mean relative depth `kdmean`.
pub fn snl1(p: &Dia, a: &[f64], cg: &[f64], kdmean: f64) -> (Vec<f64>, Vec<f64>) {
    let nth = p.nth;
    let nspec = p.nfr * nth;
    let tpiinv = 1.0 / std::f64::consts::TAU;
    let x = (p.kdcon * kdmean).max(p.kdmn);
    let x2 = (p.snls[2] * x).max(-1e15);
    let cons = p.snlc1 * (1.0 + p.snls[0] / x * (1.0 - p.snls[1] * x) * x2.exp());
    // arrays indexed by the Fortran bin ISP (1-NTH..) at ISP + NTH - 1
    let o = nth - 1;
    let mut ue = vec![0f64; p.nspecy + nth];
    let mut con = vec![0f64; nspec];
    for ifr in 0..p.nfr {
        let conx = tpiinv / p.sig[ifr] * cg[ifr];
        for ith in 0..nth {
            let is = ith + ifr * nth;
            ue[is + nth] = a[is] / conx;
            con[is] = conx;
        }
    }
    for is in p.nfr * nth..p.nfrhgh * nth {
        ue[is + nth] = ue[is] * p.fachfe;
    }
    let n = p.nspecx + nth;
    let (mut sa1, mut sa2) = (vec![0f64; n], vec![0f64; n]);
    let (mut da1c, mut da1p, mut da1m) = (vec![0f64; n], vec![0f64; n], vec![0f64; n]);
    let (mut da2c, mut da2p, mut da2m) = (vec![0f64; n], vec![0f64; n], vec![0f64; n]);
    let u = |t: &Vec<i32>, isp: usize| ue[(t[isp] + o as i32) as usize];
    let w = &p.awg;
    let [dal1, dal2, dal3] = p.dal;
    for isp in 0..p.nspecx {
        let e00 = ue[isp + nth];
        let ip = &p.ip;
        let ep1 = w[0] * u(&ip[0], isp) + w[1] * u(&ip[1], isp) + w[2] * u(&ip[2], isp) + w[3] * u(&ip[3], isp);
        let em1 = w[4] * u(&ip[4], isp) + w[5] * u(&ip[5], isp) + w[6] * u(&ip[6], isp) + w[7] * u(&ip[7], isp);
        let ep2 = w[0] * u(&ip[8], isp) + w[1] * u(&ip[9], isp) + w[2] * u(&ip[10], isp) + w[3] * u(&ip[11], isp);
        let em2 = w[4] * u(&ip[12], isp) + w[5] * u(&ip[13], isp) + w[6] * u(&ip[14], isp) + w[7] * u(&ip[15], isp);
        let factor = cons * p.af11[isp] * e00;
        let sa1a = e00 * (ep1 * dal1 + em1 * dal2);
        let sa1b = sa1a - ep1 * em1 * dal3;
        let sa2a = e00 * (ep2 * dal1 + em2 * dal2);
        let sa2b = sa2a - ep2 * em2 * dal3;
        let k = isp + nth;
        sa1[k] = factor * sa1b;
        sa2[k] = factor * sa2b;
        da1c[k] = cons * p.af11[isp] * (sa1a + sa1b);
        da1p[k] = factor * (dal1 * e00 - dal3 * em1);
        da1m[k] = factor * (dal2 * e00 - dal3 * ep1);
        da2c[k] = cons * p.af11[isp] * (sa2a + sa2b);
        da2p[k] = factor * (dal1 * e00 - dal3 * em2);
        da2m[k] = factor * (dal2 * e00 - dal3 * ep2);
    }
    let mut s = vec![0f64; nspec];
    let mut d = vec![0f64; nspec];
    let at = |v: &Vec<f64>, t: &Vec<i32>, isp: usize| v[(t[isp] + o as i32) as usize];
    for isp in 0..nspec {
        let k = isp + nth;
        let mut ss = -2.0 * (sa1[k] + sa2[k]);
        let mut dd = -2.0 * (da1c[k] + da2c[k]);
        for j in 0..8 {
            let (c1, c2) = (&p.ic[2 * j], &p.ic[2 * j + 1]);
            ss += w[j] * (at(&sa1, c1, isp) + at(&sa2, c2, isp));
            dd += if j < 4 {
                p.swg[j] * (at(&da1p, c1, isp) + at(&da2p, c2, isp))
            } else {
                p.swg[j] * (at(&da1m, c1, isp) + at(&da2m, c2, isp))
            };
        }
        s[isp] = con[isp] * ss;
        d[isp] = dd;
    }
    (s, d)
}

/// Parameters of `W3SRCE` on top of the source terms.
#[derive(Clone, Debug)]
pub struct SrcePar {
    /// Parameters of the ST4 source terms, `W3SLN1`, `W3SDB1`, `W3SBT4`.
    pub st: St4,
    /// Tables of `W3SNL1`.
    pub dia: Dia,
    /// Largest and smallest dynamic step (`DTMAX`, `DTMIN`).
    pub dtmax: f64,
    /// See [`SrcePar::dtmax`].
    pub dtmin: f64,
    /// Limiter constants `XFLT`, `XREL`, `FACP`.
    pub xflt: f64,
    /// See [`SrcePar::xflt`].
    pub xrel: f64,
    /// See [`SrcePar::xflt`].
    pub facp: f64,
    /// `NKH = INT(FACTI2 + FACTI1 LOG(FHIGH))`.
    pub facti1: f64,
    /// See [`SrcePar::facti1`].
    pub facti2: f64,
    /// Tail factor of the action spectrum.
    pub fachfa: f64,
    /// Largest Hs over depth (`FHMAX`).
    pub fhmax: f64,
    /// Smallest depth.
    pub dmin: f64,
    /// Cut-off frequency factors `FFXFM`, `FFXPM`, `FFXFA`.
    pub ffxfm: f64,
    /// See [`SrcePar::ffxfm`].
    pub ffxpm: f64,
    /// See [`SrcePar::ffxfm`].
    pub ffxfa: f64,
    /// `ICESCALES(1..4)`: ice scaling of `LN`, `IN`, `NL`, `DS`.
    pub icescales: [f64; 4],
    /// Densities of water and air.
    pub dwat: f64,
    /// See [`SrcePar::dwat`].
    pub dair: f64,
    /// Ice concentration forcing on (`INFLAGS2(4)`).
    pub ice_forcing: bool,
}

impl SrcePar {
    /// The parameters of the harness parameter file (ice forcing on, as in
    /// the HWW3 runs).
    pub fn new(p: &Params) -> Result<SrcePar> {
        let r = |n: &str| -> Result<f64> { Ok(p.r1(n)? as f64) };
        let ic = p.r("ICESCALES")?;
        let o = (1 - p.lbound("ICESCALES")?) as usize;
        Ok(SrcePar {
            st: St4::new(p)?,
            dia: Dia::new(p)?,
            dtmax: r("DTMAX")?,
            dtmin: r("DTMIN")?,
            xflt: r("XFLT")?,
            xrel: r("XREL")?,
            facp: r("FACP")?,
            facti1: r("FACTI1")?,
            facti2: r("FACTI2")?,
            fachfa: r("FACHFA")?,
            fhmax: r("FHMAX")?,
            dmin: r("DMIN")?,
            ffxfm: r("FFXFM")?,
            ffxpm: r("FFXPM")?,
            ffxfa: r("FFXFA")?,
            icescales: [ic[o] as f64, ic[o + 1] as f64, ic[o + 2] as f64, ic[o + 3] as f64],
            dwat: r("DWAT")?,
            dair: r("DAIR")?,
            ice_forcing: true,
        })
    }
}

/// Inputs of `W3SRCE` at one node besides the spectrum.
#[derive(Clone, Copy, Debug, Default)]
pub struct SrceIn {
    /// `IT = 0`: the start call (no integration, the spectrum is restored).
    pub start: bool,
    /// Global time step.
    pub dtg: f64,
    /// Water depth (`D_INP`, before the `DMIN` floor).
    pub depth: f64,
    /// Wind speed and direction (relative to the current).
    pub u10: f64,
    /// See [`SrceIn::u10`].
    pub u10dir: f64,
    /// Ice concentration.
    pub ice: f64,
    /// Sediment diameter and critical Shields number of `W3SBT4`.
    pub d50: f64,
    /// See [`SrceIn::d50`].
    pub psic: f64,
    /// Air-sea drag scaling (`ASF`, `COEF` in `W3SRCE`).
    pub coef: f64,
}

/// Outputs of `W3SRCE` (the spectrum is updated in place).
#[derive(Clone, Debug, Default)]
pub struct SrceOut {
    /// Mean dynamic step.
    pub dtdyn: f64,
    /// Cut-off frequency (Hz).
    pub fcut: f64,
    /// u* and its direction.
    pub ustar: f64,
    /// See [`SrceOut::ustar`].
    pub ustdir: f64,
    /// Wave-supported stress of the last `W3SIN4`.
    pub tauwx: f64,
    /// See [`SrceOut::tauwx`].
    pub tauwy: f64,
    /// Stress to the ocean.
    pub tauox: f64,
    /// See [`SrceOut::tauox`].
    pub tauoy: f64,
    /// Stress from the air to the waves.
    pub tauwix: f64,
    /// See [`SrceOut::tauwix`].
    pub tauwiy: f64,
    /// Stress from the waves back to the air (swell damping).
    pub tauwnx: f64,
    /// See [`SrceOut::tauwnx`].
    pub tauwny: f64,
    /// Stress from the air to the ocean.
    pub tauocx: f64,
    /// See [`SrceOut::tauocx`].
    pub tauocy: f64,
    /// Energy flux from the wind to the waves.
    pub phiaw: f64,
    /// Energy flux to the ocean.
    pub phioc: f64,
    /// Energy flux to the bottom boundary layer.
    pub phibbl: f64,
    /// Charnock coefficient.
    pub charn: f64,
    /// Wind-sea period.
    pub tws: f64,
    /// Whitecap outputs (only `WHITECAP(3)`, the wind-sea height, is computed).
    pub whitecap: [f64; 4],
    /// Bottom stress.
    pub taubbl: [f64; 2],
    /// Ice stress.
    pub tauice: [f64; 2],
    /// Energy flux to the ice.
    pub phice: f64,
    /// Mean wavenumber.
    pub wnmean: f64,
    /// Last dynamic step shaved by the limiter.
    pub shave: bool,
    /// Number of dynamic steps.
    pub nsteps: usize,
}

impl SrceOut {
    /// The values in the order of [`crate::harness::Srce::NAMES`] (`fpi`
    /// being passed through).
    pub fn as_array(&self, fpi: f64) -> [f64; 30] {
        [
            self.dtdyn,
            self.fcut,
            self.ustar,
            self.ustdir,
            self.tauwx,
            self.tauwy,
            self.tauox,
            self.tauoy,
            self.tauwix,
            self.tauwiy,
            self.tauwnx,
            self.tauwny,
            self.phiaw,
            self.charn,
            self.tws,
            self.phioc,
            self.whitecap[0],
            self.whitecap[1],
            self.whitecap[2],
            self.whitecap[3],
            self.taubbl[0],
            self.taubbl[1],
            self.tauice[0],
            self.tauice[1],
            self.phice,
            self.tauocx,
            self.tauocy,
            self.wnmean,
            fpi,
            self.shave as u8 as f64,
        ]
    }
}

/// `W3SRCE` called with `srce_direct`: explicit integration of the source
/// terms over `dtg` with the dynamic step and the action limiter, the
/// high-frequency tail, the `FHMAX` limit and the ice scaling of the
/// sources (`IC0`: no attenuation). `spec` is updated in place; `ustar` is
/// the u* of the previous call (reset for the start call). `wn` and `cg`
/// are given per frequency.
pub fn srce(q: &SrcePar, inp: &SrceIn, spec: &mut [f64], wn: &[f64], cg: &[f64], ustar: f64) -> SrceOut {
    let p = &q.st;
    let (nk, nth) = (p.nk, p.nth);
    let nspec = nk * nth;
    let tpi = std::f64::consts::TAU;
    let grav = p.grav;
    let depth = q.dmin.max(inp.depth);
    let drat = q.dair / q.dwat;
    let isc = |k: usize| (1.0 - inp.ice * q.icescales[k]).clamp(0.0, 1.0);
    let (isc_ln, isc_in, isc_nl, isc_ds) = (isc(0), isc(1), isc(2), isc(3));
    let mut dam = vec![0f64; nspec];
    let mut wn2 = vec![0f64; nspec];
    for ik in 0..nk {
        for ith in 0..nth {
            dam[ith + ik * nth] = q.facp / (p.s(ik + 1) * wn[ik].powi(3));
            wn2[ith + ik * nth] = wn[ik];
        }
    }
    let (u10, u10dir) = (inp.u10, inp.u10dir);
    let mut o = SrceOut::default();
    let (mut tauwx, mut tauwy) = (0.0, 0.0);
    // the first W3SPR4 of WW3 (IT > 0) reads LLWS before they are set; only
    // its u* and z0 are used, which do not depend on them
    let mut llws = vec![true; nspec];
    let ustar0 = if inp.start { 0.0 } else { ustar };
    let mut sp = spr4(p, spec, cg, wn, u10, u10dir, ustar0, 0.0, 0.0, &llws);
    if !inp.start && p.sintailpar[3] > 0.5 {
        let s = sin4(p, spec, cg, &wn2, u10, sp.ustar, drat, u10dir, sp.z0);
        tauwx = s.tauwx;
        tauwy = s.tauwy;
        llws = s.llws;
    }
    if inp.start {
        sp = spr4(p, spec, cg, wn, u10, u10dir, ustar0, 0.0, 0.0, &llws);
    } else {
        sp = spr4(p, spec, cg, wn, u10, u10dir, sp.ustar, tauwx, tauwy, &llws);
    }
    o.tws = 1.0 / sp.fmeanws;
    let specinit = spec.to_vec();
    let mut fhigh = (q.ffxfm * sp.fmean1.max(sp.fmeanws)).max(q.ffxpm / sp.ustar);
    let (mut dtdyn, mut dttot) = (0.0f64, 0.0f64);
    let mut nsteps = 0usize;
    let (mut phiaw, mut phibbl) = (0.0f64, 0.0f64);
    let (mut tauwix, mut tauwiy, mut tauwnx, mut tauwny) = (0.0f64, 0.0, 0.0, 0.0);
    let mut taubbl: [f64; 2];
    let mut whitecap3: f64;
    let mut shave: bool;
    let (mut vs, mut vd) = (vec![0f64; nspec], vec![0f64; nspec]);
    let int = |x: f64| x.trunc() as i64;
    loop {
        nsteps += 1;
        let mut vsln = sln1(p, wn, fhigh, sp.ustar, u10dir);
        let mut si = sin4(p, spec, cg, &wn2, u10, sp.ustar, drat, u10dir, sp.z0);
        let (mut vsnl, mut vdnl) = snl1(&q.dia, spec, cg, sp.wnmean * depth);
        let ds = sds4(p, spec, wn, depth, sp.ustar, sp.usdir, drat).expect("W3SDS4 configuration checked by the caller");
        let (mut vsds, mut vdds) = (ds.s, ds.d);
        let (vsdb, vddb, _lbreak) = sdb1(p, spec, depth, sp.emean, sp.fmean, sp.wnmean);
        let bt = sbt4(p, spec, cg, wn, depth, inp.d50, inp.psic);
        taubbl = bt.taubbl;
        let nkh = (nk as i64).min(int(q.facti2 + q.facti1 * fhigh.max(1e-7).ln()));
        let nkh1 = (nk as i64).min(nkh + 1).max(0) as usize;
        let nspech = nkh1 * nth;
        let mut dt = (inp.dtg - dttot).min(q.dtmax);
        let afilt = dam[nspec - 1].max(q.xflt * sp.amax);
        if q.ice_forcing {
            for is in 0..nspech {
                vsnl[is] *= isc_nl;
                vdnl[is] *= isc_nl;
                vsln[is] *= isc_ln;
                si.s[is] *= isc_in;
                si.d[is] *= isc_in;
                vsds[is] *= isc_ds;
                vdds[is] *= isc_ds;
            }
        }
        for is in 0..nspech {
            vs[is] = vsln[is] + si.s[is] + vsnl[is] + vsds[is] + bt.s[is];
            vd[is] = si.d[is] + vdnl[is] + vdds[is] + bt.d[is];
            let damax = dam[is].min((q.xrel * specinit[is]).max(afilt));
            let afac = 1.0 / (vs[is] / damax).abs().max(1e-10);
            dt = dt.min(afac / (1.0 + afac * vd[is].min(0.0)).max(1e-10));
        }
        dt = dt.max(0.5);
        dtdyn += dt;
        let idt = 1 + int(0.99 * (inp.dtg - dttot) / dt);
        dt = (inp.dtg - dttot) / idt as f64;
        shave = dt < q.dtmin && dt < inp.dtg - dttot;
        dt = dt.max(q.dtmin.min(inp.dtg - dttot));
        let hdt = dt;
        dttot += dt;
        for is in 0..nspech {
            let inc = vs[is] * dt / (1.0 - hdt * vd[is]).max(1.0);
            let inc = if shave { inc.abs().min(dam[is]).copysign(inc) } else { inc };
            spec[is] = (spec[is] + inc).max(0.0);
        }
        for is in 0..nspech {
            let inc = vsdb[is] * dt / (1.0 - hdt * vddb[is]).max(1.0);
            spec[is] = (spec[is] + inc).max(0.0);
        }
        let mut wc3 = 0.0;
        for ik in 0..nk {
            let factor = p.dden[ik] / cg[ik];
            for ith in 0..nth {
                let is = ik * nth + ith;
                phiaw += si.s[is] * dt * factor / (1.0 - hdt * si.d[is]).max(1.0);
                phibbl -= bt.s[is] * dt * factor / (1.0 - hdt * bt.d[is]).max(1.0);
                if si.s[is] > 0.0 {
                    wc3 += spec[is] * factor;
                }
            }
        }
        whitecap3 = 4.0 * wc3.sqrt();
        tauwix += si.tauwx * drat * dt;
        tauwiy += si.tauwy * drat * dt;
        tauwnx += si.tauwnx * drat * dt;
        tauwny += si.tauwny * drat * dt;
        sp = spr4(p, spec, cg, wn, u10, u10dir, sp.ustar, si.tauwx, si.tauwy, &si.llws);
        let fage = q.ffxfa * (0.3 * u10 * sp.fmeanws * tpi / grav).tanh();
        let fh1 = (q.ffxfm + fage) * sp.fmean1;
        let fh2 = q.ffxpm / sp.ustar;
        fhigh = p.s(nk).min(fh1.max(fh2));
        let nkh = 2.max((nkh1 as i64).min(int(q.facti2 + q.facti1 * fhigh.max(1e-7).ln()))) as usize;
        if dttot >= 0.9999 * inp.dtg {
            let hm = q.fhmax * (sp.wnmean * inp.depth.max(0.0)).tanh() / sp.wnmean.max(1e-4);
            let em = hm * hm / 16.0;
            if sp.emean > em && sp.emean > 1e-30 {
                for x in spec.iter_mut() {
                    *x = *x / sp.emean * em;
                }
                sp.emean = em;
            }
        }
        for ik in nkh..nk {
            for ith in 0..nth {
                spec[ith + ik * nth] = spec[ith + (ik - 1) * nth] * q.fachfa;
            }
        }
        let s2 = sin4(p, spec, cg, &wn2, u10, sp.ustar, drat, u10dir, sp.z0);
        tauwx = s2.tauwx;
        tauwy = s2.tauwy;
        if p.sintailpar[3] < 0.5 {
            sp = spr4(p, spec, cg, wn, u10, u10dir, sp.ustar, tauwx, tauwy, &s2.llws);
        }
        if dttot >= 0.9999 * inp.dtg {
            break;
        }
    }
    o.nsteps = nsteps;
    o.dtdyn = dtdyn / nsteps.max(1) as f64;
    o.fcut = fhigh / tpi;
    let (mut efinish, mut mwx, mut mwy) = (0.0f64, 0.0f64, 0.0f64);
    for ik in 0..nk {
        let (mut eband, mut a1, mut b1) = (0.0, 0.0, 0.0);
        for ith in 0..nth {
            let diff = specinit[ith + ik * nth] - spec[ith + ik * nth];
            eband += diff;
            a1 += diff * p.ecos[ith];
            b1 += diff * p.esin[ith];
        }
        let f = p.dden[ik] / cg[ik];
        efinish += eband * f;
        mwx += a1 * f * wn[ik] / p.s(ik + 1);
        mwy += b1 * f * wn[ik] / p.s(ik + 1);
    }
    let dtg = inp.dtg;
    o.tauox = (grav * mwx + tauwix - taubbl[0]) / dtg;
    o.tauoy = (grav * mwy + tauwiy - taubbl[1]) / dtg;
    o.tauwix = tauwix / dtg;
    o.tauwiy = tauwiy / dtg;
    o.tauwnx = tauwnx / dtg;
    o.tauwny = tauwny / dtg;
    o.taubbl = [taubbl[0] / dtg, taubbl[1] / dtg];
    o.ustar = sp.ustar;
    o.ustdir = sp.usdir;
    let c2 = q.dair * inp.coef * inp.coef * sp.ustar * sp.ustar;
    o.tauocx = c2 * sp.usdir.cos() + q.dwat * (o.tauox - o.tauwix);
    o.tauocy = c2 * sp.usdir.sin() + q.dwat * (o.tauoy - o.tauwiy);
    o.phioc = q.dwat * grav * (efinish + phiaw - phibbl) / dtg;
    o.phiaw = q.dwat * grav * phiaw / dtg;
    o.phibbl = q.dwat * grav * phibbl / dtg;
    o.charn = sp.charn;
    o.whitecap = [0.0, 0.0, whitecap3, 0.0];
    o.wnmean = sp.wnmean;
    o.tauwx = tauwx;
    o.tauwy = tauwy;
    o.shave = shave;
    if inp.start {
        spec.copy_from_slice(&specinit);
    }
    for x in spec.iter_mut() {
        *x = x.max(0.0);
    }
    o
}

// Indices of the scalar parameters of the GPU source kernel (`srce`) in
// [`SrceTables::par`].
/// Gravity.
pub const P_GRAV: usize = 0;
/// Von Karman constant.
pub const P_KAPPA: usize = 1;
/// Kinematic viscosity of air.
pub const P_NU_AIR: usize = 2;
/// Air/water density ratio.
pub const P_DRAT: usize = 3;
/// Direction step.
pub const P_DTH: usize = 4;
/// Tail factor of the energy (`W3SPR4`).
pub const P_FTE: usize = 5;
/// Tail factor of the mean frequency.
pub const P_FTF: usize = 6;
/// Tail factor of the mean wavenumber.
pub const P_SSTXFTWN: usize = 7;
/// Tail factor of `FMEAN1`.
pub const P_SSTXFTFTAIL: usize = 8;
/// Exponent of the mean wavenumber.
pub const P_WWNMEANP: usize = 9;
/// Exponent of `FMEAN1`.
pub const P_WWNMEANPTAIL: usize = 10;
/// Wind height.
pub const P_ZZWND: usize = 11;
/// Charnock constant.
pub const P_AALPHA: usize = 12;
/// `BETAMAX`.
pub const P_BBETA: usize = 13;
/// Wave age shift `ZALP`.
pub const P_ZZALP: usize = 14;
/// Sheltering coefficient.
pub const P_TTAUWSHELTER: usize = 15;
/// `SSWELLF(1..7)` (swell damping).
pub const P_SSWELLF: usize = 16;
/// Directional exponent of the wind input.
pub const P_SSINTHP: usize = 23;
/// Roughness ratio of swell damping.
pub const P_ZZ0RAT: usize = 24;
/// Largest roughness.
pub const P_ZZ0MAX: usize = 25;
/// `SINTAILPAR(1..5)`.
pub const P_SINTAILPAR: usize = 26;
/// `CAPCHNK(1..5)` (capped Charnock).
pub const P_CAPCHNK: usize = 31;
/// Step of `FWTABLE`.
pub const P_DELAB: usize = 36;
/// Origin of `FWTABLE`.
pub const P_ABMIN: usize = 37;
/// `SSDSC(2)` (saturation dissipation coefficient).
pub const P_SDSC2: usize = 38;
/// `SSDSC(6)` (isotropic part).
pub const P_SDSC6: usize = 39;
/// Saturation threshold.
pub const P_SSDSBR: usize = 40;
/// Saturation exponent.
pub const P_SSDSP: usize = 41;
/// `SSDSBM(0..4)` (depth correction).
pub const P_SSDSBM: usize = 42;
/// `W3SLN1` coefficient.
pub const P_SLNC1: usize = 47;
/// `W3SLN1` PM filter factor.
pub const P_FSPM: usize = 48;
/// `W3SLN1` high-frequency filter factor.
pub const P_FSHF: usize = 49;
/// `W3SDB1` coefficient.
pub const P_SDBC1: usize = 50;
/// `W3SDB1` breaking index.
pub const P_SDBC2: usize = 51;
/// `W3SDB1`: depth only (1) or Miche (0).
pub const P_FDONLY: usize = 52;
/// `SBTCX(1..10)` (`W3SBT4`).
pub const P_SBTCX: usize = 53;
/// Specific gravity of the sediment.
pub const P_SED_SG: usize = 63;
/// Step of `ERFTABLE`.
pub const P_DELXERF: usize = 64;
/// Half range of `ERFTABLE`.
pub const P_XERFMAX: usize = 65;
/// DIA: `KDCON`.
pub const P_KDCON: usize = 66;
/// DIA: `KDMN`.
pub const P_KDMN: usize = 67;
/// DIA: `SNLC1`.
pub const P_SNLC1: usize = 68;
/// DIA: `SNLS1`.
pub const P_SNLS1: usize = 69;
/// DIA: `SNLS2`.
pub const P_SNLS2: usize = 70;
/// DIA: `SNLS3`.
pub const P_SNLS3: usize = 71;
/// Tail factor of the energy spectrum.
pub const P_FACHFE: usize = 72;
/// DIA: `DAL1`.
pub const P_DAL1: usize = 73;
/// DIA: `DAL2`.
pub const P_DAL2: usize = 74;
/// DIA: `DAL3`.
pub const P_DAL3: usize = 75;
/// DIA: `AWG1..8`.
pub const P_AWG: usize = 76;
/// DIA: `SWG1..8`.
pub const P_SWG: usize = 84;
/// `W3SRCE`: largest dynamic step.
pub const P_DTMAX: usize = 92;
/// `W3SRCE`: smallest dynamic step.
pub const P_DTMIN: usize = 93;
/// Limiter: `XFLT`.
pub const P_XFLT: usize = 94;
/// Limiter: `XREL`.
pub const P_XREL: usize = 95;
/// Limiter: `FACP`.
pub const P_FACP: usize = 96;
/// `NKH = INT(FACTI2 + FACTI1 LOG(FHIGH))`.
pub const P_FACTI1: usize = 97;
/// See `FACTI1`.
pub const P_FACTI2: usize = 98;
/// Tail factor of the action spectrum.
pub const P_FACHFA: usize = 99;
/// Largest Hs over depth.
pub const P_FHMAX: usize = 100;
/// Smallest depth.
pub const P_DMIN: usize = 101;
/// Cut-off factor on the mean frequency.
pub const P_FFXFM: usize = 102;
/// Cut-off factor on u*.
pub const P_FFXPM: usize = 103;
/// Wave-age part of the cut-off.
pub const P_FFXFA: usize = 104;
/// `ICESCALES(1..4)`.
pub const P_ICESCALES: usize = 105;
/// Length of `FWTABLE`.
pub const P_NFW: usize = 109;
/// Length of `ERFTABLE`.
pub const P_NERF: usize = 110;
/// Ice concentration forcing on (1).
pub const P_ICE_FORCING: usize = 111;
/// Half width of the saturation sector (degrees).
pub const P_SSDSDTH: usize = 112;
/// Stress tables on (`SINTABLE = 1`): 1, else 0.
pub const P_SINTABLE: usize = 113;
/// Table steps `DELUST`, `DELALP`, `DELTAUW`, `DELU`, `DELTAIL`, `TAUWMAX`.
pub const P_DELUST: usize = 114;
/// See [`P_DELUST`].
pub const P_DELALP: usize = 115;
/// See [`P_DELUST`].
pub const P_DELTAUW: usize = 116;
/// See [`P_DELUST`].
pub const P_DELU: usize = 117;
/// See [`P_DELUST`].
pub const P_DELTAIL: usize = 118;
/// See [`P_DELUST`].
pub const P_TAUWMAX: usize = 119;
/// Table sizes `ITAUMAX`, `JUMAX`, `IUSTAR`, `IALPHA`, `ILEVTAIL`.
pub const P_ITAUMAX: usize = 120;
/// See [`P_ITAUMAX`].
pub const P_JUMAX: usize = 121;
/// See [`P_ITAUMAX`].
pub const P_IUSTAR: usize = 122;
/// See [`P_ITAUMAX`].
pub const P_IALPHA: usize = 123;
/// See [`P_ITAUMAX`].
pub const P_ILEVTAIL: usize = 124;
/// `SSDSC(3)` (`SDSCUM`, cumulative term), `SSDSC(5)` (turbulence),
/// `SSDSC(21)` (viscosity).
pub const P_SDSC3: usize = 125;
/// See [`P_SDSC3`].
pub const P_SDSC5: usize = 126;
/// See [`P_SDSC3`].
pub const P_SDSC21: usize = 127;
/// `DIKCUMUL`.
pub const P_DIKCUMUL: usize = 128;
/// Number of scalar parameters.
pub const NPAR: usize = 129;

/// The parameters and tables of [`SrcePar`] laid out for the GPU source
/// kernel: scalars at the `P_*` indices, per-frequency factors, the
/// interpolation tables and the index tables of `W3SDS4` and `W3SNL1`
/// turned into 0-based offsets in the kernel's shared arrays.
#[derive(Clone, Debug)]
pub struct SrceTables {
    /// Scalar parameters, `NPAR` values at the `P_*` indices.
    pub par: Vec<f64>,
    /// `SIG2(ISPEC)` and `DDEN2(ISPEC)` of the first direction of each frequency.
    pub sig2k: Vec<f64>,
    /// See [`SrceTables::sig2k`].
    pub dden2k: Vec<f64>,
    /// `FWTABLE` and `ERFTABLE`.
    pub fwtable: Vec<f64>,
    /// See [`SrceTables::fwtable`].
    pub erftable: Vec<f64>,
    /// Saturation weights `SATWEIGHTS(j, ith)` at `j + ith nsat`.
    pub satw: Vec<f64>,
    /// Saturation direction offsets, 0-based (`SATINDICES - 1`), at `j + ith nsat`.
    pub sati: Vec<u32>,
    /// First dimension of the saturation tables.
    pub nsat: usize,
    /// DIA: the 16 tables `IP11 .. IM24` (`NSPECX` each), as offsets in the
    /// extended spectrum `UE` stored from `1 - NTH` (value `+ NTH - 1`).
    pub dia_ip: Vec<u32>,
    /// DIA: the 16 tables `IC11, IC12, .., IC82` (`NSPEC` each), as offsets in
    /// the interaction arrays stored from `1 - NTH`.
    pub dia_ic: Vec<u32>,
    /// DIA: `AF11(1..NSPECX)`.
    pub af11: Vec<f64>,
    /// `SIG(0 .. NK+1)`, `DDEN(1 .. NK)`, `ECOS(1 .. NTH)`, `ESIN(1 .. NTH)`
    /// of WW3.
    pub sig: Vec<f64>,
    /// See [`SrceTables::sig`].
    pub dden: Vec<f64>,
    /// See [`SrceTables::sig`].
    pub ecos: Vec<f64>,
    /// See [`SrceTables::sig`].
    pub esin: Vec<f64>,
    /// DIA: bins of the interaction arrays and of the extended spectrum.
    pub nspecx: usize,
    /// See [`SrceTables::nspecx`].
    pub nspecy: usize,
    /// `TAUT` (`SINTABLE = 1`; one zero otherwise).
    pub taut: Vec<f64>,
    /// `TAUHFT2` when `TAUWSHELTER > 0`, else `TAUHFT` (`SINTABLE = 1`;
    /// one zero otherwise).
    pub tauhf: Vec<f64>,
    /// `CUMULW(IS2, IS)`, which depends on the two frequencies and the
    /// direction difference only, as `cumw[(ik2 NK + ik) NTH + |ith2 - ith|]`
    /// (`SDSCUM < 0`; one zero otherwise).
    pub cumw: Vec<f64>,
}

/// Index in [`SrceTables::par`] of a namelist parameter of the source terms:
/// `BETAMAX`, `SDSC2`, `NLPROP` (`SNLC1` of the DIA), `BJGAM` (`SDBC2`).
pub fn par_index(name: &str) -> Result<usize> {
    Ok(match name {
        "BETAMAX" => P_BBETA,
        "SDSC2" => P_SDSC2,
        "NLPROP" => P_SNLC1,
        "BJGAM" => P_SDBC2,
        _ => bail!("unknown source parameter {name} (BETAMAX, SDSC2, NLPROP, BJGAM)"),
    })
}

/// The sediment diameter `SED_D50` and critical Shields number `SED_PSIC`
/// of every sea point (`ISEA = 1 .. NSEA`), from the harness parameter file.
pub fn sediment(p: &Params) -> Result<(Vec<f32>, Vec<f32>)> {
    let from1 = |n: &str| -> Result<Vec<f32>> {
        let lb = p.lbound(n)?;
        if lb > 1 {
            bail!("{n} starts at {lb}");
        }
        Ok(p.r(n)?[(1 - lb) as usize..].to_vec())
    };
    Ok((from1("SED_D50")?, from1("SED_PSIC")?))
}

impl SrceTables {
    /// Lay out `q` for the GPU (the sediment diameter and critical Shields
    /// number are per node, see [`sediment`]).
    pub fn new(q: &SrcePar) -> Result<SrceTables> {
        let p = &q.st;
        let dia = &q.dia;
        let (nk, nth) = (p.nk, p.nth);
        let nspec = nk * nth;
        if p.sintailpar[3] < 0.5 {
            bail!("the GPU source kernel assumes SINTAILPAR(4) > 0.5 (TAUWBUG)");
        }
        if p.ssdsdth >= 180.0 || p.nsat == 0 {
            bail!("the GPU source kernel needs the saturation tables of W3SDS4 (SSDSDTH < 180)");
        }
        if p.ssdsc[0].round() as i32 != 1 || p.ssdsc[2] > 0.0 {
            bail!("the GPU source kernel needs SDSBCHOICE = 1 and SDSCUM <= 0");
        }
        let mut par = vec![0f64; NPAR];
        let mut put = |i: usize, v: &[f64]| par[i..i + v.len()].copy_from_slice(v);
        put(P_GRAV, &[p.grav, p.kappa, p.nu_air, q.dair / q.dwat, p.dth, p.fte, p.ftf, p.sstxftwn, p.sstxftftail, p.wwnmeanp, p.wwnmeanptail]);
        put(P_ZZWND, &[p.zzwnd, p.aalpha, p.bbeta, p.zzalp, p.ttauwshelter]);
        put(P_SSWELLF, &p.sswellf);
        put(P_SSINTHP, &[p.ssinthp, p.zz0rat, p.zz0max]);
        put(P_SINTAILPAR, &p.sintailpar);
        put(P_CAPCHNK, &p.capchnk);
        put(P_DELAB, &[p.delab, p.abmin, p.ssdsc[1], p.ssdsc[5], p.ssdsbr, p.ssdsp]);
        put(P_SSDSBM, &p.ssdsbm);
        put(P_SLNC1, &[p.slnc1, p.fspm, p.fshf, p.sdbc1, p.sdbc2, p.fdonly as u8 as f64]);
        put(P_SBTCX, &p.sbtcx);
        put(P_SED_SG, &[p.sed_sg, p.delxerf, p.xerfmax, dia.kdcon, dia.kdmn, dia.snlc1, dia.snls[0], dia.snls[1], dia.snls[2], dia.fachfe]);
        put(P_DAL1, &dia.dal);
        put(P_AWG, &dia.awg);
        put(P_SWG, &dia.swg);
        put(P_DTMAX, &[q.dtmax, q.dtmin, q.xflt, q.xrel, q.facp, q.facti1, q.facti2, q.fachfa, q.fhmax, q.dmin, q.ffxfm, q.ffxpm, q.ffxfa]);
        put(P_ICESCALES, &q.icescales);
        put(P_NFW, &[p.fwtable.len() as f64, p.erftable.len() as f64, q.ice_forcing as u8 as f64, p.ssdsdth]);
        let (mut taut, mut tauhf) = (vec![0.0], vec![0.0]);
        if let Some(t) = &p.tables {
            put(P_SINTABLE, &[1.0, t.delust, t.delalp, t.deltauw, t.delu, t.deltail, t.tauwmax]);
            put(P_ITAUMAX, &[t.itaumax as f64, t.jumax as f64, t.iustar as f64, t.ialpha as f64, t.ilevtail as f64]);
            taut = t.taut.clone();
            tauhf = if p.ttauwshelter > 0.0 {
                if t.tauhft2.is_empty() {
                    bail!("TAUWSHELTER > 0 needs TAUHFT2");
                }
                t.tauhft2.clone()
            } else {
                t.tauhft.clone()
            };
        }
        put(P_SDSC3, &[p.ssdsc[2], p.ssdsc[4], p.ssdsc.get(20).copied().unwrap_or(0.0), p.dikcumul as f64]);
        let mut cumw = vec![0.0];
        if p.ssdsc[2] < 0.0 {
            // CUMULW(IS2, IS) as a function of (IK2, IK, |ITH2 - ITH|), checked
            // against every used entry of the matrix
            cumw = vec![0.0; nk * nk * nth];
            let dik = p.dikcumul.max(0) as usize;
            for ik in 0..nk {
                for ik2 in 0..(ik + 1).saturating_sub(dik) {
                    for ith in 0..nth {
                        for ith2 in 0..nth {
                            let v = p.cumulw[(ik2 * nth + ith2) + (ik * nth + ith) * nspec];
                            let o = (ik2 * nk + ik) * nth + ith2.abs_diff(ith);
                            if ith == 0 || ith2 == 0 {
                                cumw[o] = v;
                            } else if cumw[o] != v {
                                bail!("CUMULW is not a function of the direction difference at ({ik2}, {ith2}; {ik}, {ith})");
                            }
                        }
                    }
                }
            }
        }
        let first = |v: &[f64]| -> Result<Vec<f64>> {
            let mut out = vec![];
            for ik in 0..nk {
                let x = v[ik * nth];
                if v[ik * nth..(ik + 1) * nth].iter().any(|&y| y != x) {
                    bail!("SIG2 or DDEN2 varies with direction at frequency {}", ik + 1);
                }
                out.push(x);
            }
            Ok(out)
        };
        let sati: Vec<u32> = p.satindices.iter().map(|&j| (j - 1) as u32).collect();
        if sati.iter().any(|&j| j as usize >= nth) {
            bail!("SATINDICES out of 1..NTH");
        }
        let off = |v: i32, hi: usize| -> Result<u32> {
            let o = v + nth as i32 - 1;
            if o < 0 || o as usize >= hi + nth {
                bail!("DIA index {v} out of 1-NTH..{hi}");
            }
            Ok(o as u32)
        };
        let mut dia_ip = Vec::with_capacity(16 * dia.nspecx);
        for t in &dia.ip {
            for &v in &t[..dia.nspecx] {
                dia_ip.push(off(v, dia.nspecy)?);
            }
        }
        let mut dia_ic = Vec::with_capacity(16 * nspec);
        for t in &dia.ic {
            for &v in &t[..nspec] {
                dia_ic.push(off(v, dia.nspecx)?);
            }
        }
        if dia.nspecx < nspec || dia.nspecy != dia.nfrhgh * nth {
            bail!("DIA sizes NSPECX {} NSPECY {} unlike NSPEC {nspec}, NFRHGH NTH {}", dia.nspecx, dia.nspecy, dia.nfrhgh * nth);
        }
        Ok(SrceTables {
            par,
            sig2k: first(&p.sig2)?,
            dden2k: first(&p.dden2)?,
            fwtable: p.fwtable.clone(),
            erftable: p.erftable.clone(),
            satw: p.satweights.clone(),
            sati,
            nsat: p.nsat,
            dia_ip,
            dia_ic,
            af11: dia.af11[..dia.nspecx].to_vec(),
            sig: p.sig.clone(),
            dden: p.dden.clone(),
            ecos: p.ecos[..nth].to_vec(),
            esin: p.esin[..nth].to_vec(),
            nspecx: dia.nspecx,
            nspecy: dia.nspecy,
            taut,
            tauhf,
            cumw,
        })
    }
}

/// `TABU_TAUHF2` of `INSIN4`: the tail-stress tables `TAUHFT(0:IUSTAR,
/// 0:IALPHA)` and `TAUHFT2(0:IUSTAR, 0:IALPHA, 0:ILEVTAIL)` for `BETAMAX =
/// bbeta`, in double precision, with their derivatives with respect to
/// `BETAMAX` (forward mode). `sigmax` is `SIG(NK)`. Returns
/// `(TAUHFT, d TAUHFT, TAUHFT2, d TAUHFT2)` in Fortran order from 0.
pub fn tabu_tauhf2(p: &St4, t: &StressTables, sigmax: f64, bbeta: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    use rayon::prelude::*;
    const JTOT: usize = 250;
    let (grav, kappa) = (p.grav, p.kappa);
    let (ius, ial, ilv) = (t.iustar, t.ialpha, t.ilevtail);
    let ustarm = 5.0;
    let alpham = 20.0 * p.aalpha;
    let delust = ustarm / ius as f64;
    let delalp = alpham / ial as f64;
    let deltail = alpham / ilv as f64;
    let const1 = bbeta / (kappa * kappa);
    let dconst1 = 1.0 / (kappa * kappa);
    let omegac = sigmax;
    let x0 = 0.05;
    let n1 = ius + 1;
    let n2 = ial + 1;
    // one K (u* bin) per task: (tauhft, d, tauhft2, d) of that K
    let rows: Vec<_> = (0..=ius)
        .into_par_iter()
        .map(|k| {
            let mut h1 = vec![0f64; n2];
            let mut dh1 = vec![0f64; n2];
            let mut h2 = vec![0f64; n2 * (ilv + 1)];
            let mut dh2 = vec![0f64; n2 * (ilv + 1)];
            let ust0 = (k as f64 * delust).max(0.000001);
            for l in 0..=ial {
                let zz0 = ust0 * ust0 * (p.aalpha + l as f64 * delalp) / grav;
                let omegacc = omegac.max(x0 * grav / ust0);
                let yc = omegacc * (zz0 / grav).sqrt();
                let dely = ((1.0 - yc) / JTOT as f64).max(0.0);
                // as in WW3, UST is set to UST0 once per (K, L): each tail
                // level I starts from the UST the previous level ended with
                let mut ust = ust0;
                let mut dust = 0.0;
                for i in 0..=ilv {
                    let levtail = i as f64 * deltail;
                    let (mut t1, mut dt1, mut t2, mut dt2) = (0.0, 0.0, 0.0, 0.0);
                    let mut tauw = ust0 * ust0;
                    let mut dtauw = 0.0;
                    for j in 1..=JTOT {
                        let w = if j == 1 || j == JTOT { 0.5 } else { 1.0 };
                        let y = yc + (j - 1) as f64 * dely;
                        let omega = y * (grav / zz0).sqrt();
                        let cm = grav / omega;
                        // TAUHFT: u* fixed at UST0
                        let zx = ust0 / cm + p.zzalp;
                        let zarg = (kappa / zx).min(20.0);
                        let zmu = (grav * zz0 / (cm * cm) * zarg.exp()).min(1.0);
                        let zlog = zmu.ln().min(0.0);
                        let zb = zmu * zlog.powi(4);
                        t1 += w * const1 * zb / y * dely;
                        dt1 += w * dconst1 * zb / y * dely;
                        // TAUHFT2: u* reduced by the stress of the tail
                        let zx = ust / cm + p.zzalp;
                        let dzx = dust / cm;
                        let zargv = kappa / zx;
                        let (zarg, dzarg) = if zargv < 20.0 { (zargv, -kappa * dzx / (zx * zx)) } else { (20.0, 0.0) };
                        let zmuv = grav * zz0 / (cm * cm) * zarg.exp();
                        let (zmu, dzmu) = if zmuv < 1.0 { (zmuv, zmuv * dzarg) } else { (1.0, 0.0) };
                        let lnz = zmu.ln();
                        let (zlog, dzlog) = if lnz < 0.0 { (lnz, dzmu / zmu) } else { (0.0, 0.0) };
                        let zbeta = const1 * zmu * zlog.powi(4);
                        let dzbeta = dconst1 * zmu * zlog.powi(4) + const1 * (dzmu * zlog.powi(4) + zmu * 4.0 * zlog.powi(3) * dzlog);
                        let r = ust / ust0;
                        let dr = dust / ust0;
                        t2 += w * zbeta * r * r / y * dely;
                        dt2 += w * (dzbeta * r * r + zbeta * 2.0 * r * dr) / y * dely;
                        tauw -= w * ust * ust * zbeta * levtail / y * dely;
                        dtauw -= w * (2.0 * ust * dust * zbeta + ust * ust * dzbeta) * levtail / y * dely;
                        if tauw > 0.0 {
                            ust = tauw.sqrt();
                            dust = 0.5 * dtauw / ust;
                        } else {
                            ust = 0.0;
                            dust = 0.0;
                        }
                    }
                    if i == ilv {
                        h1[l] = t1;
                        dh1[l] = dt1;
                    }
                    h2[l + i * n2] = t2;
                    dh2[l + i * n2] = dt2;
                }
            }
            (h1, dh1, h2, dh2)
        })
        .collect();
    let (mut tauhft, mut dtauhft) = (vec![0f64; n1 * n2], vec![0f64; n1 * n2]);
    let (mut tauhft2, mut dtauhft2) = (vec![0f64; n1 * n2 * (ilv + 1)], vec![0f64; n1 * n2 * (ilv + 1)]);
    for (k, (h1, dh1, h2, dh2)) in rows.into_iter().enumerate() {
        for l in 0..n2 {
            tauhft[k + l * n1] = h1[l];
            dtauhft[k + l * n1] = dh1[l];
            for i in 0..=ilv {
                tauhft2[k + l * n1 + i * n1 * n2] = h2[l + i * n2];
                dtauhft2[k + l * n1 + i * n1 * n2] = dh2[l + i * n2];
            }
        }
    }
    (tauhft, dtauhft, tauhft2, dtauhft2)
}

/// `TABU_STRESS` of `INSIN4`: the friction velocity table
/// `TAUT(0:ITAUMAX, 0:JUMAX) = SQRT(TAU)` as a function of the wave-supported
/// stress (`I DELTAUW`)^2 and the wind speed `J DELU`, from the steady wind
/// profile with the roughness length of Charnock corrected for the
/// quasi-linear effect, `Z0 / SQRT(1 - TAUW / TAU)`, by 10 Newton
/// iterations; zero for a zero wind. Fortran order from 0. The iterations
/// are done in single precision, as WW3 does them: near `TAUWMAX` at strong
/// winds `1 - TAUW / TAU` cancels, and the table of WW3 differs there from
/// a double-precision one by up to 1.2e-4 of its largest value (474 of the
/// 40,401 cells above 1e-5), while this one reproduces it to 8e-8. It does
/// not depend on `BETAMAX`.
pub fn tabu_stress(p: &St4, t: &StressTables) -> Vec<f64> {
    let f = |x: f64| x as f32;
    let (ni, nj) = (t.itaumax, t.jumax);
    let (xm, eps1, wcd) = (0.5f32, 0.00001f32, 0.0012875f32.sqrt());
    let (aalpha, grav, zz0max, zzwnd, kappa) = (f(p.aalpha), f(p.grav), f(p.zz0max), f(p.zzwnd), f(p.kappa));
    let mut out = vec![0f64; (ni + 1) * (nj + 1)];
    for j in 1..=nj {
        let utop = j as f32 * f(t.delu);
        for i in 0..=ni {
            let ztauw = (i as f32 * f(t.deltauw)).powi(2);
            let ustold = utop * wcd;
            let mut tauold = (ustold * ustold).max(ztauw + eps1);
            for _ in 0..10 {
                let x = ztauw / tauold;
                let mut ust = tauold.sqrt();
                let mut zz00 = aalpha * tauold / grav;
                if zz0max != 0.0 {
                    zz00 = zz00.min(zz0max);
                }
                let zz0 = zz00 / (1.0 - x).powf(xm);
                let l = (zzwnd / zz0).ln();
                let fx = ust - kappa * utop / l;
                let delf = 1.0 - kappa * utop / (l * l) * 2.0 / ust * (1.0 - (xm + 1.0) * x) / (1.0 - x);
                ust -= fx / delf;
                tauold = (ust * ust).max(ztauw + eps1);
            }
            out[i + j * (ni + 1)] = tauold.sqrt() as f64;
        }
    }
    out
}

/// The stress tables of `INSIN4` (`TAUT`, `TAUHFT` and, with sheltering,
/// `TAUHFT2`) computed here for the `BETAMAX` of `p` ([`tabu_stress`],
/// [`tabu_tauhf2`] in double precision),
/// in place of those read from the parameter file; their derivative with
/// respect to `BETAMAX` comes from the same [`tabu_tauhf2`].
pub fn rebuild_stress_tables(p: &mut St4) -> Result<()> {
    let t = p.tables.as_ref().ok_or_else(|| anyhow::anyhow!("no stress tables (SINTABLE = 0)"))?;
    let taut = tabu_stress(p, t);
    let (h1, _, h2, _) = tabu_tauhf2(p, t, p.sig[p.nk], p.bbeta);
    let tm = p.tables.as_mut().unwrap();
    tm.taut = taut;
    tm.tauhft = h1;
    if !tm.tauhft2.is_empty() {
        tm.tauhft2 = h2;
    }
    Ok(())
}
