//! The source-term part of `ww3_grid` in Rust (`W3GRID` of `w3gridmd.F90`
//! with `INSIN4`, `INSNL1`, `INSBT4`, `TABU_FW`, `TABU_ERF`): the parameter
//! file of the source terms ([`Params`], the records the WW3 harness writes)
//! built from the namelists of the run and the spectral grid, with the WW3
//! defaults of the switches of HWW3 (ST4, NL1, LN1, BT4, DB1). With it a GPU
//! run needs no `ww3_grid` and no harness.
//!
//! The scalars are computed in double precision and stored in single
//! precision as WW3 holds them; the tables follow the Fortran (`TAUT` in
//! single precision, see [`crate::sources::tabu_stress`]).

use crate::harness::{ParamArray, Params};
use crate::sources::{tabu_stress, tabu_tauhf2, St4};
use crate::spec::Spec;
use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::path::Path;

/// WW3 constants (`constants.F90`).
const GRAV: f64 = 9.806;
const DWAT: f64 = 1000.0;
const DAIR: f64 = 1.225;
const NU_AIR: f64 = 1.4e-5;
const NU_WATER: f64 = 1.31e-6;
const SED_SG: f64 = 2.65;
const KAPPA: f64 = 0.40;
const PI: f64 = 3.141592653589793;
const TPI: f64 = 2.0 * PI;
const RADE: f64 = 180.0 / PI;
const SIZEFWTABLE: usize = 300;
const ABMIN: f64 = -1.0;
const ABMAX: f64 = 8.0;
const SIZEERFTABLE: usize = 300;
const XERFMAX: f64 = 4.0;
/// Sizes and ranges of the stress tables (`W3SRC4MD`).
const ITAUMAX: usize = 200;
const JUMAX: usize = 200;
const IUSTAR: usize = 100;
const IALPHA: usize = 200;
const ILEVTAIL: usize = 50;
const UMAX: f64 = 50.0;
const TAUWMAX: f64 = 2.2361;
/// `FACHF`, power of the high-frequency tail (ST4).
const FACHF: f64 = 5.0;

/// The namelist groups of a WW3 namelist file (`&NAME KEY = value, ... /`),
/// keys in upper case.
#[derive(Clone, Debug, Default)]
pub struct Namelists {
    /// Values by group and key, as written.
    pub groups: HashMap<String, HashMap<String, String>>,
}

impl Namelists {
    /// Parse the text of a namelist file (comments after `!`, values
    /// separated by commas or new lines, a group ended by `/`).
    pub fn parse(text: &str) -> Namelists {
        let mut groups: HashMap<String, HashMap<String, String>> = HashMap::new();
        let mut body = String::new();
        for line in text.lines() {
            let l = match line.find('!') {
                Some(i) => &line[..i],
                None => line,
            };
            body.push_str(l);
            body.push('\n');
        }
        let mut rest = body.as_str();
        while let Some(i) = rest.find('&') {
            rest = &rest[i + 1..];
            let name_end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
            let name = rest[..name_end].to_uppercase();
            let end = rest.find('/').unwrap_or(rest.len());
            let content = &rest[name_end..end];
            let g = groups.entry(name).or_default();
            // KEY = value pairs: split at '=' and take the last token before each as the key
            let parts: Vec<&str> = content.split('=').collect();
            for k in 0..parts.len().saturating_sub(1) {
                let key = parts[k].split(|c: char| c == ',' || c.is_whitespace()).filter(|s| !s.is_empty()).last().unwrap_or("");
                let mut val = parts[k + 1].to_string();
                if k + 1 < parts.len() - 1 {
                    // drop the next key from the value
                    let toks: Vec<&str> = val.split(|c: char| c == ',' || c.is_whitespace()).filter(|s| !s.is_empty()).collect();
                    val = toks[..toks.len().saturating_sub(1)].join(" ");
                }
                let v = val.trim().trim_end_matches(',').trim().to_string();
                if !key.is_empty() {
                    g.insert(key.to_uppercase(), v);
                }
            }
            rest = &rest[end.min(rest.len())..];
        }
        Namelists { groups }
    }

    /// Read a namelist file.
    pub fn read(p: &Path) -> Result<Namelists> {
        Ok(Namelists::parse(&std::fs::read_to_string(p).map_err(|e| anyhow!("{}: {e}", p.display()))?))
    }

    fn raw(&self, group: &str, key: &str) -> Option<&String> {
        self.groups.get(group).and_then(|g| g.get(key))
    }

    /// Real value of `group`/`key`, or `default`.
    pub fn f(&self, group: &str, key: &str, default: f64) -> Result<f64> {
        match self.raw(group, key) {
            Some(v) => v.replace(['d', 'D'], "e").parse::<f64>().map_err(|e| anyhow!("&{group} {key} = {v}: {e}")),
            None => Ok(default),
        }
    }

    /// String value of `group`/`key`, without its quotes.
    pub fn s(&self, group: &str, key: &str) -> Option<String> {
        self.raw(group, key).map(|v| v.trim().trim_matches(|c| c == '\'' || c == '"').trim().to_string())
    }

    /// Logical value of `group`/`key`, or `default`.
    pub fn b(&self, group: &str, key: &str, default: bool) -> Result<bool> {
        match self.raw(group, key) {
            Some(v) => {
                let u = v.trim_matches('.').to_uppercase();
                match u.chars().next() {
                    Some('T') => Ok(true),
                    Some('F') => Ok(false),
                    _ => bail!("&{group} {key} = {v}: not a logical"),
                }
            }
            None => Ok(default),
        }
    }
}

/// The grid quantities the source-term parameters need, from `ww3_grid.nml`
/// and the mesh.
#[derive(Clone, Debug)]
pub struct GridSetup {
    /// Number of frequencies and directions.
    pub nk: usize,
    /// See [`GridSetup::nk`].
    pub nth: usize,
    /// Lowest frequency (Hz), frequency factor and direction offset.
    pub fr1: f64,
    /// See [`GridSetup::fr1`].
    pub xfr: f64,
    /// See [`GridSetup::fr1`].
    pub thoff: f64,
    /// Minimum depth `DMIN` (m) and the time steps `DTMAX`, `DTMIN` (s).
    pub dmin: f64,
    /// See [`GridSetup::dmin`].
    pub dtmax: f64,
    /// See [`GridSetup::dmin`].
    pub dtmin: f64,
    /// Number of sea points (`NSEA`).
    pub nsea: usize,
}

fn rec_r(p: &mut Params, name: &str, lbound: &[i32], shape: &[usize], v: &[f64]) {
    assert_eq!(shape.iter().product::<usize>(), v.len(), "{name}");
    p.map.insert(
        name.to_string(),
        ParamArray { lbound: lbound.to_vec(), shape: shape.to_vec(), real: v.iter().map(|&x| x as f32).collect(), int: vec![] },
    );
}

fn rec_i(p: &mut Params, name: &str, lbound: &[i32], shape: &[usize], v: &[i32]) {
    assert_eq!(shape.iter().product::<usize>(), v.len(), "{name}");
    p.map.insert(name.to_string(), ParamArray { lbound: lbound.to_vec(), shape: shape.to_vec(), real: vec![], int: v.to_vec() });
}

fn scalar(p: &mut Params, name: &str, v: f64) {
    rec_r(p, name, &[1], &[1], &[v]);
}

fn iscalar(p: &mut Params, name: &str, v: i32) {
    rec_i(p, name, &[1], &[1], &[v]);
}

/// `KZEONE` (ACM algorithm): `exp(x) K0` and `exp(x) K1` of the complex
/// argument `x + i y`, as `(re0, im0, re1, im1)`.
fn kzeone(x: f64, y: f64) -> (f64, f64, f64, f64) {
    const EXSQ: [f64; 8] = [
        0.5641003087264, 0.4120286874989, 0.1584889157959, 0.3078003387255e-1, 0.2778068842913e-2, 0.1000044412325e-3, 0.1059115547711e-5,
        0.1522475804254e-8,
    ];
    const TSQ: [f64; 8] = [0.0, 3.19303633920635e-1, 1.29075862295915, 2.95837445869665, 5.40903159724444, 8.80407957805676, 1.34685357432515e1, 2.02499163658709e1];
    let mut r2 = x * x + y * y;
    let (mut re0, mut im0, mut re1, mut im1);
    if r2 >= 1.96e2 {
        let mut rterm = 1.0;
        let mut iterm = 0.0;
        re0 = 1.0;
        im0 = 0.0;
        re1 = 1.0;
        im1 = 0.0;
        let p1 = 8.0 * r2;
        let p2 = r2.sqrt();
        let l = 3.91 + 8.12e1 / p2;
        let mut r1 = 1.0;
        r2 = 1.0;
        let mut m: i64 = -8;
        let mut k: i64 = 3;
        for n in 1..=(l as i64) {
            m += 8;
            k -= m;
            r1 *= (k - 4) as f32 as f64;
            r2 *= k as f32 as f64;
            let t1 = (n as f32 as f64) * p1;
            let t2 = rterm;
            rterm = (t2 * x + iterm * y) / t1;
            iterm = (-t2 * y + iterm * x) / t1;
            re0 += r1 * rterm;
            im0 += r1 * iterm;
            re1 += r2 * rterm;
            im1 += r2 * iterm;
        }
        let t1 = (p2 + x).sqrt();
        let t2 = -y / t1;
        let p1 = 8.86226925452758e-1 / p2;
        let rterm = p1 * y.cos();
        let iterm = -p1 * y.sin();
        let r1 = re0 * rterm - im0 * iterm;
        let r2b = re0 * iterm + im0 * rterm;
        re0 = t1 * r1 - t2 * r2b;
        im0 = t1 * r2b + t2 * r1;
        let r1 = re1 * rterm - im1 * iterm;
        let r2b = re1 * iterm + im1 * rterm;
        re1 = t1 * r1 - t2 * r2b;
        im1 = t1 * r2b + t2 * r1;
    } else if r2 >= 1.849e1 {
        let x2 = 2.0 * x;
        let y2 = 2.0 * y;
        let r1 = y2 * y2;
        let mut p1 = (x2 * x2 + r1).sqrt();
        let mut p2 = (p1 + x2).sqrt();
        let mut t1 = EXSQ[0] / (2.0 * p1);
        re0 = t1 * p2;
        im0 = t1 / p2;
        re1 = 0.0;
        im1 = 0.0;
        for n in 1..8 {
            let t2 = x2 + TSQ[n];
            p1 = (t2 * t2 + r1).sqrt();
            p2 = (p1 + t2).sqrt();
            t1 = EXSQ[n] / p1;
            re0 += t1 * p2;
            im0 += t1 / p2;
            t1 = EXSQ[n] * TSQ[n];
            re1 += t1 * p2;
            im1 += t1 / p2;
        }
        let t2 = -y2 * im0;
        re1 /= r2;
        let r2b = y2 * im1 / r2;
        let rterm = 1.41421356237309 * y.cos();
        let iterm = -1.41421356237309 * y.sin();
        im0 = re0 * iterm + t2 * rterm;
        re0 = re0 * rterm - t2 * iterm;
        let t1 = re1 * rterm - r2b * iterm;
        let t2 = re1 * iterm + r2b * rterm;
        re1 = t1 * x + t2 * y;
        im1 = -t1 * y + t2 * x;
    } else {
        let mut x2 = x / 2.0;
        let mut y2 = y / 2.0;
        let p1 = x2 * x2;
        let p2 = y2 * y2;
        let mut t1 = -((p1 + p2).ln() / 2.0 + 0.5772156649015329);
        let t2 = -y.atan2(x);
        x2 = p1 - p2;
        y2 *= x;
        let mut rterm = 1.0;
        let mut iterm = 0.0;
        re0 = t1;
        im0 = t2;
        t1 += 0.5;
        re1 = t1;
        im1 = t2;
        let p2 = r2.sqrt();
        let mut l = 2.106 * p2 + 4.4;
        if p2 < 8.0e-1 {
            l = 2.129 * p2 + 4.0;
        }
        for n in 1..=(l as i64) {
            let mut p1 = n as f64;
            let p2 = (n * n) as f64;
            let r1 = rterm;
            rterm = (r1 * x2 - iterm * y2) / p2;
            iterm = (r1 * y2 + iterm * x2) / p2;
            t1 += 0.5 / p1;
            re0 += t1 * rterm - t2 * iterm;
            im0 += t1 * iterm + t2 * rterm;
            p1 += 1.0;
            t1 += 0.5 / p1;
            re1 += (t1 * rterm - t2 * iterm) / p1;
            im1 += (t1 * iterm + t2 * rterm) / p1;
        }
        let r1 = x / r2 - 0.5 * (x * re1 - y * im1);
        let r2b = -y / r2 - 0.5 * (x * im1 + y * re1);
        let p1 = x.exp();
        re0 *= p1;
        im0 *= p1;
        re1 = p1 * r1;
        im1 = p1 * r2b;
    }
    (re0, im0, re1, im1)
}

/// `KERKEI`: the Kelvin functions `ker x` and `kei x` (single-precision
/// argument and results, as WW3).
fn kerkei(x: f32) -> (f32, f32) {
    let zr = x as f64 * 0.5 * 2.0f64.sqrt();
    let (cyr, cyi, _, _) = kzeone(zr, zr);
    ((cyr / zr.exp()) as f32, (cyi / zr.exp()) as f32)
}

/// `TABU_FW` (`constants.F90`): the friction factor of the wave bottom
/// boundary layer as a function of the log10 of the normalised roughness,
/// from `ABMIN` in steps `DELAB`, in single precision as WW3.
pub fn tabu_fw() -> (Vec<f64>, f64) {
    let delab = ((ABMAX - ABMIN) as f32 / SIZEFWTABLE as f32) as f64;
    let l10 = 10f32.ln();
    let mut out = vec![0f64; SIZEFWTABLE + 1];
    for (i, o) in out.iter_mut().enumerate() {
        let abrlog = ABMIN as f32 + i as f32 * delab as f32;
        let abr = (abrlog * l10).exp();
        let fact = 1.0 / abr / (21.2 * KAPPA as f32);
        let mut fsubw = 0.05f32;
        let mut dzeta0 = 0f32;
        for _ in 0..100 {
            let fsubwmemo = fsubw;
            let dzeta0memo = dzeta0;
            dzeta0 = fact * fsubw.powf(-0.5);
            let (ker, kei) = kerkei(2.0 * dzeta0.sqrt());
            fsubw = 0.08 / (ker * ker + kei * kei);
            fsubw = 0.5 * (fsubwmemo + fsubw);
            dzeta0 = 0.5 * (dzeta0memo + dzeta0);
        }
        *o = fsubw.min(0.5) as f64;
    }
    (out, delab)
}

/// `TABU_ERF` (`W3SBT4MD`): `(1 + erf x) / 2` from `-XERFMAX` in steps
/// `DELXERF` (WW3 writes `2**(1/2)` for the factor, which is 1 in integer
/// arithmetic).
pub fn tabu_erf() -> (Vec<f64>, f64) {
    let delxerf = ((2.0 * XERFMAX) as f32 / SIZEERFTABLE as f32) as f64;
    let out = (0..=SIZEERFTABLE)
        .map(|i| {
            let x = (-XERFMAX as f32 + i as f32 * delxerf as f32) as f64;
            let e = libm_erf(x);
            if x < 0.0 {
                (1.0 - e.abs()) / 2.0
            } else {
                (1.0 + e) / 2.0
            }
        })
        .collect();
    (out, delxerf)
}

/// The error function (Abramowitz and Stegun 7.1.26 is not precise enough:
/// a series for |x| < 2.5 and the continued fraction beyond, to 1e-14).
fn libm_erf(x: f64) -> f64 {
    let a = x.abs();
    let r = if a == 0.0 {
        0.0
    } else if a < 2.5 {
        // Maclaurin series
        let mut term = a;
        let mut sum = a;
        let x2 = a * a;
        let mut n = 0.0;
        loop {
            n += 1.0;
            term *= -x2 / n;
            let t = term / (2.0 * n + 1.0);
            sum += t;
            if t.abs() <= 1e-17 * sum.abs() || n > 200.0 {
                break;
            }
        }
        2.0 / PI.sqrt() * sum
    } else {
        // erfc by the continued fraction of Lentz
        let tiny = 1e-300;
        let mut f = a;
        let mut c = a;
        let mut d = 0.0;
        for k in 1..200 {
            let an = k as f64 / 2.0;
            d = a + an * d;
            d = if d.abs() < tiny { tiny } else { d };
            c = a + an / c;
            c = if c.abs() < tiny { tiny } else { c };
            d = 1.0 / d;
            let delta = c * d;
            f *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        1.0 - (-a * a).exp() / (PI.sqrt() * f)
    };
    if x < 0.0 {
        -r
    } else {
        r
    }
}

/// The parameter file of the source terms for the grid `g` and the
/// namelists `nml` (those of `namelists_<grid>.nml`; anything not set takes
/// the WW3 default of the HWW3 switches): every record the harness writes,
/// with the stress tables (`TAUT`, `TAUHFT`, `TAUHFT2`) computed for its
/// `BETAMAX`.
pub fn source_params(g: &GridSetup, nml: &Namelists) -> Result<Params> {
    let mut p = Params::default();
    let (nk, nth) = (g.nk, g.nth);
    let nspec = nk * nth;
    let sp = Spec::new(nk, nth, g.fr1 as f32, g.xfr as f32, g.thoff as f32);
    let xfr = sp.xfr as f64;
    let dth = sp.dth as f64;
    let sig: Vec<f64> = sp.sig.iter().map(|&x| x as f64).collect();
    let sigk = sig[nk];
    // spectral grid
    iscalar(&mut p, "NK", nk as i32);
    iscalar(&mut p, "NTH", nth as i32);
    iscalar(&mut p, "NSPEC", nspec as i32);
    rec_r(&mut p, "SIG", &[0], &[nk + 2], &sig);
    rec_r(&mut p, "DSIP", &[0], &[nk + 2], &sp.dsip.iter().map(|&x| x as f64).collect::<Vec<_>>());
    rec_r(&mut p, "DDEN", &[1], &[nk], &sp.dden.iter().map(|&x| x as f64).collect::<Vec<_>>());
    let sig2: Vec<f64> = (0..nspec).map(|is| sig[is / nth + 1]).collect();
    let dden2: Vec<f64> = (0..nspec).map(|is| sp.dden[is / nth] as f64).collect();
    rec_r(&mut p, "SIG2", &[1], &[nspec], &sig2);
    rec_r(&mut p, "DDEN2", &[1], &[nspec], &dden2);
    let ne = nspec + nth;
    rec_r(&mut p, "ECOS", &[1], &[ne], &(0..ne).map(|i| sp.ecos[i % nth] as f64).collect::<Vec<_>>());
    rec_r(&mut p, "ESIN", &[1], &[ne], &(0..ne).map(|i| sp.esin[i % nth] as f64).collect::<Vec<_>>());
    scalar(&mut p, "DTH", dth);
    scalar(&mut p, "XFR", xfr);
    scalar(&mut p, "FR1", g.fr1);
    // constants
    for (n, v) in [("GRAV", GRAV), ("KAPPA", KAPPA), ("NU_AIR", NU_AIR), ("DWAT", DWAT), ("DAIR", DAIR), ("SED_SG", SED_SG)] {
        scalar(&mut p, n, v);
    }
    // W3SRCE integration (MISC, TIMESTEPS)
    let f = |gr: &str, k: &str, d: f64| nml.f(gr, k, d);
    let xp = f("MISC", "XP", 0.15)?.max(1e-6);
    let xr = f("MISC", "XR", 0.10)?.max(1e-6);
    let xfilt = f("MISC", "XFILT", 0.05)?.max(0.0);
    let fmiche = f("MISC", "FMICHE", 1.6)?;
    scalar(&mut p, "FACP", xp / PI * 0.62e-3 * TPI.powi(4) / (GRAV * GRAV));
    scalar(&mut p, "XREL", xr);
    scalar(&mut p, "XFLT", xfilt);
    scalar(&mut p, "FHMAX", fmiche.max(0.01));
    let ice: Vec<f64> = ["ICESLN", "ICEWIND", "ICESNL", "ICESDS"].iter().map(|k| f("MISC", k, 1.0).map(|v| v.clamp(0.0, 1.0))).collect::<Result<_>>()?;
    rec_r(&mut p, "ICESCALES", &[1], &[4], &ice);
    scalar(&mut p, "DTMAX", g.dtmax);
    scalar(&mut p, "DTMIN", g.dtmin.max(0.0).min(g.dtmax));
    scalar(&mut p, "DMIN", g.dmin.max(1e-3));
    let facti1 = 1.0 / (xfr as f32).ln() as f64;
    scalar(&mut p, "FACTI1", facti1);
    scalar(&mut p, "FACTI2", 1.0 - ((TPI * g.fr1) as f32).ln() as f64 * facti1);
    scalar(&mut p, "FACHFA", xfr.powf(-FACHF - 2.0));
    scalar(&mut p, "FACHFE", xfr.powf(-FACHF));
    scalar(&mut p, "FTE", 0.25 * sigk * dth * sigk);
    scalar(&mut p, "FTF", 0.20 * dth * sigk);
    // SLN1
    let clin = f("SLN1", "CLIN", 80.0)?.max(0.0);
    scalar(&mut p, "SLNC1", clin * (DAIR / DWAT).powi(2) / (GRAV * GRAV));
    scalar(&mut p, "FSPM", f("SLN1", "RFPM", 1.0)?.max(0.0));
    scalar(&mut p, "FSHF", f("SLN1", "RFHF", 0.5)?.clamp(0.0, 1.0));
    // SIN4
    let alpha0 = f("SIN4", "ALPHA0", 0.0095)?;
    let betamax = f("SIN4", "BETAMAX", 1.43)?;
    scalar(&mut p, "ZZWND", f("SIN4", "ZWND", 10.0)?);
    scalar(&mut p, "AALPHA", alpha0);
    scalar(&mut p, "BBETA", betamax);
    scalar(&mut p, "ZZALP", f("SIN4", "ZALP", 0.006)?);
    scalar(&mut p, "TTAUWSHELTER", f("SIN4", "TAUWSHELTER", 0.3)?);
    scalar(&mut p, "SSINTHP", f("SIN4", "SINTHP", 2.0)?);
    scalar(&mut p, "ZZ0RAT", f("SIN4", "Z0RAT", 0.04)?);
    scalar(&mut p, "ZZ0MAX", f("SIN4", "Z0MAX", 0.0)?);
    let swellf = [
        f("SIN4", "SWELLF", 0.66)?,
        f("SIN4", "SWELLF2", -0.018)?,
        f("SIN4", "SWELLF3", 0.022)?,
        f("SIN4", "SWELLF4", 1.5e5)?,
        f("SIN4", "SWELLF5", 1.2)?,
        f("SIN4", "SWELLF6", 0.0)?,
        f("SIN4", "SWELLF7", 360000.0)?,
    ];
    rec_r(&mut p, "SSWELLF", &[1], &[7], &swellf);
    let sintable = f("SIN4", "SINTABLE", 1.0)?;
    let sintail = [sintable, f("SIN4", "SINTAIL1", 0.0)?, f("SIN4", "SINTAIL2", 0.0)?, f("SIN4", "TAUWBUG", 1.0)?, f("SIN4", "VISCSTRESS", 0.0)?];
    rec_r(&mut p, "SINTAILPAR", &[1], &[5], &sintail);
    // CAPCHNK(1:10), five of them set
    let mut capch = vec![f("SIN4", "CAPCHA", 0.0)?, f("SIN4", "CHAMIN", 0.0001)?, f("SIN4", "CHA0", alpha0)?, f("SIN4", "UCAP", 30.0)?, f("SIN4", "SIGMAUCAP", 10.0)?];
    capch.resize(10, 0.0);
    rec_r(&mut p, "CAPCHNK", &[1], &[10], &capch);
    // SDS4
    let wnmeanp = f("SDS4", "WNMEANP", 0.5)?;
    let wnmeanptail = f("SDS4", "WNMEANPTAIL", -0.5)?;
    scalar(&mut p, "WWNMEANP", wnmeanp);
    scalar(&mut p, "WWNMEANPTAIL", wnmeanptail);
    scalar(&mut p, "FFXFM", f("SDS4", "FXFM3", 2.5)? * TPI);
    scalar(&mut p, "FFXFA", f("SDS4", "FXFMAGE", 0.0)? * TPI);
    scalar(&mut p, "FFXPM", f("SDS4", "FXPM3", 4.0)? * GRAV / 28.0);
    let sdscum = f("SDS4", "SDSCUM", -0.40344)?;
    let ssdsc = [
        f("SDS4", "SDSBCHOICE", 1.0)?,
        f("SDS4", "SDSC2", -2.2e-5)?,
        sdscum,
        f("SDS4", "SDSC4", 1.0)?,
        f("SDS4", "SDSC5", 0.0)?,
        f("SDS4", "SDSC6", 0.3)?,
        f("SDS4", "WHITECAPWIDTH", 0.3)?,
        f("SDS4", "SDSSTRAIN", 0.0)?,
        f("SDS4", "SDSL", 3.5e-5)?,
        f("SDS4", "SDSSTRAINA", 15.0)? * nth as f64 / 360.0,
        f("SDS4", "SDSSTRAIN2", 0.0)?,
        f("SDS4", "CUMSIGP", 0.0)?,
        f("SDS4", "SDSMWD", 0.9)?,
        f("SDS4", "SPMSS", 0.5)?,
        f("SDS4", "SDSMWPOW", 1.0)?,
        f("SDS4", "SDKOF", 3.0)?,
        f("SDS4", "WHITECAPDUR", 0.56)?,
        f("SDS4", "SDSFACMTF", 400.0)?,
        f("SDS4", "SDSNMTF", 1.5)?,
        f("SDS4", "SDSCUMP", 2.0)?,
        f("SDS4", "SDSNUW", 0.0)?,
    ];
    rec_r(&mut p, "SSDSC", &[1], &[21], &ssdsc);
    scalar(&mut p, "SSDSBR", f("SDS4", "SDSBR", 0.9e-3)?);
    scalar(&mut p, "SSDSBT", f("SDS4", "SDSBT", 1.1e-3)?);
    scalar(&mut p, "SSDSP", f("SDS4", "SDSP", 2.0)?);
    let sdsdth = f("SDS4", "SDSDTH", 80.0)?;
    scalar(&mut p, "SSDSDTH", sdsdth);
    let sdsbm = [f("SDS4", "SDSBM0", 1.0)?, f("SDS4", "SDSBM1", 0.0)?, f("SDS4", "SDSBM2", 0.0)?, f("SDS4", "SDSBM3", 0.0)?, f("SDS4", "SDSBM4", 0.0)?];
    rec_r(&mut p, "SSDSBM", &[0], &[5], &sdsbm);
    let sdsbrf1 = f("SDS4", "SDSBRF1", 0.5)?;
    scalar(&mut p, "SSDSBRF1", sdsbrf1);
    // the tail integrals of W3SPR4
    let sk = sigk;
    scalar(&mut p, "SSTXFTFTAIL", 1.0 / (FACHF - 1.0 - wnmeanptail * 2.0) * sk.powf(2.0 + wnmeanptail * 2.0) * dth);
    scalar(&mut p, "SSTXFTWN", 1.0 / (FACHF - 1.0 - wnmeanp * 2.0) * sk * sk * (sk / GRAV.sqrt()).powf(wnmeanp * 2.0) * dth);
    // saturation of W3SDS4 (INSIN4)
    let sdscos = f("SDS4", "SDSCOS", 2.0)?;
    if sdsdth < 180.0 {
        let sdsnth = ((sdsdth / (dth * RADE)).round() as i64).min(nth as i64 / 2 - 1) as usize;
        // W3DIMS allocates 2 (NTH/2 - 1) + 1 rows; the unused ones keep index 1 and weight 0
        let ns = 2 * (nth / 2 - 1) + 1;
        let mut ind = vec![1i32; ns * nth];
        let mut w = vec![0f64; ns * nth];
        let th = |i: usize| (dth as f32 * (g.thoff as f32 + i as f32)) as f64;
        for ith in 1..=nth {
            for i_int in (ith as i64 - sdsnth as i64)..=(ith as i64 + sdsnth as i64) {
                let mut j = i_int;
                if i_int < 1 {
                    j = i_int + nth as i64;
                }
                if i_int > nth as i64 {
                    j = i_int - nth as i64;
                }
                let r = (i_int - (ith as i64 - sdsnth as i64)) as usize;
                ind[r + (ith - 1) * ns] = j as i32;
                w[r + (ith - 1) * ns] = ((th(ith - 1) - th(j as usize - 1)) as f32).cos().powf(sdscos as f32) as f64;
            }
        }
        rec_i(&mut p, "SATINDICES", &[1, 1], &[ns, nth], &ind);
        rec_r(&mut p, "SATWEIGHTS", &[1, 1], &[ns, nth], &w);
    }
    // cumulative breaking term (INSIN4)
    let mut dikcumul = 0;
    if sdscum < 0.0 {
        dikcumul = (sdsbrf1 / (xfr - 1.0)).round() as i32;
        let mut cw = vec![0f64; nspec * nspec];
        for ik in 1..=nk {
            let c = GRAV / sig[ik];
            for ith in 1..=nth {
                let is = ith + (ik - 1) * nth;
                for ik2 in 1..=(ik as i64 - dikcumul as i64).max(0) as usize {
                    let c2 = GRAV / sig[ik2];
                    for ith2 in 1..=nth {
                        let is2 = ith2 + (ik2 - 1) * nth;
                        let e = sp.ecos[(ith2 as i64 - ith as i64).unsigned_abs() as usize] as f64;
                        cw[(is2 - 1) + (is - 1) * nspec] = (c * c + c2 * c2 - 2.0 * c * c2 * e).sqrt() * sp.dsip[ik2] as f64 / (0.5 * c2) * dth;
                    }
                }
            }
        }
        rec_r(&mut p, "CUMULW", &[1, 1], &[nspec, nspec], &cw);
    }
    iscalar(&mut p, "DIKCUMUL", dikcumul);
    // SNL1 and INSNL1 (DIA)
    let lambda = f("SNL1", "LAMBDA", 0.25)?;
    scalar(&mut p, "SNLC1", f("SNL1", "NLPROP", 2.5e7)? / GRAV.powi(4));
    scalar(&mut p, "KDCON", f("SNL1", "KDCONV", 0.75)?);
    scalar(&mut p, "KDMN", f("SNL1", "KDMIN", 0.50)?);
    scalar(&mut p, "SNLS1", f("SNL1", "SNLCS1", 5.5)?);
    scalar(&mut p, "SNLS2", f("SNL1", "SNLCS2", 0.833)?);
    scalar(&mut p, "SNLS3", f("SNL1", "SNLCS3", -1.25)?);
    iscalar(&mut p, "IQTPE", 1);
    insnl1(&mut p, nk, nth, xfr, dth, &sig, lambda);
    // entrained air (MISC)
    let aircmin = f("MISC", "AIRCMIN", 2.0)?;
    scalar(&mut p, "AAIRCMIN", (GRAV / aircmin / sig[1]).ln() / xfr.ln() + 1.0);
    scalar(&mut p, "AAIRGB", f("MISC", "AIRGB", 0.2)?);
    // output flags the harness records (FLOGRD(5, 7) and (5, 8): not used by the source terms)
    iscalar(&mut p, "FLOGRD57", 0);
    iscalar(&mut p, "FLOGRD58", 0);
    // SDB1
    scalar(&mut p, "SDBC1", f("SDB1", "BJALFA", 1.0)?.max(0.0));
    scalar(&mut p, "SDBC2", f("SDB1", "BJGAM", 0.73)?.max(0.0));
    iscalar(&mut p, "FDONLY", nml.b("SDB1", "BJFLAG", true)? as i32);
    // SBT4 and its tables
    let mut sbtcx = vec![
        f("SBT4", "RIPFAC1", 0.4)?,
        f("SBT4", "RIPFAC2", -2.5)?,
        f("SBT4", "RIPFAC3", 1.2)?,
        f("SBT4", "RIPFAC4", 0.05)?,
        f("SBT4", "SIGDEPTH", 0.05)?,
        f("SBT4", "BOTROUGHMIN", 0.01)?,
        f("SBT4", "BOTROUGHFAC", 1.0)?,
    ];
    sbtcx.resize(10, 0.0);
    rec_r(&mut p, "SBTCX", &[1], &[10], &sbtcx);
    if nml.b("SBT4", "SEDMAPD50", false)? {
        bail!("SEDMAPD50 = T (a map of grain sizes) is not ported");
    }
    let d50 = (f("SBT4", "SED_D50_UNIFORM", 2.0e-4)? as f32).max(1e-5);
    let dstar = (GRAV as f32 * (SED_SG as f32 - 1.0) / (NU_WATER as f32).powi(2)).powf(0.333333) * d50;
    let psic = 0.3 / (1.0 + 1.2 * dstar) + 0.055 * (1.0 - (-0.02 * dstar).exp());
    // element 0 is not a sea point: 0 there, as WW3 allocates and fills them
    let mut v = vec![d50 as f64; g.nsea + 1];
    v[0] = 0.0;
    rec_r(&mut p, "SED_D50", &[0], &[g.nsea + 1], &v);
    let mut v = vec![psic as f64; g.nsea + 1];
    v[0] = 0.0;
    rec_r(&mut p, "SED_PSIC", &[0], &[g.nsea + 1], &v);
    let (fw, delab) = tabu_fw();
    rec_r(&mut p, "FWTABLE", &[0], &[SIZEFWTABLE + 1], &fw);
    iscalar(&mut p, "SIZEFWTABLE", SIZEFWTABLE as i32);
    scalar(&mut p, "DELAB", delab);
    scalar(&mut p, "ABMIN", ABMIN);
    let (erf, delxerf) = tabu_erf();
    rec_r(&mut p, "ERFTABLE", &[0], &[SIZEERFTABLE + 1], &erf);
    scalar(&mut p, "DELXERF", delxerf);
    scalar(&mut p, "XERFMAX", XERFMAX);
    iscalar(&mut p, "SIZEERFTABLE", SIZEERFTABLE as i32);
    // stress tables (INSIN4): steps first, then the tables computed from the other records
    if sintable > 0.5 {
        let alpham = 20.0 * alpha0;
        let tabdel = [5.0 / IUSTAR as f64, alpham / IALPHA as f64, TAUWMAX / ITAUMAX as f64, UMAX / JUMAX as f64, alpham / ILEVTAIL as f64, TAUWMAX];
        rec_r(&mut p, "TABDEL", &[1], &[6], &tabdel);
        rec_r(&mut p, "TAUT", &[0, 0], &[ITAUMAX + 1, JUMAX + 1], &vec![0.0; (ITAUMAX + 1) * (JUMAX + 1)]);
        rec_r(&mut p, "TAUHFT", &[0, 0], &[IUSTAR + 1, IALPHA + 1], &vec![0.0; (IUSTAR + 1) * (IALPHA + 1)]);
        let n2 = (IUSTAR + 1) * (IALPHA + 1) * (ILEVTAIL + 1);
        if f("SIN4", "TAUWSHELTER", 0.3)? > 0.0 {
            rec_r(&mut p, "TAUHFT2", &[0, 0, 0], &[IUSTAR + 1, IALPHA + 1, ILEVTAIL + 1], &vec![0.0; n2]);
        }
        let st = St4::new(&p)?;
        let t = st.tables.as_ref().unwrap();
        let taut = tabu_stress(&st, t);
        // from the single-precision BBETA and SIG(NK), as WW3
        let (h1, _, h2, _) = tabu_tauhf2(&st, t, st.sig[st.nk], st.bbeta);
        p.set_real("TAUT", &taut)?;
        p.set_real("TAUHFT", &h1)?;
        if p.has("TAUHFT2") {
            p.set_real("TAUHFT2", &h2)?;
        }
    }
    Ok(p)
}

/// `INSNL1`: the DIA quadruplet geometry, interpolation weights and index
/// tables, as the records `NFR`, `NFRHGH`, `NFRCHG`, `NSPECX`, `NSPECY`,
/// `DAL1`..`DAL3`, `AF11`, `AWG`, `SWG`, `IP11`..`IM24`, `IC11`..`IC82`.
fn insnl1(p: &mut Params, nk: usize, nth: usize, xfr: f64, dth: f64, sig: &[f64], lam: f64) {
    let nspec = nk * nth;
    let lamm2 = (1.0 - lam).powi(2);
    let lamp2 = (1.0 + lam).powi(2);
    let delth3 = ((lamm2 * lamm2 + 4.0 - lamp2 * lamp2) / (4.0 * lamm2)).acos();
    let delth4 = (-delth3.sin() * lamm2 / lamp2).asin();
    let dal1 = 1.0 / (1.0 + lam).powi(4);
    let dal2 = 1.0 / (1.0 - lam).powi(4);
    let dal3 = 2.0 * dal1 * dal2;
    let cthp = (delth4 / dth).abs();
    let ithp = cthp as i64;
    let ithp1 = ithp + 1;
    let wthp = cthp - ithp as f64;
    let wthp1 = 1.0 - wthp;
    let cthm = (delth3 / dth).abs();
    let ithm = cthm as i64;
    let ithm1 = ithm + 1;
    let wthm = cthm - ithm as f64;
    let wthm1 = 1.0 - wthm;
    let xfrln = xfr.ln();
    let ifrp = ((1.0 + lam).ln() / xfrln) as i64;
    let ifrp1 = ifrp + 1;
    let wfrp = (1.0 + lam - xfr.powi(ifrp as i32)) / (xfr.powi(ifrp1 as i32) - xfr.powi(ifrp as i32));
    let wfrp1 = 1.0 - wfrp;
    let ifrm = ((1.0 - lam).ln() / xfrln) as i64;
    let ifrm1 = ifrm - 1;
    let wfrm = (xfr.powi(ifrm as i32) - (1.0 - lam)) / (xfr.powi(ifrm as i32) - xfr.powi(ifrm1 as i32));
    let wfrm1 = 1.0 - wfrm;
    let nfr = nk as i64;
    let nfrhgh = nfr + ifrp1 - ifrm1;
    let nfrchg = nfr - ifrm1;
    let nspecy = (nfrhgh * nth as i64) as usize;
    let nspecx = (nfrchg * nth as i64) as usize;
    for (n, v) in [("NFR", nfr), ("NFRHGH", nfrhgh), ("NFRCHG", nfrchg), ("NSPECX", nspecx as i64), ("NSPECY", nspecy as i64)] {
        iscalar(p, n, v as i32);
    }
    let fr = |ifr: i64| -> [i64; 8] {
        [ifr + ifrp, ifr + ifrp1, (ifr + ifrm).max(0), (ifr + ifrm1).max(0), (ifr - ifrp).max(0), (ifr - ifrp1).max(0), ifr - ifrm, ifr - ifrm1]
    };
    let th = |ith: i64| -> [i64; 8] {
        let n = nth as i64;
        let mut t = [ith + ithp, ith + ithp1, ith + ithm, ith + ithm1, ith - ithp, ith - ithp1, ith - ithm, ith - ithm1];
        for (k, v) in t.iter_mut().enumerate() {
            if k < 4 && *v > n {
                *v -= n;
            }
            if k >= 4 && *v < 1 {
                *v += n;
            }
        }
        t
    };
    let n = nth as i64;
    let idx = |itx: i64, ifx: i64| (itx + (ifx - 1) * n) as i32;
    let mut tabs: Vec<(&str, Vec<i32>)> = vec![];
    // IP / IM over NSPECX: (theta index, frequency index) pairs
    let pairs_x: [(&str, usize, usize); 16] = [
        ("IP11", 1, 1),
        ("IP12", 0, 1),
        ("IP13", 1, 0),
        ("IP14", 0, 0),
        ("IM11", 7, 3),
        ("IM12", 6, 3),
        ("IM13", 7, 2),
        ("IM14", 6, 2),
        ("IP21", 5, 1),
        ("IP22", 4, 1),
        ("IP23", 5, 0),
        ("IP24", 4, 0),
        ("IM21", 3, 3),
        ("IM22", 2, 3),
        ("IM23", 3, 2),
        ("IM24", 2, 2),
    ];
    for (name, it, iff) in pairs_x {
        let v = (1..=nspecx as i64)
            .map(|isp| {
                let ifr = 1 + (isp - 1) / n;
                let ith = 1 + (isp - 1) % n;
                idx(th(ith)[it], fr(ifr)[iff])
            })
            .collect();
        tabs.push((name, v));
    }
    let pairs_c: [(&str, usize, usize); 16] = [
        ("IC11", 5, 5),
        ("IC21", 4, 5),
        ("IC31", 5, 4),
        ("IC41", 4, 4),
        ("IC51", 3, 7),
        ("IC61", 2, 7),
        ("IC71", 3, 6),
        ("IC81", 2, 6),
        ("IC12", 1, 5),
        ("IC22", 0, 5),
        ("IC32", 1, 4),
        ("IC42", 0, 4),
        ("IC52", 7, 7),
        ("IC62", 6, 7),
        ("IC72", 7, 6),
        ("IC82", 6, 6),
    ];
    for (name, it, iff) in pairs_c {
        let v = (1..=nspec as i64)
            .map(|isp| {
                let ifr = 1 + (isp - 1) / n;
                let ith = 1 + (isp - 1) % n;
                idx(th(ith)[it], fr(ifr)[iff])
            })
            .collect();
        tabs.push((name, v));
    }
    for (name, v) in tabs {
        let len = v.len();
        rec_i(p, name, &[1], &[len], &v);
    }
    scalar(p, "DAL1", dal1);
    scalar(p, "DAL2", dal2);
    scalar(p, "DAL3", dal3);
    let tpiinv = 1.0 / TPI;
    let mut af11 = vec![0f64; nspecx];
    for ifr in 1..=nk {
        let a = (sig[ifr] * tpiinv).powi(11);
        for ith in 1..=nth {
            af11[ith - 1 + (ifr - 1) * nth] = a;
        }
    }
    let mut frq = sig[nk] * tpiinv;
    for ifr in nk + 1..=nfrchg as usize {
        frq *= xfr;
        for ith in 1..=nth {
            af11[ith - 1 + (ifr - 1) * nth] = frq.powi(11);
        }
    }
    rec_r(p, "AF11", &[1], &[nspecx], &af11);
    let awg = [wthp * wfrp, wthp1 * wfrp, wthp * wfrp1, wthp1 * wfrp1, wthm * wfrm, wthm1 * wfrm, wthm * wfrm1, wthm1 * wfrm1];
    rec_r(p, "AWG", &[1], &[8], &awg);
    rec_r(p, "SWG", &[1], &[8], &awg.map(|x| x * x));
}

/// Compare two parameter files record by record: for every record of
/// `reference`, the largest difference relative to its largest value (or
/// whether the integers are equal), and the records missing in `ours`.
pub fn compare(ours: &Params, reference: &Params) -> Vec<(String, String)> {
    let mut names: Vec<&String> = reference.map.keys().collect();
    names.sort();
    let mut out = vec![];
    for name in names {
        let r = &reference.map[name];
        let line = match ours.map.get(name) {
            None => "missing".to_string(),
            Some(o) if o.shape != r.shape || o.lbound != r.lbound => {
                format!("shape {:?} from {:?} against {:?} from {:?}", o.shape, o.lbound, r.shape, r.lbound)
            }
            Some(o) if !r.int.is_empty() => {
                let bad = o.int.iter().zip(&r.int).filter(|(a, b)| a != b).count();
                if bad == 0 { "equal".to_string() } else { format!("{bad} of {} integers differ", r.int.len()) }
            }
            Some(o) => {
                let m = r.real.iter().fold(0f64, |m, &v| m.max((v as f64).abs())).max(1e-30);
                let d = o.real.iter().zip(&r.real).fold(0f64, |e, (&a, &b)| e.max((a as f64 - b as f64).abs()));
                format!("{:.2e}", d / m)
            }
        };
        out.push((name.clone(), line));
    }
    out
}
