//! CPU transcriptions of the sea-ice source terms of WW3 (7.14, switches
//! `IC2` and `IS2`), in double precision, as references for the GPU kernels:
//!
//! - `INSIS2` ([`IceParams::from_namelists`]): the tables of the floe-size
//!   dependent scattering (`SIS2ALPHAS`, `SIS2ALPHA2`, `ICEDAVETAB`) and the
//!   eigen-decomposition of the directional scattering matrix
//!   (`IS2EIGVEC`, `IS2EIGVAL`);
//! - `W3SIS2` ([`sis2`]): scattering in the marginal ice zone, ratio `R` of
//!   the energy with ice bending to the energy without ice, floe-size update
//!   by break-up and creep (inelastic) or anelastic dissipation, with
//!   `W3FSD_DAVE` ([`fsd_dave`]) and the Robinson-Palmer dispersion
//!   `W3RPWNICE` / `FINDROOTS_NR` ([`rpwnice`]);
//! - `W3SIC2` ([`sic2`]): damping by the under-ice boundary layer, laminar
//!   to turbulent (Stopa et al. 2016), branch `IC2DISPER = F`;
//! - the ice part of `W3SRCE` applied after the dynamic integration loop
//!   when `IC_NUMERICS = F` ([`ice_after_loop`], section 10.1 of
//!   `w3srcemd.F90`): implicit attenuation, isotropic or matrix-exponential
//!   scattering, and the fluxes `PHICE` and `TAUICE`.
//!
//! Spectra are `A(ISPEC)` with `ISPEC = ITH + (IK-1) NTH` stored at
//! `ith + ik nth` (0-based), as in [`crate::sources`]. Per-frequency arrays
//! (`WN`, `CG`, `R`, ...) hold `IK = 1 .. NK` at `ik - 1`.
//!
//! The parameters are rounded to single precision when read, as WW3 holds
//! them (`REAL`); the arithmetic is then done in double precision. The
//! constants are kept as written in the Fortran (`YOUNG = 5.49E+9`,
//! `POISSON = 0.3`, ice density `DENS = 1025`, `VISCM = 1.83E-6`, `GRAV =
//! 9.806`, `DWAT = 1000`), and so are the oddities of the code, each marked
//! by a comment (for instance the dissipation of `W3SIS2` added inside the
//! frequency loop, the integer division `4*4/3 = 5` of the anelastic
//! dissipation, the 520 values reshaped into the 500 of `SIS1ALPHATABLE2`).
//!
//! Not ported (an error is returned): `IC2DISPER = T` (`IC2PARS(1) > 0.5`,
//! dispersion of Liu et al. with ice forcing parameters 1 and 2),
//! `ICEDISP = T` (`LIU_FORWARD_DISPERSION` in `W3SRCE`) and
//! `ICNUMERICS = T` (ice terms inside the dynamic loop).

use crate::spec::Spec;
use crate::ww3grid::Namelists;
use anyhow::{bail, Result};

/// Acceleration of gravity (`GRAV` of `constants.F90`).
pub const GRAV: f64 = 9.806;
/// Density of water (`DWAT` of `constants.F90`).
pub const DWAT: f64 = 1000.0;
/// `PI` of `constants.F90`.
const PI: f64 = 3.141592653589793;
/// `TPI` of `constants.F90`.
const TPI: f64 = 2.0 * PI;
/// `TPIINV` of `constants.F90`.
const TPIINV: f64 = 1.0 / TPI;
/// Young modulus of the ice (`YOUNG` of `W3SIS2`, Pa).
const YOUNG: f64 = 5.49e9;
/// Poisson ratio of the ice (`POISSON`).
const POISSON: f64 = 0.3;
/// Density used for the ice in `W3SIS2` (`DENS`, kg/m³, as written).
const DENS: f64 = 1025.0;
/// Density of the ice in `W3RPWNICE` (`DENS_ICE`).
const DENS_ICE: f64 = 922.5;
/// Molecular viscosity of water at freezing (`VISCM` of `W3SIC2`).
const VISCM: f64 = 1.83e-6;
/// Number of ice thicknesses of the scattering tables (`NTHICK`).
pub const NTHICK: usize = 20;
/// Number of maximum floe diameters of `ICEDAVETAB` (`NICED`, 1 m to 500 m).
pub const NICED: usize = 500;
/// Number of frequencies of the scattering tables (`NFTAB`).
const NFTAB: usize = 25;
/// First thickness and thickness step of the tables (`THICK1`, `DTHICK`);
/// kept at 0.1 and 0.25 also for the WIM1 table that starts at 0.25 m, as
/// in WW3.
const THICK1: f64 = 0.1;
/// See [`THICK1`].
const DTHICK: f64 = 0.25;
/// `SIZEFWTABLE` of `constants.F90`.
const SIZEFWTABLE: usize = 300;
/// `ABMIN` of `constants.F90`.
const ABMIN: f64 = -1.0;

/// Parameters and tables of the ice source terms (`IS2PARS`, `IC2PARS`,
/// the `&MISC` ice switches and the tables of `INSIS2`).
#[derive(Clone, Debug)]
pub struct IceParams {
    /// Number of frequencies.
    pub nk: usize,
    /// Number of directions.
    pub nth: usize,
    /// Frequency increment factor (`XFR`).
    pub xfr: f64,
    /// `SIG(1..NK)` at `ik - 1` (rad/s).
    pub sig: Vec<f64>,
    /// `DDEN(1..NK)` at `ik - 1`.
    pub dden: Vec<f64>,
    /// `ECOS(1..NTH)` at `ith - 1`.
    pub ecos: Vec<f64>,
    /// `ESIN(1..NTH)` at `ith - 1`.
    pub esin: Vec<f64>,
    /// `IS2PARS(1..24)` at `i - 1` (`&SIS2`, see `W3GRID`): 1 `ISC1`,
    /// 2 `IS2BACKSCAT`, 3 `IS2BREAK`, 4 `IS2C2`, 5 `IS2C3`, 6 `IS2DISP`,
    /// 7 `IS2DAMP`, 8 `IS2FRAGILITY`, 9 `IS2DMIN`, 10 `IS2DUPDATE`,
    /// 11 `IS2CONC`, 12 `|IS2CREEPB|`, 13 `IS2CREEPC`, 14 `IS2CREEPD`,
    /// 15 `IS2CREEPN`, 16 `IS2BREAKE`, 17 `IS2BREAKF`, 18 `IS2WIM1`,
    /// 19 `IS2FLEXSTR`, 20 `IS2ISOSCAT`, 21 `IS2ANDISD`, 22 `IS2ANDISN`,
    /// 23 `IS2ANDISB`, 24 `IS2ANDISE` (logicals as 0 or 1).
    pub is2pars: [f64; 24],
    /// `IC2PARS(1..8)` at `i - 1` (`&SIC2`): 1 `IC2DISPER`, 2 `IC2TURB`,
    /// 3 `IC2ROUGH`, 4 `IC2REYNOLDS`, 5 `IC2SMOOTH`, 6 `IC2VISC`,
    /// 7 `IC2TURBS`, 8 `IC2DMAX`.
    pub ic2pars: [f64; 8],
    /// `IICEHMIN` (`&MISC ICEHMIN`): minimum thickness in the strain.
    pub iicehmin: f64,
    /// `IICEHINIT` (`&MISC ICEHINIT`): ice thickness without forcing.
    pub iicehinit: f64,
    /// `IICEDISP` (`&MISC ICEDISP`): ice dispersion in `W3SRCE`.
    pub iicedisp: bool,
    /// `IICESMOOTH`: never set by `ww3_grid` 7.14 nor written in the model
    /// definition file; taken as false (the value of the zeroed memory).
    pub iicesmooth: bool,
    /// `IC_NUMERICS` (`&MISC ICNUMERICS`).
    pub ic_numerics: bool,
    /// Whether the switch `IS2` is compiled (true in the HWW3 build).
    pub with_is2: bool,
    /// Whether the switch `IC2` is compiled (true in the HWW3 build).
    pub with_ic2: bool,
    /// `SIS2ALPHAS(J, IK)` at `j - 1 + (ik - 1) NTHICK`: scattering
    /// coefficient of a floe of unit diameter, by thickness and frequency.
    pub sis2alphas: Vec<f64>,
    /// `SIS2ALPHA2(J, IK)`, same layout (filled by `INSIS2`, not used by
    /// `W3SIS2`).
    pub sis2alpha2: Vec<f64>,
    /// `ICEDAVETAB(1..NICED)` at `i - 1`: mean floe diameter for a maximum
    /// diameter of `i` m.
    pub icedavetab: Vec<f64>,
    /// `ICEDMIN` of the module before any call (an uninitialised module
    /// variable in WW3, taken as 0); `W3SIS2` sets it to `IS2DMIN` when
    /// positive.
    pub icedmin: f64,
    /// `IS2EIGVEC(I, J)` at `(i - 1) NTH + (j - 1)`: column `j` is the
    /// eigenvector of eigenvalue `IS2EIGVAL(J)` of the scattering matrix.
    pub is2eigvec: Vec<f64>,
    /// `IS2EIGVAL(1..NTH)` at `j - 1`, clipped at 0 as `INSIS2` does.
    pub is2eigval: Vec<f64>,
    /// `FWTABLE(0..SIZEFWTABLE)`: friction factor table (`TABU_FW`).
    pub fwtable: Vec<f64>,
    /// `DELAB` of the friction factor table.
    pub delab: f64,
    /// `ABMIN` of the friction factor table.
    pub abmin: f64,
}

/// Round to single precision, as WW3 stores a `REAL` parameter.
fn r32(x: f64) -> f64 {
    x as f32 as f64
}

impl IceParams {
    /// Parameters of the ice source terms from the namelists `&SIS2`,
    /// `&SIC2` and `&MISC` of `ww3_grid` (WW3 defaults otherwise, see
    /// `W3GRID` sections 6.n and the `IS2PARS`/`IC2PARS` assignments), and
    /// the tables of `INSIS2` on the spectral grid `sp`.
    pub fn from_namelists(nml: &Namelists, sp: &Spec) -> Result<IceParams> {
        let f = |g: &str, k: &str, d: f64| nml.f(g, k, d).map(r32);
        let b = |g: &str, k: &str, d: bool| nml.b(g, k, d).map(|v| if v { 1.0 } else { 0.0 });
        // &SIS2 (W3GRID defaults under W3_IS2)
        let mut ip = [0f64; 24];
        ip[0] = f("SIS2", "ISC1", 1.0)?;
        ip[1] = f("SIS2", "IS2BACKSCAT", 1.0)?;
        ip[2] = b("SIS2", "IS2BREAK", false)?;
        ip[3] = f("SIS2", "IS2C2", 0.0)?;
        ip[4] = f("SIS2", "IS2C3", 0.0)?;
        ip[5] = b("SIS2", "IS2DISP", false)?;
        ip[6] = f("SIS2", "IS2DAMP", 0.0)?;
        ip[7] = f("SIS2", "IS2FRAGILITY", 0.9)?;
        ip[8] = f("SIS2", "IS2DMIN", 20.0)?;
        ip[9] = b("SIS2", "IS2DUPDATE", true)?;
        ip[10] = f("SIS2", "IS2CONC", 0.0)?;
        ip[11] = f("SIS2", "IS2CREEPB", 0.0)?.abs();
        ip[12] = f("SIS2", "IS2CREEPC", 0.4)?;
        ip[13] = f("SIS2", "IS2CREEPD", 0.5)?;
        ip[14] = f("SIS2", "IS2CREEPN", 3.0)?;
        ip[15] = f("SIS2", "IS2BREAKE", 1.0)?;
        ip[16] = f("SIS2", "IS2BREAKF", 3.6)?;
        ip[17] = f("SIS2", "IS2WIM1", 1.0)?;
        ip[18] = f("SIS2", "IS2FLEXSTR", 6.0e5)?;
        ip[19] = b("SIS2", "IS2ISOSCAT", true)?;
        ip[20] = f("SIS2", "IS2ANDISD", 2.0e-9)?;
        ip[21] = f("SIS2", "IS2ANDISN", 1.0)?;
        ip[22] = b("SIS2", "IS2ANDISB", true)?;
        ip[23] = f("SIS2", "IS2ANDISE", 0.55)?;
        // &SIC2 (W3GRID defaults under W3_IC2)
        let mut cp = [0f64; 8];
        cp[0] = b("SIC2", "IC2DISPER", false)?;
        cp[1] = f("SIC2", "IC2TURB", 1.0)?;
        cp[2] = f("SIC2", "IC2ROUGH", 0.01)?;
        cp[3] = f("SIC2", "IC2REYNOLDS", 1.5e5)?;
        cp[4] = f("SIC2", "IC2SMOOTH", 2.0e5)?;
        cp[5] = f("SIC2", "IC2VISC", 1.0)?;
        cp[6] = f("SIC2", "IC2TURBS", 0.0)?;
        cp[7] = f("SIC2", "IC2DMAX", 0.0)?;
        let (fwtable, delab) = crate::ww3grid::tabu_fw();
        let mut p = IceParams {
            nk: sp.nk,
            nth: sp.nth,
            xfr: sp.xfr as f64,
            sig: (1..=sp.nk).map(|ik| sp.sig[ik] as f64).collect(),
            dden: sp.dden.iter().map(|&x| x as f64).collect(),
            ecos: sp.ecos.iter().map(|&x| x as f64).collect(),
            esin: sp.esin.iter().map(|&x| x as f64).collect(),
            is2pars: ip,
            ic2pars: cp,
            iicehmin: f("MISC", "ICEHMIN", 0.2)?,
            iicehinit: f("MISC", "ICEHINIT", 0.5)?,
            iicedisp: nml.b("MISC", "ICEDISP", false)?,
            iicesmooth: false,
            ic_numerics: nml.b("MISC", "ICNUMERICS", false)?,
            with_is2: true,
            with_ic2: true,
            sis2alphas: Vec::new(),
            sis2alpha2: Vec::new(),
            icedavetab: Vec::new(),
            icedmin: 0.0,
            is2eigvec: Vec::new(),
            is2eigval: Vec::new(),
            fwtable,
            delab,
            abmin: ABMIN,
        };
        let ec2: Vec<f64> = sp.ec2.iter().map(|&x| x as f64).collect();
        let es2: Vec<f64> = sp.es2.iter().map(|&x| x as f64).collect();
        insis2(&mut p, &ec2, &es2);
        Ok(p)
    }

    /// `IS2PARS(i)` with the Fortran index `i` (1-based).
    #[inline]
    fn is2(&self, i: usize) -> f64 {
        self.is2pars[i - 1]
    }

    /// `IC2PARS(i)` with the Fortran index `i` (1-based).
    #[inline]
    fn ic2(&self, i: usize) -> f64 {
        self.ic2pars[i - 1]
    }
}

/// Interpolation of a `(NTHICK, NFTAB)` table at the frequencies of the
/// grid, as the loops of `INSIS2` (linear in frequency, constant outside the
/// table).
fn interp_table(sig: &[f64], tab: &[f64]) -> Vec<f64> {
    let nk = sig.len();
    let mut out = vec![0f64; NTHICK * nk];
    for i in 0..nk {
        let fr = sig[i] * TPIINV;
        for j in 0..NTHICK {
            let v = if fr < SIS1FTABLE[0] {
                tab[j]
            } else if fr > SIS1FTABLE[NFTAB - 1] {
                tab[j + (NFTAB - 1) * NTHICK]
            } else {
                let mut ind = 0;
                for k in 0..NFTAB - 1 {
                    if SIS1FTABLE[k] < fr {
                        ind = k;
                    }
                }
                let x = (fr - SIS1FTABLE[ind]) / (SIS1FTABLE[ind + 1] - SIS1FTABLE[ind]);
                tab[j + ind * NTHICK] * (1.0 - x) + tab[j + (ind + 1) * NTHICK] * x
            };
            out[j + i * NTHICK] = v;
        }
    }
    out
}

/// `INSIS2`: the tables of `W3SIS2` and the diagonalisation of the
/// scattering matrix (`EC2`, `ES2` of the spectral grid by direction).
fn insis2(p: &mut IceParams, ec2: &[f64], es2: &[f64]) {
    let nth = p.nth;
    // 1. reflection by frequency and thickness
    let tab: &[f64] = if p.is2(18) < 0.5 { &SIS1ALPHATABLE } else { &SIS1ALPHATABLE_WIM };
    p.sis2alphas = interp_table(&p.sig, tab);
    // RESHAPE keeps the first NTHICK*NFTAB = 500 of the 520 values written
    p.sis2alpha2 = interp_table(&p.sig, &SIS1ALPHATABLE2[..NTHICK * NFTAB]);
    // 2. ICEDMAX to ICEDAVE
    p.icedavetab = (1..=NICED).map(|i| fsd_dave(p.is2(9), i as f64, p.is2(8))).collect();
    // 3. scattering matrix ("example that looks like figure 12 in Masson
    //    & LeBlond") and its eigen-decomposition
    let nthf = nth as f64;
    let mut m = vec![0f64; nth * nth];
    for i in 0..nth {
        for j in 0..nth {
            let k = i.abs_diff(j);
            let mut v = -1.0 / nthf * (2.0 * ec2[k] * ec2[k] + 0.8 * p.ecos[k].powi(3));
            // ECOS < 0.001 is also true for every negative cosine (as written)
            if p.ecos[k] < 0.001 {
                v -= 1.0 / nthf * 0.8 * es2[k];
            }
            m[i * nth + j] = v;
        }
        // removes the row sum from the diagonal for energy conservation
        let s: f64 = m[i * nth..(i + 1) * nth].iter().sum();
        m[i * nth + i] -= s;
    }
    let (val, vec) = jacobi(&m, nth);
    p.is2eigval = val.iter().map(|&v| v.max(0.0)).collect();
    p.is2eigvec = vec;
}

/// Eigenvalues and eigenvectors of the symmetric matrix `a` (`n × n`,
/// row-major) by cyclic Jacobi rotations (`DIAGONALIZE` of `W3SERVMD` is the
/// Jacobi method of Numerical Recipes). Returns the eigenvalues and the
/// matrix `V` (row-major) whose column `j` is the eigenvector `j`. The order
/// of the eigenpairs does not matter to the matrix exponential of
/// [`ice_after_loop`].
fn jacobi(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a.to_vec();
    let mut v = vec![0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        let mut diag = 0.0;
        for p in 0..n {
            diag += a[p * n + p].abs();
            for q in p + 1..n {
                off += a[p * n + q].abs();
            }
        }
        if off <= 1e-300 || off <= 1e-17 * diag {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = 0.5 * (aqq - app) / apq;
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

/// `W3FSD_DAVE`: mean floe diameter of a power-law floe-size distribution
/// `P(x) = x^(-1-γ)` between `icedmin` and `icedmax`, with
/// `γ = 2 + ln(fragility) / ln 2`.
pub fn fsd_dave(icedmin: f64, icedmax: f64, fragility: f64) -> f64 {
    let icedmaxl = (icedmin * 1.01).max(icedmax);
    let gam = 2.0 + fragility.ln() / 2f64.ln();
    if icedmin == 0.0 {
        icedmaxl
    } else {
        gam / (gam - 1.0)
            * ((icedmaxl.powf(-gam + 1.0) - icedmin.powf(-gam + 1.0)) / (icedmaxl.powf(-gam) - icedmin.powf(-gam)))
    }
}

/// Minimal complex number for the Robinson-Palmer dispersion relation.
#[derive(Clone, Copy, Debug)]
struct Cx {
    re: f64,
    im: f64,
}

impl Cx {
    fn new(re: f64, im: f64) -> Cx {
        Cx { re, im }
    }
    fn add(self, o: Cx) -> Cx {
        Cx::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: Cx) -> Cx {
        Cx::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: Cx) -> Cx {
        Cx::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    fn scale(self, s: f64) -> Cx {
        Cx::new(self.re * s, self.im * s)
    }
    fn div(self, o: Cx) -> Cx {
        let d = o.re * o.re + o.im * o.im;
        Cx::new((self.re * o.re + self.im * o.im) / d, (self.im * o.re - self.re * o.im) / d)
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

/// `FUNCD_FVAL` and `FUNCD_FDERIV`: the Robinson-Palmer dispersion function
/// `C5 k^5 + C1 k - σ²/g` and its derivative.
fn funcd(x: Cx, visc_rp: f64, iceh: f64, sigma: f64) -> (Cx, Cx) {
    let flex_rigid = YOUNG * iceh.powi(3) / (12.0 * (1.0 - POISSON * POISSON));
    let draft = (DENS_ICE / DENS) * iceh;
    let dens_grav = DENS * GRAV;
    let alp = sigma * sigma / GRAV;
    let gamma = sigma * visc_rp / dens_grav;
    let c1 = Cx::new(1.0 - alp * draft, -gamma);
    let c5 = flex_rigid / dens_grav;
    let x2 = x.mul(x);
    let x4 = x2.mul(x2);
    let x5 = x4.mul(x);
    let fval = x5.scale(c5).add(c1.mul(x)).sub(Cx::new(alp, 0.0));
    let fderiv = x4.scale(5.0 * c5).add(c1);
    (fval, fderiv)
}

/// `FINDROOTS_NR`: Newton iterations from `guess`. As in WW3 the loop stops
/// after `MAXIT - 1` iterations without an error when it does not converge
/// (the test `J > MAXIT` of the Fortran is never true).
fn findroots_nr(guess: Cx, visc_rp: f64, iceh: f64, sigma: f64) -> Cx {
    const MAXIT: usize = 300;
    const TOL: f64 = 1e-9;
    let mut x0 = guess;
    let mut x = guess;
    let mut j = 1;
    while j < MAXIT {
        let (fval, fderiv) = funcd(x0, visc_rp, iceh, sigma);
        let dx = fval.div(fderiv);
        x = x0.sub(dx);
        x0 = x;
        if dx.abs() > TOL || fval.abs() > 1.0 {
            j += 1;
        } else {
            break;
        }
    }
    x
}

/// `W3RPWNICE`: wavenumber `WN_I`, damping (imaginary part of the
/// wavenumber with the artificial viscosity `VISC_RP = 10`) and group speed
/// `CG_I` of the Robinson-Palmer dispersion relation in unbroken ice of
/// thickness `iceh`, for `IK = 1 .. NK` (used when `IS2DISP = T`).
pub fn rpwnice(p: &IceParams, iceh: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    const VISC_RP: f64 = 10.0;
    let nk = p.nk;
    let mut wn_i = vec![0f64; nk];
    let mut damping = vec![0f64; nk];
    let mut cg_i = vec![0f64; nk];
    let flex_rigid = YOUNG * iceh.powi(3) / (12.0 * (1.0 - POISSON * POISSON));
    let mut gs1 = Cx::new(p.sig[0] * p.sig[0] / GRAV, 0.0);
    for ik in 0..nk {
        let s = p.sig[ik];
        let root = findroots_nr(gs1, 0.0, iceh, s);
        wn_i[ik] = root.re;
        let root = findroots_nr(gs1, VISC_RP, iceh, s);
        damping[ik] = root.im;
        gs1 = Cx::new(wn_i[ik], 0.0);
        cg_i[ik] = (5.0 * flex_rigid * wn_i[ik].powi(4) + DENS * GRAV - DENS_ICE * iceh * s * s)
            / (2.0 * s * (DENS + DENS_ICE * wn_i[ik] * iceh));
    }
    (wn_i, damping, cg_i)
}

/// Outputs of [`sis2`].
#[derive(Clone, Debug)]
pub struct Sis2Out {
    /// Source term `S` (action), scattering with back-scatter and dissipation.
    pub s: Vec<f64>,
    /// Diagonal part of the scattering `D` (`VDIR` in `W3SRCE`).
    pub d: Vec<f64>,
    /// Diagonal dissipation `DISSIP` (`VDIR2` in `W3SRCE`).
    pub dissip: Vec<f64>,
    /// `R(IK)`: ratio of the energy (with ice bending) to the wave energy
    /// without ice. Without ice (`CICE <= 0`) the Fortran leaves it
    /// undefined; `W3SRCE` sets it to 1 before the call, which is returned.
    pub r: Vec<f64>,
    /// Updated maximum floe diameter `ICEF`.
    pub icef: f64,
}

/// `W3SIS2`: floe-size dependent scattering in the marginal ice zone, ice
/// break-up and creep or anelastic dissipation, for the action spectrum `a`,
/// ice concentration `cice`, thickness `iceh`, maximum floe diameter `icef`
/// (carried from step to step) and its forcing `icedmax`, with the open
/// water `wn`, `cg` and the in-ice `wn_r`, `cg_ice` (`IK = 1 .. NK` at
/// `ik - 1`). `depth` is unused, as in WW3.
#[allow(clippy::too_many_arguments)]
pub fn sis2(p: &IceParams, a: &[f64], _depth: f64, cice: f64, iceh: f64, icef: f64, icedmax: f64, wn: &[f64], cg: &[f64], wn_r: &[f64], cg_ice: &[f64]) -> Sis2Out {
    let (nk, nth) = (p.nk, p.nth);
    let ns = nk * nth;
    let mut s = vec![0f64; ns];
    let mut d = vec![0f64; ns];
    let mut dissip = vec![0f64; ns];
    let mut r = vec![1f64; nk];
    let mut icef = icef;
    // 0. initialisations
    let sigma_c = p.is2(19);
    let mp2 = 1.0 - POISSON * POISSON;
    // variables of Cole et al. (1995)
    let alpha_d = 0.54;
    let b_cole = 1.205e-9 * (p.is2(24) * 1.60218e-19 / (1.38064852e-23 * 268.15)).exp();
    let tau_d = b_cole / 0.07;
    let s_d: Vec<f64> = p.sig.iter().map(|&sg| (sg * tau_d).ln()).collect();
    let delta_d = p.is2(21);
    let mut icedmin = p.icedmin;
    if p.is2(9) > 0.0 {
        icedmin = p.is2(9);
    }
    let creepfac = if p.is2(12) > 0.0 {
        let b = p.is2(12);
        let n = p.is2(15);
        // 2 is the ratio of <H/2^4> to (Hrms/2)^4 for a Rayleigh distribution,
        // 0.375 the average of cos^4, 0.4 is 2/5
        -2.0 * (0.25 / (n + 2.0)) * 0.375 * b * iceh.powf(n + 2.0) * (YOUNG / (2.0 * b * mp2)).powf(n + 1.0) / (DWAT * GRAV)
    } else {
        1.0
    };
    // WLG = TPI/WN only enters WSQ, which W3SIS2 does not use
    let _ = wn;
    let mut dmax = icef;
    let mut brk_crit = false;
    let nsum = (0.3f32 / (p.xfr as f32 - 1.0)).round().max(0.0) as usize;
    let strain_c = sigma_c * mp2 / YOUNG;
    // minimum floe size that can break
    let d_flex_fail = 0.5 * ((PI.powi(4) * YOUNG * iceh.powi(3)) / (48.0 * DENS * GRAV * mp2)).powf(0.25);
    // mean floe diameter from the maximum floe diameter
    let icedave: Vec<f64> = if dmax > NICED as f64 {
        vec![dmax; nk]
    } else if p.is2(9) > 0.0 {
        let i = (dmax.round() as i64).max(1) as usize;
        vec![p.icedavetab[i - 1]; nk]
    } else {
        (0..nk)
            .map(|ik| {
                if p.is2(14) * (TPI / wn_r[ik]) < dmax {
                    fsd_dave((TPI / wn_r[ik]) * p.is2(14), dmax, p.is2(8))
                } else {
                    dmax
                }
            })
            .collect()
    };
    if cice > 0.0 {
        // 1. wavelength in ice (Robinson-Palmer if IS2DISP)
        let (wn_i, wn_rp, cg_i) = if p.is2(6) > 0.5 { rpwnice(p, iceh) } else { (wn_r.to_vec(), vec![0f64; nk], cg_ice.to_vec()) };
        let wlg_i: Vec<f64> = wn_i.iter().map(|&k| TPI / k).collect();
        let cgratio: Vec<f64> = if p.is2(16) > 0.5 { (0..nk).map(|ik| cg_i[ik] / cg[ik]).collect() } else { vec![1.0; nk] };
        // 2. reflection coefficient from the table (1-based indices as the Fortran)
        let xi = (iceh - THICK1) / DTHICK;
        let mut ind1 = 1 + xi.floor() as i64;
        let mut ind2 = ind1 + 1;
        let mut w = xi - (ind1 - 1) as f64;
        if ind1 < 1 {
            ind1 = 1;
            w = 0.0;
        } else if ind2 > NTHICK as i64 {
            ind2 = NTHICK as i64;
            ind1 = NTHICK as i64;
            w = 0.0;
        }
        // IND2 is left below 1 by the Fortran for ICEH < -0.15 m (weight 0):
        // clamped here to stay in the table
        let i1 = (ind1 - 1) as usize;
        let i2 = (ind2.max(1) - 1) as usize;
        let mut curvspec = vec![0f64; nk];
        let curvtostrain = 0.25 * iceh.max(p.iicehmin).powi(2);
        for ik in 0..nk {
            let cg_eff = cice * cg_i[ik] + (1.0 - cice) * cg[ik];
            let al = &p.sis2alphas;
            let mut alpha = -1.0 * ((al[i1 + ik * NTHICK] * (1.0 - w) + al[i2 + ik * NTHICK] * w) / icedave[ik] + 2.0 * p.is2(7) * wn_rp[ik]) * cice;
            // additional scattering for pack ice (Squire et al. GRL 2009)
            alpha = p.is2(1) * alpha - 2.0 * p.is2(4) * (-1.0 * p.is2(5) / p.sig[ik]).exp();
            if p.is2(11) > 0.0 {
                if cice < 0.2 {
                    alpha = 0.0;
                }
                if cice > 0.8 {
                    alpha = 0.0;
                }
                if (0.2..=0.8).contains(&cice) {
                    alpha *= (cice - 0.2) * (0.8 - cice);
                }
            }
            // time decay
            let beta = alpha * cice * cg_eff;
            // 3. attenuation by scattering, back-scatter for energy conservation
            let mut sumalldir = 0.0;
            let mut suma = 0.0;
            for ith in 0..nth {
                let is = ith + ik * nth;
                d[is] = beta;
                s[is] = beta * a[is];
                sumalldir += s[is];
                suma += a[is];
            }
            // R: Wadhams 1973 eq. 34 (his thickness is 2h); uses DMAX = ICEF
            // even when IS2DUPDATE = F
            let smoothd = if p.iicesmooth {
                if p.is2(14) * wlg_i[ik] < dmax {
                    ((dmax - p.is2(14) * wlg_i[ik]) / (dmax * p.is2(13))).tanh()
                } else {
                    0.0
                }
            } else {
                1.0
            };
            r[ik] = 1.0 + p.is2(16) * smoothd * 4.0 * YOUNG * iceh.powi(3) * (PI / wlg_i[ik]).powi(4) / (3.0 * DWAT * GRAV * mp2);
            // action to surface elevation variance (m^2)
            let sume = suma * p.dden[ik] / cg[ik] / (r[ik] * cgratio[ik]);
            // curvature variance = elevation variance * k^4
            curvspec[ik] = sume * (2.0 * PI / wlg_i[ik]).powi(4);
            sumalldir /= nth as f64;
            // isotropic redistribution of the scattered energy
            for ith in 0..nth {
                s[ith + ik * nth] -= sumalldir * p.is2(2);
            }
        }
        // 4. update of the floe size
        if p.is2(10) < 0.5 {
            // resets the maximum floe size to the forcing or initial value
            dmax = icedmax;
            icef = icedmax;
        }
        let curv: Vec<f64> = (0..nk).map(|ik| curvspec[ik.saturating_sub(nsum)..=(ik + nsum).min(nk - 1)].iter().sum()).collect();
        let mut brk_crit_w = vec![false; nk];
        if p.is2(3) > 0.5 {
            let mut ikbreak = 0usize;
            for ik in 0..nk {
                let strain = curv[ik] * curvtostrain;
                if d_flex_fail < dmax {
                    // Williams et al. used IS2BREAKF = sqrt(2), the default is 3.6
                    let wamp = p.is2(17) * strain.sqrt();
                    if p.is2(9) == 0.0 {
                        icedmin = (TPI / wn_r[ik]) * p.is2(14);
                    }
                    let half = wlg_i[ik] / 2.0;
                    brk_crit_w[ik] = wamp > strain_c && half > icedmin && half < dmax && half > d_flex_fail;
                    if brk_crit_w[ik] {
                        ikbreak = ik;
                        brk_crit = true;
                    }
                }
            }
            // 4.b correction of the bias of the finite bandwidth sum (bounds
            // fixed at the entry of the loop, IKBREAK updated inside)
            if brk_crit {
                let lo = ikbreak.saturating_sub(nsum);
                let hi = ikbreak;
                for ik in lo..=hi {
                    if curvspec[ik] >= curvspec[ikbreak] && dmax >= wlg_i[ik] / 2.0 {
                        ikbreak = ik;
                    }
                }
                dmax = wlg_i[ikbreak] / 2.0;
                // weighting by CURVSPEC for a continuous shift of DMAX
                let ikp1 = (ikbreak + 1).min(nk - 1);
                let ikm1 = ikbreak.saturating_sub(1);
                if brk_crit_w[ikp1] && brk_crit_w[ikm1] {
                    dmax = (wlg_i[ikbreak] * curvspec[ikbreak] + wlg_i[ikp1] * curvspec[ikp1] + wlg_i[ikm1] * curvspec[ikm1])
                        / (2.0 * (curvspec[ikbreak] + curvspec[ikm1] + curvspec[ikp1]));
                }
                icef = dmax;
            }
        }
        // 5. inelastic or anelastic dissipation
        if p.is2(12) > 0.0 {
            for ik in 0..nk {
                if p.is2(14) * wlg_i[ik] < dmax {
                    let smoothd = ((dmax - p.is2(14) * wlg_i[ik]) / (dmax * p.is2(13))).tanh();
                    let v = if p.is2(23) <= 0.5 {
                        // inelastic (creep)
                        creepfac * 4.0 * curv[ik] * ((2.0 * PI) / wlg_i[ik]).powf(p.is2(15) + 1.0) / (cgratio[ik] * r[ik] * r[ik]) * smoothd
                    } else {
                        // anelastic: the Fortran `-4*4/3*SIG(IK)` is the integer
                        // division 16/3 = 5 (kept as written)
                        -5.0 * p.sig[ik] * delta_d * alpha_d * wn_i[ik].powi(4) * (YOUNG / mp2).powi(2) * (iceh / 2.0).powi(3) / 3.0 * 1.0
                            / ((alpha_d * s_d[ik]).exp() + (-alpha_d * s_d[ik]).exp())
                            * smoothd
                            / (r[ik] * r[ik] * cgratio[ik])
                            / (DWAT * GRAV)
                            * TPIINV
                    };
                    for ith in 0..nth {
                        dissip[ith + ik * nth] = v;
                    }
                }
                // As in WW3, `S = S + DISSIP*CICE*A` sits inside the IK loop:
                // the whole spectrum is updated at every IK, so the
                // dissipation of band IK is added NK - IK + 1 times.
                for is in 0..ns {
                    s[is] += dissip[is] * cice * a[is];
                }
            }
        }
    } else {
        // 6. no scattering nor dissipation
        icef = 0.0;
    }
    Sis2Out { s, d, dissip, r, icef }
}

/// `W3SIC2`: damping of the waves by the boundary layer under the ice,
/// laminar or turbulent (`IC2DISPER = F`), for the action spectrum `a`, the
/// ice thickness `iceh`, the floe diameter `icef` (used only with
/// `IC2DMAX > 0`), the open-water `cg`, `wn`, the in-ice `wn_r`, `cg_ice`
/// and the ratio `r` of [`sis2`]. `icei` is the ice concentration field
/// `ICEI` when it is read (`INFLAGS2(4)`), `southern_rlg` whether the node
/// has a negative latitude on a rectilinear lat-lon grid (for `IC2TURBS`).
/// Returns `(S, D)`. `depth` and `wn` are unused, as in WW3.
#[allow(clippy::too_many_arguments)]
pub fn sic2(
    p: &IceParams,
    a: &[f64],
    _depth: f64,
    iceh: f64,
    icef: f64,
    cg: &[f64],
    _wn: &[f64],
    wn_r: &[f64],
    cg_ice: &[f64],
    r: &[f64],
    icei: Option<f64>,
    southern_rlg: bool,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let (nk, nth) = (p.nk, p.nth);
    let ns = nk * nth;
    let mut d = vec![0f64; ns];
    // ICECOEF1 = ICEH (forced or not)
    let icecoef1 = iceh;
    let noice = icecoef1 == 0.0 || icei == Some(0.0);
    if !noice {
        if p.ic2(1) > 0.5 {
            bail!("W3SIC2 with IC2DISPER = T (Liu et al. dispersion, ice parameters 1 and 2) is not ported");
        }
        let mut smooth_dmax = vec![1f64; nk];
        let (pturb, pvisc, dturb);
        if p.ic2(2) > 0.0 {
            let mut uorb = 0.0;
            let mut aorb = 0.0;
            let mut fturb = p.ic2(2);
            // special treatment in the southern ocean
            if p.ic2(7) > 0.0 && southern_rlg {
                fturb = p.ic2(7);
            }
            for ik in 0..nk {
                let eb: f64 = a[ik * nth..(ik + 1) * nth].iter().sum();
                // W3_IS2: relative motion reduced for floes smaller than
                // IC2DMAX wavelengths
                if p.ic2(8) > 0.0 {
                    let wlg_r = TPI / wn_r[ik];
                    smooth_dmax[ik] = (0.5 * (1.0 + ((icef - p.ic2(8) * wlg_r) / (icef * 0.5)).tanh())).powi(2);
                }
                let sg = p.sig[ik];
                if r[ik] > 1.0 {
                    uorb += eb * smooth_dmax[ik] * sg * sg * p.dden[ik] / cg[ik] / (r[ik] * cg_ice[ik] / cg[ik]);
                    aorb += eb * smooth_dmax[ik] * p.dden[ik] / cg[ik] / (r[ik] * cg_ice[ik] / cg[ik]);
                } else {
                    uorb += eb * smooth_dmax[ik] * sg * sg * p.dden[ik] / cg[ik];
                    aorb += eb * smooth_dmax[ik] * p.dden[ik] / cg[ik];
                }
            }
            // significant amplitudes
            let aorb = 2.0 * aorb.sqrt();
            let uorb = 2.0 * uorb.sqrt();
            let re = uorb * aorb / VISCM;
            let smooth = 0.5 * ((re - p.ic2(4)) / p.ic2(5)).tanh();
            pturb = 0.5 + smooth;
            pvisc = 0.5 - smooth;
            let xi = ((aorb / p.ic2(3)).max(3.0).log10() - p.abmin) / p.delab;
            let ind = (SIZEFWTABLE - 1).min(xi as usize);
            let deli1 = 1f64.min(xi - ind as f64);
            let deli2 = 1.0 - deli1;
            let fw = p.fwtable[ind] * deli2 + p.fwtable[ind + 1] * deli1;
            dturb = fturb * fw * uorb / GRAV;
        } else {
            dturb = 0.0;
            pturb = 0.0;
            pvisc = 1.0;
        }
        for ik in 0..nk {
            // WN_R: only correct for unbroken ice (WW3 comment)
            let sg = p.sig[ik];
            let dvisc = p.ic2(6) * wn_r[ik] * (VISCM * sg / 2.0).sqrt();
            let d1d = -1.0 * (pturb * (dturb * sg * sg).max(dvisc) + pvisc * dvisc) * smooth_dmax[ik];
            for ith in 0..nth {
                d[ith + ik * nth] = d1d;
            }
        }
    }
    let s = (0..ns).map(|is| d[is] * a[is]).collect();
    Ok((s, d))
}

/// Ice inputs of a node for [`ice_after_loop`].
#[derive(Clone, Copy, Debug)]
pub struct IceIn {
    /// Ice concentration `ICE`.
    pub ice: f64,
    /// Ice thickness `ICEH` (`ICEHINIT` without thickness forcing).
    pub iceh: f64,
    /// Maximum floe diameter of the forcing `ICEDMAX` (1000 m without
    /// forcing).
    pub icedmax: f64,
    /// Water depth.
    pub depth: f64,
    /// `INFLAGS2(4)`: whether the ice concentration was ever read.
    pub ice_read: bool,
    /// `ICEI`, the ice concentration field read by `W3SIC2` (with
    /// `ice_read`).
    pub icei: f64,
    /// Negative latitude on a rectilinear lat-lon grid (`IC2TURBS`).
    pub southern_rlg: bool,
}

/// Fluxes returned by [`ice_after_loop`].
#[derive(Clone, Copy, Debug, Default)]
pub struct IceFlux {
    /// `PHICE`: energy flux from the waves to the ice (W/m²).
    pub phice: f64,
    /// `TAUICE(1:2)`: momentum flux to the ice, divided by `DTG` only (as
    /// in WW3, without `DWAT`).
    pub tauice: [f64; 2],
}

/// The ice part of `W3SRCE` after the dynamic integration loop
/// (`IC_NUMERICS = F`, section 10.1): with ice (`INFLAGS2(4)` and
/// `ICE > 0`), `WN_R = WN1`, `CG_ICE = CG1`, `R = 1`, then [`sis2`] and
/// [`sic2`], the implicit attenuation `ATT = exp(ICE VDIC DTG) exp(ICE
/// VDIR2 DTG)` (times `exp(ICE VDIR DTG)` when `IS2BACKSCAT = 0`), the
/// scattering (isotropic back-scatter `ISO + (SPEC - ISO) exp(VDIR
/// IS2BACKSCAT DTG)` when `IS2ISOSCAT`, else the matrix exponential of the
/// scattering matrix) and the fluxes `PHICE`, `TAUICE`. Without ice, `ICEF`
/// is set to 0 when `IS2DUPDATE = F`. Updates `spec` and `icef` in place;
/// `wn1`, `cg1` are the open-water wavenumbers and group speeds, `dtg` the
/// global time step.
pub fn ice_after_loop(p: &IceParams, spec: &mut [f64], inp: &IceIn, icef: &mut f64, wn1: &[f64], cg1: &[f64], dtg: f64) -> Result<IceFlux> {
    let (nk, nth) = (p.nk, p.nth);
    let ns = nk * nth;
    let mut out = IceFlux::default();
    let ice = inp.ice;
    if inp.ice_read && ice > 0.0 {
        if p.ic_numerics {
            bail!("IC_NUMERICS = T (ice terms in the dynamic loop) is not ported");
        }
        if p.iicedisp {
            bail!("ICEDISP = T (LIU_FORWARD_DISPERSION in W3SRCE) is not ported");
        }
        let wn_r = wn1.to_vec();
        let cg_ice = cg1.to_vec();
        let mut r = vec![1f64; nk]; // in case IC2 is defined but not IS2
        let zero = vec![0f64; ns];
        let (mut vdir, mut vdir2) = (zero.clone(), zero.clone());
        if p.with_is2 {
            let o = sis2(p, spec, inp.depth, ice, inp.iceh, *icef, inp.icedmax, wn1, cg1, &wn_r, &cg_ice);
            vdir = o.d;
            vdir2 = o.dissip;
            r = o.r;
            *icef = o.icef;
        }
        let mut vdic = zero;
        if p.with_ic2 {
            let icei = if inp.ice_read { Some(inp.icei) } else { None };
            vdic = sic2(p, spec, inp.depth, inp.iceh, *icef, cg1, wn1, &wn_r, &cg_ice, &r, icei, inp.southern_rlg)?.1;
        }
        let spec2 = spec.to_vec();
        let mut scat = vec![0f64; nth];
        for ik in 0..nk {
            let is0 = ik * nth;
            // first part: dissipation
            let mut att = 1.0;
            if p.with_ic2 {
                att = (ice * vdic[is0] * dtg).exp();
            }
            if p.with_is2 {
                att *= (ice * vdir2[is0] * dtg).exp();
                if p.is2(2) == 0.0 {
                    // without re-distribution the scattering is an attenuation
                    att *= (ice * vdir[is0] * dtg).exp();
                }
            }
            for is in is0..is0 + nth {
                spec[is] = att * spec2[is];
            }
            // second part: scattering with re-distribution in directions
            if p.with_is2 && p.is2(2) >= 0.0 {
                let band = &mut spec[is0..is0 + nth];
                if p.is2(20) > 0.5 {
                    // isotropic part (eigenvalue 0) and the rest (eigenvalue VDIR)
                    let sc = (vdir[is0] * p.is2(2) * dtg).exp();
                    let iso = band.iter().sum::<f64>() / nth as f64;
                    for x in band.iter_mut() {
                        *x = iso + (*x - iso) * sc;
                    }
                } else {
                    // matrix exponential (as bottom scattering, Ardhuin & Herbers 2002)
                    let v = &p.is2eigvec;
                    for j in 0..nth {
                        let mut y = 0.0;
                        for i in 0..nth {
                            y += v[i * nth + j] * band[i];
                        }
                        scat[j] = y * (p.is2eigval[j] * vdir[is0] * dtg * p.is2(2)).exp();
                    }
                    for i in 0..nth {
                        let mut x = 0.0;
                        for j in 0..nth {
                            x += v[i * nth + j] * scat[j];
                        }
                        band[i] = x;
                    }
                }
            }
            // 10.2 fluxes of energy and momentum to the ice
            let factor = p.dden[ik] / cg1[ik];
            let factor2 = factor * GRAV * wn1[ik] / p.sig[ik];
            for ith in 0..nth {
                let is = ith + is0;
                let dsp = spec[is] - spec2[is];
                out.phice += dsp * factor;
                out.tauice[0] -= dsp * factor2 * p.ecos[ith];
                out.tauice[1] -= dsp * factor2 * p.esin[ith];
            }
        }
        out.phice = -1.0 * DWAT * GRAV * out.phice / dtg;
        out.tauice[0] /= dtg;
        out.tauice[1] /= dtg;
    } else if p.with_is2 && p.is2(10) < 0.5 {
        *icef = 0.0;
    }
    Ok(out)
}

/// Frequencies (Hz) of the scattering tables (`SIS1FTABLE`).
const SIS1FTABLE: [f64; NFTAB] = [
    0.0420, 0.04620, 0.050820, 0.0559020, 0.06149220, 0.06764142, 0.0744055620000000, 0.0818461182, 0.09003073002,
    0.099033803022, 0.1089371833242, 0.11983090, 0.13181399, 0.144995391, 0.15949493, 0.175444423115458, 0.192988865427003,
    0.212287751969704, 0.233516527, 0.256868, 0.28255499787, 0.310810497658843, 0.341891547424728, 0.376080702167200,
    0.413688772383920,
];

/// `SIS1ALPHATABLE(NTHICK, NFTAB)` of `INSIS2` for `IS2WIM1 < 0.5` (Kohout and
/// Meylan), column-major: thickness index fastest.
const SIS1ALPHATABLE: [f64; 500] = [
    1.78E-007, 2.21E-006, 6.57E-006, 1.31E-005, 2.28E-005,
    3.60E-005, 5.23E-005, 7.19E-005, 9.60E-005, 0.0001260665,
    0.0001621645, 0.0002032202, 0.0002483457, 0.0002987263, 0.0003571903,
    0.0004247358, 0.00049714, 0.0005689017, 0.0006469729, 0.0007470985,
    2.74E-007, 3.40E-006, 1.04E-005, 2.10E-005, 3.67E-005,
    5.82E-005, 8.52E-005, 0.0001179843, 0.0001581719, 0.0002075378,
    0.0002663323, 0.0003330095, 0.0004063158, 0.000487845, 0.0005815828,
    0.0006893214, 0.0008061048, 0.0009252626, 0.0010565619, 0.0012218782,
    4.16E-007, 5.21E-006, 1.62E-005, 3.32E-005, 5.91E-005,
    9.56E-005, 0.0001423817, 0.0002001783, 0.0002716622, 0.0003599526,
    0.0004656353, 0.0005861828, 0.0007194785, 0.0008683691, 0.0010401368,
    0.0012386623, 0.0014562639, 0.0016819935, 0.0019335276, 0.0022507523,
    6.25E-007, 7.95E-006, 2.50E-005, 5.28E-005, 9.71E-005,
    0.0001615969, 0.0002467311, 0.0003540015, 0.0004891301, 0.0006588358,
    0.0008648602, 0.0011024698, 0.0013675698, 0.0016664349, 0.0020150216,
    0.0024225002, 0.0028732348, 0.0033437743, 0.0038725363, 0.0045480433,
    9.32E-007, 1.21E-005, 3.91E-005, 8.60E-005, 0.000165062,
    0.0002848805, 0.0004473356, 0.0006565411, 0.0009257279, 0.0012707025,
    0.0016966237, 0.0021939892, 0.002754291, 0.0033926056, 0.0041466541,
    0.0050385513, 0.0060311467, 0.0070669538, 0.0082363302, 0.0097527074,
    1.38E-006, 1.86E-005, 6.30E-005, 0.0001464051, 0.0002950074,
    0.0005287398, 0.0008535728, 0.0012801516, 0.0018391215, 0.0025674934,
    0.0034790152, 0.0045539812, 0.0057742581, 0.0071757314, 0.0088467499,
    0.0108382999, 0.0130588683, 0.0153657347, 0.0179697664, 0.0213785773,
    2.05E-006, 2.91E-005, 0.0001064725, 0.0002646727, 0.0005601397,
    0.0010380532, 0.0017148531, 0.0026159282, 0.0038087163, 0.0053744112,
    0.0073439386, 0.0096752268, 0.0123295445, 0.0153861547, 0.0190373907,
    0.0233881203, 0.0282220195, 0.0332084504, 0.0388053869, 0.0461221517,
    3.03E-006, 4.71E-005, 0.0001918323, 0.000513363, 0.0011329164,
    0.00214921, 0.0036015483, 0.0055437802, 0.0081104965, 0.0114597841,
    0.0156453149, 0.0205751599, 0.0261681604, 0.0325784506, 0.040175062,
    0.0491364185, 0.0589976693, 0.0690831851, 0.0802738154, 0.0946821772,
    4.53E-006, 8.02E-005, 0.0003720589, 0.001067721, 0.0024215478,
    0.0046335423, 0.0077820302, 0.0119609444, 0.0173982353, 0.0243482166,
    0.0328666235, 0.0427569128, 0.0538584883, 0.0664227317, 0.0810452442,
    0.0979544205, 0.1162816559, 0.1348568677, 0.1551243874, 0.1804646861,
    6.84E-006, 0.0001465372, 0.0007768361, 0.0023492877, 0.0053624751,
    0.0101694228, 0.0168827981, 0.0256087065, 0.0366354159, 0.0502611381,
    0.0664627541, 0.0848577616, 0.1051596929, 0.1276991046, 0.1532607056,
    0.1820303259, 0.2126339282, 0.2433779188, 0.276250064, 0.3157667291,
    1.06E-005, 0.0002908265, 0.0017246725, 0.0053239059, 0.0119340853,
    0.0220417832, 0.035687347, 0.0528302869, 0.073649217, 0.098327901,
    0.1266340957, 0.1579155214, 0.1917144685, 0.2283923343, 0.2688390962,
    0.3131039168, 0.359282913, 0.4052034867, 0.4533028057, 0.5090275342,
    1.68E-005, 0.0006285069, 0.0039687874, 0.0119764661, 0.0257390989,
    0.0456577018, 0.0713257381, 0.1021551099, 0.1379813612, 0.1787934975,
    0.2240723105, 0.2727534775, 0.3241066558, 0.3786049119, 0.4374575502,
    0.5006384418, 0.5653760852, 0.6286093824, 0.6937583927, 0.7684424911,
    2.82E-005, 0.0014590465, 0.0091291531, 0.0257006983, 0.051998859,
    0.0879406127, 0.1317806874, 0.181756664, 0.2375759087, 0.2995488732,
    0.3668698493, 0.4374715634, 0.5099661328, 0.5855233305, 0.6668652222,
    0.754201342, 0.8422038741, 0.925009729, 1.0096608603, 1.1112763605,
    5.06E-005, 0.0035218608, 0.0201532593, 0.0509283911, 0.0962607638,
    0.1553638291, 0.2233707746, 0.2967358894, 0.3765436929, 0.4650074273,
    0.5607827923, 0.6590228472, 0.756685343, 0.8572097524, 0.9677298862,
    1.0892101997, 1.2094355521, 1.3151711317, 1.4236588573, 1.5698199322,
    9.99E-005, 0.0084186275, 0.0412091805, 0.0918067318, 0.1629311719,
    0.2530998556, 0.3511295005, 0.4513397894, 0.5593550205, 0.6817564883,
    0.815653431, 0.9499632903, 1.0782538468, 1.2090808019, 1.3588213156,
    1.5298071338, 1.6953922745, 1.8273584757, 1.9646959304, 2.1824935098,
    0.0002197702, 0.0188788997, 0.076522568, 0.1517352519, 0.2567566667,
    0.3885311336, 0.5247441812, 0.6573850155, 0.8007012438, 0.9681737964,
    1.1534672992, 1.3345908371, 1.4996817772, 1.666222509, 1.8655897771,
    2.1018122262, 2.3245234997, 2.4819251275, 2.6492217053, 2.9647668115,
    0.0005376597, 0.0381101752, 0.1293683129, 0.2359311434, 0.3874916537,
    0.576671054, 0.7645781239, 0.9404856491, 1.1305780655, 1.356723506,
    1.6068566622, 1.844140891, 2.0502598744, 2.2550088512, 2.5085040228,
    2.815733869, 3.096422752, 3.2715199709, 3.462031842, 3.8799520309,
    0.0014123358, 0.0685490628, 0.203577783, 0.354894301, 0.5714749709,
    0.8380065625, 1.0957796855, 1.3307089678, 1.5810360789, 1.8762692254,
    2.197628874, 2.4940515596, 2.7425260367, 2.9852393578, 3.2886471617,
    3.6568112069, 3.9833963075, 4.1693103781, 4.3750188868, 4.868451483,
    0.0037199175, 0.1126937266, 0.3066036571, 0.5225755275, 0.8226663225,
    1.1819668289, 1.5247740065, 1.8332140124, 2.152798111, 2.5177593097,
    2.9057350396, 3.2594397982, 3.5553142762, 3.8420910788, 4.1929464318,
    4.6097641365, 4.975238806, 5.1850270882, 5.4153321104, 5.9516719058,
    0.009037082, 0.1761561171, 0.4486599083, 0.747307698, 1.1406376531,
    1.5966776005, 2.0299434584, 2.4186982894, 2.8096987013, 3.2399866752,
    3.6899433755, 4.105747098, 4.4658248514, 4.8172532951, 5.2300901821,
    5.7046450426, 6.1270633187, 6.3975380116, 6.6851833028, 7.2647998139,
    0.0191118494, 0.2660897811, 0.6359174284, 1.0316197503, 1.5313672569,
    2.0961171527, 2.6294074335, 3.1054178673, 3.5736395046, 4.0762156975,
    4.5979660806, 5.0883063988, 5.5262800229, 5.9567260476, 6.4472029911,
    6.9971641707, 7.4947435133, 7.841785986, 8.2023388524, 8.8466621683,
    0.0352335589, 0.3840118688, 0.8727819679, 1.3979077636, 2.0496249898,
    2.7683726012, 3.4325758193, 4.0133546648, 4.572769913, 5.1624475081,
    5.7663362962, 6.3283500032, 6.8264493734, 7.3114532863, 7.8576036504,
    8.4626783134, 9.0041860776, 9.378581111, 9.7670030485, 10.46036883,
    0.0588123086, 0.5407217302, 1.1866725247, 1.8744101474, 2.7029832853,
    3.586993579, 4.3802119773, 5.0573372156, 5.6991092734, 6.3694458919,
    7.0521552001, 7.6857035561, 8.2469182713, 8.7926955021, 9.4042992778,
    10.0780089999, 10.6798375161, 11.0993158393, 11.5359791079, 12.3054058274,
    0.0912925253, 0.7587053794, 1.5933132286, 2.4467380656, 3.4467575836,
    4.4845019618, 5.3910117536, 6.1520041613, 6.8749329856, 7.6395860278,
    8.4275368684, 9.1660844666, 9.8268448897, 10.4746437466, 11.2031746957,
    12.0078539018, 12.7326579385, 13.2484048884, 13.7868789237, 14.7164075718,
    0.1320035456, 1.258347597, 2.2697363962, 2.8626691529, 3.4800285532,
    4.1591450976, 4.8036627318, 5.4044338329, 6.0517560386, 6.8415021436,
    7.8080659794, 8.9257124242, 10.1640694041, 11.5476843252, 13.1494718506,
    14.9965430946, 16.972215146, 18.903468102, 20.9125176371, 23.3776351255,
];

/// `SIS1ALPHATABLE` of `INSIS2` for `IS2WIM1 >= 0.5` (WIM1, T. Williams), column-major.
const SIS1ALPHATABLE_WIM: [f64; 500] = [
    3.80373e-06, 6.02822e-06, 1.12121e-05, 2.24588e-05, 3.05165e-05,
    3.85142e-05, 5.29712e-05, 7.55453e-05, 1.05531e-04, 1.38952e-04,
    1.73378e-04, 2.09421e-04, 2.47299e-04, 2.87669e-04, 3.34304e-04,
    3.91094e-04, 4.57710e-04, 5.31901e-04, 6.14490e-04, 7.08299e-04,
    4.76980e-06, 6.95433e-06, 1.54179e-05, 2.46138e-05, 3.79955e-05,
    5.96725e-05, 8.81668e-05, 1.20881e-04, 1.57364e-04, 1.99666e-04,
    2.50723e-04, 3.13346e-04, 3.89907e-04, 4.81606e-04, 5.88164e-04,
    7.08707e-04, 8.42536e-04, 9.88622e-04, 1.14498e-03, 1.30931e-03,
    5.84556e-06, 8.58738e-06, 2.00947e-05, 3.18410e-05, 5.67385e-05,
    9.37456e-05, 1.37304e-04, 1.87832e-04, 2.49218e-04, 3.28900e-04,
    4.31988e-04, 5.59364e-04, 7.11302e-04, 8.86736e-04, 1.08074e-03,
    1.28808e-03, 1.50809e-03, 1.74230e-03, 1.98930e-03, 2.24607e-03,
    6.90305e-06, 1.12155e-05, 2.59514e-05, 5.15063e-05, 9.18902e-05,
    1.45703e-04, 2.18304e-04, 3.18476e-04, 4.51992e-04, 6.21037e-04,
    8.24428e-04, 1.05881e-03, 1.32112e-03, 1.60970e-03, 1.92396e-03,
    2.26484e-03, 2.63577e-03, 3.04238e-03, 3.49116e-03, 3.98911e-03,
    8.46004e-06, 1.46791e-05, 3.69837e-05, 8.32870e-05, 1.49151e-04,
    2.46175e-04, 3.92165e-04, 5.95871e-04, 8.57938e-04, 1.17280e-03,
    1.53668e-03, 1.95109e-03, 2.42048e-03, 2.95341e-03, 3.56468e-03,
    4.27150e-03, 5.08843e-03, 6.02927e-03, 7.11153e-03, 8.35459e-03,
    1.11513e-05, 1.95090e-05, 5.78215e-05, 1.32267e-04, 2.57932e-04,
    4.61051e-04, 7.55044e-04, 1.14193e-03, 1.62478e-03, 2.21208e-03,
    2.92101e-03, 3.77684e-03, 4.80845e-03, 6.04635e-03, 7.52256e-03,
    9.26818e-03, 1.13101e-02, 1.36712e-02, 1.63732e-02, 1.94360e-02,
    1.52571e-05, 2.78128e-05, 9.43665e-05, 2.30636e-04, 4.90057e-04,
    9.02105e-04, 1.48012e-03, 2.24822e-03, 3.24754e-03, 4.53444e-03,
    6.16979e-03, 8.21171e-03, 1.07145e-02, 1.37253e-02, 1.72805e-02,
    2.14091e-02, 2.61367e-02, 3.14840e-02, 3.74632e-02, 4.40798e-02,
    2.13731e-05, 4.29771e-05, 1.64050e-04, 4.49510e-04, 9.84576e-04,
    1.82921e-03, 3.06425e-03, 4.79802e-03, 7.14520e-03, 1.02139e-02,
    1.40940e-02, 1.88529e-02, 2.45396e-02, 3.11875e-02, 3.88141e-02,
    4.74255e-02, 5.70233e-02, 6.76049e-02, 7.91611e-02, 9.16776e-02,
    3.17688e-05, 7.05749e-05, 3.13449e-04, 9.26652e-04, 2.08059e-03,
    3.98338e-03, 6.87070e-03, 1.09528e-02, 1.63868e-02, 2.32726e-02,
    3.16649e-02, 4.15851e-02, 5.30293e-02, 6.59789e-02, 8.04085e-02,
    9.62863e-02, 1.13575e-01, 1.32236e-01, 1.52233e-01, 1.73533e-01,
    5.14019e-05, 1.24975e-04, 6.49063e-04, 2.01553e-03, 4.70028e-03,
    9.17913e-03, 1.58010e-02, 2.47576e-02, 3.61186e-02, 4.98708e-02,
    6.59568e-02, 8.42994e-02, 1.04811e-01, 1.27404e-01, 1.51996e-01,
    1.78511e-01, 2.06873e-01, 2.37011e-01, 2.68862e-01, 3.02365e-01,
    9.05394e-05, 2.44433e-04, 1.43884e-03, 4.66981e-03, 1.09558e-02,
    2.08947e-02, 3.46681e-02, 5.22109e-02, 7.33489e-02, 9.78738e-02,
    1.25579e-01, 1.56274e-01, 1.89790e-01, 2.25973e-01, 2.64687e-01,
    3.05808e-01, 3.49223e-01, 3.94830e-01, 4.42530e-01, 4.92227e-01,
    1.75074e-04, 5.25080e-04, 3.39573e-03, 1.10036e-02, 2.46312e-02,
    4.44096e-02, 6.99511e-02, 1.00742e-01, 1.36302e-01, 1.76219e-01,
    2.20147e-01, 2.67802e-01, 3.18937e-01, 3.73339e-01, 4.30813e-01,
    4.91183e-01, 5.54282e-01, 6.19954e-01, 6.88048e-01, 7.58418e-01,
    3.74908e-04, 1.22443e-03, 8.17252e-03, 2.46994e-02, 5.11207e-02,
    8.63422e-02, 1.29173e-01, 1.78641e-01, 2.34006e-01, 2.94691e-01,
    3.60235e-01, 4.30258e-01, 5.04426e-01, 5.82438e-01, 6.64019e-01,
    7.48914e-01, 8.36872e-01, 9.27649e-01, 1.02102e+00, 1.11676e+00,
    8.82800e-04, 3.03043e-03, 1.88248e-02, 5.09125e-02, 9.68861e-02,
    1.54064e-01, 2.20524e-01, 2.94949e-01, 3.76394e-01, 4.64138e-01,
    5.57588e-01, 6.56231e-01, 7.59605e-01, 8.67281e-01, 9.78849e-01,
    1.09393e+00, 1.21214e+00, 1.33315e+00, 1.45662e+00, 1.58224e+00,
    2.22521e-03, 7.51676e-03, 3.98357e-02, 9.57376e-02, 1.68983e-01,
    2.55527e-01, 3.52939e-01, 4.59644e-01, 5.74490e-01, 6.96566e-01,
    8.25086e-01, 9.59331e-01, 1.09864e+00, 1.24238e+00, 1.38997e+00,
    1.54085e+00, 1.69451e+00, 1.85049e+00, 2.00833e+00, 2.16765e+00,
    5.66833e-03, 1.75464e-02, 7.66453e-02, 1.65727e-01, 2.74951e-01,
    3.99380e-01, 5.36249e-01, 6.83721e-01, 8.40370e-01, 1.00497e+00,
    1.17642e+00, 1.35368e+00, 1.53580e+00, 1.72190e+00, 1.91118e+00,
    2.10291e+00, 2.29645e+00, 2.49123e+00, 2.68675e+00, 2.88259e+00,
    1.36539e-02, 3.72456e-02, 1.35140e-01, 2.67780e-01, 4.22901e-01,
    5.95111e-01, 7.81333e-01, 9.79307e-01, 1.18712e+00, 1.40304e+00,
    1.62548e+00, 1.85300e+00, 2.08433e+00, 2.31832e+00, 2.55400e+00,
    2.79055e+00, 3.02727e+00, 3.26361e+00, 3.49910e+00, 3.73337e+00,
    2.98722e-02, 7.15314e-02, 2.21423e-01, 4.09271e-01, 6.21810e-01,
    8.53369e-01, 1.10028e+00, 1.35949e+00, 1.62829e+00, 1.90425e+00,
    2.18522e+00, 2.46937e+00, 2.75516e+00, 3.04136e+00, 3.32695e+00,
    3.61118e+00, 3.89345e+00, 4.17331e+00, 4.45047e+00, 4.72471e+00,
    5.88578e-02, 1.25612e-01, 3.41821e-01, 5.98292e-01, 8.81793e-01,
    1.18598e+00, 1.50596e+00, 1.83748e+00, 2.17675e+00, 2.52062e+00,
    2.86649e+00, 3.21233e+00, 3.55662e+00, 3.89823e+00, 4.23636e+00,
    4.57049e+00, 4.90029e+00, 5.22559e+00, 5.54632e+00, 5.86250e+00,
    1.05468e-01, 2.04724e-01, 5.03158e-01, 8.44026e-01, 1.21416e+00,
    1.60557e+00, 2.01139e+00, 2.42573e+00, 2.84383e+00, 3.26205e+00,
    3.67776e+00, 4.08914e+00, 4.49503e+00, 4.89472e+00, 5.28788e+00,
    5.67440e+00, 6.05432e+00, 6.42782e+00, 6.79514e+00, 7.15656e+00,
    1.74545e-01, 3.14137e-01, 7.13063e-01, 1.15691e+00, 1.63126e+00,
    2.12499e+00, 2.62866e+00, 3.13493e+00, 3.63860e+00, 4.13623e+00,
    4.62576e+00, 5.10610e+00, 5.57680e+00, 6.03787e+00, 6.48955e+00,
    6.93227e+00, 7.36652e+00, 7.79283e+00, 8.21174e+00, 8.62378e+00,
    2.70834e-01, 4.59350e-01, 9.80330e-01, 1.54856e+00, 2.14570e+00,
    2.75617e+00, 3.36813e+00, 3.97376e+00, 4.56847e+00, 5.14997e+00,
    5.71743e+00, 6.27093e+00, 6.81101e+00, 7.33850e+00, 7.85432e+00,
    8.35942e+00, 8.85471e+00, 9.34106e+00, 9.81925e+00, 1.02900e+01,
    3.99120e-01, 6.46435e-01, 1.31507e+00, 2.03136e+00, 2.76957e+00,
    3.50946e+00, 4.23832e+00, 4.94958e+00, 5.64069e+00, 6.31137e+00,
    6.96253e+00, 7.59564e+00, 8.21234e+00, 8.81426e+00, 9.40295e+00,
    9.97983e+00, 1.05461e+01, 1.11030e+01, 1.16515e+01, 1.21924e+01,
    5.64538e-01, 8.82405e-01, 1.72870e+00, 2.61770e+00, 3.51364e+00,
    4.39362e+00, 5.24695e+00, 6.07052e+00, 6.86499e+00, 7.63269e+00,
    8.37647e+00, 9.09919e+00, 9.80351e+00, 1.04918e+01, 1.11660e+01,
    1.18280e+01, 1.24792e+01, 1.31209e+01, 1.37543e+01, 1.43803e+01,
    7.72889e-01, 1.17552e+00, 2.23335e+00, 3.31894e+00, 4.38704e+00,
    5.41672e+00, 6.40295e+00, 7.34788e+00, 8.25618e+00, 9.13289e+00,
    9.98272e+00, 1.08097e+01, 1.16174e+01, 1.24086e+01, 1.31857e+01,
    1.39506e+01, 1.47049e+01, 1.54501e+01, 1.61872e+01, 1.69173e+01,
];

/// `SIS1ALPHATABLE2` of `INSIS2`: the Fortran source list has 520 values for a
/// `(NTHICK, NFTAB)` = 500 array; `RESHAPE` keeps the first 500 (kept as written).
const SIS1ALPHATABLE2: [f64; 520] = [
    0.000001693306, 0.000001694535, 0.000001709985, 0.000001718371, 0.000001718689,
    0.000001715291, 0.000001711765, 0.000001709562, 0.000001708573, 0.000001708047,
    0.000001707216, 0.000001705508, 0.000001702489, 0.000001697757, 0.000001690926,
    0.000001681715, 0.000001670039, 0.000001655970, 0.000001639540, 0.000001620666,
    0.000002345944, 0.000002373942, 0.000002352788, 0.000002336058, 0.000002340157,
    0.000002361578, 0.000002389818, 0.000002414910, 0.000002430787, 0.000002435510,
    0.000002429671, 0.000002414451, 0.000002390516, 0.000002358094, 0.000002317788,
    0.000002271221, 0.000002220674, 0.000002167641, 0.000002111561, 0.000002051905,
    0.000003223057, 0.000003267049, 0.000003224977, 0.000003204685, 0.000003222416,
    0.000003261346, 0.000003299552, 0.000003321243, 0.000003319410, 0.000003293875,
    0.000003247777, 0.000003184750, 0.000003107742, 0.000003019246, 0.000002922032,
    0.000002819435, 0.000002714739, 0.000002609966, 0.000002505271, 0.000002400993,
    0.000004390462, 0.000004433012, 0.000004403102, 0.000004405652, 0.000004433604,
    0.000004455373, 0.000004447247, 0.000004400212, 0.000004316137, 0.000004201667,
    0.000004064075, 0.000003909816, 0.000003744650, 0.000003573915, 0.000003402281,
    0.000003233187, 0.000003068613, 0.000002909686, 0.000002757781, 0.000002614371,
    0.000005932907, 0.000005957402, 0.000005982223, 0.000006017129, 0.000006011597,
    0.000005931834, 0.000005775734, 0.000005561133, 0.000005309860, 0.000005038831,
    0.000004758982, 0.000004477929, 0.000004202314, 0.000003938073, 0.000003689194,
    0.000003456677, 0.000003239230, 0.000003035853, 0.000002848219, 0.000002677864,
    0.000007962387, 0.000007965182, 0.000008068497, 0.000008081519, 0.000007918056,
    0.000007586175, 0.000007143591, 0.000006652694, 0.000006157274, 0.000005680186,
    0.000005230854, 0.000004813199, 0.000004429817, 0.000004082375, 0.000003770294,
    0.000003490093, 0.000003236746, 0.000003006720, 0.000002800108, 0.000002616762,
    0.000010630745, 0.000010631503, 0.000010754695, 0.000010555056, 0.000009993969,
    0.000009206201, 0.000008341755, 0.000007500204, 0.000006728370, 0.000006039825,
    0.000005432987, 0.000004901489, 0.000004438118, 0.000004035401, 0.000003685333,
    0.000003379621, 0.000003110711, 0.000002873062, 0.000002663457, 0.000002478682,
    0.000014147503, 0.000014179502, 0.000014068022, 0.000013255509, 0.000011955952,
    0.000010517681, 0.000009161184, 0.000007970672, 0.000006956434, 0.000006102184,
    0.000005386202, 0.000004787198, 0.000004285040, 0.000003861250, 0.000003500220,
    0.000003190247, 0.000002923212, 0.000002692594, 0.000002491216, 0.000002312444,
    0.000018803447, 0.000018843681, 0.000017888469, 0.000015846997, 0.000013469781,
    0.000011291688, 0.000009484334, 0.000008030183, 0.000006862685, 0.000005921675,
    0.000005161810, 0.000004546889, 0.000004045234, 0.000003629478, 0.000003278778,
    0.000002980118, 0.000002726294, 0.000002510734, 0.000002323048, 0.000002154027,
    0.000024999293, 0.000024767047, 0.000021866213, 0.000017907275, 0.000014286707,
    0.000011453674, 0.000009339677, 0.000007754652, 0.000006540404, 0.000005593636,
    0.000004848620, 0.000004258144, 0.000003783667, 0.000003393700, 0.000003065741,
    0.000002787091, 0.000002551635, 0.000002353114, 0.000002179924, 0.000002022176,
    0.000033276027, 0.000031812088, 0.000025413911, 0.000019081783, 0.000014360014,
    0.000011113013, 0.000008878307, 0.000007287536, 0.000006111165, 0.000005216269,
    0.000004523259, 0.000003978625, 0.000003542278, 0.000003183888, 0.000002882748,
    0.000002627060, 0.000002410531, 0.000002226889, 0.000002066156, 0.000001920400,
    0.000044336466, 0.000039340417, 0.000027862847, 0.000019248910, 0.000013854771,
    0.000010491793, 0.000008293280, 0.000006775936, 0.000005680752, 0.000004862822,
    0.000004234499, 0.000003739067, 0.000003338786, 0.000003008719, 0.000002732485,
    0.000002498817, 0.000002299071, 0.000002126144, 0.000001974716, 0.000001841606,
    0.000059030780, 0.000046157443, 0.000028771759, 0.000018580063, 0.000013047269,
    0.000009812168, 0.000007739586, 0.000006322672, 0.000005313968, 0.000004570787,
    0.000004001034, 0.000003546046, 0.000003172357, 0.000002862735, 0.000002606289,
    0.000002391441, 0.000002205202, 0.000002038837, 0.000001893870, 0.000001773775,
    0.000078242681, 0.000050872198, 0.000028192943, 0.000017445281, 0.000012194039,
    0.000009215299, 0.000007293336, 0.000005968851, 0.000005030058, 0.000004344187,
    0.000003817214, 0.000003389423, 0.000003031921, 0.000002735020, 0.000002492733,
    0.000002292508, 0.000002116550, 0.000001954358, 0.000001814339, 0.000001706341,
    0.000102553944, 0.000052628108, 0.000026654291, 0.000016235235, 0.000011448810,
    0.000008738110, 0.000006951009, 0.000005700488, 0.000004811678, 0.000004163305,
    0.000003662892, 0.000003251855, 0.000002905015, 0.000002617308, 0.000002385271,
    0.000002195475, 0.000002027393, 0.000001869747, 0.000001734845, 0.000001635748,
    0.000131528803, 0.000051648919, 0.000024847696, 0.000015214276, 0.000010847813,
    0.000008336757, 0.000006656766, 0.000005471182, 0.000004622914, 0.000003999733,
    0.000003516266, 0.000003119051, 0.000002785020, 0.000002508708, 0.000002285704,
    0.000002102851, 0.000001941273, 0.000001790864, 0.000001662390, 0.000001566768,
    0.000162640998, 0.000049040647, 0.000023269162, 0.000014460197, 0.000010345625,
    0.000007943173, 0.000006352063, 0.000005238097, 0.000004434583, 0.000003834908,
    0.000003367217, 0.000002987629, 0.000002673539, 0.000002414062, 0.000002200682,
    0.000002022244, 0.000001866432, 0.000001726382, 0.000001605605, 0.000001507441,
    0.000190593962, 0.000046018523, 0.000022036249, 0.000013884361, 0.000009879926,
    0.000007534440, 0.000006029990, 0.000004998037, 0.000004246561, 0.000003673278,
    0.000003223583, 0.000002865466, 0.000002575963, 0.000002336404, 0.000002132787,
    0.000001957123, 0.000001806342, 0.000001677844, 0.000001564855, 0.000001460377,
    0.000208777708, 0.000043254960, 0.000020953050, 0.000013327645, 0.000009433705,
    0.000007161759, 0.000005746018, 0.000004790209, 0.000004085598, 0.000003536085,
    0.000003102390, 0.000002762488, 0.000002493229, 0.000002269788, 0.000002073862,
    0.000001900259, 0.000001753553, 0.000001634519, 0.000001527844, 0.000001418102,
    0.000213643676, 0.000040801760, 0.000019820045, 0.000012719494, 0.000009042103,
    0.000006888517, 0.000005551025, 0.000004645394, 0.000003968666, 0.000003433279,
    0.000003008733, 0.000002677964, 0.000002418184, 0.000002202481, 0.000002011120,
    0.000001840100, 0.000001696853, 0.000001583290, 0.000001480832, 0.000001370962,
    0.000208202335, 0.000038538552, 0.000018774435, 0.000012168047, 0.000008723021,
    0.000006681551, 0.000005399851, 0.000004523630, 0.000003863808, 0.000003339344,
    0.000002922742, 0.000002598268, 0.000002343776, 0.000002132781, 0.000001945837,
    0.000001779030, 0.000001639702, 0.000001529688, 0.000001430567, 0.000001324016,
    0.000198860865, 0.000036596393, 0.000018094381, 0.000011782298, 0.000008423914,
    0.000006432368, 0.000005194822, 0.000004356699, 0.000003726655, 0.000003224518,
    0.000002825485, 0.000002515982, 0.000002274472, 0.000002073961, 0.000001894595,
    0.000001733161, 0.000001598541, 0.000001493403, 0.000001398055, 0.000001292778,
    0.000188086831, 0.000035005432, 0.000017449034, 0.000011378985, 0.000008128438,
    0.000006206255, 0.000005018705, 0.000004216819, 0.000003612753, 0.000003129463,
    0.000002744757, 0.000002446723, 0.000002214548, 0.000002021421, 0.000001847651,
    0.000001690537, 0.000001559668, 0.000001458051, 0.000001365539, 0.000001262003,
    0.000176803511, 0.000033479878, 0.000016713766, 0.000010962586, 0.000007891418,
    0.000006057305, 0.000004907363, 0.000004122061, 0.000003528516, 0.000003054204,
    0.000002676573, 0.000002382977, 0.000002153342, 0.000001962633, 0.000001792438,
    0.000001639642, 0.000001512208, 0.000001412404, 0.000001322051, 0.000001223019,
    0.000157501278, 0.000031061181, 0.000013781133, 0.000009601049, 0.000007947466,
    0.000006760411, 0.000005655833, 0.000004680347, 0.000003892156, 0.000003277589,
    0.000002790517, 0.000002392986, 0.000002067558, 0.000001808761, 0.000001609460,
    0.000001454171, 0.000001323348, 0.000001206327, 0.000001111024, 0.000001045460,
    0.000007782240, 0.000002755736, 0.000000005024, 0.000000101927, 0.000088216814,
    0.009792270789, 0.030135654570, 0.009293514348, 0.001549599316, 0.000298014001,
    0.000055965309, 0.000007063938, 0.000000637549, 0.000000072256, 0.000000017854,
    0.000000008901, 0.000000004238, 0.000000001169, 0.000000000492, 0.000000001113,
];

#[cfg(test)]
mod tests {
    use super::*;

    const IRI: &str = "&SIC2 IC2DISPER = F, IC2TURB = 1.0, IC2ROUGH = 0.001, IC2REYNOLDS = 150000, IC2SMOOTH = 200000., IC2VISC = 2.0 /
&SIS2 ISC1 = 0.2, IS2C2 = 0.0, IS2C3 = 0., IS2BACKSCAT = 1., IS2BREAK = T, IS2DUPDATE = F, IS2CREEPB = 2E7, IS2CREEPD = 0.5 /
&MISC ICEHINIT = 0.5, ICEHMIN = 0.1, CICE0 = 0.25, CICEN = 2.00 /
";

    fn params(nk: usize, nth: usize) -> IceParams {
        let sp = Spec::new(nk, nth, 0.0339, 1.1, 0.0);
        IceParams::from_namelists(&Namelists::parse(IRI), &sp).unwrap()
    }

    /// Deep-water wavenumbers and group speeds.
    fn deep(p: &IceParams) -> (Vec<f64>, Vec<f64>) {
        (p.sig.iter().map(|s| s * s / GRAV).collect(), p.sig.iter().map(|s| GRAV / (2.0 * s)).collect())
    }

    /// A unimodal spectrum of Hs = 2 m peaked at 0.1 Hz towards θ = 0.
    fn spectrum(p: &IceParams, cg: &[f64]) -> Vec<f64> {
        let (nk, nth) = (p.nk, p.nth);
        let mut a = vec![0f64; nk * nth];
        for ik in 0..nk {
            let f = p.sig[ik] * TPIINV;
            for ith in 0..nth {
                let c = p.ecos[ith].max(0.0);
                a[ith + ik * nth] = (-((f - 0.1) / 0.02).powi(2)).exp() * c * c + 1e-6;
            }
        }
        let e: f64 = (0..nk * nth).map(|is| a[is] * p.dden[is / nth] / cg[is / nth]).sum();
        a.iter().map(|x| x * 0.25 / e).collect()
    }

    fn band_sums(p: &IceParams, a: &[f64]) -> Vec<f64> {
        (0..p.nk).map(|ik| a[ik * p.nth..(ik + 1) * p.nth].iter().sum()).collect()
    }

    #[test]
    fn ice_params_from_iri_namelists() {
        let p = params(36, 36);
        assert_eq!(p.is2pars[0], 0.2f32 as f64);
        assert_eq!(p.is2pars[2], 1.0);
        assert_eq!(p.is2pars[9], 0.0);
        assert_eq!(p.is2pars[11], 2e7);
        assert_eq!(p.is2pars[19], 1.0);
        assert_eq!(p.is2pars[22], 1.0);
        assert_eq!(p.ic2pars[2], 0.001f32 as f64);
        assert_eq!(p.ic2pars[5], 2.0);
        assert_eq!(p.iicehmin, 0.1f32 as f64);
        assert!(!p.iicedisp && !p.ic_numerics);
    }

    #[test]
    fn insis2_tables_finite_and_positive() {
        let p = params(36, 36);
        assert_eq!(p.sis2alphas.len(), NTHICK * 36);
        assert!(p.sis2alphas.iter().all(|x| x.is_finite() && *x > 0.0));
        assert!(p.sis2alpha2.iter().all(|x| x.is_finite() && *x >= 0.0));
        assert_eq!(p.icedavetab.len(), NICED);
        assert!(p.icedavetab.iter().all(|x| x.is_finite() && *x > 0.0));
        for (i, &d) in p.icedavetab.iter().enumerate() {
            // the mean lies between IS2DMIN and the maximum diameter
            assert!(d >= 20.0 && d <= (i as f64 + 1.0).max(20.2) + 1e-9, "{i} {d}");
        }
        assert!(p.is2eigval.iter().all(|x| x.is_finite() && *x >= 0.0));
        // orthonormal eigenvectors
        let n = p.nth;
        let v = &p.is2eigvec;
        for i in 0..n {
            for j in 0..n {
                let s: f64 = (0..n).map(|k| v[k * n + i] * v[k * n + j]).sum();
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((s - e).abs() < 1e-10, "{i} {j} {s}");
            }
        }
    }

    #[test]
    fn fsd_dave_limits() {
        assert_eq!(fsd_dave(0.0, 300.0, 0.9), 300.0);
        let m = fsd_dave(20.0, 300.0, 0.9);
        assert!(m > 20.0 && m < 300.0);
    }

    #[test]
    fn zero_ice_gives_zero_terms() {
        let p = params(36, 36);
        let (wn, cg) = deep(&p);
        let a = spectrum(&p, &cg);
        let o = sis2(&p, &a, 4000.0, 0.0, 0.5, 1000.0, 1000.0, &wn, &cg, &wn, &cg);
        assert!(o.s.iter().chain(&o.d).chain(&o.dissip).all(|x| *x == 0.0));
        assert!(o.r.iter().all(|x| *x == 1.0));
        assert_eq!(o.icef, 0.0);
        let r = vec![1.0; p.nk];
        let (s, d) = sic2(&p, &a, 4000.0, 0.0, 1000.0, &cg, &wn, &wn, &cg, &r, Some(0.5), false).unwrap();
        assert!(s.iter().chain(&d).all(|x| *x == 0.0));
        let (s, d) = sic2(&p, &a, 4000.0, 0.5, 1000.0, &cg, &wn, &wn, &cg, &r, Some(0.0), false).unwrap();
        assert!(s.iter().chain(&d).all(|x| *x == 0.0));
        let mut spec = a.clone();
        let mut icef = 1000.0;
        let inp = IceIn { ice: 0.0, iceh: 0.5, icedmax: 1000.0, depth: 4000.0, ice_read: true, icei: 0.0, southern_rlg: false };
        let fl = ice_after_loop(&p, &mut spec, &inp, &mut icef, &wn, &cg, 600.0).unwrap();
        assert_eq!(spec, a);
        assert_eq!(icef, 0.0); // IS2DUPDATE = F
        assert_eq!(fl.phice, 0.0);
        assert_eq!(fl.tauice, [0.0, 0.0]);
    }

    #[test]
    fn sis2_ratio_at_least_one_and_finite() {
        let p = params(36, 36);
        let (wn, cg) = deep(&p);
        let a = spectrum(&p, &cg);
        for &(cice, iceh, icef) in &[(0.5, 0.5, 1000.0), (1.0, 2.0, 100.0), (0.1, 0.05, 10.0)] {
            let o = sis2(&p, &a, 4000.0, cice, iceh, icef, 1000.0, &wn, &cg, &wn, &cg);
            assert!(o.r.iter().all(|r| r.is_finite() && *r >= 1.0));
            assert!(o.s.iter().chain(&o.d).chain(&o.dissip).all(|x| x.is_finite()));
            assert!(o.d.iter().all(|x| *x <= 0.0));
            assert!(o.dissip.iter().all(|x| *x <= 0.0));
            assert!(o.icef.is_finite() && o.icef > 0.0 && o.icef <= 1000.0);
        }
    }

    fn scattering_conserves(iso: bool) {
        let mut p = params(36, 36);
        p.with_ic2 = false;
        p.is2pars[11] = 0.0; // no creep: DISSIP = 0
        p.is2pars[19] = if iso { 1.0 } else { 0.0 };
        let (wn, cg) = deep(&p);
        let a = spectrum(&p, &cg);
        let mut spec = a.clone();
        let mut icef = 1000.0;
        let inp = IceIn { ice: 0.8, iceh: 0.5, icedmax: 1000.0, depth: 4000.0, ice_read: true, icei: 0.8, southern_rlg: false };
        let fl = ice_after_loop(&p, &mut spec, &inp, &mut icef, &wn, &cg, 3600.0).unwrap();
        let (b0, b1) = (band_sums(&p, &a), band_sums(&p, &spec));
        for ik in 0..p.nk {
            assert!((b1[ik] - b0[ik]).abs() <= 1e-12 * b0[ik], "{ik} {} {}", b0[ik], b1[ik]);
        }
        // energy redistributed towards the back directions
        let ik = (0..p.nk).max_by(|&i, &j| b0[i].partial_cmp(&b0[j]).unwrap()).unwrap();
        let back = ik * p.nth + p.nth / 2;
        assert!(spec[back] > a[back] * 1.01, "{} {}", spec[back], a[back]);
        let e0: f64 = (0..p.nk).map(|ik| b0[ik] * p.dden[ik] / cg[ik]).sum();
        assert!(fl.phice.abs() <= 1e-9 * DWAT * GRAV * e0 / 3600.0);
        assert!(fl.tauice[0] > 0.0);
    }

    #[test]
    fn isotropic_backscatter_conserves_energy() {
        scattering_conserves(true);
    }

    #[test]
    fn matrix_scattering_conserves_energy() {
        scattering_conserves(false);
    }

    #[test]
    fn iri_after_loop_dissipates() {
        let p = params(36, 36);
        let (wn, cg) = deep(&p);
        let a = spectrum(&p, &cg);
        let mut spec = a.clone();
        let mut icef = 1000.0;
        let inp = IceIn { ice: 0.6, iceh: 0.5, icedmax: 1000.0, depth: 4000.0, ice_read: true, icei: 0.6, southern_rlg: false };
        let fl = ice_after_loop(&p, &mut spec, &inp, &mut icef, &wn, &cg, 600.0).unwrap();
        assert!(spec.iter().all(|x| x.is_finite() && *x >= 0.0));
        let (b0, b1) = (band_sums(&p, &a), band_sums(&p, &spec));
        assert!((0..p.nk).all(|ik| b1[ik] <= b0[ik] * (1.0 + 1e-12)));
        assert!(fl.phice > 0.0 && fl.phice.is_finite());
        assert!(icef.is_finite() && icef > 0.0 && icef <= 1000.0);
    }

    #[test]
    fn sic2_viscous_by_hand() {
        // IC2TURB = 0: D = -IC2VISC WN_R sqrt(VISCM SIG / 2)
        let mut p = params(2, 4);
        p.ic2pars[1] = 0.0;
        p.sig[0] = 0.5;
        let a = vec![1.0; 8];
        let wn_r = [0.1, 0.2];
        let cg = [9.8, 4.9];
        let (s, d) = sic2(&p, &a, 100.0, 0.5, 1000.0, &cg, &wn_r, &wn_r, &cg, &[1.0, 1.0], Some(0.5), false).unwrap();
        for ith in 0..4 {
            assert!((d[ith] - (-0.00013527749258468682)).abs() < 1e-15);
            assert_eq!(s[ith], d[ith]);
        }
    }

    #[test]
    fn sic2_turbulent_by_hand() {
        // one frequency band of elevation variance 0.25 m²: AORB = 1 m,
        // UORB = SIG m/s, Re >> IC2REYNOLDS so that PTURB = 1
        let p = params(36, 36);
        let (wn, cg) = deep(&p);
        let ik = 15;
        let nth = p.nth;
        let mut a = vec![0f64; p.nk * nth];
        let a0 = 0.25 / (nth as f64 * p.dden[ik] / cg[ik]);
        for ith in 0..nth {
            a[ith + ik * nth] = a0;
        }
        let r = vec![1.0; p.nk];
        let (_, d) = sic2(&p, &a, 4000.0, 0.5, 1000.0, &cg, &wn, &wn, &cg, &r, Some(0.5), false).unwrap();
        let sg = p.sig[ik];
        let (aorb, uorb) = (1.0, sg);
        let re = uorb * aorb / 1.83e-6;
        // XI = (log10(AORB / IC2ROUGH) - ABMIN) / DELAB, about 4 / DELAB
        // (IC2ROUGH is 0.001 rounded to single precision)
        let xi = ((1.0 / p.ic2pars[2]).log10() + 1.0) / p.delab;
        let ind = xi.floor() as usize;
        let w = xi - ind as f64;
        let fw = p.fwtable[ind] * (1.0 - w) + p.fwtable[ind + 1] * w;
        let smooth = 0.5 * ((re - 1.5e5) / 2e5).tanh();
        let (pt, pv) = (0.5 + smooth, 0.5 - smooth);
        for jk in [0usize, ik, 35] {
            let s = p.sig[jk];
            let dturb = 1.0 * fw * uorb / GRAV;
            let dvisc = 2.0 * wn[jk] * (1.83e-6 * s / 2.0).sqrt();
            let want = -(pt * (dturb * s * s).max(dvisc) + pv * dvisc);
            assert!((d[jk * nth] - want).abs() <= 1e-9 * want.abs(), "{jk} {} {want}", d[jk * nth]);
        }
    }

    #[test]
    fn rp_dispersion_thin_ice_is_open_water() {
        let p = params(36, 36);
        let (wn_i, damp, cg_i) = rpwnice(&p, 0.001);
        for ik in 0..p.nk {
            let k0 = p.sig[ik] * p.sig[ik] / GRAV;
            assert!((wn_i[ik] / k0 - 1.0).abs() < 0.01, "{ik} {} {k0}", wn_i[ik]);
            assert!(damp[ik].is_finite() && cg_i[ik] > 0.0);
        }
    }

    #[test]
    fn not_ported_options_fail() {
        let mut p = params(4, 4);
        p.ic2pars[0] = 1.0;
        let (wn, cg) = deep(&p);
        let a = vec![1.0; 16];
        assert!(sic2(&p, &a, 100.0, 0.5, 1000.0, &cg, &wn, &wn, &cg, &[1.0; 4], Some(0.5), false).is_err());
        let mut p = params(4, 4);
        p.iicedisp = true;
        let inp = IceIn { ice: 0.5, iceh: 0.5, icedmax: 1000.0, depth: 100.0, ice_read: true, icei: 0.5, southern_rlg: false };
        let mut spec = a.clone();
        let mut icef = 1000.0;
        assert!(ice_after_loop(&p, &mut spec, &inp, &mut icef, &wn, &cg, 600.0).is_err());
    }
}
