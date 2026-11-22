//! Spectral grid constants of WW3 (`W3GRID` section 5, `W3IOGR`) and the
//! explicit dispersion relation `WAVNU3` used by `W3INIT` and `W3ULEV`.
//!
//! Frequencies `SIG(ik) = 2π FR1 XFR^(ik-1)` for `ik = 0 .. NK+1` (index 0
//! and NK+1 are the ghost bins of the frequency advection); everything is
//! computed in single precision as WW3 does.

use std::f32::consts::PI;

pub const GRAV: f32 = 9.806;
pub const RADIUS: f32 = 4.0e7 / (2.0 * PI);
pub const DERA: f32 = PI / 180.0;
/// Power of the high-frequency tail (`FACHF`, ST4 default).
pub const FACHF: f32 = 5.0;

pub struct Spec {
    pub nk: usize,
    pub nth: usize,
    pub xfr: f32,
    pub dth: f32,
    /// `SIG(0 .. NK+1)`, `DSIP(0 .. NK+1)` (rad/s).
    pub sig: Vec<f32>,
    pub dsip: Vec<f32>,
    /// `DSII(1 .. NK)` and `DDEN(1 .. NK)` stored at index `ik - 1`.
    pub dsii: Vec<f32>,
    pub dden: Vec<f32>,
    pub ecos: Vec<f32>,
    pub esin: Vec<f32>,
    pub es2: Vec<f32>,
    pub ec2: Vec<f32>,
    pub esc: Vec<f32>,
    /// `XFR^(-FACHF-2)`: tail factor of the action spectrum per bin.
    pub fachfa: f32,
    /// `FTE = 0.25 SIG(NK) DTH SIG(NK)`: energy tail of `W3OUTG`.
    pub fte: f32,
}

impl Spec {
    pub fn new(nk: usize, nth: usize, fr1: f32, xfr: f32, thoff: f32) -> Spec {
        let tpi = 2.0 * PI;
        let xfr = xfr.max(1.00001);
        let dth = tpi / nth as f32;
        let mut ecos = vec![0f32; nth];
        let mut esin = vec![0f32; nth];
        for ith in 0..nth {
            let th = dth * (thoff + ith as f32);
            let (mut s, mut c) = (th.sin(), th.cos());
            if s.abs() < 1e-5 {
                s = 0.0;
                c = if c > 0.5 { 1.0 } else { -1.0 };
            }
            if c.abs() < 1e-5 {
                c = 0.0;
                s = if s > 0.5 { 1.0 } else { -1.0 };
            }
            ecos[ith] = c;
            esin[ith] = s;
        }
        let es2 = esin.iter().map(|s| s * s).collect();
        let ec2 = ecos.iter().map(|c| c * c).collect();
        let esc = esin.iter().zip(&ecos).map(|(s, c)| s * c).collect();
        let mut sigma = fr1 * tpi / (xfr * xfr);
        let sxfr = 0.5 * (xfr - 1.0 / xfr);
        let mut sig = vec![0f32; nk + 2];
        let mut dsip = vec![0f32; nk + 2];
        for ik in 0..nk + 2 {
            sigma *= xfr;
            sig[ik] = sigma;
            dsip[ik] = sigma * sxfr;
        }
        let mut dsii = vec![0f32; nk];
        dsii[0] = 0.5 * sig[1] * (xfr - 1.0);
        for ik in 2..nk {
            dsii[ik - 1] = dsip[ik];
        }
        dsii[nk - 1] = 0.5 * sig[nk] * (xfr - 1.0) / xfr;
        let dden = (0..nk).map(|j| dth * dsii[j] * sig[j + 1]).collect();
        let fachfa = xfr.powf(-FACHF - 2.0);
        let fte = 0.25 * sig[nk] * dth * sig[nk];
        Spec { nk, nth, xfr, dth, sig, dsip, dsii, dden, ecos, esin, es2, ec2, esc, fachfa, fte }
    }
}

/// `WAVNU3`: wavenumber and group speed of radian frequency `si` at depth `h`.
pub fn wavnu3(si: f32, h: f32) -> (f32, f32) {
    let kdmax = 20.0f32;
    let zpi = 2.0 * PI;
    let tp = si / zpi;
    let kh0 = zpi * zpi * h / GRAV * tp * tp;
    let tmp = 1.55 + 1.3 * kh0 + 0.216 * kh0 * kh0;
    let kh = kh0 * (1.0 + kh0.powf(1.09) * 1.0 / tmp.min(kdmax).exp()) / kh0.min(kdmax).tanh().sqrt();
    let k = kh / h;
    let cg = 0.5 * (1.0 + (2.0 * kh / (2.0 * kh).min(kdmax).sinh())) * si / k;
    (k, cg)
}

/// Wavenumbers and group speeds of a node for `ik = 0 .. NK+1`, as `W3INIT`
/// and `W3ULEV` set them: at the water depth `max(DMIN, dw)`, unless the
/// lowest frequency is in deep water (`k d >= 4`) at the bathymetric depth
/// `max(DMIN, -zb)`, in which case that depth is used and never changes.
pub fn node_wavenumbers(sp: &Spec, dw: f32, zb: f32, dmin: f32, wn: &mut [f32], cg: &mut [f32]) -> bool {
    let dbat = dmin.max(-zb);
    let (kb, _) = wavnu3(sp.sig[1], dbat);
    let deep = kb * dbat >= 4.0;
    let depth = if deep { dbat } else { dmin.max(dw) };
    for ik in 0..sp.nk + 2 {
        let (k, c) = wavnu3(sp.sig[ik], depth);
        wn[ik] = k;
        cg[ik] = c;
    }
    deep
}
