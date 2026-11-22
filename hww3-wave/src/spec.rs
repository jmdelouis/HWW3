//! Spectral grid constants of WW3 (`W3GRID` section 5, `W3IOGR`) and the
//! explicit dispersion relation `WAVNU3` used by `W3INIT` and `W3ULEV`.
//!
//! Frequencies `SIG(ik) = 2π FR1 XFR^(ik-1)` for `ik = 0 .. NK+1` (index 0
//! and NK+1 are the ghost bins of the frequency advection); everything is
//! computed in single precision as WW3 does.

use std::f32::consts::PI;

/// Acceleration of gravity (m/s², WW3 `GRAV`).
pub const GRAV: f32 = 9.806;
/// Earth radius (m, WW3 `RADIUS`).
pub const RADIUS: f32 = 4.0e7 / (2.0 * PI);
/// Degrees to radians (WW3 `DERA`).
pub const DERA: f32 = PI / 180.0;
/// Power of the high-frequency tail (`FACHF`, ST4 default).
pub const FACHF: f32 = 5.0;

/// Spectral grid of WW3: frequencies, directions and the bin widths used
/// by the propagation and the outputs.
pub struct Spec {
    /// Number of frequencies (`NK`).
    pub nk: usize,
    /// Number of directions (`NTH`).
    pub nth: usize,
    /// Frequency increment factor (`XFR`).
    pub xfr: f32,
    /// Direction increment (rad, `DTH`).
    pub dth: f32,
    /// `SIG(0 .. NK+1)` (rad/s).
    pub sig: Vec<f32>,
    /// `DSIP(0 .. NK+1)`: frequency bandwidths (rad/s) of the propagation.
    pub dsip: Vec<f32>,
    /// `DSII(1 .. NK)` stored at index `ik - 1`: bandwidths of the integrals.
    pub dsii: Vec<f32>,
    /// `DDEN(1 .. NK)` stored at index `ik - 1`: `DTH DSII SIG`.
    pub dden: Vec<f32>,
    /// `cos θ` (`ECOS`, exactly 0 or ±1 on the axes).
    pub ecos: Vec<f32>,
    /// `sin θ` (`ESIN`, exactly 0 or ±1 on the axes).
    pub esin: Vec<f32>,
    /// `sin² θ` (`ES2`).
    pub es2: Vec<f32>,
    /// `cos² θ` (`EC2`).
    pub ec2: Vec<f32>,
    /// `sin θ cos θ` (`ESC`).
    pub esc: Vec<f32>,
    /// `XFR^(-FACHF-2)`: tail factor of the action spectrum per bin.
    pub fachfa: f32,
    /// `FTE = 0.25 SIG(NK) DTH SIG(NK)`: energy tail of `W3OUTG`.
    pub fte: f32,
}

impl Spec {
    /// Spectral grid of `nk` frequencies from `fr1` (Hz) with factor `xfr`,
    /// and `nth` directions offset by `thoff` (fraction of `DTH`), as
    /// `W3GRID` builds it.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_grid() {
        let sp = Spec::new(32, 36, 0.0373, 1.1, 0.0);
        assert_eq!((sp.sig.len(), sp.dsii.len()), (34, 32));
        assert!((sp.sig[1] / (2.0 * PI * 0.0373) - 1.0).abs() < 1e-6);
        for ik in 0..33 {
            assert!((sp.sig[ik + 1] / sp.sig[ik] / 1.1 - 1.0).abs() < 1e-6);
            assert!((sp.dsip[ik] / (sp.sig[ik] * 0.5 * (1.1 - 1.0 / 1.1)) - 1.0).abs() < 1e-6);
        }
        // the inner bandwidths tile the band between the two end half-bins
        let inner: f32 = sp.dsii[1..31].iter().sum();
        let band = 0.5 * (sp.sig[31] + sp.sig[32]) - 0.5 * (sp.sig[1] + sp.sig[2]);
        assert!((inner / band - 1.0).abs() < 1e-4);
        for k in 0..36 {
            assert!((sp.ecos[k].powi(2) + sp.esin[k].powi(2) - 1.0).abs() < 1e-6);
            assert_eq!(sp.esc[k], sp.esin[k] * sp.ecos[k]);
        }
        // directions on the axes are exact
        assert_eq!((sp.ecos[0], sp.esin[0]), (1.0, 0.0));
        assert_eq!((sp.ecos[9], sp.esin[9]), (0.0, 1.0));
        assert_eq!((sp.ecos[18], sp.esin[18]), (-1.0, 0.0));
        assert_eq!((sp.ecos[27], sp.esin[27]), (0.0, -1.0));
    }

    #[test]
    fn wavnu3_follows_the_dispersion_relation() {
        // WAVNU3 is an explicit approximation: within 0.1 % of the exact
        // wavenumber and group speed
        for f in [0.0373f32, 0.06, 0.1, 0.2, 0.5, 0.7] {
            for h in [1.0f32, 5.0, 20.0, 100.0, 1000.0, 5000.0] {
                let si = 2.0 * PI * f;
                let (k, cg) = wavnu3(si, h);
                let r = si * si / (GRAV * k * (k * h).tanh());
                assert!((r - 1.0).abs() < 2e-3, "f {f} h {h}: ratio {r}");
                let exact = crate::geom::group_speed(f as f64, h as f64) as f32;
                assert!((cg / exact - 1.0).abs() < 1e-3, "f {f} h {h}: cg {cg} against {exact}");
            }
        }
    }

    #[test]
    fn node_wavenumbers_keep_deep_nodes_at_the_bathymetric_depth() {
        let sp = Spec::new(32, 36, 0.0373, 1.1, 0.0);
        let (mut wn, mut cg) = (vec![0f32; 34], vec![0f32; 34]);
        // deep: the water level does not matter
        assert!(node_wavenumbers(&sp, 3.0, -4000.0, 1.0, &mut wn, &mut cg));
        assert_eq!(wn[5], wavnu3(sp.sig[5], 4000.0).0);
        // shallow: the water depth is used, never below DMIN
        assert!(!node_wavenumbers(&sp, 12.0, -10.0, 1.0, &mut wn, &mut cg));
        assert_eq!(wn[5], wavnu3(sp.sig[5], 12.0).0);
        assert!(!node_wavenumbers(&sp, 0.2, -10.0, 1.0, &mut wn, &mut cg));
        assert_eq!(cg[0], wavnu3(sp.sig[0], 1.0).1);
    }
}
