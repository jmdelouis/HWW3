//! CPU transcription, in double precision, of the WW3 reflection source
//! term `REF1` (`W3SREF` of `w3ref1md.F90`, WW3 7.14) and of what
//! `ww3_grid` prepares for it on the structured grids (`RECT` and `CURV`).
//!
//! Ported:
//! - `W3SREF` without the `IG1` switch (no infragravity waves) and with
//!   the structured-grid branch only ([`sref`]): reflection by the resolved
//!   shoreline (`REFLC(1) > 0`), and the diffuse reflection by sub-grid
//!   obstructions (`REFSUBGRID`, `TRNX`/`TRNY < 1`) and icebergs
//!   (`REFICEBERG`, `BERG > 0`);
//! - the namelist `&REF1` and the array `REFPARS(1..10)` of `W3GRID`
//!   ([`Ref1Params::from_namelists`]);
//! - `W3SETREF` of `w3gdatmd.F90` (the shoreline geometry `REFLC(1:4)`,
//!   `REFLD(1:6)` from the neighbours in `MAPSTA`, [`reflection_geometry`]);
//! - the grid derivatives `DXDP`, `DYDP`, `DXDQ`, `DYDQ` and the scale
//!   factors `HPFAC`, `HQFAC` of `W3GNTX` (`W3CGDM` of `w3gsrumd.F90`,
//!   finite differences of order 4 in a polar stereographic projection,
//!   [`grid_metric`]), which give the shoreline angles of `W3SETREF` and the
//!   cell sizes `DELX`, `DELY` of `W3SREF` on curvilinear grids;
//! - the obstruction file (`OBST%FILENAME`) read as `ww3_grid` reads it,
//!   giving `TRNX = 1 - OBSX`, `TRNY = 1 - OBSY` ([`read_obstructions`]).
//!
//! Not ported: the unstructured-grid branch of `W3SREF` (`IOBPD`,
//! `REFUNSTSOURCE`), the `IG1` free infragravity waves, the slope map
//! (`REFMAP > 0`, refused by [`Ref1Params::from_namelists`]), the
//! tripolar closure, and the update of `W3SETREF` by `W3ULEV` when the
//! water level dries points.
//!
//! In `W3SRCE` (section 12, after the integration of the other terms and
//! the ice terms) `W3SREF` is called once per global step `DTG` when
//! `REFLEC(1) > 0`, `REFLEC(2) > 0` or `REFLEC(4) > 0` and `BERG > 0`
//! ([`is_called`]), with `REFLEC = REFLC(:, ISEA)` except
//! `REFLEC(4) = BERG REFLC(4)` ([`reflec`]), and the spectrum becomes
//! `SPEC + DTG VREF` (the special case of unstructured grids is not
//! ported).
//!
//! Spectra are `A(ISPEC)` with `ISPEC = ITH + (IK-1) NTH` stored at
//! `ith + ik nth` (0-based), as in [`crate::sources`]. The comments
//! `Fortran:` mark the oddities of WW3 kept on purpose.

use crate::grid::Spectrum;
use crate::structgrid::{read_array, GridType, StructGrid};
use crate::ww3grid::Namelists;
use anyhow::{bail, Context, Result};
use std::path::Path;

/// `PI` of WW3's `constants.F90`.
const PI: f64 = std::f64::consts::PI;
/// `TPI = 2 PI`.
const TPI: f64 = 2.0 * PI;
/// `TPIINV = 1 / TPI`.
const TPIINV: f64 = 1.0 / TPI;
/// `DERA`: degrees to radians.
const DERA: f64 = PI / 180.0;
/// `RADIUS` of the Earth in `constants.F90` (m).
const RADIUS: f64 = 4.0e7 * TPIINV;
/// `GRAV` (m/s2).
const GRAV: f64 = 9.806;
/// `REARTH` of `w3gsrumd.F90` (without the `COAMPS` switch), m.
const REARTH: f64 = 4.0e7 / (2.0 * PI);
/// `NFD` of `W3GNTX`: order of the finite differences of the grid metric.
const NFD: usize = 4;

/// Parameters of `REF1`: `REFPARS(1..10)` as `W3GRID` sets them from the
/// namelist `&REF1`, and the spectral tables `W3SREF` reads.
#[derive(Clone, Debug)]
pub struct Ref1Params {
    /// `REFPARS(1..10)` at `i - 1`: `REFCOAST`, `REFSUBGRID`,
    /// `REFUNSTSOURCE`, `REFICEBERG`, unused (0), `REFFREQ`, `REFSLOPE`,
    /// `REFCOSP_STRAIGHT`, `REFRMAX`, `REFFREQPOW`.
    pub refpars: [f64; 10],
    /// `REFMAP` (0: the beach slope `REFLC(3)` is `REFSLOPE` everywhere).
    pub refmap: f64,
    /// Number of frequencies.
    pub nk: usize,
    /// Number of directions.
    pub nth: usize,
    /// `SIG(0..NK+1)` (rad/s).
    pub sig: Vec<f64>,
    /// `DDEN(1..NK)` at `ik - 1`.
    pub dden: Vec<f64>,
    /// `ECOS(1..NTH)` at `ith - 1`, with WW3's rounding to 0 and ±1.
    pub ecos: Vec<f64>,
    /// `ESIN(1..NTH)` at `ith - 1`.
    pub esin: Vec<f64>,
}

impl Ref1Params {
    /// `REFPARS` from the namelist group `REF1` (keys in upper case) with
    /// the defaults of `W3GRID` (`REFCOAST = 0`, `REFFREQ = 0`,
    /// `REFSLOPE = 0.22`, `REFMAP = 0`, `REFSUBGRID = 0`, `REFICEBERG = 0`,
    /// `REFCOSP_STRAIGHT = 4`, `REFRMAX = 1`, `REFFREQPOW = 2`,
    /// `REFUNSTSOURCE = 0`; `REFMAPD` is read but unused), and the spectral
    /// tables of `spec` as `W3GRID` computes them.
    pub fn from_namelists(nml: &Namelists, spec: &Spectrum) -> Result<Ref1Params> {
        let f = |k: &str, d: f64| nml.f("REF1", k, d);
        let refmap = f("REFMAP", 0.0)?;
        if refmap != 0.0 {
            bail!("&REF1 REFMAP = {refmap}: the slope map (REFMAP > 0) is not ported");
        }
        let mut refpars = [0.0; 10];
        refpars[0] = f("REFCOAST", 0.0)?;
        refpars[1] = f("REFSUBGRID", 0.0)?;
        refpars[2] = f("REFUNSTSOURCE", 0.0)?;
        refpars[3] = f("REFICEBERG", 0.0)?;
        // Fortran: REFPARS(5) is never set (0 from the allocation).
        refpars[5] = f("REFFREQ", 0.0)?;
        refpars[6] = f("REFSLOPE", 0.22)?;
        refpars[7] = f("REFCOSP_STRAIGHT", 4.0)?;
        refpars[8] = f("REFRMAX", 1.0)?;
        refpars[9] = f("REFFREQPOW", 2.0)?;
        Ok(Ref1Params::with_refpars(refpars, spec))
    }

    /// Parameters from `REFPARS` (with `REFMAP = 0`) and the spectral
    /// tables of `spec`.
    pub fn with_refpars(refpars: [f64; 10], spec: &Spectrum) -> Ref1Params {
        let (nk, nth) = (spec.nk, spec.nth);
        let dth = TPI / nth as f64;
        let rth0 = spec.thoff.clamp(-0.5, 0.5);
        let (mut ecos, mut esin) = (vec![0.0; nth], vec![0.0; nth]);
        for ith in 0..nth {
            let th = dth * (rth0 + ith as f64);
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
        let xfr = spec.xfr;
        let mut sigma = spec.freq1 * TPI / (xfr * xfr);
        let sxfr = 0.5 * (xfr - 1.0 / xfr);
        let (mut sig, mut dsip) = (vec![0.0; nk + 2], vec![0.0; nk + 2]);
        for ik in 0..nk + 2 {
            sigma *= xfr;
            sig[ik] = sigma;
            dsip[ik] = sigma * sxfr;
        }
        let mut dsii = vec![0.0; nk + 1];
        dsii[1] = 0.5 * sig[1] * (xfr - 1.0);
        for ik in 2..nk {
            dsii[ik] = dsip[ik];
        }
        dsii[nk] = 0.5 * sig[nk] * (xfr - 1.0) / xfr;
        let dden = (1..=nk).map(|ik| dth * dsii[ik] * sig[ik]).collect();
        Ref1Params { refpars, refmap: 0.0, nk, nth, sig, dden, ecos, esin }
    }
}

/// Sub-grid transparencies `TRNX`, `TRNY` of every grid point (`iy nx +
/// ix`), `1 - OBSX` and `1 - OBSY` of the obstruction file.
#[derive(Clone, Debug)]
pub struct Transparencies {
    /// `TRNX(IY, IX)` at `iy nx + ix`.
    pub trnx: Vec<f64>,
    /// `TRNY(IY, IX)` at `iy nx + ix`.
    pub trny: Vec<f64>,
}

/// `FLAGTR` of the namelist group `MISC` (default 0, values outside 0..6
/// set to 0 as `W3GRID` does; the reduction by 2 of the `IC0` switch when
/// `CICE0 = CICEN` is not applied).
pub fn flagtr(nml: &Namelists) -> Result<i64> {
    let f = nml.f("MISC", "FLAGTR", 0.0)?.round() as i64;
    Ok(if (0..=6).contains(&f) { f } else { 0 })
}

/// The transparencies of a structured grid as `ww3_grid` sets them: none
/// (`Ok(None)`) when `flagtr = 0`; otherwise `OBSX` then `OBSY` read from
/// `OBST%FILENAME` (in `data`) of the group `OBST_NML` of the grid
/// namelist, times `OBST%SF` (1 when below 1e-7 in absolute value), in the
/// layout `OBST%IDLA` / `OBST%IDFM` (default 1), clipped to [0, 1], and
/// `TRNX = 1 - OBSX`, `TRNY = 1 - OBSY`.
pub fn read_obstructions(grid_nml: &Namelists, flagtr: i64, data: &Path, nx: usize, ny: usize) -> Result<Option<Transparencies>> {
    if flagtr == 0 {
        return Ok(None);
    }
    let file = grid_nml.s("OBST_NML", "OBST%FILENAME").unwrap_or_default();
    if file.is_empty() {
        bail!("FLAGTR = {flagtr} without OBST%FILENAME");
    }
    let mut vsc = grid_nml.f("OBST_NML", "OBST%SF", 1.0)?;
    if vsc.abs() < 1e-7 {
        vsc = 1.0;
    }
    let mut idla = grid_nml.f("OBST_NML", "OBST%IDLA", 1.0)? as i64;
    if !(1..=4).contains(&idla) {
        idla = 1;
    }
    let mut idfm = grid_nml.f("OBST_NML", "OBST%IDFM", 1.0)? as i64;
    if !(1..=3).contains(&idfm) {
        idfm = 1;
    }
    let path = data.join(&file);
    // both arrays one after the other, in file order
    let raw = read_array(&path, nx, 2 * ny, 1, idfm, false).with_context(|| format!("obstructions {}", path.display()))?;
    let n = nx * ny;
    let field = |b: &[f64]| -> Vec<f64> {
        let mut w = vec![0.0; n];
        for iy in 0..ny {
            let src = if idla >= 3 { ny - 1 - iy } else { iy };
            for ix in 0..nx {
                let o = (b[src * nx + ix] * vsc).clamp(0.0, 1.0);
                w[iy * nx + ix] = 1.0 - o;
            }
        }
        w
    };
    Ok(Some(Transparencies { trnx: field(&raw[..n]), trny: field(&raw[n..2 * n]) }))
}

/// Grid derivatives and scale factors of `W3GNTX` at every point
/// (`iy nx + ix`), in degrees per grid index (`SPHERE = .FALSE.`: no
/// spherical metric, so that `HPFAC` has no `cos(latitude)` factor).
#[derive(Clone, Debug)]
pub struct Metric {
    /// `DXDP`: d(longitude)/d(ix).
    pub dxdp: Vec<f64>,
    /// `DYDP`: d(latitude)/d(ix).
    pub dydp: Vec<f64>,
    /// `DXDQ`: d(longitude)/d(iy).
    pub dxdq: Vec<f64>,
    /// `DYDQ`: d(latitude)/d(iy).
    pub dydq: Vec<f64>,
    /// `HPFAC = sqrt(max(DXDP² + DYDP², 1e-15))`.
    pub hpfac: Vec<f64>,
    /// `HQFAC = sqrt(max(DXDQ² + DYDQ², 1e-15))`.
    pub hqfac: Vec<f64>,
}

/// Finite-difference weights of `W3FDWT` (Fornberg 1998) for the first
/// derivative at 0 on the nodes `x`.
fn fdwt(x: &[f64]) -> Vec<f64> {
    let n = x.len() - 1;
    let m = 1usize;
    let z = 0.0;
    let mut c = vec![[0.0f64; 2]; n + 1];
    let mut c1 = 1.0;
    let mut c4 = x[0] - z;
    c[0][0] = 1.0;
    for i in 1..=n {
        let mn = i.min(m);
        let mut c2 = 1.0;
        let c5 = c4;
        c4 = x[i] - z;
        for j in 0..i {
            let c3 = x[i] - x[j];
            c2 *= c3;
            if j == i - 1 {
                for k in (1..=mn).rev() {
                    c[i][k] = c1 * (k as f64 * c[i - 1][k - 1] - c5 * c[i - 1][k]) / c2;
                }
                c[i][0] = -c1 * c5 * c[i - 1][0] / c2;
            }
            for k in (1..=mn).rev() {
                c[j][k] = (c4 * c[j][k] - k as f64 * c[j][k - 1]) / c3;
            }
            c[j][0] = c4 * c[j][0] / c3;
        }
        c1 = c2;
    }
    c.iter().map(|w| w[1]).collect()
}

/// `W3SPLX`: stereographic projection centred at `(lam0, phi0)` with the
/// standard circle at `c0` degrees from the centre.
fn splx(lam0: f64, phi0: f64, c0: f64, lam: f64, phi: f64) -> (f64, f64) {
    let d2r = PI / 180.0;
    let clam = ((lam - lam0) * d2r).cos();
    let slam = ((lam - lam0) * d2r).sin();
    let cphi0 = (phi0 * d2r).cos();
    let cphi = (phi * d2r).cos();
    let sphi0 = (phi0 * d2r).sin();
    let sphi = (phi * d2r).sin();
    let k0 = (0.5 * c0 * d2r).cos().powi(2);
    let k = 2.0 * k0 * REARTH / (1.0 + sphi0 * sphi + cphi0 * cphi * clam);
    (k * cphi * slam, k * (cphi0 * sphi - sphi0 * cphi * clam))
}

/// `SPDDP` / `SPDDQ`: derivatives in the polar stereographic plane turned
/// into derivatives of longitude and latitude (degrees).
fn spdd(lam0: f64, c0: f64, ihem: i32, lam: f64, phi: f64, dxd: f64, dyd: f64) -> (f64, f64) {
    let d2r = PI / 180.0;
    let k0 = (0.5 * c0 * d2r).cos().powi(2);
    let mu = (lam - lam0) * d2r;
    let a = if ihem >= 0 { 1.0 } else { -1.0 };
    let mut nu = PI / 4.0 - a * 0.5 * phi * d2r;
    let m = nu.abs().max(1e-6);
    nu = if nu >= 0.0 { m } else { -m };
    let fac = (1.0 / d2r) * 0.5 / REARTH / k0;
    let (cosmu, sinmu) = (mu.cos(), mu.sin());
    let cosnu2 = nu.cos().powi(2);
    let cotnu = 1.0 / nu.tan();
    let dlamdx = fac * cotnu * cosmu;
    let dlamdy = a * fac * cotnu * sinmu;
    let dphidx = -a * 2.0 * fac * cosnu2 * sinmu;
    let dphidy = 2.0 * fac * cosnu2 * cosmu;
    (dxd * dlamdx + dyd * dlamdy, dxd * dphidx + dyd * dphidy)
}

/// `DXYDP` (`along_p`) or `DXYDQ` at point `(ix, iy)` with a stencil of
/// `n + 1` points (`ICLO_NONE`, or `ICLO_SMPL` closed in `ix`; no mask).
fn dxyd(x: &[f32], y: &[f32], nx: usize, ny: usize, closed: bool, ix: usize, iy: usize, along_p: bool, n: usize) -> (f64, f64) {
    let (pos, len, periodic) = if along_p { (ix, nx, closed) } else { (iy, ny, false) };
    let i = if periodic {
        n / 2
    } else if pos < n / 2 {
        pos
    } else if len - 1 - pos < n / 2 {
        n + pos - (len - 1)
    } else {
        n / 2
    };
    let nodes: Vec<f64> = (0..=n).map(|j| j as f64 - i as f64).collect();
    let c = fdwt(&nodes);
    let (mut xp, mut yp) = (vec![0.0; n + 1], vec![0.0; n + 1]);
    for j in 0..=n {
        let mut k = pos as i64 + j as i64 - i as i64;
        if periodic {
            k = k.rem_euclid(len as i64);
        }
        let idx = if along_p { iy * nx + k as usize } else { k as usize * nx + ix };
        xp[j] = x[idx] as f64;
        yp[j] = y[idx] as f64;
    }
    let (x0, y0) = (x[iy * nx + ix] as f64, y[iy * nx + ix] as f64);
    let ihem = if yp.iter().cloned().fold(f64::NEG_INFINITY, f64::max) < 0.0 { -1 } else { 1 };
    let lon0 = 0.0;
    let lat0 = 90.0 * ihem as f64;
    let c0 = 90.0 - y0.abs();
    let (mut d1, mut d2) = (0.0, 0.0);
    for j in 0..=n {
        let (u, v) = splx(lon0, lat0, c0, xp[j], yp[j]);
        d1 += c[j] * u;
        d2 += c[j] * v;
    }
    spdd(lon0, c0, ihem, x0, y0, d1, d2)
}

/// The metric of `W3GNTX` (`W3CGDM` with `NFD = 4`, `SPHERE = .FALSE.`) at
/// the sea points of `g` (zero elsewhere). `RECT`: the constant spacings
/// (`DXDP = SX`, `DYDQ = SY`, the cross terms zero). `CURV`: finite
/// differences of order 4 (one-sided near the edges, periodic in `ix` with
/// `CLOS = 'SMPL'`) of the coordinates rounded to single precision (`REAL(XGRD)`)
/// in a polar stereographic projection, redone with order 2 where the
/// Jacobian `DXDP DYDQ - DXDQ DYDP` is negative; an error where it stays
/// negative (`W3CGDM ERROR`).
pub fn grid_metric(g: &StructGrid) -> Result<Metric> {
    let (nx, ny) = (g.nx, g.ny);
    let npt = nx * ny;
    let mut m = Metric {
        dxdp: vec![0.0; npt],
        dydp: vec![0.0; npt],
        dxdq: vec![0.0; npt],
        dydq: vec![0.0; npt],
        hpfac: vec![0.0; npt],
        hqfac: vec![0.0; npt],
    };
    let (sx, sy) = rect_spacing(g);
    let x: Vec<f32> = g.lon.iter().map(|&v| v as f32).collect();
    let y: Vec<f32> = g.lat.iter().map(|&v| v as f32).collect();
    for iy in 0..ny {
        for ix in 0..nx {
            let i = iy * nx + ix;
            if !g.is_sea(i) {
                continue;
            }
            let (mut dxdp, mut dydp, mut dxdq, mut dydq);
            match g.gtype {
                GridType::Rect => {
                    dxdp = sx;
                    dydp = 0.0;
                    dxdq = 0.0;
                    dydq = sy;
                }
                GridType::Curv => {
                    (dxdp, dydp) = dxyd(&x, &y, nx, ny, g.closed, ix, iy, true, NFD);
                    (dxdq, dydq) = dxyd(&x, &y, nx, ny, g.closed, ix, iy, false, NFD);
                }
            }
            let mut gsqr = dxdp * dydq - dxdq * dydp;
            if gsqr < 0.0 && NFD > 2 && g.gtype == GridType::Curv {
                (dxdp, dydp) = dxyd(&x, &y, nx, ny, g.closed, ix, iy, true, 2);
                (dxdq, dydq) = dxyd(&x, &y, nx, ny, g.closed, ix, iy, false, 2);
                gsqr = dxdp * dydq - dxdq * dydp;
            }
            if gsqr < 0.0 {
                bail!(
                    "{}: point ({ix}, {iy}): the coordinates do not define a right-handed system (W3CGDM): DXDP {dxdp} DYDP {dydp} DXDQ {dxdq} DYDQ {dydq}",
                    g.name
                );
            }
            m.dxdp[i] = dxdp;
            m.dydp[i] = dydp;
            m.dxdq[i] = dxdq;
            m.dydq[i] = dydq;
            m.hpfac[i] = (dxdp * dxdp + dydp * dydp).max(1e-15).sqrt();
            m.hqfac[i] = (dxdq * dxdq + dydq * dydq).max(1e-15).sqrt();
        }
    }
    Ok(m)
}

/// `SX`, `SY` of a `RECT` grid (degrees), from its first and last points.
fn rect_spacing(g: &StructGrid) -> (f64, f64) {
    let (nx, ny) = (g.nx, g.ny);
    ((g.lon[nx - 1] - g.lon[0]) / (nx - 1) as f64, (g.lat[(ny - 1) * nx] - g.lat[0]) / (ny - 1) as f64)
}

/// Everything `W3SREF` takes from the grid, per sea point in WW3's order
/// (`ISEA`: `ix` fastest then `iy`, `MAPSTA` 1 or 2).
#[derive(Clone, Debug)]
pub struct Ref1Geometry {
    /// Index `iy nx + ix` of each sea point (to map them to the mesh nodes
    /// through `Mesh::ij`).
    pub ij: Vec<u32>,
    /// `REFLC(1:4, ISEA)`: shoreline reflection coefficient (`REFCOAST` at
    /// coast points, else 0), sub-grid obstruction `max(1 - TRNX, 1 - TRNY)`
    /// (with `REFSUBGRID > 0`), beach slope (`REFSLOPE`), iceberg factor (1
    /// with `REFICEBERG > 0`).
    pub reflc: Vec<[f64; 4]>,
    /// `REFLD(1:6, ISEA)`: direction index (1-based) pointing into the
    /// coast, shoreline type (0 sharp corner, 1 mild corner, 2 straight),
    /// 1 when the reflection applies along `ix` (resp. `iy`), and the
    /// direction offsets of the grid axes `ix`, `iy` (0-based).
    pub refld: Vec<[i32; 6]>,
    /// `DELX` of `W3SREF` (m): `SX cos(lat) DERA RADIUS` (`RECT`) or
    /// `HPFAC DERA RADIUS` (`CURV`).
    pub delx: Vec<f64>,
    /// `DELY` of `W3SREF` (m): `SY DERA RADIUS` or `HQFAC DERA RADIUS`.
    pub dely: Vec<f64>,
    /// `TRNX(IY, IX)` at the run (1 without obstruction file).
    pub trnx: Vec<f64>,
    /// `TRNY(IY, IX)` at the run (1 without obstruction file).
    pub trny: Vec<f64>,
}

/// [`reflection_geometry_trn`] without obstruction file (`FLAGTR = 0`).
pub fn reflection_geometry(g: &StructGrid, p: &Ref1Params) -> Result<Ref1Geometry> {
    reflection_geometry_trn(g, p, None)
}

/// `REFLC`, `REFLD` of every sea point of `g` as `W3SETREF` computes them
/// in `ww3_grid` (`RECT` or `CURV`, spherical), with `DELX`, `DELY` and the
/// transparencies `W3SREF` gets at the run.
///
/// `W3SETREF` visits the points `2 <= IX <= NX-1`, `2 <= IY <= NY-1` only
/// (Fortran: also on grids closed in longitude, whose first and last columns
/// get no reflection, and without wrapping the neighbours). At a sea point
/// it sets `REFLC(2) = max(1 - TRNX, 1 - TRNY)` when `REFSUBGRID > 0`,
/// `REFLC(4) = 1` when `REFICEBERG > 0` and, when `REFCOAST > 0` and the
/// latitude is at most 85°, looks at the 8 neighbours (`8 MAPST2 + MAPSTA`:
/// 0 land, 1 sea, 2 boundary, 8 excluded) for a coast:
/// - a straight coast at `K` when the neighbours `K - 1`, `K`, `K + 1` are
///   land and `K + 4` is not (`REFLD(2) = 2`, or 1 when several `K`;
///   `REFLD(1)` from the mean of the directions `ANGLES(K)`);
/// - else a mild corner (`K`, `K + 1` land, `K + 4` not; `REFLD(2) = 1`);
/// - else a sharp corner (even `K` land, `K + 4` not; `REFLD(2) = 0`).
///
/// `ANGLES(K)` is the direction of neighbour `K` from the grid derivatives
/// ([`grid_metric`]) with `cos(lat)` on the longitude terms. `REFLC(3)` is
/// `REFSLOPE` at every sea point (`REFMAP = 0`).
///
/// `trn`: the transparencies of the obstruction file ([`read_obstructions`]);
/// without them (`FLAGTR = 0`) `ww3_grid` has `TRNX = TRNY = 0`, so that
/// `REFLC(2) = 1` at every visited sea point when `REFSUBGRID > 0`
/// (Fortran), while at the run `TRNX`, `TRNY` are not read from the model
/// definition (left unset by `W3IOGR`, 1 with the `UOST` switch): 1 here.
pub fn reflection_geometry_trn(g: &StructGrid, p: &Ref1Params, trn: Option<&Transparencies>) -> Result<Ref1Geometry> {
    let (nx, ny) = (g.nx, g.ny);
    let nth = p.nth as i64;
    let metric = grid_metric(g)?;
    let (sx, sy) = rect_spacing(g);
    let npt = nx * ny;
    // ISEA of each point
    let mut mapfs = vec![usize::MAX; npt];
    let mut ij = Vec::new();
    for i in 0..npt {
        if g.is_sea(i) {
            mapfs[i] = ij.len();
            ij.push(i as u32);
        }
    }
    let nsea = ij.len();
    let mut reflc = vec![[0.0, 0.0, p.refpars[6], 0.0]; nsea];
    let mut refld = vec![[0i32; 6]; nsea];
    // grid-time transparencies (TRNX = 0 when FLAGTR = 0)
    let tr_grid = |i: usize| -> (f64, f64) {
        match trn {
            Some(t) => (t.trnx[i], t.trny[i]),
            None => (0.0, 0.0),
        }
    };
    // 8 MAPST2 + MAPSTA as ww3_grid leaves them
    let neigh = |i: usize| -> i32 {
        match g.mapsta[i] {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => 8,
        }
    };
    let rref2 = p.refpars[1] > 0.0;
    let rref4 = p.refpars[3] > 0.0;
    for iy in 1..ny.saturating_sub(1) {
        for ix in 1..nx.saturating_sub(1) {
            let i = iy * nx + ix;
            // RREF(1) is set again at every point, and cleared near the pole
            let rref1 = p.refpars[0] > 0.0 && !(g.lat[i] > 85.0);
            if !g.is_sea(i) {
                continue;
            }
            let s = mapfs[i];
            if rref2 {
                let (tx, ty) = tr_grid(i);
                reflc[s][1] = (1.0 - tx).max(1.0 - ty);
            }
            if rref4 {
                reflc[s][3] = 1.0;
            }
            if !rref1 {
                continue;
            }
            reflc[s][0] = 0.0;
            refld[s] = [0; 6];
            //  3 2 1
            //  4 X 0
            //  5 6 7
            let at = |dx: i64, dy: i64| neigh(((iy as i64 + dy) as usize) * nx + (ix as i64 + dx) as usize);
            let n1 = [at(1, 0), at(1, 1), at(0, 1), at(-1, 1), at(-1, 0), at(-1, -1), at(0, -1), at(1, -1)];
            if n1.iter().all(|&v| v != 0) {
                continue;
            }
            let clat = (g.lat[i] * DERA).cos();
            let (dxdp, dydp, dxdq, dydq) = (metric.dxdp[i], metric.dydp[i], metric.dxdq[i], metric.dydq[i]);
            let mut angles = [0.0f64; 8];
            angles[0] = dydp.atan2(dxdp * clat);
            angles[1] = (dydp + dydq).atan2((dxdp + dxdq) * clat);
            angles[2] = dydq.atan2(dxdq * clat);
            angles[3] = (dydq - dydp).atan2((dxdq - dxdp) * clat);
            for k in 4..8 {
                angles[k] = angles[k - 4] + PI;
            }
            if n1[0] >= 1 && n1[4] >= 1 {
                refld[s][2] = 0;
            } else if n1[0] >= 1 || n1[4] >= 1 {
                refld[s][2] = 1;
            }
            // Fortran: .EQ.1 for neighbour 2 here, where REFLD(3) tests .GE.1
            if n1[2] == 1 && n1[6] >= 1 {
                refld[s][3] = 0;
            } else if n1[2] >= 1 || n1[6] >= 1 {
                refld[s][3] = 1;
            }
            let mut j = 0;
            refld[s][0] = 0;
            let (mut cosavg, mut sinavg) = (0.0f64, 0.0f64);
            refld[s][4] = ((nth + nint(angles[0] / TPI * nth as f64)) % nth) as i32;
            refld[s][5] = ((nth + nint((angles[2] / TPI - 0.25) * nth as f64)) % nth) as i32;
            for k in 0..8usize {
                if n1[k] == 0 && n1[(k + 7) % 8] == 0 && n1[(k + 1) % 8] == 0 && n1[(k + 4) % 8] != 0 {
                    reflc[s][0] = p.refpars[0];
                    refld[s][1] = 2;
                    cosavg += angles[k].cos();
                    sinavg += angles[k].sin();
                    j += 1;
                }
            }
            if j > 0 {
                if j > 1 {
                    refld[s][1] = 1;
                }
                let thavg = sinavg.atan2(cosavg);
                refld[s][0] = (1 + (nth + nint(thavg / TPI * nth as f64)) % nth) as i32;
            } else {
                // mild corners; the last K found wins
                for k in 0..8i64 {
                    let ku = k as usize;
                    if n1[ku] == 0 && n1[(ku + 1) % 8] == 0 && n1[(ku + 4) % 8] != 0 {
                        reflc[s][0] = p.refpars[0];
                        // Fortran: integer division
                        refld[s][0] = (1 + ((k * nth + (k + 1) * nth) / 16) % nth) as i32;
                        refld[s][1] = 1;
                    }
                }
                // sharp corners, not diagonals
                if reflc[s][0] <= 0.0 {
                    for k in (0..8i64).step_by(2) {
                        let ku = k as usize;
                        if n1[ku] == 0 && n1[(ku + 4) % 8] != 0 {
                            reflc[s][0] = p.refpars[0];
                            refld[s][0] = (1 + (k * nth) / 8) as i32;
                            refld[s][1] = 0;
                        }
                    }
                }
            }
        }
    }
    let mut delx = Vec::with_capacity(nsea);
    let mut dely = Vec::with_capacity(nsea);
    let mut trnx = Vec::with_capacity(nsea);
    let mut trny = Vec::with_capacity(nsea);
    for &i in &ij {
        let i = i as usize;
        match g.gtype {
            GridType::Rect => {
                delx.push(sx * (g.lat[i] * DERA).cos() * DERA * RADIUS);
                dely.push(sy * DERA * RADIUS);
            }
            GridType::Curv => {
                delx.push(metric.hpfac[i] * DERA * RADIUS);
                dely.push(metric.hqfac[i] * DERA * RADIUS);
            }
        }
        match trn {
            Some(t) => {
                trnx.push(t.trnx[i]);
                trny.push(t.trny[i]);
            }
            None => {
                trnx.push(1.0);
                trny.push(1.0);
            }
        }
    }
    Ok(Ref1Geometry { ij, reflc, refld, delx, dely, trnx, trny })
}

/// Fortran `NINT`: nearest integer, halves away from zero.
fn nint(x: f64) -> i64 {
    x.round() as i64
}

/// `REFLEC` of `W3WAVE`: `REFLC(:, ISEA)` with `REFLEC(4) = BERG REFLC(4)`.
pub fn reflec(reflc: &[f64; 4], berg: f64) -> [f64; 4] {
    [reflc[0], reflc[1], reflc[2], berg * reflc[3]]
}

/// Whether `W3SRCE` calls `W3SREF`: `REFLEC(1) > 0`, `REFLEC(2) > 0`, or
/// `REFLEC(4) > 0` and `BERG > 0`.
pub fn is_called(reflec: &[f64; 4], berg: f64) -> bool {
    reflec[0] > 0.0 || reflec[1] > 0.0 || (reflec[3] > 0.0 && berg > 0.0)
}

/// `W3SREF` on a structured grid (without `IG1`): the reflection source
/// term `VREF(NSPEC)` (action density per second) of the spectrum `a` at a
/// point with group velocities `cg(NK)`, reflection coefficients `reflec`
/// (`REFLEC` of `W3WAVE`, see [`reflec`]), shoreline codes `refld`, cell
/// sizes `delx`, `dely` (m) and transparencies `trnx`, `trny`, iceberg
/// damage `berg`. `W3SRCE` adds `DTG VREF` to the spectrum.
///
/// The arguments `WN`, `EMEAN`, `FMEAN`, `DEPTH`, `CX`, `CY`, `DT` of the
/// Fortran are unused on structured grids without `IG1` and left out.
///
/// Kept oddities of the Fortran: `FMEANA` (Hz) is bounded below by `SIG(1)`
/// (rad/s); the direction differences index `ECOS` as if `TH(1) = 0`; the
/// sub-grid and iceberg part uses `SIG(IK)` with `IK = NK + 1` (the value
/// of the loop index after the loops), takes the Miche factor with 0.007
/// instead of 0.07 and the iceberg factor `REFLEC(4)` as a slope, and drops
/// `REFLEC(4)` from `RAMP4` when `REFFREQ > 0`.
#[allow(clippy::too_many_arguments)]
pub fn sref(p: &Ref1Params, a: &[f64], cg: &[f64], reflc: &[f64; 4], refld: &[i32; 6], delx: f64, dely: f64, trnx: f64, trny: f64, berg: f64) -> Vec<f64> {
    let (nk, nth) = (p.nk, p.nth);
    let nthi = nth as i64;
    let refpars = &p.refpars;
    let sig = &p.sig;
    // ECOS(i), ESIN(i) with i 1-based
    let ecos = |i: i64| p.ecos[(i - 1) as usize];
    let esin = |i: i64| p.esin[(i - 1) as usize];
    let ai = |ith: i64, ik: usize| (ith - 1) as usize + (ik - 1) * nth;
    let mut emeana = 0.0;
    let mut fmeana = 0.0;
    let mut fmean2 = 0.0;
    for ik in 1..=nk {
        let mut eb = 0.0;
        for ith in 1..=nthi {
            eb += a[ai(ith, ik)];
        }
        eb = eb * p.dden[ik - 1] / cg[ik - 1];
        emeana += eb;
        fmean2 += eb / sig[ik].powi(2);
        fmeana += eb / sig[ik];
    }
    fmeana = TPIINV * (emeana / fmeana.max(1e-7));
    let _fmean2 = TPIINV * (emeana / fmean2.max(1e-7)).sqrt(); // used by IG1 only
    // Fortran: a frequency (Hz) bounded by a radian frequency
    fmeana = fmeana.max(sig[1]);
    let mut s = vec![0.0; nk * nth];
    let nrs = nint(refpars[7]) as i32;
    let mut ramp0;
    if refpars[5] > 0.0 {
        // Miche parameter for a beach slope REFLC(3)
        let slope = if reflc[2].is_nan() { 0.001 } else { reflc[2].max(0.001) };
        let michefac = 0.0001 * GRAV * GRAV * slope.powi(5) / (emeana.max(1e-4) * fmeana.max(0.001).powi(4));
        ramp0 = (0.07 * (michefac.log10() + 4.5) + 1.5 * michefac).max(0.005);
    } else {
        ramp0 = 1.0;
    }
    // 2. Shoreline reflection
    if reflc[0] > 0.0 {
        let fac1 = 1.0 / (0.5 * nth as f64);
        let fac2 = 1.57 / (0.5 * nth as f64);
        let fac3 = 2.0 / p.ecos.iter().map(|c| c.abs().powi(nrs)).sum::<f64>();
        let r1d = refld[0] as i64;
        let (rd3, rd4) = (refld[2] as f64, refld[3] as f64);
        for ik in 1..=nk {
            let ramp1 = if refpars[5] > 0.0 {
                let ramp = (0.75 * TPI * fmeana / sig[ik]).max(1.0).powf(refpars[9]) * ramp0;
                // RAMP2 = MIN(REFPARS(9), REFLC(2) RAMP) is computed here but unused
                refpars[8].min(reflc[0] * ramp)
            } else {
                ramp0 * reflc[0]
            };
            let cgk = cg[ik - 1];
            for ith in 1..=nthi {
                let r1 = ecos(1 + (ith - r1d).abs() % nthi);
                let r2 = ramp1 * a[ai(ith, ik)];
                if r1 > 0.0 && r2 > 0.0 {
                    for ith2 in 1..=nthi {
                        let isp = ai(ith2, ik);
                        let ith2x = 1 + (nthi + ith2 - refld[4] as i64 - 1) % nthi;
                        let ith2y = 1 + (nthi + ith2 - refld[5] as i64 - 1) % nthi;
                        let r3 = ecos(1 + (ith2 - r1d).abs() % nthi);
                        if r3 < 0.0 {
                            // specular direction: th3 = pi + 2 n - th1
                            let ith3 = 1 + (nthi / 2 + nthi + 2 * r1d - ith - 1) % nthi;
                            let r4 = ecos(1 + (ith2 - ith3).abs() % nthi);
                            if r4 > 0.0 {
                                let cx = (ecos(ith2x) / delx).abs();
                                let cy = (esin(ith2y) / dely).abs();
                                match refld[1] {
                                    0 => {
                                        s[isp] += rd3 * r2 * cgk * cx * fac1 + rd4 * r2 * cgk * cy * fac1;
                                    }
                                    1 => {
                                        s[isp] += rd3 * r2 * cgk * cx * r4.abs() * fac2 + rd4 * r2 * cgk * cy * r4.abs() * fac2;
                                    }
                                    2 => {
                                        s[isp] += rd3 * r2 * cgk * cx * r4.powi(nrs) * fac3 + rd4 * r2 * cgk * cy * r4.powi(nrs) * fac3;
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Diffuse reflection by sub-grid islands and icebergs
    if (refpars[1] > 0.0 && trnx + trny < 2.0) || (refpars[3] > 0.0 && berg > 0.0) {
        // Fortran: IK is NK + 1 here, the index after the loops above
        let sigk = sig[nk + 1];
        let (ramp2, ramp4);
        if refpars[5] > 0.0 {
            let ramp = (0.75 * TPI * fmeana / sigk).max(1.0).powf(refpars[9]) * ramp0;
            ramp2 = refpars[8].min(reflc[1] * ramp);
            // iceberg slope given by REFLC(4) (BERG times the iceberg factor)
            let slope = reflc[3].max(0.001);
            let michefac = 0.0001 * GRAV * GRAV * slope.powi(5) / (emeana.max(1e-4) * fmeana.max(0.001).powi(4));
            // Fortran: 0.007 here, 0.07 above
            ramp0 = (0.007 * (michefac.log10() + 4.5) + 1.5 * michefac).max(0.005);
            let ramp = (0.75 * TPI * fmeana / sigk).max(1.0).powf(refpars[9]) * ramp0;
            ramp4 = refpars[8].min(ramp);
        } else {
            ramp2 = ramp0 * reflc[1];
            ramp4 = ramp0 * reflc[3];
        }
        let r2x = ramp2 * refpars[1] * (1.0 - trnx).clamp(0.0, 1.0) + ramp4 * refpars[3] * (1.0 - (-berg * delx * 0.0001).exp()).clamp(0.0, 1.0);
        let r2y = ramp2 * refpars[1] * (1.0 - trny).clamp(0.0, 1.0) + ramp4 * refpars[3] * (1.0 - (-berg * dely * 0.0001).exp()).clamp(0.0, 1.0);
        let fac1 = 1.0 / (0.5 * nth as f64);
        for ik in 1..=nk {
            let cgk = cg[ik - 1];
            for ith in 1..=nthi {
                let r2 = a[ai(ith, ik)];
                if r2 > 0.0 {
                    for ith2 in 1..=nthi {
                        let r3 = ecos(1 + (nthi + ith2 - ith) % nthi);
                        if r3 < 0.0 {
                            s[ai(ith2, ik)] += cgk * r2x * r2 * (ecos(ith2) / delx).abs() * fac1 + cgk * r2y * r2 * (esin(ith2) / dely).abs() * fac1;
                        }
                    }
                }
            }
        }
    }
    s
}

/// `W3SREF` at sea point `isea` of a [`Ref1Geometry`], as `W3SRCE` calls
/// it: `None` when it is not called ([`is_called`]), else `VREF`.
pub fn sref_point(p: &Ref1Params, geo: &Ref1Geometry, isea: usize, a: &[f64], cg: &[f64], berg: f64) -> Option<Vec<f64>> {
    let r = reflec(&geo.reflc[isea], berg);
    if !is_called(&r, berg) {
        return None;
    }
    Some(sref(p, a, cg, &r, &geo.refld[isea], geo.delx[isea], geo.dely[isea], geo.trnx[isea], geo.trny[isea], berg))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(nk: usize, nth: usize) -> Spectrum {
        Spectrum { nk, nth, freq1: 0.0339, xfr: 1.1, thoff: 0.0, dmin: 2.5 }
    }

    const IRI: &str = "&REF1 REFCOAST = 0.05, REFCOSP_STRAIGHT = 4, REFFREQ = 1., REFICEBERG = 0.2,\n      REFMAP = 0., REFSLOPE = 0., REFSUBGRID = 0.1, REFRMAX = 0.5 /\n";

    /// A RECT grid of 12 × 9 points, 0.5° (from 10° W, 40° N), land for
    /// `ix >= 8`, edge rows excluded (not closed: edge columns too).
    fn coast_grid(gtype: GridType) -> StructGrid {
        let (nx, ny) = (12, 9);
        let mut lon = vec![];
        let mut lat = vec![];
        let mut mapsta = vec![];
        for iy in 0..ny {
            for ix in 0..nx {
                lon.push(-10.0 + 0.5 * ix as f64);
                lat.push(40.0 + 0.5 * iy as f64);
                mapsta.push(if ix >= 8 { 0 } else { 1 });
            }
        }
        let mut g = StructGrid {
            name: "T".into(),
            gtype,
            nx,
            ny,
            closed: false,
            lon,
            lat,
            zb: vec![-100.0; nx * ny],
            mapsta,
            spec: spec(6, 24),
        };
        g.exclude_edges();
        g
    }

    fn isea_of(geo: &Ref1Geometry, nx: usize, ix: usize, iy: usize) -> usize {
        geo.ij.iter().position(|&i| i as usize == iy * nx + ix).unwrap()
    }

    /// An isotropic-free test spectrum: energy only in direction `ith0`.
    fn spike(p: &Ref1Params, ith0: usize) -> (Vec<f64>, Vec<f64>) {
        let mut a = vec![0.0; p.nk * p.nth];
        for ik in 0..p.nk {
            a[ith0 + ik * p.nth] = 1.0 / (1.0 + ik as f64);
        }
        let cg: Vec<f64> = (0..p.nk).map(|ik| 9.806 / (2.0 * p.sig[ik + 1])).collect();
        (a, cg)
    }

    #[test]
    fn namelist_defaults_and_iri() {
        let sp = spec(36, 36);
        let p = Ref1Params::from_namelists(&Namelists::parse(IRI), &sp).unwrap();
        assert_eq!(p.refpars, [0.05, 0.1, 0.0, 0.2, 0.0, 1.0, 0.0, 4.0, 0.5, 2.0]);
        let d = Ref1Params::from_namelists(&Namelists::parse("&MISC FLAGTR = 4 /\n"), &sp).unwrap();
        assert_eq!(d.refpars, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22, 4.0, 1.0, 2.0]);
        assert_eq!(flagtr(&Namelists::parse("&MISC FLAGTR = 4 /\n")).unwrap(), 4);
        assert!(Ref1Params::from_namelists(&Namelists::parse("&REF1 REFMAP = 1. /"), &sp).is_err());
        assert_eq!(p.sig.len(), 38);
        assert!((p.sig[1] - 0.0339 * TPI).abs() < 1e-12);
        assert_eq!((p.ecos[0], p.ecos[9], p.esin[9], p.ecos[18]), (1.0, 0.0, 1.0, -1.0));
    }

    #[test]
    fn fornberg_weights() {
        let w = fdwt(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let e = [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0];
        for (a, b) in w.iter().zip(e) {
            assert!((a - b).abs() < 1e-14, "{w:?}");
        }
        let w = fdwt(&[0.0, 1.0, 2.0]);
        assert!((w[0] + 1.5).abs() < 1e-14 && (w[1] - 2.0).abs() < 1e-14 && (w[2] + 0.5).abs() < 1e-14);
    }

    #[test]
    fn straight_coast_rect_and_curv() {
        let sp = spec(6, 24);
        let p = Ref1Params::from_namelists(&Namelists::parse(IRI), &sp).unwrap();
        for gt in [GridType::Rect, GridType::Curv] {
            let g = coast_grid(gt);
            let geo = reflection_geometry(&g, &p).unwrap();
            assert_eq!(geo.ij.len(), g.counts()[1] + g.counts()[2]);
            // the point west of the coast: coast towards east (TH = 0, ITH 1)
            let s = isea_of(&geo, g.nx, 7, 4);
            assert_eq!(geo.refld[s], [1, 2, 1, 0, 0, 0], "{gt:?}");
            assert_eq!(geo.reflc[s], [0.05, 1.0, 0.0, 1.0]);
            // open sea: no shoreline reflection
            let o = isea_of(&geo, g.nx, 3, 4);
            assert_eq!(geo.reflc[o][0], 0.0);
            assert_eq!(geo.refld[o], [0; 6]);
            // edge rows/columns are excluded: no sea point at iy = 0
            assert!(geo.ij.iter().all(|&i| (i as usize) / g.nx >= 1));
            // cell sizes: 0.5° times cos(lat) and 0.5°
            let lat = g.lat[4 * g.nx + 7];
            let dx = 0.5 * (lat * DERA).cos() * DERA * RADIUS;
            let dy = 0.5 * DERA * RADIUS;
            let tol = if gt == GridType::Rect { 1e-9 } else { 1e-3 };
            if gt == GridType::Rect {
                assert!((geo.delx[s] - dx).abs() < tol * dx && (geo.dely[s] - dy).abs() < tol * dy);
            } else {
                // Fortran: HPFAC of W3GNTX has no cos(lat) on curvilinear grids
                assert!((geo.delx[s] - 0.5 * DERA * RADIUS).abs() < tol * dy, "{} {}", geo.delx[s], dy);
                assert!((geo.dely[s] - dy).abs() < tol * dy);
            }
        }
    }

    #[test]
    fn coast_and_corners() {
        // land only in the north-east quarter: the point at the diagonal of
        // the corner sees a sharp/mild corner, the one below the coast a
        // straight coast to the north
        let sp = spec(6, 24);
        let p = Ref1Params::from_namelists(&Namelists::parse(IRI), &sp).unwrap();
        let mut g = coast_grid(GridType::Rect);
        let nx = g.nx;
        for iy in 0..g.ny {
            for ix in 0..nx {
                let land = ix >= 6 && iy >= 5;
                g.mapsta[iy * nx + ix] = if land { 0 } else if g.mapsta[iy * nx + ix] == 3 { 3 } else { 1 };
            }
        }
        let geo = reflection_geometry(&g, &p).unwrap();
        // (7, 4): land at 1 (8,5), 2 (7,5), 3 (6,5): straight coast at K = 2,
        // direction north: TH = 90° -> ITH = 1 + 24/4 = 7; REFLD(4) = 1
        let s = isea_of(&geo, nx, 7, 4);
        assert_eq!(geo.refld[s], [7, 2, 0, 1, 0, 0]);
        // (5, 4): only neighbour 1 (6, 5) is land: K = 1 alone is neither a
        // straight coast nor a mild corner; not an even K: no reflection
        let c = isea_of(&geo, nx, 5, 4);
        assert_eq!(geo.reflc[c][0], 0.0);
        // (5, 5): land at 0 (6,5), 1 (6,6), 7 (6,4)? (6,4) is sea: mild corner
        // at K = 0 (0 and 1 land, 4 sea): REFLD(1) = 1 + (0 + 24)/16 = 2
        let m = isea_of(&geo, nx, 5, 5);
        assert_eq!(geo.refld[m][..2], [2, 1]);
        assert_eq!(geo.reflc[m][0], 0.05);
    }

    #[test]
    fn reflection_goes_back_from_the_coast() {
        let sp = spec(6, 24);
        let p = Ref1Params::from_namelists(&Namelists::parse(IRI), &sp).unwrap();
        let g = coast_grid(GridType::Rect);
        let geo = reflection_geometry(&g, &p).unwrap();
        let s = isea_of(&geo, g.nx, 7, 4);
        // eastward waves towards the coast at ix = 8
        let (a, cg) = spike(&p, 0);
        let v = sref_point(&p, &geo, s, &a, &cg, 0.0).unwrap();
        let nth = p.nth;
        let mut total = 0.0;
        for ik in 0..p.nk {
            for ith in 0..nth {
                let x = v[ith + ik * nth];
                assert!(x >= 0.0);
                if x > 0.0 {
                    assert!(p.ecos[ith] < 0.0, "energy into direction {ith}");
                }
                total += x;
            }
            // the specular direction (west) gets the most
            let row = &v[ik * nth..(ik + 1) * nth];
            let best = (0..nth).max_by(|&i, &j| row[i].partial_cmp(&row[j]).unwrap()).unwrap();
            assert_eq!(best, nth / 2);
        }
        assert!(total > 0.0);
        // westward waves leave the coast: nothing reflected
        let (a, cg) = spike(&p, p.nth / 2);
        let v = sref_point(&p, &geo, s, &a, &cg, 0.0).unwrap();
        assert!(v.iter().all(|&x| x == 0.0));
        // open sea, no obstruction: called (REFLC(2) = 1 as ww3_grid sets it
        // without obstruction file) but zero
        let o = isea_of(&geo, g.nx, 3, 4);
        let (a, cg) = spike(&p, 0);
        assert!(sref_point(&p, &geo, o, &a, &cg, 0.0).unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn no_reflection_without_refcoast() {
        let sp = spec(6, 24);
        let p = Ref1Params::from_namelists(&Namelists::parse("&REF1 REFCOAST = 0., REFFREQ = 1. /"), &sp).unwrap();
        let g = coast_grid(GridType::Rect);
        let geo = reflection_geometry(&g, &p).unwrap();
        assert!(geo.reflc.iter().all(|r| r[0] == 0.0 && r[1] == 0.0 && r[3] == 0.0));
        let s = isea_of(&geo, g.nx, 7, 4);
        let (a, cg) = spike(&p, 0);
        assert!(sref_point(&p, &geo, s, &a, &cg, 0.0).is_none());
        let v = sref(&p, &a, &cg, &geo.reflc[s], &geo.refld[s], geo.delx[s], geo.dely[s], 1.0, 1.0, 0.0);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn subgrid_obstruction() {
        let sp = spec(6, 24);
        let p = Ref1Params::from_namelists(&Namelists::parse(IRI), &sp).unwrap();
        let g = coast_grid(GridType::Rect);
        let n = g.nx * g.ny;
        let mut t = Transparencies { trnx: vec![1.0; n], trny: vec![1.0; n] };
        let k = 4 * g.nx + 3;
        t.trnx[k] = 0.4;
        let geo = reflection_geometry_trn(&g, &p, Some(&t)).unwrap();
        let o = isea_of(&geo, g.nx, 3, 4);
        assert!((geo.reflc[o][1] - 0.6).abs() < 1e-15);
        let o2 = isea_of(&geo, g.nx, 2, 4);
        assert_eq!(geo.reflc[o2][1], 0.0);
        assert!(sref_point(&p, &geo, o2, &spike(&p, 0).0, &spike(&p, 0).1, 0.0).is_none());
        let (a, cg) = spike(&p, 0);
        let v = sref_point(&p, &geo, o, &a, &cg, 0.0).unwrap();
        assert!(v.iter().any(|&x| x > 0.0));
        for ik in 0..p.nk {
            for ith in 0..p.nth {
                if v[ith + ik * p.nth] > 0.0 {
                    assert!(p.ecos[ith] < 0.0);
                }
            }
        }
    }

    #[test]
    fn obstruction_file() {
        let d = std::env::temp_dir().join(format!("hww3_ref1_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        // 3 x 2, OBSX then OBSY in percent (SF 0.01), one value above 100
        std::fs::write(d.join("g.obst"), "0 50 100\n10 20 130\n0 0 0\n0 25 0\n").unwrap();
        let nml = Namelists::parse("&OBST_NML\n OBST%SF = 0.010\n OBST%FILENAME = 'g.obst'\n/\n");
        assert!(read_obstructions(&nml, 0, &d, 3, 2).unwrap().is_none());
        let t = read_obstructions(&nml, 4, &d, 3, 2).unwrap().unwrap();
        let e = [1.0, 0.5, 0.0, 0.9, 0.8, 0.0];
        for (a, b) in t.trnx.iter().zip(e) {
            assert!((a - b).abs() < 1e-12, "{:?}", t.trnx);
        }
        assert!((t.trny[4] - 0.75).abs() < 1e-12 && t.trny[0] == 1.0);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
