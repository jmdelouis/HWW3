//! CPU transcription, in double precision, of the WW3 third-order geographic
//! propagation on the structured grids (`RECT` and `CURV`): the switches
//! `PR3` + `UQ` (`w3pro3md.F90` and `w3uqckmd.F90`, WW3 7.14), with the
//! averaging of Booij and Holthuijsen against the Garden-Sprinkler Effect
//! (GSE) and the sub-grid obstructions.
//!
//! Ported:
//! - `W3MAP3` (sections 1 to 4: the maps `MAPX2`, `MAPY2`, `MAPAXY`,
//!   `MAPCXY` of the UQ scheme, built from `MAPSTA`), in
//!   [`Pro3Grid::new`] and [`Pro3Grid::update_maps`];
//! - `W3UTRN` of `w3updtmd.F90` (the cell transparencies `ATRNX`, `ATRNY`
//!   from `TRNX`, `TRNY`, at the cell boundaries for `FLAGTR` 1, 3, 5 or
//!   at the cell centres for `FLAGTR` 2, 4, 6, and the ice and iceberg
//!   transparencies when `FLAGTR >= 3` and `FICEN - FICE0 >= 1e-6`, the
//!   ice part only with the `IC0` switch, [`Pro3Params::ic0`]);
//! - `W3MAPT` (the map `MAPTRN` of the points where the averaging is
//!   switched off, any transparency below `TRNMIN = 0.95`);
//! - `W3XYP3` (one spectral component over one global step `DTG`, [`xyp3`]):
//!   the number of sub-steps, the GSE averaging with `WDTHCG`, `WDTHTH`
//!   (`&PRO3`), the `x` and `y` sweeps alternating with the parity of the
//!   step, the curvilinear metric, the closure in longitude, the active
//!   boundary points and the boundary spectra of `FLBPI`;
//! - `W3QCK3` (UQ scheme with transparencies, [`qck3`]) and `W3QCK1` (the
//!   same without them, [`qck1`]); `W3QCK2` (irregular grids) is not used
//!   by `W3XYP3` and is not ported;
//! - `W3CFLXY` (the diagnostic CFL number of one sea point, [`cflxy`]);
//! - the parameters of `ww3_grid` that these routines read
//!   ([`Pro3Params::from_namelists`]).
//!
//! Not ported: `W3KTP3` (ported elsewhere) and section 5 of `W3MAP3` (its
//! spectral maps), the moving-grid switches `MGP` and `MGG` (`FGSE = 1`,
//! `VGX` and `VGY` unused), the `UNO` scheme, the tripolar closure (refused
//! by `W3XYP3` too), Cartesian grids (`FLAGLL` is true: the grids of
//! [`StructGrid`] are spherical), the test output.
//!
//! # Layout
//!
//! WW3 stores a field on the full grid as a 1-D array over `IXY =
//! (IX-1) NY + IY` (`IY` fastest), with halos: `VQ(1-NY:NY*(NX+2))` in
//! `W3XYP3`, `MAPSTX(1-2*NY:NY*(NX+2))`, `FLA(1-NY:NY*NX)` in `W3QCK3`.
//! Its index arithmetic is kept as it is (the moves `IXY +- NY` and
//! `IXY +- 1` of the averaging and of the stencils, which wrap from the top
//! of one column to the bottom of the next one, as in Fortran), on arrays
//! with the same bounds. The public functions take and return the fields in
//! the layout of [`StructGrid`] (`iy nx + ix`), which `xyp3` copies into
//! and out of the WW3 array as `W3GATH` and `W3SCAT` do (zero away from the
//! sea points, written back where `MAPSTA /= 0`). Sea points (`ISEA`) are in
//! WW3's order, `ix` fastest then `iy`; spectral components are
//! `isp = ith + ik nth` (0-based), WW3's `ISP - 1`.
//!
//! # Grid metric and closure
//!
//! The derivatives `DPDX`, `DPDY`, `DQDX`, `DQDY` and the Jacobian `GSQRT`
//! are those of `W3CGDM` (`w3gsrumd.F90`): `GSQRT = max(DXDP DYDQ - DXDQ
//! DYDP, 1e-15)`, `DPDX = DYDQ / GSQRT`, `DPDY = -DXDQ / GSQRT`, `DQDX =
//! -DYDP / GSQRT`, `DQDY = DXDP / GSQRT`, from the grid derivatives of
//! [`grid_metric`] (degrees per index, no spherical factor). The velocities
//! in m/s, `CXTOT = cos(th) CG / CLATS` and `CYTOT = sin(th) CG` (plus the
//! current), become Courant numbers in index space, `(CXTOT DPDX + CYTOT
//! DPDY) DT / (DERA RADIUS)` and the same with `DQDX`, `DQDY`; the
//! propagated quantity is `Q = A / CG CLATS GSQRT`, `CLATS = cos(latitude)`.
//! With `CLOS = 'SMPL'` (`GLOBAL`) `W3QCK3` copies the last column before
//! the first one and the first two after the last one before the `x` sweep,
//! and the averaging folds what it put in the halo columns back onto the
//! first and last columns.
//!
//! The comments `Fortran:` mark the oddities of WW3 kept on purpose. The
//! single-precision literals of WW3 (`1.E-15`, `0.575`, ...) are taken at
//! their decimal values; a sign test on a negative zero (`SIGN(1.1, CFL)`
//! with `CFL = -0.`) follows the default of the Intel compiler (`-0.` taken
//! as positive), which changes nothing since the flux is then `CFL QB = 0`.

use crate::ref1::{grid_metric, Ref1Params, Transparencies};
use crate::structgrid::StructGrid;
use crate::ww3grid::Namelists;
use anyhow::{bail, Result};

/// `PI` of WW3's `constants.F90`.
const PI: f64 = std::f64::consts::PI;
/// `DERA`: degrees to radians.
const DERA: f64 = PI / 180.0;
/// `RADIUS` of the Earth in `constants.F90` (m).
const RADIUS: f64 = 4.0e7 / (2.0 * PI);
/// `GRAV` (m/s2).
const GRAV: f64 = 9.806;
/// `TRNMIN` of `w3pro3md.F90`: below this transparency the averaging is
/// switched off.
const TRNMIN: f64 = 0.95;
/// `SMALL` of `W3CGDM`: lower bound of the Jacobian `GSQRT`.
const SMALL: f64 = 1e-15;

/// The parameters of `ww3_grid` that `W3XYP3`, `W3UTRN` and `W3MAPT` read.
#[derive(Clone, Debug, PartialEq)]
pub struct Pro3Params {
    /// `WDCG = WDTHCG` of `&PRO3`: width of the averaging along the
    /// propagation, as a fraction of the spread of the group velocities in
    /// one frequency bin (default 1.5).
    pub wdthcg: f64,
    /// `WDTH = WDTHTH` of `&PRO3`: width of the averaging across the
    /// propagation, as a fraction of the directional increment (default
    /// 1.5).
    pub wdthth: f64,
    /// `DTCFL = TIMESTEPS%DTXY` (s): the largest step of the geographic
    /// propagation for the lowest frequency.
    pub dtcfl: f64,
    /// `RUN%FLCX`: propagation along `x` (default true).
    pub flcx: bool,
    /// `RUN%FLCY`: propagation along `y` (default true).
    pub flcy: bool,
    /// `FLAGTR` of `&MISC` (0 to 6, default 0): 0 no obstructions; odd,
    /// transparencies at the cell boundaries; even, at the cell centres;
    /// 3 or more, ice transparencies too.
    pub flagtr: i64,
    /// `FICE0 = CICE0` of `&MISC` after `W3GRID`'s corrections.
    pub fice0: f64,
    /// `FICEN = CICEN` of `&MISC`, clipped to [0, 1].
    pub ficen: f64,
    /// `FICEL = LICE` of `&MISC` (m, default 0): with `LICE > 0` the ice
    /// transparency is `exp(-ICE DX / LICE)`.
    pub ficel: f64,
    /// `FFACBERG = FACBERG` of `&MISC` (default 1).
    pub ffacberg: f64,
    /// The `IC0` switch of the build: the ice concentration enters the
    /// transparencies of `W3UTRN` (`TRIX`, `TRIY`) only with it. False by
    /// default and in [`Pro3Params::from_namelists`]: the IRI build has
    /// `IC2`, with which the ice changes nothing in `W3UTRN` (only the
    /// icebergs would).
    pub ic0: bool,
}

impl Default for Pro3Params {
    /// The IRI configuration: `WDTHCG = WDTHTH = 1.5`, `DTXY = 420` s, both
    /// directions propagated, no obstructions, `CICE0 = CICEN = 0.5`,
    /// `LICE = 0`, `FACBERG = 1`.
    fn default() -> Pro3Params {
        Pro3Params {
            wdthcg: 1.5,
            wdthth: 1.5,
            dtcfl: 420.0,
            flcx: true,
            flcy: true,
            flagtr: 0,
            fice0: 0.5,
            ficen: 0.5,
            ficel: 0.0,
            ffacberg: 1.0,
            ic0: false,
        }
    }
}

impl Pro3Params {
    /// The parameters as `W3GRID` sets them: `TIMESTEPS%DTXY` (required)
    /// and `RUN%FLCX`, `RUN%FLCY` from the groups `TIMESTEPS_NML` and
    /// `RUN_NML` of `grid_nml` (`ww3_grid.nml`); `WDTHCG`, `WDTHTH` of
    /// `&PRO3` and `FLAGTR`, `CICE0`, `CICEN`, `LICE`, `FACBERG` of `&MISC`
    /// from `params_nml` (the file of `GRID%NML`; it may be the same).
    /// `FLAGTR` outside 0..6 becomes 0, `CICEN` is clipped to [0, 1],
    /// `CICE0` to [0, `CICEN`], and `CICE0 = CICEN` when `FLAGTR < 3`. The
    /// reduction of `FLAGTR` by 2 of the `IC0` switch when `CICE0 = CICEN`
    /// is not applied (the IRI configuration uses `IC2`).
    pub fn from_namelists(grid_nml: &Namelists, params_nml: &Namelists) -> Result<Pro3Params> {
        let dtcfl = grid_nml.f("TIMESTEPS_NML", "TIMESTEPS%DTXY", f64::NAN)?;
        if !(dtcfl > 0.0) {
            bail!("TIMESTEPS%DTXY missing or not positive");
        }
        let f = |g: &str, k: &str, d: f64| params_nml.f(g, k, d);
        // Fortran: WDTHTH defaults to the default of WDTHCG (1.5), not to
        // the value read for WDTHCG.
        let wdthcg = f("PRO3", "WDTHCG", 1.5)?;
        let wdthth = f("PRO3", "WDTHTH", 1.5)?;
        let mut flagtr = f("MISC", "FLAGTR", 0.0)?.round() as i64;
        if !(0..=6).contains(&flagtr) {
            flagtr = 0;
        }
        let ficen = f("MISC", "CICEN", 0.5)?.clamp(0.0, 1.0);
        let mut fice0 = f("MISC", "CICE0", 0.5)?.max(0.0).min(ficen);
        if flagtr < 3 && fice0 != ficen {
            fice0 = ficen;
        }
        Ok(Pro3Params {
            wdthcg,
            wdthth,
            dtcfl,
            flcx: grid_nml.b("RUN_NML", "RUN%FLCX", true)?,
            flcy: grid_nml.b("RUN_NML", "RUN%FLCY", true)?,
            flagtr,
            fice0,
            ficen,
            ficel: f("MISC", "LICE", 0.0)?,
            ffacberg: f("MISC", "FACBERG", 1.0)?,
            ic0: false,
        })
    }
}

/// Everything `W3MAP3`, `W3UTRN`, `W3MAPT` and `W3GNTX` prepare for
/// `W3XYP3` on one structured grid, with the spectral tables it reads.
#[derive(Clone, Debug)]
pub struct Pro3Grid {
    /// Points along a row (`NX`).
    pub nx: usize,
    /// Number of rows (`NY`).
    pub ny: usize,
    /// `ICLOSE = ICLOSE_SMPL` (`GLOBAL` in `W3XYP3`).
    pub closed: bool,
    /// The parameters of `ww3_grid`.
    pub params: Pro3Params,
    /// Number of frequencies.
    pub nk: usize,
    /// Number of directions.
    pub nth: usize,
    /// `SIG(0..NK+1)` (rad/s).
    pub sig: Vec<f64>,
    /// `ECOS(1..NTH)` at `ith`.
    pub ecos: Vec<f64>,
    /// `ESIN(1..NTH)` at `ith`.
    pub esin: Vec<f64>,
    /// `DTH` (rad).
    pub dth: f64,
    /// `XFR`.
    pub xfr: f64,
    /// Number of sea points (`NSEA`, `MAPSTA /= 0`).
    pub nsea: usize,
    /// Index `iy nx + ix` of each sea point in the layout of [`StructGrid`].
    pub sea_ij: Vec<usize>,
    /// `MAPSF(ISEA, 1:3)`: `IX`, `IY`, `IXY` (1-based).
    mapsf: Vec<[i64; 3]>,
    /// `MAPSTA` at `IXY - 1` (-2 to 2; excluded points are 0).
    mapsta: Vec<i32>,
    /// `CLATS = cos(latitude)` of each sea point.
    pub clats: Vec<f64>,
    /// `DPDX` of each sea point (index per degree).
    pub dpdx: Vec<f64>,
    /// `DPDY` of each sea point.
    pub dpdy: Vec<f64>,
    /// `DQDX` of each sea point.
    pub dqdx: Vec<f64>,
    /// `DQDY` of each sea point.
    pub dqdy: Vec<f64>,
    /// `GSQRT` of each sea point (square degrees per cell).
    pub gsqrt: Vec<f64>,
    /// `HPFAC` of each sea point (degrees).
    pub hpfac: Vec<f64>,
    /// `HQFAC` of each sea point (degrees).
    pub hqfac: Vec<f64>,
    /// `TRNX` at `IXY - 1` when `FLAGTR > 0`.
    trnx: Option<Vec<f64>>,
    /// `TRNY` at `IXY - 1` when `FLAGTR > 0`.
    trny: Option<Vec<f64>>,
    /// `MAPX2(1:NMX2)` (`IXY`, 1-based).
    mapx2: Vec<i64>,
    /// `NMX0`, `NMX1`, `NMX2`.
    nmx: [usize; 3],
    /// `MAPY2(1:NMY2)`.
    mapy2: Vec<i64>,
    /// `NMY0`, `NMY1`, `NMY2`.
    nmy: [usize; 3],
    /// `MAPAXY(1:NACT)`.
    mapaxy: Vec<i64>,
    /// `MAPCXY(1:NSEA)` as 0-based sea indices; the first `ncent` are the
    /// central points.
    mapcxy: Vec<usize>,
    /// `NCENT`.
    ncent: usize,
    /// `ATRNX(IXY, -1:1)` at `IXY - 1`, `[-1, 0, 1]`.
    atrnx: Vec<[f64; 3]>,
    /// `ATRNY(IXY, -1:1)`.
    atrny: Vec<[f64; 3]>,
    /// `MAPTRN(IXY)` at `IXY - 1`.
    maptrn: Vec<bool>,
}

/// Boundary spectra of `FLBPI` for one spectral component, interpolated in
/// time in each sub-step of [`xyp3`] (section 3.c of `W3XYP3`).
#[derive(Clone, Copy, Debug)]
pub struct BoundaryInput<'a> {
    /// `DSEC21(TBPI0, TIME)` (s), `TIME` being the end of the global step.
    pub dt0: f64,
    /// `DSEC21(TBPI0, TBPIN)` (s).
    pub dtn: f64,
    /// `ISBPI(1:NBI)`: 0-based sea index of each boundary point.
    pub isbpi: &'a [usize],
    /// `BBPI0(ISP, 1:NBI)`: action density at `TBPI0`.
    pub bbpi0: &'a [f64],
    /// `BBPIN(ISP, 1:NBI)`: action density at `TBPIN`.
    pub bbpin: &'a [f64],
}

impl Pro3Grid {
    /// Prepare the propagation on `g`, as `W3GRID`, `W3GNTX` and the first
    /// `FLMAP` update of `W3WAVE` do:
    /// 1. `MAPSTA` from `g.mapsta` (the excluded points, 3, become 0 as in
    ///    `W3GRID`), and the sea points (`MAPSTA /= 0`) in WW3's order with
    ///    `MAPSF` and `CLATS = cos(latitude)`;
    /// 2. the metric of `W3CGDM` from [`grid_metric`]: `GSQRT`, `DPDX`,
    ///    `DPDY`, `DQDX`, `DQDY`, `HPFAC`, `HQFAC` (see the module
    ///    documentation);
    /// 3. the spectral tables `SIG`, `ECOS`, `ESIN` of `W3GRID` (through
    ///    [`Ref1Params::with_refpars`]);
    /// 4. `TRNX`, `TRNY` from `trn` when `params.flagtr > 0` (an error
    ///    without them; ignored when `flagtr = 0`);
    /// 5. `W3MAP3`, `W3UTRN` without ice and `W3MAPT`
    ///    ([`Pro3Grid::update_maps`]).
    pub fn new(g: &StructGrid, trn: Option<&Transparencies>, params: &Pro3Params) -> Result<Pro3Grid> {
        let (nx, ny) = (g.nx, g.ny);
        if nx < 2 || ny < 2 {
            bail!("grid of {nx} x {ny} points");
        }
        // 1. MAPSTA and the sea points (W3GRID section 8.f)
        let mut mapsta = vec![0i32; nx * ny];
        for iy in 0..ny {
            for ix in 0..nx {
                let m = g.mapsta[iy * nx + ix];
                mapsta[ix * ny + iy] = if m == 3 { 0 } else { m as i32 };
            }
        }
        let metric = grid_metric(g)?;
        let (mut sea_ij, mut mapsf, mut clats) = (vec![], vec![], vec![]);
        let (mut dpdx, mut dpdy, mut dqdx, mut dqdy, mut gsqrt, mut hpfac, mut hqfac) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        for iy in 0..ny {
            for ix in 0..nx {
                if mapsta[ix * ny + iy] == 0 {
                    continue;
                }
                let i = iy * nx + ix;
                sea_ij.push(i);
                mapsf.push([ix as i64 + 1, iy as i64 + 1, (ix * ny + iy) as i64 + 1]);
                clats.push((g.lat[i] * DERA).cos());
                // 2. W3CGDM
                let (xp, yp, xq, yq) = (metric.dxdp[i], metric.dydp[i], metric.dxdq[i], metric.dydq[i]);
                let gs = (xp * yq - xq * yp).max(SMALL);
                gsqrt.push(gs);
                dpdx.push(yq / gs);
                dpdy.push(-xq / gs);
                dqdx.push(-yp / gs);
                dqdy.push(xp / gs);
                hpfac.push(metric.hpfac[i]);
                hqfac.push(metric.hqfac[i]);
            }
        }
        // 3. spectral tables
        let sp = Ref1Params::with_refpars([0.0; 10], &g.spec);
        // 4. transparencies, in WW3's layout
        let (trnx, trny) = if params.flagtr != 0 {
            let t = match trn {
                Some(t) => t,
                None => bail!("FLAGTR = {} without transparencies", params.flagtr),
            };
            let mut x = vec![0.0; nx * ny];
            let mut y = vec![0.0; nx * ny];
            for iy in 0..ny {
                for ix in 0..nx {
                    x[ix * ny + iy] = t.trnx[iy * nx + ix];
                    y[ix * ny + iy] = t.trny[iy * nx + ix];
                }
            }
            (Some(x), Some(y))
        } else {
            (None, None)
        };
        let nsea = sea_ij.len();
        let mut p = Pro3Grid {
            nx,
            ny,
            closed: g.closed,
            params: params.clone(),
            nk: g.spec.nk,
            nth: g.spec.nth,
            sig: sp.sig,
            ecos: sp.ecos,
            esin: sp.esin,
            dth: 2.0 * PI / g.spec.nth as f64,
            xfr: g.spec.xfr,
            nsea,
            sea_ij,
            mapsf,
            mapsta,
            clats,
            dpdx,
            dpdy,
            dqdx,
            dqdy,
            gsqrt,
            hpfac,
            hqfac,
            trnx,
            trny,
            mapx2: vec![],
            nmx: [0; 3],
            mapy2: vec![],
            nmy: [0; 3],
            mapaxy: vec![],
            mapcxy: vec![],
            ncent: 0,
            atrnx: vec![],
            atrny: vec![],
            maptrn: vec![],
        };
        // 5. maps
        p.update_maps(None, None, None)?;
        Ok(p)
    }

    /// The update of the maps of `W3WAVE` when `FLMAP` is set (section 3.5:
    /// `W3MAP3`, `W3UTRN`, `W3MAPT`), after a change of `MAPSTA` (points
    /// switched off, negative, by `W3ULEV` or `W3UICE`) or of the ice.
    /// `mapsta`: the new `MAPSTA` in the layout of [`StructGrid`] (values
    /// -2 to 2, zero exactly on the land and excluded points), or `None` to
    /// keep it. `ice`, `berg`: ice concentration and iceberg damping of
    /// each sea point, used by `W3UTRN` when `FLAGTR >= 3` and `FICEN -
    /// FICE0 >= 1e-6` (`None`: zero).
    pub fn update_maps(&mut self, mapsta: Option<&[i32]>, ice: Option<&[f64]>, berg: Option<&[f64]>) -> Result<()> {
        let (nx, ny) = (self.nx, self.ny);
        if let Some(m) = mapsta {
            if m.len() != nx * ny {
                bail!("MAPSTA of {} points, {} expected", m.len(), nx * ny);
            }
            for iy in 0..ny {
                for ix in 0..nx {
                    let (new, old) = (m[iy * nx + ix], self.mapsta[ix * ny + iy]);
                    if (new == 0) != (old == 0) || !(-2..=2).contains(&new) {
                        bail!("MAPSTA at ({ix}, {iy}): {new} (was {old}); the sea points cannot change");
                    }
                }
            }
            for iy in 0..ny {
                for ix in 0..nx {
                    self.mapsta[ix * ny + iy] = m[iy * nx + ix];
                }
            }
        }
        for (v, n) in [(ice, "ice"), (berg, "berg")] {
            if let Some(v) = v {
                if v.len() != self.nsea {
                    bail!("{n} of {} sea points, {} expected", v.len(), self.nsea);
                }
            }
        }
        self.map3();
        self.utrn(ice, berg);
        self.mapt();
        Ok(())
    }

    /// `MAPSTA(IY, IX)` read through the 1-D index `(IX-1) NY + IY`, as
    /// Fortran addresses it (`IY = 0` or `NY + 1` falls in the neighbouring
    /// column). Fortran: outside the array (never reached when the first
    /// and last rows hold no sea point of `MAPSTA = 1`) it reads 0 here.
    fn ms(&self, iy: i64, ix: i64) -> i32 {
        let k = (ix - 1) * self.ny as i64 + iy;
        if k < 1 || k > (self.nx * self.ny) as i64 {
            0
        } else {
            self.mapsta[(k - 1) as usize]
        }
    }

    /// `W3MAP3`, sections 1 to 4 (`GTYPE < 3`).
    fn map3(&mut self) {
        let (nx, ny) = (self.nx as i64, self.ny as i64);
        // 1. MAPX2. Fortran: IX2 wraps to the first column whether or not
        // the grid is closed, and the first and last rows are skipped.
        let mut mapx2 = vec![];
        let mut nmx = [0usize; 3];
        for pass in 0..3 {
            for ix in 1..=nx {
                let ixy0 = (ix - 1) * ny;
                let ix2 = 1 + ix % nx;
                for iy in 2..=ny - 1 {
                    let (a, b) = (self.ms(iy, ix) == 1, self.ms(iy, ix2) == 1);
                    let take = match pass {
                        0 => a && b,
                        1 => a && !b,
                        _ => !a && b,
                    };
                    if take {
                        mapx2.push(ixy0 + iy);
                    }
                }
            }
            nmx[pass] = mapx2.len();
        }
        // 2. MAPY2 (rows 1 to NY-1)
        let mut mapy2 = vec![];
        let mut nmy = [0usize; 3];
        for pass in 0..3 {
            for ix in 1..=nx {
                let ixy0 = (ix - 1) * ny;
                for iy in 1..=ny - 1 {
                    let (a, b) = (self.ms(iy, ix) == 1, self.ms(iy + 1, ix) == 1);
                    let take = match pass {
                        0 => a && b,
                        1 => a && !b,
                        _ => !a && b,
                    };
                    if take {
                        mapy2.push(ixy0 + iy);
                    }
                }
            }
            nmy[pass] = mapy2.len();
        }
        // 3. MAPAXY
        let mut mapaxy = vec![];
        for ix in 1..=nx {
            for iy in 2..=ny - 1 {
                if self.ms(iy, ix) == 1 {
                    mapaxy.push((ix - 1) * ny + iy);
                }
            }
        }
        // 4. MAPCXY: the central points (all eight neighbours with MAPSTA
        // >= 1) first, in sea order, the others from the end backwards.
        // Fortran: the neighbours wrap in IX whether or not the grid is
        // closed; IY0 = 0 and IY2 = NY + 1 are read through the 1-D index.
        let nsea = self.nsea;
        let mut mapcxy = vec![0usize; nsea];
        let (mut ncent, mut ncentc) = (0usize, nsea);
        for isea in 0..nsea {
            let [ix, iy, _] = self.mapsf[isea];
            let (mut ix0, mut ix2) = (ix - 1, ix + 1);
            let (iy0, iy2) = (iy - 1, iy + 1);
            if ix == nx {
                ix2 = 1;
            }
            if ix == 1 {
                ix0 = nx;
            }
            let m = self.ms(iy, ix);
            if m == 2 || m < 0 {
                ncentc -= 1;
                mapcxy[ncentc] = isea;
            } else if [(iy0, ix0), (iy0, ix), (iy0, ix2), (iy, ix0), (iy, ix2), (iy2, ix0), (iy2, ix), (iy2, ix2)]
                .iter()
                .all(|&(a, b)| self.ms(a, b) >= 1)
            {
                mapcxy[ncent] = isea;
                ncent += 1;
            } else {
                ncentc -= 1;
                mapcxy[ncentc] = isea;
            }
        }
        self.mapx2 = mapx2;
        self.nmx = nmx;
        self.mapy2 = mapy2;
        self.nmy = nmy;
        self.mapaxy = mapaxy;
        self.mapcxy = mapcxy;
        self.ncent = ncent;
    }

    /// `W3UTRN`: `ATRNX`, `ATRNY` (1 everywhere first).
    fn utrn(&mut self, ice: Option<&[f64]>, berg: Option<&[f64]>) {
        let (nx, ny) = (self.nx as i64, self.ny as i64);
        let n = (nx * ny) as usize;
        self.atrnx = vec![[1.0; 3]; n];
        self.atrny = vec![[1.0; 3]; n];
        let flagtr = self.params.flagtr;
        if flagtr == 0 {
            return;
        }
        let (trnx, trny) = (self.trnx.as_ref().unwrap(), self.trny.as_ref().unwrap());
        // Fortran: an index outside 1..NX*NY (IXY - 1 at the first point
        // of the grid, never a sea point) reads a transparency of 1 here.
        let at = |a: &Vec<f64>, k: i64| if k < 1 || k > nx * ny { 1.0 } else { a[(k - 1) as usize] };
        let isl = |k: i64| (k - 1) as usize;
        if flagtr == 1 || flagtr == 3 || flagtr == 5 {
            // data at the cell boundaries
            for isea in 0..self.nsea {
                let [ix, iy, ixy] = self.mapsf[isea];
                let (xm, xp) = if ix == 1 {
                    (at(trnx, iy + (nx - 1) * ny), at(trnx, ixy))
                } else if ix == nx {
                    (at(trnx, ixy - ny), at(trnx, iy))
                } else {
                    (at(trnx, ixy - ny), at(trnx, ixy))
                };
                self.atrnx[isl(ixy)][0] = xm;
                self.atrnx[isl(ixy)][2] = xp;
                self.atrny[isl(ixy)][0] = at(trny, ixy - 1);
                self.atrny[isl(ixy)][2] = at(trny, ixy);
            }
        } else {
            // data at the cell centres
            for isea in 0..self.nsea {
                let [ix, iy, ixy] = self.mapsf[isea];
                let (ixn, ixp) = neighbours_x(ix, iy, nx, ny);
                let (iyn, iyp) = neighbours_y(iy, ixy, ny);
                let t = |a: &Vec<f64>, k: i64| a[(k - 1) as usize];
                let (tx, ty) = (t(trnx, ixy), t(trny, ixy));
                self.atrnx[isl(ixy)][0] = (1.0 + tx) * t(trnx, ixn) / (1.0 + t(trnx, ixn));
                self.atrnx[isl(ixy)][2] = (1.0 + tx) * t(trnx, ixp) / (1.0 + t(trnx, ixp));
                self.atrny[isl(ixy)][0] = (1.0 + ty) * t(trny, iyn) / (1.0 + t(trny, iyn));
                self.atrny[isl(ixy)][2] = (1.0 + ty) * t(trny, iyp) / (1.0 + t(trny, iyp));
                // active boundary points: open towards land and the edges
                if self.ms(iy, ix) == 2 {
                    if ix == 1 || self.ms(iy, ix - 1) <= 0 {
                        self.atrnx[isl(ixy)][0] = 1.0;
                    }
                    if ix == nx || self.ms(iy, ix + 1) <= 0 {
                        self.atrnx[isl(ixy)][2] = 1.0;
                    }
                    if iy == 1 || self.ms(iy - 1, ix) <= 0 {
                        self.atrny[isl(ixy)][0] = 1.0;
                    }
                    if iy == ny || self.ms(iy + 1, ix) <= 0 {
                        self.atrny[isl(ixy)][2] = 1.0;
                    }
                }
            }
        }
        let p = &self.params;
        if flagtr < 3 || p.ficen - p.fice0 < 1e-6 || (ice.is_none() && berg.is_none()) {
            // (no ice nor berg: every TRIX = 1 and ATRN is unchanged)
            return;
        }
        // ice and icebergs, at the cell centres
        let mut trix = vec![1.0; n];
        let mut triy = vec![1.0; n];
        for isea in 0..self.nsea {
            let ixy = self.mapsf[isea][2];
            let dx = self.hpfac[isea] * RADIUS * DERA;
            let dy = self.hqfac[isea] * RADIUS * DERA;
            let ic = ice.map_or(0.0, |v| v[isea]);
            let bg = berg.map_or(0.0, |v| v[isea]);
            let k = isl(ixy);
            // Fortran: the ice part only with the IC0 switch
            if ic > 0.0 && p.ic0 {
                if p.ficel > 0.0 {
                    trix[k] = (-ic * dx / p.ficel).exp();
                    triy[k] = (-ic * dy / p.ficel).exp();
                } else {
                    let (lice0, licen) = (p.fice0 * dx, p.ficen * dx);
                    trix[k] = (licen - ic * dx) / (licen - lice0);
                    let (lice0, licen) = (p.fice0 * dy, p.ficen * dy);
                    triy[k] = (licen - ic * dy) / (licen - lice0);
                }
                trix[k] = trix[k].min(1.0).max(0.0);
                triy[k] = triy[k].min(1.0).max(0.0);
            }
            if bg > 0.0 {
                trix[k] *= (-bg * p.ffacberg * dx * 0.0001).exp();
                triy[k] *= (-bg * p.ffacberg * dy * 0.0001).exp();
            }
        }
        for isea in 0..self.nsea {
            let [ix, iy, ixy] = self.mapsf[isea];
            let (ixn, ixp) = neighbours_x(ix, iy, nx, ny);
            let (iyn, iyp) = neighbours_y(iy, ixy, ny);
            let k = isl(ixy);
            self.atrnx[k][0] *= (1.0 + trix[k]) * trix[isl(ixn)] / (1.0 + trix[isl(ixn)]);
            self.atrnx[k][2] *= (1.0 + trix[k]) * trix[isl(ixp)] / (1.0 + trix[isl(ixp)]);
            self.atrny[k][0] *= (1.0 + triy[k]) * triy[isl(iyn)] / (1.0 + triy[isl(iyn)]);
            self.atrny[k][2] *= (1.0 + triy[k]) * triy[isl(iyp)] / (1.0 + triy[isl(iyp)]);
        }
    }

    /// `W3MAPT`: `MAPTRN` at the sea points.
    fn mapt(&mut self) {
        self.maptrn = vec![false; self.nx * self.ny];
        for isea in 0..self.nsea {
            let k = (self.mapsf[isea][2] - 1) as usize;
            // Fortran: the corrected expression (ATRNX twice, ATRNY twice).
            let m = self.atrnx[k][2].min(self.atrnx[k][0]).min(self.atrny[k][2]).min(self.atrny[k][0]);
            self.maptrn[k] = m < TRNMIN;
        }
    }

    /// `MAPSTA` of point `(ix, iy)` (0-based).
    pub fn mapsta(&self, ix: usize, iy: usize) -> i32 {
        self.mapsta[ix * self.ny + iy]
    }

    /// `ATRNX(IXY, -1)` and `ATRNX(IXY, 1)` of point `(ix, iy)`.
    pub fn atrnx(&self, ix: usize, iy: usize) -> (f64, f64) {
        let a = self.atrnx[ix * self.ny + iy];
        (a[0], a[2])
    }

    /// `ATRNY(IXY, -1)` and `ATRNY(IXY, 1)` of point `(ix, iy)`.
    pub fn atrny(&self, ix: usize, iy: usize) -> (f64, f64) {
        let a = self.atrny[ix * self.ny + iy];
        (a[0], a[2])
    }

    /// `MAPTRN` of point `(ix, iy)`: no averaging there.
    pub fn maptrn(&self, ix: usize, iy: usize) -> bool {
        self.maptrn[ix * self.ny + iy]
    }

    /// `NCENT`: the number of central points of the averaging.
    pub fn ncent(&self) -> usize {
        self.ncent
    }

    /// `MAPCXY`: the sea points (0-based `ISEA`) in the order of the
    /// averaging of [`xyp3`], the [`Pro3Grid::ncent`] central points first.
    pub fn mapcxy(&self) -> &[usize] {
        &self.mapcxy
    }

    /// The counters of `W3MAP3`: `NCENT`, `NMX0`, `NMX1`, `NMX2`, `NMY0`,
    /// `NMY1`, `NMY2`, `NACT`.
    pub fn counts(&self) -> [usize; 8] {
        [self.ncent, self.nmx[0], self.nmx[1], self.nmx[2], self.nmy[0], self.nmy[1], self.nmy[2], self.mapaxy.len()]
    }
}

/// The interpolation table of `DISTAB` (`w3dispmd.F90`) and `WAVNU1`, the
/// wavenumbers and group velocities `W3INIT` gives the structured grids
/// (without the `PDLIB` switch), in single precision as in WW3.
#[derive(Clone, Debug)]
pub struct DispTable {
    /// `N1MAX = NAR1D - 1 = 120`.
    n1max: i32,
    /// `DSIE = sqrt(GRAV DFAC) / N1MAX` with `DFAC = 6`.
    dsie: f32,
    /// `EWN1(0:NAR1D)`: `k` at unit depth.
    ewn1: Vec<f32>,
    /// `ECG1(0:NAR1D)`: group velocity at unit depth.
    ecg1: Vec<f32>,
}

impl Default for DispTable {
    fn default() -> Self {
        DispTable::new()
    }
}

impl DispTable {
    /// `DISTAB`: `WAVNU2` at unit depth (`EPS = 1e-7`, 15 iterations) at
    /// the `NAR1D + 1 = 122` points `si = i DSIE`.
    pub fn new() -> DispTable {
        let g = GRAV as f32;
        let nar1d = 121usize;
        let n1max = (nar1d - 1) as i32;
        let simax = (g * 6.0f32).sqrt();
        let dsie = simax / n1max as f32;
        let mut ewn1 = vec![0f32; nar1d + 1];
        let mut ecg1 = vec![0f32; nar1d + 1];
        ecg1[0] = g.sqrt();
        for i in 1..=nar1d {
            let (k, cg) = wavnu2(i as f32 * dsie, 1.0, 1e-7, 15);
            ewn1[i] = k;
            ecg1[i] = cg;
        }
        DispTable { n1max, dsie, ewn1, ecg1 }
    }

    /// `WAVNU1`: wavenumber and group velocity of the radian frequency `si`
    /// at depth `h`, interpolated in the table, deep water beyond it.
    pub fn wavnu1(&self, si: f32, h: f32) -> (f32, f32) {
        let sqrth = h.sqrt();
        let six = si * sqrth;
        let i1 = (six / self.dsie) as i32;
        if i1 <= self.n1max && i1 >= 1 {
            let i2 = i1 + 1;
            let r1 = six / self.dsie - i1 as f32;
            let r2 = 1.0 - r1;
            let (a, b) = (i1 as usize, i2 as usize);
            ((r2 * self.ewn1[a] + r1 * self.ewn1[b]) / h, (r2 * self.ecg1[a] + r1 * self.ecg1[b]) * sqrth)
        } else {
            let g = GRAV as f32;
            (si * si / g, 0.5 * g / si)
        }
    }
}

/// `WAVNU2`: Newton iterations of the dispersion relation (single
/// precision), for the table of `DISTAB` (positive `w`).
fn wavnu2(w: f32, h: f32, eps: f32, nmax: usize) -> (f32, f32) {
    let g = GRAV as f32;
    let w0 = w.abs();
    let mut k = if w0 < (g / h).sqrt() { w0 / (g * h).sqrt() } else { w0 * w0 / g };
    let mut kold = 0f32;
    for _ in 0..nmax {
        let dif = (k - kold).abs();
        let rdif = if k != 0.0 { dif / k } else { 0.0 };
        if dif < eps && rdif < eps {
            break;
        }
        kold = k;
        let f = g * kold * (kold * h).tanh() - w0 * w0;
        let fd = if kold * h > 25.0 { g * (kold * h).tanh() } else { g * (kold * h).tanh() + g * kold * h / (kold * h).cosh().powi(2) };
        k = kold - f / fd;
    }
    let cg = if 2.0 * k * h > 25.0 { w0 / k * 0.5 } else { w0 / k * 0.5 * (1.0 + (2.0 * k * h / (2.0 * k * h).sinh())) };
    (k, cg)
}

/// `IXN`, `IXP` of `W3UTRN` (cell centres): the neighbours in `x`, wrapping
/// whether or not the grid is closed.
fn neighbours_x(ix: i64, iy: i64, nx: i64, ny: i64) -> (i64, i64) {
    if ix == 1 {
        (iy + (nx - 1) * ny, iy + ix * ny)
    } else if ix == nx {
        (iy + (ix - 2) * ny, iy)
    } else {
        (iy + (ix - 2) * ny, iy + ix * ny)
    }
}

/// `IYN`, `IYP` of `W3UTRN`: the neighbours in `y`, the point itself at the
/// first and last rows.
fn neighbours_y(iy: i64, ixy: i64, ny: i64) -> (i64, i64) {
    if iy == 1 {
        (ixy, ixy + 1)
    } else if iy == ny {
        (ixy - 1, ixy)
    } else {
        (ixy - 1, ixy + 1)
    }
}

/// `SIGN(1.1, CFL)`, `-0.` taken as positive.
fn sign11(cfl: f64) -> f64 {
    if cfl >= 0.0 {
        1.1
    } else {
        -1.1
    }
}

/// `INT` of Fortran (truncation towards zero) of a small real.
fn fint(x: f64) -> i64 {
    x.trunc() as i64
}

/// `W3QCK3`: one sweep of the ULTIMATE QUICKEST scheme with the cell
/// transparencies `trans` (`TRANS(IXY, -1:1)` at `IXY - 1`), or `W3QCK1`
/// with `trans = None` (the same fluxes, the update without the factors).
///
/// Arrays in WW3's layout (`IXY = (IX-1) MY + IY` with `INC = MY` along `x`,
/// `INC = 1` along `y`): `cfll` is `CFLL(1:MY*(MX+1))` (index `IXY - 1`),
/// `q` is `Q(1-MY:MY*(MX+2))` (index `IXY + MY - 1`), `mapact`
/// (`MAPACT(1:NACT)`) and `mapbou` (`MAPBOU(1:NB2)`) hold 1-based `IXY`.
/// With `close` the ghost columns of `q` and `cfll` are filled first
/// (`cfll` is modified, `INTENT(INOUT)` in WW3).
#[allow(clippy::too_many_arguments)]
pub fn qck3(
    mx: usize,
    my: usize,
    nx: usize,
    ny: usize,
    cfll: &mut [f64],
    trans: Option<&[[f64; 3]]>,
    q: &mut [f64],
    close: bool,
    inc: i64,
    mapact: &[i64],
    mapbou: &[i64],
    nb: [usize; 3],
) {
    let (mx, my, nx, ny) = (mx as i64, my as i64, nx as i64, ny as i64);
    assert!(cfll.len() as i64 >= my * (mx + 1) && q.len() as i64 >= my * (mx + 3));
    let qi = |i: i64| (i + my - 1) as usize;
    let ci = |i: i64| (i - 1) as usize;
    // 1. FLA(1-MY:MY*MX) and the closure
    let mut fla = vec![0.0f64; (my * (mx + 1)) as usize];
    let (iad00, iad02) = (-my, my);
    let (iadn0, iadn1, iadn2) = (iad00 + my * nx, my * nx, iad02 + my * nx);
    if close {
        for iy in 1..=ny {
            q[qi(iy + iad00)] = q[qi(iy + iadn0)]; // 1 ghost column to the left
            q[qi(iy + iadn1)] = q[qi(iy)]; // 1st ghost column to the right
            q[qi(iy + iadn2)] = q[qi(iy + iad02)]; // 2nd ghost column to the right
            cfll[ci(iy + iadn1)] = cfll[ci(iy)];
        }
    }
    // 2. central points: 3rd order with the ULTIMATE limiter
    for &ixy in &mapbou[..nb[0]] {
        let cfl = 0.5 * (cfll[ci(ixy)] + cfll[ci(ixy + inc)]);
        let s = sign11(cfl);
        let ixyc = ixy - inc * fint(0.0f64.min(s));
        let mut qb = 0.5 * ((1.0 - cfl) * q[qi(ixy + inc)] + (1.0 + cfl) * q[qi(ixy)])
            - (1.0 - cfl * cfl) / 6.0 * (q[qi(ixyc - inc)] - 2.0 * q[qi(ixyc)] + q[qi(ixyc + inc)]);
        let ixyu = ixyc - inc * fint(s);
        let ixyd = 2 * ixyc - ixyu;
        let dq = q[qi(ixyd)] - q[qi(ixyu)];
        let m = dq.abs().max(1e-15);
        let dqnz = if dq >= 0.0 { m } else { -m };
        let mut qcn = (q[qi(ixyc)] - q[qi(ixyu)]) / dqnz;
        qcn = (-0.1f64).max(qcn).min(1.1);
        let mut qbn = ((qb - q[qi(ixyu)]) / dqnz).max(qcn);
        qbn = qbn.min(1.0).min(qcn / 1e-10f64.max(cfl.abs()));
        let qbr = q[qi(ixyu)] + qbn * dq;
        let cfac = fint(2.0 * (qcn - 0.5).abs()) as f64;
        qb = (1.0 - cfac) * qbr + cfac * q[qi(ixyc)];
        fla[qi(ixy)] = cfl * qb;
    }
    // 3. boundary above: 1st order
    for &ixy in &mapbou[nb[0]..nb[1]] {
        let cfl = cfll[ci(ixy)];
        let ixyc = ixy - inc * fint(0.0f64.min(sign11(cfl)));
        fla[qi(ixy)] = cfl * q[qi(ixyc)];
    }
    // 4. boundary below: 1st order
    for &ixy in &mapbou[nb[1]..nb[2]] {
        let cfl = cfll[ci(ixy + inc)];
        let ixyc = ixy - inc * fint(0.0f64.min(sign11(cfl)));
        fla[qi(ixy)] = cfl * q[qi(ixyc)];
    }
    // 5. closure of the fluxes
    if close {
        for iy in 1..=ny {
            fla[qi(iy + iad00)] = fla[qi(iy + iadn0)];
        }
    }
    // 6. update of the active points; the transparencies scale the
    // incoming fluxes only (outgoing fluxes leave in full)
    for &ixy in mapact {
        let jn = if fla[qi(ixy - inc)] > 0.0 { -1 } else { 0 };
        let jp = if fla[qi(ixy)] < 0.0 { 1 } else { 0 };
        let k = qi(ixy);
        q[k] = match trans {
            Some(t) => {
                let tr = &t[(ixy - 1) as usize];
                (q[k] + tr[(jn + 1) as usize] * fla[qi(ixy - inc)] - tr[(jp + 1) as usize] * fla[k]).max(0.0)
            }
            None => (q[k] + fla[qi(ixy - inc)] - fla[k]).max(0.0),
        };
    }
}

/// `W3QCK1`: [`qck3`] without transparencies (same arguments).
#[allow(clippy::too_many_arguments)]
pub fn qck1(mx: usize, my: usize, nx: usize, ny: usize, cfll: &mut [f64], q: &mut [f64], close: bool, inc: i64, mapact: &[i64], mapbou: &[i64], nb: [usize; 3]) {
    qck3(mx, my, nx, ny, cfll, None, q, close, inc, mapact, mapbou, nb);
}

/// Velocities of the sub-step count of `W3XYP3` (section 1.a): `(CG0, CGX,
/// CGY, CXC, CYC)` with the deep-water group velocities `0.575 g / SIG`.
fn speeds(g: &Pro3Grid, isp: usize, cur: Option<(&[f64], &[f64])>) -> (f64, f64, f64, f64, f64) {
    let (ith, ik) = (isp % g.nth, isp / g.nth);
    let cg0 = 0.575 * GRAV / g.sig[1];
    let cga = 0.575 * GRAV / g.sig[ik + 1];
    let (mut cgx, mut cgy) = (cga * g.ecos[ith], cga * g.esin[ith]);
    let (mut cxc, mut cyc) = (0.0, 0.0);
    if let Some((cx, cy)) = cur {
        let mm = |v: &[f64]| v.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
        let (cxmin, cxmax) = mm(&cx[..g.nsea]);
        let (cymin, cymax) = mm(&cy[..g.nsea]);
        // Fortran: the current is added to the deep-water speed with the
        // extreme that gives the largest magnitude.
        cgx += if (cgx + cxmin).abs() > (cgx + cxmax).abs() { cxmin } else { cxmax };
        cgy += if (cgy + cymin).abs() > (cgy + cymax).abs() { cymin } else { cymax };
        cxc = cxmin.abs().max(cxmax.abs());
        cyc = cymin.abs().max(cymax.abs());
    }
    (cg0, cgx, cgy, cxc, cyc)
}

/// `NTLOC` of `W3XYP3`: the number of sub-steps of component `isp` over
/// `dtg`, `1 + INT(DTG / (DTCFL CG0 / CGN))` with `CGN` the largest of the
/// deep-water speed components (with the extreme currents) and
/// `0.001 CG0`. Fortran: the actual group velocities do not enter.
pub fn substeps(g: &Pro3Grid, isp: usize, dtg: f64, cur: Option<(&[f64], &[f64])>) -> usize {
    let (cg0, cgx, cgy, cxc, cyc) = speeds(g, isp, cur);
    let cgn = cgx.abs().max(cgy.abs()).max(cxc).max(cyc).max(0.001 * cg0);
    (1 + fint(dtg / (g.params.dtcfl * cg0 / cgn))) as usize
}

/// `W3XYP3`: the propagation of spectral component `isp` (`ith + ik nth`)
/// over the global step `dtg` (s), in place in `field` (action density on
/// the full grid, layout `iy nx + ix`; only the sea points are read and
/// written).
///
/// - `cg`: the group velocity `CG(IK, ISEA)` of each sea point (m/s);
/// - `cur`: the currents `CX`, `CY` of each sea point (m/s, `FLCUR`), or
///   `None`;
/// - `secs_of_day`: `DSEC21` from 00:00 of the day of `TIME` to `TIME`, the
///   end of the global step in `W3WAVE` (`TIME` is advanced before the
///   propagation): it sets the order of the sweeps;
/// - `bpi`: the boundary spectra (`FLBPI`), or `None`.
///
/// Steps (sections of `W3XYP3`):
/// 1. the number of sub-steps [`substeps`], `DTLOC = DTG / NTLOC`, and
///    `YFIRST = MOD(NINT(DTTST / DTG), 2) == 0` with `DTTST` the integer
///    seconds of the day; the widths of the averaging `FTH = WDTH DTH
///    DTLOC / (RADIUS DERA)` across and `FCG = WDCG (XFR - 1/XFR) / 2 DTLOC
///    / (RADIUS DERA) / NTLOC` along the propagation, turned into index
///    space at each sea point and split into the two "corners" of the
///    averaging (`IIXY1..4`, `RDI1..4`);
/// 2. `Q = A / CG CLATS` and the Courant numbers of each sea point;
/// 3. for each sub-step: the averaging (central points, near-coast points
///    without moving action onto land, no averaging where `MAPTRN`, the
///    active boundary points restored, the halo columns folded back when
///    closed), `Q GSQRT`, the `y` then `x` sweeps of [`qck3`] when
///    `YFIRST` (else `x` then `y`), `Q / GSQRT`, the boundary spectra, and
///    `YFIRST` flipped;
/// 4. `A = max(0, CG / CLATS Q)` at the points with `MAPSTA > 0`.
///
/// Fortran: the points with `MAPSTA < 0` come out at zero (the averaging
/// skips them and they are not propagated), and `W3SCAT` writes them back.
#[allow(clippy::too_many_arguments)]
pub fn xyp3(g: &Pro3Grid, isp: usize, dtg: f64, secs_of_day: f64, field: &mut [f64], cg: &[f64], cur: Option<(&[f64], &[f64])>, bpi: Option<&BoundaryInput>) {
    let (nx, ny) = (g.nx as i64, g.ny as i64);
    let nsea = g.nsea;
    assert_eq!(field.len(), g.nx * g.ny);
    assert!(cg.len() >= nsea);
    let (ith, ik) = (isp % g.nth, isp / g.nth);
    assert!(ik < g.nk);
    let global = g.closed;
    let p = &g.params;

    // 1.a constants
    let ntloc = substeps(g, isp, dtg, cur) as i64;
    let dtloc = dtg / ntloc as f64;
    let dtrad = dtloc / (DERA * RADIUS);
    // Fortran: DTTST is an INTEGER, the REAL of DSEC21 truncated.
    let dttst = secs_of_day.trunc();
    let mut yfirst = ((dttst / dtg).round() as i64).rem_euclid(2) == 0;

    // 1.b arrays: VQ and AQ over 1-2NY..NY(NX+2) (WW3: 1-NY..NY(NX+2); the
    // extra column below only takes writes WW3 would make out of bounds,
    // never with sea points of MAPSTA >= 1 on the first and last rows)
    let lo = 1 - 2 * ny;
    let hi = ny * (nx + 2);
    let v = |i: i64| (i - lo) as usize;
    let len = (hi - lo + 1) as usize;
    let mut vlcflx = vec![0.0f64; ((nx + 1) * ny) as usize];
    let mut vlcfly = vec![0.0f64; ((nx + 1) * ny) as usize];
    let mut mapstx = vec![0i32; len];
    for k in 1..=nx * ny {
        mapstx[v(k)] = g.mapsta[(k - 1) as usize];
    }
    if global {
        for j in 0..2 * ny {
            mapstx[v(1 - 2 * ny + j)] = g.mapsta[((nx - 2) * ny + j) as usize];
            mapstx[v(nx * ny + 1 + j)] = g.mapsta[j as usize];
        }
    }

    // 1.c interpolation data of the averaging
    let mut fth = p.wdthth * g.dth * dtloc;
    let mut fcg = p.wdthcg * 0.5 * (g.xfr - 1.0 / g.xfr) * dtloc;
    fth = fth / RADIUS / DERA;
    fcg = fcg / RADIUS / DERA;
    // Fortran: "required to match original (is this correct?)": the width
    // along the propagation is divided by the number of sub-steps.
    fcg /= ntloc as f64;
    let (ec, es) = (g.ecos[ith], g.esin[ith]);
    let (fthx, fthy, fcgx, fcgy) = (-fth * es, fth * ec, fcg * ec, fcg * es);
    let mut iixy = vec![[0i64; 4]; nsea];
    let mut rdi = vec![[0.0f64; 4]; nsea];
    for isea in 0..nsea {
        let tmpx = fthx / g.clats[isea];
        let tmpy = fthy;
        let dxcgn = g.dpdx[isea] * tmpx + g.dpdy[isea] * tmpy;
        let dycgn = g.dqdx[isea] * tmpx + g.dqdy[isea] * tmpy;
        let tmpx = fcgx / g.clats[isea];
        let tmpy = fcgy;
        let dxcgs = g.dpdx[isea] * tmpx + g.dpdy[isea] * tmpy;
        let dycgs = g.dqdx[isea] * tmpx + g.dqdy[isea] * tmpy;
        // "sum" corner then "difference" corner (and their mirror images)
        for (c, (dxcgc, dycgc)) in [(dxcgn + dxcgs, dycgn + dycgs), (dxcgn - dxcgs, dycgn - dycgs)].into_iter().enumerate() {
            let ixc = if dxcgc < 0.0 { -ny } else { ny };
            let iyc = if dycgc < 0.0 { -1 } else { 1 };
            iixy[isea][2 * c] = ixc + iyc;
            if dxcgc.abs() > dycgc.abs() {
                iixy[isea][2 * c + 1] = ixc;
                rdi[isea][2 * c] = (dycgc / dxcgc).abs();
                rdi[isea][2 * c + 1] = dxcgc.abs();
            } else {
                iixy[isea][2 * c + 1] = iyc;
                rdi[isea][2 * c] = if dycgc.abs() > 1e-5 { (dxcgc / dycgc).abs() } else { 1.0 };
                rdi[isea][2 * c + 1] = dycgc.abs();
            }
        }
    }

    // 2.a Q and the Courant numbers (W3GATH: zero away from the sea points)
    let mut vq = vec![0.0f64; len];
    let mut cxtot = vec![0.0f64; ((nx + 1) * ny) as usize];
    let mut cytot = vec![0.0f64; (nx * ny) as usize];
    for isea in 0..nsea {
        let ixy = g.mapsf[isea][2];
        let k = (ixy - 1) as usize;
        vq[v(ixy)] = field[g.sea_ij[isea]] / cg[isea] * g.clats[isea];
        cxtot[k] = ec * cg[isea] / g.clats[isea];
        cytot[k] = es * cg[isea];
    }
    if let Some((cx, cy)) = cur {
        for isea in 0..nsea {
            let k = (g.mapsf[isea][2] - 1) as usize;
            cxtot[k] += cx[isea] / g.clats[isea];
            cytot[k] += cy[isea];
        }
    }
    for isea in 0..nsea {
        let k = (g.mapsf[isea][2] - 1) as usize;
        let cp = cxtot[k] * g.dpdx[isea] + cytot[k] * g.dpdy[isea];
        let cq = cxtot[k] * g.dqdx[isea] + cytot[k] * g.dqdy[isea];
        vlcflx[k] = cp * dtrad;
        vlcfly[k] = cq * dtrad;
    }

    // 3. sub-steps
    let mut aq = vec![0.0f64; len];
    let qlo = v(1 - ny);
    for itloc in 1..=ntloc {
        // 3.a averaging
        aq.copy_from_slice(&vq);
        vq.iter_mut().for_each(|x| *x = 0.0);
        // 3.a.1 central points
        for &isea in &g.mapcxy[..g.ncent] {
            let ixy = g.mapsf[isea][2];
            if g.maptrn[(ixy - 1) as usize] {
                // Fortran: an assignment, which drops what the neighbours
                // already put at this point
                vq[v(ixy)] = aq[v(ixy)];
            } else {
                let [i1, i2, i3, i4] = iixy[isea];
                let rd1 = rdi[isea][0];
                let rd2 = (rdi[isea][1] * cg[isea]).min(1.0);
                let rd3 = rdi[isea][2];
                let rd4 = (rdi[isea][3] * cg[isea]).min(1.0);
                let a = aq[v(ixy)];
                vq[v(ixy)] += a * (3.0 - rd2 - rd4) / 3.0;
                vq[v(ixy + i1)] += a * rd2 * rd1 / 6.0;
                vq[v(ixy + i2)] += a * (1.0 - rd1) * rd2 / 6.0;
                vq[v(ixy + i3)] += a * rd4 * rd3 / 6.0;
                vq[v(ixy + i4)] += a * (1.0 - rd3) * rd4 / 6.0;
                vq[v(ixy - i1)] += a * rd2 * rd1 / 6.0;
                vq[v(ixy - i2)] += a * (1.0 - rd1) * rd2 / 6.0;
                vq[v(ixy - i3)] += a * rd4 * rd3 / 6.0;
                vq[v(ixy - i4)] += a * (1.0 - rd3) * rd4 / 6.0;
            }
        }
        // 3.a.2 near-coast points: a share that would land on a point with
        // MAPSTA <= 0 (or cross a corner between two of them) stays here
        for &isea in &g.mapcxy[g.ncent..] {
            let ixy = g.mapsf[isea][2];
            if g.mapsta[(ixy - 1) as usize] <= 0 {
                continue;
            }
            if g.maptrn[(ixy - 1) as usize] {
                vq[v(ixy)] = aq[v(ixy)];
                continue;
            }
            let [i1, i2, i3, i4] = iixy[isea];
            let rd1 = rdi[isea][0];
            let rd3 = rdi[isea][2];
            let rd2 = (rdi[isea][1] * cg[isea]).min(1.0);
            let rd4 = (rdi[isea][3] * cg[isea]).min(1.0);
            let a = aq[v(ixy)];
            vq[v(ixy)] += a * (3.0 - rd2 - rd4) / 3.0;
            let st = |k: i64| mapstx[v(k)];
            // corner 1 (diagonal: the target wet and not both side points
            // dry), then its axial neighbour
            let ixc = if i1 >= 0 { ny } else { -ny };
            let iyc = i1 - ixc;
            let w = a * rd2 * rd1 / 6.0;
            let t = if st(ixy + i1) >= 1 && !(st(ixy + ixc) <= 0 && st(ixy + iyc) <= 0) { ixy + i1 } else { ixy };
            vq[v(t)] += w;
            let t = if st(ixy - i1) >= 1 && !(st(ixy - ixc) <= 0 && st(ixy - iyc) <= 0) { ixy - i1 } else { ixy };
            vq[v(t)] += w;
            let w = a * (1.0 - rd1) * rd2 / 6.0;
            let t = if st(ixy + i2) >= 1 { ixy + i2 } else { ixy };
            vq[v(t)] += w;
            let t = if st(ixy - i2) >= 1 { ixy - i2 } else { ixy };
            vq[v(t)] += w;
            // corner 3
            let ixc = if i3 >= 0 { ny } else { -ny };
            let iyc = i3 - ixc;
            let w = a * rd4 * rd3 / 6.0;
            let t = if st(ixy + i3) >= 1 && !(st(ixy + ixc) <= 0 && st(ixy + iyc) <= 0) { ixy + i3 } else { ixy };
            vq[v(t)] += w;
            let t = if st(ixy - i3) >= 1 && !(st(ixy - ixc) <= 0 && st(ixy - iyc) <= 0) { ixy - i3 } else { ixy };
            vq[v(t)] += w;
            let w = a * (1.0 - rd3) * rd4 / 6.0;
            let t = if st(ixy + i4) >= 1 { ixy + i4 } else { ixy };
            vq[v(t)] += w;
            let t = if st(ixy - i4) >= 1 { ixy - i4 } else { ixy };
            vq[v(t)] += w;
        }
        // 3.a.3 boundary points keep their value
        for k in 1..=nx * ny {
            if g.mapsta[(k - 1) as usize] == 2 {
                vq[v(k)] = aq[v(k)];
            }
        }
        // 3.a.4 closure of the averaging (the propagation is closed in
        // W3QCK3)
        if global {
            for iy in 1..=ny {
                vq[v(iy)] += vq[v(nx * ny + iy)];
                vq[v((nx - 1) * ny + iy)] += vq[v(iy - ny)];
            }
        }
        // 3.b propagation in the straightened space
        for isea in 0..nsea {
            vq[v(g.mapsf[isea][2])] *= g.gsqrt[isea];
        }
        let sweep_y = |vq: &mut [f64], vlcfly: &mut [f64]| {
            if p.flcy {
                qck3(g.nx, g.ny, g.nx, g.ny, vlcfly, Some(g.atrny.as_slice()), &mut vq[qlo..], false, 1, &g.mapaxy, &g.mapy2, g.nmy);
            }
        };
        let sweep_x = |vq: &mut [f64], vlcflx: &mut [f64]| {
            if p.flcx {
                qck3(g.nx, g.ny, g.nx, g.ny, vlcflx, Some(g.atrnx.as_slice()), &mut vq[qlo..], global, ny, &g.mapaxy, &g.mapx2, g.nmx);
            }
        };
        if yfirst {
            sweep_y(&mut vq, &mut vlcfly);
            sweep_x(&mut vq, &mut vlcflx);
        } else {
            sweep_x(&mut vq, &mut vlcflx);
            sweep_y(&mut vq, &mut vlcfly);
        }
        for isea in 0..nsea {
            vq[v(g.mapsf[isea][2])] /= g.gsqrt[isea];
        }
        // 3.c boundary spectra, interpolated at the end of the sub-step
        if let Some(b) = bpi {
            let mut rd1 = b.dt0 - dtg * (ntloc - itloc) as f64 / ntloc as f64;
            let mut rd2 = b.dtn;
            if rd2 > 0.001 {
                rd2 = (rd1 / rd2).max(0.0).min(1.0);
                rd1 = 1.0 - rd2;
            } else {
                rd1 = 0.0;
                rd2 = 1.0;
            }
            for (ibi, &is) in b.isbpi.iter().enumerate() {
                let ixy = g.mapsf[is][2];
                vq[v(ixy)] = (rd1 * b.bbpi0[ibi] + rd2 * b.bbpin[ibi]) / cg[is] * g.clats[is];
            }
        }
        yfirst = !yfirst;
    }

    // 4. back to action density (W3SCAT: written where MAPSTA /= 0)
    for isea in 0..nsea {
        let ixy = g.mapsf[isea][2];
        let m = g.mapsta[(ixy - 1) as usize];
        let mut x = vq[v(ixy)];
        if m > 0 {
            x = (cg[isea] / g.clats[isea] * x).max(0.0);
        }
        if m != 0 {
            field[g.sea_ij[isea]] = x;
        }
    }
}

/// `W3CFLXY`: the largest Courant number of the geographic propagation at
/// sea point `isea` (0-based) over `dtg`, for the group velocities `cg`
/// (`CG(1:NK, ISEA)`) and the current `cur` (`CX`, `CY`) of that point.
/// Fortran: a diagnostic only; the signed numbers enter the maximum (no
/// absolute value, so that 0 is a lower bound) and the time step is not
/// turned into degrees (no division by `DERA RADIUS`), unlike in `W3XYP3`.
pub fn cflxy(g: &Pro3Grid, isea: usize, dtg: f64, cg: &[f64], cur: Option<(f64, f64)>) -> f64 {
    let mut cflxymax = 0.0f64;
    for ik in 0..g.nk {
        for ith in 0..g.nth {
            let mut cxtot = g.ecos[ith] * cg[ik] / g.clats[isea];
            let mut cytot = g.esin[ith] * cg[ik];
            if let Some((cx, cy)) = cur {
                cxtot += cx / g.clats[isea];
                cytot += cy;
            }
            let cp = cxtot * g.dpdx[isea] + cytot * g.dpdy[isea];
            let cq = cxtot * g.dqdx[isea] + cytot * g.dqdy[isea];
            cflxymax = (cp * dtg).max(cq * dtg).max(cflxymax);
        }
    }
    cflxymax
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Spectrum;
    use crate::structgrid::GridType;

    const SPEC: Spectrum = Spectrum { nk: 4, nth: 8, freq1: 0.05, xfr: 1.1, thoff: 0.0, dmin: 1.0 };

    /// A RECT grid of `nx × ny` points from `(x0, y0)` with steps `s`
    /// (degrees), land where `land(ix, iy)`, the edges excluded as
    /// `ww3_grid` does.
    fn rect(nx: usize, ny: usize, x0: f64, y0: f64, s: f64, closed: bool, land: impl Fn(usize, usize) -> bool) -> StructGrid {
        let (mut lon, mut lat, mut zb, mut mapsta) = (vec![], vec![], vec![], vec![]);
        for iy in 0..ny {
            for ix in 0..nx {
                lon.push(x0 + s * ix as f64);
                lat.push(y0 + s * iy as f64);
                let l = land(ix, iy);
                zb.push(if l { 10.0 } else { -100.0 });
                mapsta.push(if l { 0 } else { 1 });
            }
        }
        let mut g = StructGrid { name: "T".into(), gtype: GridType::Rect, nx, ny, closed, lon, lat, zb, mapsta, spec: SPEC };
        g.exclude_edges();
        g
    }

    /// A Gaussian bump centred at `(cx, cy)` (grid indices) at the sea
    /// points.
    fn bump(p: &Pro3Grid, cx: f64, cy: f64, w: f64) -> Vec<f64> {
        let mut f = vec![0.0; p.nx * p.ny];
        for &i in &p.sea_ij {
            let (ix, iy) = ((i % p.nx) as f64, (i / p.nx) as f64);
            f[i] = (-((ix - cx).powi(2) + (iy - cy).powi(2)) / (w * w)).exp();
        }
        f
    }

    /// `sum A CLATS` over the sea points: what the scheme conserves with a
    /// uniform group velocity on a RECT grid.
    fn mass(p: &Pro3Grid, f: &[f64]) -> f64 {
        p.sea_ij.iter().zip(&p.clats).map(|(&i, c)| f[i] * c).sum()
    }

    /// Run `n` global steps of `dtg` for component `isp` with a uniform
    /// group velocity `c`.
    fn run(p: &Pro3Grid, f: &mut [f64], isp: usize, c: f64, dtg: f64, n: usize) {
        let cg = vec![c; p.nsea];
        for s in 0..n {
            xyp3(p, isp, dtg, (s + 1) as f64 * dtg % 86400.0, f, &cg, None, None);
        }
    }

    fn params(dtcfl: f64) -> Pro3Params {
        Pro3Params { dtcfl, ..Pro3Params::default() }
    }

    /// A bump going east across the seam of a closed RECT grid near the
    /// equator conserves the action to round-off and moves by `c t`.
    #[test]
    fn closed_rect_conserves_and_moves() {
        let (nx, ny, s) = (64, 12, 0.25);
        let g = rect(nx, ny, 0.0, -1.375, s, true, |_, _| false);
        let p = Pro3Grid::new(&g, None, &params(300.0)).unwrap();
        assert!(substeps(&p, 0, 900.0, None) > 1);
        let (x0, c, dtg, n) = (56.0, 10.0, 900.0, 24);
        let mut f = bump(&p, x0, 5.5, 2.5);
        let m0 = mass(&p, &f);
        run(&p, &mut f, 0, c, dtg, n);
        let m1 = mass(&p, &f);
        assert!(((m1 - m0) / m0).abs() < 1e-12, "mass {m0} -> {m1}");
        // centroid, unwrapped around the starting point
        let (mut sx, mut sw) = (0.0, 0.0);
        for (&i, cl) in p.sea_ij.iter().zip(&p.clats) {
            let ix = (i % nx) as f64;
            let xu = (ix - x0 + 1.5 * nx as f64).rem_euclid(nx as f64) - 0.5 * nx as f64;
            sx += xu * f[i] * cl;
            sw += f[i] * cl;
        }
        let shift = sx / sw;
        let expect = c * dtg * n as f64 / (RADIUS * DERA * s);
        assert!(((shift - expect) / expect).abs() < 0.01, "shift {shift} cells, expected {expect}");
        assert!(f.iter().all(|&x| x >= 0.0));
    }

    /// A land column stops the waves: nothing beyond it, and the action
    /// that reaches it is absorbed.
    #[test]
    fn land_column_blocks() {
        let (nx, ny) = (48, 10);
        let g = rect(nx, ny, 0.0, -1.125, 0.25, false, |ix, _| ix == 25);
        let p = Pro3Grid::new(&g, None, &params(300.0)).unwrap();
        let mut f = bump(&p, 15.0, 4.5, 2.0);
        // nothing east of the land column at the start
        f.iter_mut().enumerate().filter(|(i, _)| i % nx >= 25).for_each(|(_, x)| *x = 0.0);
        let m0 = mass(&p, &f);
        run(&p, &mut f, 0, 10.0, 900.0, 48);
        for iy in 0..ny {
            for ix in 25..nx {
                assert_eq!(f[iy * nx + ix], 0.0, "({ix}, {iy})");
            }
        }
        let m1 = mass(&p, &f);
        assert!(m1 < 0.5 * m0, "mass {m0} -> {m1}");
    }

    /// The GSE averaging spreads a single cell across the propagation
    /// (along `y` for waves going east), conserving the action.
    #[test]
    fn gse_averaging_smooths() {
        let (nx, ny) = (64, 12);
        let g = rect(nx, ny, 0.0, -1.375, 0.25, true, |_, _| false);
        let mut out = vec![];
        for w in [0.0, 1.5] {
            let pp = Pro3Params { wdthcg: w, wdthth: w, ..params(300.0) };
            let p = Pro3Grid::new(&g, None, &pp).unwrap();
            let mut f = vec![0.0; nx * ny];
            f[6 * nx + 20] = 1.0;
            let m0 = mass(&p, &f);
            run(&p, &mut f, 0, 10.0, 900.0, 8);
            let m1 = mass(&p, &f);
            assert!(((m1 - m0) / m0).abs() < 1e-12, "w {w}: mass {m0} -> {m1}");
            let rows = (0..ny).filter(|&iy| (0..nx).any(|ix| f[iy * nx + ix] > 1e-12)).count();
            let fmax = f.iter().cloned().fold(0.0, f64::max);
            out.push((rows, fmax));
        }
        assert_eq!(out[0].0, 1, "without averaging the cell stays in its row");
        assert!(out[1].0 >= 3, "rows reached with averaging: {}", out[1].0);
        assert!(out[1].1 < out[0].1, "maxima {:?}", out);
    }

    /// Obstructions at the cell centres (`FLAGTR = 2`): a column of
    /// transparency 0 blocks, transparency 1 everywhere changes nothing.
    #[test]
    fn obstruction_transparency() {
        // (not closed: on a closed grid the averaging carries a trace of
        // the western tail of the bump westwards across the seam)
        let (nx, ny) = (64, 10);
        let g = rect(nx, ny, 0.0, -1.125, 0.25, false, |_, _| false);
        let run_trn = |col0: bool, flagtr: i64| {
            let trnx: Vec<f64> = (0..nx * ny).map(|i| if col0 && i % nx == 30 { 0.0 } else { 1.0 }).collect();
            let t = Transparencies { trnx, trny: vec![1.0; nx * ny] };
            let pp = Pro3Params { flagtr, ..params(300.0) };
            let p = Pro3Grid::new(&g, Some(&t), &pp).unwrap();
            let mut f = bump(&p, 20.0, 4.5, 2.0);
            // nothing east of the obstruction at the start
            f.iter_mut().enumerate().filter(|(i, _)| i % nx >= 30).for_each(|(_, x)| *x = 0.0);
            run(&p, &mut f, 0, 10.0, 900.0, 48);
            (p, f)
        };
        let (p, f) = run_trn(true, 2);
        assert!(p.maptrn(29, 4) && p.maptrn(30, 4) && p.maptrn(31, 4) && !p.maptrn(20, 4));
        assert_eq!(p.atrnx(31, 4).0, 0.0);
        for iy in 0..ny {
            for ix in 31..40 {
                assert_eq!(f[iy * nx + ix], 0.0, "({ix}, {iy})");
            }
        }
        assert!(f[4 * nx + 29] > 0.0);
        let (_, f1) = run_trn(false, 2);
        let (_, f0) = run_trn(false, 0);
        assert_eq!(f1, f0, "transparency 1 must change nothing");
    }

    /// The same grid given as CURV (coordinates at each point) gives the
    /// result of the RECT grid up to the finite differences of the metric.
    #[test]
    fn curv_matches_rect() {
        let (nx, ny) = (40, 14);
        let gr = rect(nx, ny, 10.0, 30.0, 0.25, false, |ix, iy| ix == 30 && iy < 8);
        let mut gc = gr.clone();
        gc.gtype = GridType::Curv;
        let pr = Pro3Grid::new(&gr, None, &params(300.0)).unwrap();
        let pc = Pro3Grid::new(&gc, None, &params(300.0)).unwrap();
        for isp in [0, 1, 2 * 8 + 3] {
            let mut fr = bump(&pr, 15.0, 6.5, 2.0);
            let mut fc = fr.clone();
            run(&pr, &mut fr, isp, 8.0, 900.0, 12);
            run(&pc, &mut fc, isp, 8.0, 900.0, 12);
            let fmax = fr.iter().cloned().fold(0.0, f64::max);
            let d = fr.iter().zip(&fc).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(d < 1e-4 * fmax, "isp {isp}: {d} against {fmax}");
        }
    }

    /// `WAVNU1` of the table of `DISTAB` against the Newton iterations of
    /// `WAVNU2` in shallow and intermediate water, and deep water beyond the
    /// table.
    #[test]
    fn wavnu1_table() {
        let t = DispTable::new();
        for (si, h) in [(0.2f32, 5.0f32), (0.5, 20.0), (1.0, 3.0), (0.3, 100.0)] {
            let (k1, c1) = t.wavnu1(si, h);
            let (k2, c2) = wavnu2(si, h, 1e-7, 15);
            assert!((k1 / k2 - 1.0).abs() < 2e-3 && (c1 / c2 - 1.0).abs() < 2e-3, "si {si} h {h}: {k1} {k2} {c1} {c2}");
        }
        let (k, c) = t.wavnu1(1.0, 4000.0);
        assert_eq!((k, c), (1.0 / GRAV as f32, 0.5 * GRAV as f32));
    }

    /// `W3QCK1` and `W3QCK3` with unit transparencies agree, and a uniform
    /// field stays uniform in the interior of a closed row.
    #[test]
    fn qck1_is_qck3_with_unit_transparencies() {
        let g = rect(16, 5, 0.0, -0.5, 0.25, true, |_, _| false);
        let p = Pro3Grid::new(&g, None, &params(300.0)).unwrap();
        let (nx, ny) = (16i64, 5i64);
        let mut cfl = vec![0.3; ((nx + 1) * ny) as usize];
        let mut q1 = vec![0.0; (ny * (nx + 3)) as usize];
        for ix in 0..nx {
            for iy in 0..ny {
                q1[(ix * ny + iy + ny) as usize] = 1.0 + (ix as f64 * 0.7).sin();
            }
        }
        let mut q3 = q1.clone();
        let mut cfl3 = cfl.clone();
        qck1(16, 5, 16, 5, &mut cfl, &mut q1, true, ny, &p.mapaxy, &p.mapx2, p.nmx);
        let ones = vec![[1.0; 3]; (nx * ny) as usize];
        qck3(16, 5, 16, 5, &mut cfl3, Some(&ones), &mut q3, true, ny, &p.mapaxy, &p.mapx2, p.nmx);
        assert_eq!(q1, q3);
        let mut q = vec![2.0; (ny * (nx + 3)) as usize];
        qck1(16, 5, 16, 5, &mut cfl, &mut q, true, ny, &p.mapaxy, &p.mapx2, p.nmx);
        assert!(q.iter().all(|&x| (x - 2.0).abs() < 1e-14));
    }
}
