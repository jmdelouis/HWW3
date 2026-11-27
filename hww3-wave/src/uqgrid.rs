//! The host side of the UQ/PR3 propagation of WW3's structured grids on the
//! GPU ([`crate::prop_uq`]): the grid directory of that scheme, the tables
//! its kernels read, the global steps of `W3WAVE`, and a CPU mirror of the
//! kernels checked against [`crate::pro3::xyp3`].
//!
//! # Grid directory
//!
//! `hww3-pre struct WW3_GRID.nml --data DIR --out GRID_DIR --scheme uq
//! --nml NAMELISTS` writes ([`UqSetup::write`]):
//! - the points of the model, every sea point of the grid (`MAPSTA` 1 or 2)
//!   in WW3's order (`ISEA`, `ix` fastest), as an HWW3 grid directory
//!   without triangles ([`crate::grid::Grid::read`]): `raw/lon.f64`,
//!   `raw/lat.f64` (geographic), `raw/depth.f64` (`-ZB`), an empty
//!   `raw/tri.u32`, `raw/obc.u32` (the points of `MAPSTA = 2`) and
//!   `raw/sea_ij.u32` (index `iy NX + ix` of each sea point);
//! - the structured grid itself: `raw/uq_grid.txt` (`NX NY CLOSED TYPE
//!   NAME`), `raw/uq_lon.f64`, `raw/uq_lat.f64`, `raw/uq_zb.f64` and
//!   `raw/uq_mapsta.u8` on the whole `NX x NY` grid, and the transparencies
//!   `raw/uq_trnx.f64`, `raw/uq_trny.f64` when `FLAGTR > 0`;
//! - `ww3_grid.nml` (the grid namelist) and `uq_params.nml` (the namelists
//!   of the run, for `&PRO3` and `&MISC`).
//!
//! The presence of `raw/uq_grid.txt` makes `hww3-prop run` use the UQ
//! scheme ([`UqSetup::is_uq_dir`]).
//!
//! # Tables of the kernels
//!
//! The kernels work on the whole grid in WW3's layout `k = ix NY + iy`
//! (`IXY - 1`), one direction after another ([`UqTables`]). `W3XYP3`
//! spreads the action of each sea point onto its eight neighbours (the GSE
//! averaging): the kernels gather it instead, each point summing what its
//! neighbours send it, which needs per point the wet mask of its 3 x 3
//! neighbourhood, its class (central or near-coast point of `W3MAP3`, `MAPTRN`)
//! and its position in `MAPCXY` (a point of `MAPTRN` drops what the points
//! processed before it sent, [`UqTables::gfl`]). The `x` and `y` sweeps of
//! `W3QCK3` compute the two fluxes of each active point from the values
//! before the sweep, in parallel over the points.
//!
//! The kernels need no sea point on the first and last rows, nor on the
//! first and last columns of a grid not closed in longitude (as after
//! `ww3_grid`'s exclusion of the edges without active boundary points):
//! the averaging and the stencils of `W3XYP3` then never go through the
//! one-dimensional wrap of WW3's index from a column to the next, nor into
//! halo columns that are not folded back. [`UqTables::new`] checks it.

use crate::grid::Spectrum;
use crate::pro3::{substeps, Pro3Grid, Pro3Params};
use crate::ref1::Transparencies;
use crate::structgrid::{spectrum, GridType, StructGrid};
use crate::ww3grid::Namelists;
use anyhow::{bail, ensure, Context, Result};
use std::path::Path;

/// `DERA` of `constants.F90`.
const DERA: f64 = std::f64::consts::PI / 180.0;
/// `RADIUS` of `constants.F90` (m).
const RADIUS: f64 = 4.0e7 / (2.0 * std::f64::consts::PI);

/// Bits 0 to 8 of [`UqTables::gfl`]: the wet mask (`MAPSTA >= 1`) of the 3 x 3
/// neighbourhood, bit `3 (dy + 1) + dx + 1` for the neighbour `(ix + dx, iy + dy)`.
pub const GF_WET: u32 = 0x1ff;
/// `MAPSTA = 1`.
pub const GF_MS1: u32 = 1 << 9;
/// `MAPSTA > 0`.
pub const GF_POS: u32 = 1 << 10;
/// Sea point (`MAPSTA /= 0`).
pub const GF_SEA: u32 = 1 << 11;
/// `MAPTRN`: no averaging at this point.
pub const GF_TRN: u32 = 1 << 12;
/// Central point of the averaging (`MAPCXY(1:NCENT)`).
pub const GF_CEN: u32 = 1 << 13;
/// `MAPSTA = 2`: keeps its value through the averaging.
pub const GF_MS2: u32 = 1 << 14;
/// Active point of the sweeps (`MAPAXY`: `MAPSTA = 1` off the first and last rows).
pub const GF_ACT: u32 = 1 << 15;

/// Values per spectral component in the parameter table of [`ptab`].
pub const PT: usize = 8;

/// A structured grid prepared for the UQ scheme: the grid, its
/// transparencies and the parameters of `ww3_grid` the model needs.
#[derive(Clone, Debug)]
pub struct UqSetup {
    /// The structured grid (after `ww3_grid`'s corrections of `MAPSTA`).
    pub sg: StructGrid,
    /// `TRNX`, `TRNY` when `FLAGTR > 0`.
    pub trn: Option<Transparencies>,
    /// The parameters of `W3XYP3`, `W3UTRN`, `W3MAPT`.
    pub params: Pro3Params,
    /// `TIMESTEPS%DTMAX` (s): the largest global step.
    pub dtmax: f64,
    /// `TIMESTEPS%DTKTH` (s): the largest step of `W3KTP3`.
    pub dtkth: f64,
    /// `CFLTM` of `&PRO3` (default 0.7): the CFL limit of `W3KTP3`.
    pub cfltm: f64,
}

fn write_f64s(p: &Path, v: &[f64]) -> Result<()> {
    std::fs::write(p, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).with_context(|| p.display().to_string())
}

fn read_f64s(p: &Path, n: usize) -> Result<Vec<f64>> {
    let b = std::fs::read(p).with_context(|| p.display().to_string())?;
    ensure!(b.len() == 8 * n, "{}: {} bytes for {n} values", p.display(), b.len());
    Ok(b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
}

impl UqSetup {
    /// Whether `dir` is a grid directory of the UQ scheme.
    pub fn is_uq_dir(dir: &Path) -> bool {
        dir.join("raw").join("uq_grid.txt").exists()
    }

    /// Write the grid directory of the UQ scheme (see the module
    /// documentation) for the grid `sg`, its transparencies `trn`, the text
    /// of its namelist `grid_nml` and that of the namelists of the run
    /// `params_nml`.
    pub fn write(dir: &Path, sg: &StructGrid, trn: Option<&Transparencies>, grid_nml: &str, params_nml: &str) -> Result<()> {
        let raw = dir.join("raw");
        std::fs::create_dir_all(&raw).with_context(|| raw.display().to_string())?;
        let sea: Vec<usize> = (0..sg.nx * sg.ny).filter(|&i| sg.is_sea(i)).collect();
        let pick = |a: &[f64]| sea.iter().map(|&i| a[i]).collect::<Vec<f64>>();
        write_f64s(&raw.join("lon.f64"), &pick(&sg.lon))?;
        write_f64s(&raw.join("lat.f64"), &pick(&sg.lat))?;
        write_f64s(&raw.join("depth.f64"), &sea.iter().map(|&i| -sg.zb[i]).collect::<Vec<f64>>())?;
        std::fs::write(raw.join("tri.u32"), [])?;
        let u32s = |name: &str, v: &[u32]| std::fs::write(raw.join(name), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        let obc: Vec<u32> = sea.iter().enumerate().filter(|&(_, &i)| sg.mapsta[i] == 2).map(|(k, _)| k as u32).collect();
        u32s("obc.u32", &obc)?;
        u32s("sea_ij.u32", &sea.iter().map(|&i| i as u32).collect::<Vec<u32>>())?;
        let gt = match sg.gtype {
            GridType::Rect => "RECT",
            GridType::Curv => "CURV",
        };
        std::fs::write(raw.join("uq_grid.txt"), format!("{} {} {} {} {}\n", sg.nx, sg.ny, sg.closed as u8, gt, sg.name))?;
        write_f64s(&raw.join("uq_lon.f64"), &sg.lon)?;
        write_f64s(&raw.join("uq_lat.f64"), &sg.lat)?;
        write_f64s(&raw.join("uq_zb.f64"), &sg.zb)?;
        std::fs::write(raw.join("uq_mapsta.u8"), &sg.mapsta)?;
        if let Some(t) = trn {
            write_f64s(&raw.join("uq_trnx.f64"), &t.trnx)?;
            write_f64s(&raw.join("uq_trny.f64"), &t.trny)?;
        }
        std::fs::write(dir.join("ww3_grid.nml"), grid_nml)?;
        std::fs::write(dir.join("uq_params.nml"), params_nml)?;
        Ok(())
    }

    /// Read a grid directory written by [`UqSetup::write`].
    pub fn read(dir: &Path) -> Result<UqSetup> {
        let raw = dir.join("raw");
        let head = std::fs::read_to_string(raw.join("uq_grid.txt")).with_context(|| format!("{}: not a grid directory of the UQ scheme", dir.display()))?;
        let w: Vec<&str> = head.split_whitespace().collect();
        ensure!(w.len() >= 4, "{}/raw/uq_grid.txt: {head:?}", dir.display());
        let (nx, ny): (usize, usize) = (w[0].parse()?, w[1].parse()?);
        let closed = w[2] == "1";
        let gtype = match w[3] {
            "RECT" => GridType::Rect,
            "CURV" => GridType::Curv,
            t => bail!("grid type {t}"),
        };
        let name = w.get(4).map(|s| s.to_string()).unwrap_or_default();
        let n = nx * ny;
        let gn = Namelists::read(&dir.join("ww3_grid.nml"))?;
        let pn = Namelists::read(&dir.join("uq_params.nml"))?;
        let spec: Spectrum = spectrum(&gn)?;
        let mapsta = std::fs::read(raw.join("uq_mapsta.u8"))?;
        ensure!(mapsta.len() == n, "uq_mapsta.u8: {} values for {nx} x {ny}", mapsta.len());
        let sg = StructGrid {
            name,
            gtype,
            nx,
            ny,
            closed,
            lon: read_f64s(&raw.join("uq_lon.f64"), n)?,
            lat: read_f64s(&raw.join("uq_lat.f64"), n)?,
            zb: read_f64s(&raw.join("uq_zb.f64"), n)?,
            mapsta,
            spec,
        };
        let trn = if raw.join("uq_trnx.f64").exists() {
            Some(Transparencies { trnx: read_f64s(&raw.join("uq_trnx.f64"), n)?, trny: read_f64s(&raw.join("uq_trny.f64"), n)? })
        } else {
            None
        };
        let params = Pro3Params::from_namelists(&gn, &pn)?;
        ensure!(params.flagtr == 0 || trn.is_some(), "FLAGTR = {} without uq_trnx.f64", params.flagtr);
        Ok(UqSetup {
            sg,
            trn,
            params,
            dtmax: gn.f("TIMESTEPS_NML", "TIMESTEPS%DTMAX", f64::NAN)?,
            dtkth: gn.f("TIMESTEPS_NML", "TIMESTEPS%DTKTH", f64::NAN)?,
            cfltm: pn.f("PRO3", "CFLTM", 0.7)?,
        })
    }

    /// `W3MAP3`, `W3UTRN` and `W3MAPT` of the grid ([`Pro3Grid::new`]).
    pub fn pro3(&self) -> Result<Pro3Grid> {
        Pro3Grid::new(&self.sg, self.trn.as_ref(), &self.params)
    }
}

/// The tables of the UQ kernels on one grid: the whole grid in WW3's layout
/// `k = ix NY + iy`, the sea points in WW3's order.
#[derive(Clone, Debug)]
pub struct UqTables {
    /// Points along a row.
    pub nx: usize,
    /// Rows.
    pub ny: usize,
    /// `NX NY`.
    pub ng: usize,
    /// Sea points.
    pub nsea: usize,
    /// Closed in longitude (`SMPL`).
    pub closed: bool,
    /// Sea point at `k` (`u32::MAX` off the sea).
    pub gsea: Vec<u32>,
    /// `k` of each sea point.
    pub ksea: Vec<u32>,
    /// Flags of each point `k` (`GF_*`).
    pub gfl: Vec<u32>,
    /// Position of each point `k` in `MAPCXY` (0 off the sea).
    pub gpos: Vec<u32>,
    /// `ATRNX(-1)`, `ATRNX(1)`, `ATRNY(-1)`, `ATRNY(1)` of each sea point.
    pub atrn: Vec<f64>,
    /// `CLATS`, `DPDX`, `DPDY`, `DQDX`, `DQDY`, `GSQRT` of each sea point.
    pub metric: [Vec<f64>; 6],
    /// The stencil of `W3DZXY` per sea point: the sea points `IXP`, `IXM`,
    /// `IYP`, `IYM`.
    pub dzi: Vec<u32>,
    /// Its coefficients: `DZ/DX = (Z(IXP) - Z(IXM)) c0 + (Z(IYP) - Z(IYM)) c1`,
    /// `DZ/DY = (Z(IXP) - Z(IXM)) c2 + (Z(IYP) - Z(IYM)) c3`.
    pub dzc: Vec<f64>,
}

impl UqTables {
    /// The tables of `pg` (see [`UqTables::update`] for those that follow
    /// `MAPSTA` and the ice).
    pub fn new(pg: &Pro3Grid) -> Result<UqTables> {
        let (nx, ny) = (pg.nx, pg.ny);
        let ng = nx * ny;
        let nsea = pg.nsea;
        let mut gsea = vec![u32::MAX; ng];
        let mut ksea = vec![0u32; nsea];
        for (isea, &i) in pg.sea_ij.iter().enumerate() {
            let (ix, iy) = (i % nx, i / nx);
            if iy == 0 || iy == ny - 1 || (!pg.closed && (ix == 0 || ix == nx - 1)) {
                bail!("sea point {isea} at ({}, {}) on the edge of the grid: the UQ kernels need no sea point on the first and last rows, nor on the first and last columns of a grid not closed in longitude", ix + 1, iy + 1);
            }
            gsea[ix * ny + iy] = isea as u32;
            ksea[isea] = (ix * ny + iy) as u32;
        }
        let metric = [pg.clats.clone(), pg.dpdx.clone(), pg.dpdy.clone(), pg.dqdx.clone(), pg.dqdy.clone(), pg.gsqrt.clone()];
        // W3DZXY (the sea points do not change)
        let dfac = 1.0 / (DERA * RADIUS);
        let (mut dzi, mut dzc) = (vec![0u32; 4 * nsea], vec![0.0f64; 4 * nsea]);
        let sea = |ix: usize, iy: usize| gsea[ix * ny + iy] != u32::MAX;
        for isea in 0..nsea {
            let i = pg.sea_ij[isea];
            let (ix, iy) = (i % nx, i / nx);
            let me = isea as u32;
            dzi[4 * isea..4 * isea + 4].copy_from_slice(&[me; 4]);
            if iy < 1 || iy > ny - 2 || (!pg.closed && (ix < 1 || ix > nx - 2)) {
                continue;
            }
            let (ixps, ixms) = ((ix + 1) % nx, (ix + nx - 1) % nx);
            let mut stx = 0.5;
            let ixp = if sea(ixps, iy) { ixps } else { stx = 1.0; ix };
            let ixm = if sea(ixms, iy) { ixms } else { stx = 1.0; ix };
            let mut sty = 0.5;
            let iyp = if sea(ix, iy + 1) { iy + 1 } else { sty = 1.0; iy };
            let iym = if sea(ix, iy - 1) { iy - 1 } else { sty = 1.0; iy };
            dzi[4 * isea] = gsea[ixp * ny + iy];
            dzi[4 * isea + 1] = gsea[ixm * ny + iy];
            dzi[4 * isea + 2] = gsea[ix * ny + iyp];
            dzi[4 * isea + 3] = gsea[ix * ny + iym];
            dzc[4 * isea] = stx * pg.dpdx[isea] * dfac;
            dzc[4 * isea + 1] = sty * pg.dqdx[isea] * dfac;
            dzc[4 * isea + 2] = stx * pg.dpdy[isea] * dfac;
            dzc[4 * isea + 3] = sty * pg.dqdy[isea] * dfac;
        }
        let mut t = UqTables {
            nx,
            ny,
            ng,
            nsea,
            closed: pg.closed,
            gsea,
            ksea,
            gfl: vec![0; ng],
            gpos: vec![0; ng],
            atrn: vec![1.0; 4 * nsea],
            metric,
            dzi,
            dzc,
        };
        t.update(pg);
        Ok(t)
    }

    /// The tables that follow `MAPSTA` and the ice (after
    /// [`Pro3Grid::update_maps`]): the flags, the positions in `MAPCXY` and
    /// the transparencies.
    pub fn update(&mut self, pg: &Pro3Grid) {
        let (nx, ny) = (self.nx, self.ny);
        let closed = self.closed;
        let wet = |ix: i64, iy: i64| -> bool {
            if iy < 0 || iy >= ny as i64 {
                return false;
            }
            let jx = if closed {
                ix.rem_euclid(nx as i64)
            } else if ix < 0 || ix >= nx as i64 {
                return false;
            } else {
                ix
            };
            pg.mapsta(jx as usize, iy as usize) >= 1
        };
        let mut central = vec![false; self.nsea];
        for &s in &pg.mapcxy()[..pg.ncent()] {
            central[s] = true;
        }
        for (p, &s) in pg.mapcxy().iter().enumerate() {
            self.gpos[self.ksea[s] as usize] = p as u32;
        }
        for ix in 0..nx {
            for iy in 0..ny {
                let k = ix * ny + iy;
                let m = pg.mapsta(ix, iy);
                let mut f = 0u32;
                for dy in 0..3i64 {
                    for dx in 0..3i64 {
                        if wet(ix as i64 + dx - 1, iy as i64 + dy - 1) {
                            f |= 1 << (3 * dy + dx);
                        }
                    }
                }
                if m == 1 {
                    f |= GF_MS1;
                    if iy >= 1 && iy + 2 <= ny {
                        f |= GF_ACT;
                    }
                }
                if m > 0 {
                    f |= GF_POS;
                }
                if m == 2 {
                    f |= GF_MS2;
                }
                let s = self.gsea[k];
                if s != u32::MAX {
                    f |= GF_SEA;
                    if pg.maptrn(ix, iy) {
                        f |= GF_TRN;
                    }
                    if central[s as usize] {
                        f |= GF_CEN;
                    }
                    let (xm, xp) = pg.atrnx(ix, iy);
                    let (ym, yp) = pg.atrny(ix, iy);
                    self.atrn[4 * s as usize..4 * s as usize + 4].copy_from_slice(&[xm, xp, ym, yp]);
                }
                self.gfl[k] = f;
            }
        }
    }

    /// `W3DZXY`: the gradients `(DZ/DX, DZ/DY)` (per metre) of `z` given at
    /// the sea points.
    pub fn dzxy(&self, z: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut dx = vec![0.0; self.nsea];
        let mut dy = vec![0.0; self.nsea];
        for i in 0..self.nsea {
            let d = &self.dzi[4 * i..4 * i + 4];
            let c = &self.dzc[4 * i..4 * i + 4];
            let a = z[d[0] as usize] - z[d[1] as usize];
            let b = z[d[2] as usize] - z[d[3] as usize];
            dx[i] = a * c[0] + b * c[1];
            dy[i] = a * c[2] + b * c[3];
        }
        (dx, dy)
    }
}

/// The parameters of `W3XYP3` for every spectral component `isp = ith + ik
/// NTH` over the global step `dtg` (s) with the currents `cur` (`CX`, `CY`
/// of the sea points): [`PT`] values per component, `NTLOC`, `DTLOC / (DERA
/// RADIUS)`, the widths of the averaging `FTHX`, `FTHY`, `FCGX`, `FCGY`
/// (section 1.c of `W3XYP3`, divided by `RADIUS DERA`, `FCG` also by
/// `NTLOC`), `ECOS`, `ESIN`; and `NTLOC` of each component.
pub fn ptab(pg: &Pro3Grid, dtg: f64, cur: Option<(&[f64], &[f64])>) -> (Vec<f64>, Vec<usize>) {
    let nspec = pg.nk * pg.nth;
    let p = &pg.params;
    let mut t = vec![0.0; PT * nspec];
    let mut nt = vec![0usize; nspec];
    for isp in 0..nspec {
        let ith = isp % pg.nth;
        let ntloc = substeps(pg, isp, dtg, cur);
        let dtloc = dtg / ntloc as f64;
        let dtrad = dtloc / (DERA * RADIUS);
        let mut fth = p.wdthth * pg.dth * dtloc;
        let mut fcg = p.wdthcg * 0.5 * (pg.xfr - 1.0 / pg.xfr) * dtloc;
        fth = fth / RADIUS / DERA;
        fcg = fcg / RADIUS / DERA;
        fcg /= ntloc as f64;
        let (ec, es) = (pg.ecos[ith], pg.esin[ith]);
        t[PT * isp..PT * (isp + 1)].copy_from_slice(&[ntloc as f64, dtrad, -fth * es, fth * ec, fcg * ec, fcg * es, ec, es]);
        nt[isp] = ntloc;
    }
    (t, nt)
}

/// `YFIRST` of the first sub-step of `W3XYP3` for a global step `dtg`
/// ending `secs_of_day` seconds after 00:00.
pub fn yfirst(secs_of_day: f64, dtg: f64) -> bool {
    ((secs_of_day.trunc() / dtg).round() as i64).rem_euclid(2) == 0
}

/// The global steps of `W3WAVE` (section 2): each call, from `TIME` to
/// `TCALC` (the end of the call, an output or input time), in `NT = 1 +
/// INT(DTTST / DTMAX - 0.001)` equal steps rounded to the second with the
/// remainder carried, in single precision as in WW3.
#[derive(Clone, Debug)]
pub struct W3Steps {
    /// `DTMAX` (s).
    pub dtmax: f64,
    left: i64,
    dtga: f32,
    dtres: f32,
}

impl W3Steps {
    /// No call started yet.
    pub fn new(dtmax: f64) -> W3Steps {
        W3Steps { dtmax, left: 0, dtga: 0.0, dtres: 0.0 }
    }

    /// The next global step (s) from `t` (s): when the previous call is
    /// over, a call to `end` (s) starts.
    pub fn next(&mut self, t: i64, end: i64) -> f32 {
        if self.left == 0 {
            let dttst = (end - t).max(1) as f32;
            self.left = 1 + (dttst / self.dtmax as f32 - 0.001) as i64;
            self.dtga = dttst / self.left as f32;
            self.dtres = 0.0;
        }
        let dtg = (self.dtga + self.dtres + 0.0001).round();
        self.dtres += self.dtga - dtg;
        if self.dtres.abs() < 0.001 {
            self.dtres = 0.0;
        }
        self.left -= 1;
        dtg
    }
}

// ---------------------------------------------------------------------------
// CPU mirror of the kernels of crate::prop_uq
// ---------------------------------------------------------------------------

/// The share of a source point's action that one corner of its averaging
/// (`dxc`, `dyc`: the corner in index space, `cgs` its group velocity) sends
/// to the neighbour `(tx, ty)` (0, 1, 2 for -1, 0, +1), or with `selfmode`
/// what stays at the source (the shares that a near-coast point cannot
/// send, minus `RD/3` of the corner). `wet` is the wet mask of the source's
/// neighbourhood; a central point sends everything.
#[allow(clippy::too_many_arguments)]
pub fn corner_w(dxc: f64, dyc: f64, cgs: f64, tx: u32, ty: u32, wet: u32, central: bool, selfmode: bool) -> f64 {
    let ox = if dxc < 0.0 { 0u32 } else { 2 };
    let oy = if dyc < 0.0 { 0u32 } else { 2 };
    let xmajor = dxc.abs() > dyc.abs();
    let ra = if xmajor {
        (dyc / dxc).abs()
    } else if dyc.abs() > 1e-5 {
        (dxc / dyc).abs()
    } else {
        1.0
    };
    let rb = if xmajor { dxc.abs() } else { dyc.abs() };
    let rdb = (rb * cgs).min(1.0);
    let wd = rdb * ra / 6.0;
    let wa = (1.0 - ra) * rdb / 6.0;
    let (axx, axy) = if xmajor { (ox, 1) } else { (1, oy) };
    let wb = |px: u32, py: u32| (wet >> (3 * py + px)) & 1 == 1;
    let mut w = 0.0;
    for e in 0..4 {
        let (px, py, we, diag) = match e {
            0 => (ox, oy, wd, true),
            1 => (2 - ox, 2 - oy, wd, true),
            2 => (axx, axy, wa, false),
            _ => (2 - axx, 2 - axy, wa, false),
        };
        let test = if diag { wb(px, py) && (wb(px, 1) || wb(1, py)) } else { wb(px, py) };
        let pass = central || test;
        if selfmode {
            if !pass {
                w += we;
            }
        } else if pass && px == tx && py == ty {
            w += we;
        }
    }
    if selfmode {
        w -= rdb / 3.0;
    }
    w
}

/// The share of the action of sea point `s` (metric and group velocity of
/// the tables) sent to `(tx, ty)`, or kept with `selfmode` (then plus 1).
#[allow(clippy::too_many_arguments)]
fn source_w(t: &UqTables, pt: &[f64], s: usize, cgs: f64, tx: u32, ty: u32, fl: u32, selfmode: bool) -> f64 {
    let [cl, dpdx, dpdy, dqdx, dqdy, _] = [0, 1, 2, 3, 4, 5].map(|j| t.metric[j][s]);
    let (tmpx, tmpy) = (pt[2] / cl, pt[3]);
    let dxn = dpdx * tmpx + dpdy * tmpy;
    let dyn_ = dqdx * tmpx + dqdy * tmpy;
    let (tmpx, tmpy) = (pt[4] / cl, pt[5]);
    let dxs = dpdx * tmpx + dpdy * tmpy;
    let dys = dqdx * tmpx + dqdy * tmpy;
    let (wet, cen) = (fl & GF_WET, fl & GF_CEN != 0);
    let w = corner_w(dxn + dxs, dyn_ + dys, cgs, tx, ty, wet, cen, selfmode) + corner_w(dxn - dxs, dyn_ - dys, cgs, tx, ty, wet, cen, selfmode);
    if selfmode {
        1.0 + w
    } else {
        w
    }
}

/// Neighbour `(dxi - 1, dyi - 1)` of point `k`: its index, and whether the
/// move crosses the closure in longitude; `None` outside the grid.
pub fn neighbour(t: &UqTables, k: usize, dxi: usize, dyi: usize) -> Option<(usize, bool)> {
    let (ix, iy) = (k / t.ny, k % t.ny);
    if (iy == 0 && dyi == 0) || (iy + 1 == t.ny && dyi == 2) {
        return None;
    }
    let edge = (ix == 0 && dxi == 0) || (ix + 1 == t.nx && dxi == 2);
    if edge && !t.closed {
        return None;
    }
    let jx = (ix + t.nx + dxi - 1) % t.nx;
    Some((jx * t.ny + iy + dyi - 1, edge))
}

/// The averaging of sub-step `sub` of one component (parameters `pt`) at
/// point `k`, from the values `q` (Q at the sea points), times `GSQRT`:
/// the kernel [`crate::prop_uq::uq_average()`].
pub fn average_at(t: &UqTables, pt: &[f64], cg: &[f64], q: &[f64], k: usize) -> f64 {
    let fk = t.gfl[k];
    if fk & GF_SEA == 0 {
        return 0.0;
    }
    if fk & GF_POS == 0 {
        return 0.0;
    }
    let sk = t.gsea[k] as usize;
    let ak = q[k];
    let trn = fk & GF_TRN != 0;
    let mut sum = if trn { ak } else { ak * source_w(t, pt, sk, cg[sk], 1, 1, fk, true) };
    let mut seam = 0.0;
    for dyi in 0..3 {
        for dxi in 0..3 {
            if dxi == 1 && dyi == 1 {
                continue;
            }
            let Some((s, cross)) = neighbour(t, k, dxi, dyi) else { continue };
            let fs = t.gfl[s];
            if fs & GF_POS == 0 || fs & GF_TRN != 0 || fs & GF_SEA == 0 {
                continue;
            }
            if !cross && trn && t.gpos[s] <= t.gpos[k] {
                continue;
            }
            let ss = t.gsea[s] as usize;
            let w = q[s] * source_w(t, pt, ss, cg[ss], (2 - dxi) as u32, (2 - dyi) as u32, fs, false);
            if cross {
                seam += w;
            } else {
                sum += w;
            }
        }
    }
    if fk & GF_MS2 != 0 {
        sum = ak;
    }
    (sum + seam) * t.metric[5][sk]
}

/// The Courant number of sea point `s` along `x` (`ydir = false`) or `y`.
fn cfl_at(t: &UqTables, pt: &[f64], s: usize, cgs: f64, cur: Option<(&[f64], &[f64])>, ydir: bool) -> f64 {
    let cl = t.metric[0][s];
    let mut cxtot = pt[6] * cgs / cl;
    let mut cytot = pt[7] * cgs;
    if let Some((cx, cy)) = cur {
        cxtot += cx[s] / cl;
        cytot += cy[s];
    }
    let (m1, m2) = if ydir { (t.metric[3][s], t.metric[4][s]) } else { (t.metric[1][s], t.metric[2][s]) };
    (cxtot * m1 + cytot * m2) * pt[1]
}

/// Next point along the sweep (`x` with the wrap in longitude, `y`).
fn step(t: &UqTables, k: usize, ydir: bool, fwd: bool) -> usize {
    if ydir {
        if fwd {
            k + 1
        } else {
            k - 1
        }
    } else {
        let (ix, iy) = (k / t.ny, k % t.ny);
        let jx = if fwd { (ix + 1) % t.nx } else { (ix + t.nx - 1) % t.nx };
        jx * t.ny + iy
    }
}

/// The ULTIMATE QUICKEST face value of `W3QCK3` (as the kernel's
/// `theta_flux`, the flux `CFL QB`).
fn uq_flux(ct: f64, ctp: f64, qtm: f64, qt: f64, qtp: f64, qtpp: f64) -> f64 {
    let cf = 0.5 * (ct + ctp);
    let up = cf < 0.0;
    let (qc, qcm, qcp) = if up { (qtp, qt, qtpp) } else { (qt, qtm, qtp) };
    let qb = 0.5 * ((1.0 - cf) * qtp + (1.0 + cf) * qt) - (1.0 - cf * cf) * (1.0 / 6.0) * (qcm - 2.0 * qc + qcp);
    let (qu, qd) = if up { (qcp, qcm) } else { (qcm, qcp) };
    let dq = qd - qu;
    let m = dq.abs().max(1e-15);
    let dqnz = if dq >= 0.0 { m } else { -m };
    let qcn = ((qc - qu) / dqnz).max(-0.1).min(1.1);
    let qbn = ((qb - qu) / dqnz).max(qcn).min(1.0).min(qcn / cf.abs().max(1e-10));
    let qbr = qu + qbn * dq;
    let cfac = if 2.0 * (qcn - 0.5).abs() >= 1.0 { 1.0 } else { 0.0 };
    cf * ((1.0 - cfac) * qbr + cfac * qc)
}

/// The flux through the face between `a` and the next point `b` along the
/// sweep (zero without a point of `MAPSTA = 1` on either side).
#[allow(clippy::too_many_arguments)]
fn face(t: &UqTables, pt: &[f64], cg: &[f64], cur: Option<(&[f64], &[f64])>, q: &[f64], a: usize, ydir: bool) -> f64 {
    let b = step(t, a, ydir, true);
    let (ta, tb) = (t.gfl[a] & GF_MS1 != 0, t.gfl[b] & GF_MS1 != 0);
    let cfl = |p: usize| {
        let s = t.gsea[p] as usize;
        cfl_at(t, pt, s, cg[s], cur, ydir)
    };
    if ta && tb {
        uq_flux(cfl(a), cfl(b), q[step(t, a, ydir, false)], q[a], q[b], q[step(t, b, ydir, true)])
    } else if ta || tb {
        let c = if ta { cfl(a) } else { cfl(b) };
        c * if c < 0.0 { q[b] } else { q[a] }
    } else {
        0.0
    }
}

/// One sweep of `W3QCK3` at point `k` from the values `q` before the sweep
/// (the kernel [`crate::prop_uq::uq_sweep()`], without the division by
/// `GSQRT`).
pub fn sweep_at(t: &UqTables, pt: &[f64], cg: &[f64], cur: Option<(&[f64], &[f64])>, q: &[f64], k: usize, ydir: bool) -> f64 {
    let v = q[k];
    if t.gfl[k] & GF_ACT == 0 {
        return v;
    }
    let s = t.gsea[k] as usize;
    let fl = face(t, pt, cg, cur, q, step(t, k, ydir, false), ydir);
    let fr = face(t, pt, cg, cur, q, k, ydir);
    let o = if ydir { 2 } else { 0 };
    let tl = if fl > 0.0 { t.atrn[4 * s + o] } else { 1.0 };
    let tr = if fr < 0.0 { t.atrn[4 * s + o + 1] } else { 1.0 };
    (v + tl * fl - tr * fr).max(0.0)
}

/// [`crate::pro3::xyp3`] (no boundary spectra) as the kernels of
/// [`crate::prop_uq`] compute it, on the CPU: component `isp` with the
/// parameters `pt` of [`ptab`], `field` at the sea points.
#[allow(clippy::too_many_arguments)]
pub fn xyp3_mirror(t: &UqTables, pt: &[f64], field: &mut [f64], cg: &[f64], cur: Option<(&[f64], &[f64])>, secs_of_day: f64, dtg: f64, flcx: bool, flcy: bool) {
    let mut q = vec![0.0; t.ng];
    for s in 0..t.nsea {
        q[t.ksea[s] as usize] = field[s] / cg[s] * t.metric[0][s];
    }
    let mut yf = yfirst(secs_of_day, dtg);
    for _ in 0..pt[0] as usize {
        let mut v: Vec<f64> = (0..t.ng).map(|k| average_at(t, pt, cg, &q, k)).collect();
        for (ydir, on) in if yf { [(true, flcy), (false, flcx)] } else { [(false, flcx), (true, flcy)] } {
            if on {
                v = (0..t.ng).map(|k| sweep_at(t, pt, cg, cur, &v, k, ydir)).collect();
            }
        }
        for s in 0..t.nsea {
            let k = t.ksea[s] as usize;
            v[k] /= t.metric[5][s];
        }
        q = v;
        yf = !yf;
    }
    for s in 0..t.nsea {
        let k = t.ksea[s] as usize;
        field[s] = if t.gfl[k] & GF_POS != 0 { (cg[s] / t.metric[0][s] * q[k]).max(0.0) } else { q[k] };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pro3::xyp3;

    const SPEC: Spectrum = Spectrum { nk: 4, nth: 8, freq1: 0.05, xfr: 1.1, thoff: 0.0, dmin: 1.0 };

    /// A RECT grid with land where `land`, its edges excluded.
    fn rect(nx: usize, ny: usize, closed: bool, land: impl Fn(usize, usize) -> bool) -> StructGrid {
        let (mut lon, mut lat, mut zb, mut mapsta) = (vec![], vec![], vec![], vec![]);
        for iy in 0..ny {
            for ix in 0..nx {
                lon.push(10.0 + 0.25 * ix as f64);
                lat.push(-2.0 + 0.25 * iy as f64);
                let l = land(ix, iy);
                zb.push(if l { 10.0 } else { -100.0 });
                mapsta.push(if l { 0 } else { 1 });
            }
        }
        let mut g = StructGrid { name: "T".into(), gtype: GridType::Rect, nx, ny, closed, lon, lat, zb, mapsta, spec: SPEC };
        g.exclude_edges();
        g
    }

    /// A smooth field with some structure, at the sea points.
    fn field(pg: &Pro3Grid, seed: f64) -> Vec<f64> {
        pg.sea_ij
            .iter()
            .map(|&i| {
                let (ix, iy) = ((i % pg.nx) as f64, (i / pg.nx) as f64);
                1.0 + (0.37 * ix + seed).sin() * (0.23 * iy - seed).cos() + if (i * 7919) % 13 == 0 { 2.0 } else { 0.0 }
            })
            .collect()
    }

    /// The CPU mirror of the kernels gives `xyp3` on grids with land,
    /// closure and obstructions (and the ice transparencies), for every
    /// direction of two frequencies, both sweep orders.
    #[test]
    fn mirror_matches_xyp3() {
        let (nx, ny) = (40, 16);
        let land = |ix: usize, iy: usize| (ix == 17 && iy < 9) || (ix >= 25 && ix <= 27 && iy >= 5 && iy <= 7) || (ix == 3 && iy == 12) || (ix == 39 && iy == 4);
        for closed in [true, false] {
            let g = rect(nx, ny, closed, land);
            let trnx: Vec<f64> = (0..nx * ny).map(|i| if i % nx == 8 && (i / nx) % 3 == 0 { 0.3 } else { 1.0 }).collect();
            let trny: Vec<f64> = (0..nx * ny).map(|i| if i % nx == 30 { 0.6 } else { 1.0 }).collect();
            let tr = Transparencies { trnx, trny };
            let params = Pro3Params { dtcfl: 300.0, flagtr: 4, fice0: 0.25, ficen: 1.0, ficel: 40000.0, ..Pro3Params::default() };
            let mut pg = Pro3Grid::new(&g, Some(&tr), &params).unwrap();
            let ice: Vec<f64> = pg.sea_ij.iter().map(|&i| if i % nx == 12 { 0.4 } else { 0.0 }).collect();
            pg.update_maps(None, Some(&ice), None).unwrap();
            let t = UqTables::new(&pg).unwrap();
            assert!(t.gfl.iter().any(|&f| f & GF_TRN != 0));
            let cg: Vec<f64> = (0..pg.nsea).map(|s| 9.0 + (s % 7) as f64 * 0.3).collect();
            let cx: Vec<f64> = (0..pg.nsea).map(|s| 0.3 * ((s % 5) as f64 - 2.0)).collect();
            let cy: Vec<f64> = (0..pg.nsea).map(|s| 0.2 * ((s % 3) as f64 - 1.0)).collect();
            for cur in [None, Some((cx.as_slice(), cy.as_slice()))] {
                let dtg = 900.0;
                let (tab, _) = ptab(&pg, dtg, cur);
                for isp in (0..8).chain(24..32) {
                    for secs in [900.0, 1800.0] {
                        let mut f0 = vec![0.0; nx * ny];
                        let fs = field(&pg, isp as f64);
                        for (s, &i) in pg.sea_ij.iter().enumerate() {
                            f0[i] = fs[s];
                        }
                        xyp3(&pg, isp, dtg, secs, &mut f0, &cg, cur, None);
                        let mut f1 = fs.clone();
                        xyp3_mirror(&t, &tab[PT * isp..PT * (isp + 1)], &mut f1, &cg, cur, secs, dtg, true, true);
                        let fmax = f1.iter().cloned().fold(0.0, f64::max);
                        let d = pg.sea_ij.iter().enumerate().map(|(s, &i)| (f0[i] - f1[s]).abs()).fold(0.0, f64::max);
                        assert!(d < 1e-12 * fmax, "closed {closed} cur {} isp {isp} secs {secs}: {d:e} of {fmax}", cur.is_some());
                    }
                }
            }
        }
    }

    /// The global steps of W3WAVE: 1,800 s for DTMAX 2,520 and an hour, and
    /// the remainder carried over a call of 1,000 s in 3 steps.
    #[test]
    fn w3wave_steps() {
        let mut s = W3Steps::new(2520.0);
        assert_eq!((s.next(0, 3600), s.next(1800, 3600)), (1800.0, 1800.0));
        let mut s = W3Steps::new(400.0);
        let v: Vec<f32> = (0..3).map(|_| s.next(0, 1000)).collect();
        assert_eq!(v.iter().sum::<f32>(), 1000.0);
        assert_eq!(v, vec![333.0, 334.0, 333.0]);
    }

    /// W3DZXY of a linear field on a RECT grid away from land: the exact
    /// gradient (per metre along the grid's degrees, no cosine as in WW3).
    #[test]
    fn dzxy_linear() {
        let g = rect(20, 10, true, |ix, iy| ix == 10 && iy == 5);
        let pg = Pro3Grid::new(&g, None, &Pro3Params::default()).unwrap();
        let t = UqTables::new(&pg).unwrap();
        let z: Vec<f64> = pg.sea_ij.iter().map(|&i| 3.0 * (i % 20) as f64 + 5.0 * (i / 20) as f64).collect();
        let (dx, dy) = t.dzxy(&z);
        let f = 1.0 / (DERA * RADIUS);
        for (s, &i) in pg.sea_ij.iter().enumerate() {
            let (ix, iy) = (i % 20, i / 20);
            if iy >= 2 && iy <= 7 && ix >= 1 && ix <= 18 && !(ix >= 9 && ix <= 11 && iy >= 4 && iy <= 6) {
                assert!((dx[s] - 3.0 / 0.25 * f).abs() < 1e-12 && (dy[s] - 5.0 / 0.25 * f).abs() < 1e-12, "({ix}, {iy}) {} {}", dx[s], dy[s]);
            }
        }
    }
}
