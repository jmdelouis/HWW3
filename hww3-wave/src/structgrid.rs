//! WW3 structured grids (`RECT` and `CURV` of `ww3_grid.nml`) turned into
//! the triangle meshes of the explicit N scheme (`PDLIB_W3XYPFSN2`), so that
//! `hww3-prop` runs them as it runs the HWW3 unstructured grids.
//!
//! The reader follows `ww3_grid` (`ww3_grid.F90`, namelist version):
//! - `RECT`: `NX × NY` points `X0 + ix SX`, `Y0 + iy SY` (degrees), with
//!   `SX`, `SY` divided by `RECT%SF` and `X0`, `Y0` by `RECT%SF0`;
//! - `CURV`: the longitudes and latitudes of each point read from
//!   `CURV%XCOORD%FILENAME` and `CURV%YCOORD%FILENAME`, multiplied by their
//!   `SF` and shifted by their `OFF`;
//! - the bottom level `ZB` from `DEPTH%FILENAME` times `DEPTH%SF`
//!   (negative at sea), and the status map `MAPSTA` from `MASK%FILENAME`
//!   (0 land, 1 sea, 2 active boundary, 3 excluded) or, without a mask
//!   file, sea where `ZB <= ZLIM`;
//! - the sea points of the first and last rows, and of the first and last
//!   columns when the grid is not closed in longitude (`CLOS = 'NONE'`),
//!   are excluded, as `ww3_grid` does (it counts 4,304 excluded points on
//!   IRIGLOB-7M, the sea points of its two edge rows);
//! - the files are read in free format (`IDFM = 1`, the default) or as
//!   Fortran unformatted records (`IDFM = 3`), row by row from the south
//!   (`IDLA = 1, 2`, the default) or from the north (`IDLA = 3, 4`).
//!
//! The sea points (`MAPSTA` 1 or 2) keep WW3's order (`ISEA`, `ix` fastest
//! then `iy`). Each grid cell whose four corners are sea gives two triangles,
//! cut along its shorter diagonal; a cell with three sea corners gives one.
//! With the closure `CLOS = 'SMPL'` the last column is joined to the first.
//! Sea points left in no triangle (single points between land) are dropped,
//! and the points of `MAPSTA = 2` become the open boundary of the mesh.
//!
//! A grid around a pole (the curvilinear polar grids) cannot be meshed in
//! longitude and latitude. The mesh then uses rotated coordinates in which
//! the pole sits on the equator ([`Rotation`]), with the spectral directions
//! taken from the rotated north, as WW3 does on its rotated-pole grids. The
//! geographic coordinates and the angle from the geographic to the rotated
//! east at each node are written next to the mesh, so that forcing vectors
//! can be turned into the rotated frame.
//!
//! Left out: the obstructions (`OBST%FILENAME`, sub-grid blocking of the
//! structured schemes, which the triangle scheme has no use for) and the
//! tripolar closure (`TRPL`); Cartesian grids (`COORD = 'CART'`) are not
//! handled.

use crate::grid::{Grid, Spectrum};
use crate::ww3grid::Namelists;
use anyhow::{anyhow, bail, Context, Result};
use std::f64::consts::PI;
use std::path::Path;

const DERA: f64 = PI / 180.0;

/// Grid type of `GRID%TYPE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridType {
    /// Regular longitude-latitude grid.
    Rect,
    /// Curvilinear grid (coordinates given at each point).
    Curv,
}

/// A WW3 structured grid as `ww3_grid` reads it. Arrays are `ny × nx`,
/// point `(ix, iy)` at `iy nx + ix`, `iy = 0` the first row of the files
/// (the southern one with `IDLA = 1`).
#[derive(Clone, Debug)]
pub struct StructGrid {
    /// `GRID%NAME`.
    pub name: String,
    /// `GRID%TYPE`.
    pub gtype: GridType,
    /// Points along a row.
    pub nx: usize,
    /// Number of rows.
    pub ny: usize,
    /// `CLOS = 'SMPL'`: the last column is joined to the first.
    pub closed: bool,
    /// Longitude of the points (degrees).
    pub lon: Vec<f64>,
    /// Latitude of the points (degrees).
    pub lat: Vec<f64>,
    /// Bottom level `ZB` (m, negative at sea).
    pub zb: Vec<f64>,
    /// `MAPSTA`: 0 land, 1 sea, 2 active boundary, 3 excluded.
    pub mapsta: Vec<u8>,
    /// Spectral grid and `DMIN`.
    pub spec: Spectrum,
}

/// Rotation of the coordinates of a mesh: none, or the rotated frame in
/// which the geographic north (south) pole lies on the equator, at rotated
/// longitude 0 (180), with the rotated north pole at (0° N, 180° E)
/// ((0° N, 0° E)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rotation {
    /// Geographic coordinates.
    None,
    /// The north pole moved to the rotated point (0, 0).
    North,
    /// The south pole moved to the rotated point (0, 0).
    South,
}

impl Rotation {
    /// Parse `none`, `north` or `south`.
    pub fn parse(s: &str) -> Result<Rotation> {
        Ok(match s {
            "none" => Rotation::None,
            "north" => Rotation::North,
            "south" => Rotation::South,
            _ => bail!("rotation {s}: none, north, south or auto"),
        })
    }

    /// Unit vector of `(lon, lat)` in the rotated frame.
    fn apply(self, v: [f64; 3]) -> [f64; 3] {
        match self {
            Rotation::None => v,
            // x' = z, y' = y, z' = -x: (0, 0, 1) -> (1, 0, 0)
            Rotation::North => [v[2], v[1], -v[0]],
            // x' = -z, y' = y, z' = x: (0, 0, -1) -> (1, 0, 0)
            Rotation::South => [-v[2], v[1], v[0]],
        }
    }

    /// Rotated longitude and latitude (degrees) of a point, and the angle
    /// `alpha` (degrees) from the rotated east to the geographic east,
    /// counter-clockwise: a vector of geographic components `(ue, un)` has
    /// rotated components `(ue cos a - un sin a, ue sin a + un cos a)`.
    pub fn point(self, lon: f64, lat: f64) -> (f64, f64, f64) {
        if self == Rotation::None {
            return (lon, lat, 0.0);
        }
        let (l, p) = (lon * DERA, lat * DERA);
        let v = self.apply([p.cos() * l.cos(), p.cos() * l.sin(), p.sin()]);
        let lat_r = v[2].clamp(-1.0, 1.0).asin();
        let lon_r = v[1].atan2(v[0]);
        // geographic east, in the rotated frame
        let e = self.apply([-l.sin(), l.cos(), 0.0]);
        // rotated east and north at the point
        let er = [-lon_r.sin(), lon_r.cos(), 0.0];
        let nr = [-lat_r.sin() * lon_r.cos(), -lat_r.sin() * lon_r.sin(), lat_r.cos()];
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let alpha = dot(e, nr).atan2(dot(e, er));
        (lon_r / DERA, lat_r / DERA, alpha / DERA)
    }
}

/// A triangle mesh made from a structured grid: the grid of the model and
/// what links it to the structured grid.
#[derive(Clone, Debug)]
pub struct Mesh {
    /// Nodes (in the coordinates of [`Mesh::rotation`]), depths,
    /// triangles, open boundary and spectral grid.
    pub grid: Grid,
    /// Index `iy nx + ix` of each node in the structured grid.
    pub ij: Vec<u32>,
    /// Geographic longitude of the nodes (degrees).
    pub lon_geo: Vec<f64>,
    /// Geographic latitude of the nodes (degrees).
    pub lat_geo: Vec<f64>,
    /// Angle from the rotated to the geographic east at each node (degrees,
    /// see [`Rotation::point`]); zero without rotation.
    pub angle: Vec<f64>,
    /// Rotation of the coordinates.
    pub rotation: Rotation,
    /// Sea points of the grid (`MAPSTA` 1 or 2).
    pub nsea: usize,
    /// Sea points in no triangle, left out of the mesh.
    pub dropped: usize,
}

/// Values of a WW3 input file read in free format: numbers separated by
/// blanks or commas, with Fortran `D` exponents and `r*value` repeats.
fn parse_free(text: &str, n: usize, what: &str) -> Result<Vec<f64>> {
    let mut v = Vec::with_capacity(n);
    for tok in text.split(|c: char| c.is_whitespace() || c == ',').filter(|s| !s.is_empty()) {
        let (rep, val) = match tok.split_once('*') {
            Some((r, x)) => (r.parse::<usize>().with_context(|| format!("{what}: repeat {tok}"))?, x),
            None => (1, tok),
        };
        let x: f64 = val.replace(['d', 'D'], "e").parse().with_context(|| format!("{what}: value {tok}"))?;
        for _ in 0..rep {
            v.push(x);
        }
        if v.len() >= n {
            break;
        }
    }
    if v.len() < n {
        bail!("{what}: {} values, {n} expected", v.len());
    }
    v.truncate(n);
    Ok(v)
}

/// Values of a Fortran unformatted sequential file (`IDFM = 3`): the
/// records one after the other, 4-byte reals or integers, in the byte order
/// of the record markers.
fn parse_unformatted(b: &[u8], n: usize, integer: bool, what: &str) -> Result<Vec<f64>> {
    if b.len() < 8 {
        bail!("{what}: {} bytes", b.len());
    }
    let le = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let big = !(le > 0 && le + 8 <= b.len() && u32::from_le_bytes(b[4 + le..8 + le].try_into().unwrap()) as usize == le);
    let rd = |s: &[u8]| -> [u8; 4] {
        let mut a: [u8; 4] = s.try_into().unwrap();
        if big {
            a.reverse();
        }
        a
    };
    let mut v = Vec::with_capacity(n);
    let mut p = 0;
    while p + 4 <= b.len() && v.len() < n {
        let len = u32::from_le_bytes(rd(&b[p..p + 4])) as usize;
        let body = b.get(p + 4..p + 4 + len).ok_or_else(|| anyhow!("{what}: record of {len} bytes past the end"))?;
        for c in body.chunks_exact(4) {
            let w = rd(c);
            v.push(if integer { i32::from_le_bytes(w) as f64 } else { f32::from_le_bytes(w) as f64 });
        }
        p += len + 8;
    }
    if v.len() < n {
        bail!("{what}: {} values, {n} expected", v.len());
    }
    v.truncate(n);
    Ok(v)
}

/// An array of `nx × ny` values of a WW3 input file, as `iy nx + ix` with
/// `iy = 0` the southern row (rows read from the north with `IDLA` 3 or 4).
pub fn read_array(path: &Path, nx: usize, ny: usize, idla: i64, idfm: i64, integer: bool) -> Result<Vec<f64>> {
    let n = nx * ny;
    let what = path.display().to_string();
    let v = match idfm {
        1 | 2 => parse_free(&std::fs::read_to_string(path).with_context(|| format!("read {what}"))?, n, &what)?,
        3 => parse_unformatted(&std::fs::read(path).with_context(|| format!("read {what}"))?, n, integer, &what)?,
        _ => bail!("{what}: IDFM {idfm} not handled"),
    };
    Ok(match idla {
        1 | 2 => v,
        3 | 4 => {
            let mut w = vec![0.0; n];
            for iy in 0..ny {
                w[iy * nx..(iy + 1) * nx].copy_from_slice(&v[(ny - 1 - iy) * nx..(ny - iy) * nx]);
            }
            w
        }
        _ => bail!("{what}: IDLA {idla} not handled"),
    })
}

/// The spectral grid and `DMIN` of a `ww3_grid` namelist.
pub fn spectrum(nml: &Namelists) -> Result<Spectrum> {
    Ok(Spectrum {
        nk: nml.f("SPECTRUM_NML", "SPECTRUM%NK", 32.0)? as usize,
        nth: nml.f("SPECTRUM_NML", "SPECTRUM%NTH", 36.0)? as usize,
        freq1: nml.f("SPECTRUM_NML", "SPECTRUM%FREQ1", 0.0373)?,
        xfr: nml.f("SPECTRUM_NML", "SPECTRUM%XFR", 1.1)?,
        thoff: nml.f("SPECTRUM_NML", "SPECTRUM%THOFF", 0.0)?,
        dmin: nml.f("GRID_NML", "GRID%DMIN", 1.0)?,
    })
}

impl StructGrid {
    /// Read the grid of a `ww3_grid.nml` (`RECT` or `CURV`), its files taken
    /// from `data` (where `ww3_grid` would find them).
    pub fn read(nml_path: &Path, data: &Path) -> Result<StructGrid> {
        let nml = Namelists::read(nml_path)?;
        let s = |g: &str, k: &str, d: &str| nml.s(g, k).unwrap_or_else(|| d.to_string());
        let gtype = match s("GRID_NML", "GRID%TYPE", "").to_uppercase().as_str() {
            "RECT" => GridType::Rect,
            "CURV" => GridType::Curv,
            t => bail!("{}: GRID%TYPE '{t}' (RECT or CURV here; UNST grids are read by Grid::read)", nml_path.display()),
        };
        let coord = s("GRID_NML", "GRID%COORD", "SPHE").to_uppercase();
        if coord != "SPHE" {
            bail!("{}: GRID%COORD '{coord}': only spherical grids", nml_path.display());
        }
        let closed = match s("GRID_NML", "GRID%CLOS", "NONE").to_uppercase().as_str() {
            "NONE" => false,
            "SMPL" => true,
            c => bail!("{}: GRID%CLOS '{c}' not handled", nml_path.display()),
        };
        let zlim = nml.f("GRID_NML", "GRID%ZLIM", 0.0)?;
        let (nx, ny, lon, lat);
        match gtype {
            GridType::Rect => {
                nx = nml.f("RECT_NML", "RECT%NX", 0.0)? as usize;
                ny = nml.f("RECT_NML", "RECT%NY", 0.0)? as usize;
                let sf = nml.f("RECT_NML", "RECT%SF", 1.0)?;
                let sf0 = nml.f("RECT_NML", "RECT%SF0", 1.0)?;
                let (sx, sy) = (nml.f("RECT_NML", "RECT%SX", 1.0)? / sf, nml.f("RECT_NML", "RECT%SY", 1.0)? / sf);
                let (x0, y0) = (nml.f("RECT_NML", "RECT%X0", 0.0)? / sf0, nml.f("RECT_NML", "RECT%Y0", 0.0)? / sf0);
                let mut x = Vec::with_capacity(nx * ny);
                let mut y = Vec::with_capacity(nx * ny);
                for iy in 0..ny {
                    for ix in 0..nx {
                        x.push(x0 + sx * ix as f64);
                        y.push(y0 + sy * iy as f64);
                    }
                }
                lon = x;
                lat = y;
            }
            GridType::Curv => {
                nx = nml.f("CURV_NML", "CURV%NX", 0.0)? as usize;
                ny = nml.f("CURV_NML", "CURV%NY", 0.0)? as usize;
                let coord = |c: &str| -> Result<Vec<f64>> {
                    let k = |w: &str| format!("CURV%{c}COORD%{w}");
                    let f = s("CURV_NML", &k("FILENAME"), "");
                    if f.is_empty() {
                        bail!("{}: no CURV%{c}COORD%FILENAME", nml_path.display());
                    }
                    let sf = nml.f("CURV_NML", &k("SF"), 1.0)?;
                    let off = nml.f("CURV_NML", &k("OFF"), 0.0)?;
                    let idla = nml.f("CURV_NML", &k("IDLA"), 1.0)? as i64;
                    let idfm = nml.f("CURV_NML", &k("IDFM"), 1.0)? as i64;
                    Ok(read_array(&data.join(&f), nx, ny, idla, idfm, false)?.into_iter().map(|v| v * sf + off).collect())
                };
                lon = coord("X")?;
                lat = coord("Y")?;
            }
        }
        if nx < 2 || ny < 2 {
            bail!("{}: grid of {nx} × {ny} points", nml_path.display());
        }
        let dfile = s("DEPTH_NML", "DEPTH%FILENAME", "");
        if dfile.is_empty() {
            bail!("{}: no DEPTH%FILENAME", nml_path.display());
        }
        let dsf = nml.f("DEPTH_NML", "DEPTH%SF", 1.0)?;
        let didla = nml.f("DEPTH_NML", "DEPTH%IDLA", 1.0)? as i64;
        let didfm = nml.f("DEPTH_NML", "DEPTH%IDFM", 1.0)? as i64;
        let zb: Vec<f64> = read_array(&data.join(&dfile), nx, ny, didla, didfm, false)?.into_iter().map(|v| v * dsf).collect();
        let mfile = s("MASK_NML", "MASK%FILENAME", "");
        let mapsta: Vec<u8> = if mfile.is_empty() {
            zb.iter().map(|&z| (z <= zlim) as u8).collect()
        } else {
            let midla = nml.f("MASK_NML", "MASK%IDLA", 1.0)? as i64;
            let midfm = nml.f("MASK_NML", "MASK%IDFM", 1.0)? as i64;
            read_array(&data.join(&mfile), nx, ny, midla, midfm, true)?
                .into_iter()
                .map(|v| match v.round() as i64 {
                    m @ 0..=3 => Ok(m as u8),
                    m => Err(anyhow!("{mfile}: mask value {m}")),
                })
                .collect::<Result<_>>()?
        };
        let name = s("GRID_NML", "GRID%NAME", "");
        let mut g = StructGrid { name, gtype, nx, ny, closed, lon, lat, zb, mapsta, spec: spectrum(&nml)? };
        g.exclude_edges();
        Ok(g)
    }

    /// Exclude the sea points of the first and last rows, and of the first
    /// and last columns of a grid not closed in longitude (`ww3_grid`).
    pub fn exclude_edges(&mut self) {
        let (nx, ny) = (self.nx, self.ny);
        let mut edge = |i: usize| {
            if self.mapsta[i] == 1 || self.mapsta[i] == 2 {
                self.mapsta[i] = 3;
            }
        };
        for ix in 0..nx {
            edge(ix);
            edge((ny - 1) * nx + ix);
        }
        if !self.closed {
            for iy in 0..ny {
                edge(iy * nx);
                edge(iy * nx + nx - 1);
            }
        }
    }

    /// Whether point `i` is a sea point (`MAPSTA` 1 or 2).
    pub fn is_sea(&self, i: usize) -> bool {
        self.mapsta[i] == 1 || self.mapsta[i] == 2
    }

    /// Number of points of each `MAPSTA` value (0 to 3).
    pub fn counts(&self) -> [usize; 4] {
        let mut c = [0; 4];
        for &m in &self.mapsta {
            c[m as usize] += 1;
        }
        c
    }

    /// The rotation the mesh needs: towards the pole of the hemisphere when
    /// a sea point lies within `margin` degrees of it, else none.
    pub fn auto_rotation(&self, margin: f64) -> Rotation {
        let sea = (0..self.lon.len()).filter(|&i| self.is_sea(i));
        let (mut lo, mut hi) = (90.0f64, -90.0f64);
        for i in sea {
            lo = lo.min(self.lat[i]);
            hi = hi.max(self.lat[i]);
        }
        if hi > 90.0 - margin {
            Rotation::North
        } else if lo < -90.0 + margin {
            Rotation::South
        } else {
            Rotation::None
        }
    }

    /// Triangulate the sea points (see the module documentation).
    pub fn triangulate(&self, rotation: Rotation) -> Result<Mesh> {
        let (nx, ny) = (self.nx, self.ny);
        let npt = nx * ny;
        // rotated coordinates of every point (geographic without rotation)
        let mut x = vec![0.0; npt];
        let mut y = vec![0.0; npt];
        let mut ang = vec![0.0; npt];
        for i in 0..npt {
            if self.is_sea(i) {
                let (a, b, c) = rotation.point(self.lon[i], self.lat[i]);
                x[i] = a;
                y[i] = b;
                ang[i] = c;
            }
        }
        // unit vectors, for the diagonals
        let unit = |i: usize| {
            let (l, p) = (self.lon[i] * DERA, self.lat[i] * DERA);
            [p.cos() * l.cos(), p.cos() * l.sin(), p.sin()]
        };
        let chord = |a: usize, b: usize| {
            let (u, v) = (unit(a), unit(b));
            (u[0] - v[0]).powi(2) + (u[1] - v[1]).powi(2) + (u[2] - v[2]).powi(2)
        };
        let ncol = if self.closed { nx } else { nx - 1 };
        let mut tri: Vec<[usize; 3]> = Vec::with_capacity(2 * ncol * (ny - 1));
        for iy in 0..ny - 1 {
            for ix in 0..ncol {
                let ix1 = (ix + 1) % nx;
                // corners counter-clockwise in (ix, iy)
                let q = [iy * nx + ix, iy * nx + ix1, (iy + 1) * nx + ix1, (iy + 1) * nx + ix];
                let s: Vec<bool> = q.iter().map(|&i| self.is_sea(i)).collect();
                match s.iter().filter(|&&b| b).count() {
                    4 => {
                        if chord(q[0], q[2]) <= chord(q[1], q[3]) * (1.0 + 1e-9) {
                            tri.push([q[0], q[1], q[2]]);
                            tri.push([q[0], q[2], q[3]]);
                        } else {
                            tri.push([q[0], q[1], q[3]]);
                            tri.push([q[1], q[2], q[3]]);
                        }
                    }
                    3 => {
                        let k = s.iter().position(|&b| !b).unwrap();
                        tri.push([q[(k + 1) % 4], q[(k + 2) % 4], q[(k + 3) % 4]]);
                    }
                    _ => {}
                }
            }
        }
        // nodes: the sea points in a triangle, in WW3's order
        let mut used = vec![false; npt];
        for t in &tri {
            for &v in t {
                used[v] = true;
            }
        }
        let nsea = (0..npt).filter(|&i| self.is_sea(i)).count();
        let mut node = vec![u32::MAX; npt];
        let mut ij = Vec::with_capacity(nsea);
        for i in 0..npt {
            if used[i] {
                node[i] = ij.len() as u32;
                ij.push(i as u32);
            }
        }
        let dropped = nsea - ij.len();
        let pick = |a: &[f64]| ij.iter().map(|&i| a[i as usize]).collect::<Vec<f64>>();
        let grid = Grid {
            lon: pick(&x),
            lat: pick(&y),
            depth: ij.iter().map(|&i| -self.zb[i as usize]).collect(),
            tri: tri.iter().map(|t| [node[t[0]], node[t[1]], node[t[2]]]).collect(),
            obc: ij.iter().enumerate().filter(|&(_, &i)| self.mapsta[i as usize] == 2).map(|(k, _)| k as u32).collect(),
            spec: self.spec.clone(),
        };
        if grid.lat.iter().any(|&v| v.abs() > 85.0) {
            bail!("{}: nodes within 5° of the pole of the mesh coordinates: use a rotation", self.name);
        }
        Ok(Mesh { grid, lon_geo: pick(&self.lon), lat_geo: pick(&self.lat), angle: pick(&ang), ij, rotation, nsea, dropped })
    }
}

impl Mesh {
    /// Write the mesh as an HWW3 grid directory ([`Grid::read`]):
    /// `raw/{lon,lat,depth}.f64`, `raw/tri.u32`, `raw/obc.u32`, plus
    /// `raw/ij.u32` (structured index of each node), `raw/lon_geo.f64`,
    /// `raw/lat_geo.f64` and `raw/angle.f64` (see [`Mesh::angle`]), and the
    /// namelist text `nml` as `ww3_grid.nml`.
    pub fn write(&self, dir: &Path, nml: &str) -> Result<()> {
        let raw = dir.join("raw");
        std::fs::create_dir_all(&raw).with_context(|| format!("create {}", raw.display()))?;
        let f64s = |name: &str, v: &[f64]| -> Result<()> {
            std::fs::write(raw.join(name), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).with_context(|| name.to_string())
        };
        let u32s = |name: &str, v: &[u32]| -> Result<()> {
            std::fs::write(raw.join(name), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).with_context(|| name.to_string())
        };
        f64s("lon.f64", &self.grid.lon)?;
        f64s("lat.f64", &self.grid.lat)?;
        f64s("depth.f64", &self.grid.depth)?;
        u32s("tri.u32", &self.grid.tri.concat())?;
        u32s("obc.u32", &self.grid.obc)?;
        u32s("ij.u32", &self.ij)?;
        f64s("lon_geo.f64", &self.lon_geo)?;
        f64s("lat_geo.f64", &self.lat_geo)?;
        f64s("angle.f64", &self.angle)?;
        std::fs::write(dir.join("ww3_grid.nml"), nml)?;
        Ok(())
    }

    /// Read `raw/lon_geo.f64`, `raw/lat_geo.f64` and `raw/angle.f64` of a
    /// grid directory, or the mesh coordinates and zero angles when the
    /// grid has none (an HWW3 grid).
    pub fn geographic(dir: &Path, grid: &Grid) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let raw = dir.join("raw");
        let rd = |n: &str| -> Result<Vec<f64>> {
            let b = std::fs::read(raw.join(n)).with_context(|| n.to_string())?;
            Ok(b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
        };
        if raw.join("angle.f64").exists() {
            Ok((rd("lon_geo.f64")?, rd("lat_geo.f64")?, rd("angle.f64")?))
        } else {
            Ok((grid.lon.clone(), grid.lat.clone(), vec![0.0; grid.lon.len()]))
        }
    }
}

/// Turn a vector of geographic components `(ue, un)` into the rotated frame
/// of a node whose angle is `alpha` degrees ([`Rotation::point`]).
pub fn rotate_vector(ue: f64, un: f64, alpha: f64) -> (f64, f64) {
    let (s, c) = (alpha * DERA).sin_cos();
    (ue * c - un * s, ue * s + un * c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Geometry;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("hww3_struct_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    const SPEC: &str = "&SPECTRUM_NML\n  SPECTRUM%XFR = 1.10\n  SPECTRUM%FREQ1 = 0.0339\n  SPECTRUM%NK = 4\n  SPECTRUM%NTH = 12\n/\n";

    /// A global 8 × 6 RECT grid, 45° by 10°, closed in longitude, with one
    /// land point, written row by row from the north (IDLA 3) and one
    /// repeat count, and its mask from ZLIM.
    #[test]
    fn rect_global_closed() {
        let d = tmpdir("rect");
        let (nx, ny) = (8, 6);
        let mut z = vec![-1000.0f64; nx * ny];
        z[2 * nx + 3] = 5.0; // land
        z[3 * nx + 5] = -0.05; // above ZLIM: land
        // file from the north, in mm (SF 0.001)
        let mut txt = String::new();
        for iy in (0..ny).rev() {
            let row: Vec<String> = (0..nx).map(|ix| format!("{}", (z[iy * nx + ix] * 1000.0) as i64)).collect();
            txt.push_str(&row.join(" "));
            txt.push('\n');
        }
        let txt = txt.replacen("-1000000 -1000000", "2*-1000000", 1);
        std::fs::write(d.join("g.bot"), txt).unwrap();
        let nml = format!(
            "{SPEC}&GRID_NML\n GRID%NAME = 'T1'\n GRID%TYPE = 'RECT'\n GRID%COORD = 'SPHE'\n GRID%CLOS = 'SMPL'\n GRID%ZLIM = -0.10\n GRID%DMIN = 2.5\n/\n\
             &RECT_NML\n RECT%NX = {nx}\n RECT%NY = {ny}\n RECT%SX = 45.0\n RECT%SY = 10.0\n RECT%X0 = -180.0\n RECT%Y0 = -25.0\n/\n\
             &DEPTH_NML\n DEPTH%SF = 0.001\n DEPTH%FILENAME = 'g.bot'\n DEPTH%IDLA = 3\n/\n"
        );
        std::fs::write(d.join("ww3_grid.nml"), &nml).unwrap();
        let g = StructGrid::read(&d.join("ww3_grid.nml"), &d).unwrap();
        assert_eq!((g.nx, g.ny, g.closed, g.gtype), (8, 6, true, GridType::Rect));
        assert_eq!((g.lon[1], g.lat[nx]), (-135.0, -15.0));
        assert!((g.zb[0] + 1000.0).abs() < 1e-9 && (g.zb[2 * nx + 3] - 5.0).abs() < 1e-9);
        // two edge rows excluded, two land points
        assert_eq!(g.counts(), [2, 4 * nx - 2, 0, 2 * nx]);
        let m = g.triangulate(Rotation::None).unwrap();
        assert_eq!((m.nsea, m.dropped, m.grid.lon.len()), (4 * nx - 2, 0, 4 * nx - 2));
        // WW3 order: ix fastest
        assert_eq!(&m.ij[..3], &[nx as u32, nx as u32 + 1, nx as u32 + 2]);
        // 3 rows of 8 cells (closed), the 8 cells around the two land points
        // keep one triangle each
        let full = 3 * nx;
        assert_eq!(m.grid.tri.len(), 2 * full - 8);
        // the geometry sees the closure: every triangle small and positive,
        // their areas sum to the band without the land
        let geo = Geometry::new(&m.grid);
        assert!(geo.tria.iter().all(|&a| a > 0.0 && a < 0.5 * 45.0 * 10.0 + 1e-9));
        let total: f64 = geo.tria.iter().sum();
        assert!((total - (full as f64 * 450.0 - 8.0 * 225.0)).abs() < 1e-6, "{total}");
        // no boundary across the date line: the nodes of rows 1 and 4 are
        // coast (edge rows excluded), those of rows 2-3 away from land inside
        let ix_of = |k: usize| m.ij[k] as usize % nx;
        let iy_of = |k: usize| m.ij[k] as usize / nx;
        for k in 0..m.grid.lon.len() {
            let (ix, iy) = (ix_of(k), iy_of(k));
            // the four neighbours of a land point lie on the diagonals cut
            // around it; its diagonal neighbours stay inside
            let near_land = [(3usize, 2usize), (5, 3)]
                .iter()
                .any(|&(lx, ly)| (ix as i64 - lx as i64).abs() + (iy as i64 - ly as i64).abs() == 1);
            let coast = iy == 1 || iy == 4 || near_land;
            assert_eq!(geo.iobp[k] == 0, coast, "node ({ix}, {iy})");
        }
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A CURV grid around the north pole (polar stereographic), with a mask
    /// file (land, active boundary, excluded corners), not closed.
    #[test]
    fn curv_polar_rotated() {
        let d = tmpdir("curv");
        let n = 21;
        let (mut lon, mut lat, mut z, mut mask) = (vec![], vec![], vec![], vec![]);
        let r0 = 2.0 * (PI / 4.0 - 30.0 * DERA / 2.0).tan(); // 30° N at the circle
        for iy in 0..n {
            for ix in 0..n {
                let (u, v) = ((ix as f64 / (n - 1) as f64 * 2.0 - 1.0) * r0, (iy as f64 / (n - 1) as f64 * 2.0 - 1.0) * r0);
                let r = (u * u + v * v).sqrt();
                let colat = 2.0 * (r / 2.0).atan();
                lat.push(90.0 - colat / DERA);
                lon.push(v.atan2(u) / DERA);
                z.push(-3000.0);
                mask.push(if lat.last().unwrap() < &30.0 { 3 } else if lat.last().unwrap() < &33.0 { 2 } else { 1 });
            }
        }
        mask[10 * n + 12] = 0;
        let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.10}")).collect::<Vec<_>>().join(" ");
        std::fs::write(d.join("p.lon"), fmt(&lon)).unwrap();
        std::fs::write(d.join("p.lat"), fmt(&lat)).unwrap();
        std::fs::write(d.join("p.bot"), fmt(&z.iter().map(|v| v * 1000.0).collect::<Vec<_>>())).unwrap();
        std::fs::write(d.join("p.mask"), mask.iter().map(|m: &i32| m.to_string()).collect::<Vec<_>>().join(" ")).unwrap();
        let nml = format!(
            "{SPEC}&GRID_NML\n GRID%NAME = 'P'\n GRID%TYPE = 'CURV'\n GRID%COORD = 'SPHE'\n GRID%CLOS = 'NONE'\n GRID%ZLIM = -0.10\n GRID%DMIN = 2.5\n/\n\
             &CURV_NML\n CURV%NX = {n}\n CURV%NY = {n}\n CURV%XCOORD%SF = 1.00\n CURV%XCOORD%FILENAME = 'p.lon'\n CURV%YCOORD%SF = 1.00\n CURV%YCOORD%FILENAME = 'p.lat'\n/\n\
             &DEPTH_NML\n DEPTH%SF = 0.001\n DEPTH%FILENAME = 'p.bot'\n/\n&MASK_NML\n MASK%FILENAME = 'p.mask'\n/\n"
        );
        std::fs::write(d.join("ww3_grid.nml"), &nml).unwrap();
        let g = StructGrid::read(&d.join("ww3_grid.nml"), &d).unwrap();
        assert_eq!(g.gtype, GridType::Curv);
        assert_eq!(g.auto_rotation(10.0), Rotation::North);
        assert!(g.triangulate(Rotation::None).is_err());
        let m = g.triangulate(Rotation::North).unwrap();
        assert!(!m.grid.obc.is_empty());
        assert!(m.grid.obc.iter().all(|&k| g.mapsta[m.ij[k as usize] as usize] == 2));
        // the pole node sits at the rotated origin
        let pole = m.ij.iter().position(|&i| i as usize == 10 * n + 10).unwrap();
        assert!(m.grid.lon[pole].abs() < 1e-9 && m.grid.lat[pole].abs() < 1e-9);
        // rotated nodes stay within 60° of the origin, triangles all positive
        assert!(m.grid.lat.iter().chain(&m.grid.lon).all(|v| v.abs() < 61.0));
        let geo = Geometry::new(&m.grid);
        assert!(geo.tria.iter().all(|&a| a > 0.0));
        // the rotated frame keeps lengths: sum of the triangle areas on the
        // sphere close to the spherical cap of the sea points
        let mesh_area: f64 = geo
            .tria
            .iter()
            .zip(geo.ine.chunks(3))
            .map(|(&a, v)| a * (v.iter().map(|&k| m.grid.lat[k as usize]).sum::<f64>() / 3.0 * DERA).cos())
            .sum();
        let geo_area: f64 = {
            let mut s = 0.0;
            for t in &m.grid.tri {
                let p: Vec<[f64; 3]> = t
                    .iter()
                    .map(|&k| {
                        let (l, b) = (m.lon_geo[k as usize] * DERA, m.lat_geo[k as usize] * DERA);
                        [b.cos() * l.cos(), b.cos() * l.sin(), b.sin()]
                    })
                    .collect();
                let c = [
                    (p[1][1] - p[0][1]) * (p[2][2] - p[0][2]) - (p[1][2] - p[0][2]) * (p[2][1] - p[0][1]),
                    (p[1][2] - p[0][2]) * (p[2][0] - p[0][0]) - (p[1][0] - p[0][0]) * (p[2][2] - p[0][2]),
                    (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]),
                ];
                s += 0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            }
            s / (DERA * DERA)
        };
        assert!((mesh_area / geo_area - 1.0).abs() < 0.02, "{mesh_area} {geo_area}");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn rotation_angles() {
        // a northward vector at the north-polar rotation's origin meridian
        for &(lon, lat) in &[(0.0, 60.0), (90.0, 70.0), (-135.0, 45.0), (10.0, 89.0)] {
            for rot in [Rotation::North, Rotation::South] {
                let lat = if rot == Rotation::South { -lat } else { lat };
                let (lr, br, a) = rot.point(lon, lat);
                // move a little east along the geographic east: the rotated
                // displacement has the direction of the rotated (1, 0)
                let h = 1e-6;
                let (l2, b2, _) = rot.point(lon + h / (lat * DERA).cos(), lat);
                let (dx, dy) = ((l2 - lr) * (br * DERA).cos(), b2 - br);
                let (ex, ey) = rotate_vector(1.0, 0.0, a);
                let nrm = (dx * dx + dy * dy).sqrt();
                assert!((dx / nrm - ex).abs() < 1e-5 && (dy / nrm - ey).abs() < 1e-5, "{rot:?} {lon} {lat}: {dx} {dy} vs {ex} {ey}");
            }
        }
        assert_eq!(Rotation::None.point(12.0, 34.0), (12.0, 34.0, 0.0));
    }

    #[test]
    fn unformatted_big_endian_rows() {
        // two records (rows) of 3 big-endian integers, then the same reals little-endian
        let mut b = vec![];
        for row in [[1i32, 0, 2], [3, 1, 1]] {
            b.extend(12u32.to_be_bytes());
            for v in row {
                b.extend(v.to_be_bytes());
            }
            b.extend(12u32.to_be_bytes());
        }
        assert_eq!(parse_unformatted(&b, 6, true, "t").unwrap(), vec![1.0, 0.0, 2.0, 3.0, 1.0, 1.0]);
        let mut c = vec![];
        c.extend(24u32.to_le_bytes());
        for v in [1.5f32, -2.0, 3.0, 0.25, 7.0, 8.0] {
            c.extend(v.to_le_bytes());
        }
        c.extend(24u32.to_le_bytes());
        assert_eq!(parse_unformatted(&c, 6, false, "t").unwrap(), vec![1.5, -2.0, 3.0, 0.25, 7.0, 8.0]);
    }
}
