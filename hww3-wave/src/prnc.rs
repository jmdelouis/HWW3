//! The interpolation of `ww3_prnc` (the WW3 preprocessor of forcing
//! fields) from a regular longitude-latitude grid to the nodes of an
//! unstructured grid, in single precision as WW3 computes it, with the
//! switches of HWW3 (`WNT2` for the wind, `CRT1` for the currents).
//!
//! `ww3_prnc` with `FORCING%GRID%LATLON = T` (`ITYPE = 2`) on an
//! unstructured grid takes, for each node, the source cell that holds it
//! (section 5 of `ww3_prnc.F90`, the branch `GTYPE = UNGTYPE`), bilinear
//! weights scaled by a factor `EFAC` that only differs from 1 outside the
//! grid, and interpolates each field with `INTERP`, which drops the corners
//! at `_FillValue` and divides by the sum of the remaining weights. Vector
//! fields (wind, currents) are interpolated by component and their
//! modulus is then rescaled to the interpolated root-mean-square speed
//! (`WNT2`) or mean speed (`CRT1`), at most by 1.5.
//!
//! The values come from the netCDF file as stored (`NF90_GET_VAR` into
//! `REAL`, no unpacking), are set to `_FillValue` where NaN and otherwise
//! unpacked as `x * scale_factor + add_offset` in single precision;
//! `scripts/prnc_nc.py` writes them in that form for [`RawInput`].

use crate::ww3io::Time;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// `ACC` of `ww3_prnc`: weights within this of the grid edge are clamped.
const ACC: f32 = 0.05;

/// The regular source grid of a netCDF forcing file: `NXI` longitudes from
/// `X0I` to `XNI` and `NYI` latitudes from `Y0I` to `YNI` (either order),
/// in single precision as `ww3_prnc` reads them.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceGrid {
    /// Number of longitudes `NXI`.
    pub nxi: usize,
    /// Number of latitudes `NYI`.
    pub nyi: usize,
    /// First longitude `X0I` (degrees).
    pub x0i: f32,
    /// Last longitude `XNI`.
    pub xni: f32,
    /// First latitude `Y0I`.
    pub y0i: f32,
    /// Last latitude `YNI`.
    pub yni: f32,
}

/// Interpolation weights of `ww3_prnc` for the nodes of an unstructured
/// grid: for node `i`, the corners `(ix21, iy21)`, `(ix22, iy21)`,
/// `(ix21, iy22)`, `(ix22, iy22)` (0-based) with weights `rd11`, `rd21`,
/// `rd12`, `rd22`.
#[derive(Clone, Debug)]
pub struct Weights {
    /// First longitude index of each node.
    pub ix21: Vec<u32>,
    /// Second longitude index.
    pub ix22: Vec<u32>,
    /// First latitude index.
    pub iy21: Vec<u32>,
    /// Second latitude index.
    pub iy22: Vec<u32>,
    /// Weight of `(ix21, iy21)`.
    pub rd11: Vec<f32>,
    /// Weight of `(ix22, iy21)`.
    pub rd21: Vec<f32>,
    /// Weight of `(ix21, iy22)`.
    pub rd12: Vec<f32>,
    /// Weight of `(ix22, iy22)`.
    pub rd22: Vec<f32>,
    /// Nodes outside the source grid or on its last row or column (those
    /// for which `ww3_prnc` writes message 1041).
    pub outside: usize,
    /// Source grid closed in longitude (`ICLOSE_SMPL`).
    pub closed: bool,
}

impl SourceGrid {
    /// The weights of `ww3_prnc` for nodes at longitudes `x` and latitudes
    /// `y` (degrees; `XGRD`, `YGRD` taken to single precision).
    pub fn weights(&self, x: &[f64], y: &[f64]) -> Weights {
        let (nxi, nyi) = (self.nxi as i32, self.nyi as i32);
        let sxi = (self.xni - self.x0i) / (nxi - 1) as f32;
        let syi = (self.yni - self.y0i) / (nyi - 1) as f32;
        let closed = ((nxi as f32 * sxi).abs() - 360.0).abs() < 0.1 * sxi.abs();
        let n = x.len();
        let mut w = Weights {
            ix21: vec![0; n],
            ix22: vec![0; n],
            iy21: vec![0; n],
            iy22: vec![0; n],
            rd11: vec![0.0; n],
            rd21: vec![0.0; n],
            rd12: vec![0.0; n],
            rd22: vec![0.0; n],
            outside: 0,
            closed,
        };
        for i in 0..n {
            let (xx, yy) = (x[i] as f32, y[i] as f32);
            let mut out = false;
            let xm = (360.0f32 + (xx - self.x0i)) % 360.0;
            let mut ix21 = 1 + (xm / sxi) as i32;
            let ix22;
            if !closed {
                out |= ix21 < 1 || ix21 > nxi - 1;
                ix21 = ix21.clamp(1, nxi - 1);
                ix22 = ix21 + 1;
            } else {
                ix21 = ix21.clamp(1, nxi);
                ix22 = ix21 % nxi + 1;
            }
            let mut iy21 = 1 + ((yy - self.y0i) / syi) as i32;
            out |= iy21 < 1 || iy21 > nyi - 1;
            w.outside += out as usize;
            iy21 = iy21.clamp(1, nyi - 1);
            let iy22 = iy21 + 1;
            let mut rw1 = xm / sxi - (ix21 - 1) as f32;
            let mut rw2 = (yy - self.y0i) / syi - (iy21 - 1) as f32;
            if ix21 <= 1 && rw1 < ACC && rw1 < 0.0 {
                rw1 = 0.0;
            }
            if ix21 >= nxi - 1 && rw1 > 1.0 - ACC && rw1 > 1.0 {
                rw1 = 1.0;
            }
            if iy21 <= 1 && rw2 < ACC && rw2 < 0.0 {
                rw2 = 0.0;
            }
            // as ww3_prnc: IY21 >= NYI never holds after the clamp above
            if iy21 >= nyi && rw2 > 1.0 - ACC && rw2 > 1.0 {
                rw2 = 1.0;
            }
            let e1 = ((rw1 - 0.5).abs() - 0.5).max(0.0);
            let e2 = ((rw2 - 0.5).abs() - 0.5).max(0.0);
            let efac = (e1 * e1 + e2 * e2).sqrt();
            let efac = 1.0 / (1.0 + 0.25 * (efac * efac));
            w.ix21[i] = (ix21 - 1) as u32;
            w.ix22[i] = (ix22 - 1) as u32;
            w.iy21[i] = (iy21 - 1) as u32;
            w.iy22[i] = (iy22 - 1) as u32;
            w.rd11[i] = efac * (1.0 - rw1) * (1.0 - rw2);
            w.rd21[i] = efac * rw1 * (1.0 - rw2);
            w.rd12[i] = efac * (1.0 - rw1) * rw2;
            w.rd22[i] = efac * rw1 * rw2;
        }
        w
    }
}

impl Weights {
    /// `INTERP` of `ww3_prnc`: the field `xc` of the source grid
    /// (`xc[iy nxi + ix]`) at the nodes. Corners at `fill` are left out and
    /// the others renormalised; a node whose corners have zero weight takes
    /// the first corner not at `fill`, or 0.
    pub fn interp(&self, xc: &[f32], nxi: usize, fill: f32) -> Vec<f32> {
        let n = self.rd11.len();
        let mut fa = vec![0f32; n];
        for i in 0..n {
            let at = |ix: u32, iy: u32| xc[iy as usize * nxi + ix as usize];
            let c = [
                (at(self.ix21[i], self.iy21[i]), self.rd11[i]),
                (at(self.ix22[i], self.iy21[i]), self.rd21[i]),
                (at(self.ix21[i], self.iy22[i]), self.rd12[i]),
                (at(self.ix22[i], self.iy22[i]), self.rd22[i]),
            ];
            let mut factor = 0f32;
            let mut f = 0f32;
            for (v, r) in c {
                if v != fill {
                    factor += r;
                    f += r * v;
                }
            }
            fa[i] = if factor > 0.0 {
                f / factor
            } else {
                c.iter().map(|&(v, _)| v).find(|&v| v != fill).unwrap_or(0.0)
            };
        }
        fa
    }
}

/// The forcing fields of `ww3_prnc` handled here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    /// Ice concentration (`ICE`, `IFLD = 1`).
    Ice,
    /// Water level (`LEV`, `IFLD = 2`).
    Level,
    /// Wind (`WND`, `IFLD = 3`), rescaled with `WNT2`.
    Wind,
    /// Currents (`CUR`, `IFLD = 4`), rescaled with `CRT1`.
    Current,
}

impl Field {
    /// From the name used on the command line.
    pub fn parse(s: &str) -> Result<Field> {
        Ok(match s {
            "ice" => Field::Ice,
            "level" => Field::Level,
            "wind" => Field::Wind,
            "current" => Field::Current,
            _ => bail!("field {s}: expected ice, level, wind or current"),
        })
    }

    /// `IDFLD` of the forcing file.
    pub fn id(self) -> &'static str {
        match self {
            Field::Ice => "ICE",
            Field::Level => "LEV",
            Field::Wind => "WND",
            Field::Current => "CUR",
        }
    }

    /// Number of netCDF variables (components).
    pub fn ncomp(self) -> usize {
        match self {
            Field::Ice | Field::Level => 1,
            Field::Wind | Field::Current => 2,
        }
    }

    /// The fields written for one time from the source fields `xc` (and
    /// `yc` for vectors), already unpacked: `[FA]` for a scalar, `[FX, FY]`
    /// for a vector.
    pub fn process(self, w: &Weights, nxi: usize, xc: &[f32], yc: Option<&[f32]>, fill: f32) -> Vec<Vec<f32>> {
        match self {
            Field::Ice | Field::Level => vec![w.interp(xc, nxi, fill)],
            Field::Wind | Field::Current => {
                let yc = yc.expect("vector field without its second component");
                let mut fx = w.interp(xc, nxi, fill);
                let mut fy = w.interp(yc, nxi, fill);
                let mut xtemp: Vec<f32> = xc.iter().zip(yc).map(|(&a, &b)| if a != fill && b != fill { a * a + b * b } else { fill }).collect();
                let a3 = w.interp(&xtemp, nxi, fill);
                for v in xtemp.iter_mut() {
                    if *v != fill {
                        *v = v.sqrt();
                    }
                }
                let a2 = w.interp(&xtemp, nxi, fill);
                for i in 0..fx.len() {
                    let a1 = (fx[i] * fx[i] + fy[i] * fy[i]).sqrt().max(1e-10);
                    // WNT2: root-mean-square speed; CRT1: mean speed
                    let target = if self == Field::Wind { a3[i].sqrt() } else { a2[i] };
                    let factor = (target / a1).min(1.5);
                    fx[i] *= factor;
                    fy[i] *= factor;
                }
                vec![fx, fy]
            }
        }
    }
}

/// Unpack values as `ww3_prnc` does after `NF90_GET_VAR`: NaN becomes
/// `fill`, and any other value `x * scale + offset` in single precision.
pub fn unpack(x: &mut [f32], fill: f32, scale: f32, offset: f32) {
    for v in x.iter_mut() {
        if v.is_nan() {
            *v = fill;
        }
        if *v != fill {
            *v = *v * scale + offset;
        }
    }
}

/// A netCDF forcing file in the form written by `scripts/prnc_nc.py`: a
/// directory with `meta.json` (grid size, `_FillValue`, and per variable
/// `scale_factor` and `add_offset`, all as single-precision bit patterns,
/// and the times), `lon.f32`, `lat.f32` (the axes) and one `<var>.f32` per
/// variable (little endian, `[time][lat][lon]`, the values as stored).
pub struct RawInput {
    dir: PathBuf,
    /// The source grid.
    pub grid: SourceGrid,
    /// `_FillValue` (of the last variable, as `ww3_prnc` keeps it).
    pub fill: f32,
    /// Variable names, scale factors and offsets.
    pub vars: Vec<(String, f32, f32)>,
    /// Times of the records.
    pub times: Vec<Time>,
}

fn f32_bits(v: &serde_json::Value, what: &str) -> Result<f32> {
    let b = v.as_u64().with_context(|| format!("meta.json: {what} is not an integer bit pattern"))?;
    Ok(f32::from_bits(b as u32))
}

fn read_f32s(p: &Path, offset: u64, n: usize) -> Result<Vec<f32>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(p).with_context(|| format!("open {}", p.display()))?;
    f.seek(SeekFrom::Start(offset))?;
    let mut b = vec![0u8; 4 * n];
    f.read_exact(&mut b).with_context(|| format!("read {} values at byte {offset} of {}", n, p.display()))?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

impl RawInput {
    /// Open a directory written by `scripts/prnc_nc.py`.
    pub fn open(dir: &Path) -> Result<RawInput> {
        let m: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).with_context(|| format!("read {}/meta.json", dir.display()))?)?;
        let nxi = m["nxi"].as_u64().context("meta.json: nxi")? as usize;
        let nyi = m["nyi"].as_u64().context("meta.json: nyi")? as usize;
        let lon = read_f32s(&dir.join("lon.f32"), 0, nxi)?;
        let lat = read_f32s(&dir.join("lat.f32"), 0, nyi)?;
        let grid = SourceGrid { nxi, nyi, x0i: lon[0], xni: lon[nxi - 1], y0i: lat[0], yni: lat[nyi - 1] };
        let fill = f32_bits(&m["fill"], "fill")?;
        let mut vars = vec![];
        for v in m["vars"].as_array().context("meta.json: vars")? {
            let name = v["name"].as_str().context("meta.json: var name")?.to_string();
            vars.push((name, f32_bits(&v["scale"], "scale")?, f32_bits(&v["offset"], "offset")?));
        }
        let mut times = vec![];
        for t in m["times"].as_array().context("meta.json: times")? {
            let d = t[0].as_i64().context("meta.json: time")? as i32;
            let s = t[1].as_i64().context("meta.json: time")? as i32;
            times.push(Time(d, s));
        }
        Ok(RawInput { dir: dir.to_path_buf(), grid, fill, vars, times })
    }

    /// The unpacked fields of variable `k` at record `it`.
    pub fn field(&self, k: usize, it: usize) -> Result<Vec<f32>> {
        let n = self.grid.nxi * self.grid.nyi;
        let (name, scale, offset) = &self.vars[k];
        let mut x = read_f32s(&self.dir.join(format!("{name}.f32")), (4 * n * it) as u64, n)?;
        unpack(&mut x, self.fill, *scale, *offset);
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_inside_and_at_the_edge() {
        // 0.5 degree grid from 10 W to 10 E and 40 N to 50 N
        let g = SourceGrid { nxi: 41, nyi: 21, x0i: -10.0, xni: 10.0, y0i: 40.0, yni: 50.0 };
        let w = g.weights(&[0.25, -10.0, 10.0, 0.0], &[45.25, 40.0, 50.0, 55.0]);
        assert!(!w.closed);
        // node 0: halfway between columns 20 and 21, rows 10 and 11
        assert_eq!((w.ix21[0], w.ix22[0], w.iy21[0], w.iy22[0]), (20, 21, 10, 11));
        assert!(w.rd11.iter().zip(&w.rd22).take(1).all(|(&a, &b)| (a - 0.25).abs() < 1e-6 && (b - 0.25).abs() < 1e-6));
        // the corners of the grid: all the weight on one corner
        assert_eq!((w.ix21[1], w.iy21[1], w.rd11[1]), (0, 0, 1.0));
        assert_eq!((w.ix22[2], w.iy22[2], w.rd22[2]), (40, 20, 1.0));
        // on the last row and column, and 5 degrees north of the grid
        // (extrapolated, scaled down by EFAC = 1/26)
        assert_eq!(w.outside, 2);
        let s = w.rd11[3] + w.rd21[3] + w.rd12[3] + w.rd22[3];
        assert!((s - 1.0 / 26.0).abs() < 1e-6 && w.rd11[3] < 0.0 && (w.rd12[3] - 11.0 / 26.0).abs() < 1e-6);
    }

    #[test]
    fn closed_grid_wraps() {
        // a global 1 degree grid from 0 to 359 E
        let g = SourceGrid { nxi: 360, nyi: 181, x0i: 0.0, xni: 359.0, y0i: 90.0, yni: -90.0 };
        let w = g.weights(&[359.5, -0.5], &[0.0, 0.0]);
        assert!(w.closed);
        assert_eq!((w.ix21[0], w.ix22[0]), (359, 0));
        assert_eq!((w.ix21[1], w.ix22[1]), (359, 0));
        // decreasing latitudes: row 90 is the equator
        assert_eq!((w.iy21[0], w.rd11[0]), (90, 0.5));
    }

    #[test]
    fn interp_drops_fill_values() {
        let g = SourceGrid { nxi: 2, nyi: 2, x0i: 0.0, xni: 1.0, y0i: 0.0, yni: 1.0 };
        let w = g.weights(&[0.25, 0.9], &[0.5, 0.9]);
        let fill = -32767.0;
        let xc = [1.0, 2.0, fill, 4.0];
        let fa = w.interp(&xc, 2, fill);
        // node 0: weights 0.375, 0.125 (row 0), 0.375 (fill), 0.125
        let want = (0.375 * 1.0 + 0.125 * 2.0 + 0.125 * 4.0) / 0.625;
        assert!((fa[0] - want).abs() < 1e-6);
        let all_fill = [fill; 4];
        assert_eq!(w.interp(&all_fill, 2, fill), vec![0.0, 0.0]);
    }

    #[test]
    fn vector_rescaling() {
        // opposite winds on two columns: the interpolated vector is short,
        // the rescaling (at most 1.5) brings it towards the mean speed
        let g = SourceGrid { nxi: 2, nyi: 2, x0i: 0.0, xni: 1.0, y0i: 0.0, yni: 1.0 };
        // (a longitude exact in single precision: MOD(360 + x, 360) rounds
        // others, as in ww3_prnc)
        let w = g.weights(&[0.25], &[0.5]);
        let u = [10.0, -10.0, 10.0, -10.0];
        let v = [0.0; 4];
        let f = Field::Wind.process(&w, 2, &u, Some(&v), -999.0);
        // interpolated u = 5, rms speed 10: factor capped at 1.5
        assert!((f[0][0] - 7.5).abs() < 1e-5 && f[1][0] == 0.0);
        let u = [10.0, 8.0, 10.0, 8.0];
        let f = Field::Current.process(&w, 2, &u, Some(&v), -999.0);
        // same direction: mean speed equals the interpolated speed
        assert!((f[0][0] - 9.5).abs() < 1e-5);
    }

    #[test]
    fn unpack_as_ww3() {
        let mut x = [f32::NAN, 100.0, -32767.0];
        unpack(&mut x, -32767.0, 0.5, 1.0);
        assert_eq!(x, [-32767.0, 51.0, -32767.0]);
    }
}
