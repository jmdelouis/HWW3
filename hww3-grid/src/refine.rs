//! Refinement from the bathymetry: the mesh-size functions of OceanMesh2D.
//!
//! The cell size required at a point follows the `wl`, `slp`, `fl` and `g`
//! options of OceanMesh2D (Roberts et al., 2019, GMD 12, 1847; `edgefx.m`):
//!
//! * **wavelength (`wl`)**: `h_wl = T·sqrt(g·max(h, h_min)) / wl` with
//!   `T` the M2 period (12.42 h), i.e. `wl` cells per shallow-water M2
//!   wavelength; `wave_time_s` gives `T/wl` directly;
//! * **slope (`slp`)**: `h_slp = 2π·max(h, h_min) / (slp·|∇h|)`, `slp`
//!   cells per topographic length scale; optionally restricted to
//!   `h < slope_max_depth_m` (as the depth bands of OceanMesh2D);
//! * **slope filter (`fl`)**: the bathymetry is low-pass filtered before
//!   the gradient: `fl = 0` no filter (OceanMesh2D default), `fl > 0` a
//!   low pass of wavelength `fl` km, `fl < 0` a low pass of wavelength the
//!   barotropic Rossby radius `sqrt(g h)/|f|` divided by `|fl|`
//!   (`sqrt(sqrt(g h)/(2β))` within 5° of the equator, capped at 10,000 km);
//! * **grade (`g`)**: the size field is then limited so that it grows by at
//!   most `grade` per unit distance, `h(x_j) <= h(x_i) + grade·|x_i - x_j|`
//!   (OceanMesh2D default 0.2).
//!
//! The bathymetry is first averaged on blocks of `smooth_km` (by default half
//! the finest cell size, so every finest cell holds about four blocks), the
//! counterpart of the structured grid of spacing `h0` of OceanMesh2D. `h` is
//! the block mean depth, land samples counting as `min_depth_m`. The low pass
//! is a box mean of width one wavelength (summed-area table), and `∇h` the
//! centred difference of the filtered depth between neighbouring blocks. The
//! grade limit is applied by forward and backward sweeps over the blocks.
//!
//! A wet block then requires the coarsest HEALPix depth whose cell size
//! `R·sqrt(π/3)/nside` is at most the required size, within
//! `[nside_min, nside_max]`, and every cell containing the block centre at a
//! coarser depth is split. Because of the 2:1 balance of the tree, neighbouring
//! cells still differ by up to a factor of 2 in size.

use crate::bathy::Bathy;
use crate::config::BathyRefine;
use crate::geo::Earth;
use crate::tree::uniq;
use cdshealpix::nested;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Gravity (m/s², as in OceanMesh2D).
pub const G: f64 = 9.807;
/// M2 period (s).
pub const T_M2: f64 = 12.42 * 3600.0;
const OMEGA: f64 = 7.292115e-5;

/// Cell size of HEALPix depth `d` (m): square root of the cell area.
pub fn cell_size_m(earth: &Earth, d: u8) -> f64 {
    earth.radius * (std::f64::consts::PI / 3.0).sqrt() / (1u64 << d) as f64
}

/// `T/wl` (s): `wave_time_s` if set, else the M2 period over `wl`.
pub fn wave_time(p: &BathyRefine) -> f64 {
    if p.wave_time_s > 0.0 {
        p.wave_time_s
    } else {
        T_M2 / p.wl
    }
}

/// Required cell size (m) at depth `h` (m) and slope `grad` (m/m), before
/// the grade limit.
pub fn target_size_m(p: &BathyRefine, h: f64, grad: f64) -> f64 {
    let h = h.max(p.min_depth_m);
    let mut t = wave_time(p) * (G * h).sqrt();
    let in_band = p.slope_max_depth_m <= 0.0 || h < p.slope_max_depth_m;
    if p.slp > 0.0 && in_band && grad > 0.0 {
        t = t.min(2.0 * std::f64::consts::PI * h / (p.slp * grad));
    }
    t
}

/// Wavelength (m) of the slope low pass at depth `h` and latitude `lat`
/// (degrees), 0 for no filter.
pub fn filter_length_m(p: &BathyRefine, h: f64, lat: f64, radius: f64) -> f64 {
    if p.fl > 0.0 {
        p.fl * 1e3
    } else if p.fl < 0.0 {
        let c = (G * h.max(1.0)).sqrt();
        let phi = lat.to_radians();
        let r = if lat.abs() < 5.0 {
            let beta = 2.0 * OMEGA * phi.cos() / radius;
            (c / (2.0 * beta)).sqrt()
        } else {
            c / (2.0 * OMEGA * phi.sin()).abs()
        };
        r.min(1e7) / p.fl.abs()
    } else {
        0.0
    }
}

/// Smallest depth in [dmin, dmax] whose cell size is at most `target` (m).
pub fn required_depth(earth: &Earth, target: f64, dmin: u8, dmax: u8) -> u8 {
    let mut d = dmin;
    while d < dmax && cell_size_m(earth, d) > target {
        d += 1;
    }
    d
}

/// Limit the growth of a size field on a regular lon/lat block grid:
/// `t[j] <= t[i] + grade·dist(i, j)` over the 8 neighbours, by a forward
/// and a backward sweep (chamfer propagation). `dx[j]` is the east-west
/// spacing of row `j` (m), `dy` the north-south one.
pub fn grade_limit(t: &mut [f32], nx: usize, ny: usize, dx: &[f64], dy: f64, grade: f64) {
    if grade <= 0.0 {
        return;
    }
    let relax = |t: &mut [f32], k: usize, n: usize, d: f64| {
        let v = t[n] as f64 + grade * d;
        if v < t[k] as f64 {
            t[k] = v as f32;
        }
    };
    for _ in 0..2 {
        for j in 0..ny {
            let dd = (dx[j] * dx[j] + dy * dy).sqrt();
            for i in 0..nx {
                let k = j * nx + i;
                if i > 0 {
                    relax(t, k, k - 1, dx[j]);
                }
                if j > 0 {
                    relax(t, k, k - nx, dy);
                    if i > 0 {
                        relax(t, k, k - nx - 1, dd);
                    }
                    if i + 1 < nx {
                        relax(t, k, k - nx + 1, dd);
                    }
                }
            }
        }
        for j in (0..ny).rev() {
            let dd = (dx[j] * dx[j] + dy * dy).sqrt();
            for i in (0..nx).rev() {
                let k = j * nx + i;
                if i + 1 < nx {
                    relax(t, k, k + 1, dx[j]);
                }
                if j + 1 < ny {
                    relax(t, k, k + nx, dy);
                    if i + 1 < nx {
                        relax(t, k, k + nx + 1, dd);
                    }
                    if i > 0 {
                        relax(t, k, k + nx - 1, dd);
                    }
                }
            }
        }
    }
}

/// Summed-area table of a `nx × ny` field, for box means.
struct Sat {
    nx: usize,
    ny: usize,
    s: Vec<f64>,
}

impl Sat {
    fn new(v: &[f32], nx: usize, ny: usize) -> Sat {
        let mut s = vec![0f64; (nx + 1) * (ny + 1)];
        for j in 0..ny {
            let mut row = 0f64;
            for i in 0..nx {
                row += v[j * nx + i] as f64;
                s[(j + 1) * (nx + 1) + i + 1] = s[j * (nx + 1) + i + 1] + row;
            }
        }
        Sat { nx, ny, s }
    }

    /// Mean over [i-rx, i+rx] × [j-ry, j+ry], clipped to the grid.
    fn mean(&self, i: isize, j: isize, rx: isize, ry: isize) -> f64 {
        let i0 = (i - rx).clamp(0, self.nx as isize - 1) as usize;
        let i1 = (i + rx).clamp(0, self.nx as isize - 1) as usize + 1;
        let j0 = (j - ry).clamp(0, self.ny as isize - 1) as usize;
        let j1 = (j + ry).clamp(0, self.ny as isize - 1) as usize + 1;
        let w = self.nx + 1;
        let sum = self.s[j1 * w + i1] - self.s[j0 * w + i1] - self.s[j1 * w + i0] + self.s[j0 * w + i0];
        sum / ((i1 - i0) * (j1 - j0)) as f64
    }
}

/// Result of the bathymetry pass.
pub struct BathyRefinement {
    /// uniq indices of the cells that must be split
    pub splits: HashSet<u64>,
    /// wet blocks: (hash at depth dmax of the block centre, slope, required size in km)
    pub blocks: Vec<(u64, f32, f32)>,
    /// block size in bathymetry samples
    pub block: usize,
}

/// Average the bathymetry on blocks, apply the size functions and the grade
/// limit, and list the cells to split.
pub fn bathy_refinement(b: &Bathy, earth: &Earth, p: &BathyRefine, dmin: u8, dmax: u8) -> BathyRefinement {
    let m = &b.meta;
    let smooth_km = if p.smooth_km > 0.0 { p.smooth_km } else { 0.5 * cell_size_m(earth, dmax) / 1e3 };
    let deg_km = earth.radius * std::f64::consts::PI / 180.0 / 1e3;
    let block = ((smooth_km / deg_km / m.dlat.abs()).round() as usize).max(1);
    let (nx, ny) = (m.nlon / block, m.nlat / block);
    // block means: depth h (land counted as min_depth) and wet flag
    let hb: Vec<(f32, bool)> = (0..ny)
        .into_par_iter()
        .flat_map_iter(|jb| {
            let mut row = Vec::with_capacity(nx);
            for ib in 0..nx {
                let (mut s, mut nw) = (0f64, 0u32);
                for j in jb * block..(jb + 1) * block {
                    for i in ib * block..(ib + 1) * block {
                        let z = b.z[j * m.nlon + i] as f64;
                        s += if z < 0.0 { -z } else { 0.0 };
                        nw += (z < 0.0) as u32;
                    }
                }
                let h = (s / (block * block) as f64).max(p.min_depth_m);
                row.push((h as f32, nw > 0));
            }
            row
        })
        .collect();
    let h: Vec<f32> = hb.iter().map(|x| x.0).collect();
    let lon_c = |ib: usize| m.lon0 + ((ib * block) as f64 + 0.5 * (block - 1) as f64) * m.dlon;
    let lat_c = |jb: usize| m.lat0 + ((jb * block) as f64 + 0.5 * (block - 1) as f64) * m.dlat;
    let dy = (m.dlat.abs() * block as f64).to_radians() * earth.radius;
    let dx: Vec<f64> = (0..ny)
        .map(|jb| (m.dlon.abs() * block as f64).to_radians() * earth.radius * lat_c(jb).to_radians().cos().max(1e-6))
        .collect();
    let sat = if p.fl != 0.0 { Some(Sat::new(&h, nx, ny)) } else { None };

    // slope and size before the grade limit (dry blocks: no requirement)
    let st: Vec<(f32, f32)> = (0..ny)
        .into_par_iter()
        .flat_map_iter(|jb| {
            let lat = lat_c(jb);
            let mut row = Vec::with_capacity(nx);
            for ib in 0..nx {
                let (h0, wet) = hb[jb * nx + ib];
                if !wet {
                    row.push((f32::NAN, f32::INFINITY));
                    continue;
                }
                let (i0, i1) = (ib.saturating_sub(1), (ib + 1).min(nx - 1));
                let (j0, j1) = (jb.saturating_sub(1), (jb + 1).min(ny - 1));
                let val = |i: usize, j: usize| -> f64 {
                    match &sat {
                        None => h[j * nx + i] as f64,
                        Some(s) => {
                            let l = filter_length_m(p, h0 as f64, lat, earth.radius);
                            let rx = (0.5 * l / dx[jb]).round() as isize;
                            let ry = (0.5 * l / dy).round() as isize;
                            s.mean(i as isize, j as isize, rx, ry)
                        }
                    }
                };
                let gx = (val(i1, jb) - val(i0, jb)) / (dx[jb] * (i1 - i0).max(1) as f64);
                let gy = (val(ib, j1) - val(ib, j0)) / (dy * (j1 - j0).max(1) as f64);
                let grad = gx.hypot(gy);
                row.push((grad as f32, target_size_m(p, h0 as f64, grad) as f32));
            }
            row
        })
        .collect();
    let slope: Vec<f32> = st.iter().map(|x| x.0).collect();
    let mut t: Vec<f32> = st.into_iter().map(|x| x.1).collect();
    grade_limit(&mut t, nx, ny, &dx, dy, p.grade);

    let layer = nested::get(dmax);
    let rows: Vec<(Vec<(u64, u8)>, Vec<(u64, f32, f32)>)> = (0..ny)
        .into_par_iter()
        .map(|jb| {
            let beta = earth.to_authalic(lat_c(jb).to_radians());
            let (mut req, mut diag) = (Vec::new(), Vec::new());
            for ib in 0..nx {
                let k = jb * nx + ib;
                if !hb[k].1 {
                    continue;
                }
                let mut lon = lon_c(ib);
                if lon < 0.0 {
                    lon += 360.0;
                }
                let hash = layer.hash(lon.to_radians(), beta);
                let r = required_depth(earth, t[k] as f64, dmin, dmax);
                if r > dmin {
                    req.push((hash, r));
                }
                diag.push((hash, slope[k], t[k] / 1e3));
            }
            (req, diag)
        })
        .collect();
    let mut splits = HashSet::new();
    let mut blocks = Vec::new();
    for (req, diag) in rows {
        for (hash, r) in req {
            // split every cell containing the block from depth r-1 up to dmin;
            // ancestors of an already inserted cell are already there
            for k in (dmin..r).rev() {
                if !splits.insert(uniq(k, hash >> (2 * (dmax - k) as u32))) {
                    break;
                }
            }
        }
        blocks.extend(diag);
    }
    BathyRefinement { splits, blocks, block }
}

/// Mean slope and smallest required size (km) of the blocks in each leaf;
/// NaN for leaves without a wet block.
pub fn leaf_diagnostics(leaves: &[(u8, u64)], blocks: &[(u64, f32, f32)], dmin: u8, dmax: u8) -> (Vec<f64>, Vec<f64>) {
    let index: HashMap<u64, u32> =
        leaves.iter().enumerate().map(|(i, &(d, h))| (uniq(d, h), i as u32)).collect();
    let mut acc = vec![(0f64, 0u32, f64::INFINITY); leaves.len()];
    for &(hash, s, t) in blocks {
        for k in 0..=(dmax - dmin) {
            if let Some(&i) = index.get(&uniq(dmax - k, hash >> (2 * k as u32))) {
                let a = &mut acc[i as usize];
                a.0 += s as f64;
                a.1 += 1;
                a.2 = a.2.min(t as f64);
                break;
            }
        }
    }
    acc.iter()
        .map(|&(s, n, t)| if n > 0 { (s / n as f64, t) } else { (f64::NAN, f64::NAN) })
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::Ellipsoid;

    fn params() -> BathyRefine {
        BathyRefine {
            wl: 0.0,
            wave_time_s: 200.0,
            slp: 10.0,
            fl: 0.0,
            grade: 0.0,
            slope_max_depth_m: 300.0,
            min_depth_m: 5.0,
            smooth_km: 0.0,
        }
    }

    #[test]
    fn wavelength_criterion_scales_with_sqrt_depth() {
        let p = BathyRefine { slp: 0.0, ..params() };
        let a = target_size_m(&p, 100.0, 0.0);
        let b = target_size_m(&p, 400.0, 0.0);
        assert!((b / a - 2.0).abs() < 1e-12);
        assert!((a - 200.0 * (G * 100.0).sqrt()).abs() < 1e-9);
        assert_eq!(target_size_m(&p, 0.0, 0.0), target_size_m(&p, 5.0, 0.0));
        // wl: cells per M2 wavelength
        let q = BathyRefine { wl: 100.0, wave_time_s: 0.0, ..p };
        assert!((target_size_m(&q, 100.0, 0.0) - T_M2 * (G * 100.0).sqrt() / 100.0).abs() < 1e-6);
    }

    #[test]
    fn slope_criterion_in_its_depth_band() {
        let p = params();
        let steep = target_size_m(&p, 100.0, 0.05);
        assert!((steep - 2.0 * std::f64::consts::PI * 100.0 / (10.0 * 0.05)).abs() < 1e-9);
        assert!(steep < target_size_m(&p, 100.0, 0.0));
        assert_eq!(target_size_m(&p, 1000.0, 0.05), target_size_m(&p, 1000.0, 0.0));
        let all = BathyRefine { slope_max_depth_m: 0.0, ..p };
        assert!(target_size_m(&all, 1000.0, 0.05) < target_size_m(&all, 1000.0, 0.0));
    }

    #[test]
    fn rossby_filter_length() {
        let p = BathyRefine { fl: -2.0, ..params() };
        let r = 6.371e6;
        let l = filter_length_m(&p, 100.0, 45.0, r);
        let f = 2.0 * OMEGA * 45f64.to_radians().sin();
        assert!((l - (G * 100.0).sqrt() / f / 2.0).abs() < 1e-6);
        assert!(filter_length_m(&p, 100.0, 1.0, r).is_finite());
        assert_eq!(filter_length_m(&BathyRefine { fl: 20.0, ..p }, 100.0, 45.0, r), 20e3);
        assert_eq!(filter_length_m(&params(), 100.0, 45.0, r), 0.0);
    }

    #[test]
    fn grade_limit_bounds_growth() {
        let (nx, ny) = (50, 3);
        let mut t = vec![f32::INFINITY; nx * ny];
        t[nx + 10] = 1000.0;
        let dx = vec![1000.0; ny];
        grade_limit(&mut t, nx, ny, &dx, 1000.0, 0.2);
        for i in 0..nx {
            let d = (i as f64 - 10.0).abs() * 1000.0;
            assert!((t[nx + i] as f64 - (1000.0 + 0.2 * d)).abs() < 1e-2, "{i} {}", t[nx + i]);
        }
        // diagonal neighbour
        assert!((t[11] as f64 - (1000.0 + 0.2 * 1000.0 * 2f64.sqrt())).abs() < 1e-2);
    }

    #[test]
    fn box_mean() {
        let v: Vec<f32> = (0..20).map(|k| k as f32).collect();
        let s = Sat::new(&v, 5, 4);
        assert_eq!(s.mean(2, 1, 0, 0), 7.0);
        assert_eq!(s.mean(2, 1, 1, 1), 7.0);
        assert_eq!(s.mean(0, 0, 1, 0), 0.5);
    }

    #[test]
    fn required_depth_is_the_coarsest_fine_enough() {
        let e = Earth::new(Ellipsoid::Wgs84);
        assert_eq!(required_depth(&e, 1e9, 7, 11), 7);
        assert_eq!(required_depth(&e, 1.0, 7, 11), 11);
        assert_eq!(required_depth(&e, cell_size_m(&e, 9) * 1.01, 7, 11), 9);
        assert_eq!(required_depth(&e, cell_size_m(&e, 9) * 0.99, 7, 11), 10);
    }
}
