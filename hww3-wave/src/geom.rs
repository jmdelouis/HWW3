//! Geometry, boundary flags, velocities and sub-cycling of the WW3 PDLIB
//! explicit N scheme (`w3profsmd_pdlib.F90`), for a spherical HWW3 grid.
//!
//! As in WW3, coordinates are in degrees: `PDLIB_IEN` holds the inward
//! normals of the edges opposite each vertex, `n1 = (y2 - y3, x3 - x2)`,
//! `PDLIB_SI` a third of the area of the triangles around a node, and the
//! advection velocity of component `(f, θ)` at a node is
//! `C = (FACX cg cos θ / CLATS, FACY cg sin θ)` in degrees per second, with
//! `FACX = FACY = 1 / (DERA RADIUS)`, multiplied by `IOBDP`.
//!
//! Boundary flags (`SET_UG_IOBP_PDLIB_INIT`, `SET_IOBPA_PDLIB`):
//! - `IOBP` = 0 on the coastal boundary, 1 inside, 2 on the open boundary;
//! - `IOBPA` = 1 on the open boundary (no update, `MAPSTA = 2`);
//! - `IOBDP` = 0 where the depth is below `DMIN + 1e-5` (dry);
//! - `IOBPD(θ, i)` = 1 inside and on the open boundary; on the coast, 1 when
//!   `-e_θ` points into one of the triangles around the node (waves coming
//!   from the sea), with WW3's `eDet` test.
//!
//! Sub-cycling (`PDLIB_W3XYPFSN2`, `LCALC`): for each component,
//! `ITER = ceil(DT / min_i SI_i / KKSUM_i)` over every wet node, coast and
//! open boundary included, as WW3 does with `FSBCCFL = T` (`UGBCCFL` of the
//! `UNST` namelist of `ww3_grid`, true by default and in the HWW3 runs), with
//! `KKSUM_i = Σ_T max(0, K_iT)`; [`Geometry::set_bccfl`] switches to the
//! interior nodes only (`FSBCCFL = F`).

use crate::grid::Grid;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Global time step of the HWW3 runs (s).
pub const DT: f64 = 600.0;
const GRAV: f64 = 9.806;
const RADIUS: f64 = 4.0e7 / (2.0 * PI);
const DERA: f64 = PI / 180.0;
/// WW3 `THR = TINY(1.)`.
const THR: f64 = f32::MIN_POSITIVE as f64;

/// What WW3 builds from an unstructured grid for the PDLIB explicit scheme
/// (`PDLIB_INIT`, `SET_UG_IOBP_PDLIB_INIT`), plus the nodal advection
/// velocities and the sub-cycling of every spectral component.
pub struct Geometry {
    /// Number of nodes.
    pub np: usize,
    /// Number of triangles.
    pub ne: usize,
    /// Number of frequencies.
    pub nk: usize,
    /// Number of directions (at most 63).
    pub nth: usize,
    /// Vertices per triangle, counter-clockwise in (lon, lat), `3 ne`.
    pub ine: Vec<u32>,
    /// `PDLIB_IEN`, `6 ne`: (n1x, n1y, n2x, n2y, n3x, n3y).
    pub ien: Vec<f64>,
    /// `PDLIB_SI` (degree²).
    pub si: Vec<f64>,
    /// `IOBP`: 0 on the coast, 1 inside, 2 on the open boundary.
    pub iobp: Vec<u8>,
    /// `IOBPA`: 1 on the open boundary, where the scheme does not update.
    pub iobpa: Vec<u8>,
    /// `IOBDP`: 0 where the depth is below `DMIN + 1e-5` (dry), else 1.
    pub iobdp: Vec<u8>,
    /// `IOBPD` as a bit mask per node (bit k for direction k), times `IOBDP`.
    pub mask: Vec<u64>,
    /// `IOBPD` alone (all bits set inside and on the open boundary).
    pub iobpd: Vec<u64>,
    /// `PDLIB_TRIA`: triangle areas (degree²), `ne`.
    pub tria: Vec<f64>,
    /// Triangles around each node: `slot[start[i]..start[i+1]]` = `3 t + p`,
    /// `p` the position of the node in triangle `t`.
    pub start: Vec<u32>,
    /// See [`Geometry::start`].
    pub slot: Vec<u32>,
    /// `cos θ` of the directions (`ECOS`, values below 1e-5 set to zero).
    pub ecos: Vec<f64>,
    /// `sin θ` of the directions (`ESIN`, values below 1e-5 set to zero).
    pub esin: Vec<f64>,
    /// Frequencies (Hz), `FREQ1 XFR^k`.
    pub freqs: Vec<f64>,
    /// `FACX cg / CLATS` and `FACY cg` per frequency and node (`nk np`), times `IOBDP`.
    pub cx: Vec<f64>,
    /// See [`Geometry::cx`].
    pub cy: Vec<f64>,
    /// `ITER(ik, ith)`, `nk nth`.
    pub iter: Vec<u32>,
    /// Largest `DT / dt_i` per component (WW3 `CFLXY`), `nk nth`.
    pub cflxy: Vec<f64>,
    /// `FSBCCFL`: whether `cflxy` and `iter` count every wet node (true) or
    /// the interior nodes only.
    pub bccfl: bool,
}

/// A longitude difference (degrees) brought into [-180, 180].
pub fn wrap_lon(d: f64) -> f64 {
    if d > 180.0 {
        d - 360.0
    } else if d < -180.0 {
        d + 360.0
    } else {
        d
    }
}

/// Linear group speed (m/s) of frequency `f` (Hz) at depth `h` (m).
pub fn group_speed(f: f64, h: f64) -> f64 {
    let w = 2.0 * PI * f;
    let mut k = (w * w / GRAV).max(w / (GRAV * h).sqrt());
    for _ in 0..60 {
        let t = (k * h).tanh();
        let step = (GRAV * k * t - w * w) / (GRAV * t + GRAV * k * h * (1.0 - t * t));
        k -= step;
        if step.abs() < 1e-14 * k {
            break;
        }
    }
    let kh = k * h;
    let n = 0.5 * (1.0 + if kh < 20.0 { 2.0 * kh / (2.0 * kh).sinh() } else { 0.0 });
    n * w / k
}

impl Geometry {
    /// Build the geometry, the boundary flags, the velocities at rest (no
    /// current, bathymetric depth) and `ITER` of a grid.
    pub fn new(g: &Grid) -> Geometry {
        let np = g.lon.len();
        let nk = g.spec.nk;
        let nth = g.spec.nth;
        assert!(nth <= 63, "at most 63 directions (bit 63 of the mask word pair holds IOBPA)");
        let (x, y) = (&g.lon, &g.lat);
        // longitude differences taken across the date line when shorter
        // (global grids closed in longitude, crate::structgrid); a regional
        // mesh never has an edge longer than 180°
        let dx = |p: usize, q: usize| wrap_lon(x[p] - x[q]);
        // counter-clockwise triangles (WW3 swaps the triangles of negative area)
        let tri: Vec<[u32; 3]> = g
            .tri
            .iter()
            .map(|t| {
                let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
                let a2 = dx(b, a) * (y[c] - y[a]) - dx(c, a) * (y[b] - y[a]);
                if a2 < 0.0 { [t[0], t[2], t[1]] } else { *t }
            })
            .collect();
        let ne = tri.len();
        let mut ine = Vec::with_capacity(3 * ne);
        let mut ien = Vec::with_capacity(6 * ne);
        let mut si = vec![0.0; np];
        let mut tria = Vec::with_capacity(ne);
        for t in &tri {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            ine.extend_from_slice(t);
            ien.extend_from_slice(&[
                y[b] - y[c],
                dx(c, b),
                y[c] - y[a],
                dx(a, c),
                y[a] - y[b],
                dx(b, a),
            ]);
            let area = 0.5 * (dx(b, a) * (y[c] - y[a]) - dx(c, a) * (y[b] - y[a]));
            for &v in t {
                si[v as usize] += area / 3.0;
            }
            tria.push(area);
        }
        // boundary flags
        let bnd = crate::grid::boundary_nodes(&tri, np);
        let mut iobp: Vec<u8> = bnd.iter().map(|&b| if b { 0 } else { 1 }).collect();
        for &o in &g.obc {
            iobp[o as usize] = 2;
        }
        let iobpa: Vec<u8> = iobp.iter().map(|&b| (b == 2) as u8).collect();
        let iobdp: Vec<u8> = g.depth.iter().map(|&d| (d >= g.spec.dmin + 1e-5) as u8).collect();
        // directions (W3GRID: ECOS/ESIN below 1e-5 set to zero)
        let dth = 2.0 * PI / nth as f64;
        let clean = |v: f64| if v.abs() < 1e-5 { 0.0 } else { v };
        let ecos: Vec<f64> = (0..nth).map(|k| clean(((k as f64 + g.spec.thoff) * dth).cos())).collect();
        let esin: Vec<f64> = (0..nth).map(|k| clean(((k as f64 + g.spec.thoff) * dth).sin())).collect();
        // IOBPD
        let full: u64 = (1u64 << nth) - 1;
        let mut iobpd = vec![0u64; np];
        for t in &tri {
            for q in 0..3 {
                let v = t[q] as usize;
                if iobp[v] != 0 {
                    iobpd[v] = full;
                    continue;
                }
                let (nx, pv) = (t[(q + 1) % 3] as usize, t[(q + 2) % 3] as usize);
                let (ax, ay) = (dx(nx, v), y[nx] - y[v]);
                let (bx, by) = (dx(pv, v), y[pv] - y[v]);
                for k in 0..nth {
                    let (evx, evy) = (ecos[k], esin[k]);
                    let d1 = THR - ax * evy + ay * evx;
                    let d2 = THR + bx * evy - by * evx;
                    if d1 > 0.0 && d2 > 0.0 {
                        iobpd[v] |= 1u64 << k;
                    }
                }
            }
        }
        let mask: Vec<u64> = (0..np)
            .map(|i| if iobdp[i] == 0 { 0 } else if iobpa[i] == 1 { full } else { iobpd[i] })
            .collect();
        // node -> triangles
        let mut start = vec![0u32; np + 1];
        for t in &tri {
            for &v in t {
                start[v as usize + 1] += 1;
            }
        }
        for i in 0..np {
            start[i + 1] += start[i];
        }
        let mut fill: Vec<u32> = start[..np].to_vec();
        let mut slot = vec![0u32; 3 * ne];
        for (e, t) in tri.iter().enumerate() {
            for (p, &v) in t.iter().enumerate() {
                slot[fill[v as usize] as usize] = (3 * e + p) as u32;
                fill[v as usize] += 1;
            }
        }
        // velocities
        let freqs: Vec<f64> = (0..nk).map(|k| g.spec.freq1 * g.spec.xfr.powi(k as i32)).collect();
        let fac = 1.0 / (DERA * RADIUS);
        let mut cx = vec![0.0; nk * np];
        let mut cy = vec![0.0; nk * np];
        cx.par_chunks_mut(np).zip(cy.par_chunks_mut(np)).enumerate().for_each(|(ik, (cxk, cyk))| {
            for i in 0..np {
                if iobdp[i] == 1 {
                    let cg = group_speed(freqs[ik], g.depth[i].max(g.spec.dmin));
                    cxk[i] = fac * cg / (y[i] * DERA).cos();
                    cyk[i] = fac * cg;
                }
            }
        });
        let mut geo = Geometry {
            np,
            ne,
            nk,
            nth,
            ine,
            ien,
            si,
            iobp,
            iobpa,
            iobdp,
            mask,
            iobpd,
            tria,
            start,
            slot,
            ecos,
            esin,
            freqs,
            cx,
            cy,
            iter: vec![],
            cflxy: vec![],
            bccfl: true,
        };
        geo.set_bccfl(true);
        geo
    }

    /// Sub-cycling from every wet node (`bccfl`, WW3's `FSBCCFL = T`, the
    /// default) or from the interior nodes only; recomputes `cflxy` and `iter`.
    pub fn set_bccfl(&mut self, bccfl: bool) {
        self.bccfl = bccfl;
        self.cflxy = self.cfl();
        self.iter = self.cflxy.iter().map(|&c| (c.ceil() as u32).max(1)).collect();
    }

    /// `DT max_i KKSUM_i / SI_i` per component over the wet nodes counted by
    /// `bccfl`.
    fn cfl(&self) -> Vec<f64> {
        let (np, nth) = (self.np, self.nth);
        let mut out = Vec::with_capacity(self.nk * nth);
        for ik in 0..self.nk {
            let cx = &self.cx[ik * np..(ik + 1) * np];
            let cy = &self.cy[ik * np..(ik + 1) * np];
            let best = (0..np)
                .into_par_iter()
                .filter(|&i| (self.bccfl || self.iobp[i] == 1) && self.iobdp[i] == 1)
                .fold(
                    || vec![0.0f64; nth],
                    |mut acc, i| {
                        let mut kk = vec![0.0f64; nth];
                        for s in self.start[i]..self.start[i + 1] {
                            let code = self.slot[s as usize] as usize;
                            let (t, p) = (code / 3, code % 3);
                            let v = &self.ine[3 * t..3 * t + 3];
                            let sx: f64 = v.iter().map(|&n| cx[n as usize]).sum::<f64>() / 6.0;
                            let sy: f64 = v.iter().map(|&n| cy[n as usize]).sum::<f64>() / 6.0;
                            let (nx, ny) = (self.ien[6 * t + 2 * p], self.ien[6 * t + 2 * p + 1]);
                            let (a, b) = (sx * nx, sy * ny);
                            for k in 0..nth {
                                kk[k] += (a * self.ecos[k] + b * self.esin[k]).max(0.0);
                            }
                        }
                        for k in 0..nth {
                            acc[k] = acc[k].max(DT * kk[k].max(1e-9) / self.si[i]);
                        }
                        acc
                    },
                )
                .reduce(|| vec![0.0f64; nth], |a, b| a.iter().zip(&b).map(|(x, y)| x.max(*y)).collect());
            out.extend(best);
        }
        out
    }

    /// Largest `ITER` over the directions of frequency `ik`: the number of
    /// frequency sweeps the GPU needs for it.
    pub fn max_iter(&self, ik: usize) -> u32 {
        *self.iter[ik * self.nth..(ik + 1) * self.nth].iter().max().unwrap()
    }

    /// Mask of direction `k` at node `i` (`IOBPD IOBDP`).
    pub fn m(&self, i: usize, k: usize) -> f64 {
        ((self.mask[i] >> k) & 1) as f64
    }

    /// `1 / SI`, `DT / ITER`, and the mask words of the kernels: bit k of
    /// `mlo` / `mhi` is `IOBPD(k, i) IOBDP_i` for k < 32 / k >= 32, bit 31
    /// of `mhi` is `IOBPA_i`.
    pub fn host_arrays(&self) -> (Vec<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
        let si_inv: Vec<f64> = self.si.iter().map(|&s| 1.0 / s).collect();
        let dtk: Vec<f64> = self.iter.iter().map(|&it| DT / it as f64).collect();
        let mlo: Vec<u32> = self.mask.iter().map(|&m| m as u32).collect();
        let mhi: Vec<u32> = (0..self.np)
            .map(|i| ((self.mask[i] >> 32) as u32 & 0x7fff_ffff) | ((self.iobpa[i] as u32) << 31))
            .collect();
        (si_inv, dtk, mlo, mhi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgrid::{lattice, STEP};

    #[test]
    fn group_speed_limits() {
        // deep water: cg = g / (2 ω)
        let (f, h) = (0.2, 5000.0);
        let deep = GRAV / (2.0 * 2.0 * PI * f);
        assert!((group_speed(f, h) / deep - 1.0).abs() < 1e-12);
        // shallow water: cg -> sqrt(g h)
        let (f, h) = (0.005, 2.0);
        assert!((group_speed(f, h) / (GRAV * h).sqrt() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn triangles_areas_and_normals() {
        let g = lattice(12, 10, 20.0, 15.0, true);
        let geo = Geometry::new(&g);
        assert_eq!((geo.np, geo.ne), (120, 2 * 11 * 9));
        // every triangle counter-clockwise, even those listed clockwise
        assert!(geo.tria.iter().all(|&a| a > 0.0));
        let total = 11.0 * 9.0 * STEP * STEP;
        let sum_tria: f64 = geo.tria.iter().sum();
        let sum_si: f64 = geo.si.iter().sum();
        assert!((sum_tria / total - 1.0).abs() < 1e-12);
        assert!((sum_si / total - 1.0).abs() < 1e-12);
        for e in 0..geo.ne {
            let n = &geo.ien[6 * e..6 * e + 6];
            // the three edge normals of a triangle sum to zero
            assert!((n[0] + n[2] + n[4]).abs() < 1e-14 && (n[1] + n[3] + n[5]).abs() < 1e-14);
            // the normal of the edge opposite vertex q points towards q
            let v = &geo.ine[3 * e..3 * e + 3];
            for q in 0..3 {
                let (a, b) = (v[q] as usize, v[(q + 1) % 3] as usize);
                let d = (g.lon[a] - g.lon[b]) * n[2 * q] + (g.lat[a] - g.lat[b]) * n[2 * q + 1];
                assert!(d > 0.0, "triangle {e}, vertex {q}");
            }
        }
        // the node -> triangle table lists each (triangle, vertex) once
        let mut seen = vec![false; 3 * geo.ne];
        for i in 0..geo.np {
            for s in geo.start[i]..geo.start[i + 1] {
                let code = geo.slot[s as usize] as usize;
                assert_eq!(geo.ine[code] as usize, i);
                assert!(!seen[code]);
                seen[code] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn boundary_flags() {
        let (nx, ny) = (12, 10);
        let geo = Geometry::new(&lattice(nx, ny, 20.0, 15.0, true));
        let full = (1u64 << geo.nth) - 1;
        for j in 0..ny {
            for i in 0..nx {
                let n = j * nx + i;
                let expect = if i == 0 {
                    2
                } else if j == 0 || j == ny - 1 || i == nx - 1 {
                    0
                } else {
                    1
                };
                assert_eq!(geo.iobp[n], expect, "node ({i}, {j})");
                assert_eq!(geo.iobpa[n], (expect == 2) as u8);
                assert_eq!(geo.iobdp[n], 1);
                if expect != 0 {
                    assert_eq!(geo.mask[n], full);
                }
            }
        }
    }

    #[test]
    fn coast_node_takes_waves_from_the_sea_only() {
        // a node of the southern coast (sea to the north) takes the
        // directions whose opposite points into the sea: sin θ <= 0
        let nx = 12;
        let geo = Geometry::new(&lattice(nx, 10, 20.0, 15.0, true));
        let n = 5;
        assert_eq!(geo.iobp[n], 0);
        for k in 0..geo.nth {
            let set = (geo.iobpd[n] >> k) & 1 == 1;
            assert_eq!(set, geo.esin[k] <= 0.0, "direction {k}");
            assert_eq!(geo.m(n, k), set as u8 as f64);
        }
        // a node of the eastern coast (sea to the west) takes cos θ >= 0
        let n = 4 * nx + nx - 1;
        assert_eq!(geo.iobp[n], 0);
        for k in 0..geo.nth {
            assert_eq!((geo.iobpd[n] >> k) & 1 == 1, geo.ecos[k] >= 0.0, "direction {k}");
        }
    }

    #[test]
    fn dry_node_is_masked() {
        let mut g = lattice(12, 10, 20.0, 15.0, true);
        let n = 4 * 12 + 6;
        g.depth[n] = 0.5 * g.spec.dmin;
        let geo = Geometry::new(&g);
        assert_eq!(geo.iobdp[n], 0);
        assert_eq!(geo.mask[n], 0);
        assert!((0..geo.nk).all(|ik| geo.cx[ik * geo.np + n] == 0.0 && geo.cy[ik * geo.np + n] == 0.0));
    }

    #[test]
    fn sub_cycling_keeps_the_cfl_of_every_wet_node_below_one() {
        let geo = Geometry::new(&lattice(12, 10, 20.0, 15.0, true));
        let (np, nth) = (geo.np, geo.nth);
        for ik in 0..geo.nk {
            assert_eq!(geo.max_iter(ik), *geo.iter[ik * nth..(ik + 1) * nth].iter().max().unwrap());
            for k in 0..nth {
                let it = geo.iter[ik * nth + k];
                assert_eq!(it, (geo.cflxy[ik * nth + k].ceil() as u32).max(1));
                let mut worst = 0.0f64;
                for i in (0..np).filter(|&i| geo.iobdp[i] == 1) {
                    let mut kk = 0.0;
                    for s in geo.start[i]..geo.start[i + 1] {
                        let code = geo.slot[s as usize] as usize;
                        let (t, p) = (code / 3, code % 3);
                        let v = &geo.ine[3 * t..3 * t + 3];
                        let sx: f64 = v.iter().map(|&n| geo.cx[ik * np + n as usize]).sum::<f64>() / 6.0;
                        let sy: f64 = v.iter().map(|&n| geo.cy[ik * np + n as usize]).sum::<f64>() / 6.0;
                        let kt = sx * geo.ecos[k] * geo.ien[6 * t + 2 * p] + sy * geo.esin[k] * geo.ien[6 * t + 2 * p + 1];
                        kk += kt.max(0.0);
                    }
                    worst = worst.max(DT / it as f64 * kk / geo.si[i]);
                }
                assert!(worst <= 1.0 + 1e-12, "frequency {ik}, direction {k}: CFL {worst}");
                assert!(worst > 0.0);
            }
        }
    }

    #[test]
    fn host_arrays_pack_the_masks() {
        let geo = Geometry::new(&lattice(12, 10, 20.0, 15.0, true));
        let (si_inv, dtk, mlo, mhi) = geo.host_arrays();
        for i in 0..geo.np {
            assert_eq!(si_inv[i], 1.0 / geo.si[i]);
            assert_eq!(mlo[i] as u64, geo.mask[i] & 0xffff_ffff);
            assert_eq!(mhi[i] >> 31, geo.iobpa[i] as u32);
        }
        for (d, &it) in dtk.iter().zip(&geo.iter) {
            assert_eq!(*d, DT / it as f64);
        }
    }
}
