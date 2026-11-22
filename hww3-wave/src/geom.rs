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
//! `ITER = ceil(DT / min_i SI_i / KKSUM_i)` over the interior nodes
//! (`IOBP = 1`, `FSBCCFL = F`), with `KKSUM_i = Σ_T max(0, K_iT)`.

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

pub struct Geometry {
    pub np: usize,
    pub ne: usize,
    pub nk: usize,
    pub nth: usize,
    /// Vertices per triangle, counter-clockwise in (lon, lat), `3 ne`.
    pub ine: Vec<u32>,
    /// `PDLIB_IEN`, `6 ne`: (n1x, n1y, n2x, n2y, n3x, n3y).
    pub ien: Vec<f64>,
    /// `PDLIB_SI` (degree²).
    pub si: Vec<f64>,
    pub iobp: Vec<u8>,
    pub iobpa: Vec<u8>,
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
    pub slot: Vec<u32>,
    pub ecos: Vec<f64>,
    pub esin: Vec<f64>,
    pub freqs: Vec<f64>,
    /// `FACX cg / CLATS` and `FACY cg` per frequency and node (`nk np`), times `IOBDP`.
    pub cx: Vec<f64>,
    pub cy: Vec<f64>,
    /// `ITER(ik, ith)`, `nk nth`.
    pub iter: Vec<u32>,
    /// Largest `DT / dt_i` per component (WW3 `CFLXY`), `nk nth`.
    pub cflxy: Vec<f64>,
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
    pub fn new(g: &Grid) -> Geometry {
        let np = g.lon.len();
        let nk = g.spec.nk;
        let nth = g.spec.nth;
        assert!(nth <= 63, "at most 63 directions (bit 63 of the mask word pair holds IOBPA)");
        let (x, y) = (&g.lon, &g.lat);
        // counter-clockwise triangles (WW3 swaps the triangles of negative area)
        let tri: Vec<[u32; 3]> = g
            .tri
            .iter()
            .map(|t| {
                let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
                let a2 = (x[b] - x[a]) * (y[c] - y[a]) - (x[c] - x[a]) * (y[b] - y[a]);
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
                x[c] - x[b],
                y[c] - y[a],
                x[a] - x[c],
                y[a] - y[b],
                x[b] - x[a],
            ]);
            let area = 0.5 * ((x[b] - x[a]) * (y[c] - y[a]) - (x[c] - x[a]) * (y[b] - y[a]));
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
                let (ax, ay) = (x[nx] - x[v], y[nx] - y[v]);
                let (bx, by) = (x[pv] - x[v], y[pv] - y[v]);
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
        };
        geo.cflxy = geo.cfl();
        geo.iter = geo.cflxy.iter().map(|&c| (c.ceil() as u32).max(1)).collect();
        geo
    }

    /// `DT max_i KKSUM_i / SI_i` per component over the interior nodes.
    fn cfl(&self) -> Vec<f64> {
        let (np, nth) = (self.np, self.nth);
        let mut out = Vec::with_capacity(self.nk * nth);
        for ik in 0..self.nk {
            let cx = &self.cx[ik * np..(ik + 1) * np];
            let cy = &self.cy[ik * np..(ik + 1) * np];
            let best = (0..np)
                .into_par_iter()
                .filter(|&i| self.iobp[i] == 1 && self.iobdp[i] == 1)
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
