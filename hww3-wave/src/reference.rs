//! CPU reference in double precision: `PDLIB_W3XYPFSN2` written as in WW3,
//! with the loop over triangles that scatters into the nodes, for checking
//! the gather kernels.

use crate::geom::{Geometry, DT};
use rayon::prelude::*;

/// One 600 s step of frequency `ik` (each direction with its own `ITER`) on
/// `u[i nth + k]`, as WW3 does it; `max(0, .)` if `clip`.
pub fn step(g: &Geometry, u0: &[f64], ik: usize, clip: bool) -> Vec<f64> {
    let (np, nth, ne) = (g.np, g.nth, g.ne);
    let cols: Vec<Vec<f64>> = (0..nth)
        .into_par_iter()
        .map(|k| {
            let it = g.iter[ik * nth + k];
            let c: Vec<(f64, f64)> =
                (0..np).map(|i| (g.cx[ik * np + i] * g.ecos[k], g.cy[ik * np + i] * g.esin[k])).collect();
            let mut kel = vec![[0.0f64; 3]; ne];
            let mut flall = vec![[0.0f64; 3]; ne];
            let mut nm = vec![0.0f64; ne];
            for e in 0..ne {
                let ni = &g.ine[3 * e..3 * e + 3];
                let (i1, i2, i3) = (ni[0] as usize, ni[1] as usize, ni[2] as usize);
                let ien = &g.ien[6 * e..6 * e + 6];
                let l0 = (c[i1].0 + c[i2].0 + c[i3].0) / 6.0;
                let l1 = (c[i1].1 + c[i2].1 + c[i3].1) / 6.0;
                let kt = [l0 * ien[0] + l1 * ien[1], l0 * ien[2] + l1 * ien[3], l0 * ien[4] + l1 * ien[5]];
                let neg: f64 = kt.iter().map(|&x| x.min(0.0)).sum();
                nm[e] = -1.0 / (-(f32::MIN_POSITIVE as f64)).min(neg);
                kel[e] = [kt[0].max(0.0), kt[1].max(0.0), kt[2].max(0.0)];
                let fl11 = c[i2].0 * ien[0] + c[i2].1 * ien[1];
                let fl12 = c[i3].0 * ien[0] + c[i3].1 * ien[1];
                let fl21 = c[i3].0 * ien[2] + c[i3].1 * ien[3];
                let fl22 = c[i1].0 * ien[2] + c[i1].1 * ien[3];
                let fl31 = c[i1].0 * ien[4] + c[i1].1 * ien[5];
                let fl32 = c[i2].0 * ien[4] + c[i2].1 * ien[5];
                let (fl111, fl112) = (2.0 * fl11 + fl12, 2.0 * fl12 + fl11);
                let (fl211, fl212) = (2.0 * fl21 + fl22, 2.0 * fl22 + fl21);
                let (fl311, fl312) = (2.0 * fl31 + fl32, 2.0 * fl32 + fl31);
                flall[e] = [
                    (fl311 + fl212) / 6.0 + kel[e][0],
                    (fl111 + fl312) / 6.0 + kel[e][1],
                    (fl211 + fl112) / 6.0 + kel[e][2],
                ];
            }
            let dtsi: Vec<f64> = g.si.iter().map(|&s| DT / it as f64 / s).collect();
            let mut u: Vec<f64> = (0..np).map(|i| u0[i * nth + k]).collect();
            for _ in 0..it {
                let mut st = vec![0.0f64; np];
                for e in 0..ne {
                    let ni = &g.ine[3 * e..3 * e + 3];
                    let ut = nm[e] * (0..3).map(|q| flall[e][q] * u[ni[q] as usize]).sum::<f64>();
                    for q in 0..3 {
                        let n = ni[q] as usize;
                        st[n] += kel[e][q] * (u[n] - ut);
                    }
                }
                for i in 0..np {
                    let mut v = u[i] - dtsi[i] * st[i] * (1.0 - g.iobpa[i] as f64);
                    if clip {
                        v = v.max(0.0);
                    }
                    u[i] = v * g.m(i, k);
                }
            }
            u
        })
        .collect();
    let mut out = vec![0.0; np * nth];
    for (k, col) in cols.iter().enumerate() {
        for i in 0..np {
            out[i * nth + k] = col[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Dims, Sweeper};
    use crate::testgrid::{lattice, uniform};

    /// N-scheme coefficients of one component per triangle: `K⁺` and
    /// `W = NM FLALL`, computed as in [`step`].
    struct Coef {
        kel: Vec<[f64; 3]>,
        w: Vec<[f64; 3]>,
    }

    fn coef(g: &Geometry, ik: usize, k: usize) -> Coef {
        let np = g.np;
        let c: Vec<(f64, f64)> = (0..np).map(|i| (g.cx[ik * np + i] * g.ecos[k], g.cy[ik * np + i] * g.esin[k])).collect();
        let mut kel = vec![[0.0; 3]; g.ne];
        let mut w = vec![[0.0; 3]; g.ne];
        for e in 0..g.ne {
            let v: Vec<usize> = g.ine[3 * e..3 * e + 3].iter().map(|&n| n as usize).collect();
            let n = &g.ien[6 * e..6 * e + 6];
            let l0 = (c[v[0]].0 + c[v[1]].0 + c[v[2]].0) / 6.0;
            let l1 = (c[v[0]].1 + c[v[1]].1 + c[v[2]].1) / 6.0;
            let kt: Vec<f64> = (0..3).map(|q| l0 * n[2 * q] + l1 * n[2 * q + 1]).collect();
            let nm = -1.0 / (-(f32::MIN_POSITIVE as f64)).min(kt.iter().map(|&x| x.min(0.0)).sum());
            let fl = |i: usize, q: usize| c[i].0 * n[2 * q] + c[i].1 * n[2 * q + 1];
            let (f11, f12, f21, f22, f31, f32_) = (fl(v[1], 0), fl(v[2], 0), fl(v[2], 1), fl(v[0], 1), fl(v[0], 2), fl(v[1], 2));
            let simpson = [
                (2.0 * f31 + f32_ + 2.0 * f22 + f21) / 6.0,
                (2.0 * f11 + f12 + 2.0 * f32_ + f31) / 6.0,
                (2.0 * f21 + f22 + 2.0 * f12 + f11) / 6.0,
            ];
            for q in 0..3 {
                kel[e][q] = kt[q].max(0.0);
                w[e][q] = nm * (simpson[q] + kel[e][q]);
            }
        }
        Coef { kel, w }
    }

    /// The N scheme as a CPU backend of [`Sweeper`], one sub-step at a time,
    /// with the transposed sub-step written from the formula of the GPU
    /// kernels: `z_j = m_j ȳ_j − Σ_{T∋j} [K⁺_jT v_j − W_jT Σ_{i∈T} K⁺_iT v_i]`,
    /// `v_i = m_i (1 − IOBPA_i) DTSI_i ȳ_i`.
    struct CpuSweeper<'a> {
        g: &'a Geometry,
        coefs: Vec<Coef>,
    }

    impl<'a> CpuSweeper<'a> {
        fn new(g: &'a Geometry) -> Self {
            let coefs = (0..g.nk * g.nth).map(|c| coef(g, c / g.nth, c % g.nth)).collect();
            CpuSweeper { g, coefs }
        }

        fn dtsi(&self, ik: usize, k: usize, i: usize) -> f64 {
            DT / self.g.iter[ik * self.g.nth + k] as f64 / self.g.si[i] * (1.0 - self.g.iobpa[i] as f64)
        }
    }

    impl Sweeper for CpuSweeper<'_> {
        type Buf = Vec<f64>;
        fn label(&self) -> String {
            "cpu f64".into()
        }
        fn float_bytes(&self) -> usize {
            8
        }
        fn dims(&self) -> Dims {
            Dims { np: self.g.np, ne: self.g.ne, nk: self.g.nk, nth: self.g.nth }
        }
        fn iter(&self) -> &[u32] {
            &self.g.iter
        }
        fn block(&self) -> u32 {
            1
        }
        fn set_block(&mut self, _: u32) {}
        fn variants(&self) -> &'static [&'static str] {
            &["scatter"]
        }
        fn variant(&self) -> usize {
            0
        }
        fn set_variant(&mut self, _: usize) {}
        fn coef_seconds(&self) -> f64 {
            0.0
        }
        fn upload(&self, v: &[f64]) -> Vec<f64> {
            v.to_vec()
        }
        fn download(&self, b: &Vec<f64>) -> Vec<f64> {
            b.clone()
        }
        fn empty(&self) -> Vec<f64> {
            vec![0.0; self.len()]
        }
        fn sync(&self) {}

        fn forward(&self, u: &Vec<f64>, out: &mut Vec<f64>, ik: usize, sub: u32, clip: bool) {
            let (g, nth) = (self.g, self.g.nth);
            for k in 0..nth {
                if sub >= g.iter[ik * nth + k] {
                    (0..g.np).for_each(|i| out[i * nth + k] = u[i * nth + k]);
                    continue;
                }
                let c = &self.coefs[ik * nth + k];
                let mut st = vec![0.0; g.np];
                for e in 0..g.ne {
                    let v = &g.ine[3 * e..3 * e + 3];
                    let ut: f64 = (0..3).map(|q| c.w[e][q] * u[v[q] as usize * nth + k]).sum();
                    for q in 0..3 {
                        let i = v[q] as usize;
                        st[i] += c.kel[e][q] * (u[i * nth + k] - ut);
                    }
                }
                for i in 0..g.np {
                    let mut x = u[i * nth + k] - self.dtsi(ik, k, i) * st[i];
                    if clip {
                        x = x.max(0.0);
                    }
                    out[i * nth + k] = x * g.m(i, k);
                }
            }
        }

        fn transpose(&self, y: &Vec<f64>, out: &mut Vec<f64>, ik: usize, sub: u32) {
            let (g, nth) = (self.g, self.g.nth);
            for k in 0..nth {
                if sub >= g.iter[ik * nth + k] {
                    (0..g.np).for_each(|i| out[i * nth + k] = y[i * nth + k]);
                    continue;
                }
                let c = &self.coefs[ik * nth + k];
                let v: Vec<f64> = (0..g.np).map(|i| g.m(i, k) * self.dtsi(ik, k, i) * y[i * nth + k]).collect();
                let mut z: Vec<f64> = (0..g.np).map(|j| g.m(j, k) * y[j * nth + k]).collect();
                for e in 0..g.ne {
                    let n = &g.ine[3 * e..3 * e + 3];
                    let s: f64 = (0..3).map(|q| c.kel[e][q] * v[n[q] as usize]).sum();
                    for q in 0..3 {
                        let j = n[q] as usize;
                        z[j] -= c.kel[e][q] * v[j] - c.w[e][q] * s;
                    }
                }
                (0..g.np).for_each(|j| out[j * nth + k] = z[j]);
            }
        }
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn max_abs(a: &[f64]) -> f64 {
        a.iter().fold(0.0f64, |m, x| m.max(x.abs()))
    }

    #[test]
    fn sub_steps_reproduce_the_reference_step() {
        let geo = Geometry::new(&lattice(12, 10, 20.0, 15.0, true));
        let cpu = CpuSweeper::new(&geo);
        let x = uniform(geo.np * geo.nth, 1);
        for ik in 0..geo.nk {
            for clip in [false, true] {
                let (got, _) = cpu.step(x.clone(), cpu.empty(), ik, clip);
                let want = step(&geo, &x, ik, clip);
                let diff: Vec<f64> = got.iter().zip(&want).map(|(a, b)| a - b).collect();
                assert!(max_abs(&diff) <= 1e-13 * max_abs(&want), "frequency {ik}, clip {clip}");
            }
        }
    }

    #[test]
    fn transpose_passes_the_dot_product_test() {
        let geo = Geometry::new(&lattice(12, 10, 20.0, 15.0, true));
        let cpu = CpuSweeper::new(&geo);
        let n = geo.np * geo.nth;
        let x = uniform(n, 2);
        let y: Vec<f64> = uniform(n, 3).iter().map(|v| 2.0 * v - 1.0).collect();
        for ik in 0..geo.nk {
            // one sub-step
            let mut lx = cpu.empty();
            cpu.forward(&x, &mut lx, ik, 0, false);
            let mut lty = cpu.empty();
            cpu.transpose(&y, &mut lty, ik, 0);
            let (a, b) = (dot(&lx, &y), dot(&x, &lty));
            assert!((a - b).abs() <= 1e-13 * a.abs().max(b.abs()), "sub-step of frequency {ik}: {a} against {b}");
            // a 600 s step: max ITER sub-steps, transposed in reverse order
            let (lx, _) = cpu.step(x.clone(), cpu.empty(), ik, false);
            let (lty, _) = cpu.step_transpose(y.clone(), cpu.empty(), ik);
            let (a, b) = (dot(&lx, &y), dot(&x, &lty));
            assert!((a - b).abs() <= 1e-12 * a.abs().max(b.abs()), "step of frequency {ik}: {a} against {b}");
        }
        assert!(geo.max_iter(0) > 1, "the test grid should sub-cycle");
    }

    #[test]
    fn uniform_state_stays_uniform_with_constant_velocity() {
        // with a constant velocity, NM Σ FLALL = 1 on every triangle, so a
        // uniform state is a fixed point of a sub-step at the interior
        // nodes. One sub-step only: the coast masks zero the directions
        // coming from land, and further sub-steps would carry those zeros
        // inwards.
        let mut geo = Geometry::new(&lattice(12, 10, 20.0, 15.0, true));
        let np = geo.np;
        for ik in 0..geo.nk {
            let (cx, cy) = (geo.cx[ik * np + np / 2], geo.cy[ik * np + np / 2]);
            geo.cx[ik * np..(ik + 1) * np].iter_mut().for_each(|v| *v = cx);
            geo.cy[ik * np..(ik + 1) * np].iter_mut().for_each(|v| *v = cy);
        }
        geo.iter.iter_mut().for_each(|it| *it = 1);
        let u = vec![1.0; np * geo.nth];
        for ik in 0..geo.nk {
            let out = step(&geo, &u, ik, false);
            for i in (0..np).filter(|&i| geo.iobp[i] != 0) {
                for k in 0..geo.nth {
                    assert!((out[i * geo.nth + k] - 1.0).abs() < 1e-12, "node {i}, direction {k}");
                }
            }
        }
    }

    #[test]
    fn step_is_linear_without_the_clip() {
        let geo = Geometry::new(&lattice(8, 7, 30.0, 10.0, false));
        let n = geo.np * geo.nth;
        let (x, y) = (uniform(n, 4), uniform(n, 5));
        let xy: Vec<f64> = x.iter().zip(&y).map(|(a, b)| 2.0 * a - 3.0 * b).collect();
        let (lx, ly, lxy) = (step(&geo, &x, 1, false), step(&geo, &y, 1, false), step(&geo, &xy, 1, false));
        for j in 0..n {
            assert!((lxy[j] - (2.0 * lx[j] - 3.0 * ly[j])).abs() < 1e-12);
        }
    }
}
