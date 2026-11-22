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
