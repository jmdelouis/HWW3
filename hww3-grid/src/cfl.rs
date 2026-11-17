//! Largest explicit time step of the WW3 unstructured propagation schemes.
//!
//! WW3 computes it in `w3profsmd.F90` (`W3XYPFSN2`, `W3CFLUG`): for a
//! triangle with vertices 1, 2, 3, the flux coefficients are
//! $K_k = \lambda\cdot\mathbf n_k$, with $\lambda$ half the mean of the
//! three nodal advection velocities and $\mathbf n_k$ the inward normal of the
//! edge opposite vertex $k$, of length that edge (`IEN`). For every spectral
//! component the largest stable step at node $i$ is
//!
//! $$ \Delta t_i = S_i \Big/ \sum_{T \ni i} \max(0, K_{i,T}), $$
//!
//! with $S_i$ a third of the area of the triangles around $i$ (`SI`), and the
//! global step is the minimum over the interior nodes (`IOBP = 1`; open and
//! coastal boundary nodes are left out unless `FSBCCFL`). The explicit
//! schemes sub-cycle: each component takes `ceil(DT / min Δt_i)` steps.
//!
//! For a velocity $c$ in direction $\theta$, $\Delta t_i = L_i(\theta)/c$
//! with $L_i(\theta) = S_i / \sum_T \max(0, \tfrac12\mathbf e_\theta\cdot\mathbf n_{i,T})$
//! a length that depends only on the geometry of the triangles. This module
//! computes $L_i = \min_\theta L_i(\theta)$ over the model directions and
//! $\Delta t_i = L_i / c_g$ with the group speed of the lowest model
//! frequency at the node depth (the fastest component), currents excluded.

use crate::geo::{dot, tangent_basis};
use rayon::prelude::*;
use serde::Serialize;

/// Radius used for the lengths (m).
const R_EARTH: f64 = 6_371_007.0;
const G: f64 = 9.806;

/// Linear group speed (m/s) of frequency `f` (Hz) at depth `h` (m).
pub fn group_speed(f: f64, h: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * f;
    let h = h.max(0.1);
    // solve w^2 = g k tanh(k h) by Newton from the deep-water guess
    let mut k = (w * w / G).max(w / (G * h).sqrt());
    for _ in 0..50 {
        let t = (k * h).tanh();
        let fk = G * k * t - w * w;
        let dk = G * t + G * k * h * (1.0 - t * t);
        let step = fk / dk;
        k -= step;
        if step.abs() < 1e-12 * k {
            break;
        }
    }
    let kh = k * h;
    let n = 0.5 * (1.0 + if kh < 20.0 { 2.0 * kh / (2.0 * kh).sinh() } else { 0.0 });
    n * w / k
}

/// Per-node CFL length $L_i$ (m) and interior flag.
///
/// `nodes` are unit vectors, `tris` counter-clockwise, `nth` directions
/// starting at `dir_offset` (fraction of the direction step, as
/// `SPECTRUM%THOFF`).
pub fn cfl_length(nodes: &[[f64; 3]], tris: &[[u32; 3]], nth: u32, dir_offset: f64) -> (Vec<f64>, Vec<bool>) {
    let n = nodes.len();
    let mut start = vec![0usize; n + 1];
    for t in tris {
        for &v in t {
            start[v as usize + 1] += 1;
        }
    }
    for i in 0..n {
        start[i + 1] += start[i];
    }
    let mut fill = start.clone();
    let mut adj = vec![0u32; start[n]];
    for (k, t) in tris.iter().enumerate() {
        for &v in t {
            adj[fill[v as usize]] = k as u32;
            fill[v as usize] += 1;
        }
    }
    let dth = 2.0 * std::f64::consts::PI / nth as f64;
    let dirs: Vec<[f64; 2]> = (0..nth)
        .map(|k| {
            let th = (k as f64 + dir_offset) * dth;
            [th.cos(), th.sin()]
        })
        .collect();
    (0..n)
        .into_par_iter()
        .map(|i| {
            let p = &nodes[i];
            let (e, nn) = tangent_basis(p);
            let proj = |q: &[f64; 3]| -> [f64; 2] {
                let w = dot(q, p);
                [R_EARTH * dot(q, &e) / w, R_EARTH * dot(q, &nn) / w]
            };
            let ts = &adj[start[i]..start[i + 1]];
            let mut si = 0.0;
            let mut normals: Vec<[f64; 2]> = Vec::with_capacity(ts.len());
            // edges around i: each counted twice when i is interior
            let mut edge_count: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for &k in ts {
                let t = tris[k as usize];
                let pos = t.iter().position(|&v| v as usize == i).unwrap();
                let a = proj(&nodes[t[0] as usize]);
                let b = proj(&nodes[t[1] as usize]);
                let c = proj(&nodes[t[2] as usize]);
                let area = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]));
                si += area / 3.0;
                let pts = [a, b, c];
                let (j, l) = (pts[(pos + 1) % 3], pts[(pos + 2) % 3]);
                // inward normal of the edge opposite i (WW3: N1 = (y2 - y3, x3 - x2))
                normals.push([j[1] - l[1], l[0] - j[0]]);
                for o in [t[(pos + 1) % 3], t[(pos + 2) % 3]] {
                    *edge_count.entry(o).or_insert(0) += 1;
                }
            }
            let interior = !ts.is_empty() && edge_count.values().all(|&c| c == 2);
            let mut l = f64::INFINITY;
            for d in &dirs {
                let kk: f64 = normals.iter().map(|m| (0.5 * (d[0] * m[0] + d[1] * m[1])).max(0.0)).sum();
                if kk > 0.0 {
                    l = l.min(si / kk);
                }
            }
            (l, interior)
        })
        .unzip()
}

fn node_length(i: usize, ts: &[u32], nodes: &[[f64; 3]], tris: &[[u32; 3]], dirs: &[[f64; 2]]) -> f64 {
    let p = &nodes[i];
    let (e, nn) = tangent_basis(p);
    let proj = |q: &[f64; 3]| -> [f64; 2] {
        let w = dot(q, p);
        [dot(q, &e) / w, dot(q, &nn) / w]
    };
    let mut si = 0.0;
    let mut normals: Vec<[f64; 2]> = Vec::with_capacity(ts.len());
    for &k in ts {
        let t = tris[k as usize];
        let pos = t.iter().position(|&v| v as usize == i).unwrap();
        let a = proj(&nodes[t[0] as usize]);
        let b = proj(&nodes[t[1] as usize]);
        let c = proj(&nodes[t[2] as usize]);
        si += ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 6.0;
        let pts = [a, b, c];
        let (j, l) = (pts[(pos + 1) % 3], pts[(pos + 2) % 3]);
        normals.push([j[1] - l[1], l[0] - j[0]]);
    }
    let mut l = f64::INFINITY;
    for d in dirs {
        let kk: f64 = normals.iter().map(|m| (0.5 * (d[0] * m[0] + d[1] * m[1])).max(0.0)).sum();
        if kk > 0.0 {
            l = l.min(si / kk);
        }
    }
    l
}

/// Edge flips that lengthen the explicit time step where it is short.
///
/// For an interior edge whose quadrilateral holds an interior node with a
/// CFL length below `threshold` times its cell size, the edge is flipped
/// when the quadrilateral is convex, the level jump across the new edge is
/// at most `max_level_jump`, the new triangles keep their angles within
/// [`min_angle`, `max_angle`] (radians) and the smallest CFL length over
/// the interior nodes of the quadrilateral (divided by their cell sizes)
/// increases by more than 0.1 %. The nodes do not move. Returns the number
/// of flips.
#[allow(clippy::too_many_arguments)]
pub fn cfl_flips(
    nodes: &[[f64; 3]],
    levels: &[u8],
    nth: u32,
    dir_offset: f64,
    threshold: f64,
    min_angle: f64,
    max_angle: f64,
    max_level_jump: u8,
    tris: &mut [[u32; 3]],
) -> usize {
    use std::collections::HashMap;
    let n = nodes.len();
    let dth = 2.0 * std::f64::consts::PI / nth as f64;
    let dirs: Vec<[f64; 2]> = (0..nth)
        .map(|k| {
            let th = (k as f64 + dir_offset) * dth;
            [th.cos(), th.sin()]
        })
        .collect();
    // cell size on the unit sphere
    let cell = |d: u8| (std::f64::consts::PI / 3.0).sqrt() / (1u64 << d) as f64;
    let mut inc: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (k, t) in tris.iter().enumerate() {
        for &v in t {
            inc[v as usize].push(k as u32);
        }
    }
    // interior nodes: every edge around them is shared by two triangles
    let interior: Vec<bool> = (0..n)
        .map(|i| {
            if inc[i].is_empty() {
                return false;
            }
            let mut c: HashMap<u32, u32> = HashMap::new();
            for &k in &inc[i] {
                for &v in &tris[k as usize] {
                    if v as usize != i {
                        *c.entry(v).or_insert(0) += 1;
                    }
                }
            }
            c.values().all(|&x| x == 2)
        })
        .collect();
    let eff = |i: usize, inc: &[Vec<u32>], tris: &[[u32; 3]]| -> f64 {
        if !interior[i] {
            return f64::INFINITY;
        }
        node_length(i, &inc[i], nodes, tris, &dirs) / cell(levels[i])
    };
    let mut cur: Vec<f64> = (0..n).map(|i| eff(i, &inc, tris)).collect();
    let mut total = 0;
    for _pass in 0..20 {
        let mut dir: HashMap<(u32, u32), (u32, u32)> = HashMap::with_capacity(3 * tris.len());
        for (k, t) in tris.iter().enumerate() {
            for i in 0..3 {
                dir.insert((t[i], t[(i + 1) % 3]), (k as u32, t[(i + 2) % 3]));
            }
        }
        let mut edges: Vec<(u32, u32)> = dir
            .keys()
            .copied()
            .filter(|&(u, v)| u < v)
            .filter(|&(u, v)| {
                let (Some(&(_, w)), Some(&(_, x))) = (dir.get(&(u, v)), dir.get(&(v, u))) else {
                    return false;
                };
                [u, v, w, x].iter().any(|&q| cur[q as usize] < threshold)
            })
            .collect();
        edges.sort_unstable();
        let mut flips = 0;
        for (u, v) in edges {
            let (Some(&(k1, w)), Some(&(k2, x))) = (dir.get(&(u, v)), dir.get(&(v, u))) else {
                continue;
            };
            if w == x || dir.contains_key(&(w, x)) || dir.contains_key(&(x, w)) {
                continue;
            }
            if levels[w as usize].abs_diff(levels[x as usize]) > max_level_jump {
                continue;
            }
            let (pu, pv, pw, px) =
                (&nodes[u as usize], &nodes[v as usize], &nodes[w as usize], &nodes[x as usize]);
            if crate::mesh::orientation(pw, pu, px) <= 0.0 || crate::mesh::orientation(px, pv, pw) <= 0.0 {
                continue;
            }
            let n1 = [w, u, x];
            let n2 = [x, v, w];
            let ok = |t: &[u32; 3]| {
                let a = crate::mesh::tri_angles(&nodes[t[0] as usize], &nodes[t[1] as usize], &nodes[t[2] as usize]);
                a.iter().all(|&x| x >= min_angle && x <= max_angle)
            };
            if !ok(&n1) || !ok(&n2) {
                continue;
            }
            let quad = [u, v, w, x];
            let before = quad.iter().map(|&q| cur[q as usize]).fold(f64::INFINITY, f64::min);
            if !before.is_finite() {
                continue;
            }
            // apply, evaluate, revert if not better
            let (o1, o2) = (tris[k1 as usize], tris[k2 as usize]);
            tris[k1 as usize] = n1;
            tris[k2 as usize] = n2;
            inc[u as usize].retain(|&k| k != k2);
            inc[v as usize].retain(|&k| k != k1);
            inc[w as usize].push(k2);
            inc[x as usize].push(k1);
            let after_v: Vec<f64> = quad.iter().map(|&q| eff(q as usize, &inc, tris)).collect();
            let after = after_v.iter().cloned().fold(f64::INFINITY, f64::min);
            if after > before * 1.001 {
                for (q, a) in quad.iter().zip(after_v) {
                    cur[*q as usize] = a;
                }
                dir.remove(&(u, v));
                dir.remove(&(v, u));
                for (k, t) in [(k1, n1), (k2, n2)] {
                    for i in 0..3 {
                        dir.insert((t[i], t[(i + 1) % 3]), (k, t[(i + 2) % 3]));
                    }
                }
                flips += 1;
            } else {
                tris[k1 as usize] = o1;
                tris[k2 as usize] = o2;
                inc[w as usize].retain(|&k| k != k2);
                inc[x as usize].retain(|&k| k != k1);
                inc[u as usize].push(k2);
                inc[v as usize].push(k1);
            }
        }
        total += flips;
        if flips == 0 {
            break;
        }
    }
    total
}

/// Statistics of the explicit time step for one nside.
#[derive(Debug, Serialize, Clone)]
pub struct LevelCfl {
    /// nside
    pub nside: u32,
    /// interior nodes
    pub n: usize,
    /// smallest explicit step (s)
    pub dt_min_s: f64,
    /// 1 % quantile of the explicit step (s)
    pub dt_p01_s: f64,
    /// median explicit step (s)
    pub dt_median_s: f64,
    /// smallest CFL length divided by the cell size
    pub efficiency_min: f64,
    /// median CFL length divided by the cell size
    pub efficiency_median: f64,
}

/// Explicit-step summary of a grid.
#[derive(Debug, Serialize, Clone)]
pub struct CflReport {
    /// frequency of the fastest component (`SPECTRUM%FREQ1`, Hz)
    pub freq_hz: f64,
    /// smallest explicit step over the interior nodes (s): the step of the
    /// explicit schemes before sub-cycling
    pub dt_min_s: f64,
    /// (lon, lat, nside, depth m) of the node that sets `dt_min_s`
    pub worst_node: (f64, f64, u32, f64),
    /// interior nodes whose explicit step is below `TIMESTEPS%DTXY`
    pub n_below_dtxy: usize,
    /// sub-cycles of the fastest component for a step of `TIMESTEPS%DTXY`
    pub subcycles_at_dtxy: u32,
    /// per nside, from nside_min to nside_max
    pub per_nside: Vec<LevelCfl>,
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// Explicit step per node (s, +inf on boundary nodes) and the report.
#[allow(clippy::too_many_arguments)]
pub fn explicit_steps(
    nodes: &[[f64; 3]],
    tris: &[[u32; 3]],
    level: &[u8],
    depth: &[f64],
    lon: &[f64],
    lat: &[f64],
    freq1: f64,
    nth: u32,
    dir_offset: f64,
    dtxy: f64,
    levels: std::ops::RangeInclusive<u8>,
) -> (Vec<f64>, Vec<f64>, CflReport) {
    let (len, interior) = cfl_length(nodes, tris, nth, dir_offset);
    let dt: Vec<f64> = (0..nodes.len())
        .map(|i| if interior[i] { len[i] / group_speed(freq1, depth[i]) } else { f64::INFINITY })
        .collect();
    let cell = |d: u8| R_EARTH * (std::f64::consts::PI / 3.0).sqrt() / (1u64 << d) as f64;
    let mut worst = (f64::INFINITY, 0usize);
    for i in 0..dt.len() {
        if dt[i] < worst.0 {
            worst = (dt[i], i);
        }
    }
    let mut per = Vec::new();
    for d in levels {
        let idx: Vec<usize> = (0..dt.len()).filter(|&i| interior[i] && level[i] == d).collect();
        let mut s: Vec<f64> = idx.iter().map(|&i| dt[i]).collect();
        let mut e: Vec<f64> = idx.iter().map(|&i| len[i] / cell(d)).collect();
        per.push(LevelCfl {
            nside: 1 << d,
            n: idx.len(),
            dt_min_s: quantile(&mut s, 0.0),
            dt_p01_s: quantile(&mut s, 0.01),
            dt_median_s: quantile(&mut s, 0.5),
            efficiency_min: quantile(&mut e, 0.0),
            efficiency_median: quantile(&mut e, 0.5),
        });
    }
    let w = worst.1;
    let report = CflReport {
        freq_hz: freq1,
        dt_min_s: worst.0,
        worst_node: (lon[w], lat[w], 1u32 << level[w], depth[w]),
        n_below_dtxy: dt.iter().filter(|&&x| x < dtxy).count(),
        subcycles_at_dtxy: (dtxy / worst.0).ceil() as u32,
        per_nside: per,
    };
    (dt, len, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_speed_limits() {
        // deep water: g / (4 pi f)
        let f = 0.0373;
        let deep = G / (4.0 * std::f64::consts::PI * f);
        assert!((group_speed(f, 5000.0) - deep).abs() / deep < 1e-3);
        // shallow water: sqrt(g h)
        let h = 2.0;
        assert!((group_speed(f, h) - (G * h).sqrt()).abs() / (G * h).sqrt() < 0.01);
    }
}
