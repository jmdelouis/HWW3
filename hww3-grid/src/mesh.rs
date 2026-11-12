//! Triangulation of the cell centres from the HEALPix adjacency.
//!
//! Every leaf registers the 8 points of its boundary that can be shared with
//! a neighbour: its 4 corners and the 4 midpoints of its edges (the corners
//! of its children). With a 2:1 balanced tree, the leaves registered at a
//! point are exactly the leaves touching that point, except at quarter points
//! of a coarse edge where at most two fine leaves are registered.
//!
//! Around every point, the registered leaves sorted by angle form a polygon of
//! cell centres (the dual cell of the point). Each maximal run of consecutive
//! *active* (wet) leaves with at least 3 members is triangulated by ear
//! clipping. Two leaves sharing a boundary segment [p, q] appear consecutively
//! around both p and q, so their connecting edge belongs to the polygons of p
//! and q only: the result is a conforming triangulation.

use crate::geo::{cross, dot, lonlat_to_vec, normalize, sub, tangent_basis};
use cdshealpix::nested;
use rayon::prelude::*;
use std::collections::HashMap;

const Q: f64 = (1u64 << 36) as f64;

fn qkey(v: &[f64; 3]) -> [i64; 3] {
    [(v[0] * Q).round() as i64, (v[1] * Q).round() as i64, (v[2] * Q).round() as i64]
}

/// The 8 shareable boundary points of a leaf (corners + edge midpoints).
fn boundary_points(d: u8, h: u64) -> Vec<[f64; 3]> {
    let (clo, cla) = nested::center(d, h);
    let centre = lonlat_to_vec(clo, cla);
    let mut pts: Vec<[f64; 3]> = Vec::with_capacity(9);
    for k in 0..4 {
        for (lo, la) in nested::vertices(d + 1, 4 * h + k) {
            let v = lonlat_to_vec(lo, la);
            if pts.iter().any(|p| {
                let s = sub(p, &v);
                dot(&s, &s) < 1e-20
            }) {
                continue;
            }
            pts.push(v);
        }
    }
    // drop the cell centre (shared corner of the 4 children)
    pts.retain(|p| {
        let s = sub(p, &centre);
        dot(&s, &s) > 1e-24
    });
    pts
}

pub struct Triangulation {
    /// triangles as leaf indices, counter-clockwise seen from outside
    pub tris: Vec<[u32; 3]>,
    /// active leaves sharing a point with an `outside` leaf
    pub touches_outside: Vec<bool>,
    pub n_points: usize,
}

/// `centres`: unit vectors of the leaf centres; `active`: wet leaves;
/// `outside`: leaves outside the model domain (for open-boundary detection).
pub fn triangulate(
    leaves: &[(u8, u64)],
    centres: &[[f64; 3]],
    active: &[bool],
    outside: &[bool],
) -> Triangulation {
    // 1. boundary points of every leaf
    let per_leaf: Vec<Vec<[f64; 3]>> =
        leaves.par_iter().map(|&(d, h)| boundary_points(d, h)).collect();

    // 2. merge identical points (quantised keys with neighbour search)
    let mut keys: HashMap<[i64; 3], u32> = HashMap::new();
    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(8 * leaves.len());
    for (li, pts) in per_leaf.iter().enumerate() {
        for p in pts {
            let k = qkey(p);
            let id = if let Some(&id) = keys.get(&k) {
                id
            } else {
                let mut found = None;
                'search: for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if let Some(&id) = keys.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                                found = Some(id);
                                break 'search;
                            }
                        }
                    }
                }
                match found {
                    Some(id) => id,
                    None => {
                        let id = points.len() as u32;
                        points.push(*p);
                        keys.insert(k, id);
                        id
                    }
                }
            };
            pairs.push((id, li as u32));
        }
    }
    drop(per_leaf);
    drop(keys);
    pairs.par_sort_unstable();

    // 3. group by point
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut s = 0;
    while s < pairs.len() {
        let mut e = s + 1;
        while e < pairs.len() && pairs[e].0 == pairs[s].0 {
            e += 1;
        }
        if e - s >= 2 {
            groups.push((s, e));
        }
        s = e;
    }

    // 4. polygons and triangles
    let results: Vec<(Vec<[u32; 3]>, Vec<u32>)> = groups
        .par_iter()
        .map(|&(s, e)| {
            let p = points[pairs[s].0 as usize];
            let members: Vec<u32> = pairs[s..e].iter().map(|x| x.1).collect();
            let mut touch = Vec::new();
            if members.iter().any(|&m| outside[m as usize]) {
                touch.extend(members.iter().copied().filter(|&m| active[m as usize]));
            }
            if members.iter().filter(|&&m| active[m as usize]).count() < 3 {
                return (Vec::new(), touch);
            }
            (polygon_triangles(&p, &members, centres, active), touch)
        })
        .collect();

    let mut tris = Vec::new();
    let mut touches_outside = vec![false; leaves.len()];
    for (t, touch) in results {
        tris.extend(t);
        for m in touch {
            touches_outside[m as usize] = true;
        }
    }
    Triangulation { tris, touches_outside, n_points: points.len() }
}

fn polygon_triangles(p: &[f64; 3], members: &[u32], centres: &[[f64; 3]], active: &[bool]) -> Vec<[u32; 3]> {
    let (e, n) = tangent_basis(p);
    // gnomonic projection on the tangent plane at p
    let proj = |c: &[f64; 3]| -> [f64; 2] {
        let w = dot(c, p);
        [dot(c, &e) / w, dot(c, &n) / w]
    };
    let mut ring: Vec<(f64, u32, [f64; 2])> = members
        .iter()
        .map(|&m| {
            let xy = proj(&centres[m as usize]);
            (xy[1].atan2(xy[0]), m, xy)
        })
        .collect();
    ring.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let nr = ring.len();

    let mut out = Vec::new();
    if ring.iter().all(|r| active[r.1 as usize]) {
        let pts: Vec<[f64; 2]> = ring.iter().map(|r| r.2).collect();
        for t in ear_clip(&pts) {
            out.push([ring[t[0]].1, ring[t[1]].1, ring[t[2]].1]);
        }
        return out;
    }
    // start right after an inactive member, then collect maximal active runs
    let start = (0..nr).find(|&i| !active[ring[i].1 as usize]).unwrap();
    let mut run: Vec<usize> = Vec::new();
    for k in 1..=nr {
        let i = (start + k) % nr;
        if active[ring[i].1 as usize] {
            run.push(i);
        } else {
            if run.len() >= 3 {
                let pts: Vec<[f64; 2]> = run.iter().map(|&i| ring[i].2).collect();
                for t in ear_clip(&pts) {
                    out.push([ring[run[t[0]]].1, ring[run[t[1]]].1, ring[run[t[2]]].1]);
                }
            }
            run.clear();
        }
    }
    out
}

fn cross2(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn min_angle(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2]) -> f64 {
    let ang = |p: &[f64; 2], q: &[f64; 2], r: &[f64; 2]| {
        let u = [q[0] - p[0], q[1] - p[1]];
        let v = [r[0] - p[0], r[1] - p[1]];
        (u[0] * v[1] - u[1] * v[0]).abs().atan2(u[0] * v[0] + u[1] * v[1])
    };
    ang(a, b, c).min(ang(b, c, a)).min(ang(c, a, b))
}

fn inside(t: [&[f64; 2]; 3], q: &[f64; 2], eps: f64) -> bool {
    cross2(t[0], t[1], q) > eps && cross2(t[1], t[2], q) > eps && cross2(t[2], t[0], q) > eps
}

/// Ear clipping of a counter-clockwise simple polygon, choosing at each step
/// the ear with the largest minimum angle. Degenerate ears are skipped.
pub fn ear_clip(pts: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let n = pts.len();
    if n < 3 {
        return Vec::new();
    }
    let scale = pts
        .iter()
        .flat_map(|p| [p[0].abs(), p[1].abs()])
        .fold(0.0, f64::max)
        .max(1e-300);
    let eps = 1e-12 * scale * scale;
    let mut idx: Vec<usize> = (0..n).collect();
    let mut out = Vec::new();
    while idx.len() > 3 {
        let m = idx.len();
        let mut best: Option<(usize, f64)> = None;
        for i in 0..m {
            let (a, b, c) = (idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]);
            if cross2(&pts[a], &pts[b], &pts[c]) <= eps {
                continue;
            }
            let tri = [&pts[a], &pts[b], &pts[c]];
            if idx.iter().any(|&j| j != a && j != b && j != c && inside(tri, &pts[j], -eps)) {
                continue;
            }
            let q = min_angle(&pts[a], &pts[b], &pts[c]);
            if best.map_or(true, |(_, bq)| q > bq) {
                best = Some((i, q));
            }
        }
        match best {
            Some((i, _)) => {
                let m = idx.len();
                out.push([idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]]);
                idx.remove(i);
            }
            None => break,
        }
    }
    if idx.len() == 3 && cross2(&pts[idx[0]], &pts[idx[1]], &pts[idx[2]]) > eps {
        out.push([idx[0], idx[1], idx[2]]);
    }
    out
}

/// Signed orientation of a spherical triangle (> 0: counter-clockwise seen
/// from outside the sphere).
pub fn orientation(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    let n = cross(&sub(b, a), &sub(c, a));
    let m = normalize([a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]]);
    dot(&n, &m)
}

/// Number of disjoint triangle fans around every vertex (0 for unused vertices).
/// A valid WW3 mesh has at most one fan per vertex: two fans touching at a single
/// vertex ("pinch" or bow-tie node) break the boundary walk of `w3triamd`.
pub fn fan_counts(n: usize, tris: &[[u32; 3]]) -> Vec<u32> {
    let mut vt: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (k, t) in tris.iter().enumerate() {
        for &v in t {
            vt[v as usize].push(k as u32);
        }
    }
    vt.iter().enumerate().map(|(v, ts)| fans(v as u32, ts, tris).1).collect()
}

/// Fan label of each triangle around `v` (same order as `ts`) and number of fans.
/// Two triangles around `v` are in the same fan when they share an edge through `v`.
fn fans(v: u32, ts: &[u32], tris: &[[u32; 3]]) -> (Vec<u32>, u32) {
    let m = ts.len();
    let mut parent: Vec<usize> = (0..m).collect();
    fn root(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    let others = |k: u32| -> [u32; 2] {
        let t = tris[k as usize];
        let o: Vec<u32> = t.iter().cloned().filter(|&x| x != v).collect();
        [o[0], o[1]]
    };
    let o: Vec<[u32; 2]> = ts.iter().map(|&k| others(k)).collect();
    for a in 0..m {
        for b in (a + 1)..m {
            if o[a].iter().any(|x| o[b].contains(x)) {
                let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }
    let mut label = vec![u32::MAX; m];
    let mut ids: Vec<usize> = Vec::new();
    for a in 0..m {
        let r = root(&mut parent, a);
        let id = match ids.iter().position(|&x| x == r) {
            Some(p) => p,
            None => {
                ids.push(r);
                ids.len() - 1
            }
        };
        label[a] = id as u32;
    }
    (label, ids.len() as u32)
}

/// Remove pinch nodes: at every vertex with several triangle fans, keep the largest
/// fan and drop the triangles of the others; repeat until every vertex has at most one
/// fan. Returns the number of triangles removed.
pub fn remove_pinches(n: usize, tris: &mut Vec<[u32; 3]>) -> usize {
    let mut removed = 0;
    loop {
        let mut vt: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (k, t) in tris.iter().enumerate() {
            for &v in t {
                vt[v as usize].push(k as u32);
            }
        }
        let mut drop = vec![false; tris.len()];
        let mut any = false;
        for (v, ts) in vt.iter().enumerate() {
            let live: Vec<u32> = ts.iter().cloned().filter(|&k| !drop[k as usize]).collect();
            if live.len() < 2 {
                continue;
            }
            let (label, nf) = fans(v as u32, &live, tris);
            if nf < 2 {
                continue;
            }
            let mut size = vec![0usize; nf as usize];
            for &l in &label {
                size[l as usize] += 1;
            }
            let keep = (0..nf as usize).max_by_key(|&i| size[i]).unwrap() as u32;
            for (a, &k) in live.iter().enumerate() {
                if label[a] != keep {
                    drop[k as usize] = true;
                    any = true;
                }
            }
        }
        if !any {
            return removed;
        }
        let before = tris.len();
        let mut k = 0;
        tris.retain(|_| {
            let keep = !drop[k];
            k += 1;
            keep
        });
        removed += before - tris.len();
    }
}
