//! Topological and geometric checks of a triangle mesh on the sphere.

use crate::geo::{angle, cross, normalize};
use crate::mesh::orientation;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Default, Serialize, Clone)]
pub struct MeshReport {
    pub n_nodes: usize,
    pub n_triangles: usize,
    pub n_edges: usize,
    pub n_boundary_edges: usize,
    /// edges shared by more than two triangles
    pub n_nonmanifold_edges: usize,
    /// interior edges traversed twice in the same direction
    pub n_inconsistent_edges: usize,
    /// clockwise triangles
    pub n_negative: usize,
    /// nodes not used by any triangle
    pub n_isolated_nodes: usize,
    /// V - E + F (1 for a disc, 1 - h for a disc with h holes)
    pub euler: i64,
    pub min_angle_deg: f64,
    pub max_angle_deg: f64,
    pub n_angle_below_20: usize,
    pub n_angle_above_120: usize,
    /// histogram of minimum angles in 5 degree bins [0,5), [5,10) ... [55,60]
    pub min_angle_hist: Vec<usize>,
    /// largest level difference across an edge
    pub max_level_jump: u8,
}

impl MeshReport {
    pub fn is_valid(&self) -> bool {
        self.n_nonmanifold_edges == 0 && self.n_inconsistent_edges == 0 && self.n_negative == 0
    }
}

fn tri_angles(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> [f64; 3] {
    // angle at vertex v between the great circles to the two others
    let at = |v: &[f64; 3], p: &[f64; 3], q: &[f64; 3]| {
        let t1 = normalize(cross(&cross(v, p), v));
        let t2 = normalize(cross(&cross(v, q), v));
        angle(&t1, &t2)
    };
    [at(a, b, c), at(b, c, a), at(c, a, b)]
}

pub fn check(nodes: &[[f64; 3]], levels: &[u8], tris: &[[u32; 3]]) -> MeshReport {
    let mut r = MeshReport { n_nodes: nodes.len(), n_triangles: tris.len(), ..Default::default() };
    r.min_angle_deg = 180.0;
    r.min_angle_hist = vec![0; 12];
    // directed edge counts
    let mut edges: HashMap<(u32, u32), (u32, u32)> = HashMap::new(); // (min,max) -> (fwd, bwd)
    let mut used = vec![false; nodes.len()];
    for t in tris {
        let (a, b, c) = (&nodes[t[0] as usize], &nodes[t[1] as usize], &nodes[t[2] as usize]);
        if orientation(a, b, c) <= 0.0 {
            r.n_negative += 1;
        }
        let ang = tri_angles(a, b, c);
        let mn = ang.iter().cloned().fold(f64::MAX, f64::min).to_degrees();
        let mx = ang.iter().cloned().fold(0.0, f64::max).to_degrees();
        r.min_angle_deg = r.min_angle_deg.min(mn);
        r.max_angle_deg = r.max_angle_deg.max(mx);
        if mn < 20.0 {
            r.n_angle_below_20 += 1;
        }
        if mx > 120.0 {
            r.n_angle_above_120 += 1;
        }
        r.min_angle_hist[((mn / 5.0) as usize).min(11)] += 1;
        for k in 0..3 {
            let (i, j) = (t[k], t[(k + 1) % 3]);
            used[i as usize] = true;
            let e = edges.entry((i.min(j), i.max(j))).or_insert((0, 0));
            if i < j {
                e.0 += 1
            } else {
                e.1 += 1
            }
            let jump = levels[i as usize].abs_diff(levels[j as usize]);
            r.max_level_jump = r.max_level_jump.max(jump);
        }
    }
    r.n_edges = edges.len();
    for (f, b) in edges.values() {
        let tot = f + b;
        if tot == 1 {
            r.n_boundary_edges += 1;
        } else if tot > 2 {
            r.n_nonmanifold_edges += 1;
        } else if *f != 1 {
            r.n_inconsistent_edges += 1;
        }
    }
    r.n_isolated_nodes = used.iter().filter(|u| !**u).count();
    let v = (nodes.len() - r.n_isolated_nodes) as i64;
    r.euler = v - r.n_edges as i64 + tris.len() as i64;
    r
}
