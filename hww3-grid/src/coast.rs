//! Nearest-coast-point queries, bucketed on HEALPix cells.

use crate::geo::{angle, vec_to_lonlat};
use cdshealpix::nested;
use std::collections::{HashMap, HashSet};

/// Coastline points bucketed by HEALPix cell, for nearest-distance queries.
///
/// The query visits the bucket containing the point, then rings of
/// neighbouring buckets, and stops when the ring is farther than the best
/// distance found or than the search radius.
pub struct CoastIndex {
    depth: u8,
    buckets: HashMap<u64, Vec<[f64; 3]>>,
    /// lower bound of the angular width of a bucket cell
    width: f64,
    /// number of points kept after thinning
    pub n_points: usize,
}

impl CoastIndex {
    /// `points` are unit vectors on the HEALPix (authalic) sphere.
    /// Points are thinned to one per cell of depth `thin_depth`.
    pub fn new(points: &[[f64; 3]], bucket_depth: u8, thin_depth: u8) -> Self {
        let thin = nested::get(thin_depth);
        let mut seen = HashSet::new();
        let mut buckets: HashMap<u64, Vec<[f64; 3]>> = HashMap::new();
        let mut n = 0;
        for p in points {
            let (lon, lat) = vec_to_lonlat(p);
            let h = thin.hash(lon, lat);
            if !seen.insert(h) {
                continue;
            }
            let b = h >> (2 * (thin_depth - bucket_depth) as u32);
            buckets.entry(b).or_default().push(*p);
            n += 1;
        }
        let omega = 4.0 * std::f64::consts::PI / (12.0 * 4f64.powi(bucket_depth as i32));
        CoastIndex { depth: bucket_depth, buckets, width: 0.5 * omega.sqrt(), n_points: n }
    }

    /// Angular distance (rad) from `p` to the nearest coast point, capped at
    /// `max_ang`.
    pub fn nearest(&self, p: &[f64; 3], max_ang: f64) -> f64 {
        if self.buckets.is_empty() {
            return max_ang;
        }
        let (lon, lat) = vec_to_lonlat(p);
        let start = nested::hash(self.depth, lon, lat);
        let mut visited: HashSet<u64> = HashSet::new();
        visited.insert(start);
        let mut frontier = vec![start];
        let mut best = max_ang;
        let mut k = 0usize;
        loop {
            for c in &frontier {
                if let Some(v) = self.buckets.get(c) {
                    for q in v {
                        let a = angle(p, q);
                        if a < best {
                            best = a;
                        }
                    }
                }
            }
            let bound = k as f64 * self.width;
            if best <= bound || bound >= max_ang || frontier.is_empty() {
                break;
            }
            let mut next = Vec::new();
            for c in &frontier {
                for n in nested::neighbours(self.depth, *c, false).values_vec() {
                    if visited.insert(n) {
                        next.push(n);
                    }
                }
            }
            frontier = next;
            k += 1;
        }
        best
    }
}
