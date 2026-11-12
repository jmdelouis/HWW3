//! Adaptive NESTED HEALPix quadtree: refinement by distance to the coast and
//! 2:1 balance between neighbouring leaves (8-neighbourhood).

use crate::coast::CoastIndex;
use crate::config::{depth_of_nside, Config};
use crate::geo::{angle, healpix_to_geodetic_deg, lonlat_to_vec, Earth};
use cdshealpix::nested;
use rayon::prelude::*;
use std::collections::HashSet;

/// NESTED uniq index of cell `h` at depth `d` (`4 * 4^d + h`).
#[inline]
pub fn uniq(d: u8, h: u64) -> u64 {
    nested::to_uniq(d, h)
}

/// Angular radius of a cell: max distance centre -> vertex (rad).
pub fn cell_radius(d: u8, h: u64) -> f64 {
    let (lo, la) = nested::center(d, h);
    let c = lonlat_to_vec(lo, la);
    nested::vertices(d, h)
        .iter()
        .map(|(vl, va)| angle(&c, &lonlat_to_vec(*vl, *va)))
        .fold(0.0, f64::max)
}

struct Refiner<'a> {
    coast: &'a CoastIndex,
    /// (max angular distance, depth), sorted by decreasing depth
    rules: Vec<(f64, u8)>,
    dmin: u8,
    dmax: u8,
    max_rule_ang: f64,
}

impl<'a> Refiner<'a> {
    fn required_depth(&self, d: u8, h: u64) -> u8 {
        let (lo, la) = nested::center(d, h);
        let c = lonlat_to_vec(lo, la);
        let r = cell_radius(d, h);
        let dist = self.coast.nearest(&c, self.max_rule_ang + r) - r;
        let mut req = self.dmin;
        for (ang, depth) in &self.rules {
            if dist < *ang && *depth > req {
                req = *depth;
            }
        }
        req
    }

    fn refine(&self, d: u8, h: u64, leaves: &mut Vec<(u8, u64)>, internal: &mut Vec<u64>) {
        if d < self.dmax && self.required_depth(d, h) > d {
            internal.push(uniq(d, h));
            for k in 0..4 {
                self.refine(d + 1, 4 * h + k, leaves, internal);
            }
        } else {
            leaves.push((d, h));
        }
    }
}

/// Leaves of the refined and balanced tree.
pub struct Tree {
    /// depth of nside_min
    pub dmin: u8,
    /// depth of nside_max
    pub dmax: u8,
    /// leaves sorted in NESTED (Morton) order at depth dmax
    pub leaves: Vec<(u8, u64)>,
    /// leaves split by the 2:1 balance
    pub n_balance_splits: usize,
}

/// Morton key of a leaf at depth dmax (first descendant).
pub fn morton(d: u8, h: u64, dmax: u8) -> u64 {
    h << (2 * (dmax - d) as u32)
}

/// Build the tree.
///
/// Roots are the cells of depth `dmin` whose centre lies in the domain plus a
/// margin of three cell widths. A cell of depth `d` is split while
/// `d < dmax` and a refinement rule requires a finer depth: the distance
/// from the cell to the coast (distance from the centre to the nearest coast
/// point minus the cell radius) is below the rule's `max_dist_km`. Then
/// leaves are split until no leaf has a neighbour (edge or corner) more
/// than one level finer (2:1 balance).
pub fn build(cfg: &Config, earth: &Earth, coast: &CoastIndex) -> Tree {
    let dmin = cfg.dmin();
    let dmax = cfg.dmax();
    let mut rules: Vec<(f64, u8)> = cfg
        .refine
        .iter()
        .map(|r| (r.max_dist_km * 1e3 / earth.radius, depth_of_nside(r.nside).unwrap()))
        .collect();
    rules.sort_by(|a, b| b.1.cmp(&a.1));
    let max_rule_ang = rules.iter().map(|r| r.0).fold(0.0, f64::max);
    let refiner = Refiner { coast, rules, dmin, dmax, max_rule_ang };

    // roots: cells of depth dmin whose centre lies in the domain + margin
    let omega = 4.0 * std::f64::consts::PI / (12.0 * 4f64.powi(dmin as i32));
    let margin = 3.0 * omega.sqrt().to_degrees();
    let nroot = 12u64 << (2 * dmin as u32);
    let roots: Vec<u64> = (0..nroot)
        .into_par_iter()
        .filter(|&h| {
            let (lo, la) = nested::center(dmin, h);
            let (lon, lat) = healpix_to_geodetic_deg(earth, lo, la);
            cfg.domain.contains_with_margin(lon, lat, margin)
        })
        .collect();

    let parts: Vec<(Vec<(u8, u64)>, Vec<u64>)> = roots
        .par_iter()
        .map(|&h| {
            let mut l = Vec::new();
            let mut i = Vec::new();
            refiner.refine(dmin, h, &mut l, &mut i);
            (l, i)
        })
        .collect();
    let mut leaves: HashSet<u64> = HashSet::new();
    let mut internal: HashSet<u64> = HashSet::new();
    for (l, i) in parts {
        leaves.extend(l.into_iter().map(|(d, h)| uniq(d, h)));
        internal.extend(i);
    }

    // 2:1 balance
    let mut n_split = 0;
    loop {
        let to_split: Vec<u64> = leaves
            .par_iter()
            .copied()
            .filter(|&u| {
                let (d, h) = nested::from_uniq(u);
                if d + 1 >= dmax {
                    return false;
                }
                nested::neighbours(d, h, false).values_vec().into_iter().any(|n| {
                    internal.contains(&uniq(d, n))
                        && (0..4).any(|k| internal.contains(&uniq(d + 1, 4 * n + k)))
                })
            })
            .collect();
        if to_split.is_empty() {
            break;
        }
        n_split += to_split.len();
        for u in to_split {
            let (d, h) = nested::from_uniq(u);
            leaves.remove(&u);
            internal.insert(u);
            for k in 0..4 {
                leaves.insert(uniq(d + 1, 4 * h + k));
            }
        }
    }

    let mut leaves: Vec<(u8, u64)> = leaves.into_iter().map(nested::from_uniq).collect();
    leaves.par_sort_unstable_by_key(|&(d, h)| morton(d, h, dmax));
    Tree { dmin, dmax, leaves, n_balance_splits: n_split }
}
