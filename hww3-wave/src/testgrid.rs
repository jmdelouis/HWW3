//! Small synthetic grid shared by the unit tests.

use crate::grid::{Grid, Spectrum};

/// Three frequencies from 0.0373 Hz and twelve 30° directions.
pub(crate) fn spectrum() -> Spectrum {
    Spectrum { nk: 3, nth: 12, freq1: 0.0373, xfr: 1.1, thoff: 0.0, dmin: 1.0 }
}

/// Lattice spacing (degrees).
pub(crate) const STEP: f64 = 0.05;

/// `nx × ny` nodes on a 0.05° lattice from (10° W, 45° N), node `j nx + i`
/// at column `i` and row `j`. Each cell is cut into two triangles along
/// alternating diagonals, and every third triangle is listed clockwise (WW3
/// and [`crate::geom::Geometry::new`] reorient them). The depth is `d0 + dd i`
/// metres. With `open_west`, the western column is the open boundary.
pub(crate) fn lattice(nx: usize, ny: usize, d0: f64, dd: f64, open_west: bool) -> Grid {
    let id = |i: usize, j: usize| (j * nx + i) as u32;
    let mut lon = Vec::with_capacity(nx * ny);
    let mut lat = Vec::with_capacity(nx * ny);
    let mut depth = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            lon.push(-10.0 + STEP * i as f64);
            lat.push(45.0 + STEP * j as f64);
            depth.push(d0 + dd * i as f64);
        }
    }
    let mut tri = Vec::with_capacity(2 * (nx - 1) * (ny - 1));
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
            if (i + j) % 2 == 0 {
                tri.push([a, b, c]);
                tri.push([a, c, d]);
            } else {
                tri.push([a, b, d]);
                tri.push([b, c, d]);
            }
        }
    }
    for (n, t) in tri.iter_mut().enumerate() {
        if n % 3 == 1 {
            t.swap(1, 2);
        }
    }
    let obc = if open_west { (0..ny).map(|j| id(0, j)).collect() } else { vec![] };
    Grid { lon, lat, depth, tri, obc, spec: spectrum() }
}

/// Deterministic pseudo-random numbers in [0, 1) (64-bit LCG).
pub(crate) fn uniform(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect()
}
