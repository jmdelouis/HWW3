//! HWW3 grid as written by `hww3-grid` (`raw/*` arrays and `ww3_grid.nml`).

use anyhow::{Context, Result};
use std::path::Path;

/// Spectral discretisation read from `ww3_grid.nml`.
#[derive(Clone, Debug)]
pub struct Spectrum {
    pub nk: usize,
    pub nth: usize,
    pub freq1: f64,
    pub xfr: f64,
    pub thoff: f64,
    pub dmin: f64,
}

/// Nodes (degrees, depth in m, positive down), triangles and open-boundary nodes.
#[derive(Clone, Debug)]
pub struct Grid {
    pub lon: Vec<f64>,
    pub lat: Vec<f64>,
    pub depth: Vec<f64>,
    pub tri: Vec<[u32; 3]>,
    /// 0-based open-boundary node indices (WW3 `IOBP = 2`, `MAPSTA = 2`).
    pub obc: Vec<u32>,
    pub spec: Spectrum,
}

fn read_f64(p: &Path) -> Result<Vec<f64>> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
}

fn read_u32(p: &Path) -> Result<Vec<u32>> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// First `key = value` of a namelist text (as `explicit_cost.py`).
fn nml_value(txt: &str, key: &str) -> Option<f64> {
    let mut from = 0;
    while let Some(pos) = txt[from..].find(key) {
        let rest = txt[from + pos + key.len()..].trim_start();
        if let Some(rest) = rest.strip_prefix('=') {
            let rest = rest.trim_start();
            let end = rest
                .find(|c: char| !(c.is_ascii_digit() || "+-.eE".contains(c)))
                .unwrap_or(rest.len());
            if let Ok(v) = rest[..end].parse() {
                return Some(v);
            }
        }
        from += pos + key.len();
    }
    None
}

impl Grid {
    /// Read `<dir>/raw/{lon,lat,depth}.f64`, `tri.u32`, `obc.u32` and `<dir>/ww3_grid.nml`.
    pub fn read(dir: &Path) -> Result<Grid> {
        let raw = dir.join("raw");
        let lon = read_f64(&raw.join("lon.f64"))?;
        let lat = read_f64(&raw.join("lat.f64"))?;
        let depth = read_f64(&raw.join("depth.f64"))?;
        let t = read_u32(&raw.join("tri.u32"))?;
        let tri = t.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let obc = if raw.join("obc.u32").exists() { read_u32(&raw.join("obc.u32"))? } else { vec![] };
        let txt = std::fs::read_to_string(dir.join("ww3_grid.nml")).unwrap_or_default();
        let spec = Spectrum {
            nk: nml_value(&txt, "SPECTRUM%NK").unwrap_or(32.0) as usize,
            nth: nml_value(&txt, "SPECTRUM%NTH").unwrap_or(36.0) as usize,
            freq1: nml_value(&txt, "SPECTRUM%FREQ1").unwrap_or(0.0373),
            xfr: nml_value(&txt, "SPECTRUM%XFR").unwrap_or(1.1),
            thoff: nml_value(&txt, "SPECTRUM%THOFF").unwrap_or(0.0),
            dmin: nml_value(&txt, "GRID%DMIN").unwrap_or(1.0),
        };
        Ok(Grid { lon, lat, depth, tri, obc, spec })
    }

    /// Keep the triangles with three nodes in the box (as `scripts/subdomain.py`).
    /// The open boundary of the sub-mesh is its boundary nodes that are interior
    /// in the full mesh, plus the open-boundary nodes of the full mesh in the box.
    pub fn cut_box(&self, lon0: f64, lon1: f64, lat0: f64, lat1: f64) -> Grid {
        let n = self.lon.len();
        let inside: Vec<bool> = (0..n)
            .map(|i| self.lon[i] >= lon0 && self.lon[i] <= lon1 && self.lat[i] >= lat0 && self.lat[i] <= lat1)
            .collect();
        let kept: Vec<[u32; 3]> =
            self.tri.iter().copied().filter(|t| t.iter().all(|&v| inside[v as usize])).collect();
        let mut remap = vec![u32::MAX; n];
        let mut used = vec![];
        for t in &kept {
            for &v in t {
                if remap[v as usize] == u32::MAX {
                    remap[v as usize] = 0;
                    used.push(v);
                }
            }
        }
        used.sort_unstable();
        for (k, &v) in used.iter().enumerate() {
            remap[v as usize] = k as u32;
        }
        let tri: Vec<[u32; 3]> =
            kept.iter().map(|t| [remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]]).collect();
        let full_b = boundary_nodes(&self.tri, n);
        let sub_b = boundary_nodes(&tri, used.len());
        let mut is_obc = vec![false; n];
        for &o in &self.obc {
            is_obc[o as usize] = true;
        }
        let obc = used
            .iter()
            .enumerate()
            .filter(|&(k, &v)| (sub_b[k] && !full_b[v as usize]) || is_obc[v as usize])
            .map(|(k, _)| k as u32)
            .collect();
        let pick = |a: &[f64]| used.iter().map(|&v| a[v as usize]).collect::<Vec<f64>>();
        Grid { lon: pick(&self.lon), lat: pick(&self.lat), depth: pick(&self.depth), tri, obc, spec: self.spec.clone() }
    }
}

/// Nodes on an edge used by a single triangle.
pub fn boundary_nodes(tri: &[[u32; 3]], n: usize) -> Vec<bool> {
    use rayon::prelude::*;
    let mut e: Vec<(u32, u32)> = Vec::with_capacity(3 * tri.len());
    for t in tri {
        for q in 0..3 {
            let (a, b) = (t[q], t[(q + 1) % 3]);
            e.push((a.min(b), a.max(b)));
        }
    }
    e.par_sort_unstable();
    let mut b = vec![false; n];
    let mut i = 0;
    while i < e.len() {
        let mut j = i + 1;
        while j < e.len() && e[j] == e[i] {
            j += 1;
        }
        if j - i == 1 {
            b[e[i].0 as usize] = true;
            b[e[i].1 as usize] = true;
        }
        i = j;
    }
    b
}
