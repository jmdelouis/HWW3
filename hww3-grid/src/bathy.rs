//! Bathymetry on a regular geodetic lon/lat grid stored as raw little-endian
//! f32 (elevation in metres, positive up, row-major with longitude fastest),
//! described by a JSON meta file.

use crate::geo::{lonlat_to_vec, Earth};
use crate::tree::uniq;
use anyhow::{bail, Result};
use cdshealpix::nested;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct BathyMeta {
    pub nlon: usize,
    pub nlat: usize,
    /// centre of the first sample, degrees
    pub lon0: f64,
    pub lat0: f64,
    pub dlon: f64,
    pub dlat: f64,
    /// raw f32 file, relative to the meta file
    pub file: String,
}

pub struct Bathy {
    pub meta: BathyMeta,
    pub z: Vec<f32>,
}

impl Bathy {
    pub fn load(meta_path: &str) -> Result<Bathy> {
        let meta: BathyMeta = serde_json::from_str(&std::fs::read_to_string(meta_path)?)?;
        let base = Path::new(meta_path).parent().unwrap_or(Path::new("."));
        let bytes = std::fs::read(base.join(&meta.file))?;
        if bytes.len() != 4 * meta.nlon * meta.nlat {
            bail!("bathymetry size {} != 4*{}*{}", bytes.len(), meta.nlon, meta.nlat);
        }
        let z = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(Bathy { meta, z })
    }

    pub fn lon(&self, i: usize) -> f64 {
        self.meta.lon0 + i as f64 * self.meta.dlon
    }
    pub fn lat(&self, j: usize) -> f64 {
        self.meta.lat0 + j as f64 * self.meta.dlat
    }

    /// nearest sample, None outside the grid
    pub fn nearest(&self, lon: f64, lat: f64) -> Option<f32> {
        let fi = ((lon - self.meta.lon0) / self.meta.dlon).round();
        let fj = ((lat - self.meta.lat0) / self.meta.dlat).round();
        if fi < 0.0 || fj < 0.0 || fi >= self.meta.nlon as f64 || fj >= self.meta.nlat as f64 {
            return None;
        }
        Some(self.z[fj as usize * self.meta.nlon + fi as usize])
    }

    /// Coast points (unit vectors on the HEALPix sphere) at the zero crossings
    /// between adjacent samples.
    pub fn coast_points(&self, earth: &Earth) -> Vec<[f64; 3]> {
        let (nx, ny) = (self.meta.nlon, self.meta.nlat);
        (0..ny)
            .into_par_iter()
            .flat_map_iter(|j| {
                let mut out = Vec::new();
                for i in 0..nx {
                    let z0 = self.z[j * nx + i];
                    let mut push = |lon: f64, lat: f64| {
                        out.push(lonlat_to_vec(lon.to_radians(), earth.to_authalic(lat.to_radians())))
                    };
                    if i + 1 < nx {
                        let z1 = self.z[j * nx + i + 1];
                        if (z0 < 0.0) != (z1 < 0.0) {
                            let t = z0 as f64 / (z0 as f64 - z1 as f64);
                            push(self.lon(i) + t * self.meta.dlon, self.lat(j));
                        }
                    }
                    if j + 1 < ny {
                        let z1 = self.z[(j + 1) * nx + i];
                        if (z0 < 0.0) != (z1 < 0.0) {
                            let t = z0 as f64 / (z0 as f64 - z1 as f64);
                            push(self.lon(i), self.lat(j) + t * self.meta.dlat);
                        }
                    }
                }
                out
            })
            .collect()
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct CellBathy {
    pub n: u32,
    pub n_wet: u32,
    /// mean elevation over all samples (m, positive up)
    pub mean_elev: f64,
    /// mean depth over wet samples (m, positive down)
    pub mean_wet_depth: f64,
}

/// Average the bathymetry samples falling in each leaf. Leaves without any
/// sample take the nearest sample at their centre.
pub fn cell_bathy(b: &Bathy, earth: &Earth, leaves: &[(u8, u64)], dmin: u8, dmax: u8) -> Vec<CellBathy> {
    let index: HashMap<u64, u32> =
        leaves.iter().enumerate().map(|(i, &(d, h))| (uniq(d, h), i as u32)).collect();
    let layer = nested::get(dmax);
    let nx = b.meta.nlon;
    let chunk = 64usize;
    let nchunks = b.meta.nlat.div_ceil(chunk);
    let mut acc = vec![(0u32, 0u32, 0f64, 0f64); leaves.len()];
    // process bands of rows in parallel, accumulate serially
    let bands: Vec<usize> = (0..nchunks).collect();
    for group in bands.chunks(rayon::current_num_threads().max(1) * 4) {
        let parts: Vec<Vec<(u32, f32)>> = group
            .par_iter()
            .map(|&c| {
                let mut out = Vec::new();
                for j in c * chunk..((c + 1) * chunk).min(b.meta.nlat) {
                    let lat = b.lat(j);
                    if lat.abs() > 90.0 {
                        continue;
                    }
                    let beta = earth.to_authalic(lat.to_radians());
                    for i in 0..nx {
                        let lon = b.lon(i).to_radians();
                        let h = layer.hash(lon, beta);
                        for k in 0..=(dmax - dmin) {
                            let u = uniq(dmax - k, h >> (2 * k as u32));
                            if let Some(&idx) = index.get(&u) {
                                out.push((idx, b.z[j * nx + i]));
                                break;
                            }
                        }
                    }
                }
                out
            })
            .collect();
        for part in parts {
            for (idx, z) in part {
                let a = &mut acc[idx as usize];
                a.0 += 1;
                a.2 += z as f64;
                if z < 0.0 {
                    a.1 += 1;
                    a.3 -= z as f64;
                }
            }
        }
    }
    leaves
        .par_iter()
        .zip(acc.par_iter())
        .map(|(&(d, h), &(n, nw, s, sw))| {
            if n > 0 {
                CellBathy {
                    n,
                    n_wet: nw,
                    mean_elev: s / n as f64,
                    mean_wet_depth: if nw > 0 { sw / nw as f64 } else { 0.0 },
                }
            } else {
                let (lo, la) = nested::center(d, h);
                let lon = {
                    let mut l = lo.to_degrees();
                    if l > 180.0 {
                        l -= 360.0
                    }
                    l
                };
                let lat = earth.to_geodetic(la).to_degrees();
                match b.nearest(lon, lat) {
                    Some(z) => CellBathy {
                        n: 0,
                        n_wet: (z < 0.0) as u32,
                        mean_elev: z as f64,
                        mean_wet_depth: if z < 0.0 { -z as f64 } else { 0.0 },
                    },
                    None => CellBathy::default(),
                }
            }
        })
        .collect()
}
