//! Reader of the GSHHG native binary format (`gshhs_[cflhi].b`, version 2.x).
//!
//! Each polygon: a header of 11 big-endian i32
//! (id, n, flag, west, east, south, north, area, area_full, container, ancestor)
//! followed by `n` pairs of big-endian i32 (lon, lat) in micro-degrees.
//! `flag & 0xff` is the level (1 land, 2 lake, 3 island in lake, 4 pond,
//! 5 Antarctic ice front, 6 Antarctic grounding line).

use anyhow::{bail, Result};

/// One GSHHG polygon.
pub struct Polygon {
    /// GSHHG level (1 land, 2 lake, ...)
    pub level: u8,
    /// (lon, lat) in degrees, lon in (-180, 180]
    pub points: Vec<(f64, f64)>,
}

fn be_i32(b: &[u8], i: usize) -> i32 {
    i32::from_be_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

/// Read polygons of the selected levels whose bounding box intersects
/// [lon_min, lon_max] x [lat_min, lat_max] (degrees, lon in (-180, 180]).
pub fn read(
    path: &str,
    levels: &[u8],
    lon_min: f64,
    lon_max: f64,
    lat_min: f64,
    lat_max: f64,
) -> Result<Vec<Polygon>> {
    let buf = std::fs::read(path)?;
    let mut off = 0usize;
    let mut out = Vec::new();
    while off + 44 <= buf.len() {
        let n = be_i32(&buf, off + 4);
        let flag = be_i32(&buf, off + 8);
        let west = be_i32(&buf, off + 12) as f64 * 1e-6;
        let east = be_i32(&buf, off + 16) as f64 * 1e-6;
        let south = be_i32(&buf, off + 20) as f64 * 1e-6;
        let north = be_i32(&buf, off + 24) as f64 * 1e-6;
        if n < 0 {
            bail!("corrupt GSHHG file at offset {off}");
        }
        let n = n as usize;
        let data = off + 44;
        let end = data + 8 * n;
        if end > buf.len() {
            bail!("truncated GSHHG file");
        }
        off = end;
        let level = (flag & 0xff) as u8;
        if !levels.contains(&level) {
            continue;
        }
        if north < lat_min || south > lat_max {
            continue;
        }
        // bbox longitude in [0, 360) possibly with east > 360; test overlap
        // with the domain shifted by multiples of 360
        let overlaps = [-360.0, 0.0, 360.0]
            .iter()
            .any(|s| !(east < lon_min + s || west > lon_max + s));
        if !overlaps {
            continue;
        }
        let mut pts = Vec::with_capacity(n);
        for k in 0..n {
            let mut lon = be_i32(&buf, data + 8 * k) as f64 * 1e-6;
            let lat = be_i32(&buf, data + 8 * k + 4) as f64 * 1e-6;
            while lon > 180.0 {
                lon -= 360.0;
            }
            while lon <= -180.0 {
                lon += 360.0;
            }
            pts.push((lon, lat));
        }
        out.push(Polygon { level, points: pts });
    }
    Ok(out)
}
