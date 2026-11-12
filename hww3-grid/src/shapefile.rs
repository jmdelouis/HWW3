//! Minimal ESRI shapefile (.shp) reader for coastlines in geographic
//! coordinates: PolyLine / Polygon records (types 3, 5 and their Z/M
//! variants 13, 15, 23, 25). Each part is returned as a chain of
//! (lon, lat) points in degrees. Attributes (.dbf) are ignored.

use anyhow::{bail, Result};

fn be_i32(b: &[u8], i: usize) -> i32 {
    i32::from_be_bytes(b[i..i + 4].try_into().unwrap())
}
fn le_i32(b: &[u8], i: usize) -> i32 {
    i32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}
fn le_f64(b: &[u8], i: usize) -> f64 {
    f64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

/// Read all parts whose record bounding box intersects the given box
/// (degrees, lon in (-180, 180]).
pub fn read(path: &str, lon_min: f64, lon_max: f64, lat_min: f64, lat_max: f64) -> Result<Vec<Vec<(f64, f64)>>> {
    let buf = std::fs::read(path)?;
    if buf.len() < 100 || be_i32(&buf, 0) != 9994 {
        bail!("{path} is not a shapefile");
    }
    let mut off = 100usize;
    let mut out = Vec::new();
    while off + 8 <= buf.len() {
        let len = 2 * be_i32(&buf, off + 4) as usize;
        let c = off + 8;
        off = c + len;
        if off > buf.len() {
            bail!("truncated shapefile");
        }
        if len < 4 {
            continue;
        }
        let st = le_i32(&buf, c);
        if ![3, 5, 13, 15, 23, 25].contains(&st) {
            continue; // null shape or unsupported type
        }
        let (xmin, ymin, xmax, ymax) = (le_f64(&buf, c + 4), le_f64(&buf, c + 12), le_f64(&buf, c + 20), le_f64(&buf, c + 28));
        if ymax < lat_min || ymin > lat_max || xmax < lon_min || xmin > lon_max {
            continue;
        }
        let nparts = le_i32(&buf, c + 36) as usize;
        let npts = le_i32(&buf, c + 40) as usize;
        let parts = c + 44;
        let pts = parts + 4 * nparts;
        if pts + 16 * npts > off {
            bail!("corrupt shapefile record");
        }
        for p in 0..nparts {
            let s = le_i32(&buf, parts + 4 * p) as usize;
            let e = if p + 1 < nparts { le_i32(&buf, parts + 4 * (p + 1)) as usize } else { npts };
            let chain: Vec<(f64, f64)> =
                (s..e).map(|k| (le_f64(&buf, pts + 16 * k), le_f64(&buf, pts + 16 * k + 8))).collect();
            if chain.len() >= 2 {
                out.push(chain);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polyline_round_trip() {
        // one PolyLine record with two parts
        let pts: [(f64, f64); 5] = [(-5.0, 48.0), (-4.0, 48.5), (-3.0, 48.7), (10.0, 60.0), (11.0, 61.0)];
        let parts = [0i32, 3];
        let mut rec = Vec::new();
        rec.extend_from_slice(&3i32.to_le_bytes());
        for v in [-5.0f64, 48.0, 11.0, 61.0] {
            rec.extend_from_slice(&v.to_le_bytes());
        }
        rec.extend_from_slice(&(parts.len() as i32).to_le_bytes());
        rec.extend_from_slice(&(pts.len() as i32).to_le_bytes());
        for p in parts {
            rec.extend_from_slice(&p.to_le_bytes());
        }
        for (x, y) in pts {
            rec.extend_from_slice(&x.to_le_bytes());
            rec.extend_from_slice(&y.to_le_bytes());
        }
        let mut f = vec![0u8; 100];
        f[0..4].copy_from_slice(&9994i32.to_be_bytes());
        f.extend_from_slice(&1i32.to_be_bytes());
        f.extend_from_slice(&((rec.len() / 2) as i32).to_be_bytes());
        f.extend_from_slice(&rec);
        let path = std::env::temp_dir().join("hww3_test.shp");
        std::fs::write(&path, f).unwrap();
        let chains = read(path.to_str().unwrap(), -10.0, 0.0, 40.0, 50.0).unwrap();
        assert_eq!(chains.len(), 2);
        assert_eq!(chains[0], pts[0..3].to_vec());
        assert_eq!(chains[1], pts[3..5].to_vec());
        assert!(read(path.to_str().unwrap(), 20.0, 30.0, 0.0, 10.0).unwrap().is_empty());
    }
}
