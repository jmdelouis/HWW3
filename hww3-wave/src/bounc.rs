//! `ww3_bounc` (the WW3 preprocessor of open-boundary spectra) in
//! `WRITE` mode, in single precision as WW3 computes it.
//!
//! `ww3_bounc` reads one spectrum file per station (`efth(time, station,
//! frequency, direction)`, as `scripts/boundary_spectra.py` writes them on
//! the spectral grid of the model, so that no spectral conversion
//! `W3CSPC` is needed), divides the spectra by 2 pi, and for each
//! open-boundary node (`MAPSTA = 2`, in sea-point order) takes the two
//! nearest stations by the haversine distance (`DIST_HAVERSINE`, in
//! degrees) and, with `INTERP = 2`, linear weights from the projection of
//! the node on the segment between them in longitude and latitude. The
//! boundary file (`nest.ww3`) holds the station spectra and these indices
//! and weights; the model interpolates at each step.
//!
//! The spectra come from the netCDF files through `scripts/bounc_nc.py`,
//! read by [`RawSpectra`].

use crate::ww3io::Time;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

const PI: f32 = std::f32::consts::PI;

/// `DIST_HAVERSINE` of `W3SERVMD`: the great-circle distance in degrees,
/// in single precision.
pub fn dist_haversine(lon1: f32, lat1: f32, lon2: f32, lat2: f32) -> f32 {
    let dera = PI / 180.0;
    let rade = 180.0 / PI;
    let dlat = (lat2 - lat1) * dera;
    let dlon = (lon2 - lon1) * dera;
    let s1 = (dlat / 2.0).sin();
    let s2 = (dlon / 2.0).sin();
    let a = s1 * s1 + (lat1 * dera).cos() * (lat2 * dera).cos() * (s2 * s2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).max(0.0).sqrt());
    rade * c
}

/// The indices (1-based, as in the file) and weights of `ww3_bounc` for
/// boundary nodes at `(xbpo, ybpo)` from stations at `(lons, lats)`:
/// `ipbpo[i] = [nearest, second, 1, 1]`, `rdbpo[i]` their weights
/// (`interp` = 2: linear between the two; otherwise all on the nearest).
pub fn station_weights(lons: &[f32], lats: &[f32], xbpo: &[f32], ybpo: &[f32], interp: i32) -> (Vec<[i32; 4]>, Vec<[f32; 4]>) {
    let nbo2 = lons.len();
    let mut ip = vec![[1i32; 4]; xbpo.len()];
    let mut rd = vec![[1f32, 0.0, 0.0, 0.0]; xbpo.len()];
    for k in 0..xbpo.len() {
        let none = 360.0f32 + 180.0;
        let (mut dmin, mut dmin2) = (none, none);
        for s in 0..nbo2 {
            let dist = dist_haversine(lons[s], lats[s], xbpo[k], ybpo[k]);
            if dmin == none {
                if dist < dmin {
                    ip[k][0] = s as i32 + 1;
                    dmin = dist;
                }
            } else if dist < dmin2 {
                if dist < dmin {
                    ip[k][1] = ip[k][0];
                    dmin2 = dmin;
                    ip[k][0] = s as i32 + 1;
                    dmin = dist;
                } else {
                    ip[k][1] = s as i32 + 1;
                    dmin2 = dist;
                }
            }
        }
        if interp > 1 && nbo2 > 1 {
            let (i1, i2) = (ip[k][0] as usize - 1, ip[k][1] as usize - 1);
            let mut dlon = lons[i2] - lons[i1];
            let dlat = lats[i2] - lats[i1];
            let mut dlo = xbpo[k] - lons[i1];
            if dlon > 180.0 {
                dlon -= 360.0;
            }
            if dlon < -180.0 {
                dlon += 360.0;
            }
            if dlo > 180.0 {
                dlo -= 360.0;
            }
            if dlo < -180.0 {
                dlo += 360.0;
            }
            let dist = (dlon * dlon + dlat * dlat).sqrt();
            let cos1 = (dlo * dlon + (ybpo[k] - lats[i1]) * dlat) / (dist * dist);
            // MIN(1., MAX(0., COS1)) as gfortran evaluates it (the first
            // argument unless the second is larger), which keeps the sign of
            // a zero COS1, as WW3's file has it
            let m = if 0.0 > cos1 { 0.0 } else { cos1 };
            let c = if 1.0 < m { 1.0 } else { m };
            rd[k][0] = 1.0 - c;
            rd[k][1] = c;
        }
    }
    (ip, rd)
}

/// The spectrum files of the stations in the form written by
/// `scripts/bounc_nc.py`: a directory with `meta.json` (number of
/// stations, frequencies and directions, the times of the first file, per
/// station its longitude, latitude, `_FillValue`, `scale_factor`,
/// `add_offset` and whether `efth` is stored as short integers, and the
/// frequencies and directions of the first file, all single-precision bit
/// patterns) and `efth.f32` (little endian, `[station][time][frequency]
/// [direction]`, the values as stored).
pub struct RawSpectra {
    dir: PathBuf,
    /// Number of stations `NBO2`.
    pub nst: usize,
    /// Number of frequencies `NKI`.
    pub nk: usize,
    /// Number of directions `NTHI`.
    pub nth: usize,
    /// Times of the records.
    pub times: Vec<Time>,
    /// Longitudes of the stations.
    pub lons: Vec<f32>,
    /// Latitudes of the stations.
    pub lats: Vec<f32>,
    /// Frequencies (Hz).
    pub freq: Vec<f32>,
    /// Directions (degrees, nautical "to" as stored).
    pub dirs: Vec<f32>,
    /// Per station: `_FillValue`, `scale_factor`, `add_offset`, short integers.
    pub coding: Vec<(f32, f32, f32, bool)>,
}

fn bits(v: &serde_json::Value) -> Result<f32> {
    Ok(f32::from_bits(v.as_u64().context("meta.json: expected a bit pattern")? as u32))
}

impl RawSpectra {
    /// Open a directory written by `scripts/bounc_nc.py`.
    pub fn open(dir: &Path) -> Result<RawSpectra> {
        let m: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).with_context(|| format!("read {}/meta.json", dir.display()))?)?;
        let nst = m["nst"].as_u64().context("meta.json: nst")? as usize;
        let nk = m["nk"].as_u64().context("meta.json: nk")? as usize;
        let nth = m["nth"].as_u64().context("meta.json: nth")? as usize;
        let list = |k: &str| -> Result<Vec<f32>> { m[k].as_array().with_context(|| format!("meta.json: {k}"))?.iter().map(bits).collect() };
        let mut coding = vec![];
        for c in m["coding"].as_array().context("meta.json: coding")? {
            coding.push((bits(&c[0])?, bits(&c[1])?, bits(&c[2])?, c[3].as_bool().context("meta.json: short flag")?));
        }
        let mut times = vec![];
        for t in m["times"].as_array().context("meta.json: times")? {
            times.push(Time(t[0].as_i64().context("time")? as i32, t[1].as_i64().context("time")? as i32));
        }
        let r = RawSpectra { dir: dir.to_path_buf(), nst, nk, nth, times, lons: list("lon")?, lats: list("lat")?, freq: list("freq")?, dirs: list("dir")?, coding };
        anyhow::ensure!(r.lons.len() == nst && r.lats.len() == nst && r.coding.len() == nst, "meta.json: {nst} stations but other lengths");
        anyhow::ensure!(r.freq.len() == nk && r.dirs.len() == nth, "meta.json: frequency or direction count");
        Ok(r)
    }

    /// The spectra of station `s` at every time, decoded as `ww3_bounc`
    /// does and divided by 2 pi, in the order of the model
    /// (`[time][ik nth + ith]`).
    pub fn station(&self, s: usize) -> Result<Vec<Vec<f32>>> {
        use std::io::{Read, Seek, SeekFrom};
        let (nt, nspec) = (self.times.len(), self.nk * self.nth);
        let mut f = std::fs::File::open(self.dir.join("efth.f32")).context("open efth.f32")?;
        f.seek(SeekFrom::Start((4 * nspec * nt * s) as u64))?;
        let mut b = vec![0u8; 4 * nspec * nt];
        f.read_exact(&mut b).with_context(|| format!("efth.f32: station {s}"))?;
        let (fill, factor, offset, short) = self.coding[s];
        let tpiinv = 1.0f32 / (2.0 * PI);
        let ln10 = 10.0f32.ln();
        Ok(b.chunks_exact(4 * nspec)
            .map(|rec| {
                rec.chunks_exact(4)
                    .map(|c| {
                        let mut x = f32::from_le_bytes(c.try_into().unwrap());
                        if x != fill {
                            x = if short { (x * factor * ln10).exp() - 1e-12 } else { x * factor + offset };
                        }
                        x * tpiinv
                    })
                    .collect()
            })
            .collect())
    }

    /// `SPCONV` of `ww3_bounc`: whether the spectra need the conversion
    /// `W3CSPC` to the grid of the model (`nk`, `nth`, `xfr`, `fr1`, first
    /// direction `th1`, increment `dth`), which is not ported.
    pub fn needs_conversion(&self, nk: usize, nth: usize, xfr: f32, fr1: f32, th1: f32, dth: f32) -> bool {
        let tpi = 2.0 * PI;
        let fr1i = self.freq[0];
        let xfri = ((self.freq[self.nk - 1] / self.freq[0]).ln() / (self.nk - 1) as f32).exp();
        let th1i = (2.5 * PI - (PI / 180.0) * self.dirs[0]) % tpi;
        self.nk != nk || self.nth != nth || (xfri / xfr - 1.0).abs() > 0.01 || (fr1i / fr1 - 1.0).abs() > 0.01 || (th1i - th1).abs() > 0.01 * dth
    }
}

/// Check that a node list is a list of open-boundary nodes in sea-point
/// order (`MAPSTA = 2` in increasing `ISEA`).
pub fn boundary_nodes(obc: &[u32]) -> Result<Vec<usize>> {
    let mut v: Vec<usize> = obc.iter().map(|&i| i as usize).collect();
    v.sort_unstable();
    v.dedup();
    if v.is_empty() {
        bail!("the grid has no open-boundary node");
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haversine_degrees() {
        // a quarter of the equator, and one degree of latitude
        assert!((dist_haversine(0.0, 0.0, 90.0, 0.0) - 90.0).abs() < 1e-4);
        assert!((dist_haversine(10.0, 45.0, 10.0, 46.0) - 1.0).abs() < 1e-5);
        assert_eq!(dist_haversine(-5.0, 50.0, -5.0, 50.0), 0.0);
    }

    #[test]
    fn two_nearest_stations_and_weights() {
        // stations along the equator at 0, 1, 2 degrees east
        let lons = [0.0, 1.0, 2.0];
        let lats = [0.0, 0.0, 0.0];
        let (ip, rd) = station_weights(&lons, &lats, &[1.25, -0.5, 0.75], &[0.1, 0.0, 0.0], 2);
        // node at 1.25 E: stations 2 (nearest) and 3, a quarter of the way
        assert_eq!(ip[0], [2, 3, 1, 1]);
        assert!((rd[0][0] - 0.75).abs() < 1e-6 && (rd[0][1] - 0.25).abs() < 1e-6);
        // west of the first station: all the weight on it
        assert_eq!((ip[1][0], rd[1][0], rd[1][1]), (1, 1.0, 0.0));
        // nearest first, then the second nearest found later in the list
        assert_eq!(&ip[2][..2], &[2, 1]);
        assert!((rd[2][0] - 0.75).abs() < 1e-6);
        // a node at the longitude of its nearest station and south of it,
        // the second station to the west on the same latitude: both terms of
        // COS1 are -0, and its sign is kept as WW3 keeps it
        let (_, rd) = station_weights(&[45.0, 44.0], &[70.0, 70.0], &[45.0], &[69.8], 2);
        assert_eq!((rd[0][0], rd[0][1].to_bits()), (1.0, (-0.0f32).to_bits()));
        // INTERP = 1: nearest station only
        let (ip, rd) = station_weights(&lons, &lats, &[1.25], &[0.1], 1);
        assert_eq!((ip[0][0], rd[0]), (2, [1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn boundary_nodes_sorted() {
        assert_eq!(boundary_nodes(&[7, 3, 5, 3]).unwrap(), vec![3, 5, 7]);
        assert!(boundary_nodes(&[]).is_err());
    }
}
