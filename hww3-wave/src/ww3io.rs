//! Readers for the WW3 binary files of a run: restart (`W3IORS`), forcing
//! fields written by `ww3_prnc` (`W3FLDO` / `W3FLDG`) and open-boundary
//! spectra written by `ww3_bounc` (`W3IOBC`).
//!
//! The HWW3 build of WW3 writes big-endian files. Forcing and boundary
//! files are Fortran sequential unformatted files (4-byte record markers);
//! the restart is a stream file of fixed-length records of `4 NSPEC` bytes.

use anyhow::{anyhow, bail, Context, Result};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// A WW3 time `(yyyymmdd, hhmmss)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Time(pub i32, pub i32);

impl Time {
    /// Seconds since 1970-01-01 00:00 (proleptic Gregorian), as `DSEC21` differences need.
    pub fn secs(&self) -> i64 {
        let (d, s) = (self.0, self.1);
        let (y, m, day) = ((d / 10000) as i64, ((d / 100) % 100) as i64, (d % 100) as i64);
        // days from civil (H. Hinnant)
        let y2 = if m <= 2 { y - 1 } else { y };
        let era = if y2 >= 0 { y2 } else { y2 - 399 } / 400;
        let yoe = y2 - era * 400;
        let mp = (m + 9) % 12;
        let doy = (153 * mp + 2) / 5 + day - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        let days = era * 146097 + doe - 719468;
        let (hh, mm, ss) = ((s / 10000) as i64, ((s / 100) % 100) as i64, (s % 100) as i64);
        days * 86400 + hh * 3600 + mm * 60 + ss
    }

    pub fn from_secs(t: i64) -> Time {
        let days = t.div_euclid(86400);
        let sod = t.rem_euclid(86400);
        let z = days + 719468;
        let era = if z >= 0 { z } else { z - 146096 } / 146097;
        let doe = z - era * 146097;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };
        Time((y * 10000 + m * 100 + d) as i32, ((sod / 3600) * 10000 + ((sod / 60) % 60) * 100 + sod % 60) as i32)
    }

    pub fn parse(s: &str) -> Result<Time> {
        let s: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
        if s.len() < 8 {
            bail!("time {s}: expected yyyymmdd[hhmmss]");
        }
        let d: i32 = s[..8].parse()?;
        let t: i32 = if s.len() >= 14 { s[8..14].parse()? } else { 0 };
        Ok(Time(d, t))
    }
}

impl std::fmt::Display for Time {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:08} {:06}", self.0, self.1)
    }
}

fn be_i32(b: &[u8]) -> i32 {
    i32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

fn be_f32(b: &[u8]) -> f32 {
    f32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

pub fn be_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(be_f32).collect()
}

pub fn be_i32s(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4).map(be_i32).collect()
}

/// Fortran sequential unformatted file with big-endian 4-byte record markers.
pub struct SeqFile {
    r: BufReader<File>,
    name: String,
}

impl SeqFile {
    pub fn open(p: &Path) -> Result<SeqFile> {
        let f = File::open(p).with_context(|| format!("open {}", p.display()))?;
        Ok(SeqFile { r: BufReader::with_capacity(1 << 22, f), name: p.display().to_string() })
    }

    /// Next record, or `None` at the end of the file.
    pub fn record(&mut self) -> Result<Option<Vec<u8>>> {
        let mut h = [0u8; 4];
        match self.r.read_exact(&mut h) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let n = be_i32(&h);
        if n < 0 {
            bail!("{}: negative record length {n} (not a big-endian sequential file?)", self.name);
        }
        let mut body = vec![0u8; n as usize];
        self.r.read_exact(&mut body)?;
        let mut t = [0u8; 4];
        self.r.read_exact(&mut t)?;
        if be_i32(&t) != n {
            bail!("{}: record markers {n} and {} differ", self.name, be_i32(&t));
        }
        Ok(Some(body))
    }

    fn need(&mut self, what: &str) -> Result<Vec<u8>> {
        self.record()?.ok_or_else(|| anyhow!("{}: end of file while reading {what}", self.name))
    }
}

// ---------------------------------------------------------------------------
// Restart
// ---------------------------------------------------------------------------

/// The fields of a `FULL` WW3 restart that the propagation needs.
pub struct Restart {
    pub time: Time,
    pub nsea: usize,
    pub nspec: usize,
    /// `VA(isp, isea)` as written, `nsea nspec`, spectrum of each sea point contiguous.
    pub va: Vec<f32>,
    pub tlev: Time,
    pub tice: Time,
    pub wlv: Vec<f32>,
    pub ice: Vec<f32>,
    pub mapsta: Vec<i32>,
    pub mapst2: Vec<i32>,
    /// `COLD` restart (`ww3_strt`): spectra only; `time` is zero, levels
    /// and ice are zero and `mapsta` is 1 (the caller sets the grid's).
    pub cold: bool,
}

impl Restart {
    /// Read a restart of an unstructured grid (`NX = NSEA`, `NY = 1`).
    pub fn read(p: &Path) -> Result<Restart> {
        let mut f = File::open(p).with_context(|| format!("open {}", p.display()))?;
        let mut head = vec![0u8; 26 + 10 + 30 + 4 + 8];
        f.read_exact(&mut head)?;
        let id = String::from_utf8_lossy(&head[..26]).to_string();
        if id != "WAVEWATCH III RESTART FILE" {
            bail!("{}: not a WW3 restart (id {id:?})", p.display());
        }
        let ver = String::from_utf8_lossy(&head[26..36]).to_string();
        let typ = String::from_utf8_lossy(&head[66..70]).to_string();
        if typ != "FULL" && typ != "COLD" {
            bail!("{}: restart type {typ:?}, expected FULL or COLD", p.display());
        }
        let cold = typ == "COLD";
        let nsea = be_i32(&head[70..74]) as usize;
        let nspec = be_i32(&head[74..78]) as usize;
        let lrecl = 4 * nspec.max(6 + 25 / 4 + 9 / 4 + 29 / 4 + 3 / 4);
        let len = f.metadata()?.len() as usize;
        let nsize = lrecl / 4;
        let npart = (nsea + nsize - 1) / nsize;
        let need = if cold { lrecl * (nsea + 2) } else { lrecl * (nsea + 3 + 2 * npart + npart) };
        if len < need {
            bail!("{}: {len} bytes, expected at least {need} for NSEA {nsea}, NSPEC {nspec} (version {ver})", p.display());
        }
        let rec = |n: usize| (lrecl * (n - 1)) as u64;
        let mut buf = vec![0u8; 8];
        f.seek(SeekFrom::Start(rec(2)))?;
        f.read_exact(&mut buf)?;
        let time = Time(be_i32(&buf[..4]), be_i32(&buf[4..]));
        // spectra: records 3 .. NSEA+2
        f.seek(SeekFrom::Start(rec(3)))?;
        let mut raw = vec![0u8; nsea * lrecl];
        f.read_exact(&mut raw)?;
        let mut va = vec![0f32; nsea * nspec];
        for (s, chunk) in raw.chunks_exact(lrecl).enumerate() {
            for (j, b) in chunk[..4 * nspec].chunks_exact(4).enumerate() {
                va[s * nspec + j] = be_f32(b);
            }
        }
        drop(raw);
        if cold {
            let (wlv, ice) = (vec![0f32; nsea], vec![0f32; nsea]);
            let (mapsta, mapst2) = (vec![1i32; nsea], vec![0i32; nsea]);
            return Ok(Restart { time, nsea, nspec, va, tlev: time, tice: time, wlv, ice, mapsta, mapst2, cold });
        }
        let mut tb = vec![0u8; 40];
        f.seek(SeekFrom::Start(rec(nsea + 3)))?;
        f.read_exact(&mut tb)?;
        let tlev = Time(be_i32(&tb[0..4]), be_i32(&tb[4..8]));
        let tice = Time(be_i32(&tb[8..12]), be_i32(&tb[12..16]));
        let mut read_parts = |first: usize, n: usize| -> Result<Vec<u8>> {
            let mut out = Vec::with_capacity(4 * n);
            for ip in 0..(n + nsize - 1) / nsize {
                let m = nsize.min(n - ip * nsize);
                let mut b = vec![0u8; 4 * m];
                f.seek(SeekFrom::Start(rec(first + ip)))?;
                f.read_exact(&mut b)?;
                out.extend_from_slice(&b);
            }
            Ok(out)
        };
        let r0 = nsea + 4;
        let wlv = be_f32s(&read_parts(r0, nsea)?);
        let ice = be_f32s(&read_parts(r0 + npart, nsea)?);
        let maptmp = be_i32s(&read_parts(r0 + 2 * npart, nsea)?);
        let mapsta: Vec<i32> = maptmp.iter().map(|&m| (m + 2).rem_euclid(8) - 2).collect();
        let mapst2: Vec<i32> = maptmp.iter().zip(&mapsta).map(|(&m, &s)| (m - s) / 8).collect();
        Ok(Restart { time, nsea, nspec, va, tlev, tice, wlv, ice, mapsta, mapst2, cold })
    }
}

// ---------------------------------------------------------------------------
// Forcing fields (current.ww3, level.ww3, ice.ww3)
// ---------------------------------------------------------------------------

pub struct FieldFile {
    f: SeqFile,
    pub id: String,
    pub nx: usize,
    pub ny: usize,
    /// Fields per time: 2 for currents, 1 for levels and ice.
    pub nfield: usize,
}

impl FieldFile {
    pub fn open(p: &Path) -> Result<FieldFile> {
        let mut f = SeqFile::open(p)?;
        let h = f.need("header")?;
        if h.len() < 28 || &h[..13] != b"WAVEWATCH III" {
            bail!("{}: not a WW3 forcing file", p.display());
        }
        let id = String::from_utf8_lossy(&h[13..16]).to_string();
        let nx = be_i32(&h[16..20]) as usize;
        let ny = be_i32(&h[20..24]) as usize;
        let nfield = match id.as_str() {
            "CUR" | "WND" => 2,
            "LEV" | "ICE" => 1,
            other => bail!("{}: field {other} not handled", p.display()),
        };
        Ok(FieldFile { f, id, nx, ny, nfield })
    }

    /// Next time and its fields, or `None` at the end of the file.
    pub fn next(&mut self) -> Result<Option<(Time, Vec<Vec<f32>>)>> {
        let t = match self.f.record()? {
            Some(t) => t,
            None => return Ok(None),
        };
        let time = Time(be_i32(&t[0..4]), be_i32(&t[4..8]));
        let mut fields = Vec::with_capacity(self.nfield);
        for _ in 0..self.nfield {
            let b = self.f.need("field")?;
            if b.len() != 4 * self.nx * self.ny {
                bail!("field {} at {time}: {} bytes, expected {}", self.id, b.len(), 4 * self.nx * self.ny);
            }
            fields.push(be_f32s(&b));
        }
        Ok(Some((time, fields)))
    }
}

// ---------------------------------------------------------------------------
// Boundary spectra (nest.ww3)
// ---------------------------------------------------------------------------

pub struct NestFile {
    f: SeqFile,
    pub nk: usize,
    pub nth: usize,
    pub xfr: f32,
    pub fr1: f32,
    pub th1: f32,
    pub nbi: usize,
    pub xbpi: Vec<f32>,
    pub ybpi: Vec<f32>,
    /// `IPBPI(i, j)`, `nbi * 4`, row i: the 4 spectra of boundary point i (1-based, 0 = none).
    pub ipbpi: Vec<i32>,
    pub rdbpi: Vec<f32>,
}

impl NestFile {
    pub fn open(p: &Path) -> Result<NestFile> {
        let mut f = SeqFile::open(p)?;
        let h = f.need("header")?;
        if &h[..32] != b"WAVEWATCH III BOUNDARY DATA FILE" {
            bail!("{}: not a WW3 boundary file", p.display());
        }
        let o = 42;
        let nk = be_i32(&h[o..o + 4]) as usize;
        let nth = be_i32(&h[o + 4..o + 8]) as usize;
        let xfr = be_f32(&h[o + 8..o + 12]);
        let fr1 = be_f32(&h[o + 12..o + 16]);
        let th1 = be_f32(&h[o + 16..o + 20]);
        let nbi = be_i32(&h[o + 20..o + 24]) as usize;
        let b = f.need("boundary points")?;
        let n = nbi;
        let xbpi = be_f32s(&b[..4 * n]);
        let ybpi = be_f32s(&b[4 * n..8 * n]);
        let ipc = be_i32s(&b[8 * n..24 * n]); // column-major (I fastest)
        let rdc = be_f32s(&b[24 * n..40 * n]);
        let mut ipbpi = vec![0i32; 4 * n];
        let mut rdbpi = vec![0f32; 4 * n];
        for i in 0..n {
            for j in 0..4 {
                ipbpi[4 * i + j] = ipc[j * n + i];
                rdbpi[4 * i + j] = rdc[j * n + i];
            }
        }
        Ok(NestFile { f, nk, nth, xfr, fr1, th1, nbi, xbpi, ybpi, ipbpi, rdbpi })
    }

    /// Next time and its station spectra (`ABPI`, `nbi2` spectra of `nk nth` values).
    pub fn next(&mut self) -> Result<Option<(Time, Vec<Vec<f32>>)>> {
        let t = match self.f.record()? {
            Some(t) => t,
            None => return Ok(None),
        };
        let time = Time(be_i32(&t[0..4]), be_i32(&t[4..8]));
        let nbi2 = be_i32(&t[8..12]) as usize;
        let mut spectra = Vec::with_capacity(nbi2);
        for _ in 0..nbi2 {
            let b = self.f.need("spectrum")?;
            if b.len() != 4 * self.nk * self.nth {
                bail!("boundary spectrum at {time}: {} bytes, expected {}", b.len(), 4 * self.nk * self.nth);
            }
            spectra.push(be_f32s(&b));
        }
        Ok(Some((time, spectra)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_roundtrip() {
        for s in ["20260108000000", "20260131233000", "20240229120000", "19991231235959"] {
            let t = Time::parse(s).unwrap();
            assert_eq!(Time::from_secs(t.secs()), t);
        }
        let a = Time::parse("20260108000000").unwrap();
        let b = Time::parse("20260110000000").unwrap();
        assert_eq!(b.secs() - a.secs(), 2 * 86400);
    }
}
