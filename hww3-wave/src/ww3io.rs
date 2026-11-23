//! Readers for the WW3 binary files of a run: restart (`W3IORS`), forcing
//! fields written by `ww3_prnc` (`W3FLDO` / `W3FLDG`) and open-boundary
//! spectra written by `ww3_bounc` (`W3IOBC`).
//!
//! The HWW3 build of WW3 writes big-endian files. Forcing and boundary
//! files are Fortran sequential unformatted files (4-byte record markers);
//! the restart is a stream file of fixed-length records of `4 NSPEC` bytes,
//! which [`Restart::write`] also writes.

use anyhow::{anyhow, bail, Context, Result};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
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

    /// Inverse of [`Time::secs`].
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

    /// Parse `yyyymmdd[hhmmss]`; other characters are ignored, so
    /// `2026-01-08 00:00:00` is accepted.
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

/// Big-endian `f32` values of a byte buffer.
pub fn be_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(be_f32).collect()
}

/// Big-endian `i32` values of a byte buffer.
pub fn be_i32s(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4).map(be_i32).collect()
}

/// Fortran sequential unformatted file with big-endian 4-byte record markers.
pub struct SeqFile {
    r: BufReader<File>,
    name: String,
}

impl SeqFile {
    /// Open a file for reading its records in order.
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

/// `VERINI` of `W3IORS`: the version a restart must carry to be read by
/// the WW3 of HWW3.
pub const RESTART_VERSION: &str = "2024-04-26";

/// Record length of a restart (`LRECL` of `W3IORS`): `NSPEC` reals, or
/// the header if longer.
fn restart_lrecl(nspec: usize) -> usize {
    4 * nspec.max(6 + 25 / 4 + 9 / 4 + 29 / 4 + 3 / 4)
}

/// The fields of a WW3 restart (`FULL`, or the spectra of a `COLD` one).
pub struct Restart {
    /// Model time of the restart.
    pub time: Time,
    /// Number of sea points (all nodes on an HWW3 grid).
    pub nsea: usize,
    /// Values per spectrum, `NK NTH`.
    pub nspec: usize,
    /// `VA(isp, isea)` as written, `nsea nspec`, spectrum of each sea point contiguous.
    pub va: Vec<f32>,
    /// Time of the water levels (`TLEV`).
    pub tlev: Time,
    /// Time of the ice concentrations (`TICE`).
    pub tice: Time,
    /// Water levels per sea point (m).
    pub wlv: Vec<f32>,
    /// Ice concentrations per sea point.
    pub ice: Vec<f32>,
    /// `MAPSTA` per node (1 sea, 2 open boundary, negative when masked by ice or drying).
    pub mapsta: Vec<i32>,
    /// `MAPST2` per node (reasons for masking).
    pub mapst2: Vec<i32>,
    /// `COLD` restart (`ww3_strt`): spectra only; `time` is zero, levels
    /// and ice are zero and `mapsta` is 1 (the caller sets the grid's).
    pub cold: bool,
    /// Version of the header (`VERINI`).
    pub version: String,
    /// Grid name of the header (`GNAME`, 30 characters).
    pub gname: String,
    /// Rest of the header record after `NSPEC`: the `FLOGRR` flags of the
    /// coupling fields, then zeros (empty: all false).
    pub flogrr: Vec<u8>,
    /// `TRHO`, `TIC1`, `TIC5` of a `FULL` restart (`(-1, 0)`: none read).
    pub tother: [Time; 3],
    /// `UST`, `USTDIR`, `ASF`, `FPIS` per sea point of a `FULL` restart
    /// (empty for a `COLD` one). The GPU model recomputes u* at each
    /// source call and does not read them.
    pub ust: Vec<f32>,
    /// See [`Restart::ust`].
    pub ustdir: Vec<f32>,
    /// See [`Restart::ust`].
    pub asf: Vec<f32>,
    /// See [`Restart::ust`].
    pub fpis: Vec<f32>,
    /// The bytes after the values of `FPIS`: the zero padding of its last
    /// record, or the coupling fields when `FLOGRR` has any.
    pub tail: Vec<u8>,
}

impl Restart {
    /// The start from calm conditions of `W3IORS` when there is no restart
    /// file (`TYPE = 'CALM'` with the `LN1` switch of HWW3): zero spectra,
    /// as a `COLD` restart (the caller sets the time and the grid's
    /// `MAPSTA`; levels and ice are zero until the first fields).
    pub fn calm(nsea: usize, nspec: usize) -> Restart {
        Restart {
            time: Time(0, 0),
            nsea,
            nspec,
            va: vec![0f32; nsea * nspec],
            tlev: Time(0, 0),
            tice: Time(0, 0),
            wlv: vec![0f32; nsea],
            ice: vec![0f32; nsea],
            mapsta: vec![1i32; nsea],
            mapst2: vec![0i32; nsea],
            cold: true,
            version: RESTART_VERSION.into(),
            gname: String::new(),
            flogrr: vec![],
            tother: [Time(-1, 0); 3],
            ust: vec![],
            ustdir: vec![],
            asf: vec![],
            fpis: vec![],
            tail: vec![],
        }
    }

    /// Read a restart of an unstructured grid (`NX = NSEA`, `NY = 1`).
    pub fn read(p: &Path) -> Result<Restart> {
        let mut f = File::open(p).with_context(|| format!("open {}", p.display()))?;
        let mut head = vec![0u8; 26 + 10 + 30 + 4 + 8];
        f.read_exact(&mut head)?;
        let gname = String::from_utf8_lossy(&head[36..66]).to_string();
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
        let lrecl = restart_lrecl(nspec);
        let mut flogrr = vec![0u8; lrecl - 78];
        f.read_exact(&mut flogrr)?;
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
            return Ok(Restart {
                time,
                nsea,
                nspec,
                va,
                tlev: time,
                tice: time,
                wlv,
                ice,
                mapsta,
                mapst2,
                cold,
                version: ver,
                gname,
                flogrr,
                tother: [Time(-1, 0); 3],
                ust: vec![],
                ustdir: vec![],
                asf: vec![],
                fpis: vec![],
                tail: vec![],
            });
        }
        let mut tb = vec![0u8; 40];
        f.seek(SeekFrom::Start(rec(nsea + 3)))?;
        f.read_exact(&mut tb)?;
        let tlev = Time(be_i32(&tb[0..4]), be_i32(&tb[4..8]));
        let tice = Time(be_i32(&tb[8..12]), be_i32(&tb[12..16]));
        let tother = [16, 24, 32].map(|k| Time(be_i32(&tb[k..k + 4]), be_i32(&tb[k + 4..k + 8])));
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
        // UST, USTDIR, ASF, FPIS, then what follows the values of FPIS
        let fpis_end = lrecl * (r0 + 7 * npart - 2) + 4 * (nsea - (npart - 1) * nsize);
        let (mut ust, mut ustdir, mut asf, mut fpis, mut tail) = (vec![], vec![], vec![], vec![], vec![]);
        if len >= fpis_end {
            ust = be_f32s(&read_parts(r0 + 3 * npart, nsea)?);
            ustdir = be_f32s(&read_parts(r0 + 4 * npart, nsea)?);
            asf = be_f32s(&read_parts(r0 + 5 * npart, nsea)?);
            fpis = be_f32s(&read_parts(r0 + 6 * npart, nsea)?);
            f.seek(SeekFrom::Start(fpis_end as u64))?;
            f.read_to_end(&mut tail)?;
        }
        Ok(Restart {
            time,
            nsea,
            nspec,
            va,
            tlev,
            tice,
            wlv,
            ice,
            mapsta,
            mapst2,
            cold,
            version: ver,
            gname,
            flogrr,
            tother,
            ust,
            ustdir,
            asf,
            fpis,
            tail,
        })
    }

    /// Write a `FULL` restart in the layout of `W3IORS` (unstructured grid,
    /// one process): header, time, one record per spectrum, the times of
    /// the fields, then `WLV`, `ICE`, `MAPSTA + 8 MAPST2`, `UST`, `USTDIR`,
    /// `ASF` and `FPIS` in records of `LRECL / 4` values, and
    /// [`Restart::tail`]. Empty `UST`, `USTDIR`, `FPIS` are written as
    /// zeros and an empty `ASF` as ones. A `FULL` restart read by
    /// [`Restart::read`] is written back byte for byte.
    pub fn write(&self, p: &Path) -> Result<()> {
        let (nsea, nspec) = (self.nsea, self.nspec);
        anyhow::ensure!(self.va.len() == nsea * nspec, "restart: {} spectral values for NSEA {nsea} NSPEC {nspec}", self.va.len());
        let lrecl = restart_lrecl(nspec);
        let nsize = lrecl / 4;
        let mut w = BufWriter::with_capacity(1 << 22, File::create(p).with_context(|| format!("create {}", p.display()))?);
        let mut rec = vec![0u8; lrecl];
        let mut put = |w: &mut BufWriter<File>, body: &[u8]| -> Result<()> {
            rec.iter_mut().for_each(|b| *b = 0);
            rec[..body.len()].copy_from_slice(body);
            w.write_all(&rec)?;
            Ok(())
        };
        let fixed = |s: &str, n: usize| -> Vec<u8> {
            let mut b: Vec<u8> = s.bytes().take(n).collect();
            b.resize(n, b' ');
            b
        };
        let mut head = b"WAVEWATCH III RESTART FILE".to_vec();
        head.extend(fixed(&self.version, 10));
        head.extend(fixed(&self.gname, 30));
        head.extend(b"FULL");
        head.extend((nsea as i32).to_be_bytes());
        head.extend((nspec as i32).to_be_bytes());
        let room = lrecl - head.len();
        head.extend(&self.flogrr[..self.flogrr.len().min(room)]);
        put(&mut w, &head)?;
        let times = |t: &[Time]| -> Vec<u8> { t.iter().flat_map(|t| [t.0.to_be_bytes(), t.1.to_be_bytes()].concat()).collect() };
        put(&mut w, &times(&[self.time]))?;
        let mut b = vec![0u8; 4 * nspec];
        for s in 0..nsea {
            for (j, v) in self.va[s * nspec..(s + 1) * nspec].iter().enumerate() {
                b[4 * j..4 * j + 4].copy_from_slice(&v.to_be_bytes());
            }
            put(&mut w, &b)?;
        }
        put(&mut w, &times(&[self.tlev, self.tice, self.tother[0], self.tother[1], self.tother[2]]))?;
        let fill = |v: &[f32], x: f32| -> Vec<u8> {
            if v.is_empty() {
                std::iter::repeat(x.to_be_bytes()).take(nsea).flatten().collect()
            } else {
                v.iter().flat_map(|x| x.to_be_bytes()).collect()
            }
        };
        let maptmp: Vec<u8> = self.mapsta.iter().zip(&self.mapst2).flat_map(|(&s, &m2)| (s + 8 * m2).to_be_bytes()).collect();
        let parts = [
            fill(&self.wlv, 0.0),
            fill(&self.ice, 0.0),
            maptmp,
            fill(&self.ust, 0.0),
            fill(&self.ustdir, 0.0),
            fill(&self.asf, 1.0),
            fill(&self.fpis, 0.0),
        ];
        let mut last = 0;
        for (k, part) in parts.iter().enumerate() {
            anyhow::ensure!(part.len() == 4 * nsea, "restart: field {k} has {} values for NSEA {nsea}", part.len() / 4);
            for chunk in part.chunks(4 * nsize) {
                put(&mut w, chunk)?;
                last = chunk.len();
            }
        }
        w.flush()?;
        if !self.tail.is_empty() {
            // the tail starts right after the values of FPIS, over the
            // padding of their last record
            let mut f = w.into_inner().map_err(|e| anyhow!("restart: {e}"))?;
            f.seek(SeekFrom::Current(last as i64 - lrecl as i64))?;
            f.write_all(&self.tail)?;
            let end = f.stream_position()?;
            f.set_len(end)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Forcing fields (current.ww3, level.ww3, ice.ww3)
// ---------------------------------------------------------------------------

/// A forcing file written by `ww3_prnc` (`current.ww3`, `level.ww3`,
/// `ice.ww3`, `wind.ww3`), read one time at a time.
pub struct FieldFile {
    f: SeqFile,
    /// Field identifier: `CUR`, `WND`, `LEV` or `ICE`.
    pub id: String,
    /// First dimension of the fields (the number of nodes on an HWW3 grid).
    pub nx: usize,
    /// Second dimension of the fields (1 on an unstructured grid).
    pub ny: usize,
    /// Fields per time: 2 for currents, 1 for levels and ice.
    pub nfield: usize,
    /// Grid type of the header (`GTYPE`, 3 for an unstructured grid).
    pub gtype: i32,
}

impl FieldFile {
    /// Open a forcing file and read its header.
    pub fn open(p: &Path) -> Result<FieldFile> {
        let mut f = SeqFile::open(p)?;
        let h = f.need("header")?;
        if h.len() < 28 || &h[..13] != b"WAVEWATCH III" {
            bail!("{}: not a WW3 forcing file", p.display());
        }
        let id = String::from_utf8_lossy(&h[13..16]).to_string();
        let nx = be_i32(&h[16..20]) as usize;
        let ny = be_i32(&h[20..24]) as usize;
        let gtype = be_i32(&h[24..28]);
        let nfield = match id.as_str() {
            "CUR" | "WND" => 2,
            "LEV" | "ICE" => 1,
            other => bail!("{}: field {other} not handled", p.display()),
        };
        Ok(FieldFile { f, id, nx, ny, nfield, gtype })
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

/// A forcing file as `ww3_prnc` writes it (`W3FLDO`, `W3FLDG`): a header
/// record (`'WAVEWATCH III'`, `IDFLD`, `NX`, `NY`, `GTYPE`, two filler
/// words written as zero, `TIDEFLAG` = 0), then per time a record with the
/// time and one record per field.
pub struct FieldWriter {
    w: BufWriter<File>,
    n: usize,
    nfield: usize,
}

fn write_record(w: &mut impl Write, body: &[u8]) -> Result<()> {
    let n = (body.len() as i32).to_be_bytes();
    w.write_all(&n)?;
    w.write_all(body)?;
    w.write_all(&n)?;
    Ok(())
}

impl FieldWriter {
    /// Create a forcing file of field `id` (`CUR`, `WND`, `LEV`, `ICE`).
    pub fn create(p: &Path, id: &str, nx: usize, ny: usize, gtype: i32) -> Result<FieldWriter> {
        let nfield = match id {
            "CUR" | "WND" => 2,
            "LEV" | "ICE" => 1,
            other => bail!("field {other} not handled"),
        };
        let mut w = BufWriter::with_capacity(1 << 22, File::create(p).with_context(|| format!("create {}", p.display()))?);
        let mut h = b"WAVEWATCH III".to_vec();
        h.extend(id.bytes());
        for v in [nx as i32, ny as i32, gtype, 0, 0, 0] {
            h.extend(v.to_be_bytes());
        }
        write_record(&mut w, &h)?;
        Ok(FieldWriter { w, n: nx * ny, nfield })
    }

    /// Write the fields of one time.
    pub fn write(&mut self, t: Time, fields: &[Vec<f32>]) -> Result<()> {
        anyhow::ensure!(fields.len() == self.nfield, "{} fields for {} per time", fields.len(), self.nfield);
        let mut tb = t.0.to_be_bytes().to_vec();
        tb.extend(t.1.to_be_bytes());
        write_record(&mut self.w, &tb)?;
        for f in fields {
            anyhow::ensure!(f.len() == self.n, "field of {} values for {}", f.len(), self.n);
            let b: Vec<u8> = f.iter().flat_map(|v| v.to_be_bytes()).collect();
            write_record(&mut self.w, &b)?;
        }
        Ok(())
    }

    /// Flush the file.
    pub fn finish(mut self) -> Result<()> {
        self.w.flush()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Boundary spectra (nest.ww3)
// ---------------------------------------------------------------------------

/// An open-boundary file written by `ww3_bounc` (`nest.ww3`), read one
/// time at a time.
pub struct NestFile {
    f: SeqFile,
    /// Number of frequencies of the spectra.
    pub nk: usize,
    /// Number of directions of the spectra.
    pub nth: usize,
    /// Frequency increment factor.
    pub xfr: f32,
    /// Lowest frequency (Hz).
    pub fr1: f32,
    /// First direction (rad).
    pub th1: f32,
    /// Number of boundary points.
    pub nbi: usize,
    /// Longitude of the boundary points (degrees).
    pub xbpi: Vec<f32>,
    /// Latitude of the boundary points (degrees).
    pub ybpi: Vec<f32>,
    /// `IPBPI(i, j)`, `nbi * 4`, row i: the 4 spectra of boundary point i (1-based, 0 = none).
    pub ipbpi: Vec<i32>,
    /// `RDBPI(i, j)`, `nbi * 4`: interpolation weights of those spectra.
    pub rdbpi: Vec<f32>,
}

impl NestFile {
    /// Open a boundary file and read its header and boundary points.
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
    use std::path::PathBuf;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("hww3_wave_{name}_{}", std::process::id()))
    }

    fn i32s(v: &[i32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_be_bytes()).collect()
    }

    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_be_bytes()).collect()
    }

    /// Append a Fortran sequential record with big-endian markers.
    fn rec(out: &mut Vec<u8>, body: &[u8]) {
        out.extend((body.len() as i32).to_be_bytes());
        out.extend(body);
        out.extend((body.len() as i32).to_be_bytes());
    }

    #[test]
    fn sequential_records() {
        let p = tmp("seq");
        let mut b = vec![];
        rec(&mut b, b"abc");
        rec(&mut b, &[1, 2, 3, 4, 5]);
        std::fs::write(&p, &b).unwrap();
        let mut f = SeqFile::open(&p).unwrap();
        assert_eq!(f.record().unwrap().unwrap(), b"abc");
        assert_eq!(f.record().unwrap().unwrap(), vec![1, 2, 3, 4, 5]);
        assert!(f.record().unwrap().is_none());
        // markers that differ are an error
        let n = b.len();
        b[n - 1] = 9;
        std::fs::write(&p, &b).unwrap();
        let mut f = SeqFile::open(&p).unwrap();
        f.record().unwrap();
        assert!(f.record().is_err());
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn forcing_file() {
        let p = tmp("cur");
        let mut b = vec![];
        let mut head = b"WAVEWATCH IIICUR".to_vec();
        head.extend(i32s(&[4, 1, -1]));
        rec(&mut b, &head);
        for (t, s) in [(20260108, 0), (20260108, 3000)] {
            rec(&mut b, &i32s(&[t, s]));
            rec(&mut b, &f32s(&[0.1, 0.2, 0.3, s as f32]));
            rec(&mut b, &f32s(&[-0.1, -0.2, -0.3, -0.4]));
        }
        std::fs::write(&p, &b).unwrap();
        let mut f = FieldFile::open(&p).unwrap();
        assert_eq!((f.id.as_str(), f.nx, f.ny, f.nfield), ("CUR", 4, 1, 2));
        let (t, v) = f.next().unwrap().unwrap();
        assert_eq!(t, Time(20260108, 0));
        assert_eq!(v, vec![vec![0.1, 0.2, 0.3, 0.0], vec![-0.1, -0.2, -0.3, -0.4]]);
        let (t, v) = f.next().unwrap().unwrap();
        assert_eq!((t, v[0][3]), (Time(20260108, 3000), 3000.0));
        assert!(f.next().unwrap().is_none());
        std::fs::remove_file(&p).unwrap();
    }

    /// A restart of `nsea` points and `nspec` values per spectrum, in the
    /// layout of `W3IORS`: fixed records of `LRECL` bytes.
    fn restart_bytes(typ: &str, nsea: usize, nspec: usize, maptmp: &[i32]) -> Vec<u8> {
        let lrecl = 4 * nspec.max(21);
        let mut recs: Vec<Vec<u8>> = vec![];
        let mut head = b"WAVEWATCH III RESTART FILE2021-04-06".to_vec();
        head.extend(format!("{:30}", "test grid").bytes());
        head.extend(typ.bytes());
        head.extend(i32s(&[nsea as i32, nspec as i32]));
        recs.push(head);
        recs.push(i32s(&[20260108, 0]));
        for s in 0..nsea {
            recs.push(f32s(&(0..nspec).map(|j| (s * 1000 + j) as f32).collect::<Vec<f32>>()));
        }
        if typ == "FULL" {
            recs.push(i32s(&[20260108, 3000, 20260101, 0]));
            let nsize = lrecl / 4;
            let wlv: Vec<f32> = (0..nsea).map(|s| 0.25 * s as f32).collect();
            let ice: Vec<f32> = (0..nsea).map(|s| if s % 7 == 0 { 0.5 } else { 0.0 }).collect();
            for part in [f32s(&wlv), f32s(&ice), i32s(maptmp)] {
                for chunk in part.chunks(4 * nsize) {
                    recs.push(chunk.to_vec());
                }
            }
        }
        recs.iter_mut().flat_map(|r| {
            r.resize(lrecl, 0);
            r.clone()
        }).collect()
    }

    #[test]
    fn cold_and_full_restarts() {
        let (nsea, nspec) = (30, 24);
        let p = tmp("restart");
        std::fs::write(&p, restart_bytes("COLD", nsea, nspec, &[])).unwrap();
        let r = Restart::read(&p).unwrap();
        assert!(r.cold);
        assert_eq!((r.nsea, r.nspec, r.time), (nsea, nspec, Time(20260108, 0)));
        assert_eq!((r.va[0], r.va[nspec + 3], r.va[nsea * nspec - 1]), (0.0, 1003.0, 29023.0));
        assert!(r.mapsta.iter().all(|&m| m == 1));
        // MAPSTA and MAPST2 are stored together as MAPSTA + 8 MAPST2
        let mut maptmp = vec![1i32; nsea];
        maptmp[3] = 2;
        maptmp[4] = -1 + 8;
        maptmp[5] = -2 + 16;
        std::fs::write(&p, restart_bytes("FULL", nsea, nspec, &maptmp)).unwrap();
        let r = Restart::read(&p).unwrap();
        std::fs::remove_file(&p).unwrap();
        assert!(!r.cold);
        assert_eq!((r.tlev, r.tice), (Time(20260108, 3000), Time(20260101, 0)));
        assert_eq!((r.wlv[29], r.ice[7], r.ice[8]), (7.25, 0.5, 0.0));
        assert_eq!(&r.mapsta[2..6], &[1, 2, -1, -2]);
        assert_eq!(&r.mapst2[2..6], &[0, 0, 1, 2]);
        assert_eq!(r.va[2 * nspec + 5], 2005.0);
    }

    #[test]
    fn forcing_written_back() {
        let p = tmp("forcing_rw");
        let mut w = FieldWriter::create(&p, "CUR", 3, 1, 3).unwrap();
        w.write(Time(20260101, 0), &[vec![1.0, 2.0, 3.0], vec![-1.0, -2.0, -3.0]]).unwrap();
        w.write(Time(20260101, 10000), &[vec![4.0, 5.0, 6.0], vec![0.5, 0.25, 0.0]]).unwrap();
        w.finish().unwrap();
        let mut f = FieldFile::open(&p).unwrap();
        assert_eq!((f.id.as_str(), f.nx, f.ny, f.nfield, f.gtype), ("CUR", 3, 1, 2, 3));
        let (t, v) = f.next().unwrap().unwrap();
        assert_eq!((t, v[1][2]), (Time(20260101, 0), -3.0));
        let (t, v) = f.next().unwrap().unwrap();
        assert_eq!((t, v[0][1], v[1][1]), (Time(20260101, 10000), 5.0, 0.25));
        assert!(f.next().unwrap().is_none());
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn restart_written_back() {
        // the layout of W3IORS: UST, USTDIR, ASF, FPIS after MAPTMP, then
        // coupling fields from the end of the values of FPIS
        let (nsea, nspec) = (30, 24);
        let lrecl = 4 * nspec;
        let maptmp: Vec<i32> = (0..nsea as i32).map(|s| if s == 4 { 7 } else { 1 }).collect();
        let mut b = restart_bytes("FULL", nsea, nspec, &maptmp);
        // TRHO, TIC1, TIC5 after TLEV, TICE
        let r = lrecl * (nsea + 2);
        b[r + 16..r + 40].copy_from_slice(&i32s(&[-1, 0, 20260101, 0, -1, 0]));
        b[66 + 4 + 8..66 + 4 + 8 + 4].copy_from_slice(&i32s(&[1]));
        for x in [0.3f32, 1.2, 1.0, 0.1] {
            let v: Vec<f32> = (0..nsea).map(|s| x + s as f32).collect();
            for chunk in f32s(&v).chunks(lrecl) {
                let mut c = chunk.to_vec();
                c.resize(lrecl, 0);
                b.extend(c);
            }
        }
        let p = tmp("restart_rw");
        let q = tmp("restart_rw2");
        for extra in [vec![], vec![9u8; 37]] {
            let mut bb = b.clone();
            if !extra.is_empty() {
                // a coupling field written from the end of the values of FPIS
                let end = bb.len() - lrecl + 4 * (nsea - lrecl / 4);
                bb.truncate(end);
                bb.extend(&extra);
            }
            std::fs::write(&p, &bb).unwrap();
            let r = Restart::read(&p).unwrap();
            assert_eq!((r.ust[2], r.asf[0], r.fpis[29]), (2.3, 1.0, 29.1));
            assert_eq!(r.tother, [Time(-1, 0), Time(20260101, 0), Time(-1, 0)]);
            r.write(&q).unwrap();
            assert_eq!(std::fs::read(&q).unwrap(), bb);
        }
        let mut c = Restart::calm(nsea, nspec);
        c.time = Time(20260101, 1000);
        c.write(&q).unwrap();
        let r = Restart::read(&q).unwrap();
        assert!(!r.cold && r.va.iter().all(|&v| v == 0.0) && r.asf.iter().all(|&v| v == 1.0));
        assert_eq!((r.time, r.version.as_str()), (Time(20260101, 1000), RESTART_VERSION));
        std::fs::remove_file(&p).unwrap();
        std::fs::remove_file(&q).unwrap();
    }

    #[test]
    fn boundary_file() {
        let (nk, nth, nbi) = (2usize, 3usize, 3usize);
        let p = tmp("nest");
        let mut b = vec![];
        let mut head = b"WAVEWATCH III BOUNDARY DATA FILE2021-04-06".to_vec();
        head.extend(i32s(&[nk as i32, nth as i32]));
        head.extend(f32s(&[1.1, 0.0373, 0.0]));
        head.extend(i32s(&[nbi as i32]));
        rec(&mut b, &head);
        // IPBPI and RDBPI are written column-major: IPBPI(i, j) at j nbi + i
        let mut pts = f32s(&[-10.0, -10.0, -10.0]);
        pts.extend(f32s(&[45.0, 45.1, 45.2]));
        pts.extend(i32s(&(0..4 * nbi).map(|c| (10 * (c % nbi) + c / nbi) as i32).collect::<Vec<i32>>()));
        pts.extend(f32s(&(0..4 * nbi).map(|c| if c < nbi { 1.0 } else { 0.0 }).collect::<Vec<f32>>()));
        rec(&mut b, &pts);
        rec(&mut b, &i32s(&[20260108, 0, 2]));
        rec(&mut b, &f32s(&vec![1.0; nk * nth]));
        rec(&mut b, &f32s(&vec![2.0; nk * nth]));
        std::fs::write(&p, &b).unwrap();
        let mut f = NestFile::open(&p).unwrap();
        assert_eq!((f.nk, f.nth, f.nbi, f.xfr, f.fr1), (nk, nth, nbi, 1.1, 0.0373));
        assert_eq!(f.ybpi, vec![45.0, 45.1, 45.2]);
        for i in 0..nbi {
            for j in 0..4 {
                assert_eq!(f.ipbpi[4 * i + j], (10 * i + j) as i32);
                assert_eq!(f.rdbpi[4 * i + j], if j == 0 { 1.0 } else { 0.0 });
            }
        }
        let (t, s) = f.next().unwrap().unwrap();
        assert_eq!((t, s.len(), s[1][0]), (Time(20260108, 0), 2, 2.0));
        assert!(f.next().unwrap().is_none());
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn time_parse_and_display() {
        let t = Time::parse("2026-01-08 06:30:00").unwrap();
        assert_eq!(t, Time(20260108, 63000));
        assert_eq!(t.to_string(), "20260108 063000");
        assert_eq!(Time::parse("20260108").unwrap(), Time(20260108, 0));
        assert!(Time::parse("2026").is_err());
        assert_eq!(Time::parse("19700101").unwrap().secs(), 0);
    }

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
