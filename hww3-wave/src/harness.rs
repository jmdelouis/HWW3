//! Reader of the WW3 source-term harness of milestone M2a
//! (`harness/src_harness.F90`).
//!
//! The harness calls the source terms of WW3 node by node, on the spectra
//! of a restart and the wind of a forcing file, and writes two files:
//!
//! - `OUT`: the inputs and outputs of each call, node by node, in the order
//!   of the first iteration of `W3SRCE` (`W3SPR4`, `W3SIN4`, `W3SPR4`,
//!   `W3SLN1`, `W3SIN4`, `W3SNL1`, `W3SDS4`, `W3SDB1`, `W3SBT4`), then a
//!   full `W3SRCE` from the same state ([`Harness`]);
//! - `OUT.params`: the parameters and tables those routines use, as WW3
//!   holds them after `W3IOGR` ([`Params`]).
//!
//! The ports of the source terms read [`Params`] rather than computing the
//! tables again, and are checked against [`Harness`] node by node.

use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

/// One record of the parameter file.
#[derive(Clone, Debug, Default)]
pub struct ParamArray {
    /// Fortran lower bounds.
    pub lbound: Vec<i32>,
    /// Fortran shape (first index fastest in the values).
    pub shape: Vec<usize>,
    /// Values of a real record.
    pub real: Vec<f32>,
    /// Values of an integer record.
    pub int: Vec<i32>,
}

/// The parameters and tables of the source terms (`OUT.params`).
#[derive(Clone, Debug, Default)]
pub struct Params {
    /// Records by name.
    pub map: HashMap<String, ParamArray>,
}

impl Params {
    /// Read a parameter file.
    pub fn read(p: &Path) -> Result<Params> {
        let mut b = vec![];
        std::fs::File::open(p)?.read_to_end(&mut b)?;
        Params::parse(&b)
    }

    /// Parse the bytes of a parameter file.
    pub fn parse(b: &[u8]) -> Result<Params> {
        if b.len() < 8 || &b[..8] != b"HWW3PAR1" {
            bail!("not a harness parameter file");
        }
        let mut pos = 8;
        let mut map = HashMap::new();
        let i32_at = |p: usize| -> Result<i32> {
            b.get(p..p + 4).map(|s| i32::from_le_bytes(s.try_into().unwrap())).ok_or_else(|| anyhow!("truncated parameter file"))
        };
        while pos + 17 <= b.len() {
            let name = String::from_utf8_lossy(&b[pos..pos + 16]).trim().to_string();
            let kind = b[pos + 16];
            pos += 17;
            let ndim = i32_at(pos)? as usize;
            pos += 4;
            let mut lbound = vec![];
            for _ in 0..ndim {
                lbound.push(i32_at(pos)?);
                pos += 4;
            }
            let mut shape = vec![];
            for _ in 0..ndim {
                shape.push(i32_at(pos)? as usize);
                pos += 4;
            }
            let n: usize = shape.iter().product();
            let raw = b.get(pos..pos + 4 * n).ok_or_else(|| anyhow!("truncated record {name}"))?;
            pos += 4 * n;
            let mut a = ParamArray { lbound, shape, ..Default::default() };
            match kind {
                b'R' => a.real = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
                b'I' => a.int = raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect(),
                k => bail!("record {name}: unknown kind {}", k as char),
            }
            map.insert(name, a);
        }
        Ok(Params { map })
    }

    fn get(&self, name: &str) -> Result<&ParamArray> {
        self.map.get(name).ok_or_else(|| anyhow!("parameter {name} not in the harness file"))
    }

    /// Real array `name`, in Fortran order from its lower bound.
    pub fn r(&self, name: &str) -> Result<&[f32]> {
        let a = self.get(name)?;
        if a.real.is_empty() && !a.int.is_empty() {
            bail!("parameter {name} is an integer");
        }
        Ok(&a.real)
    }

    /// Real scalar `name`.
    pub fn r1(&self, name: &str) -> Result<f32> {
        self.r(name)?.first().copied().ok_or_else(|| anyhow!("parameter {name} is empty"))
    }

    /// Integer array `name`, in Fortran order from its lower bound.
    pub fn i(&self, name: &str) -> Result<&[i32]> {
        let a = self.get(name)?;
        if a.int.is_empty() && !a.real.is_empty() {
            bail!("parameter {name} is real");
        }
        Ok(&a.int)
    }

    /// Integer scalar `name`.
    pub fn i1(&self, name: &str) -> Result<i32> {
        self.i(name)?.first().copied().ok_or_else(|| anyhow!("parameter {name} is empty"))
    }

    /// Whether the file holds `name`.
    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    /// Lower bounds of every dimension of `name`.
    pub fn get_lbounds(&self, name: &str) -> Result<Vec<i32>> {
        Ok(self.get(name)?.lbound.clone())
    }

    /// Lower bound of the first dimension of `name`.
    pub fn lbound(&self, name: &str) -> Result<i32> {
        Ok(self.get(name)?.lbound.first().copied().unwrap_or(1))
    }

    /// Shape of `name`.
    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.get(name)?.shape)
    }

    /// Replace the values of the real record `name` (same length, rounded
    /// to single precision as WW3 holds them).
    pub fn set_real(&mut self, name: &str, v: &[f64]) -> Result<()> {
        let a = self.map.get_mut(name).ok_or_else(|| anyhow!("parameter {name} not in the harness file"))?;
        if a.real.len() != v.len() {
            bail!("parameter {name}: {} values for {}", v.len(), a.real.len());
        }
        a.real = v.iter().map(|&x| x as f32).collect();
        Ok(())
    }

    /// Write the parameter file (records in the order of their names).
    pub fn write(&self, p: &Path) -> Result<()> {
        let mut b: Vec<u8> = b"HWW3PAR1".to_vec();
        let mut names: Vec<&String> = self.map.keys().collect();
        names.sort();
        for name in names {
            let a = &self.map[name];
            if name.len() > 16 {
                bail!("record name {name} longer than 16 characters");
            }
            b.extend_from_slice(format!("{name:<16}").as_bytes());
            let real = !a.real.is_empty() || a.int.is_empty();
            b.push(if real { b'R' } else { b'I' });
            b.extend_from_slice(&(a.shape.len() as i32).to_le_bytes());
            for &l in &a.lbound {
                b.extend_from_slice(&l.to_le_bytes());
            }
            for &s in &a.shape {
                b.extend_from_slice(&(s as i32).to_le_bytes());
            }
            if real {
                a.real.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
            } else {
                a.int.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
            }
        }
        std::fs::File::create(p)?.write_all(&b)?;
        Ok(())
    }
}

/// Outputs of `W3SPR4`.
#[derive(Clone, Debug, Default)]
pub struct Spr4 {
    /// EMEAN FMEAN FMEAN1 WNMEAN AMAX USTAR USTDIR TAUWX TAUWY CD Z0 CHARN FMEANWS DLWMEAN.
    pub v: [f32; 14],
    /// LLWS: wind-sea flags of the spectral bins.
    pub llws: Vec<bool>,
}

impl Spr4 {
    /// Names of [`Spr4::v`].
    pub const NAMES: [&'static str; 14] =
        ["EMEAN", "FMEAN", "FMEAN1", "WNMEAN", "AMAX", "USTAR", "USTDIR", "TAUWX", "TAUWY", "CD", "Z0", "CHARN", "FMEANWS", "DLWMEAN"];
}

/// Outputs of `W3SIN4`.
#[derive(Clone, Debug, Default)]
pub struct Sin4 {
    /// Source term S.
    pub s: Vec<f32>,
    /// Diagonal term D.
    pub d: Vec<f32>,
    /// TAUWX TAUWY TAUWNX TAUWNY.
    pub tau: [f32; 4],
    /// BRLAMBDA as passed in (unchanged by `W3SIN4`).
    pub brlambda: Vec<f32>,
}

/// A source term and its diagonal part.
#[derive(Clone, Debug, Default)]
pub struct SourceD {
    /// Source term S.
    pub s: Vec<f32>,
    /// Diagonal term D.
    pub d: Vec<f32>,
}

/// Inputs of a node.
#[derive(Clone, Debug, Default)]
pub struct NodeInputs {
    /// DW DEPTH CLATS U10 U10DIR UST0 USTDIR0 ICE ASF FPIS0 D50 PSIC.
    pub v: [f32; 12],
    /// Wavenumbers `WN(1..NK)`.
    pub wn: Vec<f32>,
    /// Group speeds `CG(1..NK)`.
    pub cg: Vec<f32>,
    /// Spectrum `VA(ISPEC)`, `ISPEC = ITH + (IK-1) NTH`.
    pub spec: Vec<f32>,
}

impl NodeInputs {
    /// Names of [`NodeInputs::v`].
    pub const NAMES: [&'static str; 12] = ["DW", "DEPTH", "CLATS", "U10", "U10DIR", "UST0", "USTDIR0", "ICE", "ASF", "FPIS0", "D50", "PSIC"];
}

/// Outputs of the full `W3SRCE` call.
#[derive(Clone, Debug, Default)]
pub struct Srce {
    /// Spectrum after the 600 s integration.
    pub spec: Vec<f32>,
    /// DTDYN FCUT USTAR USTDIR TAUWX TAUWY TAUOX TAUOY TAUWIX TAUWIY TAUWNX TAUWNY PHIAW CHARN TWS PHIOC
    /// WHITECAP(4) TAUBBL(2) TAUICE(2) PHICE TAUOCX TAUOCY WNMEAN FPI SHAVE.
    pub v: [f32; 30],
    /// ALPHA(1..NK).
    pub alpha: Vec<f32>,
}

impl Srce {
    /// Names of [`Srce::v`].
    pub const NAMES: [&'static str; 30] = [
        "DTDYN", "FCUT", "USTAR", "USTDIR", "TAUWX", "TAUWY", "TAUOX", "TAUOY", "TAUWIX", "TAUWIY", "TAUWNX", "TAUWNY", "PHIAW",
        "CHARN", "TWS", "PHIOC", "WHITECAP1", "WHITECAP2", "WHITECAP3", "WHITECAP4", "TAUBBL1", "TAUBBL2", "TAUICE1", "TAUICE2",
        "PHICE", "TAUOCX", "TAUOCY", "WNMEAN", "FPI", "SHAVE",
    ];
}

/// The calls of one node.
#[derive(Clone, Debug, Default)]
pub struct HarnessNode {
    /// Sea-point index (1-based, as in WW3).
    pub isea: usize,
    /// Inputs.
    pub inputs: NodeInputs,
    /// First `W3SPR4`.
    pub spr4a: Spr4,
    /// `W3SIN4` pre-call (TAUWBUG).
    pub sin4a: Sin4,
    /// Second `W3SPR4`.
    pub spr4b: Spr4,
    /// FHIGH and `W3SLN1`.
    pub fhigh: f32,
    /// `W3SLN1` source term.
    pub sln1: Vec<f32>,
    /// `W3SIN4` of the integration loop.
    pub sin4: Sin4,
    /// `W3SNL1` (DIA).
    pub snl1: SourceD,
    /// `W3SDS4` and WHITECAP(4).
    pub sds4: SourceD,
    /// WHITECAP of `W3SDS4`.
    pub whitecap: [f32; 4],
    /// `W3SDB1` and LBREAK.
    pub sdb1: SourceD,
    /// LBREAK of `W3SDB1`.
    pub lbreak: bool,
    /// `W3SBT4`.
    pub sbt4: SourceD,
    /// TAUBBL(2) and BEDFORM(3) of `W3SBT4`.
    pub sbt4_extra: [f32; 5],
    /// Full `W3SRCE`.
    pub srce: Srce,
}

/// The harness output file.
#[derive(Clone, Debug, Default)]
pub struct Harness {
    /// Number of frequencies.
    pub nk: usize,
    /// Number of directions.
    pub nth: usize,
    /// `NK NTH`.
    pub nspec: usize,
    /// Restart time `(yyyymmdd, hhmmss)`.
    pub restart_time: (i32, i32),
    /// Wind time `(yyyymmdd, hhmmss)`.
    pub wind_time: (i32, i32),
    /// Global time step of the `W3SRCE` call (s).
    pub dtg: f32,
    /// The nodes.
    pub nodes: Vec<HarnessNode>,
}

/// Number of values per node for `nk` frequencies and `nspec` bins.
pub fn record_len(nk: usize, nspec: usize) -> usize {
    (12 + 2 * nk + nspec) + 2 * (14 + nspec) + 2 * (3 * nspec + 4) + (1 + nspec) + 2 * nspec + (2 * nspec + 4) + (1 + 2 * nspec)
        + (2 * nspec + 5)
        + (nspec + 30 + nk)
}

struct Cursor<'a> {
    v: &'a [f32],
    p: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Vec<f32> {
        let s = self.v[self.p..self.p + n].to_vec();
        self.p += n;
        s
    }
    fn one(&mut self) -> f32 {
        self.p += 1;
        self.v[self.p - 1]
    }
    fn arr<const N: usize>(&mut self) -> [f32; N] {
        let mut a = [0f32; N];
        a.copy_from_slice(&self.v[self.p..self.p + N]);
        self.p += N;
        a
    }
}

impl Harness {
    /// Read a harness output file.
    pub fn read(p: &Path) -> Result<Harness> {
        let mut b = vec![];
        std::fs::File::open(p)?.read_to_end(&mut b)?;
        Harness::parse(&b)
    }

    /// Parse the bytes of a harness output file.
    pub fn parse(b: &[u8]) -> Result<Harness> {
        if b.len() < 48 || &b[..8] != b"HWW3SRC1" {
            bail!("not a harness output file");
        }
        let i = |k: usize| i32::from_le_bytes(b[8 + 4 * k..12 + 4 * k].try_into().unwrap());
        let (nk, nth, nspec, nnode, nrec) = (i(0) as usize, i(1) as usize, i(2) as usize, i(3) as usize, i(4) as usize);
        if nspec != nk * nth || nrec != record_len(nk, nspec) {
            bail!("harness layout: NK {nk}, NTH {nth}, NSPEC {nspec}, {nrec} values per node");
        }
        let dtg = f32::from_le_bytes(b[44..48].try_into().unwrap());
        let mut h = Harness { nk, nth, nspec, restart_time: (i(5), i(6)), wind_time: (i(7), i(8)), dtg, nodes: vec![] };
        let rec_bytes = 4 + 4 * nrec;
        if b.len() != 48 + nnode * rec_bytes {
            bail!("harness file: {} bytes for {nnode} nodes of {nrec} values", b.len());
        }
        for n in 0..nnode {
            let o = 48 + n * rec_bytes;
            let isea = i32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
            let v: Vec<f32> = b[o + 4..o + rec_bytes].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let mut c = Cursor { v: &v, p: 0 };
            let inputs = NodeInputs { v: c.arr(), wn: c.take(nk), cg: c.take(nk), spec: c.take(nspec) };
            let spr4 = |c: &mut Cursor| Spr4 { v: c.arr(), llws: c.take(nspec).iter().map(|&x| x > 0.5).collect() };
            let sin4 = |c: &mut Cursor| Sin4 { s: c.take(nspec), d: c.take(nspec), tau: c.arr(), brlambda: c.take(nspec) };
            let spr4a = spr4(&mut c);
            let sin4a = sin4(&mut c);
            let spr4b = spr4(&mut c);
            let fhigh = c.one();
            let sln1 = c.take(nspec);
            let sin4m = sin4(&mut c);
            let snl1 = SourceD { s: c.take(nspec), d: c.take(nspec) };
            let sds4 = SourceD { s: c.take(nspec), d: c.take(nspec) };
            let whitecap = c.arr();
            let lbreak = c.one() > 0.5;
            let sdb1 = SourceD { s: c.take(nspec), d: c.take(nspec) };
            let sbt4 = SourceD { s: c.take(nspec), d: c.take(nspec) };
            let sbt4_extra = c.arr();
            let srce = Srce { spec: c.take(nspec), v: c.arr(), alpha: c.take(nk) };
            debug_assert_eq!(c.p, nrec);
            h.nodes.push(HarnessNode {
                isea,
                inputs,
                spr4a,
                sin4a,
                spr4b,
                fhigh,
                sln1,
                sin4: sin4m,
                snl1,
                sds4,
                whitecap,
                sdb1,
                lbreak,
                sbt4,
                sbt4_extra,
                srce,
            });
        }
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harness_files_round_trip() {
        // parameter file: a real scalar, a real array with lower bound 0, an integer 2-D array
        let mut p = b"HWW3PAR1".to_vec();
        let mut rec = |name: &str, kind: u8, lb: &[i32], shape: &[i32], vals: &[u8]| {
            let mut nm = [b' '; 16];
            nm[..name.len()].copy_from_slice(name.as_bytes());
            p.extend_from_slice(&nm);
            p.push(kind);
            p.extend_from_slice(&(lb.len() as i32).to_le_bytes());
            lb.iter().chain(shape).for_each(|x| p.extend_from_slice(&x.to_le_bytes()));
            p.extend_from_slice(vals);
        };
        rec("DTH", b'R', &[1], &[1], &0.5f32.to_le_bytes());
        let sig: Vec<u8> = [0.1f32, 0.2, 0.3].iter().flat_map(|x| x.to_le_bytes()).collect();
        rec("SIG", b'R', &[0], &[3], &sig);
        let sat: Vec<u8> = (1..=6i32).flat_map(|x| x.to_le_bytes()).collect();
        rec("SATINDICES", b'I', &[1, 1], &[3, 2], &sat);
        let par = Params::parse(&p).unwrap();
        assert_eq!(par.r1("DTH").unwrap(), 0.5);
        assert_eq!(par.r("SIG").unwrap(), &[0.1, 0.2, 0.3]);
        assert_eq!(par.lbound("SIG").unwrap(), 0);
        assert_eq!(par.shape("SATINDICES").unwrap(), &[3, 2]);
        assert_eq!(par.i("SATINDICES").unwrap()[4], 5);
        assert!(par.r("SATINDICES").is_err() && par.r1("NONE").is_err());

        // output file: two nodes of NK = 2, NTH = 3, values numbered in order
        let (nk, nth) = (2usize, 3usize);
        let nspec = nk * nth;
        let nrec = record_len(nk, nspec);
        let mut b = b"HWW3SRC1".to_vec();
        for x in [nk as i32, nth as i32, nspec as i32, 2, nrec as i32, 20260108, 0, 20260108, 0] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b.extend_from_slice(&600f32.to_le_bytes());
        for isea in [7i32, 9] {
            b.extend_from_slice(&isea.to_le_bytes());
            for k in 0..nrec {
                b.extend_from_slice(&(k as f32).to_le_bytes());
            }
        }
        let h = Harness::parse(&b).unwrap();
        assert_eq!((h.nk, h.nth, h.nspec, h.nodes.len(), h.dtg), (2, 3, 6, 2, 600.0));
        let n = &h.nodes[1];
        assert_eq!(n.isea, 9);
        assert_eq!(n.inputs.v[0], 0.0);
        assert_eq!(n.inputs.wn, vec![12.0, 13.0]);
        assert_eq!(n.inputs.spec[0], 16.0);
        // the last value is ALPHA(NK)
        assert_eq!(*n.srce.alpha.last().unwrap(), (nrec - 1) as f32);
        assert_eq!(n.srce.v[0], (nrec - 30 - nk) as f32);
        assert!(Harness::parse(&b[..b.len() - 4]).is_err());
    }
}
