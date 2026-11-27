//! Reader of the output of the second source-term harness
//! (`harness/src_harness2.F90`): the ice terms `W3SIS2`, `W3SIC2`, the
//! reflection `W3SREF` and full `W3SRCE` calls with and without them, node
//! by node, on a structured grid of the IRI configuration.

use anyhow::{bail, Context, Result};
use std::path::Path;

/// Inputs and outputs of one node (single precision as WW3 wrote them).
#[derive(Clone, Debug, Default)]
pub struct Node2 {
    /// Sea-point index `ISEA` (1-based) and grid indices `IX`, `IY`.
    pub isea: i32,
    /// See [`Node2::isea`].
    pub ix: i32,
    /// See [`Node2::isea`].
    pub iy: i32,
    /// Water depth `DW`, depth used by the sources, ice concentration,
    /// ice thickness, floe diameter, largest floe diameter and iceberg
    /// damping, as passed to `W3SRCE`.
    pub dw: f32,
    /// See [`Node2::dw`].
    pub depth: f32,
    /// See [`Node2::dw`].
    pub ice: f32,
    /// See [`Node2::dw`].
    pub iceh: f32,
    /// See [`Node2::dw`].
    pub icef: f32,
    /// See [`Node2::dw`].
    pub icedmax: f32,
    /// See [`Node2::dw`].
    pub berg: f32,
    /// Wind speed and direction (radians), u* and its direction of the
    /// restart, `ASF`, `FPIS`, `D50`, `PSIC`.
    pub u10: f32,
    /// See [`Node2::u10`].
    pub u10dir: f32,
    /// See [`Node2::u10`].
    pub ust: f32,
    /// See [`Node2::u10`].
    pub ustdir: f32,
    /// See [`Node2::u10`].
    pub asf: f32,
    /// See [`Node2::u10`].
    pub fpis: f32,
    /// See [`Node2::u10`].
    pub d50: f32,
    /// See [`Node2::u10`].
    pub psic: f32,
    /// Cell sizes `DELX`, `DELY` (m), cell area `DELA`, transparencies
    /// `TRNX`, `TRNY`.
    pub delx: f32,
    /// See [`Node2::delx`].
    pub dely: f32,
    /// See [`Node2::delx`].
    pub dela: f32,
    /// See [`Node2::delx`].
    pub trnx: f32,
    /// See [`Node2::delx`].
    pub trny: f32,
    /// `REFLEC(1..4)` and `REFLED(1..6)` passed to `W3SREF`.
    pub reflec: [f32; 4],
    /// See [`Node2::reflec`].
    pub refled: [i32; 6],
    /// `CLATS`.
    pub clats: f32,
    /// Wavenumbers and group speeds (`NK`), spectrum of the restart.
    pub wn: Vec<f32>,
    /// See [`Node2::wn`].
    pub cg: Vec<f32>,
    /// See [`Node2::wn`].
    pub spec0: Vec<f32>,
    /// `EMEAN`, `FMEAN` of `W3SPR4` on the restart spectrum.
    pub emean: f32,
    /// See [`Node2::emean`].
    pub fmean: f32,
    /// `W3SIS2`: S, D, DISSIP (`NSPEC`), R (`NK`) and the floe diameter
    /// after the call.
    pub sis2_s: Vec<f32>,
    /// See [`Node2::sis2_s`].
    pub sis2_d: Vec<f32>,
    /// See [`Node2::sis2_s`].
    pub sis2_dissip: Vec<f32>,
    /// See [`Node2::sis2_s`].
    pub sis2_r: Vec<f32>,
    /// See [`Node2::sis2_s`].
    pub sis2_icef: f32,
    /// `W3SIC2`: S, D.
    pub sic2_s: Vec<f32>,
    /// See [`Node2::sic2_s`].
    pub sic2_d: Vec<f32>,
    /// `W3SREF`: S (`VREF`).
    pub sref: Vec<f32>,
    /// Full `W3SRCE` with ice and reflection: spectrum, mean dynamic
    /// step, floe diameter, `PHICE`, `TAUICE(2)`, u*.
    pub srce_spec: Vec<f32>,
    /// See [`Node2::srce_spec`].
    pub srce_dtdyn: f32,
    /// See [`Node2::srce_spec`].
    pub srce_icef: f32,
    /// See [`Node2::srce_spec`].
    pub srce_phice: f32,
    /// See [`Node2::srce_spec`].
    pub srce_tauice: [f32; 2],
    /// See [`Node2::srce_spec`].
    pub srce_ustar: f32,
    /// The same `W3SRCE` with no ice and no reflection: spectrum, mean
    /// dynamic step.
    pub srce0_spec: Vec<f32>,
    /// See [`Node2::srce0_spec`].
    pub srce0_dtdyn: f32,
}

/// A harness file `HWW3SRC2`.
#[derive(Clone, Debug)]
pub struct Harness2 {
    /// Number of frequencies, directions and spectral bins.
    pub nk: usize,
    /// See [`Harness2::nk`].
    pub nth: usize,
    /// See [`Harness2::nk`].
    pub nspec: usize,
    /// Restart and wind times (`YYYYMMDD`, `HHMMSS`) and the global step.
    pub restart_time: (i32, i32),
    /// See [`Harness2::restart_time`].
    pub wind_time: (i32, i32),
    /// See [`Harness2::restart_time`].
    pub dtg: f32,
    /// The nodes.
    pub nodes: Vec<Node2>,
}

impl Harness2 {
    /// Read a harness file.
    pub fn read(p: &Path) -> Result<Harness2> {
        let b = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
        if b.len() < 48 || &b[..8] != b"HWW3SRC2" {
            bail!("{}: not a HWW3SRC2 harness file", p.display());
        }
        let i32at = |o: usize| i32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let (nk, nth, nspec, nnode, nrec) = (i32at(8) as usize, i32at(12) as usize, i32at(16) as usize, i32at(20) as usize, i32at(24) as usize);
        let restart_time = (i32at(28), i32at(32));
        let wind_time = (i32at(36), i32at(40));
        let dtg = f32::from_le_bytes(b[44..48].try_into().unwrap());
        let expect = (33 + 2 * nk + nspec) + 2 + (3 * nspec + nk + 1) + 2 * nspec + nspec + (nspec + 6) + (nspec + 1);
        if nrec != expect {
            bail!("{}: {nrec} values per node, layout gives {expect}", p.display());
        }
        let rec = 4 + 4 * nrec;
        if b.len() != 48 + nnode * rec {
            bail!("{}: {} bytes for {nnode} nodes of {rec}", p.display(), b.len());
        }
        let mut nodes = Vec::with_capacity(nnode);
        for n in 0..nnode {
            let o = 48 + n * rec;
            let v: Vec<f32> = b[o + 4..o + rec].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let mut k = 0usize;
            let mut take = |m: usize| -> Vec<f32> {
                let s = v[k..k + m].to_vec();
                k += m;
                s
            };
            let s = take(33);
            let mut nd = Node2 {
                isea: i32at(o),
                ix: s[0] as i32,
                iy: s[1] as i32,
                dw: s[2],
                depth: s[3],
                ice: s[4],
                iceh: s[5],
                icef: s[6],
                icedmax: s[7],
                berg: s[8],
                u10: s[9],
                u10dir: s[10],
                ust: s[11],
                ustdir: s[12],
                asf: s[13],
                fpis: s[14],
                d50: s[15],
                psic: s[16],
                delx: s[17],
                dely: s[18],
                dela: s[19],
                trnx: s[20],
                trny: s[21],
                reflec: [s[22], s[23], s[24], s[25]],
                refled: [s[26] as i32, s[27] as i32, s[28] as i32, s[29] as i32, s[30] as i32, s[31] as i32],
                clats: s[32],
                ..Default::default()
            };
            nd.wn = take(nk);
            nd.cg = take(nk);
            nd.spec0 = take(nspec);
            let m = take(2);
            nd.emean = m[0];
            nd.fmean = m[1];
            nd.sis2_s = take(nspec);
            nd.sis2_d = take(nspec);
            nd.sis2_dissip = take(nspec);
            nd.sis2_r = take(nk);
            nd.sis2_icef = take(1)[0];
            nd.sic2_s = take(nspec);
            nd.sic2_d = take(nspec);
            nd.sref = take(nspec);
            nd.srce_spec = take(nspec);
            let t = take(6);
            nd.srce_dtdyn = t[0];
            nd.srce_icef = t[1];
            nd.srce_phice = t[2];
            nd.srce_tauice = [t[3], t[4]];
            nd.srce_ustar = t[5];
            nd.srce0_spec = take(nspec);
            nd.srce0_dtdyn = take(1)[0];
            nodes.push(nd);
        }
        Ok(Harness2 { nk, nth, nspec, restart_time, wind_time, dtg, nodes })
    }
}
