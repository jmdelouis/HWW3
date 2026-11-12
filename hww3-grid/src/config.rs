//! JSON configuration of a grid.

use crate::geo::Ellipsoid;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Config {
    /// grid name, used for the output file names
    pub name: String,
    pub domain: Domain,
    #[serde(default = "default_ellipsoid")]
    pub ellipsoid: Ellipsoid,
    /// coarsest HEALPix nside (open ocean)
    pub nside_min: u32,
    /// finest HEALPix nside (coast)
    pub nside_max: u32,
    /// maximum number of levels between nside_min and nside_max
    #[serde(default = "default_max_span")]
    pub max_level_span: u32,
    /// refinement rules: a cell closer than `max_dist_km` to the coast is
    /// refined up to `nside`
    pub refine: Vec<Rule>,
    /// cells shallower than this (m) are dry
    #[serde(default = "default_min_depth")]
    pub min_depth_m: f64,
    /// minimum fraction of wet bathymetry samples for a cell to be wet
    #[serde(default = "default_wet_fraction")]
    pub wet_fraction: f64,
    /// keep only the largest connected wet region
    #[serde(default = "default_true")]
    pub keep_largest_component: bool,
    /// path of the bathymetry meta.json (see `bathy::BathyMeta`)
    pub bathy: String,
    pub coast: CoastSource,
    pub output_dir: String,
    #[serde(default)]
    pub ww3: Ww3Params,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Domain {
    pub lon_min: f64,
    pub lon_max: f64,
    pub lat_min: f64,
    pub lat_max: f64,
}

impl Domain {
    /// geodetic degrees, lon in (-180, 180]
    pub fn contains(&self, lon: f64, lat: f64) -> bool {
        lat >= self.lat_min && lat <= self.lat_max && lon >= self.lon_min && lon <= self.lon_max
    }

    pub fn contains_with_margin(&self, lon: f64, lat: f64, margin: f64) -> bool {
        let c = lat.abs().min(85.0).to_radians().cos();
        lat >= self.lat_min - margin
            && lat <= self.lat_max + margin
            && lon >= self.lon_min - margin / c
            && lon <= self.lon_max + margin / c
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Rule {
    pub max_dist_km: f64,
    pub nside: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "source", rename_all = "lowercase")]
pub enum CoastSource {
    /// GSHHG native binary file (e.g. gshhs_f.b); `levels` selects the
    /// polygon levels used as coastline (1 = land/ocean boundary)
    Gshhg {
        path: String,
        #[serde(default = "default_levels")]
        levels: Vec<u8>,
    },
    /// coastline = zero crossings of the bathymetry grid
    Bathy,
}

/// WAVEWATCH III parameters written to ww3_grid.nml
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Ww3Params {
    pub nth: u32,
    pub nk: u32,
    pub freq1: f64,
    pub xfr: f64,
    pub dir_offset: f64,
    pub dtmax: f64,
    pub dtxy: f64,
    pub dtkth: f64,
    pub dtmin: f64,
    pub zlim: f64,
    pub dmin: f64,
}

impl Default for Ww3Params {
    fn default() -> Self {
        Ww3Params {
            nth: 36,
            nk: 32,
            freq1: 0.0373,
            xfr: 1.1,
            dir_offset: 0.0,
            dtmax: 600.0,
            dtxy: 600.0,
            dtkth: 300.0,
            dtmin: 10.0,
            zlim: -0.1,
            dmin: 2.0,
        }
    }
}

fn default_ellipsoid() -> Ellipsoid {
    Ellipsoid::Wgs84
}
fn default_max_span() -> u32 {
    4
}
fn default_min_depth() -> f64 {
    5.0
}
fn default_wet_fraction() -> f64 {
    0.5
}
fn default_true() -> bool {
    true
}
fn default_levels() -> Vec<u8> {
    vec![1]
}

pub fn depth_of_nside(nside: u32) -> Result<u8> {
    if nside == 0 || !nside.is_power_of_two() || nside > (1 << 29) {
        bail!("nside {nside} is not a power of two in [1, 2^29]");
    }
    Ok(nside.trailing_zeros() as u8)
}

impl Config {
    pub fn load(path: &Path) -> Result<Config> {
        let txt = std::fs::read_to_string(path)?;
        let mut cfg: Config = serde_json::from_str(&txt)?;
        // relative paths are relative to the config file
        let base = path.parent().unwrap_or(Path::new("."));
        let fix = |p: &str| -> String {
            let pp = Path::new(p);
            if pp.is_absolute() { p.to_string() } else { base.join(pp).to_string_lossy().into_owned() }
        };
        cfg.bathy = fix(&cfg.bathy);
        cfg.output_dir = fix(&cfg.output_dir);
        if let CoastSource::Gshhg { path, .. } = &mut cfg.coast {
            *path = fix(path);
        }
        cfg.check()?;
        Ok(cfg)
    }

    pub fn check(&self) -> Result<()> {
        let dmin = depth_of_nside(self.nside_min)?;
        let dmax = depth_of_nside(self.nside_max)?;
        if dmax < dmin {
            bail!("nside_max < nside_min");
        }
        if (dmax - dmin) as u32 > self.max_level_span {
            bail!(
                "nside_max/nside_min = 2^{} exceeds max_level_span = {}",
                dmax - dmin,
                self.max_level_span
            );
        }
        for r in &self.refine {
            let d = depth_of_nside(r.nside)?;
            if d < dmin || d > dmax {
                bail!("refine rule nside {} outside [nside_min, nside_max]", r.nside);
            }
        }
        let d = &self.domain;
        if !(d.lon_min < d.lon_max && d.lat_min < d.lat_max) {
            bail!("empty domain");
        }
        Ok(())
    }

    pub fn dmin(&self) -> u8 {
        depth_of_nside(self.nside_min).unwrap()
    }
    pub fn dmax(&self) -> u8 {
        depth_of_nside(self.nside_max).unwrap()
    }
}
