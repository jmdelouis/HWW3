//! JSON configuration of a grid.
//!
//! A configuration file describes the domain, the HEALPix resolution range,
//! the refinement rules, the bathymetry and coastline sources, and the
//! WAVEWATCH III parameters written to `ww3_grid.nml`. Relative paths are
//! resolved against the directory of the configuration file.
//!
//! ```json
//! {
//!   "name": "natl",
//!   "domain": {"lon_min": -100, "lon_max": 45, "lat_min": 0, "lat_max": 80},
//!   "nside_min": 128, "nside_max": 2048,
//!   "refine": [{"max_dist_km": 200, "nside": 512},
//!              {"max_dist_km": 50, "nside": 1024},
//!              {"max_dist_km": 10, "nside": 2048}],
//!   "bathy": "../data/gebco/natl/meta.json",
//!   "coast": {"source": "shapefile", "path": "../data/gshhs_land.shp"},
//!   "output_dir": "../grids/natl"
//! }
//! ```

use crate::geo::Ellipsoid;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Complete description of a grid, read from JSON with [`Config::load`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Config {
    /// Grid name, used for the output file names (`<name>.msh`, ...).
    pub name: String,
    /// Geographic box of the model domain; cells whose centre is outside are
    /// not wet, and wet cells touching them become open-boundary nodes.
    pub domain: Domain,
    /// Earth model: `"wgs84"` (default, authalic-latitude conversion) or
    /// `"sphere"`.
    #[serde(default = "default_ellipsoid")]
    pub ellipsoid: Ellipsoid,
    /// Coarsest HEALPix nside (open ocean); a power of two.
    pub nside_min: u32,
    /// Finest HEALPix nside (coast); a power of two.
    pub nside_max: u32,
    /// Maximum number of refinement levels between `nside_min` and
    /// `nside_max` (default 4).
    #[serde(default = "default_max_span")]
    pub max_level_span: u32,
    /// Refinement from the bathymetry (depth and slope criteria, see
    /// [`crate::refine`]).
    #[serde(default)]
    pub refine_bathy: Option<BathyRefine>,
    /// Optional refinement by distance to the coast: a cell closer than
    /// `max_dist_km` to the coast is refined up to `nside`. The finest
    /// applicable rule wins. Empty by default.
    #[serde(default)]
    pub refine: Vec<Rule>,
    /// Cells whose mean wet depth is shallower than this (m) are dry
    /// (default 5).
    #[serde(default = "default_min_depth")]
    pub min_depth_m: f64,
    /// Minimum fraction of wet bathymetry samples for a cell to be wet
    /// (default 0.5).
    #[serde(default = "default_wet_fraction")]
    pub wet_fraction: f64,
    /// Keep only the largest connected wet region (default true).
    #[serde(default = "default_true")]
    pub keep_largest_component: bool,
    /// Path of the bathymetry `meta.json` (see [`crate::bathy::BathyMeta`]).
    pub bathy: String,
    /// Coastline used by the distance refinement `refine` (default: zero
    /// crossings of the bathymetry; not read when `refine` is empty).
    #[serde(default = "default_coast")]
    pub coast: CoastSource,
    /// Directory where the grid files are written.
    pub output_dir: String,
    /// WAVEWATCH III parameters for `ww3_grid.nml`.
    #[serde(default)]
    pub ww3: Ww3Params,
    /// Triangle-quality control.
    #[serde(default)]
    pub quality: Quality,
}

/// Triangle-quality control applied after the triangulation.
///
/// The nodes are fixed HEALPix centres, so angles are improved by changing
/// the connectivity only: edge flips (max-min angle, i.e. Delaunay) inside
/// the mesh, then removal of bad triangles on the boundary. If `enforce` is
/// true, the program fails when a triangle angle is still outside
/// [`min_angle_deg`, `max_angle_deg`].
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Quality {
    /// Smallest accepted triangle angle (degrees).
    pub min_angle_deg: f64,
    /// Largest accepted triangle angle (degrees).
    pub max_angle_deg: f64,
    /// Apply Delaunay edge flips (default true).
    pub flips: bool,
    /// Largest HEALPix level difference allowed across a flipped edge
    /// (default 1, as in the 2:1 balanced tree).
    pub max_level_jump: u8,
    /// Remove boundary triangles outside the angle bounds (default true).
    pub trim_boundary: bool,
    /// Fail if an angle is still outside the bounds (default true).
    pub enforce: bool,
}

impl Default for Quality {
    fn default() -> Self {
        Quality {
            min_angle_deg: 20.0,
            max_angle_deg: 130.0,
            flips: true,
            max_level_jump: 1,
            trim_boundary: true,
            enforce: true,
        }
    }
}

/// Longitude/latitude box in geodetic degrees, longitudes in (-180, 180].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Domain {
    /// Western edge (degrees).
    pub lon_min: f64,
    /// Eastern edge (degrees).
    pub lon_max: f64,
    /// Southern edge (degrees).
    pub lat_min: f64,
    /// Northern edge (degrees).
    pub lat_max: f64,
}

impl Domain {
    /// True if (`lon`, `lat`), geodetic degrees with lon in (-180, 180], is
    /// inside the box.
    pub fn contains(&self, lon: f64, lat: f64) -> bool {
        lat >= self.lat_min && lat <= self.lat_max && lon >= self.lon_min && lon <= self.lon_max
    }

    /// Same as [`Domain::contains`] with the box enlarged by `margin` degrees
    /// of arc (the longitude margin is divided by cos(lat), capped at 85°).
    pub fn contains_with_margin(&self, lon: f64, lat: f64, margin: f64) -> bool {
        let c = lat.abs().min(85.0).to_radians().cos();
        lat >= self.lat_min - margin
            && lat <= self.lat_max + margin
            && lon >= self.lon_min - margin / c
            && lon <= self.lon_max + margin / c
    }
}

/// Refinement rule: cells closer than `max_dist_km` to the coast are refined
/// to at least `nside`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Rule {
    /// Distance threshold from the coast (km), measured from the cell edge.
    pub max_dist_km: f64,
    /// Target nside, within [`Config::nside_min`, `Config::nside_max`].
    pub nside: u32,
}

/// Refinement from the bathymetry with the mesh-size functions of
/// OceanMesh2D (see [`crate::refine`]): required cell size
/// `min(T·sqrt(g h)/wl, 2π h/(slp·|∇h|))` with a low pass `fl` of the
/// bathymetry before the gradient, then the grade limit `grade`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BathyRefine {
    /// Cells per shallow-water M2 wavelength, `T_M2·sqrt(g h)/wl`
    /// (OceanMesh2D `wl`); used when `wave_time_s` is 0.
    #[serde(default)]
    pub wl: f64,
    /// `T/wl` in seconds, overriding `wl` when > 0: cell size
    /// `wave_time_s·sqrt(g h)`.
    #[serde(default)]
    pub wave_time_s: f64,
    /// Cells per topographic length scale `2π h/|∇h|` (OceanMesh2D `slp`);
    /// 0 disables the slope criterion.
    #[serde(default, alias = "slope_points")]
    pub slp: f64,
    /// Low pass of the bathymetry before the gradient (OceanMesh2D `fl`):
    /// 0 none (default), > 0 wavelength in km, < 0 barotropic Rossby radius
    /// divided by `|fl|`.
    #[serde(default)]
    pub fl: f64,
    /// Maximum growth of the size field per unit distance (OceanMesh2D `g`,
    /// whose default is 0.2); 0 disables the limit (default).
    #[serde(default)]
    pub grade: f64,
    /// The slope criterion applies only shallower than this (m); 0 = at all
    /// depths (default).
    #[serde(default)]
    pub slope_max_depth_m: f64,
    /// Floor of the depth used by the size functions (m, default 1 as in
    /// OceanMesh2D); land samples count as this depth.
    #[serde(default = "default_one")]
    pub min_depth_m: f64,
    /// Size of the blocks on which the bathymetry is averaged before the
    /// size functions (km; default 0 = half the finest cell size).
    #[serde(default)]
    pub smooth_km: f64,
}

/// Source of the coastline, selected by the `"source"` key.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "source", rename_all = "lowercase")]
pub enum CoastSource {
    /// GSHHG native binary file (e.g. `gshhs_f.b`).
    Gshhg {
        /// Path of the `.b` file.
        path: String,
        /// Polygon levels used as coastline (default `[1]`, land/ocean).
        #[serde(default = "default_levels")]
        levels: Vec<u8>,
    },
    /// ESRI shapefile of coastline polylines or land polygons in geographic
    /// WGS84 coordinates (e.g. `gshhs_land.shp`).
    Shapefile {
        /// Path of the `.shp` file.
        path: String,
    },
    /// Coastline = zero crossings of the bathymetry grid.
    Bathy,
}

/// WAVEWATCH III parameters written to `ww3_grid.nml`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Ww3Params {
    /// Number of directions (`SPECTRUM%NTH`, default 36).
    pub nth: u32,
    /// Number of frequencies (`SPECTRUM%NK`, default 32).
    pub nk: u32,
    /// First frequency in Hz (`SPECTRUM%FREQ1`, default 0.0373).
    pub freq1: f64,
    /// Frequency increment factor (`SPECTRUM%XFR`, default 1.1).
    pub xfr: f64,
    /// Relative offset of the first direction (`SPECTRUM%THOFF`, default 0).
    pub dir_offset: f64,
    /// Maximum global time step in s (`TIMESTEPS%DTMAX`, default 600).
    pub dtmax: f64,
    /// Maximum spatial-propagation time step in s (`TIMESTEPS%DTXY`, default 600).
    pub dtxy: f64,
    /// Maximum spectral-propagation time step in s (`TIMESTEPS%DTKTH`, default 300).
    pub dtkth: f64,
    /// Minimum source-term time step in s (`TIMESTEPS%DTMIN`, default 10).
    pub dtmin: f64,
    /// Coastline limit depth in m (`GRID%ZLIM`, default -0.1).
    pub zlim: f64,
    /// Minimum water depth in m (`GRID%DMIN`, default 2); also the floor of
    /// the depths written to the mesh.
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
fn default_one() -> f64 {
    1.0
}
fn default_coast() -> CoastSource {
    CoastSource::Bathy
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

/// HEALPix depth of an nside (`nside = 2^depth`); errors if `nside` is not a
/// power of two in [1, 2^29].
///
/// ```
/// assert_eq!(hww3_grid::config::depth_of_nside(2048).unwrap(), 11);
/// assert!(hww3_grid::config::depth_of_nside(1000).is_err());
/// ```
pub fn depth_of_nside(nside: u32) -> Result<u8> {
    if nside == 0 || !nside.is_power_of_two() || nside > (1 << 29) {
        bail!("nside {nside} is not a power of two in [1, 2^29]");
    }
    Ok(nside.trailing_zeros() as u8)
}

impl Config {
    /// Read a JSON configuration, resolve relative paths against the
    /// directory of the file, and [check](Config::check) it.
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
        match &mut cfg.coast {
            CoastSource::Gshhg { path, .. } | CoastSource::Shapefile { path } => *path = fix(path),
            CoastSource::Bathy => {}
        }
        cfg.check()?;
        Ok(cfg)
    }

    /// Check the nsides (powers of two, `nside_min <= nside_max`, span at
    /// most `max_level_span` levels), the rule nsides and the domain.
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
        if let Some(p) = &self.refine_bathy {
            if !(p.wave_time_s > 0.0 || p.wl > 0.0) || p.slp < 0.0 || p.grade < 0.0 || !(p.min_depth_m > 0.0) {
                bail!("refine_bathy: wl or wave_time_s must be > 0, min_depth_m > 0, slp and grade >= 0");
            }
        }
        let d = &self.domain;
        if !(d.lon_min < d.lon_max && d.lat_min < d.lat_max) {
            bail!("empty domain");
        }
        Ok(())
    }

    /// HEALPix depth of `nside_min`.
    pub fn dmin(&self) -> u8 {
        depth_of_nside(self.nside_min).unwrap()
    }
    /// HEALPix depth of `nside_max`.
    pub fn dmax(&self) -> u8 {
        depth_of_nside(self.nside_max).unwrap()
    }
}
