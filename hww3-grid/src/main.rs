//! `hww3-grid` command-line program.
//!
//! - `hww3-grid <config.json>`: build and write the grid, print the summary
//!   as JSON on stdout, exit with an error if the mesh is invalid or if a
//!   triangle angle is outside the bounds of `quality` (when enforced);
//! - `hww3-grid cells <grid_dir>`: write the HEALPix cell vertices of every
//!   node (for plots).

use anyhow::{bail, Result};
use cdshealpix::nested;
use hww3_grid::config::Config;
use hww3_grid::geo::{healpix_to_geodetic_deg, Earth};
use hww3_grid::{pipeline, ww3};
use rayon::prelude::*;
use std::path::Path;

/// `hww3-grid cells <grid_dir>`: write the 4 vertices (S, E, N, W) of every
/// node's HEALPix cell, geodetic degrees, as raw/cell_lon.f64 and
/// raw/cell_lat.f64 (n x 4).
fn cells(dir: &Path) -> Result<()> {
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)?;
    let cfg: Config = serde_json::from_value(meta["config"].clone())?;
    let earth = Earth::new(cfg.ellipsoid);
    let bytes = std::fs::read(dir.join("raw/uniq.u64"))?;
    let uniq: Vec<u64> = bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    let v: Vec<[(f64, f64); 4]> = uniq
        .par_iter()
        .map(|&u| {
            let (d, h) = nested::from_uniq(u);
            nested::vertices(d, h).map(|(lo, la)| healpix_to_geodetic_deg(&earth, lo, la))
        })
        .collect();
    let lon: Vec<f64> = v.iter().flat_map(|c| c.iter().map(|p| p.0)).collect();
    let lat: Vec<f64> = v.iter().flat_map(|c| c.iter().map(|p| p.1)).collect();
    ww3::write_raw(&dir.join("raw/cell_lon.f64"), &lon)?;
    ww3::write_raw(&dir.join("raw/cell_lat.f64"), &lat)?;
    eprintln!("wrote vertices of {} cells", uniq.len());
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 3 && args[1] == "cells" {
        return cells(Path::new(&args[2]));
    }
    if args.len() != 2 {
        bail!("usage: hww3-grid <config.json> | hww3-grid cells <grid_dir>");
    }
    let cfg = Config::load(Path::new(&args[1]))?;
    let (grid, summary) = pipeline::build(&cfg)?;
    pipeline::write(&cfg, &grid, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if !summary.mesh.is_valid() {
        bail!("mesh validation failed");
    }
    let q = &cfg.quality;
    if q.enforce && !summary.mesh.angles_within(q.min_angle_deg, q.max_angle_deg) {
        bail!(
            "triangle angles {:.1}..{:.1} deg outside the accepted range {}..{} deg",
            summary.mesh.min_angle_deg,
            summary.mesh.max_angle_deg,
            q.min_angle_deg,
            q.max_angle_deg
        );
    }
    Ok(())
}
