//! End-to-end grid construction.

use crate::bathy::{cell_bathy, Bathy};
use crate::coast::CoastIndex;
use crate::config::{CoastSource, Config};
use crate::geo::{angle, healpix_to_geodetic_deg, lonlat_to_vec, normalize, Earth};
use crate::gshhg;
use crate::mesh::triangulate;
use crate::tree;
use crate::validate::{check, MeshReport};
use crate::ww3;
use anyhow::Result;
use cdshealpix::nested;
use rayon::prelude::*;
use serde::Serialize;
use std::path::Path;
use std::time::Instant;

/// Statistics of a grid build, printed as JSON by the `hww3-grid` program
/// and stored in `meta.json`.
#[derive(Debug, Serialize, Clone)]
pub struct Summary {
    /// grid name
    pub name: String,
    /// earth model used
    pub ellipsoid: String,
    /// coarsest nside
    pub nside_min: u32,
    /// finest nside
    pub nside_max: u32,
    /// coastline points kept in the index (after thinning)
    pub n_coast_points: usize,
    /// leaves of the refined and balanced tree (wet and dry)
    pub n_leaves: usize,
    /// leaves split by the 2:1 balance
    pub n_balance_splits: usize,
    /// wet leaves inside the domain
    pub n_wet_leaves: usize,
    /// connected components of the triangulation before keeping the largest
    pub n_components: usize,
    /// triangles dropped to remove pinch nodes
    pub n_pinch_tris_removed: usize,
    /// angle statistics of the triangulation before the quality step
    pub quality_before: QualityBefore,
    /// edge flips made by the Delaunay step
    pub n_flips: usize,
    /// bad boundary triangles removed (including pinch removal after trimming)
    pub n_trimmed: usize,
    /// nodes of the final mesh
    pub n_nodes: usize,
    /// open-boundary nodes (wet nodes touching a cell outside the domain)
    pub n_open_boundary: usize,
    /// node count per nside, from nside_min to nside_max
    pub nodes_per_nside: Vec<(u32, usize)>,
    /// validation of the final mesh
    pub mesh: MeshReport,
    /// wall time of each step (s)
    pub timings_s: Vec<(String, f64)>,
}

/// Angle statistics of the raw triangulation (before flips and trimming).
#[derive(Debug, Serialize, Clone)]
pub struct QualityBefore {
    /// smallest angle (degrees)
    pub min_angle_deg: f64,
    /// largest angle (degrees)
    pub max_angle_deg: f64,
    /// triangles with an angle below 20 degrees
    pub n_angle_below_20: usize,
    /// triangles with an angle above 120 degrees
    pub n_angle_above_120: usize,
}

/// The final grid: one entry per node, nodes in NESTED (Morton) order.
pub struct Grid {
    /// HEALPix NESTED uniq index of each node (`4 * 4^depth + ipix`)
    pub uniq: Vec<u64>,
    /// HEALPix depth of each node (`nside = 2^depth`)
    pub level: Vec<u8>,
    /// longitude (degrees, geodetic, in (-180, 180])
    pub lon: Vec<f64>,
    /// latitude (degrees, geodetic on the chosen ellipsoid)
    pub lat: Vec<f64>,
    /// mean water depth over the wet samples of the cell (m, positive down)
    pub depth: Vec<f64>,
    /// triangles, 0-based node indices, counter-clockwise
    pub tris: Vec<[u32; 3]>,
    /// 0-based indices of the open-boundary nodes
    pub obc: Vec<u32>,
}

/// Densify chains of (lon, lat) degrees into unit vectors on the HEALPix
/// sphere, with a maximum angular step `step` between consecutive points.
fn densify(chains: &[Vec<(f64, f64)>], earth: &Earth, step: f64) -> Vec<[f64; 3]> {
    chains
        .par_iter()
        .flat_map_iter(|c| {
            let v: Vec<[f64; 3]> = c
                .iter()
                .map(|&(lo, la)| lonlat_to_vec(lo.to_radians(), earth.to_authalic(la.to_radians())))
                .collect();
            let mut out = Vec::with_capacity(v.len());
            for k in 0..v.len() {
                let a = v[k];
                out.push(a);
                if k + 1 == v.len() {
                    break;
                }
                let b = v[k + 1];
                let n = (angle(&a, &b) / step).ceil() as usize;
                for s in 1..n {
                    let t = s as f64 / n as f64;
                    out.push(normalize([
                        a[0] * (1.0 - t) + b[0] * t,
                        a[1] * (1.0 - t) + b[1] * t,
                        a[2] * (1.0 - t) + b[2] * t,
                    ]));
                }
            }
            out
        })
        .collect()
}

fn coast_points(cfg: &Config, earth: &Earth, bathy: &Bathy) -> Result<Vec<[f64; 3]>> {
    let d = &cfg.domain;
    let m = 5.0;
    let (x0, x1, y0, y1) = (d.lon_min - m, d.lon_max + m, d.lat_min - m, d.lat_max + m);
    // densify to half the finest cell size
    let step = 0.5 * (4.0 * std::f64::consts::PI / (12.0 * (cfg.nside_max as f64).powi(2))).sqrt();
    match &cfg.coast {
        CoastSource::Bathy => Ok(bathy.coast_points(earth)),
        CoastSource::Gshhg { path, levels } => {
            let chains: Vec<Vec<(f64, f64)>> = gshhg::read(path, levels, x0, x1, y0, y1)?
                .into_iter()
                .map(|p| {
                    let mut c = p.points;
                    if let Some(&f) = c.first() {
                        c.push(f); // close the polygon
                    }
                    c
                })
                .collect();
            Ok(densify(&chains, earth, step))
        }
        CoastSource::Shapefile { path } => {
            let chains = crate::shapefile::read(path, x0, x1, y0, y1)?;
            Ok(densify(&chains, earth, step))
        }
    }
}

fn find(p: &mut [u32], mut i: u32) -> u32 {
    while p[i as usize] != i {
        p[i as usize] = p[p[i as usize] as usize];
        i = p[i as usize];
    }
    i
}

/// Build the grid described by `cfg` (all steps listed in the crate
/// documentation) and return it with its [`Summary`]. Nothing is written.
pub fn build(cfg: &Config) -> Result<(Grid, Summary)> {
    let mut timings = Vec::new();
    let mut t0 = Instant::now();
    let mut lap = |name: &str, t0: &mut Instant| {
        timings.push((name.to_string(), t0.elapsed().as_secs_f64()));
        eprintln!("[hww3-grid] {name}: {:.2}s", t0.elapsed().as_secs_f64());
        *t0 = Instant::now();
    };
    let earth = Earth::new(cfg.ellipsoid);
    let (dmin, dmax) = (cfg.dmin(), cfg.dmax());

    let bathy = Bathy::load(&cfg.bathy)?;
    lap("load bathymetry", &mut t0);

    let cpts = coast_points(cfg, &earth, &bathy)?;
    let bucket_depth = dmax.saturating_sub(2).max(1);
    let thin_depth = (dmax + 2).min(29);
    let coast = CoastIndex::new(&cpts, bucket_depth, thin_depth);
    drop(cpts);
    lap("coast index", &mut t0);

    let tr = tree::build(cfg, &earth, &coast);
    let leaves = tr.leaves;
    lap("refinement + 2:1 balance", &mut t0);

    let cb = cell_bathy(&bathy, &earth, &leaves, dmin, dmax);
    drop(bathy);
    lap("cell bathymetry", &mut t0);

    let n = leaves.len();
    let geo: Vec<([f64; 3], f64, f64)> = leaves
        .par_iter()
        .map(|&(d, h)| {
            let (lo, la) = nested::center(d, h);
            let (lon, lat) = healpix_to_geodetic_deg(&earth, lo, la);
            (lonlat_to_vec(lo, la), lon, lat)
        })
        .collect();
    let centres: Vec<[f64; 3]> = geo.iter().map(|g| g.0).collect();
    let inside: Vec<bool> = geo.iter().map(|g| cfg.domain.contains(g.1, g.2)).collect();
    let outside: Vec<bool> = inside.iter().map(|b| !b).collect();
    let wet: Vec<bool> = (0..n)
        .map(|i| {
            let c = &cb[i];
            let frac = if c.n > 0 { c.n_wet as f64 / c.n as f64 } else { c.n_wet as f64 };
            inside[i] && frac >= cfg.wet_fraction && c.mean_wet_depth >= cfg.min_depth_m
        })
        .collect();
    let n_wet = wet.iter().filter(|w| **w).count();

    let mut tri = triangulate(&leaves, &centres, &wet, &outside);
    // WW3 needs a single triangle fan around every node
    let n_pinch_tris_removed = crate::mesh::remove_pinches(n, &mut tri.tris);
    lap("triangulation", &mut t0);

    // triangle quality: Delaunay flips, then bad boundary triangles
    let q = &cfg.quality;
    let leaf_levels: Vec<u8> = leaves.iter().map(|l| l.0).collect();
    let before_quality = check(&centres, &leaf_levels, &tri.tris);
    let n_flips = if q.flips {
        crate::mesh::delaunay_flips(&centres, &leaf_levels, q.max_level_jump, &mut tri.tris)
    } else {
        0
    };
    let n_trimmed = if q.trim_boundary {
        crate::mesh::trim_boundary(
            n,
            &centres,
            q.min_angle_deg.to_radians(),
            q.max_angle_deg.to_radians(),
            &mut tri.tris,
        )
    } else {
        0
    };
    lap("quality (flips + boundary trim)", &mut t0);

    // connected components over triangle edges
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for t in &tri.tris {
        for k in 0..3 {
            let (a, b) = (find(&mut parent, t[k]), find(&mut parent, t[(k + 1) % 3]));
            if a != b {
                parent[a as usize] = b;
            }
        }
    }
    let mut in_tri = vec![false; n];
    for t in &tri.tris {
        for &v in t {
            in_tri[v as usize] = true;
        }
    }
    let mut comp_size = std::collections::HashMap::new();
    for i in 0..n {
        if in_tri[i] {
            *comp_size.entry(find(&mut parent, i as u32)).or_insert(0usize) += 1;
        }
    }
    let n_components = comp_size.len();
    let keep_root = comp_size.iter().max_by_key(|(_, s)| **s).map(|(r, _)| *r);
    let keep: Vec<bool> = (0..n)
        .map(|i| {
            in_tri[i]
                && (!cfg.keep_largest_component || Some(find(&mut parent, i as u32)) == keep_root)
        })
        .collect();

    // renumber (leaves are already in Morton order)
    let mut new_id = vec![u32::MAX; n];
    let mut grid = Grid {
        uniq: vec![],
        level: vec![],
        lon: vec![],
        lat: vec![],
        depth: vec![],
        tris: vec![],
        obc: vec![],
    };
    for i in 0..n {
        if keep[i] {
            new_id[i] = grid.uniq.len() as u32;
            let (d, h) = leaves[i];
            grid.uniq.push(tree::uniq(d, h));
            grid.level.push(d);
            grid.lon.push(geo[i].1);
            grid.lat.push(geo[i].2);
            grid.depth.push(cb[i].mean_wet_depth.max(cfg.ww3.dmin));
            if tri.touches_outside[i] {
                grid.obc.push(new_id[i]);
            }
        }
    }
    grid.tris = tri
        .tris
        .iter()
        .filter(|t| keep[t[0] as usize])
        .map(|t| [new_id[t[0] as usize], new_id[t[1] as usize], new_id[t[2] as usize]])
        .collect();

    let kept_centres: Vec<[f64; 3]> = (0..n).filter(|&i| keep[i]).map(|i| centres[i]).collect();
    let report = check(&kept_centres, &grid.level, &grid.tris);
    lap("components + validation", &mut t0);

    let mut per = Vec::new();
    for d in dmin..=dmax {
        per.push((1u32 << d, grid.level.iter().filter(|&&l| l == d).count()));
    }
    let summary = Summary {
        name: cfg.name.clone(),
        ellipsoid: format!("{:?}", cfg.ellipsoid),
        nside_min: cfg.nside_min,
        nside_max: cfg.nside_max,
        n_coast_points: coast.n_points,
        n_leaves: n,
        n_balance_splits: tr.n_balance_splits,
        n_wet_leaves: n_wet,
        n_components,
        n_pinch_tris_removed,
        quality_before: QualityBefore {
            min_angle_deg: before_quality.min_angle_deg,
            max_angle_deg: before_quality.max_angle_deg,
            n_angle_below_20: before_quality.n_angle_below_20,
            n_angle_above_120: before_quality.n_angle_above_120,
        },
        n_flips,
        n_trimmed,
        n_nodes: grid.uniq.len(),
        n_open_boundary: grid.obc.len(),
        nodes_per_nside: per,
        mesh: report,
        timings_s: timings,
    };
    Ok((grid, summary))
}

/// Write the grid to `cfg.output_dir`: `<name>.msh`, `<name>.obc`,
/// `ww3_grid.nml`, `namelists_<name>.nml`, the raw arrays in `raw/` and
/// `meta.json` (configuration, summary and description of the arrays).
pub fn write(cfg: &Config, grid: &Grid, summary: &Summary) -> Result<()> {
    let out = Path::new(&cfg.output_dir);
    std::fs::create_dir_all(out)?;
    let n = &cfg.name;
    ww3::write_msh(&out.join(format!("{n}.msh")), &grid.lon, &grid.lat, &grid.depth, &grid.tris)?;
    ww3::write_obc(&out.join(format!("{n}.obc")), &grid.obc)?;
    ww3::write_grid_nml(&out.join("ww3_grid.nml"), cfg)?;
    ww3::write_namelists(&out.join(format!("namelists_{n}.nml")))?;
    let raw = out.join("raw");
    std::fs::create_dir_all(&raw)?;
    ww3::write_raw(&raw.join("uniq.u64"), &grid.uniq)?;
    ww3::write_raw(&raw.join("level.u8"), &grid.level)?;
    ww3::write_raw(&raw.join("lon.f64"), &grid.lon)?;
    ww3::write_raw(&raw.join("lat.f64"), &grid.lat)?;
    ww3::write_raw(&raw.join("depth.f64"), &grid.depth)?;
    let flat: Vec<u32> = grid.tris.iter().flatten().copied().collect();
    ww3::write_raw(&raw.join("tri.u32"), &flat)?;
    ww3::write_raw(&raw.join("obc.u32"), &grid.obc)?;
    let meta = serde_json::json!({
        "config": cfg,
        "summary": summary,
        "arrays": {
            "uniq.u64": "HEALPix NESTED uniq index of each node (4*4^depth + ipix)",
            "level.u8": "HEALPix depth of each node (nside = 2^depth)",
            "lon.f64": "longitude (deg, geodetic)",
            "lat.f64": "latitude (deg, geodetic on the chosen ellipsoid)",
            "depth.f64": "mean water depth over wet samples (m, positive down)",
            "tri.u32": "triangles, 3 x 0-based node indices, counter-clockwise",
            "obc.u32": "0-based open-boundary node indices"
        }
    });
    std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    Ok(())
}
