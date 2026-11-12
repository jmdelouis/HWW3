//! End-to-end tests on synthetic bathymetries.

use hww3_grid::config::{CoastSource, Config, Domain, Quality, Rule, Ww3Params};
use hww3_grid::geo::Ellipsoid;
use hww3_grid::pipeline;
use std::path::PathBuf;

/// Write a bathymetry grid on [lon0, lon1] x [lat0, lat1] with step `step`
/// and elevation `f(lon, lat)`, return the meta path.
fn write_bathy(
    name: &str,
    lon0: f64,
    lon1: f64,
    lat0: f64,
    lat1: f64,
    step: f64,
    f: impl Fn(f64, f64) -> f32,
) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hww3_test_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    let nlon = ((lon1 - lon0) / step).round() as usize + 1;
    let nlat = ((lat1 - lat0) / step).round() as usize + 1;
    let mut bytes = Vec::with_capacity(4 * nlon * nlat);
    for j in 0..nlat {
        for i in 0..nlon {
            let z = f(lon0 + i as f64 * step, lat0 + j as f64 * step);
            bytes.extend_from_slice(&z.to_le_bytes());
        }
    }
    std::fs::write(dir.join("elev.f32"), bytes).unwrap();
    let meta = serde_json::json!({
        "nlon": nlon, "nlat": nlat, "lon0": lon0, "lat0": lat0,
        "dlon": step, "dlat": step, "file": "elev.f32"
    });
    let p = dir.join("meta.json");
    std::fs::write(&p, meta.to_string()).unwrap();
    p
}

fn config(name: &str, bathy: &PathBuf, domain: Domain, nmin: u32, nmax: u32, rules: Vec<Rule>) -> Config {
    Config {
        name: name.into(),
        domain,
        ellipsoid: Ellipsoid::Wgs84,
        nside_min: nmin,
        nside_max: nmax,
        max_level_span: 4,
        refine: rules,
        min_depth_m: 5.0,
        wet_fraction: 0.5,
        keep_largest_component: true,
        bathy: bathy.to_string_lossy().into(),
        coast: CoastSource::Bathy,
        output_dir: std::env::temp_dir().join(format!("hww3_out_{name}")).to_string_lossy().into(),
        ww3: Ww3Params::default(),
        quality: Quality::default(),
    }
}

/// Uniform grid across a HEALPix base-face corner (lon 45, lat 41.8): the
/// mesh must be a disc (Euler characteristic 1) with consistent orientation.
#[test]
fn uniform_open_ocean() {
    let b = write_bathy("uniform", 20.0, 70.0, 20.0, 65.0, 0.1, |_, _| -3000.0);
    let d = Domain { lon_min: 30.0, lon_max: 60.0, lat_min: 30.0, lat_max: 55.0 };
    let cfg = config("uniform", &b, d, 64, 64, vec![]);
    let (grid, s) = pipeline::build(&cfg).unwrap();
    println!("{}", serde_json::to_string_pretty(&s).unwrap());
    assert!(s.mesh.is_valid());
    assert_eq!(s.mesh.euler, 1);
    assert_eq!(s.mesh.n_isolated_nodes, 0);
    assert_eq!(s.mesh.max_level_jump, 0);
    // ~2 triangles per node in a uniform grid
    let ratio = grid.tris.len() as f64 / grid.uniq.len() as f64;
    assert!(ratio > 1.8 && ratio < 2.0, "{ratio}");
    assert!(s.n_open_boundary > 0);
    pipeline::write(&cfg, &grid, &s).unwrap();
}

/// Multi-resolution grid around a circular island sitting on the base-face
/// corner: one hole (Euler characteristic 0), 2:1 transitions only.
#[test]
fn island_multiscale() {
    let (cx, cy) = (45.0f64, 41.81f64);
    let b = write_bathy("island", 20.0, 70.0, 20.0, 65.0, 0.02, move |lon, lat| {
        let dx = (lon - cx) * lat.to_radians().cos();
        let dy = lat - cy;
        let r = (dx * dx + dy * dy).sqrt();
        if r < 2.0 {
            (300.0 * (1.0 - r / 2.0)) as f32 + 1.0
        } else {
            (-4000.0 * ((r - 2.0) / 3.0).min(1.0) - 10.0) as f32
        }
    });
    let d = Domain { lon_min: 30.0, lon_max: 60.0, lat_min: 30.0, lat_max: 55.0 };
    let rules = vec![
        Rule { max_dist_km: 30.0, nside: 512 },
        Rule { max_dist_km: 100.0, nside: 256 },
        Rule { max_dist_km: 250.0, nside: 128 },
    ];
    let cfg = config("island", &b, d, 32, 512, rules);
    let (grid, s) = pipeline::build(&cfg).unwrap();
    println!("{}", serde_json::to_string_pretty(&s).unwrap());
    assert!(s.mesh.is_valid());
    assert_eq!(s.mesh.euler, 0, "one island -> one hole");
    assert_eq!(s.mesh.n_isolated_nodes, 0);
    assert_eq!(s.mesh.max_level_jump, 1);
    // every level is used
    for (nside, count) in &s.nodes_per_nside {
        assert!(*count > 0, "no node at nside {nside}");
    }
    // the quality step brings every angle within the default bounds
    assert!(s.quality_before.n_angle_below_20 + s.quality_before.n_angle_above_120 > 0);
    assert!(s.n_flips > 0);
    assert!(s.mesh.angles_within(cfg.quality.min_angle_deg, cfg.quality.max_angle_deg));
    pipeline::write(&cfg, &grid, &s).unwrap();
}

/// Ear clipping keeps all triangles counter-clockwise.
#[test]
fn ear_clip_square() {
    let pts = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.7, -0.7]];
    let t = hww3_grid::mesh::ear_clip(&pts);
    assert_eq!(t.len(), 3);
}

fn unit(lon: f64, lat: f64) -> [f64; 3] {
    hww3_grid::geo::lonlat_to_vec(lon.to_radians(), lat.to_radians())
}

/// A flat quadrilateral split along its long diagonal is flipped to the
/// short diagonal, which raises the smallest angle.
#[test]
fn delaunay_flip_short_diagonal() {
    use hww3_grid::mesh::{delaunay_flips, tri_angles};
    // u(0) and v(1) far apart along the equator, w(2) and x(3) close above/below
    let nodes = [unit(-1.0, 0.0), unit(1.0, 0.0), unit(0.0, 0.2), unit(0.0, -0.2)];
    let levels = [0u8; 4];
    // (u, v, w) and (v, u, x), both counter-clockwise
    let mut tris = vec![[0u32, 1, 2], [1, 0, 3]];
    let min_angle = |t: &[[u32; 3]]| {
        t.iter()
            .flat_map(|t| tri_angles(&nodes[t[0] as usize], &nodes[t[1] as usize], &nodes[t[2] as usize]))
            .fold(f64::MAX, f64::min)
    };
    let before = min_angle(&tris);
    assert_eq!(delaunay_flips(&nodes, &levels, 1, &mut tris), 1);
    let after = min_angle(&tris);
    assert!(after > before, "{before} -> {after}");
    // the new edge is (w, x)
    assert!(tris.iter().all(|t| t.contains(&2) && t.contains(&3)));
    // a second call does nothing
    assert_eq!(delaunay_flips(&nodes, &levels, 1, &mut tris), 0);
}

/// A sliver triangle on the boundary is removed; good triangles are kept.
#[test]
fn trim_boundary_sliver() {
    use hww3_grid::mesh::trim_boundary;
    // a good triangle (0,1,2) and a sliver (1,3,2) glued along edge (1,2)
    let nodes = [unit(0.0, 0.0), unit(1.0, 0.0), unit(0.5, 0.8), unit(4.0, 0.45)];
    let mut tris = vec![[0u32, 1, 2], [1, 3, 2]];
    let n = trim_boundary(4, &nodes, 20f64.to_radians(), 120f64.to_radians(), &mut tris);
    assert_eq!(n, 1);
    assert_eq!(tris, vec![[0, 1, 2]]);
}

#[test]
fn bow_tie_pinch_removed() {
    use hww3_grid::mesh::{fan_counts, remove_pinches};
    // two fans around vertex 0: {0,1,2},{0,2,3} and {0,4,5}
    let mut tris = vec![[0u32, 1, 2], [0, 2, 3], [0, 4, 5]];
    assert_eq!(fan_counts(6, &tris)[0], 2);
    let removed = remove_pinches(6, &mut tris);
    assert_eq!(removed, 1);
    assert_eq!(tris, vec![[0, 1, 2], [0, 2, 3]]);
    assert!(fan_counts(6, &tris).iter().all(|&c| c <= 1));
}
