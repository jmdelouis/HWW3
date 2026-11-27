//! Preprocessing of an HWW3 run in Rust, in place of the WW3 tools.
//!
//! ```text
//! hww3-pre prnc <grid dir> --field wind|current|level|ice --input DIR --out FILE.ww3
//!     [--msh FILE.msh] [--start YYYYMMDDHHMMSS] [--stop YYYYMMDDHHMMSS]
//!     The forcing file of ww3_prnc (hww3_wave::prnc): the netCDF fields of
//!     DIR (written by scripts/prnc_nc.py) interpolated on the nodes of the
//!     grid. With --msh the node coordinates are those of the gmsh file, as
//!     WW3 reads them, instead of the grid's raw arrays (they must agree to
//!     1e-5 degree).
//!     On a mesh of a WW3 structured grid in rotated coordinates (hww3-pre
//!     struct), the fields are interpolated at the geographic positions of
//!     the nodes and vectors turned into the rotated frame.
//! hww3-pre cmp-fields A.ww3 B.ww3
//!     Compare two forcing files time by time: number of values that
//!     differ and largest difference, per field, over the times of A found
//!     in B.
//! hww3-pre bounc <grid dir> --input DIR --out nest.ww3 [--msh FILE.msh] [--interp 2]
//!     The boundary file of ww3_bounc (hww3_wave::bounc): the station
//!     spectra of DIR (written by scripts/bounc_nc.py), and for each
//!     open-boundary node its two nearest stations and their weights.
//! hww3-pre cmp-nest A.ww3 B.ww3
//!     Compare two boundary files: header, boundary points, indices and
//!     weights, and the spectra over the times of A found in B.
//! hww3-pre struct WW3_GRID.nml --data DIR --out GRID_DIR [--rotate auto|none|north|south] [--nml NAMELISTS]
//!     A WW3 structured grid (RECT or CURV, files in DIR) triangulated into
//!     an HWW3 grid directory (hww3_wave::structgrid), with the namelist
//!     copied as GRID_DIR/ww3_grid.nml. The default rotation (auto) moves
//!     the pole onto the equator when a sea point lies within 10° of it.
//!     With --nml (the namelists of the run), also the reflection
//!     geometry of W3SETREF per node (raw/ref1_reflc.f64, ref1_refld.u32,
//!     ref1_delx.f64, ref1_dely.f64, ref1_trnx.f64, ref1_trny.f64), read by
//!     hww3-prop run --ice-ref.
//! hww3-pre struct WW3_GRID.nml --data DIR --out GRID_DIR --scheme uq --nml NAMELISTS
//!     The grid directory of the UQ scheme of WW3 (hww3_wave::uqgrid): every
//!     sea point of the structured grid in WW3's order as a grid without
//!     triangles, the structured grid itself (coordinates, bottom, MAPSTA,
//!     obstructions) and the namelists, for hww3-prop run with the
//!     geographic propagation of W3XYP3 (hww3_wave::prop_uq); with the
//!     reflection geometry of W3SETREF per sea point.
//! hww3-pre restart-uq <mesh dir> --restart IN.ww3 --uq-grid UQ_DIR --out OUT.ww3
//!     A restart of hww3-prop on the mesh of a structured grid (hww3-pre
//!     struct) written for the grid directory of the UQ scheme of the same
//!     grid: one spectrum per sea point in WW3's order, the points left
//!     out of the mesh with zero spectra, levels and ice; MAPSTA that of
//!     the structured grid (1 or 2) with the sign of the node's.
//! hww3-pre restart-struct <grid dir> --restart IN.ww3 --out OUT.ww3 --grid-nml WW3_GRID.nml --data DIR
//!     A restart of hww3-prop on the mesh of a structured grid (hww3-pre
//!     struct) written in WW3's order of the sea points of the structured
//!     grid (raw/sea_ij.u32), for WW3 programs run on that grid: the points
//!     left out of the mesh get zero spectra, levels and ice, u* 1e-5 and
//!     ASF 1; MAPSTA is that of the nodes (that of the grid, 1 or 2, for the points left out),
//!     written over the whole NX x NY grid as W3IORS does on structured
//!     grids (0 on land, MAPST2 = 1 on the excluded points), from the
//!     structured grid of --grid-nml and --data.
//! hww3-pre ideal-wind <grid dir> --out wind.ww3 --start YYYYMMDDHHMMSS --hours H [--speed U]
//!     An idealised hourly wind on the nodes, for runs without forcing:
//!     westerlies and trade winds of amplitude U (default 10 m/s) with
//!     waves of 3 lows around the globe moving east at 15° per day, turned
//!     into the rotated frame of a rotated mesh.
//! ```

use anyhow::{bail, Context, Result};
use hww3_wave::grid::Grid;
use hww3_wave::bounc::{boundary_nodes, station_weights, RawSpectra};
use hww3_wave::prnc::{Field, RawInput};
use hww3_wave::structgrid::{rotate_vector, Mesh, Rotation, StructGrid};
use hww3_wave::ww3io::{FieldFile, FieldWriter, NestFile, NestWriter, Time};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// `GTYPE` of an unstructured grid (`UNGTYPE` of `W3GDATMD`).
const UNGTYPE: i32 = 3;

struct Args {
    cmd: String,
    pos: Vec<String>,
    opts: HashMap<String, String>,
}

fn parse() -> Result<Args> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.is_empty() {
        bail!("usage: hww3-pre prnc|cmp-fields ... (see the source header)");
    }
    let mut pos = vec![];
    let mut opts = HashMap::new();
    let mut i = 1;
    while i < a.len() {
        if a[i].starts_with("--") {
            let v = a.get(i + 1).with_context(|| format!("{} needs a value", a[i]))?;
            opts.insert(a[i].clone(), v.clone());
            i += 2;
        } else {
            pos.push(a[i].clone());
            i += 1;
        }
    }
    Ok(Args { cmd: a[0].clone(), pos, opts })
}

/// Node coordinates of a gmsh 2 file (`$Nodes`), read in double precision
/// as `W3TRIAMD` reads them.
fn msh_nodes(p: &Path) -> Result<(Vec<f64>, Vec<f64>)> {
    let text = std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
    let mut lines = text.lines();
    while let Some(l) = lines.next() {
        if l.trim() == "$Nodes" {
            let n: usize = lines.next().context("msh: node count")?.trim().parse()?;
            let (mut x, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
            for _ in 0..n {
                let l = lines.next().context("msh: end of file in $Nodes")?;
                let mut it = l.split_whitespace().skip(1);
                x.push(it.next().context("msh: x")?.parse::<f64>()?);
                y.push(it.next().context("msh: y")?.parse::<f64>()?);
            }
            return Ok((x, y));
        }
    }
    bail!("{}: no $Nodes section", p.display())
}

/// Node coordinates: those of the gmsh file of `--msh` (checked against
/// the grid's) or the grid's raw arrays.
fn node_coordinates(a: &Args, grid: &Grid) -> Result<(Vec<f64>, Vec<f64>)> {
    Ok(match a.opts.get("--msh") {
        Some(m) => {
            let (x, y) = msh_nodes(Path::new(m))?;
            anyhow::ensure!(x.len() == grid.lon.len(), "{m}: {} nodes, grid {}", x.len(), grid.lon.len());
            let d = x.iter().zip(&grid.lon).chain(y.iter().zip(&grid.lat)).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            anyhow::ensure!(d < 1e-5, "{m}: coordinates differ from the grid's by {d:.2e} degree");
            println!("node coordinates of {m} (within {d:.1e} degree of the grid's)");
            (x, y)
        }
        None => (grid.lon.clone(), grid.lat.clone()),
    })
}

fn prnc(a: &Args) -> Result<()> {
    let grid_dir = PathBuf::from(a.pos.first().context("prnc needs the grid directory")?);
    let field = Field::parse(a.opts.get("--field").context("prnc needs --field")?)?;
    let inp = RawInput::open(Path::new(a.opts.get("--input").context("prnc needs --input")?))?;
    let out = PathBuf::from(a.opts.get("--out").context("prnc needs --out")?);
    let start = a.opts.get("--start").map(|s| Time::parse(s)).transpose()?;
    let stop = a.opts.get("--stop").map(|s| Time::parse(s)).transpose()?;
    anyhow::ensure!(inp.vars.len() == field.ncomp(), "{} variables in the input for {} components", inp.vars.len(), field.ncomp());
    let t0 = Instant::now();
    let grid = Grid::read(&grid_dir)?;
    // a mesh of a structured grid in rotated coordinates: interpolate at the
    // geographic positions, then turn the vectors into the rotated frame
    let (lon_geo, lat_geo, angle) = Mesh::geographic(&grid_dir, &grid)?;
    let rotated = angle.iter().any(|&v| v != 0.0);
    let (x, y) = if rotated && !a.opts.contains_key("--msh") { (lon_geo, lat_geo) } else { node_coordinates(a, &grid)? };
    let g = &inp.grid;
    let w = g.weights(&x, &y);
    println!(
        "{}: source grid {} x {}, longitudes {} to {}, latitudes {} to {}{}, fill {}, {} times; {} nodes, {} outside or on the last row or column; {:.1} s",
        field.id(),
        g.nxi,
        g.nyi,
        g.x0i,
        g.xni,
        g.y0i,
        g.yni,
        if w.closed { " (closed in longitude)" } else { "" },
        inp.fill,
        inp.times.len(),
        x.len(),
        w.outside,
        t0.elapsed().as_secs_f64()
    );
    let mut fw = FieldWriter::create(&out, field.id(), x.len(), 1, UNGTYPE)?;
    let mut n = 0;
    for (it, &t) in inp.times.iter().enumerate() {
        if start.is_some_and(|s| t < s) {
            continue;
        }
        if stop.is_some_and(|s| t > s) {
            break;
        }
        let xc = inp.field(0, it)?;
        let yc = if field.ncomp() == 2 { Some(inp.field(1, it)?) } else { None };
        let mut f = field.process(&w, g.nxi, &xc, yc.as_deref(), inp.fill);
        if rotated && f.len() == 2 {
            for i in 0..f[0].len() {
                let (u, v) = rotate_vector(f[0][i] as f64, f[1][i] as f64, angle[i]);
                f[0][i] = u as f32;
                f[1][i] = v as f32;
            }
        }
        fw.write(t, &f)?;
        n += 1;
    }
    fw.finish()?;
    println!("{n} times written to {} in {:.1} s", out.display(), t0.elapsed().as_secs_f64());
    Ok(())
}

fn cmp_fields(a: &Args) -> Result<()> {
    let (pa, pb) = match a.pos.as_slice() {
        [p, q] => (Path::new(p), Path::new(q)),
        _ => bail!("cmp-fields needs two files"),
    };
    let mut fa = FieldFile::open(pa)?;
    let mut fb = FieldFile::open(pb)?;
    anyhow::ensure!(
        fa.id == fb.id && fa.nx == fb.nx && fa.ny == fb.ny,
        "headers differ: {} {} x {} against {} {} x {}",
        fa.id,
        fa.nx,
        fa.ny,
        fb.id,
        fb.nx,
        fb.ny
    );
    println!("{}: {} x {}, GTYPE {} and {}", fa.id, fa.nx, fa.ny, fa.gtype, fb.gtype);
    let mut b = fb.next()?;
    let (mut times, mut missing) = (0usize, 0usize);
    let mut ndiff = vec![0usize; fa.nfield];
    let mut worst = vec![(0f32, Time(0, 0), 0usize, 0f32, 0f32); fa.nfield];
    let mut vmax = vec![0f32; fa.nfield];
    while let Some((t, va)) = fa.next()? {
        while b.as_ref().is_some_and(|(tb, _)| *tb < t) {
            b = fb.next()?;
        }
        let vb = match &b {
            Some((tb, vb)) if *tb == t => vb,
            _ => {
                missing += 1;
                continue;
            }
        };
        times += 1;
        for k in 0..fa.nfield {
            for (i, (&x, &y)) in va[k].iter().zip(&vb[k]).enumerate() {
                vmax[k] = vmax[k].max(x.abs());
                if x.to_bits() != y.to_bits() {
                    ndiff[k] += 1;
                    let d = (x - y).abs();
                    if d > worst[k].0 {
                        worst[k] = (d, t, i, x, y);
                    }
                }
            }
        }
    }
    println!("{times} times compared, {missing} times of A not in B");
    for k in 0..fa.nfield {
        let (d, t, i, x, y) = worst[k];
        println!(
            "field {}: {} values differ of {}; largest difference {:.3e} ({:.2e} of the largest value {:.4}) at {t}, node {i}: {x} against {y}",
            k + 1,
            ndiff[k],
            times * fa.nx * fa.ny,
            d,
            d / vmax[k].max(f32::MIN_POSITIVE),
            vmax[k]
        );
    }
    Ok(())
}

fn bounc(a: &Args) -> Result<()> {
    let grid_dir = PathBuf::from(a.pos.first().context("bounc needs the grid directory")?);
    let inp = RawSpectra::open(Path::new(a.opts.get("--input").context("bounc needs --input")?))?;
    let out = PathBuf::from(a.opts.get("--out").context("bounc needs --out")?);
    let interp: i32 = a.opts.get("--interp").map(|s| s.parse()).transpose()?.unwrap_or(2);
    let t0 = Instant::now();
    let grid = Grid::read(&grid_dir)?;
    let (x, y) = node_coordinates(a, &grid)?;
    let s = &grid.spec;
    let (nk, nth) = (s.nk, s.nth);
    let dth = 2.0 * std::f32::consts::PI / nth as f32;
    let (xfr, fr1, th1) = (s.xfr as f32, s.freq1 as f32, s.thoff as f32 * dth);
    if inp.needs_conversion(nk, nth, xfr, fr1, th1, dth) {
        bail!("the station spectra ({} x {}, from {} Hz) are not on the grid of the model: the conversion W3CSPC is not ported", inp.nk, inp.nth, inp.freq[0]);
    }
    let nodes = boundary_nodes(&grid.obc)?;
    let xbpo: Vec<f32> = nodes.iter().map(|&i| x[i] as f32).collect();
    let ybpo: Vec<f32> = nodes.iter().map(|&i| y[i] as f32).collect();
    let (ip, rd) = station_weights(&inp.lons, &inp.lats, &xbpo, &ybpo, interp);
    let mut w = NestWriter::create(&out, nk, nth, xfr, fr1, th1, &xbpo, &ybpo, &ip, &rd)?;
    let spectra: Vec<Vec<Vec<f32>>> = (0..inp.nst).map(|k| inp.station(k)).collect::<Result<_>>()?;
    for (it, &t) in inp.times.iter().enumerate() {
        let at: Vec<Vec<f32>> = spectra.iter().map(|sp| sp[it].clone()).collect();
        w.write(t, &at)?;
    }
    w.finish()?;
    println!(
        "{} boundary nodes from {} stations ({} x {} spectra), {} times ({} to {}) written to {} in {:.1} s",
        nodes.len(),
        inp.nst,
        inp.nk,
        inp.nth,
        inp.times.len(),
        inp.times.first().map_or(String::new(), |t| t.to_string()),
        inp.times.last().map_or(String::new(), |t| t.to_string()),
        out.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn cmp_nest(a: &Args) -> Result<()> {
    let (pa, pb) = match a.pos.as_slice() {
        [p, q] => (Path::new(p), Path::new(q)),
        _ => bail!("cmp-nest needs two files"),
    };
    let mut fa = NestFile::open(pa)?;
    let mut fb = NestFile::open(pb)?;
    println!(
        "header: NK {} {}, NTH {} {}, XFR {} {}, FR1 {} {}, TH1 {} {}, NBI {} {}",
        fa.nk, fb.nk, fa.nth, fb.nth, fa.xfr, fb.xfr, fa.fr1, fb.fr1, fa.th1, fb.th1, fa.nbi, fb.nbi
    );
    anyhow::ensure!(fa.nk == fb.nk && fa.nth == fb.nth && fa.nbi == fb.nbi, "the files differ in size");
    let maxdiff = |u: &[f32], v: &[f32]| u.iter().zip(v).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let ndiff = |u: &[f32], v: &[f32]| u.iter().zip(v).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    println!(
        "boundary points: XBPI {} differ (max {:.2e}), YBPI {} differ (max {:.2e}), IPBPI {} differ, RDBPI {} differ (max {:.2e})",
        ndiff(&fa.xbpi, &fb.xbpi),
        maxdiff(&fa.xbpi, &fb.xbpi),
        ndiff(&fa.ybpi, &fb.ybpi),
        maxdiff(&fa.ybpi, &fb.ybpi),
        fa.ipbpi.iter().zip(&fb.ipbpi).filter(|(x, y)| x != y).count(),
        ndiff(&fa.rdbpi, &fb.rdbpi),
        maxdiff(&fa.rdbpi, &fb.rdbpi)
    );
    // the times of A found in B (B may hold a part of A's period)
    let (mut times, mut missing, mut nd, mut worst, mut vmax) = (0usize, 0usize, 0usize, 0f32, 0f32);
    let mut b = fb.next()?;
    while let Some((ta, sa)) = fa.next()? {
        while b.as_ref().is_some_and(|(tb, _)| *tb < ta) {
            b = fb.next()?;
        }
        let sb = match &b {
            Some((tb, sb)) if *tb == ta => sb,
            _ => {
                missing += 1;
                continue;
            }
        };
        anyhow::ensure!(sa.len() == sb.len(), "{ta}: {} spectra against {}", sa.len(), sb.len());
        for (u, v) in sa.iter().zip(sb) {
            nd += ndiff(u, v);
            worst = worst.max(maxdiff(u, v));
            vmax = u.iter().fold(vmax, |m, x| m.max(x.abs()));
        }
        times += 1;
    }
    println!(
        "{times} times compared, {missing} times of A not in B: {nd} spectral values differ, largest difference {worst:.3e} ({:.2e} of the largest value {vmax:.4e})",
        worst / vmax.max(f32::MIN_POSITIVE)
    );
    Ok(())
}

fn structured(a: &Args) -> Result<()> {
    let nml = PathBuf::from(a.pos.first().context("struct needs the ww3_grid.nml")?);
    let data = PathBuf::from(a.opts.get("--data").context("struct needs --data")?);
    let out = PathBuf::from(a.opts.get("--out").context("struct needs --out")?);
    let t0 = Instant::now();
    let g = StructGrid::read(&nml, &data)?;
    if a.opts.get("--scheme").map(String::as_str) == Some("uq") {
        return structured_uq(a, &g, &nml, &data, &out, t0);
    }
    let c = g.counts();
    println!(
        "{} ({:?}, {} x {}, closed {}): land {}, sea {}, active boundary {}, excluded {} ({:.1} s)",
        g.name,
        g.gtype,
        g.nx,
        g.ny,
        g.closed,
        c[0],
        c[1],
        c[2],
        c[3],
        t0.elapsed().as_secs_f64()
    );
    let rot = match a.opts.get("--rotate").map(String::as_str).unwrap_or("auto") {
        "auto" => g.auto_rotation(10.0),
        r => Rotation::parse(r)?,
    };
    let m = g.triangulate(rot)?;
    let shallow = m.grid.depth.iter().filter(|&&d| d < g.spec.dmin + 1e-5).count();
    let (lo, hi) = m.grid.lat.iter().fold((90.0f64, -90.0f64), |(l, h), &v| (l.min(v), h.max(v)));
    println!(
        "mesh: rotation {:?}, {} nodes of {} sea points ({} in no triangle), {} triangles, {} open-boundary nodes, {} nodes shallower than DMIN {} (dry for the scheme), mesh latitudes {:.2} to {:.2}",
        rot,
        m.grid.lon.len(),
        m.nsea,
        m.dropped,
        m.grid.tri.len(),
        m.grid.obc.len(),
        shallow,
        g.spec.dmin,
        lo,
        hi
    );
    m.write(&out, &std::fs::read_to_string(&nml)?)?;
    if let Some(rn) = a.opts.get("--nml") {
        use hww3_wave::ref1::{flagtr, read_obstructions, reflection_geometry_trn, Ref1Params};
        use hww3_wave::ww3grid::Namelists;
        let rnl = Namelists::read(Path::new(rn))?;
        let gn = Namelists::read(&nml)?;
        let rp = Ref1Params::from_namelists(&rnl, &g.spec)?;
        let trn = read_obstructions(&gn, flagtr(&rnl)?, &data, g.nx, g.ny)?;
        let geo = reflection_geometry_trn(&g, &rp, trn.as_ref())?;
        // node k -> sea point of the structured grid
        let raw = out.join("raw");
        let (mut reflc, mut refld) = (Vec::with_capacity(4 * m.ij.len()), Vec::with_capacity(6 * m.ij.len()));
        let (mut dx, mut dy, mut tx, mut ty) = (vec![], vec![], vec![], vec![]);
        for &i in &m.ij {
            let s = geo.ij.binary_search(&i).map_err(|_| anyhow::anyhow!("node at {i} not a sea point"))?;
            reflc.extend_from_slice(&geo.reflc[s]);
            refld.extend(geo.refld[s].iter().map(|&v| v.max(0) as u32));
            dx.push(geo.delx[s]);
            dy.push(geo.dely[s]);
            tx.push(geo.trnx[s]);
            ty.push(geo.trny[s]);
        }
        let f64s = |name: &str, v: &[f64]| std::fs::write(raw.join(name), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        f64s("ref1_reflc.f64", &reflc)?;
        std::fs::write(raw.join("ref1_refld.u32"), refld.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
        f64s("ref1_delx.f64", &dx)?;
        f64s("ref1_dely.f64", &dy)?;
        f64s("ref1_trnx.f64", &tx)?;
        f64s("ref1_trny.f64", &ty)?;
        let ncoast = (0..m.ij.len()).filter(|&k| reflc[4 * k] > 0.0).count();
        println!("reflection geometry: {} nodes with a coast reflection, REFPARS {:?}", ncoast, rp.refpars);
    }
    println!("written to {} in {:.1} s", out.display(), t0.elapsed().as_secs_f64());
    Ok(())
}

/// `struct --scheme uq`: the grid directory of the UQ scheme.
fn structured_uq(a: &Args, g: &StructGrid, nml: &Path, data: &Path, out: &Path, t0: Instant) -> Result<()> {
    use hww3_wave::ref1::{flagtr, read_obstructions, reflection_geometry_trn, Ref1Params};
    use hww3_wave::uqgrid::{UqSetup, UqTables};
    use hww3_wave::ww3grid::Namelists;
    let rn = a.opts.get("--nml").context("struct --scheme uq needs --nml (the namelists of the run)")?;
    let rnl = Namelists::read(Path::new(rn))?;
    let gn = Namelists::read(nml)?;
    let trn = read_obstructions(&gn, flagtr(&rnl)?, data, g.nx, g.ny)?;
    UqSetup::write(out, g, trn.as_ref(), &std::fs::read_to_string(nml)?, &std::fs::read_to_string(rn)?)?;
    let setup = UqSetup::read(out)?;
    let pg = setup.pro3()?;
    let t = UqTables::new(&pg)?;
    let c = g.counts();
    println!(
        "{} ({:?}, {} x {}, closed {}): land {}, sea {}, active boundary {}, excluded {}; UQ scheme: {} sea points, NCENT {}, W3MAP3 counters {:?}, FLAGTR {}, DTMAX {} s, DTXY {} s, DTKTH {} s",
        g.name,
        g.gtype,
        g.nx,
        g.ny,
        g.closed,
        c[0],
        c[1],
        c[2],
        c[3],
        t.nsea,
        pg.ncent(),
        pg.counts(),
        setup.params.flagtr,
        setup.dtmax,
        setup.params.dtcfl,
        setup.dtkth
    );
    // the reflection geometry of W3SETREF per sea point (hww3-prop run --ice-ref)
    let rp = Ref1Params::from_namelists(&rnl, &g.spec)?;
    let geo = reflection_geometry_trn(g, &rp, trn.as_ref())?;
    let raw = out.join("raw");
    let (mut reflc, mut refld) = (vec![], vec![]);
    let (mut dx, mut dy, mut tx, mut ty) = (vec![], vec![], vec![], vec![]);
    for &i in &pg.sea_ij {
        let s = geo.ij.binary_search(&(i as u32)).map_err(|_| anyhow::anyhow!("point {i} not a sea point of the reflection geometry"))?;
        reflc.extend_from_slice(&geo.reflc[s]);
        refld.extend(geo.refld[s].iter().map(|&v| v.max(0) as u32));
        dx.push(geo.delx[s]);
        dy.push(geo.dely[s]);
        tx.push(geo.trnx[s]);
        ty.push(geo.trny[s]);
    }
    let f64s = |name: &str, v: &[f64]| std::fs::write(raw.join(name), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    f64s("ref1_reflc.f64", &reflc)?;
    std::fs::write(raw.join("ref1_refld.u32"), refld.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    f64s("ref1_delx.f64", &dx)?;
    f64s("ref1_dely.f64", &dy)?;
    f64s("ref1_trnx.f64", &tx)?;
    f64s("ref1_trny.f64", &ty)?;
    println!("written to {} in {:.1} s", out.display(), t0.elapsed().as_secs_f64());
    Ok(())
}

/// `restart-uq`: a restart of the mesh of a structured grid for the grid
/// directory of its UQ scheme.
fn restart_uq(a: &Args) -> Result<()> {
    use hww3_wave::ww3io::Restart;
    let mesh = PathBuf::from(a.pos.first().context("restart-uq needs the mesh directory")?);
    let uqd = PathBuf::from(a.opts.get("--uq-grid").context("restart-uq needs --uq-grid")?);
    let inp = PathBuf::from(a.opts.get("--restart").context("restart-uq needs --restart")?);
    let out = PathBuf::from(a.opts.get("--out").context("restart-uq needs --out")?);
    let t0 = Instant::now();
    let rd = |d: &Path, n: &str| -> Result<Vec<u32>> {
        let b = std::fs::read(d.join("raw").join(n)).with_context(|| format!("{}/raw/{n}", d.display()))?;
        Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    };
    let (ij, sea_ij, uq_sea, uq_obc) = (rd(&mesh, "ij.u32")?, rd(&mesh, "sea_ij.u32")?, rd(&uqd, "sea_ij.u32")?, rd(&uqd, "obc.u32")?);
    anyhow::ensure!(sea_ij == uq_sea, "the mesh and the UQ grid do not have the same sea points");
    let r = Restart::read(&inp)?;
    anyhow::ensure!(r.nsea == ij.len(), "restart of {} points, mesh of {} nodes", r.nsea, ij.len());
    let (nspec, nsea) = (r.nspec, sea_ij.len());
    let mut node = vec![usize::MAX; nsea];
    for (k, &i) in ij.iter().enumerate() {
        let s = sea_ij.binary_search(&i).map_err(|_| anyhow::anyhow!("node {k} at {i} is not a sea point"))?;
        node[s] = k;
    }
    let mut base = vec![1i32; nsea];
    for &o in &uq_obc {
        base[o as usize] = 2;
    }
    let pick = |v: &[f32], x: f32| -> Vec<f32> {
        if v.is_empty() {
            return vec![];
        }
        node.iter().map(|&k| if k == usize::MAX { x } else { v[k] }).collect()
    };
    let mut va = vec![0f32; nsea * nspec];
    for (s, &k) in node.iter().enumerate() {
        if k != usize::MAX {
            va[s * nspec..(s + 1) * nspec].copy_from_slice(&r.va[k * nspec..(k + 1) * nspec]);
        }
    }
    let mapsta: Vec<i32> = node.iter().zip(&base).map(|(&k, &b)| if k != usize::MAX && r.mapsta[k] < 0 { -b } else { b }).collect();
    let mapst2: Vec<i32> = node.iter().map(|&k| if k == usize::MAX { 0 } else { r.mapst2[k] }).collect();
    let left = node.iter().filter(|&&k| k == usize::MAX).count();
    let s = Restart {
        nsea,
        va,
        wlv: pick(&r.wlv, 0.0),
        ice: pick(&r.ice, 0.0),
        mapsta,
        mapst2,
        ust: pick(&r.ust, 1e-5),
        ustdir: pick(&r.ustdir, 0.0),
        asf: pick(&r.asf, 1.0),
        fpis: pick(&r.fpis, 0.0),
        tail: vec![],
        map_full: None,
        ..r
    };
    s.write(&out)?;
    println!("{}: {nsea} sea points ({left} out of the mesh, at zero), NSPEC {nspec}, time {}, written in {:.1} s", out.display(), s.time, t0.elapsed().as_secs_f64());
    Ok(())
}

fn ideal_wind(a: &Args) -> Result<()> {
    let dir = PathBuf::from(a.pos.first().context("ideal-wind needs the grid directory")?);
    let out = PathBuf::from(a.opts.get("--out").context("ideal-wind needs --out")?);
    let start = Time::parse(a.opts.get("--start").context("ideal-wind needs --start")?)?;
    let hours: i64 = a.opts.get("--hours").context("ideal-wind needs --hours")?.parse()?;
    let u0: f64 = a.opts.get("--speed").map(|s| s.parse()).transpose()?.unwrap_or(10.0);
    let grid = Grid::read(&dir)?;
    let (lon, lat, ang) = Mesh::geographic(&dir, &grid)?;
    let np = lon.len();
    let mut w = FieldWriter::create(&out, "WND", np, 1, UNGTYPE)?;
    let d = std::f64::consts::PI / 180.0;
    for h in 0..=hours {
        let t = Time::from_secs(start.secs() + 3600 * h);
        let phase = 15.0 * h as f64 / 24.0;
        let (mut u, mut v) = (vec![0f32; np], vec![0f32; np]);
        for i in 0..np {
            let (l, p) = (lon[i] * d, lat[i] * d);
            // zonal belts (trades below 30°, westerlies above) and three
            // travelling lows on the westerlies
            let s = (3.0 * (lon[i] - phase) * d).sin();
            let ue = u0 * (-(2.0 * p).cos() * 0.8 + 0.5 * s * (2.0 * p).sin().abs());
            let un = u0 * 0.5 * (3.0 * (lon[i] - phase) * d).cos() * (2.0 * p).sin() + 0.1 * u0 * l.sin();
            let (x, y) = rotate_vector(ue, un, ang[i]);
            u[i] = x as f32;
            v[i] = y as f32;
        }
        w.write(t, &[u, v])?;
    }
    w.finish()?;
    println!("{}: {} hourly winds from {start} on {np} nodes", out.display(), hours + 1);
    Ok(())
}

fn restart_struct(a: &Args) -> Result<()> {
    use hww3_wave::ww3io::Restart;
    let dir = PathBuf::from(a.pos.first().context("restart-struct needs the grid directory")?);
    let inp = PathBuf::from(a.opts.get("--restart").context("restart-struct needs --restart")?);
    let out = PathBuf::from(a.opts.get("--out").context("restart-struct needs --out")?);
    let t0 = Instant::now();
    let rd = |n: &str| -> Result<Vec<u32>> {
        let b = std::fs::read(dir.join("raw").join(n)).with_context(|| n.to_string())?;
        Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    };
    let (ij, sea_ij, obc) = (rd("ij.u32")?, rd("sea_ij.u32")?, rd("obc.u32")?);
    let r = Restart::read(&inp)?;
    anyhow::ensure!(r.nsea == ij.len(), "restart of {} points, mesh of {} nodes", r.nsea, ij.len());
    let nspec = r.nspec;
    let nsea = sea_ij.len();
    let mut is_obc = vec![false; ij.len()];
    for &o in &obc {
        is_obc[o as usize] = true;
    }
    // node of each sea point (usize::MAX when left out of the mesh)
    let mut node = vec![usize::MAX; nsea];
    for (k, &i) in ij.iter().enumerate() {
        let s = sea_ij.binary_search(&i).map_err(|_| anyhow::anyhow!("node {k} at {i} is not a sea point"))?;
        node[s] = k;
    }
    let pick = |v: &[f32], x: f32| -> Vec<f32> {
        if v.is_empty() {
            return vec![];
        }
        node.iter().map(|&k| if k == usize::MAX { x } else { v[k] }).collect()
    };
    let mut va = vec![0f32; nsea * nspec];
    for (s, &k) in node.iter().enumerate() {
        if k != usize::MAX {
            va[s * nspec..(s + 1) * nspec].copy_from_slice(&r.va[k * nspec..(k + 1) * nspec]);
        }
    }
    let mapsta: Vec<i32> = node.iter().map(|&k| if k != usize::MAX && is_obc[k] { 2 } else { 1 }).collect();
    let mapst2: Vec<i32> = node.iter().map(|&k| if k == usize::MAX { 0 } else { r.mapst2[k] }).collect();
    let g = hww3_wave::structgrid::StructGrid::read(
        Path::new(a.opts.get("--grid-nml").context("restart-struct needs --grid-nml")?),
        Path::new(a.opts.get("--data").context("restart-struct needs --data")?),
    )?;
    anyhow::ensure!(g.counts()[1] + g.counts()[2] == nsea, "{}: {} sea points, mesh directory {nsea}", g.name, g.counts()[1] + g.counts()[2]);
    let mut node_of_point = vec![usize::MAX; g.nx * g.ny];
    for (s, &i) in sea_ij.iter().enumerate() {
        node_of_point[i as usize] = node[s];
    }
    let full: Vec<i32> = (0..g.nx * g.ny)
        .map(|i| match g.mapsta[i] {
            0 => 0,
            3 => 8,
            _ => match node_of_point[i] {
                usize::MAX => g.mapsta[i] as i32,
                k => r.mapsta[k] + 8 * r.mapst2[k],
            },
        })
        .collect();
    let s = Restart {
        nsea,
        gname: g.name.clone(),
        map_full: Some((g.nx, g.ny, full)),
        va,
        wlv: pick(&r.wlv, 0.0),
        ice: pick(&r.ice, 0.0),
        mapsta,
        mapst2,
        ust: pick(&r.ust, 1e-5),
        ustdir: pick(&r.ustdir, 0.0),
        asf: pick(&r.asf, 1.0),
        fpis: pick(&r.fpis, 0.0),
        tail: vec![],
        ..r
    };
    s.write(&out)?;
    println!(
        "{}: {} sea points ({} from the mesh), NSPEC {}, time {}, ice > 0 at {} points, written in {:.1} s",
        out.display(),
        nsea,
        ij.len(),
        nspec,
        s.time,
        s.ice.iter().filter(|&&c| c > 0.0).count(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() -> Result<()> {
    let a = parse()?;
    match a.cmd.as_str() {
        "prnc" => prnc(&a),
        "cmp-fields" => cmp_fields(&a),
        "bounc" => bounc(&a),
        "cmp-nest" => cmp_nest(&a),
        "struct" => structured(&a),
        "ideal-wind" => ideal_wind(&a),
        "restart-struct" => restart_struct(&a),
        "restart-uq" => restart_uq(&a),
        other => bail!("unknown command {other}"),
    }
}
