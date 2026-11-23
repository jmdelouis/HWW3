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
//! hww3-pre cmp-fields A.ww3 B.ww3
//!     Compare two forcing files time by time: number of values that
//!     differ and largest difference, per field, over the times of A found
//!     in B.
//! ```

use anyhow::{bail, Context, Result};
use hww3_wave::grid::Grid;
use hww3_wave::prnc::{Field, RawInput};
use hww3_wave::ww3io::{FieldFile, FieldWriter, Time};
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
    let (x, y) = match a.opts.get("--msh") {
        Some(m) => {
            let (x, y) = msh_nodes(Path::new(m))?;
            anyhow::ensure!(x.len() == grid.lon.len(), "{m}: {} nodes, grid {}", x.len(), grid.lon.len());
            let d = x.iter().zip(&grid.lon).chain(y.iter().zip(&grid.lat)).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            anyhow::ensure!(d < 1e-5, "{m}: coordinates differ from the grid's by {d:.2e} degree");
            println!("node coordinates of {m} (within {d:.1e} degree of the grid's)");
            (x, y)
        }
        None => (grid.lon.clone(), grid.lat.clone()),
    };
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
        let f = field.process(&w, g.nxi, &xc, yc.as_deref(), inp.fill);
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

fn main() -> Result<()> {
    let a = parse()?;
    match a.cmd.as_str() {
        "prnc" => prnc(&a),
        "cmp-fields" => cmp_fields(&a),
        other => bail!("unknown command {other}"),
    }
}
