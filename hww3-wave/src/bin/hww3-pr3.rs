//! The geographic propagation of WW3's structured grids (UQ/PR3 with the
//! GSE averaging, [`hww3_wave::pro3`]) on the CPU, as `W3WAVE` drives it
//! with the spectral propagation and the source terms off (`RUN%FLCX =
//! FLCY = T`, `FLCTH = FLCK = FLSOU = F`): a reference to check the
//! transcription against `ww3_shel` on the same grid from the same restart.
//!
//! ```text
//! hww3-pr3 WW3_GRID.nml --data DIR --nml NAMELISTS.nml --mesh GRID_DIR
//!          --restart RESTART.ww3 --start YYYYMMDDHHMMSS --hours H --out DIR
//!          [--dtmax S]
//! hww3-pr3 WW3_GRID.nml --data DIR --nml NAMELISTS.nml --harness FILE
//! ```
//!
//! The restart is one of `hww3-prop` on the mesh `GRID_DIR` of the grid
//! (`hww3-pre struct`); its spectra go to the sea points through
//! `raw/ij.u32` (the points left out of the mesh start at zero), with its
//! levels, ice and masked points (negative `MAPSTA`; the ice would enter
//! the transparencies of `W3UTRN` with `FLAGTR >= 3` and the `IC0` switch,
//! which the IRI build does not have). The group speeds are those of `W3INIT` (`WAVNU1` at
//! `max(DMIN, WLV - ZB)`, no currents). As in `W3WAVE`, each hour (the
//! output times) is cut into `NT = 1 + INT(3600 / DTMAX - 0.001)` equal
//! global steps (1,800 s for `DTMAX = 2520`), and each spectral component
//! goes through `W3XYP3` (in parallel over the components).
//!
//! With `--harness FILE` (the output of `harness/xyp3_harness.F90`, only
//! the grid, `--data` and `--nml` needed) it compares instead the maps,
//! transparencies and fields of `pro3` with WW3's, point by point, for the
//! components and the two global steps of the harness. `OUT/hs_YYYYMMDDHH.f32` holds `H_s` (m,
//! little endian) on the whole `NX x NY` grid (`iy NX + ix`, -999.9 off the
//! sea points) at the start and every hour, with the tail of `W3OUTG`.

use anyhow::{Context, Result};
use hww3_wave::pro3::{xyp3, DispTable, Pro3Grid, Pro3Params};
use hww3_wave::ref1::{flagtr, read_obstructions};
use hww3_wave::spec::Spec;
use hww3_wave::structgrid::StructGrid;
use hww3_wave::ww3grid::Namelists;
use hww3_wave::ww3io::{Restart, Time};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: hww3-pr3 WW3_GRID.nml --data DIR --nml NAMELISTS.nml --mesh GRID_DIR --restart RESTART.ww3 --start T --hours H --out DIR [--dtmax S]";
    let grid_nml = PathBuf::from(a.first().context(usage)?);
    let mut o: HashMap<String, String> = HashMap::new();
    let mut i = 1;
    while i + 1 < a.len() {
        o.insert(a[i].clone(), a[i + 1].clone());
        i += 2;
    }
    let get = |k: &str| o.get(k).cloned().with_context(|| format!("{k} missing; {usage}"));
    let data = PathBuf::from(get("--data")?);
    let t0 = Instant::now();
    let gn = Namelists::read(&grid_nml)?;
    let pn = Namelists::read(Path::new(&get("--nml")?))?;
    let dtmax: f64 = match o.get("--dtmax") {
        Some(s) => s.parse()?,
        None => gn.f("TIMESTEPS_NML", "TIMESTEPS%DTMAX", 600.0)?,
    };
    let g = StructGrid::read(&grid_nml, &data)?;
    let trn = read_obstructions(&gn, flagtr(&pn)?, &data, g.nx, g.ny)?;
    let params = Pro3Params::from_namelists(&gn, &pn)?;
    let mut pg = Pro3Grid::new(&g, trn.as_ref(), &params)?;
    if let Some(h) = o.get("--harness") {
        return check_harness(&mut pg, Path::new(h));
    }
    let mesh = PathBuf::from(get("--mesh")?);
    let out = PathBuf::from(get("--out")?);
    let start = Time::parse(&get("--start")?)?;
    let hours: i64 = get("--hours")?.parse()?;
    std::fs::create_dir_all(&out)?;
    let s = &g.spec;
    let sp = Spec::new(s.nk, s.nth, s.freq1 as f32, s.xfr as f32, s.thoff as f32);
    let (nk, nth, nx, ny) = (s.nk, s.nth, g.nx, g.ny);
    let nspec = nk * nth;
    // sea points (ISEA order)
    let sea: Vec<usize> = (0..nx * ny).filter(|&i| g.is_sea(i)).collect();
    let nsea = sea.len();
    let mut isea_of = vec![usize::MAX; nx * ny];
    for (isea, &ip) in sea.iter().enumerate() {
        isea_of[ip] = isea;
    }
    // spectra of the restart (mesh order) on the whole grid, per component
    let ij: Vec<u32> = std::fs::read(mesh.join("raw").join("ij.u32"))?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let rs = Restart::read(Path::new(&get("--restart")?))?;
    anyhow::ensure!(rs.nsea == ij.len() && rs.nspec == nspec, "restart {} x {} against mesh {} x {nspec}", rs.nsea, rs.nspec, ij.len());
    anyhow::ensure!(rs.time == start, "restart at {}, --start {start}", rs.time);
    // levels and ice of the restart at the sea points (zero off the mesh)
    let (mut wlv, mut ice) = (vec![0f32; nsea], vec![0f64; nsea]);
    for (k, &p) in ij.iter().enumerate() {
        let isea = isea_of[p as usize];
        anyhow::ensure!(isea != usize::MAX, "mesh node {k} on a land point");
        wlv[isea] = rs.wlv[k];
        ice[isea] = rs.ice[k] as f64;
    }
    // MAPSTA of the restart (negative where the restart masks a point for
    // ice or drying; the grid's elsewhere), then W3UTRN with the ice of the
    // restart, as the first FLMAP update of W3WAVE (no ice forcing: it
    // stays)
    let mut full: Vec<i32> = (0..nx * ny).map(|i| if g.is_sea(i) { g.mapsta[i] as i32 } else { 0 }).collect();
    for (k, &p) in ij.iter().enumerate() {
        if rs.mapsta[k] < 0 {
            full[p as usize] = -full[p as usize].abs();
        }
    }
    let nneg = full.iter().filter(|&&m| m < 0).count();
    pg.update_maps(Some(&full), Some(&ice), None)?;
    println!("MAPSTA of the restart: {nneg} sea points masked");
    // wavenumbers and group speeds of W3INIT (WAVNU1 at max(DMIN, DW),
    // DW = max(0, WLV - ZB), single precision)
    let tab = DispTable::new();
    let mut cgk = vec![vec![0f64; nsea]; nk + 2];
    for (isea, &ip) in sea.iter().enumerate() {
        let dw = (wlv[isea] - g.zb[ip] as f32).max(0.0);
        let depth = (s.dmin as f32).max(dw);
        for ik in 0..nk + 2 {
            cgk[ik][isea] = tab.wavnu1(sp.sig[ik], depth).1 as f64;
        }
    }
    let mut fields: Vec<Vec<f64>> = (0..nspec)
        .into_par_iter()
        .map(|isp| {
            let mut f = vec![0f64; nx * ny];
            for (k, &p) in ij.iter().enumerate() {
                f[p as usize] = rs.va[k * nspec + isp] as f64;
            }
            f
        })
        .collect();
    drop(rs);
    println!(
        "{} ({} x {}, {} sea points, closed {}), NK {nk} NTH {nth}, DTMAX {dtmax} s, DTXY {} s, WDTHCG {} WDTHTH {}, FLAGTR {}: set-up {:.1} s",
        g.name,
        nx,
        ny,
        nsea,
        g.closed,
        params.dtcfl,
        params.wdthcg,
        params.wdthth,
        params.flagtr,
        t0.elapsed().as_secs_f64()
    );
    let write_hs = |t: Time, fields: &Vec<Vec<f64>>| -> Result<()> {
        let mut hs = vec![-999.9f32; nx * ny];
        let vals: Vec<f32> = (0..nsea)
            .into_par_iter()
            .map(|isea| {
                let ip = sea[isea];
                let (mut et, mut ab) = (0f64, 0f64);
                for ik in 0..nk {
                    ab = (0..nth).map(|th| fields[ik * nth + th][ip]).sum::<f64>();
                    et += ab * sp.dden[ik] as f64 / cgk[ik + 1][isea];
                }
                et += sp.fte as f64 * ab / cgk[nk][isea];
                (4.0 * et.max(0.0).sqrt()) as f32
            })
            .collect();
        for (isea, &ip) in sea.iter().enumerate() {
            hs[ip] = vals[isea];
        }
        let tag = format!("{:08}{:02}", t.0, t.1 / 10000);
        std::fs::write(out.join(format!("hs_{tag}.f32")), hs.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
        let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / nsea as f64;
        println!("{t}: mean H_s {mean:.4} m");
        Ok(())
    };
    write_hs(start, &fields)?;
    let stop = start.secs() + 3600 * hours;
    let mut t = start.secs();
    let mut steps = 0usize;
    let t1 = Instant::now();
    while t < stop {
        // W3WAVE section 2: from TIME to the next output time TCALC, NT
        // equal steps of at most DTMAX, rounded to the second with the
        // remainder carried (single precision, as in WW3)
        let next_hour = (t / 3600 + 1) * 3600;
        let dttst = (next_hour - t) as f32;
        let nt = 1 + (dttst / dtmax as f32 - 0.001) as i64;
        let dtga = dttst / nt as f32;
        let mut dtres = 0f32;
        for _ in 0..nt {
            let dtg = (dtga + dtres + 0.0001).round();
            dtres += dtga - dtg;
            if dtres.abs() < 0.001 {
                dtres = 0.0;
            }
            t += dtg as i64;
            let secs_of_day = (t.rem_euclid(86400)) as f64;
            fields.par_iter_mut().enumerate().for_each(|(isp, f)| {
                let ik = isp / nth;
                xyp3(&pg, isp, dtg as f64, secs_of_day, f, &cgk[ik + 1], None, None);
            });
            steps += 1;
        }
        if t % 3600 == 0 {
            write_hs(Time::from_secs(t), &fields)?;
        }
    }
    println!("{steps} global steps in {:.1} s", t1.elapsed().as_secs_f64());
    Ok(())
}

/// Little-endian reader of the output of the WW3 harness
/// `harness/xyp3_harness.F90`.
struct Rd {
    b: Vec<u8>,
    p: usize,
}

impl Rd {
    fn i(&mut self) -> i32 {
        let v = i32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn f(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn fv(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.f() as f64).collect()
    }
}

/// `--harness FILE`: run `W3XYP3`'s transcription on the inputs of the WW3
/// harness (status map, ice, group velocities, fields of a few components
/// of the restart) and compare the maps, the transparencies and the fields
/// after one and two global steps, point by point.
fn check_harness(pg: &mut Pro3Grid, path: &Path) -> Result<()> {
    let mut r = Rd { b: std::fs::read(path)?, p: 8 };
    anyhow::ensure!(&r.b[..8] == b"HWW3XYP3", "{}: not a harness file", path.display());
    let (nx, ny, nsea, _nk, _nth, nisp) = (r.i() as usize, r.i() as usize, r.i() as usize, r.i(), r.i(), r.i() as usize);
    let rtime = [r.i(), r.i()];
    let dtg = r.f() as f64;
    let counts: Vec<usize> = (0..8).map(|_| r.i() as usize).collect();
    anyhow::ensure!(nx == pg.nx && ny == pg.ny && nsea == pg.nsea, "harness {nx} x {ny}, {nsea} sea points; grid {} x {}, {}", pg.nx, pg.ny, pg.nsea);
    // MAPSTA in WW3's order (IXY) to the order iy nx + ix
    let mut mapsta = vec![0i32; nx * ny];
    for ix in 0..nx {
        for iy in 0..ny {
            mapsta[iy * nx + ix] = r.i();
        }
    }
    let ice = r.fv(nsea);
    let _wlv = r.fv(nsea);
    let ww3_tr: Vec<Vec<f64>> = (0..5).map(|_| r.fv(nsea)).collect();
    pg.update_maps(Some(&mapsta), Some(&ice), None)?;
    println!("W3MAP3 counters (NCENT NMX0-2 NMY0-2 NACT): WW3 {:?}, ours {:?}", counts, pg.counts());
    let mut dtr = [0f64; 4];
    let mut ntrn = 0usize;
    for (isea, &i) in pg.sea_ij.iter().enumerate() {
        let (ix, iy) = (i % nx, i / nx);
        let (xm, xp) = pg.atrnx(ix, iy);
        let (ym, yp) = pg.atrny(ix, iy);
        for (k, v) in [xm, xp, ym, yp].into_iter().enumerate() {
            dtr[k] = dtr[k].max((v - ww3_tr[k][isea]).abs());
        }
        if pg.maptrn(ix, iy) != (ww3_tr[4][isea] > 0.5) {
            ntrn += 1;
        }
    }
    println!("ATRNX(-1) ATRNX(1) ATRNY(-1) ATRNY(1) largest differences {dtr:?}; MAPTRN differs at {ntrn} points");
    let secs0 = ((rtime[1] / 10000) * 3600 + (rtime[1] / 100 % 100) * 60 + rtime[1] % 100) as f64;
    for _ in 0..nisp {
        let isp = (r.i() - 1) as usize;
        let cg = r.fv(nsea);
        let f0 = r.fv(nsea);
        let ww3 = [r.fv(nsea), r.fv(nsea)];
        let mut f = vec![0f64; nx * ny];
        for (isea, &i) in pg.sea_ij.iter().enumerate() {
            f[i] = f0[isea];
        }
        for (step, w) in ww3.iter().enumerate() {
            if step == 1 {
                // the second step from WW3's first one
                for (isea, &i) in pg.sea_ij.iter().enumerate() {
                    f[i] = ww3[0][isea];
                }
            }
            let secs = (secs0 + dtg * (step + 1) as f64).rem_euclid(86400.0);
            xyp3(pg, isp, dtg, secs, &mut f, &cg, None, None);
            let fmax = w.iter().cloned().fold(0.0, f64::max).max(1e-30);
            let mut worst: Vec<(f64, usize)> = pg.sea_ij.iter().enumerate().map(|(isea, &i)| ((f[i] - w[isea]).abs(), isea)).collect();
            let nbad = worst.iter().filter(|x| x.0 > 1e-4 * fmax).count();
            worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let rms = (worst.iter().map(|x| x.0 * x.0).sum::<f64>() / nsea as f64).sqrt();
            println!(
                "ISP {:4} (IK {:2} ITH {:2}) step {}: max |diff| / max {:.3e}, RMS / max {:.3e}, points above 1e-4 of the max: {nbad}",
                isp + 1,
                isp / pg.nth + 1,
                isp % pg.nth + 1,
                step + 1,
                worst[0].0 / fmax,
                rms / fmax
            );
            for &(d, isea) in worst.iter().take(if nbad > 0 { 4 } else { 0 }) {
                let i = pg.sea_ij[isea];
                let (ix, iy) = (i % nx, i / nx);
                println!(
                    "    (IX {} IY {}) MAPSTA {} MAPTRN {}: WW3 {:.6e} ours {:.6e} diff {:.3e}",
                    ix + 1,
                    iy + 1,
                    pg.mapsta(ix, iy),
                    pg.maptrn(ix, iy),
                    w[isea],
                    f[i],
                    d
                );
            }
        }
    }
    Ok(())
}
