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
//! ```
//!
//! The restart is one of `hww3-prop` on the mesh `GRID_DIR` of the grid
//! (`hww3-pre struct`); its spectra go to the sea points through
//! `raw/ij.u32` (the points left out of the mesh start at zero). The
//! wavenumbers and group speeds are those of the bathymetric depth (no
//! levels, no currents). Global steps of `DTMAX` (default the grid's
//! `TIMESTEPS%DTMAX`) are cut at the hours, as `W3WAVE` cuts them at the
//! output times; each spectral component goes through `W3XYP3` (in
//! parallel over the components). `OUT/hs_YYYYMMDDHH.f32` holds `H_s` (m,
//! little endian) on the whole `NX x NY` grid (`iy NX + ix`, -999.9 off the
//! sea points) at the start and every hour, with the tail of `W3OUTG`.

use anyhow::{Context, Result};
use hww3_wave::pro3::{xyp3, Pro3Grid, Pro3Params};
use hww3_wave::ref1::{flagtr, read_obstructions};
use hww3_wave::spec::{node_wavenumbers, Spec};
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
    let mesh = PathBuf::from(get("--mesh")?);
    let out = PathBuf::from(get("--out")?);
    let start = Time::parse(&get("--start")?)?;
    let hours: i64 = get("--hours")?.parse()?;
    std::fs::create_dir_all(&out)?;
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
    let pg = Pro3Grid::new(&g, trn.as_ref(), &params)?;
    let s = &g.spec;
    let sp = Spec::new(s.nk, s.nth, s.freq1 as f32, s.xfr as f32, s.thoff as f32);
    let (nk, nth, nx, ny) = (s.nk, s.nth, g.nx, g.ny);
    let nspec = nk * nth;
    // sea points (ISEA order) and their wavenumbers and group speeds
    let sea: Vec<usize> = (0..nx * ny).filter(|&i| g.is_sea(i)).collect();
    let nsea = sea.len();
    let mut cgk = vec![vec![0f64; nsea]; nk + 2];
    {
        let (mut wn, mut cg) = (vec![0f32; nk + 2], vec![0f32; nk + 2]);
        for (isea, &ip) in sea.iter().enumerate() {
            let zb = g.zb[ip] as f32;
            node_wavenumbers(&sp, (-zb).max(0.0), zb, s.dmin as f32, &mut wn, &mut cg);
            for ik in 0..nk + 2 {
                cgk[ik][isea] = cg[ik] as f64;
            }
        }
    }
    // spectra of the restart (mesh order) on the whole grid, per component
    let ij: Vec<u32> = std::fs::read(mesh.join("raw").join("ij.u32"))?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let rs = Restart::read(Path::new(&get("--restart")?))?;
    anyhow::ensure!(rs.nsea == ij.len() && rs.nspec == nspec, "restart {} x {} against mesh {} x {nspec}", rs.nsea, rs.nspec, ij.len());
    anyhow::ensure!(rs.time == start, "restart at {}, --start {start}", rs.time);
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
        let next_hour = (t / 3600 + 1) * 3600;
        let dtg = (dtmax as i64).min(next_hour - t);
        t += dtg;
        let secs_of_day = (t.rem_euclid(86400)) as f64;
        fields.par_iter_mut().enumerate().for_each(|(isp, f)| {
            let ik = isp / nth;
            xyp3(&pg, isp, dtg as f64, secs_of_day, f, &cgk[ik + 1], None, None);
        });
        steps += 1;
        if t % 3600 == 0 {
            write_hs(Time::from_secs(t), &fields)?;
        }
    }
    println!("{steps} global steps in {:.1} s", t1.elapsed().as_secs_f64());
    Ok(())
}
