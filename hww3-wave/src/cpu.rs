//! CPU transcription of `W3KTP3`, `W3QCK1` and `W3QCK2` (WW3 `w3pro3md`,
//! `w3uqckmd`) in single precision, keeping WW3's storage (`VQ` with the
//! ghost bins, `MAPTH2`, `MAPWN2`, 1-based offsets), to check the GPU
//! kernels of the `prop` module, which are written differently (one unit per
//! line of the spectrum), and of `PDLIB_W3XYPUG` with currents and
//! open-boundary spectra.

use crate::spec::Spec;

/// Inputs of `W3KTP3` for one node.
pub struct Ktp3Node<'a> {
    /// Group speeds `CG(0:NK+1)` (m/s).
    pub cg: &'a [f32],
    /// Wavenumbers `WN(0:NK+1)` (rad/m).
    pub wn: &'a [f32],
    /// Water depth (m).
    pub depth: f32,
    /// Great-circle turning factor `CTHG0` (`-tan(lat) / RADIUS`, 1/m).
    pub cthg0: f32,
    /// Depth gradient `∂d/∂x` (m/m).
    pub dddx: f32,
    /// Depth gradient `∂d/∂y` (m/m).
    pub dddy: f32,
    /// Eastward current (m/s).
    pub cx: f32,
    /// Northward current (m/s).
    pub cy: f32,
    /// Current gradient `∂u/∂x` (1/s).
    pub dcxdx: f32,
    /// Current gradient `∂u/∂y` (1/s).
    pub dcxdy: f32,
    /// Current gradient `∂v/∂x` (1/s).
    pub dcydx: f32,
    /// Current gradient `∂v/∂y` (1/s).
    pub dcydy: f32,
}

fn sign(a: f32, b: f32) -> f32 {
    if b >= 0.0 { a.abs() } else { -a.abs() }
}

/// `INT(x)` of Fortran (truncation toward zero).
fn int(x: f32) -> i64 {
    x.trunc() as i64
}

/// `W3QCK1` on `q` indexed as Fortran `Q(1-MY : MY*(MX+2))` (offset `MY`),
/// `cfll` as `CFLL(1 : MY*(MX+1))` (offset 1).
#[allow(clippy::too_many_arguments)]
fn w3qck1(mx: usize, my: usize, nx: usize, ny: usize, cfll: &mut [f32], q: &mut [f32], close: bool, inc: i64, mapact: &[usize], mapbou: &[usize], nb0: usize, nb1: usize, nb2: usize) {
    let qo = my as i64; // Q(j) -> q[j + qo - 1 + 1] ... index j maps to q[(j - (1 - my)) as usize]
    let qi = |j: i64| (j - (1 - my as i64)) as usize;
    let _ = qo;
    let ci = |j: i64| (j - 1) as usize;
    let mut fla = vec![0f32; my * mx + my]; // FLA(1-MY : MY*MX)
    let fi = |j: i64| (j - (1 - my as i64)) as usize;
    if close {
        let iad00 = -(my as i64);
        let iad02 = my as i64;
        let iadn0 = iad00 + (my * nx) as i64;
        let iadn1 = (my * nx) as i64;
        let iadn2 = iad02 + (my * nx) as i64;
        for iy in 1..=ny as i64 {
            q[qi(iy + iad00)] = q[qi(iy + iadn0)];
            q[qi(iy + iadn1)] = q[qi(iy)];
            q[qi(iy + iadn2)] = q[qi(iy + iad02)];
            cfll[ci(iy + iadn1)] = cfll[ci(iy)];
        }
    }
    for ip in 0..nb0 {
        let ixy = mapbou[ip] as i64;
        let cfl = 0.5 * (cfll[ci(ixy)] + cfll[ci(ixy + inc)]);
        let ixyc = ixy - inc * int(0f32.min(sign(1.1, cfl)));
        let mut qb = 0.5 * ((1.0 - cfl) * q[qi(ixy + inc)] + (1.0 + cfl) * q[qi(ixy)])
            - (1.0 - cfl * cfl) / 6.0 * (q[qi(ixyc - inc)] - 2.0 * q[qi(ixyc)] + q[qi(ixyc + inc)]);
        let ixyu = ixyc - inc * int(sign(1.1, cfl));
        let ixyd = 2 * ixyc - ixyu;
        let dq = q[qi(ixyd)] - q[qi(ixyu)];
        let dqnz = sign(1e-15f32.max(dq.abs()), dq);
        let mut qcn = (q[qi(ixyc)] - q[qi(ixyu)]) / dqnz;
        qcn = 1.1f32.min((-0.1f32).max(qcn));
        let mut qbn = ((qb - q[qi(ixyu)]) / dqnz).max(qcn);
        qbn = qbn.min(1.0).min(qcn / 1e-10f32.max(cfl.abs()));
        let qbr = q[qi(ixyu)] + qbn * dq;
        let cfac = int(2.0 * (qcn - 0.5).abs()) as f32;
        qb = (1.0 - cfac) * qbr + cfac * q[qi(ixyc)];
        fla[fi(ixy)] = cfl * qb;
    }
    for ip in nb0..nb1 {
        let ixy = mapbou[ip] as i64;
        let cfl = cfll[ci(ixy)];
        let ixyc = ixy - inc * int(0f32.min(sign(1.1, cfl)));
        fla[fi(ixy)] = cfl * q[qi(ixyc)];
    }
    for ip in nb1..nb2 {
        let ixy = mapbou[ip] as i64;
        let cfl = cfll[ci(ixy + inc)];
        let ixyc = ixy - inc * int(0f32.min(sign(1.1, cfl)));
        fla[fi(ixy)] = cfl * q[qi(ixyc)];
    }
    if close {
        let iad00 = -(my as i64);
        let iadn0 = iad00 + (my * nx) as i64;
        for iy in 1..=ny as i64 {
            fla[fi(iy + iad00)] = fla[fi(iy + iadn0)];
        }
    }
    for &ixy in mapact {
        let ixy = ixy as i64;
        q[qi(ixy)] = 0f32.max(q[qi(ixy)] + fla[fi(ixy - inc)] - fla[fi(ixy)]);
    }
}

/// `W3QCK2` (variable grid): `velo`, `dx1` as `(1 : MY*(MX+1))`, `dx2` as `(1-MY : MY*(MX+1))`.
#[allow(clippy::too_many_arguments)]
fn w3qck2(my: usize, velo: &[f32], dt: f32, dx1: &[f32], dx2: &[f32], q: &mut [f32], inc: i64, mapact: &[usize], mapbou: &[usize], nb0: usize, nb1: usize, nb2: usize) {
    let qi = |j: i64| (j - (1 - my as i64)) as usize;
    let ci = |j: i64| (j - 1) as usize;
    let di = |j: i64| (j - (1 - my as i64)) as usize;
    let mut fla = vec![0f32; q.len()];
    for ip in 0..nb0 {
        let ixy = mapbou[ip] as i64;
        let vel = 0.5 * (velo[ci(ixy)] + velo[ci(ixy + inc)]);
        let cfl = dt * vel / dx2[di(ixy)];
        let ixyc = ixy - inc * int(0f32.min(sign(1.1, cfl)));
        let mut qb = 0.5 * ((1.0 - cfl) * q[qi(ixy + inc)] + (1.0 + cfl) * q[qi(ixy)])
            - dx2[di(ixy)].powi(2) / dx1[ci(ixyc)] * (1.0 - cfl * cfl) / 6.0
                * ((q[qi(ixyc + inc)] - q[qi(ixyc)]) / dx2[di(ixyc)] - (q[qi(ixyc)] - q[qi(ixyc - inc)]) / dx2[di(ixyc - inc)]);
        let ixyu = ixyc - inc * int(sign(1.1, cfl));
        let ixyd = 2 * ixyc - ixyu;
        let dq = q[qi(ixyd)] - q[qi(ixyu)];
        let dqnz = sign(1e-15f32.max(dq.abs()), dq);
        let mut qcn = (q[qi(ixyc)] - q[qi(ixyu)]) / dqnz;
        qcn = 1.1f32.min((-0.1f32).max(qcn));
        let mut qbn = ((qb - q[qi(ixyu)]) / dqnz).max(qcn);
        qbn = qbn.min(1.0).min(qcn / 1e-10f32.max(cfl.abs()));
        let qbr = q[qi(ixyu)] + qbn * dq;
        let cfac = int(2.0 * (qcn - 0.5).abs()) as f32;
        qb = (1.0 - cfac) * qbr + cfac * q[qi(ixyc)];
        fla[qi(ixy)] = vel * qb;
    }
    for ip in nb0..nb1 {
        let ixy = mapbou[ip] as i64;
        let vel = velo[ci(ixy)];
        let ixyc = ixy - inc * int(0f32.min(sign(1.1, vel)));
        fla[qi(ixy)] = vel * q[qi(ixyc)];
    }
    for ip in nb1..nb2 {
        let ixy = mapbou[ip] as i64;
        let vel = velo[ci(ixy + inc)];
        let ixyc = ixy - inc * int(0f32.min(sign(1.1, vel)));
        fla[qi(ixy)] = vel * q[qi(ixyc)];
    }
    for &ixy in mapact {
        let ixy = ixy as i64;
        q[qi(ixy)] = 0f32.max(q[qi(ixy)] + dt / dx1[ci(ixy)] * (fla[qi(ixy - inc)] - fla[qi(ixy)]));
    }
}

/// `W3KTP3` for one node; `va` in WW3 order (`isp = ith + ik nth`).
///
/// `facth` and `fack` are WW3's `FACTH` and `FACK` (time step over the
/// direction and frequency increments), `ctmax` the CFL cap, `flcur`,
/// `flcth` and `flck` the switches of the current terms, of refraction and
/// of the frequency shift, and `itstep` the step counter whose parity sets
/// the order of the two advections.
#[allow(clippy::too_many_arguments)]
pub fn ktp3(sp: &Spec, n: &Ktp3Node, va: &mut [f32], facth: f32, fack: f32, ctmax: f32, flcur: bool, flcth: bool, flck: bool, itstep: u64) {
    let (nk, nth) = (sp.nk, sp.nth);
    let nk2 = nk + 2;
    let nspec = nk * nth;
    let mapth2: Vec<usize> = (0..nspec).map(|isp| (isp / nth + 2) + (isp % nth) * nk2).collect();
    let mut mapwn2 = Vec::with_capacity(nspec + nth);
    for ik in 1..nk {
        for ith in 1..=nth {
            mapwn2.push((ik + 1) + (ith - 1) * nk2);
        }
    }
    for ith in 1..=nth {
        mapwn2.push(nk + 1 + (ith - 1) * nk2);
    }
    for ith in 1..=nth {
        mapwn2.push(1 + (ith - 1) * nk2);
    }
    let depth = n.depth;
    let mut dsdd = vec![0f32; nk + 2];
    for ik in 0..nk + 2 {
        dsdd[ik] = if depth * n.wn[ik] < 5.0 { (n.cg[ik] * n.wn[ik] - 0.5 * sp.sig[ik]).max(0.0) / depth } else { 0.0 };
    }
    // VQ(-NK-1 : NK2*(NTH+2)) -> index j + NK2 (since 1 - MY = -NK-1)
    let mut vq = vec![0f32; nk2 * (nth + 2) + nk2];
    let qi = |j: usize| j + nk2 - 1;
    for isp in 0..nspec {
        vq[qi(mapth2[isp])] = va[isp];
    }
    let mut vcflt = vec![0f32; nk2 * (nth + 1)];
    if flcth {
        let fdg = facth * n.cthg0;
        let mut frk = vec![0f32; nk + 1];
        let mut frg = vec![0f32; nk + 1];
        for ik in 1..=nk {
            frk[ik] = facth * dsdd[ik] / n.wn[ik];
            frg[ik] = fdg * n.cg[ik];
        }
        if flcur {
            let dcyx = facth * n.dcydx;
            let dcxxyy = facth * (n.dcxdx - n.dcydy);
            let dcxy = facth * n.dcxdy;
            for isp in 0..nspec {
                let th = isp % nth;
                vcflt[mapth2[isp] - 1] = sp.es2[th] * dcyx + sp.esc[th] * dcxxyy - sp.ec2[th] * dcxy;
            }
        }
        for isp in 0..nspec {
            let (th, ik) = (isp % nth, isp / nth + 1);
            let v = vcflt[mapth2[isp] - 1] + frg[ik] * sp.ecos[th] + frk[ik] * (sp.esin[th] * n.dddx - sp.ecos[th] * n.dddy);
            vcflt[mapth2[isp] - 1] = sign(v.abs().min(ctmax), v);
        }
    }
    let mut cflk = vec![0f32; nk2 * nth];
    let mut db = vec![0f32; nk2 * (nth + 1)];
    let mut dm = vec![0f32; nk2 * (nth + 2)]; // DM(NK2, 0:NTH+1) -> dx2 offset handled below
    if flck {
        let fkd = n.cx * n.dddx + n.cy * n.dddy;
        let dcxx = -n.dcxdx;
        let dcxyyx = -(n.dcxdy + n.dcydx);
        let dcyy = -n.dcydy;
        let fkc: Vec<f32> = (0..nth).map(|th| sp.ec2[th] * dcxx + sp.esc[th] * dcxyyx + sp.es2[th] * dcyy).collect();
        // DB(IK+1, ITH), DM(IK+1, ITH) stored column-major with leading dimension NK2
        for ith in 0..nth {
            for ik in 0..=nk {
                db[ith * nk2 + ik] = sp.dsip[ik] / n.cg[ik];
                dm[(ith + 1) * nk2 + ik] = n.wn[ik + 1] - n.wn[ik];
            }
            db[ith * nk2 + nk + 1] = sp.dsip[nk + 1] / n.cg[nk + 1];
            dm[(ith + 1) * nk2 + nk + 1] = 0.0;
        }
        for ik in 0..=nk + 1 {
            let fkd0 = fkd / n.cg[ik] * dsdd[ik];
            let velfac = fack / db[ik];
            for ith in 0..nth {
                let v = (fkd0 + n.wn[ik] * fkc[ith]) * velfac;
                cflk[ith * nk2 + ik] = sign(v.abs().min(ctmax), v) / velfac;
            }
        }
    }
    let do_k = |vq: &mut Vec<f32>| {
        for ith in 0..nth {
            vq[qi(nk + 2 + ith * nk2)] = sp.fachfa * vq[qi(nk + 1 + ith * nk2)];
        }
        // DX2 = DM viewed from (1 - MY): DM(NK2, 0:NTH+1) starts at column 0
        w3qck2(nk2, &cflk, fack, &db, &dm, vq, 1, &mapth2, &mapwn2, nspec - nth, nspec, nspec + nth);
    };
    let do_th = |vq: &mut Vec<f32>, vcflt: &mut Vec<f32>| {
        w3qck1(nth, nk2, nth, nk2, vcflt, vq, true, nk2 as i64, &mapth2, &mapth2, nspec, nspec, nspec);
    };
    if itstep % 2 == 0 {
        if flck {
            do_k(&mut vq);
        }
        if flcth {
            do_th(&mut vq, &mut vcflt);
        }
    } else {
        if flcth {
            do_th(&mut vq, &mut vcflt);
        }
        if flck {
            do_k(&mut vq);
        }
    }
    for isp in 0..nspec {
        va[isp] = vq[qi(mapth2[isp])];
    }
}

/// Inputs of the CPU transcription of `PDLIB_W3XYPUG` / `PDLIB_W3XYPFSN2`
/// for one frequency.
pub struct GeoFreq<'a> {
    /// Vertices per triangle (`3 ne`, counter-clockwise).
    pub ine: &'a [u32],
    /// `PDLIB_IEN` (`6 ne`).
    pub ien: &'a [f64],
    /// `PDLIB_SI` per node.
    pub si: &'a [f64],
    /// `IOBP` per node.
    pub iobp: &'a [u8],
    /// `IOBPA` per node.
    pub iobpa: &'a [u8],
    /// `IOBPD` bits per node (bit k for direction k).
    pub iobpd: &'a [u64],
    /// `IOBDP` per node (0 dry, 1 wet).
    pub iobdp: &'a [u8],
    /// `CG(IK, node)` (m/s).
    pub cg: &'a [f32],
    /// `CLATS`: cosine of the latitude per node.
    pub clats: &'a [f32],
    /// Eastward current `CX` per node (m/s).
    pub cx: &'a [f32],
    /// Northward current `CY` per node (m/s).
    pub cy: &'a [f32],
    /// `ECOS` of the directions.
    pub ecos: &'a [f32],
    /// `ESIN` of the directions.
    pub esin: &'a [f32],
    /// `FACX = 1 / (DERA RADIUS)`.
    pub facx: f32,
    /// Global time step (s).
    pub dt: f32,
    /// Boundary: node of each point, `BBPI0`, `BBPIN` of this frequency (`nbnd nth`).
    pub bnode: &'a [u32],
    /// `BBPI0`: boundary spectra at the start of the boundary interval.
    pub bb0: &'a [f32],
    /// `BBPIN`: boundary spectra at the end of the boundary interval.
    pub bbn: &'a [f32],
    /// `RD10`: time from the start of the boundary interval to the end of
    /// the step (s).
    pub rd10: f32,
    /// `RD20`: length of the boundary interval (s).
    pub rd20: f32,
}

/// One geographic step of every direction of a frequency, as WW3 writes it
/// (triangle loop with scatter, `KELEM`, `NM`, `FLALL` and `U` in double,
/// `AC` stored in single between sub-iterations). `va` is `[np nth]`;
/// returns `ITER` per direction.
pub fn geo_freq(g: &GeoFreq, va: &mut [f32], nth: usize) -> Vec<u32> {
    let np = g.si.len();
    let ne = g.ine.len() / 3;
    let thr = f32::MIN_POSITIVE as f64;
    let mut iters = vec![0u32; nth];
    let mut bpos = vec![usize::MAX; np];
    for (b, &i) in g.bnode.iter().enumerate() {
        bpos[i as usize] = b;
    }
    for ith in 0..nth {
        let ccos = g.facx * g.ecos[ith];
        let csin = g.facx * g.esin[ith];
        let mut ac = vec![0f32; np];
        let mut c = vec![[0f64; 2]; np];
        for i in 0..np {
            ac[i] = va[i * nth + ith] / g.cg[i] * g.clats[i];
            let mut vx = ccos * g.cg[i] / g.clats[i];
            let mut vy = csin * g.cg[i];
            if g.iobp[i] > 0 {
                vx += g.facx * g.cx[i] / g.clats[i];
                vy += g.facx * g.cy[i];
            }
            c[i] = [(vx * g.iobdp[i] as f32) as f64, (vy * g.iobdp[i] as f32) as f64];
        }
        let mut kelem = vec![[0f64; 3]; ne];
        let mut nm = vec![0f64; ne];
        let mut flall = vec![[0f64; 3]; ne];
        for e in 0..ne {
            let (i1, i2, i3) = (g.ine[3 * e] as usize, g.ine[3 * e + 1] as usize, g.ine[3 * e + 2] as usize);
            let ien = &g.ien[6 * e..6 * e + 6];
            let l0 = (c[i1][0] + c[i2][0] + c[i3][0]) / 6.0;
            let l1 = (c[i1][1] + c[i2][1] + c[i3][1]) / 6.0;
            let k = [l0 * ien[0] + l1 * ien[1], l0 * ien[2] + l1 * ien[3], l0 * ien[4] + l1 * ien[5]];
            nm[e] = -1.0 / (-thr).min(k.iter().map(|&x| x.min(0.0)).sum::<f64>());
            let kp = [k[0].max(0.0), k[1].max(0.0), k[2].max(0.0)];
            kelem[e] = kp;
            let dotn = |i: usize, q: usize| c[i][0] * ien[2 * q] + c[i][1] * ien[2 * q + 1];
            let (fl11, fl12) = (dotn(i2, 0), dotn(i3, 0));
            let (fl21, fl22) = (dotn(i3, 1), dotn(i1, 1));
            let (fl31, fl32) = (dotn(i1, 2), dotn(i2, 2));
            let (fl111, fl112) = (2.0 * fl11 + fl12, 2.0 * fl12 + fl11);
            let (fl211, fl212) = (2.0 * fl21 + fl22, 2.0 * fl22 + fl21);
            let (fl311, fl312) = (2.0 * fl31 + fl32, 2.0 * fl32 + fl31);
            flall[e] = [(fl311 + fl212) / 6.0 + kp[0], (fl111 + fl312) / 6.0 + kp[1], (fl211 + fl112) / 6.0 + kp[2]];
        }
        // ITER (LCALC)
        let mut kksum = vec![0f64; np];
        for e in 0..ne {
            for q in 0..3 {
                kksum[g.ine[3 * e + q] as usize] += kelem[e][q];
            }
        }
        let mut dtmax = 10.0e10f64;
        for i in 0..np {
            if g.iobp[i] == 1 {
                dtmax = dtmax.min(g.si[i] / (1e-9f64).max(kksum[i] * g.iobdp[i] as f64));
            }
        }
        let cflxy = g.dt as f64 / dtmax;
        let rest = (cflxy % 1.0).abs();
        let nint = cflxy.round().abs() as u32;
        let iter = if rest < thr { nint } else if rest < 0.5 { nint + 1 } else { nint };
        let iter = iter.max(1);
        iters[ith] = iter;
        let dtsi: Vec<f64> = (0..np).map(|i| g.dt as f64 / iter as f64 / g.si[i]).collect();
        for it in 1..=iter {
            let u: Vec<f64> = ac.iter().map(|&x| x as f64).collect();
            let mut st = vec![0f64; np];
            for e in 0..ne {
                let ni = [g.ine[3 * e] as usize, g.ine[3 * e + 1] as usize, g.ine[3 * e + 2] as usize];
                let ut = nm[e] * (flall[e][0] * u[ni[0]] + flall[e][1] * u[ni[1]] + flall[e][2] * u[ni[2]]);
                for q in 0..3 {
                    st[ni[q]] += kelem[e][q] * (u[ni[q]] - ut);
                }
            }
            for i in 0..np {
                let m = (((g.iobpd[i] >> ith) & 1) as f64) * g.iobdp[i] as f64;
                let v = (u[i] - dtsi[i] * st[i] * (1 - g.iobpa[i]) as f64).max(0.0) * m;
                ac[i] = v as f32;
            }
            let mut rd1 = g.rd10 - g.dt * (iter - it) as f32 / iter as f32;
            let mut rd2 = g.rd20;
            if rd2 > 0.001 {
                rd2 = (rd1 / rd2).clamp(0.0, 1.0);
                rd1 = 1.0 - rd2;
            } else {
                rd1 = 0.0;
                rd2 = 1.0;
            }
            for (b, &i) in g.bnode.iter().enumerate() {
                let i = i as usize;
                ac[i] = (rd1 * g.bb0[b * nth + ith] + rd2 * g.bbn[b * nth + ith]) * g.clats[i];
            }
        }
        let _ = &bpos;
        for i in 0..np {
            va[i * nth + ith] = (g.cg[i] / g.clats[i] * ac[i]).max(0.0);
        }
    }
    iters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{group_speed, Geometry};
    use crate::spec::{node_wavenumbers, DERA, RADIUS};
    use crate::testgrid::{lattice, uniform};

    fn node<'a>(cg: &'a [f32], wn: &'a [f32], depth: f32, dddx: f32) -> Ktp3Node<'a> {
        Ktp3Node {
            cg,
            wn,
            depth,
            cthg0: 0.0,
            dddx,
            dddy: 0.0,
            cx: 0.0,
            cy: 0.0,
            dcxdx: 0.0,
            dcxdy: 0.0,
            dcydx: 0.0,
            dcydy: 0.0,
        }
    }

    /// A smooth positive spectrum in WW3 order (`isp = ith + ik nth`).
    fn spectrum(sp: &Spec) -> Vec<f32> {
        let mut va = vec![0f32; sp.nk * sp.nth];
        for ik in 0..sp.nk {
            for ith in 0..sp.nth {
                let th = ith as f32 * sp.dth;
                va[ith + ik * sp.nth] = (1.0 + 0.6 * (th - 0.8).cos()) * (1.0 + ik as f32) / sp.nk as f32;
            }
        }
        va
    }

    #[test]
    fn ktp3_leaves_deep_still_water_unchanged() {
        let sp = Spec::new(8, 12, 0.0373, 1.1, 0.0);
        let (mut wn, mut cg) = (vec![0f32; sp.nk + 2], vec![0f32; sp.nk + 2]);
        node_wavenumbers(&sp, 4000.0, -4000.0, 1.0, &mut wn, &mut cg);
        let n = node(&cg, &wn, 4000.0, 1e-3);
        let va0 = spectrum(&sp);
        for itstep in [0, 1] {
            let mut va = va0.clone();
            ktp3(&sp, &n, &mut va, 300.0 / sp.dth, 1.0, 0.7, false, true, true, itstep);
            assert_eq!(va, va0);
        }
    }

    #[test]
    fn refraction_conserves_the_energy_of_each_frequency() {
        let sp = Spec::new(8, 12, 0.0373, 1.1, 0.0);
        let (mut wn, mut cg) = (vec![0f32; sp.nk + 2], vec![0f32; sp.nk + 2]);
        let depth = 15.0;
        node_wavenumbers(&sp, depth, -depth, 1.0, &mut wn, &mut cg);
        let n = node(&cg, &wn, depth, 5e-4);
        let va0 = spectrum(&sp);
        let mut va = va0.clone();
        ktp3(&sp, &n, &mut va, 300.0 / sp.dth, 1.0, 0.7, false, true, false, 0);
        let mut changed = 0f32;
        for ik in 0..sp.nk {
            let line = |v: &[f32]| v[ik * sp.nth..(ik + 1) * sp.nth].iter().sum::<f32>();
            let (before, after) = (line(&va0), line(&va));
            assert!((after / before - 1.0).abs() < 1e-5, "frequency {ik}: {before} -> {after}");
            for ith in 0..sp.nth {
                changed = changed.max((va[ith + ik * sp.nth] - va0[ith + ik * sp.nth]).abs());
            }
        }
        assert!(changed > 1e-3, "refraction should turn the spectrum");
        assert!(va.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn geographic_step_matches_the_reference() {
        // PDLIB_W3XYPUG (single-precision state, no current, no boundary
        // spectra) against PDLIB_W3XYPFSN2 in double precision
        let grid = lattice(12, 10, 20.0, 15.0, false);
        let geo = Geometry::new(&grid);
        let (np, nth) = (geo.np, geo.nth);
        let clats: Vec<f32> = grid.lat.iter().map(|&y| (y as f32 * DERA).cos()).collect();
        let ecos: Vec<f32> = geo.ecos.iter().map(|&v| v as f32).collect();
        let esin: Vec<f32> = geo.esin.iter().map(|&v| v as f32).collect();
        let zero = vec![0f32; np];
        let u = uniform(np * nth, 7);
        for ik in 0..geo.nk {
            let cg: Vec<f32> =
                grid.depth.iter().map(|&d| group_speed(geo.freqs[ik], d.max(grid.spec.dmin)) as f32).collect();
            let g = GeoFreq {
                ine: &geo.ine,
                ien: &geo.ien,
                si: &geo.si,
                iobp: &geo.iobp,
                iobpa: &geo.iobpa,
                iobpd: &geo.iobpd,
                iobdp: &geo.iobdp,
                cg: &cg,
                clats: &clats,
                cx: &zero,
                cy: &zero,
                ecos: &ecos,
                esin: &esin,
                facx: 1.0 / (DERA * RADIUS),
                dt: crate::geom::DT as f32,
                bnode: &[],
                bb0: &[],
                bbn: &[],
                rd10: 0.0,
                rd20: 0.0,
            };
            // W3XYPUG propagates VA CLATS / CG: start it from u
            let mut va: Vec<f32> = (0..np * nth).map(|j| (u[j] * cg[j / nth] as f64 / clats[j / nth] as f64) as f32).collect();
            let iters = geo_freq(&g, &mut va, nth);
            assert_eq!(iters, geo.iter[ik * nth..(ik + 1) * nth].to_vec(), "ITER of frequency {ik}");
            let want = crate::reference::step(&geo, &u, ik, true);
            let scale = want.iter().fold(0f64, |m, &x| m.max(x));
            for j in 0..np * nth {
                let got = va[j] as f64 * clats[j / nth] as f64 / cg[j / nth] as f64;
                assert!((got - want[j]).abs() < 1e-5 * scale, "frequency {ik}, value {j}: {got} against {}", want[j]);
            }
        }
    }
}
