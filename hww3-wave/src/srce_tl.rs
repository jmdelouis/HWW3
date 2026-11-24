//! Tangent-linear CubeCL kernel of the source terms (milestone M3): the
//! `W3SRCE` integration of [`crate::srce`] for the full ST4 (stress tables,
//! cumulative breaking term) differentiated in forward mode with respect to
//! one direction: a tangent of the spectrum `dA` (from the propagation), of
//! the wind (`dU10`, `dU10D`) and of the source parameters (`dpar`, with the
//! derivative of the tail-stress table for `BETAMAX`).
//!
//! Every line of the value kernel is kept, with its derivative next to it;
//! the values are those of `srce`. The discrete choices (number of dynamic
//! steps, `NKH`, the shaving limiter, the branches of `MIN`, `MAX` and of the
//! tables) are taken from the values and are piecewise constant, so their
//! derivative is zero: the tangent is exact where the model is smooth.
//!
//! Tangents sit next to the values: `ad` for the spectrum, `vsd`, `vdd` for
//! the source sums, `ued`, `b1d`, `b2d`, `dnd` for the DIA and the
//! saturation, `kd` and `sd` for the per-frequency and scalar slots of `kb`
//! and `sc`, `drd` for the per-direction stress partials.

// The `#[cube]` macro generates, for every kernel, a launch struct and
// functions that carry no documentation; the kernels themselves are
// documented below.
#![allow(missing_docs)]

use crate::sources::*;
use crate::srce::*;
use cubecl::prelude::*;

/// Derivative of [`table`] at `x` along `dx`.
#[cube]
fn table_d<F: Float>(tab: &[F], n: usize, x: F, dx: F) -> F {
    let size = n - 1;
    let xi = u32::cast_from(max(x.trunc(), F::new(0.0f32))) as usize;
    let ind = select(xi < size - 1, xi, size - 1);
    let f = x - F::cast_from(ind as u32);
    select(f < F::new(1.0f32), (tab[ind + 1] - tab[ind]) * dx, F::new(0.0f32))
}

/// `CALC_USTAR` with `SINTABLE = 1` and its tangent: u* from `TAUT`, then
/// z0; writes the values to `sc`, the tangents to `sd`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn calc_ustar_tl<F: Float>(par: &[F], taut: &[F], sc: &mut Shared<[F]>, sd: &mut Shared<[F]>, u: F, du: F, tauw: F, dtauw: F) {
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let grav = par[P_GRAV];
    let kappa = par[P_KAPPA];
    let chath = par[P_AALPHA];
    let itm = u32::cast_from(par[P_ITAUMAX]) as usize;
    let jum = u32::cast_from(par[P_JUMAX]) as usize;
    let tmax = par[P_TAUWMAX];
    let tl = max(min(tauw, tmax), zero);
    let dtl = select(tauw < tmax && tauw > zero, dtauw, zero);
    let xi = tl.sqrt() / par[P_DELTAUW];
    let dxi = select(tl > zero, F::new(0.5f32) * dtl / tl.sqrt() / par[P_DELTAUW], zero);
    let xif = u32::cast_from(xi.trunc()) as usize;
    let ind = select(xif < itm - 1, xif, itm - 1);
    let fi = xi - F::cast_from(ind as u32);
    let deli1 = min(one, fi);
    let ddeli1 = select(fi < one, dxi, zero);
    let deli2 = one - deli1;
    let xj = u / par[P_DELU];
    let dxj = du / par[P_DELU];
    let xjf = u32::cast_from(xj.trunc()) as usize;
    let j = select(xjf < jum - 1, xjf, jum - 1);
    let fj = xj - F::cast_from(j as u32);
    let delj1 = min(one, fj);
    let ddelj1 = select(fj < one, dxj, zero);
    let delj2 = one - delj1;
    let n = itm + 1;
    let t00 = taut[ind + j * n];
    let t10 = taut[ind + 1 + j * n];
    let t01 = taut[ind + (j + 1) * n];
    let t11 = taut[ind + 1 + (j + 1) * n];
    let lo = t00 * deli2 + t10 * deli1;
    let hi = t01 * deli2 + t11 * deli1;
    let ust = lo * delj2 + hi * delj1;
    let dust = ((t10 - t00) * delj2 + (t11 - t01) * delj1) * ddeli1 + (hi - lo) * ddelj1;
    let mut z0 = chath * F::new(0.000001f32) / grav;
    let mut dz0 = zero;
    let mut charn = chath;
    if ust > zero {
        let r = u / ust;
        let dr = (du * ust - u * dust) / (ust * ust);
        let s = min(r, F::new(100.0f32));
        let ds = select(r < F::new(100.0f32), dr, zero);
        z0 = par[P_ZZWND] * (-kappa * s).exp();
        dz0 = -kappa * z0 * ds;
        if ust > F::new(0.001f32) {
            charn = grav * z0 / (ust * ust);
        }
    }
    if par[P_CAPCHNK] == one {
        charn = max(min(charn, F::new(0.09f32)), chath);
    }
    sc[S_USTAR] = ust;
    sc[S_Z0] = z0;
    sc[S_CHARN] = charn;
    sd[S_USTAR] = dust;
    sd[S_Z0] = dz0;
}

/// `W3SPR4` and its tangent (mean parameters, then u* from `TAUT`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn spr4_tl<F: Float>(
    par: &[F],
    taut: &[F],
    sig: &[F],
    dden: &[F],
    a: &Shared<[F]>,
    ad: &Shared<[F]>,
    ws: &Shared<[F]>,
    kb: &mut Shared<[F]>,
    kd: &mut Shared<[F]>,
    sc: &mut Shared<[F]>,
    sd: &mut Shared<[F]>,
    u: F,
    du: F,
    tauwx: F,
    dtauwx: F,
    tauwy: F,
    dtauwy: F,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
) {
    let t = UNIT_POS as usize;
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    if t < nk {
        let mut eb = zero;
        let mut eb2 = zero;
        let mut amax = zero;
        let mut deb = zero;
        let mut deb2 = zero;
        for th in 0..nth {
            let x = a[t * nth + th];
            let dx = ad[t * nth + th];
            eb += x;
            eb2 += x * ws[t * nth + th];
            amax = max(amax, x);
            deb += dx;
            deb2 += dx * ws[t * nth + th];
        }
        kb[K_EB * nk + t] = eb;
        kb[K_EB2 * nk + t] = eb2;
        kb[K_AMAX * nk + t] = amax;
        kd[K_EB * nk + t] = deb;
        kd[K_EB2 * nk + t] = deb2;
    }
    sync_cube();
    if t == 0 {
        let tw = par[P_WWNMEANPTAIL];
        let wp = par[P_WWNMEANP];
        let two = F::new(2.0f32);
        let (mut emean, mut fmean, mut fmean1, mut wnmean, mut emeanws, mut fmeanws) = (zero, zero, zero, zero, zero, zero);
        let (mut de, mut dfm, mut df1, mut dwn, mut dews, mut dfws) = (zero, zero, zero, zero, zero, zero);
        let mut amax = zero;
        for ik in 0..nk {
            let f = dden[ik] / kb[K_CG * nk + ik];
            let eb = kb[K_EB * nk + ik] * f;
            let eb2 = kb[K_EB2 * nk + ik] * f;
            let deb = kd[K_EB * nk + ik] * f;
            let deb2 = kd[K_EB2 * nk + ik] * f;
            let s = sig[ik + 1];
            let sp = s.powf(two * tw);
            let wq = kb[K_WN * nk + ik].powf(wp);
            emean += eb;
            fmean += eb / s;
            fmean1 += eb * sp;
            wnmean += eb * wq;
            emeanws += eb2;
            fmeanws += eb2 * sp;
            de += deb;
            dfm += deb / s;
            df1 += deb * sp;
            dwn += deb * wq;
            dews += deb2;
            dfws += deb2 * sp;
            amax = max(amax, kb[K_AMAX * nk + ik]);
        }
        let ddl = dden[nk - 1];
        let cgl = kb[K_CG * nk + nk - 1];
        let eband = kb[K_EB * nk + nk - 1] * (ddl / cgl) / ddl;
        let deband = kd[K_EB * nk + nk - 1] * (ddl / cgl) / ddl;
        emean += eband * par[P_FTE];
        fmean += eband * par[P_FTF];
        fmean1 += eband * par[P_SSTXFTFTAIL];
        wnmean += eband * par[P_SSTXFTWN];
        de += deband * par[P_FTE];
        dfm += deband * par[P_FTF];
        df1 += deband * par[P_SSTXFTFTAIL];
        dwn += deband * par[P_SSTXFTWN];
        let eband2 = kb[K_EB2 * nk + nk - 1] * (ddl / cgl) / ddl;
        let deband2 = kd[K_EB2 * nk + nk - 1] * (ddl / cgl) / ddl;
        emeanws += eband2 * par[P_FTE];
        fmeanws += eband2 * par[P_SSTXFTFTAIL];
        dews += deband2 * par[P_FTE];
        dfws += deband2 * par[P_SSTXFTFTAIL];
        let e7 = F::new(1.0e-7f32);
        // derivatives of MAX(1E-7, x)
        let dem = select(emean > e7, de, zero);
        let em = max(e7, emean);
        let fmv = max(e7, fmean);
        let dfmv = select(fmean > e7, dfm, zero);
        let fmean_o = tpiinv::<F>() * emean / fmv;
        let dfmean_o = tpiinv::<F>() * (de * fmv - emean * dfmv) / (fmv * fmv);
        let q1 = one / (two * tw);
        let mut fmean1_o = tpiinv::<F>() * sig[nk];
        let mut dfmean1_o = zero;
        if fmean1 >= e7 {
            let r = max(e7, fmean1) / em;
            let dr = (df1 * em - fmean1 * dem) / (em * em);
            fmean1_o = tpiinv::<F>() * r.powf(q1);
            dfmean1_o = tpiinv::<F>() * q1 * r.powf(q1 - one) * dr;
        }
        let wv = max(e7, wnmean);
        let dwv = select(wnmean > e7, dwn, zero);
        let r2 = wv / em;
        let dr2 = (dwv * em - wv * dem) / (em * em);
        let wnmean_o = r2.powf(one / wp);
        let dwnmean_o = one / wp * r2.powf(one / wp - one) * dr2;
        let mut fmeanws_o = tpiinv::<F>() * sig[nk];
        let mut dfmeanws_o = zero;
        if fmeanws >= e7 && emeanws >= e7 {
            let r = fmeanws / emeanws;
            let dr = (dfws * emeanws - fmeanws * dews) / (emeanws * emeanws);
            fmeanws_o = tpiinv::<F>() * r.powf(q1);
            dfmeanws_o = tpiinv::<F>() * q1 * r.powf(q1 - one) * dr;
        }
        sc[S_EMEAN] = emean;
        sc[S_FMEAN] = fmean_o;
        sc[S_FMEAN1] = fmean1_o;
        sc[S_WNMEAN] = wnmean_o;
        sc[S_AMAX] = amax;
        sc[S_FMEANWS] = fmeanws_o;
        sd[S_EMEAN] = de;
        sd[S_FMEAN] = dfmean_o;
        sd[S_FMEAN1] = dfmean1_o;
        sd[S_WNMEAN] = dwnmean_o;
        sd[S_FMEANWS] = dfmeanws_o;
        let tauw = (tauwx * tauwx + tauwy * tauwy).sqrt();
        let dtauw = select(tauw > zero, (tauwx * dtauwx + tauwy * dtauwy) / tauw, zero);
        calc_ustar_tl::<F>(par, taut, sc, sd, u, du, tauw, dtauw);
    }
    sync_cube();
}

/// Tangent of [`tauhf2`] from the tangents of its weights `ddeli1` and
/// `ddelj1` (the table itself fixed).
#[cube]
#[allow(clippy::too_many_arguments)]
fn tauhf2_d<F: Float>(tab: &[F], base: usize, ind: usize, j: usize, n1: usize, deli1: F, deli2: F, delj1: F, ddeli1: F, ddelj1: F) -> F {
    let delj2 = F::new(1.0f32) - delj1;
    let a00 = tab[ind + j * n1 + base];
    let a10 = tab[ind + 1 + j * n1 + base];
    let a01 = tab[ind + (j + 1) * n1 + base];
    let a11 = tab[ind + 1 + (j + 1) * n1 + base];
    ((a10 - a00) * delj2 + (a11 - a01) * delj1) * ddeli1 + ((a01 * deli2 + a11 * deli1) - (a00 * deli2 + a10 * deli1)) * ddelj1
}

/// `W3SIN4` with `SINTABLE = 1` and its tangent; `dbeta` is the derivative
/// of `BETAMAX` and `dtauhf` that of the tail-stress table. `beta` is the
/// `BETAMAX` of the node: with tables in `BETAMAX` ([`P_BT_N`] > 1, for
/// `BETAMAX(U)`), the tail stress is the cubic Hermite interpolation in
/// `BETAMAX` of the tables and their derivatives, and its tangent in `BETAMAX`
/// comes from that interpolation (`dtauhf` is then not used).
#[cube]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn sin4_tl<F: Float>(
    par: &[F],
    tauhf: &[F],
    dtauhf: &[F],
    sig: &[F],
    dden: &[F],
    ecos: &[F],
    esin: &[F],
    sig2k: &[F],
    dden2k: &[F],
    fwtab: &[F],
    a: &Shared<[F]>,
    ad: &Shared<[F]>,
    vs: &mut Shared<[F]>,
    vsd: &mut Shared<[F]>,
    vd: &mut Shared<[F]>,
    vdd: &mut Shared<[F]>,
    ws: &mut Shared<[F]>,
    kb: &mut Shared<[F]>,
    kd: &mut Shared<[F]>,
    sc: &mut Shared<[F]>,
    sd: &mut Shared<[F]>,
    dr: &mut Shared<[F]>,
    drd: &mut Shared<[F]>,
    ustar: F,
    dustar: F,
    z0: F,
    dz0: F,
    usdir: F,
    dusdir: F,
    beta: F,
    dbeta: F,
    scale: F,
    accumulate: bool,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
) {
    let t = UNIT_POS as usize;
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let half = F::new(0.5f32);
    let two = F::new(2.0f32);
    let grav = par[P_GRAV];
    let kappa = par[P_KAPPA];
    let nu_air = par[P_NU_AIR];
    let drat = par[P_DRAT];
    let const1 = beta / (kappa * kappa);
    let dconst1 = dbeta / (kappa * kappa);
    let const0 = const1 * drat;
    let dconst0 = dconst1 * drat;
    let tsh = par[P_TTAUWSHELTER].abs();
    if t < nk {
        let mut eb = zero;
        let mut deb = zero;
        for th in 0..nth {
            eb += a[t * nth + th];
            deb += ad[t * nth + th];
        }
        kb[K_EB * nk + t] = eb;
        kd[K_EB * nk + t] = deb;
    }
    sync_cube();
    if t == 0 {
        let mut uo = zero;
        let mut ao = zero;
        let mut duo = zero;
        let mut dao = zero;
        for ik in 0..nk {
            let eb = kb[K_EB * nk + ik];
            let deb = kd[K_EB * nk + ik];
            let s = sig[ik + 1];
            let c = kb[K_CG * nk + ik];
            uo += eb * s * s * dden[ik] / c;
            ao += eb * dden[ik] / c;
            duo += deb * s * s * dden[ik] / c;
            dao += deb * dden[ik] / c;
        }
        let uorb = two * uo.sqrt();
        let duorb = select(uo > zero, duo / uo.sqrt(), zero);
        let ex = one - half * par[P_SSWELLF + 5];
        let aorb1 = two * ao.powf(ex);
        let daorb1 = select(ao > zero, two * ex * ao.powf(ex - one) * dao, zero);
        let re = F::new(4.0f32) * uorb * aorb1 / nu_air;
        let dre = F::new(4.0f32) * (duorb * aorb1 + uorb * daorb1) / nu_air;
        let mut pturb = one;
        let mut pvisc = one;
        let mut dpturb = zero;
        if par[P_SSWELLF + 3] > zero {
            if par[P_SSWELLF + 6] > zero {
                let th = ((re - par[P_SSWELLF + 3]) / par[P_SSWELLF + 6]).tanh();
                let smooth = half * th;
                pturb = half + smooth;
                pvisc = half - smooth;
                dpturb = half * (one - th * th) * dre / par[P_SSWELLF + 6];
            } else if re <= par[P_SSWELLF + 3] {
                pturb = zero;
                pvisc = one;
            } else {
                pturb = one;
                pvisc = zero;
            }
        }
        let mut fw = max(par[P_SSWELLF + 2].abs(), zero);
        let mut dfw = zero;
        let mut fu = zero;
        let mut fud = zero;
        if par[P_SSWELLF + 1] != zero {
            fu = par[P_SSWELLF + 2].abs();
            fud = par[P_SSWELLF + 1];
            let z0visc = F::new(0.1f32) * nu_air / max(ustar, F::new(0.0001f32));
            let dz0visc = select(ustar > F::new(0.0001f32), -z0visc * dustar / ustar, zero);
            let z0r = par[P_ZZ0RAT] * z0;
            let dz0r = par[P_ZZ0RAT] * dz0;
            let z0noz = max(z0visc, z0r);
            let dz0noz = select(z0visc > z0r, dz0visc, dz0r);
            let aorb2 = two * ao.sqrt();
            let daorb2 = select(ao > zero, dao / ao.sqrt(), zero);
            let r = aorb2 / z0noz;
            let drr = (daorb2 * z0noz - aorb2 * dz0noz) / (z0noz * z0noz);
            let rr = max(r, F::new(3.0f32));
            let drv = select(r > F::new(3.0f32), drr, zero);
            let xi = (log10::<F>(rr) - par[P_ABMIN]) / par[P_DELAB];
            let dxi = drv / (rr * F::new(2.3025851f32)) / par[P_DELAB];
            let nfw = u32::cast_from(par[P_NFW]) as usize;
            fw = table::<F>(fwtab, nfw, xi);
            dfw = table_d::<F>(fwtab, nfw, xi, dxi);
        }
        sc[S_UORB] = uorb;
        sc[S_FW] = fw;
        sc[S_FU] = fu;
        sc[S_FUD] = fud;
        sc[S_PTURB] = pturb;
        sc[S_PVISC] = pvisc;
        sd[S_UORB] = duorb;
        sd[S_FW] = dfw;
        sd[S_PTURB] = dpturb;
        let cu = usdir.cos();
        let su = usdir.sin();
        sc[S_TAUX] = ustar * ustar * cu;
        sc[S_TAUY] = ustar * ustar * su;
        sd[S_TAUX] = two * ustar * dustar * cu - ustar * ustar * su * dusdir;
        sd[S_TAUY] = two * ustar * dustar * su + ustar * ustar * cu * dusdir;
        sc[S_SX] = zero;
        sc[S_SY] = zero;
        sc[S_SNX] = zero;
        sc[S_SNY] = zero;
        sd[S_SX] = zero;
        sd[S_SY] = zero;
    }
    sync_cube();
    let uorb = sc[S_UORB];
    let fw = sc[S_FW];
    let fu = sc[S_FU];
    let fud = sc[S_FUD];
    let pturb = sc[S_PTURB];
    let pvisc = sc[S_PVISC];
    let duorb = sd[S_UORB];
    let dfw = sd[S_FW];
    let dpturb = sd[S_PTURB];
    let sw0 = par[P_SSWELLF];
    let sw2 = par[P_SSWELLF + 2];
    let sw4 = par[P_SSWELLF + 4];
    for ik in 0..nk {
        if t == 0 {
            let taupx = sc[S_TAUX] - tsh * sc[S_SX];
            let taupy = sc[S_TAUY] - tsh * sc[S_SY];
            let dtaupx = sd[S_TAUX] - tsh * sd[S_SX];
            let dtaupy = sd[S_TAUY] - tsh * sd[S_SY];
            let p = taupx * taupx + taupy * taupy;
            let dp = two * (taupx * dtaupx + taupy * dtaupy);
            let r4 = p.powf(F::new(0.25f32));
            let dr4 = select(p > zero, F::new(0.25f32) * r4 / p * dp, zero);
            let m = max(ustar, F::new(0.3f32));
            let dm = select(ustar > F::new(0.3f32), dustar, zero);
            let ustp = min(r4, m);
            let dustp = select(r4 < m, dr4, dm);
            let usdirp = taupy.atan2(taupx);
            let dusdirp = select(p > zero, (taupx * dtaupy - taupy * dtaupx) / p, zero);
            let cosu = usdirp.cos();
            let sinu = usdirp.sin();
            sc[S_USTP] = ustp;
            sc[S_COSU] = cosu;
            sc[S_SINU] = sinu;
            sd[S_USTP] = dustp;
            sd[S_COSU] = -sinu * dusdirp;
            sd[S_SINU] = cosu * dusdirp;
        }
        sync_cube();
        if t < nth {
            let ustp = sc[S_USTP];
            let cosu = sc[S_COSU];
            let sinu = sc[S_SINU];
            let dustp = sd[S_USTP];
            let dcosu = sd[S_COSU];
            let dsinu = sd[S_SINU];
            let k = kb[K_WN * nk + ik];
            let c = kb[K_CG * nk + ik];
            let s2 = sig2k[ik];
            let cm = k / s2;
            let ucn = ustp * cm + par[P_ZZALP];
            let ducn = dustp * cm;
            let s = sig[ik + 1];
            let const2 = dden2k[ik] / c * grav / (s / k * drat);
            let cnst = s2 * const0;
            let dcnst = s2 * dconst0;
            let zcn = (k * z0).ln();
            let dzcn = dz0 / z0;
            let swellcoefv = -sw4 * drat * two * k * (two * nu_air * s2).sqrt();
            let swellcoeft = -drat * sw0 * F::new(16.0f32) * s2 * s2 / grav;
            let is = ik * nth + t;
            let coswind = ecos[t] * cosu + esin[t] * sinu;
            let dcw = ecos[t] * dcosu + esin[t] * dsinu;
            let mut dst = zero;
            let mut ddst = zero;
            let mut llws = zero;
            if coswind > F::new(0.01f32) {
                let x = coswind * ucn;
                let dx = dcw * ucn + coswind * ducn;
                let zarg = kappa / x;
                let dzarg = -kappa * dx / (x * x);
                let zlog = zcn + zarg;
                let dzlog = dzcn + dzarg;
                if zlog < zero {
                    let pe = coswind.powf(par[P_SSINTHP]);
                    dst = cnst * zlog.exp() * zlog.powi(4) * ucn * ucn * pe;
                    ddst = dst
                        * (dcnst / cnst + dzlog + F::new(4.0f32) * dzlog / zlog + two * ducn / ucn + par[P_SSINTHP] * dcw / coswind);
                    llws = one;
                }
                if F::new(28.0f32) * cm * ustar * coswind >= one {
                    llws = one;
                }
            }
            if (sw0 != zero && dst < F::new(1.0e-7f32) * s2) || sw2 > zero {
                let dvisc = swellcoefv;
                let dturb = swellcoeft * (fw * uorb + (fu + fud * coswind) * ustp);
                let ddturb = swellcoeft * (dfw * uorb + fw * duorb + fud * dcw * ustp + (fu + fud * coswind) * dustp);
                dst += pturb * dturb + pvisc * dvisc;
                ddst += dpturb * dturb + pturb * ddturb - dpturb * dvisc;
            }
            let x = a[is];
            let dxa = ad[is];
            let temp2 = const2 * dst * x;
            let dtemp2 = const2 * (ddst * x + dst * dxa);
            dr[t] = temp2 * ecos[t];
            dr[nth + t] = temp2 * esin[t];
            dr[2 * nth + t] = select(dst < zero, one, zero);
            drd[t] = dtemp2 * ecos[t];
            drd[nth + t] = dtemp2 * esin[t];
            ws[is] = llws;
            if accumulate {
                vs[is] += scale * (dst * x);
                vd[is] += scale * dst;
                vsd[is] += scale * (ddst * x + dst * dxa);
                vdd[is] += scale * ddst;
            }
        }
        sync_cube();
        if t == 0 {
            let mut sx = sc[S_SX];
            let mut sy = sc[S_SY];
            let mut snx = sc[S_SNX];
            let mut sny = sc[S_SNY];
            let mut dsx = sd[S_SX];
            let mut dsy = sd[S_SY];
            for th in 0..nth {
                if dr[2 * nth + th] > zero {
                    snx += dr[th];
                    sny += dr[nth + th];
                } else {
                    sx += dr[th];
                    sy += dr[nth + th];
                    dsx += drd[th];
                    dsy += drd[nth + th];
                }
            }
            sc[S_SX] = sx;
            sc[S_SY] = sy;
            sc[S_SNX] = snx;
            sc[S_SNY] = sny;
            sd[S_SX] = dsx;
            sd[S_SY] = dsy;
        }
        sync_cube();
    }
    if t == 0 {
        let sx = sc[S_SX];
        let sy = sc[S_SY];
        let dsx = sd[S_SX];
        let dsy = sd[S_SY];
        let taupx = sc[S_TAUX] - tsh * sx;
        let taupy = sc[S_TAUY] - tsh * sy;
        let dtaupx = sd[S_TAUX] - tsh * dsx;
        let dtaupy = sd[S_TAUY] - tsh * dsy;
        let p = taupx * taupx + taupy * taupy;
        let dp = two * (taupx * dtaupx + taupy * dtaupy);
        let ust = p.powf(F::new(0.25f32));
        let dust = select(p > zero, F::new(0.25f32) * ust / p * dp, zero);
        let usdirp = taupy.atan2(taupx);
        let dusdirp = select(p > zero, (taupx * dtaupy - taupy * dtaupx) / p, zero);
        let cosu = sc[S_COSU];
        let sinu = sc[S_SINU];
        let dcosu = sd[S_COSU];
        let dsinu = sd[S_SINU];
        let snk = sig[nk];
        let c0 = par[P_DTH] * snk.powi(5) / (grav * grav * tpi::<F>()) * tpi::<F>() * snk / kb[K_CG * nk + nk - 1];
        let mut temp = zero;
        let mut dtemp = zero;
        for th in 0..nth {
            let cwv = ecos[th] * cosu + esin[th] * sinu;
            let dcwv = ecos[th] * dcosu + esin[th] * dsinu;
            let cw = max(cwv, zero);
            let dcw = select(cwv > zero, dcwv, zero);
            let x = a[(nk - 1) * nth + th];
            let dx = ad[(nk - 1) * nth + th];
            temp += x * cw * cw * cw;
            dtemp += dx * cw * cw * cw + x * F::new(3.0f32) * cw * cw * dcw;
        }
        let levtail0 = c0 * temp;
        let dlev = c0 * dtemp;
        // SINTABLE = 1: tail stress interpolated in TAUHFT2 (sheltering) or TAUHFT
        let ius = u32::cast_from(par[P_IUSTAR]) as usize;
        let ial = u32::cast_from(par[P_IALPHA]) as usize;
        let xi = ust / par[P_DELUST];
        let dxi = dust / par[P_DELUST];
        let xif = u32::cast_from(xi.trunc()) as usize;
        let ind = select(xif < 1, 1, select(xif < ius - 1, xif, ius - 1));
        let fi = xi - F::cast_from(ind as u32);
        let deli1 = max(min(one, fi), zero);
        let ddeli1 = select(fi < one && fi > zero, dxi, zero);
        let deli2 = one - deli1;
        let um = max(ust, F::new(0.00001f32));
        let dum = select(ust > F::new(0.00001f32), dust, zero);
        let q = grav * z0 / (um * um) - par[P_AALPHA];
        let dq = grav * (dz0 / (um * um) - two * z0 * dum / (um * um * um));
        let xj = max(zero, q / par[P_DELALP]);
        let dxj = select(q > zero, dq / par[P_DELALP], zero);
        let xjf = u32::cast_from(xj.trunc()) as usize;
        let j = select(xjf < 1, 1, select(xjf < ial - 1, xjf, ial - 1));
        let fj = xj - F::cast_from(j as u32);
        let delj1 = max(zero, min(one, fj));
        let ddelj1 = select(fj < one && fj > zero, dxj, zero);
        let delj2 = one - delj1;
        let n1 = ius + 1;
        let n2 = ial + 1;
        let mut tau1 = zero;
        let mut dtau1 = zero;
        // tables in BETAMAX (BETAMAX(U)): cubic Hermite weights hw of the
        // tables and their derivatives at the two grid values around BETAMAX,
        // and the derivatives hd of the weights with respect to BETAMAX
        // (0 where BETAMAX is clamped to the grid)
        let nbt = u32::cast_from(par[P_BT_N]);
        let stride = u32::cast_from(par[P_BT_STRIDE]) as usize;
        let nb = nbt as usize;
        let mut m = 0u32;
        let mut hw = Array::<F>::new(4usize);
        let mut hd = Array::<F>::new(4usize);
        let mut hb = Array::<u32>::new(4usize);
        if nbt > 1u32 {
            let db = par[P_BT_DB];
            let xr = (beta - par[P_BT_B0]) / db;
            let top = F::cast_from(nbt - 1u32);
            let xb = max(zero, min(top, xr));
            m = min(u32::cast_from(xb.trunc()), nbt - 2u32);
            let tt = xb - F::cast_from(m);
            let t2 = tt * tt;
            let t3 = t2 * tt;
            let three = F::new(3.0f32);
            let six = F::new(6.0f32);
            hw[0] = two * t3 - three * t2 + one;
            hw[1] = (t3 - two * t2 + tt) * db;
            hw[2] = three * t2 - two * t3;
            hw[3] = (t3 - t2) * db;
            let inside = xr >= zero && xr <= top;
            hd[0] = select(inside, (six * t2 - six * tt) / db, zero);
            hd[1] = select(inside, three * t2 - F::new(4.0f32) * tt + one, zero);
            hd[2] = select(inside, (six * tt - six * t2) / db, zero);
            hd[3] = select(inside, three * t2 - two * tt, zero);
            let mu = m as usize;
            hb[0] = (mu * stride) as u32;
            hb[1] = ((nb + mu) * stride) as u32;
            hb[2] = ((mu + 1) * stride) as u32;
            hb[3] = ((nb + mu + 1) * stride) as u32;
        }
        if nbt > 1u32 {
            if par[P_TTAUWSHELTER] > zero {
                let ilv = u32::cast_from(par[P_ILEVTAIL]) as usize;
                let xk = levtail0 / par[P_DELTAIL];
                let dxk = dlev / par[P_DELTAIL];
                let xkf = u32::cast_from(xk.trunc()) as usize;
                let i = select(xkf < ilv - 1, xkf, ilv - 1);
                let fk = xk - F::cast_from(i as u32);
                let delk1 = min(one, fk);
                let ddelk1 = select(fk < one, dxk, zero);
                let delk2 = one - delk1;
                let o0 = i * n1 * n2;
                let o1 = (i + 1) * n1 * n2;
                for c in 0..4u32 {
                    let bc = hb[c as usize] as usize;
                    let la = tauhf2::<F>(tauhf, bc + o0, ind, j, n1, deli1, deli2, delj1, delj2);
                    let lb = tauhf2::<F>(tauhf, bc + o1, ind, j, n1, deli1, deli2, delj1, delj2);
                    let dla = tauhf2_d::<F>(tauhf, bc + o0, ind, j, n1, deli1, deli2, delj1, ddeli1, ddelj1);
                    let dlb = tauhf2_d::<F>(tauhf, bc + o1, ind, j, n1, deli1, deli2, delj1, ddeli1, ddelj1);
                    let tc = la * delk2 + lb * delk1;
                    tau1 += hw[c as usize] * tc;
                    dtau1 += hw[c as usize] * (dla * delk2 + dlb * delk1 + (lb - la) * ddelk1) + hd[c as usize] * dbeta * tc;
                }
            } else {
                for c in 0..4u32 {
                    let bc = hb[c as usize] as usize;
                    let tc = tauhf2::<F>(tauhf, bc, ind, j, n1, deli1, deli2, delj1, delj2);
                    tau1 += hw[c as usize] * tc;
                    dtau1 += hw[c as usize] * tauhf2_d::<F>(tauhf, bc, ind, j, n1, deli1, deli2, delj1, ddeli1, ddelj1) + hd[c as usize] * dbeta * tc;
                }
            }
        } else if par[P_TTAUWSHELTER] > zero {
            let ilv = u32::cast_from(par[P_ILEVTAIL]) as usize;
            let xk = levtail0 / par[P_DELTAIL];
            let dxk = dlev / par[P_DELTAIL];
            let xkf = u32::cast_from(xk.trunc()) as usize;
            let i = select(xkf < ilv - 1, xkf, ilv - 1);
            let fk = xk - F::cast_from(i as u32);
            let delk1 = min(one, fk);
            let ddelk1 = select(fk < one, dxk, zero);
            let delk2 = one - delk1;
            let o0 = i * n1 * n2;
            let o1 = (i + 1) * n1 * n2;
            let a00 = tauhf[ind + j * n1 + o0];
            let a10 = tauhf[ind + 1 + j * n1 + o0];
            let a01 = tauhf[ind + (j + 1) * n1 + o0];
            let a11 = tauhf[ind + 1 + (j + 1) * n1 + o0];
            let b00 = tauhf[ind + j * n1 + o1];
            let b10 = tauhf[ind + 1 + j * n1 + o1];
            let b01 = tauhf[ind + (j + 1) * n1 + o1];
            let b11 = tauhf[ind + 1 + (j + 1) * n1 + o1];
            let la = (a00 * deli2 + a10 * deli1) * delj2 + (a01 * deli2 + a11 * deli1) * delj1;
            let lb = (b00 * deli2 + b10 * deli1) * delj2 + (b01 * deli2 + b11 * deli1) * delj1;
            tau1 = la * delk2 + lb * delk1;
            let dla_i = ((a10 - a00) * delj2 + (a11 - a01) * delj1) * ddeli1;
            let dla_j = ((a01 * deli2 + a11 * deli1) - (a00 * deli2 + a10 * deli1)) * ddelj1;
            let dlb_i = ((b10 - b00) * delj2 + (b11 - b01) * delj1) * ddeli1;
            let dlb_j = ((b01 * deli2 + b11 * deli1) - (b00 * deli2 + b10 * deli1)) * ddelj1;
            // derivative of the table itself (BETAMAX)
            let ea00 = dtauhf[ind + j * n1 + o0];
            let ea10 = dtauhf[ind + 1 + j * n1 + o0];
            let ea01 = dtauhf[ind + (j + 1) * n1 + o0];
            let ea11 = dtauhf[ind + 1 + (j + 1) * n1 + o0];
            let eb00 = dtauhf[ind + j * n1 + o1];
            let eb10 = dtauhf[ind + 1 + j * n1 + o1];
            let eb01 = dtauhf[ind + (j + 1) * n1 + o1];
            let eb11 = dtauhf[ind + 1 + (j + 1) * n1 + o1];
            let ela = (ea00 * deli2 + ea10 * deli1) * delj2 + (ea01 * deli2 + ea11 * deli1) * delj1;
            let elb = (eb00 * deli2 + eb10 * deli1) * delj2 + (eb01 * deli2 + eb11 * deli1) * delj1;
            dtau1 = (dla_i + dla_j + ela) * delk2 + (dlb_i + dlb_j + elb) * delk1 + (lb - la) * ddelk1;
        } else {
            let a00 = tauhf[ind + j * n1];
            let a10 = tauhf[ind + 1 + j * n1];
            let a01 = tauhf[ind + (j + 1) * n1];
            let a11 = tauhf[ind + 1 + (j + 1) * n1];
            tau1 = (a00 * deli2 + a10 * deli1) * delj2 + (a01 * deli2 + a11 * deli1) * delj1;
            let ea00 = dtauhf[ind + j * n1];
            let ea10 = dtauhf[ind + 1 + j * n1];
            let ea01 = dtauhf[ind + (j + 1) * n1];
            let ea11 = dtauhf[ind + 1 + (j + 1) * n1];
            dtau1 = ((a10 - a00) * delj2 + (a11 - a01) * delj1) * ddeli1
                + ((a01 * deli2 + a11 * deli1) - (a00 * deli2 + a10 * deli1)) * ddelj1
                + (ea00 * deli2 + ea10 * deli1) * delj2
                + (ea01 * deli2 + ea11 * deli1) * delj1;
        }
        let tauhf_v = levtail0 * ust * ust * tau1;
        let dtauhf_v = dlev * ust * ust * tau1 + levtail0 * two * ust * dust * tau1 + levtail0 * ust * ust * dtau1;
        let cp = usdirp.cos();
        let sp = usdirp.sin();
        let mut tauwx = sx + tauhf_v * cp;
        let mut tauwy = sy + tauhf_v * sp;
        let mut dtauwx = dsx + dtauhf_v * cp - tauhf_v * sp * dusdirp;
        let mut dtauwy = dsy + dtauhf_v * sp + tauhf_v * cp * dusdirp;
        let tauw = (tauwx * tauwx + tauwy * tauwy).sqrt();
        let dtauw = select(tauw > zero, (tauwx * dtauwx + tauwy * dtauwy) / tauw, zero);
        let eps2 = F::new(0.000001f32);
        let um2 = max(ustar, eps2);
        let dum2 = select(ustar > eps2, dustar, zero);
        let ust2 = um2 * um2;
        let dust2 = two * um2 * dum2;
        let limv = ust2 - F::new(0.00001f32);
        let lim = max(limv, eps2 * eps2);
        let dlim = select(limv > eps2 * eps2, dust2, zero);
        let tauwb = min(tauw, lim);
        let dtauwb = select(tauw < lim, dtauw, dlim);
        if tauwb < tauw {
            let r = tauwb / tauw;
            let drt = (dtauwb * tauw - tauwb * dtauw) / (tauw * tauw);
            dtauwx = dtauwx * r + tauwx * drt;
            dtauwy = dtauwy * r + tauwy * drt;
            tauwx *= r;
            tauwy *= r;
        }
        sc[S_TAUWX] = tauwx;
        sc[S_TAUWY] = tauwy;
        sd[S_TAUWX] = dtauwx;
        sd[S_TAUWY] = dtauwy;
    }
    sync_cube();
}

/// `W3SLN1` and its tangent.
#[cube]
#[allow(clippy::too_many_arguments)]
fn sln1_tl<F: Float>(
    par: &[F],
    sig: &[F],
    ecos: &[F],
    esin: &[F],
    kb: &Shared<[F]>,
    vs: &mut Shared<[F]>,
    vsd: &mut Shared<[F]>,
    fhigh: F,
    dfhigh: F,
    ustar: F,
    dustar: F,
    usdir: F,
    dusdir: F,
    scale: F,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
    #[comptime] nu: usize,
) {
    let t = UNIT_POS as usize;
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let nspec = nk * nth;
    let cosu = usdir.cos();
    let sinu = usdir.sin();
    let u2 = ustar * ustar;
    let fac = par[P_SLNC1] * (u2 * u2);
    let dfac = par[P_SLNC1] * F::new(4.0f32) * u2 * ustar * dustar;
    let ff1 = par[P_FSPM] * par[P_GRAV] / (F::new(28.0f32) * ustar);
    let dff1 = -ff1 * dustar / ustar;
    let ff2 = par[P_FSHF] * min(sig[nk], fhigh);
    let dff2 = select(fhigh < sig[nk], par[P_FSHF] * dfhigh, zero);
    let mx = max(ff1, ff2);
    let dmx = select(ff1 > ff2, dff1, dff2);
    let ffilt = min(mx, F::new(2.0f32) * sig[nk]);
    let dffilt = select(mx < F::new(2.0f32) * sig[nk], dmx, zero);
    for r in 0..(nspec + nu - 1) / nu {
        let is = r * nu + t;
        if is < nspec {
            let ik = is / nth;
            let th = is - ik * nth;
            let rfr = sig[ik + 1] / ffilt;
            let drfr = -rfr * dffilt / ffilt;
            let mut wnf = zero;
            let mut dwnf = zero;
            if rfr >= F::new(0.5f32) {
                let r2 = rfr * rfr;
                let e = (-(one / (r2 * r2))).exp();
                let k = kb[K_WN * nk + ik];
                wnf = fac / k * e;
                dwnf = dfac / k * e + wnf * F::new(4.0f32) / (r2 * r2 * rfr) * drfr;
            }
            let cwv = ecos[th] * cosu + esin[th] * sinu;
            let dcwv = -ecos[th] * sinu * dusdir + esin[th] * cosu * dusdir;
            let d = max(cwv, zero);
            let dd = select(cwv > zero, dcwv, zero);
            let d2 = d * d;
            vs[is] += scale * (wnf * (d2 * d2));
            vsd[is] += scale * (dwnf * (d2 * d2) + wnf * F::new(4.0f32) * d2 * d * dd);
        }
    }
}

/// `W3SNL1` (DIA) and its tangent; `dsnlc1` is the derivative of `NLPROP`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn snl1_tl<F: Float>(
    par: &[F],
    sig: &[F],
    dia_ip: &[u32],
    dia_ic: &[u32],
    af11: &[F],
    a: &Shared<[F]>,
    ad: &Shared<[F]>,
    kb: &Shared<[F]>,
    ue: &mut Shared<[F]>,
    ued: &mut Shared<[F]>,
    b1: &mut Shared<[F]>,
    b1d: &mut Shared<[F]>,
    b2: &mut Shared<[F]>,
    b2d: &mut Shared<[F]>,
    dn: &mut Shared<[F]>,
    dnd: &mut Shared<[F]>,
    vs: &mut Shared<[F]>,
    vsd: &mut Shared<[F]>,
    vd: &mut Shared<[F]>,
    vdd: &mut Shared<[F]>,
    kdmean: F,
    dkdmean: F,
    dsnlc1: F,
    scale: F,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
    #[comptime] nspecx: usize,
    #[comptime] nspecy: usize,
    #[comptime] nu: usize,
) {
    let t = UNIT_POS as usize;
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let two = F::new(2.0f32);
    let nspec = nk * nth;
    let xv = par[P_KDCON] * kdmean;
    let x = max(xv, par[P_KDMN]);
    let dx = select(xv > par[P_KDMN], par[P_KDCON] * dkdmean, zero);
    let x2v = par[P_SNLS3] * x;
    let x2 = max(F::new(-1.0e15f32), x2v);
    let dx2 = select(x2v > F::new(-1.0e15f32), par[P_SNLS3] * dx, zero);
    let ex = x2.exp();
    let s1 = par[P_SNLS1];
    let s2 = par[P_SNLS2];
    let g1 = one + s1 / x * (one - s2 * x) * ex;
    let dg1 = s1 * ((-dx / (x * x)) * (one - s2 * x) * ex + (one / x) * (-s2 * dx) * ex + (one / x) * (one - s2 * x) * ex * dx2);
    let cons = par[P_SNLC1] * g1;
    let dcons = dsnlc1 * g1 + par[P_SNLC1] * dg1;
    let dal1 = par[P_DAL1];
    let dal2 = par[P_DAL2];
    let dal3 = par[P_DAL3];
    for r in 0..(nspecy + nth + nu - 1) / nu {
        let u = r * nu + t;
        if u < nspecy + nth {
            let mut v = zero;
            let mut dv = zero;
            if u >= nth && u < nspec + nth {
                let is = u - nth;
                let ik = is / nth;
                let conx = tpiinv::<F>() / sig[ik + 1] * kb[K_CG * nk + ik];
                v = a[is] / conx;
                dv = ad[is] / conx;
            }
            ue[u] = v;
            ued[u] = dv;
        }
    }
    for r in 0..(nth + nu - 1) / nu {
        let u = r * nu + t;
        if u < nth {
            b1[u] = zero;
            b2[u] = zero;
            b1d[u] = zero;
            b2d[u] = zero;
        }
    }
    sync_cube();
    if t < nth {
        for ifr in nk..nspecy / nth {
            ue[nth + ifr * nth + t] = ue[ifr * nth + t] * par[P_FACHFE];
            ued[nth + ifr * nth + t] = ued[ifr * nth + t] * par[P_FACHFE];
        }
    }
    sync_cube();
    for pass in 0..3u32 {
        for r in 0..(nspecx + nu - 1) / nu {
            let isp = r * nu + t;
            if isp < nspecx {
                let e00 = ue[isp + nth];
                let de00 = ued[isp + nth];
                let i0 = dia_ip[isp] as usize;
                let i1 = dia_ip[nspecx + isp] as usize;
                let i2 = dia_ip[2 * nspecx + isp] as usize;
                let i3 = dia_ip[3 * nspecx + isp] as usize;
                let i4 = dia_ip[4 * nspecx + isp] as usize;
                let i5 = dia_ip[5 * nspecx + isp] as usize;
                let i6 = dia_ip[6 * nspecx + isp] as usize;
                let i7 = dia_ip[7 * nspecx + isp] as usize;
                let i8 = dia_ip[8 * nspecx + isp] as usize;
                let i9 = dia_ip[9 * nspecx + isp] as usize;
                let i10 = dia_ip[10 * nspecx + isp] as usize;
                let i11 = dia_ip[11 * nspecx + isp] as usize;
                let i12 = dia_ip[12 * nspecx + isp] as usize;
                let i13 = dia_ip[13 * nspecx + isp] as usize;
                let i14 = dia_ip[14 * nspecx + isp] as usize;
                let i15 = dia_ip[15 * nspecx + isp] as usize;
                let w0 = par[P_AWG];
                let w1 = par[P_AWG + 1];
                let w2 = par[P_AWG + 2];
                let w3 = par[P_AWG + 3];
                let w4 = par[P_AWG + 4];
                let w5 = par[P_AWG + 5];
                let w6 = par[P_AWG + 6];
                let w7 = par[P_AWG + 7];
                let ep1 = w0 * ue[i0] + w1 * ue[i1] + w2 * ue[i2] + w3 * ue[i3];
                let em1 = w4 * ue[i4] + w5 * ue[i5] + w6 * ue[i6] + w7 * ue[i7];
                let ep2 = w0 * ue[i8] + w1 * ue[i9] + w2 * ue[i10] + w3 * ue[i11];
                let em2 = w4 * ue[i12] + w5 * ue[i13] + w6 * ue[i14] + w7 * ue[i15];
                let dep1 = w0 * ued[i0] + w1 * ued[i1] + w2 * ued[i2] + w3 * ued[i3];
                let dem1 = w4 * ued[i4] + w5 * ued[i5] + w6 * ued[i6] + w7 * ued[i7];
                let dep2 = w0 * ued[i8] + w1 * ued[i9] + w2 * ued[i10] + w3 * ued[i11];
                let dem2 = w4 * ued[i12] + w5 * ued[i13] + w6 * ued[i14] + w7 * ued[i15];
                let caf = cons * af11[isp];
                let dcaf = dcons * af11[isp];
                let factor = caf * e00;
                let dfactor = dcaf * e00 + caf * de00;
                let sa1a = e00 * (ep1 * dal1 + em1 * dal2);
                let dsa1a = de00 * (ep1 * dal1 + em1 * dal2) + e00 * (dep1 * dal1 + dem1 * dal2);
                let sa1b = sa1a - ep1 * em1 * dal3;
                let dsa1b = dsa1a - (dep1 * em1 + ep1 * dem1) * dal3;
                let sa2a = e00 * (ep2 * dal1 + em2 * dal2);
                let dsa2a = de00 * (ep2 * dal1 + em2 * dal2) + e00 * (dep2 * dal1 + dem2 * dal2);
                let sa2b = sa2a - ep2 * em2 * dal3;
                let dsa2b = dsa2a - (dep2 * em2 + ep2 * dem2) * dal3;
                if pass == 0u32 {
                    b1[isp + nth] = factor * sa1b;
                    b2[isp + nth] = factor * sa2b;
                    b1d[isp + nth] = dfactor * sa1b + factor * dsa1b;
                    b2d[isp + nth] = dfactor * sa2b + factor * dsa2b;
                } else if pass == 1u32 {
                    let p1 = dal1 * e00 - dal3 * em1;
                    let p2 = dal1 * e00 - dal3 * em2;
                    b1[isp + nth] = factor * p1;
                    b2[isp + nth] = factor * p2;
                    b1d[isp + nth] = dfactor * p1 + factor * (dal1 * de00 - dal3 * dem1);
                    b2d[isp + nth] = dfactor * p2 + factor * (dal1 * de00 - dal3 * dem2);
                    if isp < nspec {
                        dn[isp] = -two * (caf * (sa1a + sa1b) + caf * (sa2a + sa2b));
                        dnd[isp] = -two * (dcaf * (sa1a + sa1b) + caf * (dsa1a + dsa1b) + dcaf * (sa2a + sa2b) + caf * (dsa2a + dsa2b));
                    }
                } else {
                    let m1 = dal2 * e00 - dal3 * ep1;
                    let m2 = dal2 * e00 - dal3 * ep2;
                    b1[isp + nth] = factor * m1;
                    b2[isp + nth] = factor * m2;
                    b1d[isp + nth] = dfactor * m1 + factor * (dal2 * de00 - dal3 * dep1);
                    b2d[isp + nth] = dfactor * m2 + factor * (dal2 * de00 - dal3 * dep2);
                }
            }
        }
        sync_cube();
        for r in 0..(nspec + nu - 1) / nu {
            let is = r * nu + t;
            if is < nspec {
                if pass == 0u32 {
                    let ik = is / nth;
                    let con = tpiinv::<F>() / sig[ik + 1] * kb[K_CG * nk + ik];
                    let mut ss = -two * (b1[is + nth] + b2[is + nth]);
                    let mut dss = -two * (b1d[is + nth] + b2d[is + nth]);
                    for j in 0..8u32 {
                        let jj = j as usize;
                        let c1 = dia_ic[2 * jj * nspec + is] as usize;
                        let c2 = dia_ic[(2 * jj + 1) * nspec + is] as usize;
                        ss += par[P_AWG + jj] * (b1[c1] + b2[c2]);
                        dss += par[P_AWG + jj] * (b1d[c1] + b2d[c2]);
                    }
                    vs[is] += scale * (con * ss);
                    vsd[is] += scale * (con * dss);
                } else if pass == 1u32 {
                    let mut d = dn[is];
                    let mut dd = dnd[is];
                    for j in 0..4u32 {
                        let jj = j as usize;
                        let c1 = dia_ic[2 * jj * nspec + is] as usize;
                        let c2 = dia_ic[(2 * jj + 1) * nspec + is] as usize;
                        d += par[P_SWG + jj] * (b1[c1] + b2[c2]);
                        dd += par[P_SWG + jj] * (b1d[c1] + b2d[c2]);
                    }
                    dn[is] = d;
                    dnd[is] = dd;
                } else {
                    let mut d = dn[is];
                    let mut dd = dnd[is];
                    for j in 4..8u32 {
                        let jj = j as usize;
                        let c1 = dia_ic[2 * jj * nspec + is] as usize;
                        let c2 = dia_ic[(2 * jj + 1) * nspec + is] as usize;
                        d += par[P_SWG + jj] * (b1[c1] + b2[c2]);
                        dd += par[P_SWG + jj] * (b1d[c1] + b2d[c2]);
                    }
                    vd[is] += scale * d;
                    vdd[is] += scale * dd;
                }
            }
        }
        sync_cube();
    }
}

/// `W3SDS4` (saturation, cumulative, turbulent and viscous terms) and its
/// tangent; `dsdsc2` is the derivative of `SDSC2`. `dn`, `dnd` hold the
/// breaking-crest lengths and their tangent.
#[cube]
#[allow(clippy::too_many_arguments)]
fn sds4_tl<F: Float>(
    par: &[F],
    sig: &[F],
    ecos: &[F],
    esin: &[F],
    satw: &[F],
    sati: &[u32],
    cumw: &[F],
    a: &Shared<[F]>,
    ad: &Shared<[F]>,
    kb: &mut Shared<[F]>,
    kd: &mut Shared<[F]>,
    b1: &mut Shared<[F]>,
    b1d: &mut Shared<[F]>,
    dn: &mut Shared<[F]>,
    dnd: &mut Shared<[F]>,
    vs: &mut Shared<[F]>,
    vsd: &mut Shared<[F]>,
    vd: &mut Shared<[F]>,
    vdd: &mut Shared<[F]>,
    depth: F,
    ustar: F,
    dustar: F,
    usdir: F,
    dusdir: F,
    dsdsc2: F,
    scale: F,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
    #[comptime] nsat: usize,
    #[comptime] nu: usize,
) {
    let t = UNIT_POS as usize;
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let two = F::new(2.0f32);
    let nspec = nk * nth;
    for r in 0..(nspec + nu - 1) / nu {
        let is = r * nu + t;
        if is < nspec {
            let ik = is / nth;
            let th = is - ik * nth;
            let k = kb[K_WN * nk + ik];
            let facsat = sig[ik + 1] * (k * k * k) * par[P_DTH];
            let mut dot = zero;
            let mut ddot = zero;
            for j in 0..nsat {
                let o = ik * nth + sati[j + th * nsat] as usize;
                dot += satw[j + th * nsat] * a[o];
                ddot += satw[j + th * nsat] * ad[o];
            }
            b1[is] = dot * facsat;
            b1d[is] = ddot * facsat;
        }
    }
    sync_cube();
    if t < nk {
        let mut m = b1[t * nth];
        let mut dm = b1d[t * nth];
        for th in 1..nth {
            let v = b1[t * nth + th];
            if v > m {
                m = v;
                dm = b1d[t * nth + th];
            }
        }
        kb[K_BTH0 * nk + t] = m;
        kd[K_BTH0 * nk + t] = dm;
    }
    let br = par[P_SSDSBR];
    let c3 = par[P_SDSC3];
    let c5 = par[P_SDSC5];
    let c21 = par[P_SDSC21];
    let extra = c3 != zero || c5 != zero || c21 != zero;
    if c3 != zero {
        let epsr = br.sqrt();
        let pi = F::new(3.1415927f32);
        let cb = F::new(28.16f32) / (two * (pi * pi));
        for r in 0..(nspec + nu - 1) / nu {
            let is = r * nu + t;
            if is < nspec {
                let sq = b1[is].sqrt();
                let qv = sq - epsr;
                let q = max(qv, zero);
                let dq = select(qv > zero, b1d[is] / (two * sq), zero);
                dn[is] = q * q * F::new(28.16f32) / (two * (pi * pi));
                dnd[is] = two * q * dq * cb;
            }
        }
    }
    sync_cube();
    let c2 = par[P_SDSC2];
    let c6 = par[P_SDSC6];
    let dik = u32::cast_from(par[P_DIKCUMUL]) as usize;
    let facturb = c5 * ustar * ustar / par[P_GRAV] * par[P_DRAT];
    let dfacturb = c5 * two * ustar * dustar / par[P_GRAV] * par[P_DRAT];
    let cu = usdir.cos();
    let su = usdir.sin();
    for r in 0..(nspec + nu - 1) / nu {
        let is = r * nu + t;
        if is < nspec {
            let ik = is / nth;
            let mut miche = one;
            if par[P_SSDSBM] != one {
                let x = min(kb[K_WN * nk + ik] * depth, F::new(10.0f32)).tanh();
                let p = x * (par[P_SSDSBM + 1] + x * (par[P_SSDSBM + 2] + x * (par[P_SSDSBM + 3] + x * par[P_SSDSBM + 4])));
                miche = p * p;
            }
            let coef1 = br * miche;
            let s = sig[ik + 1];
            let qv = kb[K_BTH0 * nk + ik] / coef1 - one;
            let q = max(qv, zero);
            let dq = select(qv > zero, kd[K_BTH0 * nk + ik] / coef1, zero);
            let sdiagiso = c2 * s * c6 * (q * q);
            let dsdiagiso = dsdsc2 * s * c6 * (q * q) + c2 * s * c6 * two * q * dq;
            let coef2 = c2 * s * (one - c6) / (coef1 * coef1);
            let dcoef2 = dsdsc2 * s * (one - c6) / (coef1 * coef1);
            let rv = b1[is] - coef1;
            let rr = max(rv, zero);
            let drr = select(rv > zero, b1d[is], zero);
            let rp = rr.powf(par[P_SSDSP]);
            let drp = select(rr > zero, par[P_SSDSP] * rp / rr * drr, zero);
            let d = sdiagiso + coef2 * rp;
            let dd = dsdiagiso + dcoef2 * rp + coef2 * drp;
            let x = a[is];
            let dxa = ad[is];
            let mut sv = d * x;
            let mut dsv = dd * x + d * dxa;
            let mut dv = d;
            let mut ddv = dd;
            if extra {
                let th = is - ik * nth;
                let k = kb[K_WN * nk + ik];
                let mut renewal = zero;
                let mut drenewal = zero;
                if c3 < zero && ik + 1 > dik {
                    for ik2 in 0..(ik + 1 - dik) {
                        if kb[K_BTH0 * nk + ik2] > br {
                            let mut dot = zero;
                            let mut ddot = zero;
                            for th2 in 0..nth {
                                let dth = select(th2 >= th, th2 - th, th - th2);
                                let w = cumw[(ik2 * nk + ik) * nth + dth];
                                dot += w * dn[ik2 * nth + th2];
                                ddot += w * dnd[ik2 * nth + th2];
                            }
                            renewal += dot;
                            drenewal += ddot;
                        }
                    }
                }
                let facturb2 = F::new(-2.0f32) * s * k * facturb;
                let dfacturb2 = F::new(-2.0f32) * s * k * dfacturb;
                let dvisc = F::new(-4.0f32) * c21 * k * k;
                let cwv = ecos[th] * cu + esin[th] * su;
                let dcwv = -ecos[th] * su * dusdir + esin[th] * cu * dusdir;
                let cw = max(zero, cwv);
                let dcw = select(cwv > zero, dcwv, zero);
                let dturb = facturb2 * cw;
                let ddturb = dfacturb2 * cw + facturb2 * dcw;
                let diag2 = c3 * renewal + dturb + dvisc;
                let ddiag2 = c3 * drenewal + ddturb;
                dv += diag2;
                ddv += ddiag2;
                sv += x * diag2;
                dsv += dxa * diag2 + x * ddiag2;
            }
            vs[is] += scale * sv;
            vd[is] += scale * dv;
            vsd[is] += scale * dsv;
            vdd[is] += scale * ddv;
        }
    }
    sync_cube();
}

/// `W3SDB1` and its tangent, on unit 0: writes `D` and its tangent to
/// `sc[S_SCALE + 1]`, `sd[S_SCALE + 1]`; `dsdbc2` is the derivative of
/// `BJGAM`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn sdb1_tl<F: Float>(
    par: &[F],
    sc: &mut Shared<[F]>,
    sd: &mut Shared<[F]>,
    emean: F,
    demean: F,
    fmean: F,
    dfmean: F,
    wnmean: F,
    dwnmean: F,
    depth: F,
    dsdbc2: F,
) {
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let two = F::new(2.0f32);
    let thr = F::new(1.1754944e-38f32);
    let mut d = zero;
    let mut dd = zero;
    if emean >= thr {
        let mut hm = par[P_SDBC2] * depth;
        let mut dhm = dsdbc2 * depth;
        if par[P_FDONLY] == zero {
            let th = (wnmean * max(depth, zero)).tanh();
            let dth = (one - th * th) * max(depth, zero) * dwnmean;
            hm = par[P_SDBC2] / wnmean * th;
            dhm = dsdbc2 / wnmean * th - par[P_SDBC2] * dwnmean / (wnmean * wnmean) * th + par[P_SDBC2] / wnmean * dth;
        }
        let hrms = (F::new(8.0f32) * emean).sqrt();
        let dhrms = F::new(4.0f32) * demean / hrms;
        let mut bb = zero;
        let mut dbb = zero;
        let mut b = zero;
        let mut db = zero;
        if hm > thr {
            bb = hrms * hrms / (hm * hm);
            dbb = two * hrms * dhrms / (hm * hm) - two * hrms * hrms * dhm / (hm * hm * hm);
            b = bb.sqrt();
            db = select(b > zero, dbb / (two * b), zero);
        }
        let mut q0 = zero;
        let mut dq0 = zero;
        if b > F::new(0.5f32) && b <= one {
            let y = two * b - one;
            q0 = y * y;
            dq0 = two * y * two * db;
        }
        let mut qb = zero;
        let mut dqb = zero;
        if b > F::new(0.2f32) && b < one {
            let arg = ((q0 - one) / bb).exp();
            let darg = arg * (dq0 * bb - (q0 - one) * dbb) / (bb * bb);
            let num = q0 - arg;
            let den = bb - arg;
            qb = q0 - bb * num / den;
            dqb = dq0 - (dbb * num / den + bb * ((dq0 - darg) * den - num * (dbb - darg)) / (den * den));
            for _ in 0..3u32 {
                let qn = ((qb - one) / bb).exp();
                dqb = qn * (dqb * bb - (qb - one) * dbb) / (bb * bb);
                qb = qn;
            }
        } else if b >= one {
            qb = one - thr;
            dqb = zero;
        }
        if bb > thr && (bb - qb).abs() > thr {
            if bb < one {
                d = -(two * par[P_SDBC1] * qb * fmean / bb);
                dd = -(two * par[P_SDBC1] * (dqb * fmean / bb + qb * dfmean / bb - qb * fmean * dbb / (bb * bb)));
            } else {
                d = -(two * par[P_SDBC1] * fmean * bb);
                dd = -(two * par[P_SDBC1] * (dfmean * bb + fmean * dbb));
            }
        }
    }
    sc[S_SCALE + 1] = d;
    sd[S_SCALE + 1] = dd;
}

/// `W3SBT4` and its tangent (no ice scaling). Ends with a cube
/// synchronisation.
#[cube]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn sbt4_tl<F: Float>(
    par: &[F],
    sig: &[F],
    dden: &[F],
    fwtab: &[F],
    erftab: &[F],
    a: &Shared<[F]>,
    ad: &Shared<[F]>,
    kb: &mut Shared<[F]>,
    kd: &mut Shared<[F]>,
    sc: &mut Shared<[F]>,
    sd: &mut Shared<[F]>,
    vs: &mut Shared<[F]>,
    vsd: &mut Shared<[F]>,
    vd: &mut Shared<[F]>,
    vdd: &mut Shared<[F]>,
    depth: F,
    d50_: F,
    psic: F,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
    #[comptime] nu: usize,
) {
    let t = UNIT_POS as usize;
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let two = F::new(2.0f32);
    let nspec = nk * nth;
    if t < nk {
        let mut eb = zero;
        let mut deb = zero;
        for th in 0..nth {
            eb += a[t * nth + th];
            deb += ad[t * nth + th];
        }
        kb[K_EB * nk + t] = eb;
        kd[K_EB * nk + t] = deb;
    }
    sync_cube();
    if t == 0 {
        let grav = par[P_GRAV];
        let sg1 = par[P_SED_SG] - one;
        let nfw = u32::cast_from(par[P_NFW]) as usize;
        let d50 = min(max(d50_, F::new(0.00001f32)), one);
        let background = max(par[P_SBTCX + 5], par[P_SBTCX + 6] * d50);
        let (mut sh0, mut sh1, mut sh2) = (zero, zero, zero);
        let (mut dsh0, mut dsh1, mut dsh2) = (zero, zero, zero);
        let mut uorb = zero;
        let mut duorb = zero;
        let mut uorb2 = zero;
        let mut duorb2 = zero;
        let mut aorb = zero;
        let mut daorb = zero;
        for isub in 0..3u32 {
            let mut xsub = zero;
            if isub == 0u32 {
                xsub = F::new(-0.001f32);
            } else if isub == 1u32 {
                xsub = F::new(0.001f32);
            }
            let dsub = depth * (one + xsub);
            let (mut u, mut ao, mut du, mut dao) = (zero, zero, zero, zero);
            for ik in 0..nk {
                let k = kb[K_WN * nk + ik];
                let mut cb = zero;
                if k * dsub < F::new(6.0f32) {
                    let s = sig[ik + 1];
                    let sh = (k * dsub).sinh();
                    cb = F::new(0.5f32) * s * s / (grav * (sh * sh));
                    let factor = (dden[ik] / kb[K_CG * nk + ik]) * two * cb * grav;
                    u += kb[K_EB * nk + ik] * factor;
                    ao += kb[K_EB * nk + ik] * factor / (s * s);
                    du += kd[K_EB * nk + ik] * factor;
                    dao += kd[K_EB * nk + ik] * factor / (s * s);
                }
                kb[K_CBETA * nk + ik] = cb;
            }
            uorb2 = two * u;
            duorb2 = two * du;
            let e7 = F::new(1.0e-7f32);
            uorb = max(e7, uorb2).sqrt();
            duorb = select(uorb2 > e7, duorb2 / (two * uorb), zero);
            let a2 = two * ao;
            aorb = max(e7, a2).sqrt();
            daorb = select(a2 > e7, two * dao / (two * aorb), zero);
            let rv = aorb / d50;
            let r = max(rv, F::new(0.3f32));
            let dr = select(rv > F::new(0.3f32), daorb / d50, zero);
            let xiv = (log10::<F>(r) - par[P_ABMIN]) / par[P_DELAB];
            let xi = max(xiv, one);
            let dxi = select(xiv > one, dr / (r * F::new(2.3025851f32)) / par[P_DELAB], zero);
            let fw = table::<F>(fwtab, nfw, xi);
            let dfw = table_d::<F>(fwtab, nfw, xi, dxi);
            let cpsi = two * grav * sg1 * d50;
            let shields = fw * uorb2 / cpsi / psic;
            let dshields = (dfw * uorb2 + fw * duorb2) / cpsi / psic;
            if isub == 0u32 {
                sh0 = shields;
                dsh0 = dshields;
            } else if isub == 1u32 {
                sh1 = shields;
                dsh1 = dshields;
            } else {
                sh2 = shields;
                dsh2 = dshields;
            }
        }
        let sb0 = par[P_SBTCX];
        let sb1 = par[P_SBTCX + 1];
        let sb2 = par[P_SBTCX + 2];
        let sb3 = par[P_SBTCX + 3];
        let xs = F::new(0.001f32) - F::new(-0.001f32);
        let dpsi = (sh1 - sh0) / xs * par[P_SBTCX + 4];
        let ddpsi = (dsh1 - dsh0) / xs * par[P_SBTCX + 4];
        let rq = uorb2 / (sg1 * grav * aorb);
        let drq = (duorb2 * aorb - uorb2 * daorb) / (sg1 * grav * aorb * aorb);
        let ripple = F::new(0.0655f32) * rq.powf(F::new(1.4f32));
        let dripple = select(rq > zero, F::new(1.4f32) * ripple / rq * drq, zero);
        let mbv = aorb * sb3;
        let mb = max(background, mbv);
        let dmb = select(mbv > background, daorb * sb3, zero);
        let mut ksubn = mb;
        let mut dksubn = dmb;
        if dpsi.abs() < F::new(0.0001f32) * sh2 || dpsi.abs() < F::new(1.0e-8f32) {
            if sh2 > sb2 {
                let p = sh2.powf(sb1);
                let dp = select(sh2 > zero, sb1 * p / sh2 * dsh2, zero);
                ksubn = aorb * sb0 * p + aorb * ripple;
                dksubn = daorb * sb0 * p + aorb * sb0 * dp + daorb * ripple + aorb * dripple;
            }
        } else {
            let psix = (sb2 - sh2) / dpsi;
            let dpsix = (-dsh2 * dpsi - (sb2 - sh2) * ddpsi) / (dpsi * dpsi);
            let pxv = (psix + par[P_XERFMAX]) / par[P_DELXERF];
            let psixt = max(pxv, zero);
            let dpsixt = select(pxv > zero, dpsix / par[P_DELXERF], zero);
            let nerf = u32::cast_from(par[P_NERF]) as usize;
            let p2v = table::<F>(erftab, nerf, psixt);
            let dp2v = table_d::<F>(erftab, nerf, psixt, dpsixt);
            let proba2 = min(max(p2v, zero), one);
            let dproba2 = select(p2v > zero && p2v < one, dp2v, zero);
            let proba1 = one - proba2;
            let g = (-(F::new(0.5f32) * psix * psix)).exp() / tpi::<F>().sqrt();
            let dg = -psix * dpsix * g;
            let den = proba2 + F::new(0.0001f32);
            let term = g * dpsi / den;
            let dterm = (dg * dpsi + g * ddpsi) / den - g * dpsi * dproba2 / (den * den);
            let pnv = sh2 + term;
            let psin2 = max(pnv, sb2);
            let dpsin2 = select(pnv > sb2, dsh2 + dterm, zero);
            let pp = psin2.powf(sb1);
            let dpp = select(psin2 > zero, sb1 * pp / psin2 * dpsin2, zero);
            let inner = sb0 * pp + ripple;
            let dinner = sb0 * dpp + dripple;
            ksubn = proba1 * mb + proba2 * aorb * inner;
            dksubn = -dproba2 * mb + proba1 * dmb + dproba2 * aorb * inner + proba2 * (daorb * inner + aorb * dinner);
        }
        let rv = aorb / ksubn;
        let drv = (daorb * ksubn - aorb * dksubn) / (ksubn * ksubn);
        let r = max(rv, F::new(0.3f32));
        let dr = select(rv > F::new(0.3f32), drv, zero);
        let xiv = (log10::<F>(r) - par[P_ABMIN]) / par[P_DELAB];
        let xi = max(xiv, one);
        let dxi = select(xiv > one, dr / (r * F::new(2.3025851f32)) / par[P_DELAB], zero);
        let fw = table::<F>(fwtab, nfw, xi);
        let dfw = table_d::<F>(fwtab, nfw, xi, dxi);
        sc[S_FBT] = fw * uorb;
        sd[S_FBT] = dfw * uorb + fw * duorb;
    }
    sync_cube();
    let fbt = sc[S_FBT];
    let dfbt = sd[S_FBT];
    for r in 0..(nspec + nu - 1) / nu {
        let is = r * nu + t;
        if is < nspec {
            let ik = is / nth;
            let cb = kb[K_CBETA * nk + ik];
            let dsum = -fbt * cb;
            let ddsum = -dfbt * cb;
            vs[is] += dsum * a[is];
            vsd[is] += ddsum * a[is] + dsum * ad[is];
            vd[is] += dsum;
            vdd[is] += ddsum;
        }
    }
    sync_cube();
}

/// `W3SRCE` (`srce_direct`) with its tangent, for the full ST4, at every
/// node with `MAPSTA = 1`: one cube per node, `nu` units per cube. The
/// values are those of [`crate::srce::srce()`] (not the start call); `vad` is
/// the tangent of the spectrum (updated in place), `du10`, `du10d` that of
/// the wind, `dpar` that of the scalar parameters (`BETAMAX`, `SDSC2`,
/// `NLPROP`, `BJGAM` are used) and `dtauhf` that of the tail-stress table.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn srce_tl<F: Float>(
    va: &mut [F],
    vad: &mut [F],
    wn: &[F],
    cg: &[F],
    dw: &[F],
    u10: &[F],
    u10d: &[F],
    du10: &[F],
    du10d: &[F],
    ice: &[F],
    d50: &[F],
    psic: &[F],
    nflag: &[u32],
    ust: &mut [F],
    dtdyn: &mut [F],
    par: &[F],
    dpar: &[F],
    sig: &[F],
    dden: &[F],
    ecos: &[F],
    esin: &[F],
    sig2k: &[F],
    dden2k: &[F],
    fwtab: &[F],
    erftab: &[F],
    satw: &[F],
    sati: &[u32],
    dia_ip: &[u32],
    dia_ic: &[u32],
    af11: &[F],
    taut: &[F],
    tauhf: &[F],
    dtauhf: &[F],
    cumw: &[F],
    dtg_: f32,
    np: usize,
    #[comptime] nk: usize,
    #[comptime] nth: usize,
    #[comptime] nspecx: usize,
    #[comptime] nspecy: usize,
    #[comptime] nsat: usize,
    #[comptime] nu: usize,
) {
    let i = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let nspec = nk * nth;
    if i < np {
        if (nflag[i] & 4u32) != 0u32 {
            let zero = F::new(0.0f32);
            let one = F::new(1.0f32);
            let dtg = F::cast_from(dtg_);
            let mut a = Shared::<[F]>::new_slice(nk * nth);
            let mut ad = Shared::<[F]>::new_slice(nk * nth);
            let mut vs = Shared::<[F]>::new_slice(nk * nth);
            let mut vsd = Shared::<[F]>::new_slice(nk * nth);
            let mut vd = Shared::<[F]>::new_slice(nk * nth);
            let mut vdd = Shared::<[F]>::new_slice(nk * nth);
            let mut ws = Shared::<[F]>::new_slice(nk * nth);
            let mut dn = Shared::<[F]>::new_slice(nk * nth);
            let mut dnd = Shared::<[F]>::new_slice(nk * nth);
            let mut ue = Shared::<[F]>::new_slice(nspecy + nth);
            let mut ued = Shared::<[F]>::new_slice(nspecy + nth);
            let mut b1 = Shared::<[F]>::new_slice(nspecx + nth);
            let mut b1d = Shared::<[F]>::new_slice(nspecx + nth);
            let mut b2 = Shared::<[F]>::new_slice(nspecx + nth);
            let mut b2d = Shared::<[F]>::new_slice(nspecx + nth);
            let mut kb = Shared::<[F]>::new_slice(NKB * nk);
            let mut kd = Shared::<[F]>::new_slice(NKB * nk);
            let mut sc = Shared::<[F]>::new_slice(NSC);
            let mut sd = Shared::<[F]>::new_slice(NSC);
            let mut dr = Shared::<[F]>::new_slice(3 * nth);
            let mut drd = Shared::<[F]>::new_slice(2 * nth);
            let mut red = Shared::<[F]>::new_slice(nu);
            for r in 0..(nspec + nu - 1) / nu {
                let is = r * nu + t;
                if is < nspec {
                    let ik = is / nth;
                    let th = is - ik * nth;
                    a[is] = va[(ik * np + i) * nth + th];
                    ad[is] = vad[(ik * np + i) * nth + th];
                    ws[is] = one;
                }
            }
            if t < nk {
                let w = wn[(t + 1) * np + i];
                kb[K_WN * nk + t] = w;
                kb[K_CG * nk + t] = cg[(t + 1) * np + i];
                kb[K_DAM * nk + t] = par[P_FACP] / (sig[t + 1] * (w * w * w));
            }
            sync_cube();
            let d_inp = dw[i];
            let depth = max(par[P_DMIN], d_inp);
            let u = u10[i];
            let udir = u10d[i];
            let du = du10[i];
            let dudir = du10d[i];
            let c = ice[i];
            let sd50 = d50[i];
            let spsic = psic[i];
            // BETAMAX of the node and its tangent: with BETAMAX(U),
            // d ln beta = d ln P_BBETA + sum_k (dc_k phi_k(U) + c_k phi_k'(U) dU),
            // so that the direction of coefficient k is the field beta phi_k(U)
            let beta = par[P_BBETA] * betau_ln::<F>(par, u).exp();
            let nbk = u32::cast_from(par[P_BU_NK]) as usize;
            let mut dbeta = dpar[P_BBETA];
            if nbk > 0 {
                let mut dln = dpar[P_BBETA] / par[P_BBETA];
                for k in 0..nbk {
                    dln += dpar[P_BU_COEF + k] * betau_phi::<F>(par, u, k, nbk) + par[P_BU_COEF + k] * betau_dphi::<F>(par, u, k, nbk) * du;
                }
                dbeta = beta * dln;
            }
            let dsdsc2 = dpar[P_SDSC2];
            let dsnlc1 = dpar[P_SNLC1];
            let dsdbc2 = dpar[P_SDBC2];
            let mut isc_ln = one;
            let mut isc_in = one;
            let mut isc_nl = one;
            let mut isc_ds = one;
            if par[P_ICE_FORCING] > zero {
                isc_ln = max(zero, min(one, one - c * par[P_ICESCALES]));
                isc_in = max(zero, min(one, one - c * par[P_ICESCALES + 1]));
                isc_nl = max(zero, min(one, one - c * par[P_ICESCALES + 2]));
                isc_ds = max(zero, min(one, one - c * par[P_ICESCALES + 3]));
            }
            // before the loop: W3SPR4, W3SIN4 (TAUWBUG), W3SPR4
            spr4_tl::<F>(par, taut, sig, dden, &a, &ad, &ws, &mut kb, &mut kd, &mut sc, &mut sd, u, du, zero, zero, zero, zero, nk, nth);
            let ust0 = sc[S_USTAR];
            let z00 = sc[S_Z0];
            let dust0 = sd[S_USTAR];
            let dz00 = sd[S_Z0];
            sin4_tl::<F>(
                par, tauhf, dtauhf, sig, dden, ecos, esin, sig2k, dden2k, fwtab, &a, &ad, &mut vs, &mut vsd, &mut vd, &mut vdd, &mut ws, &mut kb, &mut kd,
                &mut sc, &mut sd, &mut dr, &mut drd, ust0, dust0, z00, dz00, udir, dudir, beta, dbeta, one, false, nk, nth,
            );
            let tauwx = sc[S_TAUWX];
            let tauwy = sc[S_TAUWY];
            let dtauwx = sd[S_TAUWX];
            let dtauwy = sd[S_TAUWY];
            spr4_tl::<F>(par, taut, sig, dden, &a, &ad, &ws, &mut kb, &mut kd, &mut sc, &mut sd, u, du, tauwx, dtauwx, tauwy, dtauwy, nk, nth);
            if t == 0 {
                let f1 = par[P_FFXFM] * max(sc[S_FMEAN1], sc[S_FMEANWS]);
                let df1 = par[P_FFXFM] * select(sc[S_FMEAN1] > sc[S_FMEANWS], sd[S_FMEAN1], sd[S_FMEANWS]);
                let f2 = par[P_FFXPM] / sc[S_USTAR];
                let df2 = -f2 * sd[S_USTAR] / sc[S_USTAR];
                sc[S_FHIGH] = max(f1, f2);
                sd[S_FHIGH] = select(f1 > f2, df1, df2);
                sc[S_DTTOT] = zero;
                sc[S_DTDYN] = zero;
                sc[S_NSTEPS] = zero;
                sc[S_DONE] = zero;
            }
            sync_cube();
            loop {
                let ustar = sc[S_USTAR];
                let z0 = sc[S_Z0];
                let fhigh = sc[S_FHIGH];
                let wnmean = sc[S_WNMEAN];
                let emean = sc[S_EMEAN];
                let fmean = sc[S_FMEAN];
                let dustar = sd[S_USTAR];
                let dz0 = sd[S_Z0];
                let dfhigh = sd[S_FHIGH];
                let dwnmean = sd[S_WNMEAN];
                let demean = sd[S_EMEAN];
                let dfmean = sd[S_FMEAN];
                for r in 0..(nspec + nu - 1) / nu {
                    let is = r * nu + t;
                    if is < nspec {
                        vs[is] = zero;
                        vd[is] = zero;
                        vsd[is] = zero;
                        vdd[is] = zero;
                    }
                }
                sync_cube();
                sln1_tl::<F>(par, sig, ecos, esin, &kb, &mut vs, &mut vsd, fhigh, dfhigh, ustar, dustar, udir, dudir, isc_ln, nk, nth, nu);
                sync_cube();
                sin4_tl::<F>(
                    par, tauhf, dtauhf, sig, dden, ecos, esin, sig2k, dden2k, fwtab, &a, &ad, &mut vs, &mut vsd, &mut vd, &mut vdd, &mut ws, &mut kb,
                    &mut kd, &mut sc, &mut sd, &mut dr, &mut drd, ustar, dustar, z0, dz0, udir, dudir, beta, dbeta, isc_in, true, nk, nth,
                );
                let stauwx = sc[S_TAUWX];
                let stauwy = sc[S_TAUWY];
                let dstauwx = sd[S_TAUWX];
                let dstauwy = sd[S_TAUWY];
                snl1_tl::<F>(
                    par, sig, dia_ip, dia_ic, af11, &a, &ad, &kb, &mut ue, &mut ued, &mut b1, &mut b1d, &mut b2, &mut b2d, &mut dn, &mut dnd, &mut vs,
                    &mut vsd, &mut vd, &mut vdd, wnmean * depth, dwnmean * depth, dsnlc1, isc_nl, nk, nth, nspecx, nspecy, nu,
                );
                sds4_tl::<F>(
                    par, sig, ecos, esin, satw, sati, cumw, &a, &ad, &mut kb, &mut kd, &mut b1, &mut b1d, &mut dn, &mut dnd, &mut vs, &mut vsd, &mut vd,
                    &mut vdd, depth, ustar, dustar, udir, dudir, dsdsc2, isc_ds, nk, nth, nsat, nu,
                );
                if t == 0 {
                    sdb1_tl::<F>(par, &mut sc, &mut sd, emean, demean, fmean, dfmean, wnmean, dwnmean, depth, dsdbc2);
                }
                sbt4_tl::<F>(
                    par, sig, dden, fwtab, erftab, &a, &ad, &mut kb, &mut kd, &mut sc, &mut sd, &mut vs, &mut vsd, &mut vd, &mut vdd, depth, sd50,
                    spsic, nk, nth, nu,
                );
                let dbd = sc[S_SCALE + 1];
                let ddbd = sd[S_SCALE + 1];
                // dynamic step: NKH from the FHIGH of the previous step
                let nkh = min(F::cast_from(nk as u32), (par[P_FACTI2] + par[P_FACTI1] * max(F::new(1.0e-7f32), fhigh).ln()).trunc());
                let nkh1 = min(F::cast_from(nk as u32), nkh + one);
                let nspech = u32::cast_from(max(nkh1, zero)) as usize * nth;
                let dttot = sc[S_DTTOT];
                let afilt = max(kb[K_DAM * nk + nk - 1], par[P_XFLT] * sc[S_AMAX]);
                let mut dtl = min(dtg - dttot, par[P_DTMAX]);
                for r in 0..(nspec + nu - 1) / nu {
                    let is = r * nu + t;
                    if is < nspech {
                        let ik = is / nth;
                        let th = is - ik * nth;
                        let ainit = va[(ik * np + i) * nth + th];
                        let damax = min(kb[K_DAM * nk + ik], max(par[P_XREL] * ainit, afilt));
                        let afac = one / max(F::new(1.0e-10f32), (vs[is] / damax).abs());
                        dtl = min(dtl, afac / max(F::new(1.0e-10f32), one + afac * min(zero, vd[is])));
                    }
                }
                red[t] = dtl;
                sync_cube();
                if t == 0 {
                    let mut dt = red[0];
                    for j in 1..nu {
                        dt = min(dt, red[j]);
                    }
                    dt = max(F::new(0.5f32), dt);
                    sc[S_DTDYN] = sc[S_DTDYN] + dt;
                    sc[S_NSTEPS] = sc[S_NSTEPS] + one;
                    let rem = dtg - dttot;
                    let idt = one + (F::new(0.99f32) * rem / dt).trunc();
                    dt = rem / idt;
                    let shave = dt < par[P_DTMIN] && dt < rem;
                    dt = max(dt, min(par[P_DTMIN], rem));
                    sc[S_DT] = dt;
                    sc[S_SHAVE] = select(shave, one, zero);
                    sc[S_DTTOT] = dttot + dt;
                }
                sync_cube();
                let dt = sc[S_DT];
                let shave = sc[S_SHAVE] > zero;
                let hdt = dt;
                for r in 0..(nspec + nu - 1) / nu {
                    let is = r * nu + t;
                    if is < nspech {
                        let ik = is / nth;
                        let aold = a[is];
                        let daold = ad[is];
                        let denv = one - hdt * vd[is];
                        let den = max(one, denv);
                        let dden_ = select(denv > one, -hdt * vdd[is], zero);
                        let mut inc = vs[is] * dt / den;
                        let mut dinc = dt * (vsd[is] * den - vs[is] * dden_) / (den * den);
                        if shave {
                            let dam = kb[K_DAM * nk + ik];
                            let clamped = inc.abs() > dam;
                            let m = min(dam, inc.abs());
                            inc = select(inc >= zero, m, -m);
                            dinc = select(clamped, zero, dinc);
                        }
                        let s1 = aold + inc;
                        let a1 = max(zero, s1);
                        let da1 = select(s1 > zero, daold + dinc, zero);
                        let den2v = one - hdt * dbd;
                        let den2 = max(one, den2v);
                        let dden2 = select(den2v > one, -hdt * ddbd, zero);
                        let incdb = (dbd * aold) * dt / den2;
                        let dincdb = dt * ((ddbd * aold + dbd * daold) * den2 - dbd * aold * dden2) / (den2 * den2);
                        let s2 = a1 + incdb;
                        a[is] = max(zero, s2);
                        ad[is] = select(s2 > zero, da1 + dincdb, zero);
                    }
                }
                sync_cube();
                spr4_tl::<F>(par, taut, sig, dden, &a, &ad, &ws, &mut kb, &mut kd, &mut sc, &mut sd, u, du, stauwx, dstauwx, stauwy, dstauwy, nk, nth);
                if t == 0 {
                    let g = par[P_GRAV];
                    let arg = F::new(0.3f32) * u * sc[S_FMEANWS] * tpi::<F>() / g;
                    let darg = F::new(0.3f32) * (du * sc[S_FMEANWS] + u * sd[S_FMEANWS]) * tpi::<F>() / g;
                    let th = arg.tanh();
                    let fage = par[P_FFXFA] * th;
                    let dfage = par[P_FFXFA] * (one - th * th) * darg;
                    let fh1 = (par[P_FFXFM] + fage) * sc[S_FMEAN1];
                    let dfh1 = dfage * sc[S_FMEAN1] + (par[P_FFXFM] + fage) * sd[S_FMEAN1];
                    let fh2 = par[P_FFXPM] / sc[S_USTAR];
                    let dfh2 = -fh2 * sd[S_USTAR] / sc[S_USTAR];
                    let mx = max(fh1, fh2);
                    let dmx = select(fh1 > fh2, dfh1, dfh2);
                    let fh = min(sig[nk], mx);
                    sc[S_FHIGH] = fh;
                    sd[S_FHIGH] = select(mx < sig[nk], dmx, zero);
                    let nk2 = max(F::new(2.0f32), min(nkh1, (par[P_FACTI2] + par[P_FACTI1] * max(F::new(1.0e-7f32), fh).ln()).trunc()));
                    sc[S_NKH] = nk2;
                    let last = sc[S_DTTOT] >= F::new(0.9999f32) * dtg;
                    sc[S_DONE] = select(last, one, zero);
                    // FHMAX limit at the end of the global step
                    sc[S_SCALE] = zero;
                    if last {
                        let wnm = sc[S_WNMEAN];
                        let dwnm = sd[S_WNMEAN];
                        let dpt = max(zero, d_inp);
                        let thw = (wnm * dpt).tanh();
                        let wm = max(F::new(0.0001f32), wnm);
                        let dwm = select(wnm > F::new(0.0001f32), dwnm, zero);
                        let hm = par[P_FHMAX] * thw / wm;
                        let dhm = par[P_FHMAX] * ((one - thw * thw) * dpt * dwnm * wm - thw * dwm) / (wm * wm);
                        let em = hm * hm / F::new(16.0f32);
                        let dem = hm * dhm / F::new(8.0f32);
                        let e = sc[S_EMEAN];
                        if e > em && e > F::new(1.0e-30f32) {
                            sc[S_SCALE] = one;
                            sc[S_CONS] = em;
                            sd[S_CONS] = dem;
                        }
                    }
                }
                sync_cube();
                if sc[S_SCALE] > zero {
                    let fac = sc[S_CONS];
                    let dfac = sd[S_CONS];
                    let e = sc[S_EMEAN];
                    let de = sd[S_EMEAN];
                    for r in 0..(nspec + nu - 1) / nu {
                        let is = r * nu + t;
                        if is < nspec {
                            let x = a[is];
                            let dx = ad[is];
                            a[is] = x / e * fac;
                            ad[is] = (dx * e - x * de) / (e * e) * fac + x / e * dfac;
                        }
                    }
                    sync_cube();
                }
                // high-frequency tail above NKH
                let nkt = u32::cast_from(sc[S_NKH]) as usize;
                if t < nth {
                    for ik in nkt..nk {
                        a[ik * nth + t] = a[(ik - 1) * nth + t] * par[P_FACHFA];
                        ad[ik * nth + t] = ad[(ik - 1) * nth + t] * par[P_FACHFA];
                    }
                }
                sync_cube();
                if sc[S_DONE] > zero {
                    break;
                }
            }
            if t == 0 {
                ust[i] = sc[S_USTAR];
                dtdyn[i] = sc[S_DTDYN] / max(one, sc[S_NSTEPS]);
            }
            for r in 0..(nspec + nu - 1) / nu {
                let is = r * nu + t;
                if is < nspec {
                    let ik = is / nth;
                    let th = is - ik * nth;
                    let x = a[is];
                    va[(ik * np + i) * nth + th] = max(zero, x);
                    vad[(ik * np + i) * nth + th] = select(x > zero, ad[is], zero);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

use crate::device::Real;
use crate::srce::{bf, SourceKernel, SrceArgs};
use cubecl::client::Client;
use cubecl::server::Handle;

/// The tangent inputs of [`srce_tl()`]: the tangent of the spectrum `vad`
/// (laid out as `va`, updated in place), of the wind speed and direction
/// `du10`, `du10d` (per node), of the scalar parameters `dpar` (`NPAR`
/// values at the `P_*` indices) and of the tail-stress table `dtauhf`
/// (laid out as the table of the kernel, `TAUHFT2` with sheltering, else
/// `TAUHFT`).
pub struct TangentArgs<'a> {
    /// See [`TangentArgs`].
    pub vad: &'a Handle,
    /// See [`TangentArgs`].
    pub du10: &'a Handle,
    /// See [`TangentArgs`].
    pub du10d: &'a Handle,
    /// See [`TangentArgs`].
    pub dpar: &'a Handle,
    /// See [`TangentArgs`].
    pub dtauhf: &'a Handle,
}

impl<F: Real> SourceKernel<F> {
    /// Launch [`srce_tl()`] on `np` nodes for one global step `dtg`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_tl(&self, client: &Client, s: &SrceArgs, d: &TangentArgs, dtg: f32, np: usize, nk: usize, nth: usize) {
        let nspec = nk * nth;
        unsafe {
            srce_tl::launch_unchecked::<F>(
                client,
                CubeCount::Static(np as u32, 1, 1),
                CubeDim::new_1d(self.nu as u32),
                bf(s.va, nspec * np),
                bf(d.vad, nspec * np),
                bf(s.wn, (nk + 2) * np),
                bf(s.cg, (nk + 2) * np),
                bf(s.dw, np),
                bf(s.u10, np),
                bf(s.u10d, np),
                bf(d.du10, np),
                bf(d.du10d, np),
                bf(s.ice, np),
                bf(s.d50, np),
                bf(s.psic, np),
                bf(s.nflag, np),
                bf(s.ust, np),
                bf(s.dtdyn, np),
                bf(&self.par, NPAR),
                bf(d.dpar, NPAR),
                bf(&self.sig, nk + 2),
                bf(&self.dden, nk),
                bf(&self.ecos, nth),
                bf(&self.esin, nth),
                bf(&self.sig2k, nk),
                bf(&self.dden2k, nk),
                bf(&self.fwtab, self.nfw),
                bf(&self.erftab, self.nerf),
                bf(&self.satw, self.nsat * nth),
                bf(&self.sati, self.nsat * nth),
                bf(&self.dia_ip, 16 * self.nspecx),
                bf(&self.dia_ic, 16 * nspec),
                bf(&self.af11, self.nspecx),
                bf(&self.taut, self.ntaut),
                bf(&self.tauhf, self.ntauhf),
                bf(d.dtauhf, self.ntauhf),
                bf(&self.cumw, self.ncumw),
                dtg,
                np,
                nk,
                nth,
                self.nspecx,
                self.nspecy,
                self.nsat,
                self.nu,
            );
        }
    }
}
