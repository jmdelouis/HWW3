//! Milestone M4 prototypes in CUDA C: the hand-written adjoints of the DIA
//! and of one geographic sub-step of [`crate::adjoint`], written in CUDA C,
//! compiled with NVRTC and launched through `cudarc` on their own copies of
//! the inputs, as a third candidate next to the CubeCL kernels and Burn's
//! autodiff. The kernels have `__launch_bounds__(NU, MINB)`, so the
//! compiler is told how many cubes per SM to aim for (CubeCL only declares
//! the cube size).

use anyhow::Result;
use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

/// CUDA C source of the adjoints (double precision).
pub const SRC: &str = r#"
#define NSPEC (NK * NTH)
#define NUE (NSPECY + NTH)
typedef unsigned int uint;

extern "C" __global__ void __launch_bounds__(NU, MINB)
dia_adjoint(const double *__restrict__ va, const double *__restrict__ cg, const double *__restrict__ kdm,
            const double *__restrict__ par, const double *__restrict__ sig, const uint *__restrict__ dia_ip,
            const double *__restrict__ af11, const double *__restrict__ sbar, const double *__restrict__ dbar,
            const uint *__restrict__ tb1s, const uint *__restrict__ tb1e, const uint *__restrict__ tb2s,
            const uint *__restrict__ tb2e, const uint *__restrict__ tps, const uint *__restrict__ tpe,
            double *__restrict__ abar, uint np)
{
    extern __shared__ double sh[];
    double *ue = sh, *ueb = ue + NUE, *gs = ueb + NUE, *gd = gs + NSPEC, *eb = gd + NSPEC, *conx = eb + NSPECX;
    const uint i = blockIdx.x, t = threadIdx.x;
    if (i >= np) return;
    const double zero = 0.0, one = 1.0, two = 2.0;
    double e00b[(NSPECX + NU - 1) / NU], ep1b[(NSPECX + NU - 1) / NU], em1b[(NSPECX + NU - 1) / NU];
    double ep2b[(NSPECX + NU - 1) / NU], em2b[(NSPECX + NU - 1) / NU], acc[(NUE + NU - 1) / NU];
    if (t < NK) conx[t] = (double)0.15915494f / sig[t + 1] * cg[i * NK + t];
    __syncthreads();
    for (uint r = 0; r < (NUE + NU - 1) / NU; r++) {
        uint u = r * NU + t;
        if (u < NUE) {
            double v = zero;
            if (u >= NTH && u < NSPEC + NTH) { uint is = u - NTH; v = va[i * NSPEC + is] / conx[is / NTH]; }
            ue[u] = v;
        }
    }
    for (uint r = 0; r < (NSPEC + NU - 1) / NU; r++) {
        uint is = r * NU + t;
        if (is < NSPEC) { gs[is] = conx[is / NTH] * sbar[i * NSPEC + is]; gd[is] = dbar[i * NSPEC + is]; }
    }
    __syncthreads();
    if (t < NTH)
        for (uint ifr = NK; ifr < NSPECY / NTH; ifr++) ue[NTH + ifr * NTH + t] = ue[ifr * NTH + t] * par[P_FACHFE];
    __syncthreads();
    const double xv = par[P_KDCON] * kdm[i];
    const double x = fmax(xv, par[P_KDMN]);
    const double x2 = fmax(-1.0e15, par[P_SNLS3] * x);
    const double cons = par[P_SNLC1] * (one + par[P_SNLS1] / x * (one - par[P_SNLS2] * x) * exp(x2));
    const double dal1 = par[P_DAL1], dal2 = par[P_DAL2], dal3 = par[P_DAL3];
    const double w0 = par[P_AWG], w1 = par[P_AWG + 1], w2 = par[P_AWG + 2], w3 = par[P_AWG + 3];
    const double w4 = par[P_AWG + 4], w5 = par[P_AWG + 5], w6 = par[P_AWG + 6], w7 = par[P_AWG + 7];
    for (uint r = 0; r < (NSPECX + NU - 1) / NU; r++) {
        uint isp = r * NU + t;
        e00b[r] = ep1b[r] = em1b[r] = ep2b[r] = em2b[r] = zero;
        if (isp >= NSPECX) continue;
        uint xb = isp + NTH;
        double b10 = zero, b11 = zero, b12 = zero, b20 = zero, b21 = zero, b22 = zero;
        if (isp < NSPEC) { b10 = -two * gs[isp]; b20 = -two * gs[isp]; }
        for (uint q = tb1s[xb]; q < tb1s[xb + 1]; q++) {
            uint e = tb1e[q], is = e / 8, j = e - 8 * is;
            b10 += par[P_AWG + j] * gs[is];
            if (j < 4) b11 += par[P_SWG + j] * gd[is]; else b12 += par[P_SWG + j] * gd[is];
        }
        for (uint q = tb2s[xb]; q < tb2s[xb + 1]; q++) {
            uint e = tb2e[q], is = e / 8, j = e - 8 * is;
            b20 += par[P_AWG + j] * gs[is];
            if (j < 4) b21 += par[P_SWG + j] * gd[is]; else b22 += par[P_SWG + j] * gd[is];
        }
        double e00 = ue[isp + NTH];
        double ep1 = w0 * ue[dia_ip[isp]] + w1 * ue[dia_ip[NSPECX + isp]] + w2 * ue[dia_ip[2 * NSPECX + isp]] + w3 * ue[dia_ip[3 * NSPECX + isp]];
        double em1 = w4 * ue[dia_ip[4 * NSPECX + isp]] + w5 * ue[dia_ip[5 * NSPECX + isp]] + w6 * ue[dia_ip[6 * NSPECX + isp]] + w7 * ue[dia_ip[7 * NSPECX + isp]];
        double ep2 = w0 * ue[dia_ip[8 * NSPECX + isp]] + w1 * ue[dia_ip[9 * NSPECX + isp]] + w2 * ue[dia_ip[10 * NSPECX + isp]] + w3 * ue[dia_ip[11 * NSPECX + isp]];
        double em2 = w4 * ue[dia_ip[12 * NSPECX + isp]] + w5 * ue[dia_ip[13 * NSPECX + isp]] + w6 * ue[dia_ip[14 * NSPECX + isp]] + w7 * ue[dia_ip[15 * NSPECX + isp]];
        double caf = cons * af11[isp], factor = caf * e00;
        double sa1a = e00 * (ep1 * dal1 + em1 * dal2), sa1b = sa1a - ep1 * em1 * dal3;
        double sa2a = e00 * (ep2 * dal1 + em2 * dal2), sa2b = sa2a - ep2 * em2 * dal3;
        double p1 = dal1 * e00 - dal3 * em1, p2 = dal1 * e00 - dal3 * em2;
        double m1 = dal2 * e00 - dal3 * ep1, m2 = dal2 * e00 - dal3 * ep2;
        double ge = b10 * (caf * sa1b + factor * (ep1 * dal1 + em1 * dal2)) + b20 * (caf * sa2b + factor * (ep2 * dal1 + em2 * dal2));
        double gp1 = b10 * factor * (e00 * dal1 - em1 * dal3), gm1 = b10 * factor * (e00 * dal2 - ep1 * dal3);
        double gp2 = b20 * factor * (e00 * dal1 - em2 * dal3), gm2 = b20 * factor * (e00 * dal2 - ep2 * dal3);
        ge += b11 * (caf * p1 + factor * dal1) + b21 * (caf * p2 + factor * dal1);
        gm1 += -b11 * factor * dal3;
        gm2 += -b21 * factor * dal3;
        if (isp < NSPEC) {
            double gdn = -two * caf * gd[isp];
            ge += gdn * two * (ep1 * dal1 + em1 * dal2 + ep2 * dal1 + em2 * dal2);
            gp1 += gdn * (two * e00 * dal1 - em1 * dal3);
            gm1 += gdn * (two * e00 * dal2 - ep1 * dal3);
            gp2 += gdn * (two * e00 * dal1 - em2 * dal3);
            gm2 += gdn * (two * e00 * dal2 - ep2 * dal3);
        }
        ge += b12 * (caf * m1 + factor * dal2) + b22 * (caf * m2 + factor * dal2);
        gp1 += -b12 * factor * dal3;
        gp2 += -b22 * factor * dal3;
        e00b[r] = ge; ep1b[r] = gp1; em1b[r] = gm1; ep2b[r] = gp2; em2b[r] = gm2;
    }
    for (uint r = 0; r < (NUE + NU - 1) / NU; r++) acc[r] = zero;
    for (uint qg = 0; qg < 5; qg++) {
        for (uint r = 0; r < (NSPECX + NU - 1) / NU; r++) {
            uint isp = r * NU + t;
            if (isp < NSPECX) eb[isp] = qg == 0 ? e00b[r] : qg == 1 ? ep1b[r] : qg == 2 ? em1b[r] : qg == 3 ? ep2b[r] : em2b[r];
        }
        __syncthreads();
        for (uint r = 0; r < (NUE + NU - 1) / NU; r++) {
            uint u = r * NU + t;
            if (u >= NUE) continue;
            if (qg == 0) {
                if (u >= NTH && u - NTH < NSPECX) acc[r] += eb[u - NTH];
            } else {
                uint g = qg - 1;
                for (uint q = tps[g * (NUE + 1) + u]; q < tps[g * (NUE + 1) + u + 1]; q++) {
                    uint e = tpe[q], isp = e / 16, k = e - 16 * isp;
                    acc[r] += par[P_AWG + (k % 8)] * eb[isp];
                }
            }
        }
        __syncthreads();
    }
    for (uint r = 0; r < (NUE + NU - 1) / NU; r++) { uint u = r * NU + t; if (u < NUE) ueb[u] = acc[r]; }
    __syncthreads();
    if (t < NTH)
        for (uint ifr = NSPECY / NTH - 1; ifr >= NK; ifr--) ueb[ifr * NTH + t] += par[P_FACHFE] * ueb[NTH + ifr * NTH + t];
    __syncthreads();
    for (uint r = 0; r < (NSPEC + NU - 1) / NU; r++) {
        uint is = r * NU + t;
        if (is < NSPEC) abar[i * NSPEC + is] = ueb[is + NTH] / conx[is / NTH];
    }
}

/* one geographic sub-step: the triangle values (forward), then the three
   passes of the adjoint, four directions per thread as the CubeCL kernels;
   coef holds the 12 coefficients per triangle of the frequency */
extern "C" __global__ void __launch_bounds__(GB, 1)
geo_tri(const double *__restrict__ u, double *__restrict__ ut, const double *__restrict__ coef,
        const double *__restrict__ ccoef, const double *__restrict__ ecos, const double *__restrict__ esin,
        const uint *__restrict__ iter, const uint *__restrict__ ine, uint ne, uint sub)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x, nw = NTH / 4;
    if (g >= ne * nw) return;
    uint t = g / nw, k0 = (g - t * nw) * 4;
    const double *c12 = coef + 12 * t, *c6 = ccoef + 6 * t;
    uint r1 = ine[3 * t] * NTH, r2 = ine[3 * t + 1] * NTH, r3 = ine[3 * t + 2] * NTH;
    for (uint d = 0; d < 4; d++) {
        uint k = k0 + d;
        if (sub >= iter[k]) continue;
        double c = ecos[k], s = esin[k];
        double k1 = c12[0] * c + c12[3] * s + c6[0], k2 = c12[1] * c + c12[4] * s + c6[1], k3 = c12[2] * c + c12[5] * s + c6[2];
        double kn = fmin(k1, 0.0) + fmin(k2, 0.0) + fmin(k3, 0.0);
        double nm = -1.0 / fmin(-(double)1.1754944e-38f, kn);
        double fa1 = c12[6] * c + c12[9] * s + c6[3] + fmax(k1, 0.0);
        double fa2 = c12[7] * c + c12[10] * s + c6[4] + fmax(k2, 0.0);
        double fa3 = c12[8] * c + c12[11] * s + c6[5] + fmax(k3, 0.0);
        ut[t * NTH + k] = nm * (fa1 * u[r1 + k] + fa2 * u[r2 + k] + fa3 * u[r3 + k]);
    }
}

extern "C" __global__ void __launch_bounds__(GB, 1)
geo_adj_node1(const double *__restrict__ u, const double *__restrict__ ut, const double *__restrict__ ybar,
              double *__restrict__ ub, double *__restrict__ stb, const double *__restrict__ coef,
              const double *__restrict__ ccoef, const double *__restrict__ ecos, const double *__restrict__ esin,
              const double *__restrict__ dtk, const uint *__restrict__ iter, const double *__restrict__ si_inv,
              const uint *__restrict__ mlo, const uint *__restrict__ mhi, const uint *__restrict__ start,
              const uint *__restrict__ slot, const uint *__restrict__ bidx, uint np, uint sub)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x, nw = NTH / 4;
    if (g >= np * nw) return;
    uint i = g / nw, k0 = (g - i * nw) * 4, base = i * NTH + k0;
    uint wlo = mlo[i], whi = mhi[i];
    double sa = si_inv[i] * (1.0 - (double)((whi >> 31) & 1u));
    bool bnd = bidx[i] != 0xffffffffu;
    for (uint d = 0; d < 4; d++) {
        uint k = k0 + d;
        double c = ecos[k], s = esin[k], ui = u[base + d], st = 0.0, ks = 0.0;
        for (uint q = start[i]; q < start[i + 1]; q++) {
            uint code = slot[q], t = code / 3, p = code - 3 * t;
            double kp = fmax(coef[12 * t + p] * c + coef[12 * t + 3 + p] * s + ccoef[6 * t + p], 0.0);
            st += kp * (ui - ut[t * NTH + k]);
            ks += kp;
        }
        double dt = dtk[k];
        double m = (double)(((k < 32 ? wlo : whi) >> (k % 32)) & 1u);
        bool active = sub < iter[k];
        double x = ui - dt * sa * st, y = ybar[base + d];
        double w = (active && !bnd && x > 0.0) ? y * m : 0.0;
        double direct = active ? (bnd ? 0.0 : w) : y;
        double sb = -dt * sa * w;
        stb[base + d] = sb;
        ub[base + d] = direct + sb * ks;
    }
}

extern "C" __global__ void __launch_bounds__(GB, 1)
geo_adj_tri(const double *__restrict__ stb, double *__restrict__ utb, const double *__restrict__ coef,
            const double *__restrict__ ccoef, const double *__restrict__ ecos, const double *__restrict__ esin,
            const uint *__restrict__ ine, uint ne)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x, nw = NTH / 4;
    if (g >= ne * nw) return;
    uint t = g / nw, k0 = (g - t * nw) * 4;
    const double *c12 = coef + 12 * t, *c6 = ccoef + 6 * t;
    uint r1 = ine[3 * t] * NTH, r2 = ine[3 * t + 1] * NTH, r3 = ine[3 * t + 2] * NTH;
    for (uint d = 0; d < 4; d++) {
        uint k = k0 + d;
        double c = ecos[k], s = esin[k];
        double k1 = fmax(c12[0] * c + c12[3] * s + c6[0], 0.0);
        double k2 = fmax(c12[1] * c + c12[4] * s + c6[1], 0.0);
        double k3 = fmax(c12[2] * c + c12[5] * s + c6[2], 0.0);
        utb[t * NTH + k] = -(k1 * stb[r1 + k] + k2 * stb[r2 + k] + k3 * stb[r3 + k]);
    }
}

extern "C" __global__ void __launch_bounds__(GB, 1)
geo_adj_node2(const double *__restrict__ ub, const double *__restrict__ utb, double *__restrict__ abar,
              const double *__restrict__ coef, const double *__restrict__ ccoef, const double *__restrict__ ecos,
              const double *__restrict__ esin, const uint *__restrict__ start, const uint *__restrict__ slot, uint np)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x, nw = NTH / 4;
    if (g >= np * nw) return;
    uint i = g / nw, k0 = (g - i * nw) * 4, base = i * NTH + k0;
    for (uint d = 0; d < 4; d++) {
        uint k = k0 + d;
        double c = ecos[k], s = esin[k], acc = ub[base + d];
        for (uint q = start[i]; q < start[i + 1]; q++) {
            uint code = slot[q], t = code / 3, p = code - 3 * t;
            const double *c12 = coef + 12 * t, *c6 = ccoef + 6 * t;
            double k1 = c12[0] * c + c12[3] * s + c6[0], k2 = c12[1] * c + c12[4] * s + c6[1], k3 = c12[2] * c + c12[5] * s + c6[2];
            double kn = fmin(k1, 0.0) + fmin(k2, 0.0) + fmin(k3, 0.0);
            double nm = -1.0 / fmin(-(double)1.1754944e-38f, kn);
            double kpp = fmax(c12[p] * c + c12[3 + p] * s + c6[p], 0.0);
            double fa = c12[6 + p] * c + c12[9 + p] * s + c6[3 + p] + kpp;
            acc += nm * fa * utb[t * NTH + k];
        }
        abar[base + d] = acc;
    }
}
"#;

fn compile(defs: Vec<String>) -> Result<(Arc<CudaContext>, Arc<cudarc::driver::CudaModule>)> {
    let ctx = CudaContext::new(0)?;
    let opts = CompileOptions { arch: Some("compute_80"), options: defs, ..Default::default() };
    let ptx = compile_ptx_with_opts(SRC, opts)?;
    let module = ctx.load_module(ptx)?;
    Ok((ctx, module))
}

fn par_defs() -> Vec<String> {
    use crate::sources::*;
    [
        ("P_FACHFE", P_FACHFE),
        ("P_KDCON", P_KDCON),
        ("P_KDMN", P_KDMN),
        ("P_SNLS1", P_SNLS1),
        ("P_SNLS2", P_SNLS2),
        ("P_SNLS3", P_SNLS3),
        ("P_SNLC1", P_SNLC1),
        ("P_DAL1", P_DAL1),
        ("P_DAL2", P_DAL2),
        ("P_DAL3", P_DAL3),
        ("P_AWG", P_AWG),
        ("P_SWG", P_SWG),
    ]
    .iter()
    .map(|(n, v)| format!("-D{n}={v}u"))
    .collect()
}

/// The CUDA C adjoint of the DIA on `n` nodes (the inputs of
/// [`crate::adjoint::dia_check`], the transposed tables of
/// [`crate::adjoint::DiaAdjTables`]), with `MINB` cubes per SM asked of the
/// compiler: `abar`, the mean time of `reps` launches and the registers per
/// thread.
#[allow(clippy::too_many_arguments)]
pub fn dia_adjoint_cuda(
    t: &crate::sources::SrceTables,
    tab: &crate::adjoint::DiaAdjTables,
    va: &[f64],
    cg: &[f64],
    kdm: &[f64],
    sbar: &[f64],
    dbar: &[f64],
    n: usize,
    nk: usize,
    nth: usize,
    minb: u32,
    reps: usize,
) -> Result<(Vec<f64>, f64, i32)> {
    let nu = 128u32;
    let (nspec, nspecx, nspecy) = (nk * nth, t.nspecx, t.nspecy);
    let mut defs = par_defs();
    defs.extend([
        format!("-DNK={nk}u"),
        format!("-DNTH={nth}u"),
        format!("-DNSPECX={nspecx}u"),
        format!("-DNSPECY={nspecy}u"),
        format!("-DNU={nu}u"),
        format!("-DMINB={minb}"),
        "-DGB=256".to_string(),
    ]);
    let (ctx, module) = compile(defs)?;
    let f = module.load_function("dia_adjoint")?;
    let smem = ((2 * (nspecy + nth) + 2 * nspec + nspecx + nk) * 8) as u32;
    f.set_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32)?;
    let regs = f.num_regs()?;
    let st = ctx.default_stream();
    let (d_va, d_cg, d_kdm, d_par, d_sig) = (st.clone_htod(va)?, st.clone_htod(cg)?, st.clone_htod(kdm)?, st.clone_htod(&t.par)?, st.clone_htod(&t.sig)?);
    let (d_ip, d_af, d_sb, d_db) = (st.clone_htod(&t.dia_ip)?, st.clone_htod(&t.af11)?, st.clone_htod(sbar)?, st.clone_htod(dbar)?);
    let (d_t1s, d_t1e, d_t2s, d_t2e, d_tps, d_tpe) =
        (st.clone_htod(&tab.tb1s)?, st.clone_htod(&tab.tb1e)?, st.clone_htod(&tab.tb2s)?, st.clone_htod(&tab.tb2e)?, st.clone_htod(&tab.tps)?, st.clone_htod(&tab.tpe)?);
    let mut d_ab = st.alloc_zeros::<f64>(n * nspec)?;
    let cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (nu, 1, 1), shared_mem_bytes: smem };
    let np = n as u32;
    let launch = |d_ab: &mut cudarc::driver::CudaSlice<f64>| -> Result<()> {
        let mut b = st.launch_builder(&f);
        b.arg(&d_va).arg(&d_cg).arg(&d_kdm).arg(&d_par).arg(&d_sig).arg(&d_ip).arg(&d_af).arg(&d_sb).arg(&d_db);
        b.arg(&d_t1s).arg(&d_t1e).arg(&d_t2s).arg(&d_t2e).arg(&d_tps).arg(&d_tpe).arg(d_ab).arg(&np);
        unsafe { b.launch(cfg) }?;
        Ok(())
    };
    launch(&mut d_ab)?;
    st.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        launch(&mut d_ab)?;
    }
    st.synchronize()?;
    let per = t0.elapsed().as_secs_f64() / reps as f64;
    Ok((st.clone_dtoh(&d_ab)?, per, regs))
}

/// The CUDA C adjoint of one geographic sub-step (`sub = 0`) on the host
/// copies of [`crate::burn_m4::GeoHost`]: the triangle values recomputed,
/// then the three passes; `abar`, the mean time of `reps` adjoints and the
/// registers per thread of the four kernels.
pub fn geo_adjoint_cuda(h: &crate::burn_m4::GeoHost, ybar: &[f64], reps: usize) -> Result<(Vec<f64>, f64, [i32; 4])> {
    let (np, ne, nth) = (h.np, h.ne, h.nth);
    let gb = 256u32;
    let defs = vec![
        format!("-DNTH={nth}u"),
        format!("-DGB={gb}"),
        "-DNK=1u".into(),
        "-DNSPECX=1u".into(),
        "-DNSPECY=1u".into(),
        "-DNU=128u".into(),
        "-DMINB=1".into(),
    ];
    let mut defs = defs;
    defs.extend(par_defs());
    let (ctx, module) = compile(defs)?;
    let f_tri = module.load_function("geo_tri")?;
    let f_n1 = module.load_function("geo_adj_node1")?;
    let f_tr = module.load_function("geo_adj_tri")?;
    let f_n2 = module.load_function("geo_adj_node2")?;
    let regs = [f_tri.num_regs()?, f_n1.num_regs()?, f_tr.num_regs()?, f_n2.num_regs()?];
    let st = ctx.default_stream();
    let up = |v: &[f64]| st.clone_htod(v);
    let upu = |v: &[u32]| st.clone_htod(v);
    let (d_u, d_y, d_coef, d_cc, d_ec, d_es, d_dtk) = (up(&h.u)?, up(ybar)?, up(&h.coef)?, up(&h.ccoef)?, up(&h.ecos)?, up(&h.esin)?, up(&h.dtk)?);
    let (d_iter, d_si, d_mlo, d_mhi, d_start, d_slot, d_ine, d_bidx) =
        (upu(&h.iter)?, up(&h.si_inv)?, upu(&h.mlo)?, upu(&h.mhi)?, upu(&h.start)?, upu(&h.slot)?, upu(&h.ine)?, upu(&h.bidx)?);
    let mut d_ut = st.alloc_zeros::<f64>(ne * nth)?;
    let mut d_ub = st.alloc_zeros::<f64>(np * nth)?;
    let mut d_stb = st.alloc_zeros::<f64>(np * nth)?;
    let mut d_utb = st.alloc_zeros::<f64>(ne * nth)?;
    let mut d_ab = st.alloc_zeros::<f64>(np * nth)?;
    let cfg = |n: usize| LaunchConfig { grid_dim: ((n as u32).div_ceil(gb), 1, 1), block_dim: (gb, 1, 1), shared_mem_bytes: 0 };
    let (np32, ne32, sub) = (np as u32, ne as u32, 0u32);
    let mut run = || -> Result<()> {
        let mut b = st.launch_builder(&f_tri);
        b.arg(&d_u).arg(&mut d_ut).arg(&d_coef).arg(&d_cc).arg(&d_ec).arg(&d_es).arg(&d_iter).arg(&d_ine).arg(&ne32).arg(&sub);
        unsafe { b.launch(cfg(ne * nth / 4)) }?;
        let mut b = st.launch_builder(&f_n1);
        b.arg(&d_u).arg(&d_ut).arg(&d_y).arg(&mut d_ub).arg(&mut d_stb).arg(&d_coef).arg(&d_cc).arg(&d_ec).arg(&d_es);
        b.arg(&d_dtk).arg(&d_iter).arg(&d_si).arg(&d_mlo).arg(&d_mhi).arg(&d_start).arg(&d_slot).arg(&d_bidx).arg(&np32).arg(&sub);
        unsafe { b.launch(cfg(np * nth / 4)) }?;
        let mut b = st.launch_builder(&f_tr);
        b.arg(&d_stb).arg(&mut d_utb).arg(&d_coef).arg(&d_cc).arg(&d_ec).arg(&d_es).arg(&d_ine).arg(&ne32);
        unsafe { b.launch(cfg(ne * nth / 4)) }?;
        let mut b = st.launch_builder(&f_n2);
        b.arg(&d_ub).arg(&d_utb).arg(&mut d_ab).arg(&d_coef).arg(&d_cc).arg(&d_ec).arg(&d_es).arg(&d_start).arg(&d_slot).arg(&np32);
        unsafe { b.launch(cfg(np * nth / 4)) }?;
        Ok(())
    };
    run()?;
    st.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        run()?;
    }
    st.synchronize()?;
    let per = t0.elapsed().as_secs_f64() / reps as f64;
    drop(run);
    Ok((st.clone_dtoh(&d_ab)?, per, regs))
}
