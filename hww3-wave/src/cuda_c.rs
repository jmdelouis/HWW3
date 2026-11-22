//! Fallback backend: the same two kernels in CUDA C, compiled at run time
//! with NVRTC and launched through `cudarc` (no CubeCL, no Burn).

use crate::backend::{Dims, Scalar, Sweeper};
use crate::geom::Geometry;
use anyhow::Result;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

/// Kernels of [`crate::kernels`], written in CUDA C (same arithmetic).
pub const SRC: &str = r#"
#ifndef REAL
#define REAL float
#endif

#define TRIANGLE_COEFFICIENTS                                                    \
    unsigned int code = slot[s], t = code / 3u, p = code - 3u * t;              \
    unsigned int n1 = ine[3u * t], n2 = ine[3u * t + 1u], n3 = ine[3u * t + 2u]; \
    REAL c1x = cx[off + n1] * co, c1y = cy[off + n1] * sn;                       \
    REAL c2x = cx[off + n2] * co, c2y = cy[off + n2] * sn;                       \
    REAL c3x = cx[off + n3] * co, c3y = cy[off + n3] * sn;                       \
    const REAL *e = ien + 6u * t;                                                \
    REAL e1x = e[0], e1y = e[1], e2x = e[2], e2y = e[3], e3x = e[4], e3y = e[5]; \
    REAL lx = (c1x + c2x + c3x) * sixth, ly = (c1y + c2y + c3y) * sixth;         \
    REAL k1 = lx * e1x + ly * e1y, k2 = lx * e2x + ly * e2y, k3 = lx * e3x + ly * e3y; \
    REAL kp1 = k1 > zero ? k1 : zero, kp2 = k2 > zero ? k2 : zero, kp3 = k3 > zero ? k3 : zero; \
    REAL kn = (k1 < zero ? k1 : zero) + (k2 < zero ? k2 : zero) + (k3 < zero ? k3 : zero); \
    REAL nm = -one / (-thr < kn ? -thr : kn);                                    \
    REAL fl11 = c2x * e1x + c2y * e1y, fl12 = c3x * e1x + c3y * e1y;             \
    REAL fl21 = c3x * e2x + c3y * e2y, fl22 = c1x * e2x + c1y * e2y;             \
    REAL fl31 = c1x * e3x + c1y * e3y, fl32 = c2x * e3x + c2y * e3y;             \
    REAL fa1 = (two * fl31 + fl32 + two * fl22 + fl21) * sixth + kp1;            \
    REAL fa2 = (two * fl11 + fl12 + two * fl32 + fl31) * sixth + kp2;            \
    REAL fa3 = (two * fl21 + fl22 + two * fl12 + fl11) * sixth + kp3;

#define ARGS                                                                     \
    const REAL *__restrict__ cx, const REAL *__restrict__ cy,                    \
    const REAL *__restrict__ ecos, const REAL *__restrict__ esin,                \
    const REAL *__restrict__ dtk, const unsigned int *__restrict__ iter,         \
    const REAL *__restrict__ si_inv, const unsigned int *__restrict__ mlo,       \
    const unsigned int *__restrict__ mhi, const unsigned int *__restrict__ start,\
    const unsigned int *__restrict__ slot, const unsigned int *__restrict__ ine, \
    const REAL *__restrict__ ien, unsigned int np, unsigned int ik, unsigned int sub

#define CONSTANTS                                                                \
    const REAL zero = (REAL)0, one = (REAL)1, two = (REAL)2;                     \
    const REAL sixth = one / (REAL)6, thr = (REAL)1.17549435e-38f;

extern "C" __global__ void nscheme_forward(const REAL *__restrict__ u, REAL *__restrict__ out, ARGS, int clip)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np * NTH) return;
    unsigned int i = g / NTH, k = g - i * NTH, kk = ik * NTH + k;
    REAL ui = u[g];
    if (sub >= iter[kk]) { out[g] = ui; return; }
    CONSTANTS
    REAL co = ecos[k], sn = esin[k];
    unsigned int off = ik * np;
    REAL st = zero;
    for (unsigned int s = start[i]; s < start[i + 1]; ++s) {
        TRIANGLE_COEFFICIENTS
        REAL ut = nm * (fa1 * u[n1 * NTH + k] + fa2 * u[n2 * NTH + k] + fa3 * u[n3 * NTH + k]);
        REAL kpi = p == 0u ? kp1 : (p == 1u ? kp2 : kp3);
        st += kpi * (ui - ut);
    }
    unsigned int w = k < 32u ? mlo[i] : mhi[i];
    REAL m = (REAL)((w >> (k % 32u)) & 1u);
    REAL a = one - (REAL)((mhi[i] >> 31) & 1u);
    REAL v = ui - dtk[kk] * si_inv[i] * st * a;
    if (clip && v < zero) v = zero;
    out[g] = v * m;
}

extern "C" __global__ void nscheme_transpose(const REAL *__restrict__ y, REAL *__restrict__ out, ARGS)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np * NTH) return;
    unsigned int j = g / NTH, k = g - j * NTH, kk = ik * NTH + k;
    REAL yj = y[g];
    if (sub >= iter[kk]) { out[g] = yj; return; }
    CONSTANTS
    REAL co = ecos[k], sn = esin[k], dt = dtk[kk];
    unsigned int off = ik * np, sh = k % 32u;
    bool lo = k < 32u;
#define V(n) ((REAL)(((lo ? mlo[n] : mhi[n]) >> sh) & 1u) * (one - (REAL)((mhi[n] >> 31) & 1u)) * si_inv[n] * y[(n) * NTH + k])
    REAL mj = (REAL)(((lo ? mlo[j] : mhi[j]) >> sh) & 1u);
    REAL vj = dt * V(j);
    REAL acc = zero;
    for (unsigned int s = start[j]; s < start[j + 1]; ++s) {
        TRIANGLE_COEFFICIENTS
        REAL kv = dt * (kp1 * V(n1) + kp2 * V(n2) + kp3 * V(n3));
        REAL kpj = p == 0u ? kp1 : (p == 1u ? kp2 : kp3);
        REAL faj = p == 0u ? fa1 : (p == 1u ? fa2 : fa3);
        acc += kpj * vj - nm * faj * kv;
    }
    out[g] = mj * yj - acc;
}

/* Two-phase variant (see kernels.rs): twelve coefficients per triangle and
   frequency, one pass over (triangle, direction), one node gather. */

extern "C" __global__ void nscheme_coef(const REAL *__restrict__ cx, const REAL *__restrict__ cy,
    const unsigned int *__restrict__ ine, const REAL *__restrict__ ien, REAL *__restrict__ coef,
    unsigned int np, unsigned int ne, unsigned int ik)
{
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ne) return;
    const REAL two = (REAL)2, sixth = (REAL)1 / (REAL)6;
    unsigned int off = ik * np;
    unsigned int n1 = ine[3u * t], n2 = ine[3u * t + 1u], n3 = ine[3u * t + 2u];
    REAL x1 = cx[off + n1], x2 = cx[off + n2], x3 = cx[off + n3];
    REAL y1 = cy[off + n1], y2 = cy[off + n2], y3 = cy[off + n3];
    const REAL *e = ien + 6u * t;
    REAL e1x = e[0], e1y = e[1], e2x = e[2], e2y = e[3], e3x = e[4], e3y = e[5];
    REAL sx = (x1 + x2 + x3) * sixth, sy = (y1 + y2 + y3) * sixth;
    REAL *c = coef + ((size_t)ik * ne + t) * 12u;
    c[0] = sx * e1x; c[1] = sx * e2x; c[2] = sx * e3x;
    c[3] = sy * e1y; c[4] = sy * e2y; c[5] = sy * e3y;
    c[6] = (two * x1 * e3x + x2 * e3x + two * x1 * e2x + x3 * e2x) * sixth;
    c[7] = (two * x2 * e1x + x3 * e1x + two * x2 * e3x + x1 * e3x) * sixth;
    c[8] = (two * x3 * e2x + x1 * e2x + two * x3 * e1x + x2 * e1x) * sixth;
    c[9] = (two * y1 * e3y + y2 * e3y + two * y1 * e2y + y3 * e2y) * sixth;
    c[10] = (two * y2 * e1y + y3 * e1y + two * y2 * e3y + y1 * e3y) * sixth;
    c[11] = (two * y3 * e2y + y1 * e2y + two * y3 * e1y + y2 * e1y) * sixth;
}

#define TRI_K                                                                    \
    const REAL *c = coef + ((size_t)ik * ne + t) * 12u;                          \
    REAL co = ecos[k], sn = esin[k];                                             \
    REAL k1 = c[0] * co + c[3] * sn, k2 = c[1] * co + c[4] * sn, k3 = c[2] * co + c[5] * sn; \
    REAL kp1 = k1 > zero ? k1 : zero, kp2 = k2 > zero ? k2 : zero, kp3 = k3 > zero ? k3 : zero; \
    REAL kn = (k1 < zero ? k1 : zero) + (k2 < zero ? k2 : zero) + (k3 < zero ? k3 : zero); \
    REAL nm = -one / (-thr < kn ? -thr : kn);                                    \
    unsigned int n1 = ine[3u * t], n2 = ine[3u * t + 1u], n3 = ine[3u * t + 2u];

extern "C" __global__ void nscheme_tri_forward(const REAL *__restrict__ u, REAL *__restrict__ ut,
    const REAL *__restrict__ coef, const REAL *__restrict__ ecos, const REAL *__restrict__ esin,
    const unsigned int *__restrict__ iter, const unsigned int *__restrict__ ine,
    unsigned int ne, unsigned int ik, unsigned int sub)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ne * NTH) return;
    unsigned int t = g / NTH, k = g - t * NTH;
    if (sub >= iter[ik * NTH + k]) return;
    CONSTANTS
    TRI_K
    REAL fa1 = c[6] * co + c[9] * sn + kp1;
    REAL fa2 = c[7] * co + c[10] * sn + kp2;
    REAL fa3 = c[8] * co + c[11] * sn + kp3;
    ut[g] = nm * (fa1 * u[n1 * NTH + k] + fa2 * u[n2 * NTH + k] + fa3 * u[n3 * NTH + k]);
}

extern "C" __global__ void nscheme_node_forward(const REAL *__restrict__ u, REAL *__restrict__ out,
    const REAL *__restrict__ ut, const REAL *__restrict__ coef, const REAL *__restrict__ ecos,
    const REAL *__restrict__ esin, const REAL *__restrict__ dtk, const unsigned int *__restrict__ iter,
    const REAL *__restrict__ si_inv, const unsigned int *__restrict__ mlo, const unsigned int *__restrict__ mhi,
    const unsigned int *__restrict__ start, const unsigned int *__restrict__ slot,
    unsigned int np, unsigned int ne, unsigned int ik, unsigned int sub, int clip)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np * NTH) return;
    unsigned int i = g / NTH, k = g - i * NTH, kk = ik * NTH + k;
    REAL ui = u[g];
    if (sub >= iter[kk]) { out[g] = ui; return; }
    CONSTANTS
    REAL co = ecos[k], sn = esin[k];
    const REAL *cb = coef + (size_t)ik * ne * 12u;
    REAL st = zero;
    for (unsigned int q = start[i]; q < start[i + 1]; ++q) {
        unsigned int code = slot[q], t = code / 3u, p = code - 3u * t;
        const REAL *c = cb + (size_t)t * 12u + p;
        REAL kp = c[0] * co + c[3] * sn;
        kp = kp > zero ? kp : zero;
        st += kp * (ui - ut[t * NTH + k]);
    }
    unsigned int w = k < 32u ? mlo[i] : mhi[i];
    REAL m = (REAL)((w >> (k % 32u)) & 1u);
    REAL a = one - (REAL)((mhi[i] >> 31) & 1u);
    REAL v = ui - dtk[kk] * si_inv[i] * st * a;
    if (clip && v < zero) v = zero;
    out[g] = v * m;
}

extern "C" __global__ void nscheme_tri_transpose(const REAL *__restrict__ y, REAL *__restrict__ gt,
    const REAL *__restrict__ coef, const REAL *__restrict__ ecos, const REAL *__restrict__ esin,
    const REAL *__restrict__ dtk, const unsigned int *__restrict__ iter, const REAL *__restrict__ si_inv,
    const unsigned int *__restrict__ mlo, const unsigned int *__restrict__ mhi,
    const unsigned int *__restrict__ ine, unsigned int ne, unsigned int ik, unsigned int sub)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ne * NTH) return;
    unsigned int t = g / NTH, k = g - t * NTH, kk = ik * NTH + k;
    if (sub >= iter[kk]) return;
    CONSTANTS
    TRI_K
    unsigned int sh = k % 32u;
    bool lo = k < 32u;
    gt[g] = nm * dtk[kk] * (kp1 * V(n1) + kp2 * V(n2) + kp3 * V(n3));
}

extern "C" __global__ void nscheme_node_transpose(const REAL *__restrict__ y, REAL *__restrict__ out,
    const REAL *__restrict__ gt, const REAL *__restrict__ coef, const REAL *__restrict__ ecos,
    const REAL *__restrict__ esin, const REAL *__restrict__ dtk, const unsigned int *__restrict__ iter,
    const REAL *__restrict__ si_inv, const unsigned int *__restrict__ mlo, const unsigned int *__restrict__ mhi,
    const unsigned int *__restrict__ start, const unsigned int *__restrict__ slot,
    unsigned int np, unsigned int ne, unsigned int ik, unsigned int sub)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np * NTH) return;
    unsigned int j = g / NTH, k = g - j * NTH, kk = ik * NTH + k;
    REAL yj = y[g];
    if (sub >= iter[kk]) { out[g] = yj; return; }
    CONSTANTS
    REAL co = ecos[k], sn = esin[k];
    unsigned int sh = k % 32u;
    bool lo = k < 32u;
    REAL mj = (REAL)(((lo ? mlo[j] : mhi[j]) >> sh) & 1u);
    REAL vj = dtk[kk] * V(j);
    const REAL *cb = coef + (size_t)ik * ne * 12u;
    REAL acc = zero;
    for (unsigned int q = start[j]; q < start[j + 1]; ++q) {
        unsigned int code = slot[q], t = code / 3u, p = code - 3u * t;
        const REAL *c = cb + (size_t)t * 12u + p;
        REAL kp = c[0] * co + c[3] * sn;
        kp = kp > zero ? kp : zero;
        REAL fa = c[6] * co + c[9] * sn + kp;
        acc += kp * vj - fa * gt[t * NTH + k];
    }
    out[g] = mj * yj - acc;
}

/* Two-phase variant with DPT directions per unit (NTH a multiple of DPT):
   the triangle decoding, the vertex indices and the coefficients are loaded
   once for DPT directions. */

#ifndef DPT
#define DPT 4
#endif
#define NW (NTH / DPT)

extern "C" __global__ void nscheme_tri_forward_v(const REAL *__restrict__ u, REAL *__restrict__ ut,
    const REAL *__restrict__ coef, const REAL *__restrict__ ecos, const REAL *__restrict__ esin,
    const unsigned int *__restrict__ iter, const unsigned int *__restrict__ ine,
    unsigned int ne, unsigned int ik, unsigned int sub)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ne * NW) return;
    unsigned int t = g / NW, k0 = (g - t * NW) * DPT;
    CONSTANTS
    const REAL *c = coef + ((size_t)ik * ne + t) * 12u;
    REAL a1 = c[0], a2 = c[1], a3 = c[2], b1 = c[3], b2 = c[4], b3 = c[5];
    REAL A1 = c[6], A2 = c[7], A3 = c[8], B1 = c[9], B2 = c[10], B3 = c[11];
    const REAL *u1 = u + ine[3u * t] * NTH, *u2 = u + ine[3u * t + 1u] * NTH, *u3 = u + ine[3u * t + 2u] * NTH;
#pragma unroll
    for (unsigned int d = 0; d < DPT; ++d) {
        unsigned int k = k0 + d;
        if (sub >= iter[ik * NTH + k]) continue;
        REAL co = ecos[k], sn = esin[k];
        REAL k1 = a1 * co + b1 * sn, k2 = a2 * co + b2 * sn, k3 = a3 * co + b3 * sn;
        REAL kp1 = k1 > zero ? k1 : zero, kp2 = k2 > zero ? k2 : zero, kp3 = k3 > zero ? k3 : zero;
        REAL kn = (k1 < zero ? k1 : zero) + (k2 < zero ? k2 : zero) + (k3 < zero ? k3 : zero);
        REAL nm = -one / (-thr < kn ? -thr : kn);
        REAL fa1 = A1 * co + B1 * sn + kp1, fa2 = A2 * co + B2 * sn + kp2, fa3 = A3 * co + B3 * sn + kp3;
        ut[(size_t)t * NTH + k] = nm * (fa1 * u1[k] + fa2 * u2[k] + fa3 * u3[k]);
    }
}

extern "C" __global__ void nscheme_node_forward_v(const REAL *__restrict__ u, REAL *__restrict__ out,
    const REAL *__restrict__ ut, const REAL *__restrict__ coef, const REAL *__restrict__ ecos,
    const REAL *__restrict__ esin, const REAL *__restrict__ dtk, const unsigned int *__restrict__ iter,
    const REAL *__restrict__ si_inv, const unsigned int *__restrict__ mlo, const unsigned int *__restrict__ mhi,
    const unsigned int *__restrict__ start, const unsigned int *__restrict__ slot,
    unsigned int np, unsigned int ne, unsigned int ik, unsigned int sub, int clip)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np * NW) return;
    unsigned int i = g / NW, k0 = (g - i * NW) * DPT;
    CONSTANTS
    REAL co[DPT], sn[DPT], ui[DPT], st[DPT];
#pragma unroll
    for (unsigned int d = 0; d < DPT; ++d) {
        co[d] = ecos[k0 + d]; sn[d] = esin[k0 + d]; ui[d] = u[i * NTH + k0 + d]; st[d] = zero;
    }
    const REAL *cb = coef + (size_t)ik * ne * 12u;
    for (unsigned int q = start[i]; q < start[i + 1]; ++q) {
        unsigned int code = slot[q], t = code / 3u, p = code - 3u * t;
        const REAL *c = cb + (size_t)t * 12u + p;
        REAL a = c[0], b = c[3];
        const REAL *utr = ut + (size_t)t * NTH + k0;
#pragma unroll
        for (unsigned int d = 0; d < DPT; ++d) {
            REAL kp = a * co[d] + b * sn[d];
            kp = kp > zero ? kp : zero;
            st[d] += kp * (ui[d] - utr[d]);
        }
    }
    unsigned int wlo = mlo[i], whi = mhi[i];
    REAL aa = one - (REAL)((whi >> 31) & 1u), sii = si_inv[i];
#pragma unroll
    for (unsigned int d = 0; d < DPT; ++d) {
        unsigned int k = k0 + d, kk = ik * NTH + k;
        if (sub >= iter[kk]) { out[i * NTH + k] = ui[d]; continue; }
        unsigned int w = k < 32u ? wlo : whi;
        REAL m = (REAL)((w >> (k % 32u)) & 1u);
        REAL v = ui[d] - dtk[kk] * sii * st[d] * aa;
        if (clip && v < zero) v = zero;
        out[i * NTH + k] = v * m;
    }
}

extern "C" __global__ void nscheme_tri_transpose_v(const REAL *__restrict__ y, REAL *__restrict__ gt,
    const REAL *__restrict__ coef, const REAL *__restrict__ ecos, const REAL *__restrict__ esin,
    const REAL *__restrict__ dtk, const unsigned int *__restrict__ iter, const REAL *__restrict__ si_inv,
    const unsigned int *__restrict__ mlo, const unsigned int *__restrict__ mhi,
    const unsigned int *__restrict__ ine, unsigned int ne, unsigned int ik, unsigned int sub)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ne * NW) return;
    unsigned int t = g / NW, k0 = (g - t * NW) * DPT;
    CONSTANTS
    const REAL *c = coef + ((size_t)ik * ne + t) * 12u;
    REAL a1 = c[0], a2 = c[1], a3 = c[2], b1 = c[3], b2 = c[4], b3 = c[5];
    unsigned int n1 = ine[3u * t], n2 = ine[3u * t + 1u], n3 = ine[3u * t + 2u];
    unsigned int l1 = mlo[n1], h1 = mhi[n1], l2 = mlo[n2], h2 = mhi[n2], l3 = mlo[n3], h3 = mhi[n3];
    REAL s1 = (one - (REAL)((h1 >> 31) & 1u)) * si_inv[n1];
    REAL s2 = (one - (REAL)((h2 >> 31) & 1u)) * si_inv[n2];
    REAL s3 = (one - (REAL)((h3 >> 31) & 1u)) * si_inv[n3];
    const REAL *y1 = y + n1 * NTH, *y2 = y + n2 * NTH, *y3 = y + n3 * NTH;
#pragma unroll
    for (unsigned int d = 0; d < DPT; ++d) {
        unsigned int k = k0 + d, kk = ik * NTH + k, sh = k % 32u;
        if (sub >= iter[kk]) continue;
        bool lo = k < 32u;
        REAL co = ecos[k], sn = esin[k];
        REAL k1 = a1 * co + b1 * sn, k2 = a2 * co + b2 * sn, k3 = a3 * co + b3 * sn;
        REAL kp1 = k1 > zero ? k1 : zero, kp2 = k2 > zero ? k2 : zero, kp3 = k3 > zero ? k3 : zero;
        REAL kn = (k1 < zero ? k1 : zero) + (k2 < zero ? k2 : zero) + (k3 < zero ? k3 : zero);
        REAL nm = -one / (-thr < kn ? -thr : kn);
        REAL v1 = (REAL)(((lo ? l1 : h1) >> sh) & 1u) * s1 * y1[k];
        REAL v2 = (REAL)(((lo ? l2 : h2) >> sh) & 1u) * s2 * y2[k];
        REAL v3 = (REAL)(((lo ? l3 : h3) >> sh) & 1u) * s3 * y3[k];
        gt[(size_t)t * NTH + k] = nm * dtk[kk] * (kp1 * v1 + kp2 * v2 + kp3 * v3);
    }
}

extern "C" __global__ void nscheme_node_transpose_v(const REAL *__restrict__ y, REAL *__restrict__ out,
    const REAL *__restrict__ gt, const REAL *__restrict__ coef, const REAL *__restrict__ ecos,
    const REAL *__restrict__ esin, const REAL *__restrict__ dtk, const unsigned int *__restrict__ iter,
    const REAL *__restrict__ si_inv, const unsigned int *__restrict__ mlo, const unsigned int *__restrict__ mhi,
    const unsigned int *__restrict__ start, const unsigned int *__restrict__ slot,
    unsigned int np, unsigned int ne, unsigned int ik, unsigned int sub)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np * NW) return;
    unsigned int j = g / NW, k0 = (g - j * NW) * DPT;
    CONSTANTS
    unsigned int wlo = mlo[j], whi = mhi[j];
    REAL sj = (one - (REAL)((whi >> 31) & 1u)) * si_inv[j];
    REAL co[DPT], sn[DPT], yj[DPT], mj[DPT], vj[DPT], acc[DPT];
#pragma unroll
    for (unsigned int d = 0; d < DPT; ++d) {
        unsigned int k = k0 + d;
        co[d] = ecos[k]; sn[d] = esin[k]; yj[d] = y[j * NTH + k]; acc[d] = zero;
        mj[d] = (REAL)(((k < 32u ? wlo : whi) >> (k % 32u)) & 1u);
        vj[d] = mj[d] * sj * dtk[ik * NTH + k] * yj[d];
    }
    const REAL *cb = coef + (size_t)ik * ne * 12u;
    for (unsigned int q = start[j]; q < start[j + 1]; ++q) {
        unsigned int code = slot[q], t = code / 3u, p = code - 3u * t;
        const REAL *c = cb + (size_t)t * 12u + p;
        REAL a = c[0], b = c[3], A = c[6], B = c[9];
        const REAL *gtr = gt + (size_t)t * NTH + k0;
#pragma unroll
        for (unsigned int d = 0; d < DPT; ++d) {
            REAL kp = a * co[d] + b * sn[d];
            kp = kp > zero ? kp : zero;
            REAL fa = A * co[d] + B * sn[d] + kp;
            acc[d] += kp * vj[d] - fa * gtr[d];
        }
    }
#pragma unroll
    for (unsigned int d = 0; d < DPT; ++d) {
        unsigned int k = k0 + d;
        out[j * NTH + k] = sub >= iter[ik * NTH + k] ? yj[d] : mj[d] * yj[d] - acc[d];
    }
}
"#;

/// Directions per unit of the `two_phase_x4` kernels.
const DPT: usize = 4;

pub trait CReal: Scalar + DeviceRepr + ValidAsZeroBits + Default {
    const C_NAME: &'static str;
}
impl CReal for f32 {
    const C_NAME: &'static str = "float";
}
impl CReal for f64 {
    const C_NAME: &'static str = "double";
}

pub struct CudaGrid<F: CReal> {
    stream: Arc<CudaStream>,
    fwd: CudaFunction,
    tr: CudaFunction,
    np: usize,
    ne: usize,
    nk: usize,
    nth: usize,
    block: u32,
    iter: Vec<u32>,
    cx: CudaSlice<F>,
    cy: CudaSlice<F>,
    ecos: CudaSlice<F>,
    esin: CudaSlice<F>,
    dtk: CudaSlice<F>,
    iter_d: CudaSlice<u32>,
    si_inv: CudaSlice<F>,
    mlo: CudaSlice<u32>,
    mhi: CudaSlice<u32>,
    start: CudaSlice<u32>,
    slot: CudaSlice<u32>,
    ine: CudaSlice<u32>,
    ien: CudaSlice<F>,
    fwd_tri: CudaFunction,
    fwd_node: CudaFunction,
    tr_tri: CudaFunction,
    tr_node: CudaFunction,
    fwd_tri_v: CudaFunction,
    fwd_node_v: CudaFunction,
    tr_tri_v: CudaFunction,
    tr_node_v: CudaFunction,
    /// `nk ne 12` coefficients of the two-phase kernels.
    coef: CudaSlice<F>,
    /// `ne nth` values per triangle and direction, written by the triangle pass.
    scratch: CudaSlice<F>,
    /// Kernel variant: 0 fused, 1 two-phase, 2 two-phase with four directions per unit.
    variant: usize,
    pub compile_seconds: f64,
    coef_seconds: f64,
}

impl<F: CReal> CudaGrid<F> {
    pub fn new(ctx: &Arc<CudaContext>, g: &Geometry, block: u32) -> Result<Self> {
        let t = std::time::Instant::now();
        let opts = CompileOptions {
            arch: Some("compute_80"),
            options: vec![format!("-DREAL={}", F::C_NAME), format!("-DNTH={}u", g.nth), format!("-DDPT={DPT}u")],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(SRC, opts)?;
        let module = ctx.load_module(ptx)?;
        let fwd = module.load_function("nscheme_forward")?;
        let tr = module.load_function("nscheme_transpose")?;
        let fwd_tri = module.load_function("nscheme_tri_forward")?;
        let fwd_node = module.load_function("nscheme_node_forward")?;
        let tr_tri = module.load_function("nscheme_tri_transpose")?;
        let tr_node = module.load_function("nscheme_node_transpose")?;
        let coef_fn = module.load_function("nscheme_coef")?;
        let fwd_tri_v = module.load_function("nscheme_tri_forward_v")?;
        let fwd_node_v = module.load_function("nscheme_node_forward_v")?;
        let tr_tri_v = module.load_function("nscheme_tri_transpose_v")?;
        let tr_node_v = module.load_function("nscheme_node_transpose_v")?;
        let compile_seconds = t.elapsed().as_secs_f64();
        let stream = ctx.default_stream();
        let upf = |v: &[f64]| -> Result<CudaSlice<F>> {
            let w: Vec<F> = v.iter().map(|&x| F::of_f64(x)).collect();
            Ok(stream.clone_htod(&w)?)
        };
        let (si_inv, dtk, mlo, mhi) = g.host_arrays();
        let cx = upf(&g.cx)?;
        let cy = upf(&g.cy)?;
        let ine = stream.clone_htod(&g.ine)?;
        let ien = upf(&g.ien)?;
        let mut coef = stream.alloc_zeros::<F>(g.nk * g.ne * 12)?;
        stream.synchronize()?;
        let t = std::time::Instant::now();
        let (np, ne) = (g.np as u32, g.ne as u32);
        let cfg = LaunchConfig { grid_dim: (ne.div_ceil(block), 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
        for ik in 0..g.nk as u32 {
            let mut b = stream.launch_builder(&coef_fn);
            b.arg(&cx).arg(&cy).arg(&ine).arg(&ien).arg(&mut coef).arg(&np).arg(&ne).arg(&ik);
            unsafe { b.launch(cfg) }?;
        }
        stream.synchronize()?;
        let coef_seconds = t.elapsed().as_secs_f64();
        Ok(CudaGrid {
            fwd_tri,
            fwd_node,
            tr_tri,
            tr_node,
            fwd_tri_v,
            fwd_node_v,
            tr_tri_v,
            tr_node_v,
            coef,
            scratch: stream.alloc_zeros::<F>(g.ne * g.nth)?,
            variant: 0,
            coef_seconds,
            np: g.np,
            ne: g.ne,
            nk: g.nk,
            nth: g.nth,
            block,
            iter: g.iter.clone(),
            cx,
            cy,
            ecos: upf(&g.ecos)?,
            esin: upf(&g.esin)?,
            dtk: upf(&dtk)?,
            iter_d: stream.clone_htod(&g.iter)?,
            si_inv: upf(&si_inv)?,
            mlo: stream.clone_htod(&mlo)?,
            mhi: stream.clone_htod(&mhi)?,
            start: stream.clone_htod(&g.start)?,
            slot: stream.clone_htod(&g.slot)?,
            ine,
            ien,
            fwd,
            tr,
            stream,
            compile_seconds,
        })
    }

    fn cfg_for(&self, n: usize) -> LaunchConfig {
        let n = n as u32;
        LaunchConfig { grid_dim: (n.div_ceil(self.block), 1, 1), block_dim: (self.block, 1, 1), shared_mem_bytes: 0 }
    }

    fn cfg(&self) -> LaunchConfig {
        self.cfg_for(self.np * self.nth)
    }

    fn forward_fused(&self, u: &CudaSlice<F>, out: &mut CudaSlice<F>, ik: usize, sub: u32, clip: bool) {
        let (np, ik, clip) = (self.np as u32, ik as u32, clip as i32);
        let mut b = self.stream.launch_builder(&self.fwd);
        b.arg(u).arg(out);
        b.arg(&self.cx).arg(&self.cy).arg(&self.ecos).arg(&self.esin).arg(&self.dtk).arg(&self.iter_d);
        b.arg(&self.si_inv).arg(&self.mlo).arg(&self.mhi).arg(&self.start).arg(&self.slot);
        b.arg(&self.ine).arg(&self.ien).arg(&np).arg(&ik).arg(&sub).arg(&clip);
        unsafe { b.launch(self.cfg()) }.expect("forward launch");
    }

    fn transpose_fused(&self, y: &CudaSlice<F>, out: &mut CudaSlice<F>, ik: usize, sub: u32) {
        let (np, ik) = (self.np as u32, ik as u32);
        let mut b = self.stream.launch_builder(&self.tr);
        b.arg(y).arg(out);
        b.arg(&self.cx).arg(&self.cy).arg(&self.ecos).arg(&self.esin).arg(&self.dtk).arg(&self.iter_d);
        b.arg(&self.si_inv).arg(&self.mlo).arg(&self.mhi).arg(&self.start).arg(&self.slot);
        b.arg(&self.ine).arg(&self.ien).arg(&np).arg(&ik).arg(&sub);
        unsafe { b.launch(self.cfg()) }.expect("transpose launch");
    }

    // The triangle pass writes `scratch` through a shared reference: one
    // stream orders it before the node pass that reads it.
    fn forward_two_phase(&self, u: &CudaSlice<F>, out: &mut CudaSlice<F>, ik: usize, sub: u32, clip: bool, x4: bool) {
        let (np, ne, ik, clip) = (self.np as u32, self.ne as u32, ik as u32, clip as i32);
        let per = if x4 { DPT } else { 1 };
        let mut b = self.stream.launch_builder(if x4 { &self.fwd_tri_v } else { &self.fwd_tri });
        b.arg(u).arg(&self.scratch).arg(&self.coef).arg(&self.ecos).arg(&self.esin).arg(&self.iter_d);
        b.arg(&self.ine).arg(&ne).arg(&ik).arg(&sub);
        unsafe { b.launch(self.cfg_for(self.ne * self.nth / per)) }.expect("triangle pass launch");
        let mut b = self.stream.launch_builder(if x4 { &self.fwd_node_v } else { &self.fwd_node });
        b.arg(u).arg(out).arg(&self.scratch).arg(&self.coef).arg(&self.ecos).arg(&self.esin);
        b.arg(&self.dtk).arg(&self.iter_d).arg(&self.si_inv).arg(&self.mlo).arg(&self.mhi);
        b.arg(&self.start).arg(&self.slot).arg(&np).arg(&ne).arg(&ik).arg(&sub).arg(&clip);
        unsafe { b.launch(self.cfg_for(self.np * self.nth / per)) }.expect("node pass launch");
    }

    fn transpose_two_phase(&self, y: &CudaSlice<F>, out: &mut CudaSlice<F>, ik: usize, sub: u32, x4: bool) {
        let (np, ne, ik) = (self.np as u32, self.ne as u32, ik as u32);
        let per = if x4 { DPT } else { 1 };
        let mut b = self.stream.launch_builder(if x4 { &self.tr_tri_v } else { &self.tr_tri });
        b.arg(y).arg(&self.scratch).arg(&self.coef).arg(&self.ecos).arg(&self.esin).arg(&self.dtk);
        b.arg(&self.iter_d).arg(&self.si_inv).arg(&self.mlo).arg(&self.mhi).arg(&self.ine);
        b.arg(&ne).arg(&ik).arg(&sub);
        unsafe { b.launch(self.cfg_for(self.ne * self.nth / per)) }.expect("triangle pass launch");
        let mut b = self.stream.launch_builder(if x4 { &self.tr_node_v } else { &self.tr_node });
        b.arg(y).arg(out).arg(&self.scratch).arg(&self.coef).arg(&self.ecos).arg(&self.esin);
        b.arg(&self.dtk).arg(&self.iter_d).arg(&self.si_inv).arg(&self.mlo).arg(&self.mhi);
        b.arg(&self.start).arg(&self.slot).arg(&np).arg(&ne).arg(&ik).arg(&sub);
        unsafe { b.launch(self.cfg_for(self.np * self.nth / per)) }.expect("node pass launch");
    }
}

impl<F: CReal> Sweeper for CudaGrid<F> {
    type Buf = CudaSlice<F>;

    fn label(&self) -> String {
        format!("cuda-c {} {}", F::NAME, self.variants()[self.variant])
    }
    fn variants(&self) -> &'static [&'static str] {
        if self.nth % DPT == 0 {
            &["fused", "two_phase", "two_phase_x4"]
        } else {
            &["fused", "two_phase"]
        }
    }
    fn variant(&self) -> usize {
        self.variant
    }
    fn set_variant(&mut self, v: usize) {
        self.variant = v;
    }
    fn coef_seconds(&self) -> f64 {
        self.coef_seconds
    }
    fn float_bytes(&self) -> usize {
        std::mem::size_of::<F>()
    }
    fn dims(&self) -> Dims {
        Dims { np: self.np, ne: self.ne, nk: self.nk, nth: self.nth }
    }
    fn iter(&self) -> &[u32] {
        &self.iter
    }
    fn block(&self) -> u32 {
        self.block
    }
    fn set_block(&mut self, b: u32) {
        self.block = b;
    }

    fn upload(&self, v: &[f64]) -> CudaSlice<F> {
        let w: Vec<F> = v.iter().map(|&x| F::of_f64(x)).collect();
        self.stream.clone_htod(&w).expect("host to device copy")
    }

    fn download(&self, b: &CudaSlice<F>) -> Vec<f64> {
        self.stream.clone_dtoh(b).expect("device to host copy").iter().map(|&x| x.as_f64()).collect()
    }

    fn empty(&self) -> CudaSlice<F> {
        self.stream.alloc_zeros::<F>(self.np * self.nth).expect("device allocation")
    }

    fn sync(&self) {
        self.stream.synchronize().expect("stream synchronize");
    }

    fn forward(&self, u: &CudaSlice<F>, out: &mut CudaSlice<F>, ik: usize, sub: u32, clip: bool) {
        match self.variant {
            0 => self.forward_fused(u, out, ik, sub, clip),
            1 => self.forward_two_phase(u, out, ik, sub, clip, false),
            _ => self.forward_two_phase(u, out, ik, sub, clip, true),
        }
    }

    fn transpose(&self, y: &CudaSlice<F>, out: &mut CudaSlice<F>, ik: usize, sub: u32) {
        match self.variant {
            0 => self.transpose_fused(y, out, ik, sub),
            1 => self.transpose_two_phase(y, out, ik, sub, false),
            _ => self.transpose_two_phase(y, out, ik, sub, true),
        }
    }
}
