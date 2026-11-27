// The propagation of the model in CUDA C (milestone M4): the value kernels
// needed to recompute the forward pass inside a step (W3KTP3 refraction and
// frequency shift, the geographic N-scheme sub-step, the conversions between
// VA and AC) and their adjoints, the transposes of the tangent kernels of
// hww3_wave::prop_tl (M3). The arithmetic of the values is that of the
// CubeCL kernels of hww3_wave::prop (double precision, the f32 constants of
// those kernels). Layouts: va[(ik np + i) NTH + th], wn and cg [j np + i]
// for j = 0 .. NK + 1, the geographic arrays as the CubeCL kernels (coef
// [(ik ne + t) 12 + c], ccoef [6 t + c], iter and dtk [ik NTH + th]). NK, NTH
// and GB (threads per block) are macros.

typedef unsigned int uint;

__device__ __forceinline__ double fl(float x) { return (double)x; }

// ---------------------------------------------------------------------------
// ULTIMATE QUICKEST (W3QCK1 / W3QCK2) and its transpose
// ---------------------------------------------------------------------------

__device__ double uq_limit(double qb, double qc, double qu, double qd, double cfl)
{
    double dq = qd - qu;
    double m = fmax(fl(1.0e-15f), fabs(dq));
    double dqnz = dq >= 0.0 ? m : -m;
    double qcn = fmin(fl(1.1f), fmax(fl(-0.1f), (qc - qu) / dqnz));
    double qbn0 = fmax((qb - qu) / dqnz, qcn);
    double qbn = fmin(fmin(qbn0, 1.0), qcn / fmax(fl(1.0e-10f), fabs(cfl)));
    double qbr = qu + qbn * dq;
    double cfac = 2.0 * fabs(qcn - 0.5) >= 1.0 ? 1.0 : 0.0;
    return (1.0 - cfac) * qbr + cfac * qc;
}

// transpose of uq_limit_d of prop_tl: from the cotangent rb of the limited
// value, adds those of (qb, qc, qu, qd)
__device__ void uq_limit_adj(double qb, double qc, double qu, double qd, double cfl, double rb, double &bb, double &bc,
                             double &bu, double &bd)
{
    const double eps = fl(1.0e-15f);
    double dq = qd - qu;
    double m = fmax(eps, fabs(dq));
    double dqnz = dq >= 0.0 ? m : -m;
    double r = (qc - qu) / dqnz;
    double qcn = fmin(fl(1.1f), fmax(fl(-0.1f), r));
    double s = (qb - qu) / dqnz;
    double qbn0 = fmax(s, qcn);
    double cl = fmax(fl(1.0e-10f), fabs(cfl));
    double l2 = qcn / cl;
    double a = fmin(qbn0, 1.0);
    double qbn = fmin(a, l2);
    double cfac = 2.0 * fabs(qcn - 0.5) >= 1.0 ? 1.0 : 0.0;
    double gc = cfac * rb, gbr = (1.0 - cfac) * rb;
    double gu = gbr, gbn = gbr * dq, gdq = gbr * qbn;
    double ga = 0.0, gcn = 0.0;
    if (a < l2) ga = gbn; else gcn += gbn / cl;
    double gbn0 = qbn0 < 1.0 ? ga : 0.0;
    double gs = 0.0;
    if (s > qcn) gs = gbn0; else gcn += gbn0;
    double gnz = 0.0;
    double gb = gs / dqnz;
    gu -= gs / dqnz;
    gnz += -gs * s / dqnz;
    double gr = (r > fl(-0.1f) && r < fl(1.1f)) ? gcn : 0.0;
    gc += gr / dqnz;
    gu -= gr / dqnz;
    gnz += -gr * r / dqnz;
    if (fabs(dq) > eps) gdq += gnz;
    bb += gb;
    bc += gc;
    bu += gu - gdq;
    bd += gdq;
}

// ---------------------------------------------------------------------------
// W3KTP3: refraction (theta)
// ---------------------------------------------------------------------------

__device__ double dsdd(double wn, double cg, double sig, double depth)
{
    double v = fmax(0.0, cg * wn - 0.5 * sig) / depth;
    return depth * wn < 5.0 ? v : 0.0;
}

__device__ double theta_cfl(double es2, double esc, double ec2, double ec, double es, double dcyx, double dcxxyy, double dcxy,
                            double frg, double frk, double gx, double gy, double ctmax)
{
    double vc = es2 * dcyx + esc * dcxxyy - ec2 * dcxy;
    double v = vc + frg * ec + frk * (es * gx - ec * gy);
    double lim = fmin(fabs(v), ctmax);
    return v >= 0.0 ? lim : -lim;
}

__device__ double theta_flux(double ct, double ctp, double qtm, double qt, double qtp, double qtpp)
{
    const double sixth = 1.0 / 6.0;
    double cf = 0.5 * (ct + ctp);
    bool up = cf < 0.0;
    double qc = up ? qtp : qt, qcm = up ? qt : qtm, qcp = up ? qtpp : qtp;
    double qb = 0.5 * ((1.0 - cf) * qtp + (1.0 + cf) * qt) - (1.0 - cf * cf) * sixth * (qcm - 2.0 * qc + qcp);
    double qu = up ? qcp : qcm, qd = up ? qcm : qcp;
    return cf * uq_limit(qb, qc, qu, qd, cf);
}

// transpose of theta_flux_d: from the cotangent fb of the flux, adds those
// of the four cells t - 1 .. t + 2
__device__ void theta_flux_adj(double ct, double ctp, double qtm, double qt, double qtp, double qtpp, double fb, double &bm,
                               double &b0, double &bp, double &bpp)
{
    const double sixth = 1.0 / 6.0;
    double cf = 0.5 * (ct + ctp);
    bool up = cf < 0.0;
    double qc = up ? qtp : qt, qcm = up ? qt : qtm, qcp = up ? qtpp : qtp;
    double qb = 0.5 * ((1.0 - cf) * qtp + (1.0 + cf) * qt) - (1.0 - cf * cf) * sixth * (qcm - 2.0 * qc + qcp);
    double qu = up ? qcp : qcm, qd = up ? qcm : qcp;
    double gb = 0.0, gc = 0.0, gu = 0.0, gd = 0.0;
    uq_limit_adj(qb, qc, qu, qd, cf, cf * fb, gb, gc, gu, gd);
    double gcm = 0.0, gcp = 0.0;
    if (up) { gcp += gu; gcm += gd; } else { gcm += gu; gcp += gd; }
    double k6 = -(1.0 - cf * cf) * sixth * gb;
    bp += 0.5 * (1.0 - cf) * gb;
    b0 += 0.5 * (1.0 + cf) * gb;
    gcm += k6;
    gc += -2.0 * k6;
    gcp += k6;
    if (up) { bp += gc; b0 += gcm; bpp += gcp; } else { b0 += gc; bm += gcm; bp += gcp; }
}

#define PROP_THETA_ARGS                                                                                               \
    const double *wn, const double *cg, const double *sig, const double *dw, const double *dddx, const double *dddy, \
        const double *dcxdx, const double *dcxdy, const double *dcydx, const double *dcydy, const double *cthg0,     \
        const uint *nflag, const double *ecos, const double *esin, const double *es2, const double *ec2,             \
        const double *esc, double facth, double ctmax, double flcur, double dmin, uint np

// the CFL numbers of the row of node i, frequency ik
__device__ bool theta_row(PROP_THETA_ARGS, uint i, uint ik, double *c)
{
    if ((nflag[i] & 5u) != 5u) return false;
    double depth = fmax(dmin, dw[i]);
    double w = wn[(ik + 1) * np + i], cgv = cg[(ik + 1) * np + i];
    double frk = facth * dsdd(w, cgv, sig[ik + 1], depth) / w;
    double frg = facth * cthg0[i] * cgv;
    double dcyx = facth * dcydx[i] * flcur;
    double dcxxyy = facth * (dcxdx[i] - dcydy[i]) * flcur;
    double dcxy = facth * dcxdy[i] * flcur;
    double gx = dddx[i], gy = dddy[i];
    for (uint th = 0; th < NTH; th++)
        c[th] = theta_cfl(es2[th], esc[th], ec2[th], ecos[th], esin[th], dcyx, dcxxyy, dcxy, frg, frk, gx, gy, ctmax);
    return true;
}

// the value: one thread per (node, frequency) row, in place
extern "C" __global__ void __launch_bounds__(GB)
ktp3_theta_fwd(double *va, PROP_THETA_ARGS)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np * NK) return;
    uint ik = g / np, i = g - ik * np;
    double c[NTH], q[NTH];
    if (!theta_row(wn, cg, sig, dw, dddx, dddy, dcxdx, dcxdy, dcydx, dcydy, cthg0, nflag, ecos, esin, es2, ec2, esc, facth, ctmax,
                   flcur, dmin, np, i, ik, c))
        return;
    double *row = va + (unsigned long long)g * NTH;
    for (uint th = 0; th < NTH; th++) q[th] = row[th];
    const uint l = NTH - 1;
    double fprev = theta_flux(c[l], c[0], q[l - 1], q[l], q[0], q[1]);
    for (uint th = 0; th < NTH; th++) {
        uint tp = (th + 1) % NTH, tpp = (th + 2) % NTH, tm = (th + NTH - 1) % NTH;
        double f = theta_flux(c[th], c[tp], q[tm], q[th], q[tp], q[tpp]);
        row[th] = fmax(0.0, q[th] + fprev - f);
        fprev = f;
    }
}

// the adjoint: va the values before the refraction, ab the cotangent after
// it, replaced by the cotangent before it
extern "C" __global__ void __launch_bounds__(GB)
ktp3_theta_adj(const double *va, double *ab, PROP_THETA_ARGS)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np * NK) return;
    uint ik = g / np, i = g - ik * np;
    double c[NTH], q[NTH], gb[NTH], qb[NTH];
    if (!theta_row(wn, cg, sig, dw, dddx, dddy, dcxdx, dcxdy, dcydx, dcydy, cthg0, nflag, ecos, esin, es2, ec2, esc, facth, ctmax,
                   flcur, dmin, np, i, ik, c))
        return;
    const double *row = va + (unsigned long long)g * NTH;
    double *brow = ab + (unsigned long long)g * NTH;
    for (uint th = 0; th < NTH; th++) q[th] = row[th];
    const uint l = NTH - 1;
    // the values, for the clamp at zero
    double fprev = theta_flux(c[l], c[0], q[l - 1], q[l], q[0], q[1]);
    for (uint th = 0; th < NTH; th++) {
        uint tp = (th + 1) % NTH, tpp = (th + 2) % NTH, tm = (th + NTH - 1) % NTH;
        double f = theta_flux(c[th], c[tp], q[tm], q[th], q[tp], q[tpp]);
        double v = q[th] + fprev - f;
        gb[th] = v > 0.0 ? brow[th] : 0.0;
        qb[th] = gb[th];
        fprev = f;
    }
    // flux th (interface th | th + 1): - in cell th, + in cell th + 1
    for (uint th = 0; th < NTH; th++) {
        uint tp = (th + 1) % NTH, tpp = (th + 2) % NTH, tm = (th + NTH - 1) % NTH;
        double fb = gb[tp] - gb[th];
        theta_flux_adj(c[th], c[tp], q[tm], q[th], q[tp], q[tpp], fb, qb[tm], qb[th], qb[tp], qb[tpp]);
    }
    for (uint th = 0; th < NTH; th++) brow[th] = qb[th];
}

// ---------------------------------------------------------------------------
// W3KTP3: frequency shift (k)
// ---------------------------------------------------------------------------

__device__ double k_vel(double w, double c, double s, double db, double depth, double fkd, double fkc, double fack, double ctmax)
{
    double d = dsdd(w, c, s, depth);
    double velfac = fack / db;
    double v = (fkd / c * d + w * fkc) * velfac;
    double lim = fmin(fabs(v), ctmax);
    return (v >= 0.0 ? lim : -lim) / velfac;
}

#define PROP_K_ARGS                                                                                                   \
    const double *wn, const double *cg, const double *sig, const double *dsip, const double *dw, const double *dddx, \
        const double *dddy, const double *cx, const double *cy, const double *dcxdx, const double *dcxdy,            \
        const double *dcydx, const double *dcydy, const uint *nflag, const double *es2, const double *ec2,           \
        const double *esc, double fack, double ctmax, double fachfa, double dmin, uint np

// the cells 1 .. NK + 1 of column (i, th): velocities and widths
__device__ bool k_column(PROP_K_ARGS, uint i, uint th, double *vel, double *db)
{
    if ((nflag[i] & 5u) != 5u) return false;
    double depth = fmax(dmin, dw[i]);
    double fkd = cx[i] * dddx[i] + cy[i] * dddy[i];
    double fkc = ec2[th] * (-dcxdx[i]) + esc[th] * (-(dcxdy[i] + dcydx[i])) + es2[th] * (-dcydy[i]);
    for (uint j = 1; j <= NK + 1; j++) {
        double c = cg[j * np + i];
        db[j] = dsip[j] / c;
        vel[j] = k_vel(wn[j * np + i], c, sig[j], db[j], depth, fkd, fkc, fack, ctmax);
    }
    return true;
}

// the face value at interface m | m + 1 (1 <= m < NK) from cells m - 1 .. m + 2
// (q[0] = 0, q[NK + 1] = FACHFA q[NK]); its parameters for the transpose
struct KFace {
    double v, cf, qb, qc, qu, qd, curv, dmi, dmim;
    bool up;
};

__device__ KFace k_face(const double *q, const double *vel, const double *db, const double *wn, uint np, uint i, uint m, double fack)
{
    const double sixth = 1.0 / 6.0;
    KFace f;
    double dm_l = wn[m * np + i] - wn[(m - 1) * np + i];
    double dm_c = wn[(m + 1) * np + i] - wn[m * np + i];
    double dm_h = wn[(m + 2) * np + i] - wn[(m + 1) * np + i];
    f.v = 0.5 * (vel[m] + vel[m + 1]);
    f.cf = fack * f.v / dm_c;
    f.up = f.cf < 0.0;
    double q_l = q[m - 1], q_c = q[m], q_h = q[m + 1], q_hh = q[m + 2];
    f.qc = f.up ? q_h : q_c;
    double qcm = f.up ? q_c : q_l, qcp = f.up ? q_hh : q_h;
    double dbi = f.up ? db[m + 1] : db[m];
    f.dmi = f.up ? dm_h : dm_c;
    f.dmim = f.up ? dm_c : dm_l;
    f.curv = dm_c * dm_c / dbi * (1.0 - f.cf * f.cf) * sixth;
    f.qb = 0.5 * ((1.0 - f.cf) * q_h + (1.0 + f.cf) * q_c) - f.curv * ((qcp - f.qc) / f.dmi - (f.qc - qcm) / f.dmim);
    f.qu = f.up ? qcp : qcm;
    f.qd = f.up ? qcm : qcp;
    return f;
}

// the fluxes F[m] through the interfaces m | m + 1, m = 0 .. NK
__device__ void k_fluxes(const double *q, const double *vel, const double *db, const double *wn, uint np, uint i, double fack, double *F)
{
    F[0] = vel[1] * (vel[1] >= 0.0 ? 0.0 : q[1]);
    for (uint m = 1; m < NK; m++) {
        KFace f = k_face(q, vel, db, wn, np, i, m, fack);
        F[m] = f.v * uq_limit(f.qb, f.qc, f.qu, f.qd, f.cf);
    }
    F[NK] = vel[NK] * (vel[NK] >= 0.0 ? q[NK] : q[NK + 1]);
}

// the value: one thread per (node, direction) column, in place
extern "C" __global__ void __launch_bounds__(GB)
ktp3_k_fwd(double *va, PROP_K_ARGS)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np * NTH) return;
    uint i = g / NTH, th = g - i * NTH;
    double vel[NK + 2], db[NK + 2], q[NK + 3], F[NK + 1];
    if (!k_column(wn, cg, sig, dsip, dw, dddx, dddy, cx, cy, dcxdx, dcxdy, dcydx, dcydy, nflag, es2, ec2, esc, fack, ctmax, fachfa,
                  dmin, np, i, th, vel, db))
        return;
    const unsigned long long stride = (unsigned long long)np * NTH, c0 = (unsigned long long)i * NTH + th;
    q[0] = 0.0;
    for (uint j = 1; j <= NK; j++) q[j] = va[c0 + (j - 1) * stride];
    q[NK + 1] = fachfa * q[NK];
    q[NK + 2] = 0.0;
    k_fluxes(q, vel, db, wn, np, i, fack, F);
    for (uint m = 1; m <= NK; m++) va[c0 + (m - 1) * stride] = fmax(0.0, q[m] + fack / db[m] * (F[m - 1] - F[m]));
}

// the adjoint: va the values before the frequency shift, ab the cotangent
// after it, replaced by the cotangent before it
extern "C" __global__ void __launch_bounds__(GB)
ktp3_k_adj(const double *va, double *ab, PROP_K_ARGS)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np * NTH) return;
    uint i = g / NTH, th = g - i * NTH;
    double vel[NK + 2], db[NK + 2], q[NK + 3], F[NK + 1], qb[NK + 3];
    if (!k_column(wn, cg, sig, dsip, dw, dddx, dddy, cx, cy, dcxdx, dcxdy, dcydx, dcydy, nflag, es2, ec2, esc, fack, ctmax, fachfa,
                  dmin, np, i, th, vel, db))
        return;
    const unsigned long long stride = (unsigned long long)np * NTH, c0 = (unsigned long long)i * NTH + th;
    q[0] = 0.0;
    for (uint j = 1; j <= NK; j++) q[j] = va[c0 + (j - 1) * stride];
    q[NK + 1] = fachfa * q[NK];
    q[NK + 2] = 0.0;
    k_fluxes(q, vel, db, wn, np, i, fack, F);
    for (uint j = 0; j < NK + 3; j++) qb[j] = 0.0;
    double Fb[NK + 1];
    for (uint m = 0; m <= NK; m++) Fb[m] = 0.0;
    for (uint m = 1; m <= NK; m++) {
        double v = q[m] + fack / db[m] * (F[m - 1] - F[m]);
        double gm = v > 0.0 ? ab[c0 + (m - 1) * stride] : 0.0;
        qb[m] += gm;
        Fb[m - 1] += gm * fack / db[m];
        Fb[m] -= gm * fack / db[m];
    }
    if (vel[1] < 0.0) qb[1] += Fb[0] * vel[1];
    for (uint m = 1; m < NK; m++) {
        KFace f = k_face(q, vel, db, wn, np, i, m, fack);
        double bb = 0.0, bc = 0.0, bu = 0.0, bd = 0.0;
        uq_limit_adj(f.qb, f.qc, f.qu, f.qd, f.cf, f.v * Fb[m], bb, bc, bu, bd);
        double bcm = 0.0, bcp = 0.0;
        if (f.up) { bcp += bu; bcm += bd; } else { bcm += bu; bcp += bd; }
        // qb = 0.5 ((1 - cf) q_h + (1 + cf) q_c) - curv ((qcp - qc) / dmi - (qc - qcm) / dmim)
        qb[m + 1] += 0.5 * (1.0 - f.cf) * bb;
        qb[m] += 0.5 * (1.0 + f.cf) * bb;
        double kk = -f.curv * bb;
        bcp += kk / f.dmi;
        bc += -kk / f.dmi - kk / f.dmim;
        bcm += kk / f.dmim;
        if (f.up) { qb[m + 1] += bc; qb[m] += bcm; qb[m + 2] += bcp; } else { qb[m] += bc; qb[m - 1] += bcm; qb[m + 1] += bcp; }
    }
    qb[NK] += Fb[NK] * vel[NK] * (vel[NK] >= 0.0 ? 1.0 : 0.0);
    qb[NK + 1] += Fb[NK] * vel[NK] * (vel[NK] >= 0.0 ? 0.0 : 1.0);
    // the ghost cell NK + 1 = FACHFA q[NK]; cell 0 and NK + 2 carry nothing
    qb[NK] += fachfa * qb[NK + 1];
    for (uint m = 1; m <= NK; m++) ab[c0 + (m - 1) * stride] = qb[m];
}

// ---------------------------------------------------------------------------
// The geographic sub-steps of frequency ik (N-scheme, PDLIB_W3XYPFSN2)
// ---------------------------------------------------------------------------

// AC = VA CLATS / CG of frequency ik
extern "C" __global__ void __launch_bounds__(GB)
va_to_ac(const double *va, double *ac, const double *cg, const double *clats, uint np, uint ik)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np * NTH) return;
    uint i = g / NTH;
    ac[g] = va[(unsigned long long)ik * np * NTH + g] / cg[(ik + 1) * np + i] * clats[i];
}

// adjoint of VA(ik) = max(0, CG / CLATS AC): acb from the cotangent of VA(ik)
extern "C" __global__ void __launch_bounds__(GB)
ac_to_va_adj(const double *ac, const double *vab, double *acb, const double *cg, const double *clats, uint np, uint ik)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np * NTH) return;
    uint i = g / NTH;
    double f = cg[(ik + 1) * np + i] / clats[i];
    acb[g] = f * ac[g] > 0.0 ? f * vab[(unsigned long long)ik * np * NTH + g] : 0.0;
}

// adjoint of AC = VA(ik) CLATS / CG: the cotangent of VA(ik) (replaced)
extern "C" __global__ void __launch_bounds__(GB)
va_to_ac_adj(const double *acb, double *vab, const double *cg, const double *clats, uint np, uint ik)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np * NTH) return;
    uint i = g / NTH;
    vab[(unsigned long long)ik * np * NTH + g] = acb[g] / cg[(ik + 1) * np + i] * clats[i];
}

// the triangle values of sub-step sub (as geo_tri_x4)
extern "C" __global__ void __launch_bounds__(GB)
geo_tri(const double *u, double *ut, const double *coef, const double *ccoef, const double *ecos, const double *esin,
        const uint *iter, const uint *ine, uint ne, uint ik, uint sub)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x, nw = NTH / 4;
    if (g >= ne * nw) return;
    uint t = g / nw, k0 = (g - t * nw) * 4;
    const double *c12 = coef + ((unsigned long long)ik * ne + t) * 12, *c6 = ccoef + 6 * t;
    uint r1 = ine[3 * t] * NTH, r2 = ine[3 * t + 1] * NTH, r3 = ine[3 * t + 2] * NTH;
    for (uint d = 0; d < 4; d++) {
        uint k = k0 + d;
        if (sub >= iter[ik * NTH + k]) continue;
        double c = ecos[k], s = esin[k];
        double k1 = c12[0] * c + c12[3] * s + c6[0], k2 = c12[1] * c + c12[4] * s + c6[1], k3 = c12[2] * c + c12[5] * s + c6[2];
        double kn = fmin(k1, 0.0) + fmin(k2, 0.0) + fmin(k3, 0.0);
        double nm = -1.0 / fmin(-fl(1.1754944e-38f), kn);
        double fa1 = c12[6] * c + c12[9] * s + c6[3] + fmax(k1, 0.0);
        double fa2 = c12[7] * c + c12[10] * s + c6[4] + fmax(k2, 0.0);
        double fa3 = c12[8] * c + c12[11] * s + c6[5] + fmax(k3, 0.0);
        ut[t * NTH + k] = nm * (fa1 * u[r1 + k] + fa2 * u[r2 + k] + fa3 * u[r3 + k]);
    }
}

#define GEO_NODE_ARGS                                                                                                 \
    const double *coef, const double *ccoef, const double *ecos, const double *esin, const double *dtk,              \
        const uint *iter, const double *si_inv, const uint *mlo, const uint *mhi, const uint *start, const uint *slot, \
        const uint *bidx, uint np, uint ne, uint ik, uint sub

// the node update of sub-step sub (as geo_node_x4); the boundary nodes take
// the boundary spectrum interpolated at the end of the sub-step
extern "C" __global__ void __launch_bounds__(GB)
geo_node(const double *u, double *out, const double *ut, GEO_NODE_ARGS, const double *bb0, const double *bbn,
         const double *clats, double rd10, double rd20, double dtg)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x, nw = NTH / 4;
    if (g >= np * nw) return;
    uint i = g / nw, k0 = (g - i * nw) * 4, base = i * NTH + k0;
    uint wlo = mlo[i], whi = mhi[i];
    double sa = si_inv[i] * (1.0 - (double)((whi >> 31) & 1u));
    uint bi = bidx[i];
    bool bnd = bi != 0xffffffffu;
    unsigned long long boff = (unsigned long long)(bnd ? bi : 0u) * (NK * NTH) + ik * NTH + k0;
    for (uint d = 0; d < 4; d++) {
        uint k = k0 + d, kk = ik * NTH + k;
        double c = ecos[k], s = esin[k], ui = u[base + d], st = 0.0;
        for (uint q = start[i]; q < start[i + 1]; q++) {
            uint code = slot[q], t = code / 3, p = code - 3 * t;
            const double *c12 = coef + ((unsigned long long)ik * ne + t) * 12;
            double kp = fmax(c12[p] * c + c12[3 + p] * s + ccoef[6 * t + p], 0.0);
            st += kp * (ui - ut[t * NTH + k]);
        }
        double m = (double)(((k < 32 ? wlo : whi) >> (k % 32)) & 1u);
        uint it = iter[kk];
        double v = fmax(ui - dtk[kk] * sa * st, 0.0) * m;
        double itf = (double)it;
        double rd1 = rd10 - dtg * (itf - (double)(sub + 1)) / itf;
        double r = fmin(1.0, fmax(0.0, rd1 / rd20));
        double rd2 = rd20 > fl(0.001f) ? r : 1.0;
        double bv = ((1.0 - rd2) * bb0[boff + d] + rd2 * bbn[boff + d]) * clats[i];
        out[base + d] = sub < it ? (bnd ? bv : v) : ui;
    }
}

// adjoint, first node pass: from the cotangent y of the node values after
// the sub-step, the direct part of the cotangent of u (ub) and that of the
// sums ST (stb)
extern "C" __global__ void __launch_bounds__(GB)
geo_adj_node1(const double *u, const double *ut, const double *y, double *ub, double *stb, GEO_NODE_ARGS)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x, nw = NTH / 4;
    if (g >= np * nw) return;
    uint i = g / nw, k0 = (g - i * nw) * 4, base = i * NTH + k0;
    uint wlo = mlo[i], whi = mhi[i];
    double sa = si_inv[i] * (1.0 - (double)((whi >> 31) & 1u));
    bool bnd = bidx[i] != 0xffffffffu;
    for (uint d = 0; d < 4; d++) {
        uint k = k0 + d, kk = ik * NTH + k;
        double c = ecos[k], s = esin[k], ui = u[base + d], st = 0.0, ks = 0.0;
        for (uint q = start[i]; q < start[i + 1]; q++) {
            uint code = slot[q], t = code / 3, p = code - 3 * t;
            const double *c12 = coef + ((unsigned long long)ik * ne + t) * 12;
            double kp = fmax(c12[p] * c + c12[3 + p] * s + ccoef[6 * t + p], 0.0);
            st += kp * (ui - ut[t * NTH + k]);
            ks += kp;
        }
        double dt = dtk[kk];
        double m = (double)(((k < 32 ? wlo : whi) >> (k % 32)) & 1u);
        bool active = sub < iter[kk];
        double x = ui - dt * sa * st, yv = y[base + d];
        double w = (active && !bnd && x > 0.0) ? yv * m : 0.0;
        double direct = active ? (bnd ? 0.0 : w) : yv;
        double sb = -dt * sa * w;
        stb[base + d] = sb;
        ub[base + d] = direct + sb * ks;
    }
}

// adjoint, triangle pass: the cotangent of the triangle values
extern "C" __global__ void __launch_bounds__(GB)
geo_adj_tri(const double *stb, double *utb, const double *coef, const double *ccoef, const double *ecos, const double *esin,
            const uint *ine, uint ne, uint ik)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x, nw = NTH / 4;
    if (g >= ne * nw) return;
    uint t = g / nw, k0 = (g - t * nw) * 4;
    const double *c12 = coef + ((unsigned long long)ik * ne + t) * 12, *c6 = ccoef + 6 * t;
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

// adjoint, second node pass: the cotangent of u before the sub-step
extern "C" __global__ void __launch_bounds__(GB)
geo_adj_node2(const double *ub, const double *utb, double *acb, const double *coef, const double *ccoef, const double *ecos,
              const double *esin, const uint *iter, const uint *start, const uint *slot, uint np, uint ne, uint ik, uint sub)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x, nw = NTH / 4;
    if (g >= np * nw) return;
    uint i = g / nw, k0 = (g - i * nw) * 4, base = i * NTH + k0;
    for (uint d = 0; d < 4; d++) {
        uint k = k0 + d;
        double c = ecos[k], s = esin[k], acc = ub[base + d];
        if (sub < iter[ik * NTH + k])
            for (uint q = start[i]; q < start[i + 1]; q++) {
                uint code = slot[q], t = code / 3, p = code - 3 * t;
                const double *c12 = coef + ((unsigned long long)ik * ne + t) * 12, *c6 = ccoef + 6 * t;
                double k1 = c12[0] * c + c12[3] * s + c6[0], k2 = c12[1] * c + c12[4] * s + c6[1], k3 = c12[2] * c + c12[5] * s + c6[2];
                double kn = fmin(k1, 0.0) + fmin(k2, 0.0) + fmin(k3, 0.0);
                double nm = -1.0 / fmin(-fl(1.1754944e-38f), kn);
                double kpp = fmax(c12[p] * c + c12[3 + p] * s + c6[p], 0.0);
                double fa = c12[6 + p] * c + c12[9 + p] * s + c6[3 + p] + kpp;
                acc += nm * fa * utb[t * NTH + k];
            }
        acb[base + d] = acc;
    }
}

// zero the cotangent of the listed nodes (the open boundary, the nodes the
// forcing overwrites)
extern "C" __global__ void __launch_bounds__(GB)
zero_nodes_adj(double *ab, const uint *nodes, uint np, uint nlist)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= nlist * NK * NTH) return;
    uint b = g / (NK * NTH), isp = g - b * (NK * NTH), ik = isp / NTH, th = isp - ik * NTH;
    ab[((unsigned long long)ik * np + nodes[b]) * NTH + th] = 0.0;
}

// ---------------------------------------------------------------------------
// The outputs H_s and T_m02 (hww3_wave::prop::outputs)
// ---------------------------------------------------------------------------

// adjoint of outputs_tl of prop_tl: from the cotangents hsb, t02b of H_s and
// T_m02 at each node (ignored where MAPSTA <= 0, flag 16 unset), adds the
// cotangent of the spectra to ab; one thread per node
extern "C" __global__ void __launch_bounds__(GB)
outputs_adj(const double *va, const double *cg, const double *sig, const double *dden, const uint *nflag,
            const double *hsb, const double *t02b, double *ab, double fte, double dth, uint np)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np || (nflag[i] & 16u) == 0) return;
    double et = 0.0, et02 = 0.0, e = 0.0, c = 1.0;
    for (uint k = 0; k < NK; k++) {
        e = 0.0;
        for (uint th = 0; th < NTH; th++) e += va[((unsigned long long)k * np + i) * NTH + th];
        c = cg[(k + 1) * np + i];
        double s = sig[k + 1];
        et += e * dden[k] / c;
        et02 += e * dden[k] / c * s * s;
    }
    et += fte * e / c;
    double sn = sig[NK], tail = 0.5 * sn * sn * sn * sn * dth;
    et02 += e / c * tail;
    bool ok = et02 > fl(1.0e-7f) && et > 0.0;
    double tm = fl(6.2831855f) * sqrt(et / fmax(et02, fl(1.0e-30f)));
    double detb = (et > 0.0 ? 2.0 * hsb[i] / sqrt(et) : 0.0) + (ok ? 0.5 * tm * t02b[i] / et : 0.0);
    double det02b = ok ? -0.5 * tm * t02b[i] / et02 : 0.0;
    for (uint k = 0; k < NK; k++) {
        double ck = cg[(k + 1) * np + i], s = sig[k + 1];
        double g = detb * dden[k] / ck + det02b * dden[k] / ck * s * s;
        if (k == NK - 1) g += detb * fte / ck + det02b * tail / ck;
        for (uint th = 0; th < NTH; th++) ab[((unsigned long long)k * np + i) * NTH + th] += g;
    }
}

// ---------------------------------------------------------------------------
// The level remap of W3ULEV (hww3_wave::prop::ulev_remap, ulev_node)
// ---------------------------------------------------------------------------

// the transpose of one column of the remap: b holds the cotangent of the
// remapped column (NK values), replaced by that of the column before it
__device__ void ulev_col_adj(double *b, const uint *i1, const double *r1, uint t0, const double *own, const double *dwn,
                             double fachfa)
{
    double tb[NK];
    for (int k = NK - 1; k >= 1; k--)
        if ((uint)k + 1 >= t0) { b[k - 1] += fachfa * b[k]; b[k] = 0.0; }
    for (uint k = 0; k < NK; k++) b[k] = b[k] / dwn[k];
    for (uint k = 0; k < NK; k++) {
        double g = 0.0;
        uint j1 = i1[k];
        if (j1 <= NK) {
            double w1 = r1[k];
            if (j1 >= 1) g += w1 * b[j1 - 1];
            if (j1 < NK) g += (1.0 - w1) * b[j1];
        }
        tb[k] = g;
    }
    for (uint k = 0; k < NK; k++) b[k] = tb[k] * own[k];
}

// adjoint of ulev_remap (weights from the host): one thread per (listed node,
// direction), on the cotangent ab of the spectra
extern "C" __global__ void __launch_bounds__(GB)
ulev_remap_adj(double *ab, const uint *nodes, const double *own, const double *dwn, const uint *i1, const double *r1,
               const uint *tail, double fachfa, uint np, uint nlist)
{
    uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= nlist * NTH) return;
    uint l = g / NTH, th = g - l * NTH, i = nodes[l];
    double b[NK];
    for (uint k = 0; k < NK; k++) b[k] = ab[((unsigned long long)k * np + i) * NTH + th];
    ulev_col_adj(b, i1 + l * NK, r1 + l * NK, tail[l], own + l * NK, dwn + l * NK, fachfa);
    for (uint k = 0; k < NK; k++) ab[((unsigned long long)k * np + i) * NTH + th] = b[k];
}

// adjoint of ulev_node (weights on the device): wno, cgo the wavenumbers and
// group speeds before the level change, wnn, cgn after it, mode the node
// modes before it (2: remapped when the first bin moved by 5 % of its width)
extern "C" __global__ void __launch_bounds__(GB)
ulev_node_adj(double *ab, const double *wno, const double *cgo, const double *wnn, const double *cgn, const uint *mode,
              const double *dsip, double fachfa, uint np)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np || mode[i] != 2u) return;
    double wo[NK + 2], wnw[NK + 2];
    for (uint j = 0; j < NK + 2; j++) { wo[j] = wno[j * np + i]; wnw[j] = wnn[j * np + i]; }
    double dwn1 = dsip[1] / cgn[np + i];
    double rdk = fabs(wo[1] - wnw[1]) / dwn1;
    if (rdk < fl(0.05f)) return;
    uint wi1[NK];
    double wr1[NK], own[NK], dwn[NK];
    for (uint k = 0; k < NK; k++) { wi1[k] = NK + 1; wr1[k] = 0.0; }
    uint ik0 = 1, ii1 = 1, ii2 = 2;
    bool cond = true;
    if (wo[1] < wnw[1]) {
        ii1 = 0;
        ii2 = 1;
        bool found = false;
        for (uint j = 2; j < NK + 2; j++)
            if (!found && wo[j] >= wnw[1]) { found = true; ik0 = j - 1; }
        cond = found;
    }
    double rd2 = 1.0;
    if (cond) {
        for (uint ik = ik0; ik < NK + 1; ik++) {
            while (ii1 <= NK && wo[ik] > wnw[ii2]) {
                ii1 += 1;
                if (ii1 <= NK) ii2 = ii1 + 1;
            }
            if (ii1 <= NK) {
                double rd1 = ii1 == 0 ? (wnw[1] - wo[ik]) / dwn1 : (wnw[ii2] - wo[ik]) / (wnw[ii2] - wnw[ii1]);
                rd2 = 1.0 - rd1;
                wi1[ik - 1] = ii1;
                wr1[ik - 1] = rd1;
            }
        }
    } else ii2 = NK + 1;
    uint t0 = ii2 > NK + 1 ? ii2 : NK + 1;
    if (ii2 <= NK && rd2 <= fl(0.95f)) t0 = ii2 > 2 ? ii2 : 2;
    for (uint k = 0; k < NK; k++) {
        own[k] = dsip[k + 1] / cgo[(k + 1) * np + i];
        dwn[k] = dsip[k + 1] / cgn[(k + 1) * np + i];
    }
    double b[NK];
    for (uint th = 0; th < NTH; th++) {
        for (uint k = 0; k < NK; k++) b[k] = ab[((unsigned long long)k * np + i) * NTH + th];
        ulev_col_adj(b, wi1, wr1, t0, own, dwn, fachfa);
        for (uint k = 0; k < NK; k++) ab[((unsigned long long)k * np + i) * NTH + th] = b[k];
    }
}
