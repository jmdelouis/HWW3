// W3SRCE in CUDA C (milestone M4): the value of the source terms of the
// CubeCL kernel `srce` (hww3_wave::srce), transcribed in double precision for
// one block of NU threads per node, so that the adjoint (w3srce_adj.cuh) can
// recompute any part of the forward pass. Only the configurations of the M3
// tangent are written: SINTABLE = 1 (tail stress from the tables), with or
// without sheltering, the tables in BETAMAX (P_BT_N > 1) or not, SDSBCHOICE
// 1 (with the cumulative term) or 3 (Romero). The parameter slots P_* come
// from the Rust constants as macros (cuda_adj.rs); NK, NTH, NSPECX, NSPECY,
// NSAT and NU are macros too.

#define NSPEC (NK * NTH)
#define NUE (NSPECY + NTH)
#define NB1 (NSPECX + NTH)
#define RS ((NSPEC + NU - 1) / NU)

typedef unsigned int uint;
typedef unsigned char uchar;

// scalar slots of sc
#define S_EMEAN 0
#define S_FMEAN 1
#define S_FMEAN1 2
#define S_WNMEAN 3
#define S_AMAX 4
#define S_USTAR 5
#define S_Z0 6
#define S_CHARN 7
#define S_FMEANWS 8
#define S_TAUWX 9
#define S_TAUWY 10
#define S_FHIGH 11
#define S_DTTOT 12
#define S_DT 13
#define S_SHAVE 14
#define S_DBD 15
#define S_COSU 17
#define S_SINU 18
#define S_USTP 19
#define S_UORB 20
#define S_FW 21
#define S_FU 22
#define S_FUD 23
#define S_PTURB 24
#define S_PVISC 25
#define S_SX 26
#define S_SY 27
#define S_SNX 28
#define S_SNY 29
#define S_CONS 30
#define S_NKH 31
#define S_FBT 32
#define S_DTDYN 33
#define S_NSTEPS 34
#define S_DONE 35
#define S_TAUX 36
#define S_TAUY 37
#define S_SCALE 38
#define NSC 40

// per-frequency slots of kb (kb[slot NK + ik])
#define K_WN 0
#define K_CG 1
#define K_DAM 2
#define K_EB 3
#define K_EB2 4
#define K_EBX 5
#define K_EBY 6
#define K_CBETA 7
#define K_BTH0 8
#define K_AMAX 9
#define K_SPW 10
#define K_WNP 11
#define K_DCG 12
#define NKB 13

// per-frequency values of Romero (in b2, R_* NK + ik; DLWMEAN at R_N NK)
#define R_MSS 0
#define R_MSSD 1
#define R_MC2 2
#define R_MS2 3
#define R_ELC 4
#define R_ELS 5
#define R_BTH0 6
#define R_SL 7
#define R_ISO 8
#define R_N 9

#define TPIINV ((double)0.15915494f)
#define TPI ((double)6.2831855f)
#define THR38 ((double)1.1754944e-38f)

__device__ __forceinline__ double flt(float x) { return (double)x; }
__device__ __forceinline__ double log10_ww3(double x) { return log(x) * flt(0.4342945f); }

// interpolation in a table on a uniform axis as WW3 (index MIN(SIZE - 1, INT(x)))
__device__ double table1(const double *tab, uint n, double x)
{
    uint size = n - 1;
    uint xi = (uint)fmax(trunc(x), 0.0);
    uint ind = xi < size - 1 ? xi : size - 1;
    double deli1 = fmin(1.0, x - (double)ind);
    return tab[ind] * (1.0 - deli1) + tab[ind + 1] * deli1;
}

// CALC_USTAR with SINTABLE = 1: u* from TAUT, then z0 and Charnock
__device__ void calc_ustar(const double *par, const double *taut, double *sc, double windspeed, double tauw)
{
    const double grav = par[P_GRAV], kappa = par[P_KAPPA];
    double chath = par[P_AALPHA];
    const bool capped = par[P_CAPCHNK] == 1.0;
    uint itm = (uint)par[P_ITAUMAX], jum = (uint)par[P_JUMAX];
    double tl = fmax(fmin(tauw, par[P_TAUWMAX]), 0.0);
    double xi = sqrt(tl) / par[P_DELTAUW];
    uint xif = (uint)trunc(xi);
    uint ind = xif < itm - 1 ? xif : itm - 1;
    double deli1 = fmin(1.0, xi - (double)ind), deli2 = 1.0 - deli1;
    double xj = windspeed / par[P_DELU];
    uint xjf = (uint)trunc(xj);
    uint j = xjf < jum - 1 ? xjf : jum - 1;
    double delj1 = fmin(1.0, xj - (double)j), delj2 = 1.0 - delj1;
    uint n = itm + 1;
    double ust = (taut[ind + j * n] * deli2 + taut[ind + 1 + j * n] * deli1) * delj2 +
                 (taut[ind + (j + 1) * n] * deli2 + taut[ind + 1 + (j + 1) * n] * deli1) * delj1;
    double z0 = chath * flt(0.000001f) / grav, charn = chath;
    if (ust > flt(0.001f)) {
        double sqrtcdm1 = fmin(windspeed / ust, 100.0);
        z0 = par[P_ZZWND] * exp(-kappa * sqrtcdm1);
        charn = grav * z0 / (ust * ust);
    } else if (ust > 0.0) {
        z0 = par[P_ZZWND] * exp(-kappa * fmin(windspeed / ust, 100.0));
    }
    if (capped) charn = fmax(fmin(charn, flt(0.09f)), chath);
    sc[S_USTAR] = ust;
    sc[S_Z0] = z0;
    sc[S_CHARN] = charn;
}

// W3SPR4: mean parameters of a with the wind-sea flags ws, then u*, z0
__device__ void spr4(const double *par, const double *taut, const double *sig, const double *dden, const double *a,
                     const uchar *ws, double *kb, double *sc, double u, double tauwx, double tauwy)
{
    const uint t = threadIdx.x;
    if (t < NK) {
        double eb = 0.0, eb2 = 0.0, amax = 0.0;
        for (uint th = 0; th < NTH; th++) {
            double x = a[t * NTH + th];
            eb += x;
            if (ws[t * NTH + th] != 0) eb2 += x;
            amax = fmax(amax, x);
        }
        kb[K_EB * NK + t] = eb;
        kb[K_EB2 * NK + t] = eb2;
        kb[K_AMAX * NK + t] = amax;
    }
    __syncthreads();
    if (t == 0) {
        const double tw = par[P_WWNMEANPTAIL], wp = par[P_WWNMEANP];
        double emean = 0.0, fmean = 0.0, fmean1 = 0.0, wnmean = 0.0, emeanws = 0.0, fmeanws = 0.0, amax = 0.0;
        for (uint ik = 0; ik < NK; ik++) {
            double f = kb[K_DCG * NK + ik];
            double eb = kb[K_EB * NK + ik] * f, eb2 = kb[K_EB2 * NK + ik] * f;
            double s = sig[ik + 1], sp = kb[K_SPW * NK + ik];
            emean += eb;
            fmean += eb / s;
            fmean1 += eb * sp;
            wnmean += eb * kb[K_WNP * NK + ik];
            emeanws += eb2;
            fmeanws += eb2 * sp;
            amax = fmax(amax, kb[K_AMAX * NK + ik]);
        }
        double ddl = dden[NK - 1], cgl = kb[K_CG * NK + NK - 1];
        double eband = kb[K_EB * NK + NK - 1] * (ddl / cgl) / ddl;
        emean += eband * par[P_FTE];
        fmean += eband * par[P_FTF];
        fmean1 += eband * par[P_SSTXFTFTAIL];
        wnmean += eband * par[P_SSTXFTWN];
        double eband2 = kb[K_EB2 * NK + NK - 1] * (ddl / cgl) / ddl;
        emeanws += eband2 * par[P_FTE];
        fmeanws += eband2 * par[P_SSTXFTFTAIL];
        const double e7 = flt(1.0e-7f);
        fmean = TPIINV * emean / fmax(e7, fmean);
        if (fmean1 < e7) fmean1 = TPIINV * sig[NK];
        else fmean1 = TPIINV * pow(fmax(e7, fmean1) / fmax(e7, emean), 1.0 / (2.0 * tw));
        wnmean = pow(fmax(e7, wnmean) / fmax(e7, emean), 1.0 / wp);
        if (fmeanws < e7 || emeanws < e7) fmeanws = TPIINV * sig[NK];
        else fmeanws = TPIINV * pow(fmax(e7, fmeanws) / fmax(e7, emeanws), 1.0 / (2.0 * tw));
        sc[S_EMEAN] = emean;
        sc[S_FMEAN] = fmean;
        sc[S_FMEAN1] = fmean1;
        sc[S_WNMEAN] = wnmean;
        sc[S_AMAX] = amax;
        sc[S_FMEANWS] = fmeanws;
        double tauw = sqrt(tauwx * tauwx + tauwy * tauwy);
        calc_ustar(par, taut, sc, u, tauw);
    }
    __syncthreads();
}

// BETAMAX(U): hat function k of the nk knots, its derivative, the log factor
__device__ double betau_phi(const double *par, double u, uint k, uint nk)
{
    double uc = fmax(par[P_BU_KNOT], fmin(par[P_BU_KNOT + nk - 1], u));
    double xk = par[P_BU_KNOT + k], v = 1.0;
    if (uc >= xk) {
        if (k + 1 < nk) { double xr = par[P_BU_KNOT + k + 1]; v = fmax(0.0, (xr - uc) / (xr - xk)); }
    } else if (k > 0) {
        double xl = par[P_BU_KNOT + k - 1];
        v = fmax(0.0, (uc - xl) / (xk - xl));
    }
    return v;
}

__device__ double betau_dphi(const double *par, double u, uint k, uint nk)
{
    double v = 0.0;
    if (u > par[P_BU_KNOT] && u < par[P_BU_KNOT + nk - 1]) {
        double xk = par[P_BU_KNOT + k];
        if (u >= xk) {
            if (k + 1 < nk) { double xr = par[P_BU_KNOT + k + 1]; v = u < xr ? -1.0 / (xr - xk) : 0.0; }
        } else if (k > 0) {
            double xl = par[P_BU_KNOT + k - 1];
            v = u > xl ? 1.0 / (xk - xl) : 0.0;
        }
    }
    return v;
}

__device__ double betau_ln(const double *par, double u)
{
    uint nk = (uint)par[P_BU_NK];
    double s = 0.0;
    for (uint k = 0; k < nk; k++) s += par[P_BU_COEF + k] * betau_phi(par, u, k, nk);
    return s;
}

// bilinear interpolation in a table of the tail stress starting at base
__device__ double tauhf2(const double *tab, uint base, uint ind, uint j, uint n1, double deli1, double deli2, double delj1, double delj2)
{
    return (tab[ind + j * n1 + base] * deli2 + tab[ind + 1 + j * n1 + base] * deli1) * delj2 +
           (tab[ind + (j + 1) * n1 + base] * deli2 + tab[ind + 1 + (j + 1) * n1 + base] * deli1) * delj1;
}

// the interpolation indices and weights of the tail stress (sin4's end block)
struct TailIdx {
    uint ind, j, n1, n2, i, nbt, m, stride, nb;
    double deli1, deli2, delj1, delj2, delk1, delk2;
    double hw[4];
};

__device__ void tail_index(const double *par, double ust, double z0, double levtail0, double beta, TailIdx &x)
{
    const double grav = par[P_GRAV];
    uint ius = (uint)par[P_IUSTAR], ial = (uint)par[P_IALPHA];
    double xi = ust / par[P_DELUST];
    uint xif = (uint)trunc(xi);
    x.ind = xif < 1 ? 1 : (xif < ius - 1 ? xif : ius - 1);
    x.deli1 = fmax(fmin(1.0, xi - (double)x.ind), 0.0);
    x.deli2 = 1.0 - x.deli1;
    double um = fmax(ust, flt(0.00001f));
    double xj = fmax(0.0, (grav * z0 / (um * um) - par[P_AALPHA]) / par[P_DELALP]);
    uint xjf = (uint)trunc(xj);
    x.j = xjf < 1 ? 1 : (xjf < ial - 1 ? xjf : ial - 1);
    x.delj1 = fmax(0.0, fmin(1.0, xj - (double)x.j));
    x.delj2 = 1.0 - x.delj1;
    x.n1 = ius + 1;
    x.n2 = ial + 1;
    x.nbt = (uint)par[P_BT_N];
    x.stride = (uint)par[P_BT_STRIDE];
    x.nb = x.nbt;
    x.m = 0;
    if (x.nbt > 1) {
        double db = par[P_BT_DB];
        double xb = fmax(0.0, fmin((double)(x.nbt - 1), (beta - par[P_BT_B0]) / db));
        uint mm = (uint)trunc(xb);
        x.m = mm < x.nbt - 2 ? mm : x.nbt - 2;
        double tt = xb - (double)x.m, t2 = tt * tt, t3 = t2 * tt;
        x.hw[0] = 2.0 * t3 - 3.0 * t2 + 1.0;
        x.hw[1] = (t3 - 2.0 * t2 + tt) * db;
        x.hw[2] = 3.0 * t2 - 2.0 * t3;
        x.hw[3] = (t3 - t2) * db;
    }
    x.i = 0;
    x.delk1 = 0.0;
    x.delk2 = 1.0;
    if (par[P_TTAUWSHELTER] > 0.0) {
        uint ilv = (uint)par[P_ILEVTAIL];
        double xk = levtail0 / par[P_DELTAIL];
        uint xkf = (uint)trunc(xk);
        x.i = xkf < ilv - 1 ? xkf : ilv - 1;
        x.delk1 = fmin(1.0, xk - (double)x.i);
        x.delk2 = 1.0 - x.delk1;
    }
}

// the interpolated tail stress TAU1 (the arithmetic of sin4)
__device__ double tail_tau1(const double *par, const double *tauhf, const TailIdx &x)
{
    uint o0 = x.i * x.n1 * x.n2, o1 = (x.i + 1) * x.n1 * x.n2;
    if (x.nbt > 1) {
        uint mb[4] = {x.m * x.stride, (x.nb + x.m) * x.stride, (x.m + 1) * x.stride, (x.nb + x.m + 1) * x.stride};
        double tau1 = 0.0;
        for (int c = 0; c < 4; c++) {
            double v;
            if (par[P_TTAUWSHELTER] > 0.0)
                v = tauhf2(tauhf, mb[c] + o0, x.ind, x.j, x.n1, x.deli1, x.deli2, x.delj1, x.delj2) * x.delk2 +
                    tauhf2(tauhf, mb[c] + o1, x.ind, x.j, x.n1, x.deli1, x.deli2, x.delj1, x.delj2) * x.delk1;
            else
                v = tauhf2(tauhf, mb[c], x.ind, x.j, x.n1, x.deli1, x.deli2, x.delj1, x.delj2);
            tau1 = c == 0 ? x.hw[0] * v : tau1 + x.hw[c] * v;
        }
        return tau1;
    }
    if (par[P_TTAUWSHELTER] > 0.0)
        return tauhf2(tauhf, o0, x.ind, x.j, x.n1, x.deli1, x.deli2, x.delj1, x.delj2) * x.delk2 +
               tauhf2(tauhf, o1, x.ind, x.j, x.n1, x.deli1, x.deli2, x.delj1, x.delj2) * x.delk1;
    return tauhf2(tauhf, 0, x.ind, x.j, x.n1, x.deli1, x.deli2, x.delj1, x.delj2);
}

// W3SIN4 (SINTABLE = 1): wind input with sheltering and swell damping; the
// wind-sea flags ws, the wave-supported stress in sc; with acc, scale S and
// scale D added to vs, vd. The per-frequency (TAUPX, TAUPY) at the start of
// each frequency of the loop are written to tp[2 ik], tp[2 ik + 1] when tp
// is given (for the adjoint).
__device__ void sin4(const double *par, const double *tauhf, const double *sig, const double *dden, const double *ecos,
                     const double *esin, const double *sig2k, const double *dden2k, const double *fwtab, const double *a,
                     double *vs, double *vd, uchar *ws, double *kb, double *sc, double *dr, double ustar, double z0,
                     double usdir, double beta, double scale, bool acc, double *tp)
{
    const uint t = threadIdx.x;
    const double grav = par[P_GRAV], kappa = par[P_KAPPA], nu_air = par[P_NU_AIR], drat = par[P_DRAT];
    const double const1 = beta / (kappa * kappa), const0 = const1 * drat;
    const double tsh = fabs(par[P_TTAUWSHELTER]);
    if (t < NK) {
        double eb = 0.0;
        for (uint th = 0; th < NTH; th++) eb += a[t * NTH + th];
        kb[K_EB * NK + t] = eb;
    }
    __syncthreads();
    if (t == 0) {
        double uorb = 0.0, aorb = 0.0;
        for (uint ik = 0; ik < NK; ik++) {
            double eb = kb[K_EB * NK + ik], s = sig[ik + 1], c = kb[K_CG * NK + ik];
            uorb += eb * s * s * dden[ik] / c;
            aorb += eb * dden[ik] / c;
        }
        uorb = 2.0 * sqrt(uorb);
        double aorb1 = 2.0 * pow(aorb, 1.0 - 0.5 * par[P_SSWELLF + 5]);
        double re = 4.0 * uorb * aorb1 / nu_air;
        double pturb = 1.0, pvisc = 1.0;
        if (par[P_SSWELLF + 3] > 0.0) {
            if (par[P_SSWELLF + 6] > 0.0) {
                double smooth = 0.5 * tanh((re - par[P_SSWELLF + 3]) / par[P_SSWELLF + 6]);
                pturb = 0.5 + smooth;
                pvisc = 0.5 - smooth;
            } else if (re <= par[P_SSWELLF + 3]) { pturb = 0.0; pvisc = 1.0; }
            else { pturb = 1.0; pvisc = 0.0; }
        }
        double fw = fmax(fabs(par[P_SSWELLF + 2]), 0.0), fu = 0.0, fud = 0.0;
        if (par[P_SSWELLF + 1] != 0.0) {
            fu = fabs(par[P_SSWELLF + 2]);
            fud = par[P_SSWELLF + 1];
            double z0visc = flt(0.1f) * nu_air / fmax(ustar, flt(0.0001f));
            double z0noz = fmax(z0visc, par[P_ZZ0RAT] * z0);
            double aorb2 = 2.0 * sqrt(aorb);
            double xi = (log10_ww3(fmax(aorb2 / z0noz, 3.0)) - par[P_ABMIN]) / par[P_DELAB];
            fw = table1(fwtab, (uint)par[P_NFW], xi);
        }
        sc[S_UORB] = uorb;
        sc[S_FW] = fw;
        sc[S_FU] = fu;
        sc[S_FUD] = fud;
        sc[S_PTURB] = pturb;
        sc[S_PVISC] = pvisc;
        sc[S_TAUX] = ustar * ustar * cos(usdir);
        sc[S_TAUY] = ustar * ustar * sin(usdir);
        sc[S_SX] = sc[S_SY] = sc[S_SNX] = sc[S_SNY] = 0.0;
    }
    __syncthreads();
    const double uorb = sc[S_UORB], fw = sc[S_FW], fu = sc[S_FU], fud = sc[S_FUD], pturb = sc[S_PTURB], pvisc = sc[S_PVISC];
    const double sw0 = par[P_SSWELLF], sw2 = par[P_SSWELLF + 2], sw4 = par[P_SSWELLF + 4];
    for (uint ik = 0; ik < NK; ik++) {
        if (t == 0) {
            double taupx = sc[S_TAUX] - tsh * sc[S_SX], taupy = sc[S_TAUY] - tsh * sc[S_SY];
            if (tp) { tp[2 * ik] = taupx; tp[2 * ik + 1] = taupy; }
            double r = sqrt(taupx * taupx + taupy * taupy);
            sc[S_USTP] = fmin(sqrt(r), fmax(ustar, flt(0.3f)));
            sc[S_COSU] = r > 0.0 ? taupx / r : 1.0;
            sc[S_SINU] = r > 0.0 ? taupy / r : 0.0;
        }
        __syncthreads();
        if (t < NTH) {
            double ustp = sc[S_USTP], cosu = sc[S_COSU], sinu = sc[S_SINU];
            double k = kb[K_WN * NK + ik], c = kb[K_CG * NK + ik], s2 = sig2k[ik];
            double cm = k / s2, ucn = ustp * cm + par[P_ZZALP], s = sig[ik + 1];
            double const2 = dden2k[ik] / c * grav / (s / k * drat);
            double cnst = s2 * const0, zcn = log(k * z0);
            double swellcoefv = -sw4 * drat * 2.0 * k * sqrt(2.0 * nu_air * s2);
            double swellcoeft = -drat * sw0 * 16.0 * s2 * s2 / grav;
            uint is = ik * NTH + t;
            double coswind = ecos[t] * cosu + esin[t] * sinu;
            double dst = 0.0, llws = 0.0;
            if (coswind > flt(0.01f)) {
                double x = coswind * ucn, zarg = kappa / x, zlog = zcn + zarg;
                if (zlog < 0.0) {
                    double z2 = zlog * zlog;
                    dst = cnst * exp(zlog) * (z2 * z2) * ucn * ucn * pow(coswind, par[P_SSINTHP]);
                    llws = 1.0;
                }
                if (28.0 * cm * ustar * coswind >= 1.0) llws = 1.0;
            }
            if ((sw0 != 0.0 && dst < flt(1.0e-7f) * s2) || sw2 > 0.0) {
                double dturb = swellcoeft * (fw * uorb + (fu + fud * coswind) * ustp);
                dst += pturb * dturb + pvisc * swellcoefv;
            }
            double x = a[is], temp2 = const2 * dst * x;
            dr[t] = temp2 * ecos[t];
            dr[NTH + t] = temp2 * esin[t];
            dr[2 * NTH + t] = dst < 0.0 ? 1.0 : 0.0;
            ws[is] = llws > 0.0 ? 1 : 0;
            if (acc) { vs[is] += scale * (dst * x); vd[is] += scale * dst; }
        }
        __syncthreads();
        if (t == 0) {
            double sx = sc[S_SX], sy = sc[S_SY], snx = sc[S_SNX], sny = sc[S_SNY];
            for (uint th = 0; th < NTH; th++) {
                if (dr[2 * NTH + th] > 0.0) { snx += dr[th]; sny += dr[NTH + th]; }
                else { sx += dr[th]; sy += dr[NTH + th]; }
            }
            sc[S_SX] = sx; sc[S_SY] = sy; sc[S_SNX] = snx; sc[S_SNY] = sny;
        }
        __syncthreads();
    }
    if (t == 0) {
        double sx = sc[S_SX], sy = sc[S_SY];
        double taupx = sc[S_TAUX] - tsh * sx, taupy = sc[S_TAUY] - tsh * sy;
        double ust = pow(taupx * taupx + taupy * taupy, flt(0.25f));
        double usdirp = atan2(taupy, taupx);
        double cosu = sc[S_COSU], sinu = sc[S_SINU];
        double snk = sig[NK], s5 = snk * snk * snk * snk * snk;
        double c0 = par[P_DTH] * s5 / (grav * grav * TPI) * TPI * snk / kb[K_CG * NK + NK - 1];
        double temp = 0.0;
        for (uint th = 0; th < NTH; th++) {
            double cw = fmax(ecos[th] * cosu + esin[th] * sinu, 0.0);
            temp += a[(NK - 1) * NTH + th] * cw * cw * cw;
        }
        double levtail0 = c0 * temp;
        TailIdx ix;
        tail_index(par, ust, z0, levtail0, beta, ix);
        double tau1 = tail_tau1(par, tauhf, ix);
        double tauhf_v = levtail0 * ust * ust * tau1;
        double tauwx = sx + tauhf_v * cos(usdirp), tauwy = sy + tauhf_v * sin(usdirp);
        double tauw = sqrt(tauwx * tauwx + tauwy * tauwy);
        const double eps2 = flt(0.000001f);
        double um = fmax(ustar, eps2), ust2 = um * um;
        double tauwb = fmin(tauw, fmax(ust2 - flt(0.00001f), eps2 * eps2));
        if (tauwb < tauw) { tauwx *= tauwb / tauw; tauwy *= tauwb / tauw; }
        sc[S_TAUWX] = tauwx;
        sc[S_TAUWY] = tauwy;
    }
    __syncthreads();
}

// W3SLN1: linear wind input, scale S added to vs
__device__ void sln1(const double *par, const double *sig, const double *ecos, const double *esin, const double *kb,
                     double *vs, double fhigh, double ustar, double usdir, double scale)
{
    const uint t = threadIdx.x;
    double cosu = cos(usdir), sinu = sin(usdir), u2 = ustar * ustar;
    double fac = par[P_SLNC1] * (u2 * u2);
    double ff1 = par[P_FSPM] * par[P_GRAV] / (28.0 * ustar);
    double ff2 = par[P_FSHF] * fmin(sig[NK], fhigh);
    double ffilt = fmin(fmax(ff1, ff2), 2.0 * sig[NK]);
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is >= NSPEC) continue;
        uint ik = is / NTH, th = is - ik * NTH;
        double rfr = sig[ik + 1] / ffilt, wnf = 0.0;
        if (rfr >= 0.5) { double r2 = rfr * rfr; wnf = fac / kb[K_WN * NK + ik] * exp(-(1.0 / (r2 * r2))); }
        double d = fmax(ecos[th] * cosu + esin[th] * sinu, 0.0), d2 = d * d;
        vs[is] += scale * (wnf * (d2 * d2));
    }
}

// W3SNL1 (DIA) for the mean relative depth kdmean: scale S, scale D added
__device__ void snl1(const double *par, const double *sig, const uint *dia_ip, const uint *dia_ic, const double *af11,
                     const double *a, const double *kb, double *ue, double *b1, double *b2, double *dn, double *vs,
                     double *vd, double kdmean, double scale)
{
    const uint t = threadIdx.x;
    double x = fmax(par[P_KDCON] * kdmean, par[P_KDMN]);
    double x2 = fmax(-1.0e15, par[P_SNLS3] * x);
    double cons = par[P_SNLC1] * (1.0 + par[P_SNLS1] / x * (1.0 - par[P_SNLS2] * x) * exp(x2));
    const double dal1 = par[P_DAL1], dal2 = par[P_DAL2], dal3 = par[P_DAL3];
    for (uint r = 0; r < (NUE + NU - 1) / NU; r++) {
        uint u = r * NU + t;
        if (u >= NUE) continue;
        double v = 0.0;
        if (u >= NTH && u < NSPEC + NTH) {
            uint is = u - NTH, ik = is / NTH;
            double conx = TPIINV / sig[ik + 1] * kb[K_CG * NK + ik];
            v = a[is] / conx;
        }
        ue[u] = v;
    }
    if (t < NTH) { b1[t] = 0.0; b2[t] = 0.0; }
    __syncthreads();
    if (t < NTH)
        for (uint ifr = NK; ifr < NSPECY / NTH; ifr++) ue[NTH + ifr * NTH + t] = ue[ifr * NTH + t] * par[P_FACHFE];
    __syncthreads();
    const double w0 = par[P_AWG], w1 = par[P_AWG + 1], w2 = par[P_AWG + 2], w3 = par[P_AWG + 3];
    const double w4 = par[P_AWG + 4], w5 = par[P_AWG + 5], w6 = par[P_AWG + 6], w7 = par[P_AWG + 7];
    for (uint pass = 0; pass < 3; pass++) {
        for (uint r = 0; r < (NSPECX + NU - 1) / NU; r++) {
            uint isp = r * NU + t;
            if (isp >= NSPECX) continue;
            double e00 = ue[isp + NTH];
            double ep1 = w0 * ue[dia_ip[isp]] + w1 * ue[dia_ip[NSPECX + isp]] + w2 * ue[dia_ip[2 * NSPECX + isp]] + w3 * ue[dia_ip[3 * NSPECX + isp]];
            double em1 = w4 * ue[dia_ip[4 * NSPECX + isp]] + w5 * ue[dia_ip[5 * NSPECX + isp]] + w6 * ue[dia_ip[6 * NSPECX + isp]] + w7 * ue[dia_ip[7 * NSPECX + isp]];
            double ep2 = w0 * ue[dia_ip[8 * NSPECX + isp]] + w1 * ue[dia_ip[9 * NSPECX + isp]] + w2 * ue[dia_ip[10 * NSPECX + isp]] + w3 * ue[dia_ip[11 * NSPECX + isp]];
            double em2 = w4 * ue[dia_ip[12 * NSPECX + isp]] + w5 * ue[dia_ip[13 * NSPECX + isp]] + w6 * ue[dia_ip[14 * NSPECX + isp]] + w7 * ue[dia_ip[15 * NSPECX + isp]];
            double caf = cons * af11[isp], factor = caf * e00;
            double sa1a = e00 * (ep1 * dal1 + em1 * dal2), sa1b = sa1a - ep1 * em1 * dal3;
            double sa2a = e00 * (ep2 * dal1 + em2 * dal2), sa2b = sa2a - ep2 * em2 * dal3;
            if (pass == 0) { b1[isp + NTH] = factor * sa1b; b2[isp + NTH] = factor * sa2b; }
            else if (pass == 1) {
                b1[isp + NTH] = factor * (dal1 * e00 - dal3 * em1);
                b2[isp + NTH] = factor * (dal1 * e00 - dal3 * em2);
                if (isp < NSPEC) dn[isp] = -2.0 * (caf * (sa1a + sa1b) + caf * (sa2a + sa2b));
            } else {
                b1[isp + NTH] = factor * (dal2 * e00 - dal3 * ep1);
                b2[isp + NTH] = factor * (dal2 * e00 - dal3 * ep2);
            }
        }
        __syncthreads();
        for (uint r = 0; r < RS; r++) {
            uint is = r * NU + t;
            if (is >= NSPEC) continue;
            if (pass == 0) {
                uint ik = is / NTH;
                double con = TPIINV / sig[ik + 1] * kb[K_CG * NK + ik];
                double ss = -2.0 * (b1[is + NTH] + b2[is + NTH]);
                for (uint j = 0; j < 8; j++) ss += par[P_AWG + j] * (b1[dia_ic[2 * j * NSPEC + is]] + b2[dia_ic[(2 * j + 1) * NSPEC + is]]);
                vs[is] += scale * (con * ss);
            } else if (pass == 1) {
                double d = dn[is];
                for (uint j = 0; j < 4; j++) d += par[P_SWG + j] * (b1[dia_ic[2 * j * NSPEC + is]] + b2[dia_ic[(2 * j + 1) * NSPEC + is]]);
                dn[is] = d;
            } else {
                double d = dn[is];
                for (uint j = 4; j < 8; j++) d += par[P_SWG + j] * (b1[dia_ic[2 * j * NSPEC + is]] + b2[dia_ic[(2 * j + 1) * NSPEC + is]]);
                vd[is] += scale * d;
            }
        }
        __syncthreads();
    }
}

// W3SDS4 with SDSBCHOICE = 1, the saturation tables and the cumulative term
__device__ void sds4(const double *par, const double *sig, const double *ecos, const double *esin, const double *satw,
                     const uint *sati, const double *cumw, const double *a, double *kb, double *b1, double *dn, double *vs,
                     double *vd, double depth, double ustar, double usdir, double scale)
{
    const uint t = threadIdx.x;
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is >= NSPEC) continue;
        uint ik = is / NTH, th = is - ik * NTH;
        double k = kb[K_WN * NK + ik];
        double facsat = sig[ik + 1] * (k * k * k) * par[P_DTH];
        double dot = 0.0;
        for (uint j = 0; j < NSAT; j++) dot += satw[j + th * NSAT] * a[ik * NTH + sati[j + th * NSAT]];
        b1[is] = dot * facsat;
    }
    __syncthreads();
    if (t < NK) {
        double m = b1[t * NTH];
        for (uint th = 1; th < NTH; th++) m = fmax(m, b1[t * NTH + th]);
        kb[K_BTH0 * NK + t] = m;
    }
    const double br = par[P_SSDSBR], c3 = par[P_SDSC3], c5 = par[P_SDSC5], c21 = par[P_SDSC21];
    const bool extra = c3 != 0.0 || c5 != 0.0 || c21 != 0.0;
    if (c3 != 0.0) {
        double epsr = sqrt(br), pi = flt(3.1415927f);
        for (uint r = 0; r < RS; r++) {
            uint is = r * NU + t;
            if (is >= NSPEC) continue;
            double q = fmax(sqrt(b1[is]) - epsr, 0.0);
            dn[is] = q * q * flt(28.16f) / (2.0 * (pi * pi));
        }
    }
    __syncthreads();
    const double c2 = par[P_SDSC2], c6 = par[P_SDSC6];
    const uint dik = (uint)par[P_DIKCUMUL];
    double facturb = c5 * ustar * ustar / par[P_GRAV] * par[P_DRAT];
    double cu = cos(usdir), su = sin(usdir);
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is >= NSPEC) continue;
        uint ik = is / NTH;
        double miche = 1.0;
        if (par[P_SSDSBM] != 1.0) {
            double x = tanh(fmin(kb[K_WN * NK + ik] * depth, 10.0));
            double p = x * (par[P_SSDSBM + 1] + x * (par[P_SSDSBM + 2] + x * (par[P_SSDSBM + 3] + x * par[P_SSDSBM + 4])));
            miche = p * p;
        }
        double coef1 = br * miche, s = sig[ik + 1];
        double q = fmax(kb[K_BTH0 * NK + ik] / coef1 - 1.0, 0.0);
        double sdiagiso = c2 * s * c6 * (q * q);
        double coef2 = c2 * s * (1.0 - c6) / (coef1 * coef1);
        double d = sdiagiso + coef2 * pow(fmax(b1[is] - coef1, 0.0), par[P_SSDSP]);
        double sv = d * a[is], dv = d;
        if (extra) {
            uint th = is - ik * NTH;
            double k = kb[K_WN * NK + ik], renewal = 0.0;
            if (c3 < 0.0 && ik + 1 > dik)
                for (uint ik2 = 0; ik2 < ik + 1 - dik; ik2++)
                    if (kb[K_BTH0 * NK + ik2] > br) {
                        double dot = 0.0;
                        for (uint th2 = 0; th2 < NTH; th2++) {
                            uint dth = th2 >= th ? th2 - th : th - th2;
                            dot += cumw[(ik2 * NK + ik) * NTH + dth] * dn[ik2 * NTH + th2];
                        }
                        renewal += dot;
                    }
            double facturb2 = -2.0 * s * k * facturb, dvisc = -4.0 * c21 * k * k;
            double coswind = ecos[th] * cu + esin[th] * su;
            double diag2 = c3 * renewal + facturb2 * fmax(0.0, coswind) + dvisc;
            dv += diag2;
            sv += a[is] * diag2;
        }
        vs[is] += scale * sv;
        vd[is] += scale * dv;
    }
    __syncthreads();
}

// W3SDS4 with SDSBCHOICE = 3 (Romero 2019) and the isotropic cumulative term
__device__ void sds4_romero(const double *par, const double *sig, const double *ecos, const double *esin, const double *a,
                            const double *kb, double *b1, double *b2, double *dn, double *vs, double *vd, double ustar,
                            double usdir, double scale)
{
    const uint t = threadIdx.x;
    const double dth = par[P_DTH], tiny = flt(1.0e-14f);
    if (t < NK) {
        double k = kb[K_WN * NK + t], s = sig[t + 1], k3 = k * k * k;
        double wml = pow(k, par[P_RO + 8]) * kb[K_DCG * NK + t];
        double sigfac = pow(s, par[P_RO + 11]) * kb[K_DCG * NK + t];
        double msp = 0.0, mc2 = 0.0, ms2 = 0.0, elc = 0.0, els = 0.0, b0 = 0.0;
        for (uint th = 0; th < NTH; th++) {
            double av = a[t * NTH + th], ml = wml * av;
            mc2 += ml * ecos[th];
            ms2 += ml * esin[th];
            msp += ml;
            elc += av * ecos[th] * sigfac;
            els += av * esin[th] * sigfac;
            b0 += fmax(av * s * k3, tiny) * dth;
        }
        b2[R_MSS * NK + t] = msp; b2[R_MC2 * NK + t] = mc2; b2[R_MS2 * NK + t] = ms2;
        b2[R_ELC * NK + t] = elc; b2[R_ELS * NK + t] = els; b2[R_BTH0 * NK + t] = b0;
    }
    __syncthreads();
    if (t == 0) {
        double sp = 0.0, sc2 = 0.0, ss2 = 0.0, elcs = 0.0, elsn = 0.0, pi = flt(3.1415927f);
        for (uint ik = 0; ik < NK; ik++) {
            sp += b2[R_MSS * NK + ik]; sc2 += b2[R_MC2 * NK + ik]; ss2 += b2[R_MS2 * NK + ik];
            elcs += b2[R_ELC * NK + ik]; elsn += b2[R_ELS * NK + ik];
            b2[R_MSS * NK + ik] = sp;
            double d = atan2(ss2, sc2);
            b2[R_MSSD * NK + ik] = d < 0.0 ? d + pi : d;
        }
        b2[R_N * NK] = atan2(elsn, elcs);
    }
    __syncthreads();
    const double grav = par[P_GRAV], br = par[P_SSDSBR], c2 = par[P_SDSC2];
    const double sdsl = par[P_RO + 1], mwd = par[P_RO + 2], spmss = par[P_RO + 3], mwpow = par[P_RO + 4];
    const double kof = par[P_RO + 5], facmtf = par[P_RO + 6], nmtf = par[P_RO + 7], cump = par[P_RO + 8], strain2 = par[P_RO + 10];
    double ko = (grav / (ustar * ustar)) / ((28.0 / kof) * (28.0 / kof));
    double dlwmean = b2[R_N * NK];
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is >= NSPEC) continue;
        uint ik = is / NTH, th = is - ik * NTH;
        double k = kb[K_WN * NK + ik], s = sig[ik + 1];
        double bth = fmax(a[is] * s * (k * k * k), tiny), lmod = 1.0;
        if (facmtf > 0.0) {
            double dirf = strain2 > 0.0 ? b2[R_MSSD * NK + ik] : dlwmean;
            double cosd = ecos[th] * cos(dirf) + esin[th] * sin(dirf);
            double kloc = pow(k, 2.0 - cump);
            double fs = 1.0 + facmtf * (pow(b2[R_MSS * NK + ik] * kloc, spmss) * (cosd * cosd));
            lmod = pow(fs, nmtf);
        }
        double fk = (1.0 + mwd * pow(fmax(1.0, k / ko), mwpow)) / (mwd + 1.0);
        double btover = sqrt(b2[R_BTH0 * NK + ik]) - sqrt(par[P_RO]);
        double q = fmax(0.0, btover);
        double brm12 = c2 * (q * q * sqrt(q));
        double l = btover >= 0.0 ? sdsl * exp(-br / bth) * fk * lmod : 0.0;
        dn[is] = l;
        double cph = s / k, c5 = cph * cph * cph * cph * cph;
        b1[is] = brm12 / (grav * grav) * l * c5 / s;
    }
    __syncthreads();
    const double c3 = par[P_SDSC3], c5 = par[P_SDSC5], c21 = par[P_SDSC21];
    const uint dik = (uint)par[P_RO + 12];
    if (t < NK) {
        double sl = 0.0;
        for (uint th = 0; th < NTH; th++) sl += dn[t * NTH + th];
        b2[R_SL * NK + t] = sl;
    }
    __syncthreads();
    if (t < NK) {
        double renewal = 0.0;
        if (c3 > 0.0 && t + 1 > dik) {
            double cph = sig[t + 1] / kb[K_WN * NK + t];
            for (uint ik2 = 0; ik2 < t + 1 - dik; ik2++) {
                double cc2 = sig[ik2 + 1] / kb[K_WN * NK + ik2];
                double w = fabs(cc2 - cph) * par[P_DSIP + ik2] / (0.5 * cc2) * dth;
                renewal -= w * b2[R_SL * NK + ik2];
            }
        }
        b2[R_ISO * NK + t] = renewal;
    }
    __syncthreads();
    double facturb = c5 * ustar * ustar / grav * par[P_DRAT];
    double cu = cos(usdir), su = sin(usdir);
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is >= NSPEC) continue;
        uint ik = is / NTH, th = is - ik * NTH;
        double k = kb[K_WN * NK + ik], s = sig[ik + 1];
        double bth = fmax(a[is] * s * (k * k * k), tiny);
        const double tiny20 = flt(1.0e-20f);
        double srhs = b1[is];
        double ddiag = srhs * br / fmax(tiny20, bth) / fmax(tiny20, a[is]);
        double facturb2 = -2.0 * s * k * facturb, dvisc = -4.0 * c21 * k * k;
        double coswind = ecos[th] * cu + esin[th] * su;
        double diag2 = c3 * b2[R_ISO * NK + ik] + facturb2 * fmax(0.0, coswind) + dvisc;
        vs[is] += scale * (srhs + a[is] * diag2);
        vd[is] += scale * (ddiag + diag2);
    }
    __syncthreads();
}

// W3SDB1: the coefficient D of depth-induced breaking (S = D A)
__device__ double sdb1(const double *par, double emean, double fmean, double wnmean, double depth)
{
    double d = 0.0;
    if (emean >= THR38) {
        double hm = par[P_SDBC2] * depth;
        if (par[P_FDONLY] == 0.0) hm = par[P_SDBC2] / wnmean * tanh(wnmean * fmax(depth, 0.0));
        double hrms = sqrt(8.0 * emean), bb = 0.0, b = 0.0;
        if (hm > THR38) { bb = hrms * hrms / (hm * hm); b = sqrt(bb); }
        double q0 = 0.0;
        if (b > 0.5 && b <= 1.0) { double y = 2.0 * b - 1.0; q0 = y * y; }
        double qb = 0.0;
        if (b > flt(0.2f) && b < 1.0) {
            double arg = exp((q0 - 1.0) / bb);
            qb = q0 - bb * (q0 - arg) / (bb - arg);
            for (int it = 0; it < 3; it++) qb = exp((qb - 1.0) / bb);
        } else if (b >= 1.0) qb = 1.0 - THR38;
        double cbj = 0.0;
        if (bb > THR38 && fabs(bb - qb) > THR38) {
            if (bb < 1.0) cbj = 2.0 * par[P_SDBC1] * qb * fmean / bb;
            else cbj = 2.0 * par[P_SDBC1] * fmean * bb;
        }
        d = -cbj;
    }
    return d;
}

// W3SBT4: bottom friction with movable-bed roughness (no ice scaling)
__device__ void sbt4(const double *par, const double *sig, const double *dden, const double *ecos, const double *esin,
                     const double *fwtab, const double *erftab, const double *a, double *kb, double *sc, double *vs,
                     double *vd, double depth, double d50_, double psic)
{
    const uint t = threadIdx.x;
    if (t < NK) {
        double eb = 0.0, ebx = 0.0, eby = 0.0;
        for (uint th = 0; th < NTH; th++) {
            double x = a[t * NTH + th];
            eb += x; ebx += x * ecos[th]; eby += x * esin[th];
        }
        kb[K_EB * NK + t] = eb; kb[K_EBX * NK + t] = ebx; kb[K_EBY * NK + t] = eby;
    }
    __syncthreads();
    if (t == 0) {
        const double grav = par[P_GRAV], sg1 = par[P_SED_SG] - 1.0;
        uint nfw = (uint)par[P_NFW];
        double d50 = fmin(fmax(d50_, flt(0.00001f)), 1.0);
        double background = fmax(par[P_SBTCX + 5], par[P_SBTCX + 6] * d50);
        double sh0 = 0.0, sh1 = 0.0, sh2 = 0.0, uorb = 0.0, uorb2 = 0.0, aorb = 0.0;
        for (uint isub = 0; isub < 3; isub++) {
            double xsub = isub == 0 ? flt(-0.001f) : (isub == 1 ? flt(0.001f) : 0.0);
            double dsub = depth * (1.0 + xsub);
            double u = 0.0, ao = 0.0, ax = 0.0, ay = 0.0;
            for (uint ik = 0; ik < NK; ik++) {
                double k = kb[K_WN * NK + ik], cb = 0.0;
                if (k * dsub < 6.0) {
                    double s = sig[ik + 1], sh = sinh(k * dsub);
                    cb = 0.5 * s * s / (grav * (sh * sh));
                    double factor = (dden[ik] / kb[K_CG * NK + ik]) * 2.0 * cb * grav;
                    double varu = kb[K_EB * NK + ik] * factor;
                    u += varu;
                    ao += varu / (s * s);
                    ax += kb[K_EBX * NK + ik] * factor;
                    ay += kb[K_EBY * NK + ik] * factor;
                }
                kb[K_CBETA * NK + ik] = cb;
            }
            uorb2 = 2.0 * u;
            uorb = sqrt(fmax(flt(1.0e-7f), uorb2));
            aorb = sqrt(fmax(flt(1.0e-7f), 2.0 * ao));
            double xi = fmax((log10_ww3(fmax(aorb / d50, flt(0.3f))) - par[P_ABMIN]) / par[P_DELAB], 1.0);
            double fw = table1(fwtab, nfw, xi);
            double psi = fw * uorb2 / (2.0 * grav * sg1 * d50);
            double shields = psi / psic;
            if (isub == 0) sh0 = shields; else if (isub == 1) sh1 = shields; else sh2 = shields;
        }
        double sb0 = par[P_SBTCX], sb1 = par[P_SBTCX + 1], sb2 = par[P_SBTCX + 2], sb3 = par[P_SBTCX + 3];
        double dpsi = (sh1 - sh0) / (flt(0.001f) - flt(-0.001f)) * par[P_SBTCX + 4];
        double ripple = flt(0.0655f) * pow(uorb2 / (sg1 * grav * aorb), flt(1.4f));
        double ksubn = fmax(background, aorb * sb3);
        if (fabs(dpsi) < flt(0.0001f) * sh2 || fabs(dpsi) < flt(1.0e-8f)) {
            if (sh2 > sb2) ksubn = aorb * sb0 * pow(sh2, sb1) + aorb * ripple;
        } else {
            double psix = (sb2 - sh2) / dpsi;
            double psixt = fmax((psix + par[P_XERFMAX]) / par[P_DELXERF], 0.0);
            double proba2 = fmin(fmax(table1(erftab, (uint)par[P_NERF], psixt), 0.0), 1.0), proba1 = 1.0 - proba2;
            double psin2 = fmax(sh2 + exp(-(0.5 * psix * psix)) / sqrt(TPI) * dpsi / (proba2 + flt(0.0001f)), sb2);
            ksubn = proba1 * fmax(background, aorb * sb3) + proba2 * aorb * (sb0 * pow(psin2, sb1) + ripple);
        }
        double xi = fmax((log10_ww3(fmax(aorb / ksubn, flt(0.3f))) - par[P_ABMIN]) / par[P_DELAB], 1.0);
        sc[S_FBT] = table1(fwtab, nfw, xi) * uorb;
    }
    __syncthreads();
    double fbt = sc[S_FBT];
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is >= NSPEC) continue;
        double dsum = -fbt * kb[K_CBETA * NK + is / NTH];
        vs[is] += dsum * a[is];
        vd[is] += dsum;
    }
    __syncthreads();
}
