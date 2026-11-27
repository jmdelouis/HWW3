// The value kernel of W3SRCE in CUDA C (one block of NU threads per node,
// the layout of hww3_wave::srce): va[(ik np + i) NTH + th] updated in place,
// u* and the mean dynamic step written to ust and dtdyn. ROMERO selects
// SDSBCHOICE = 3. With tape non-null, the spectrum and the scalars at the
// start of each dynamic sub-step are written for the adjoint (tape_off[i]
// gives the node's first record, tape_cap[i] its number of records; each
// record holds NSPEC + NSC values), and the number of sub-steps to nsub[i].
// The block b handles node i0 + b (the nodes of a batch).

struct SrceIn {
    const double *wn, *cg, *dw, *u10, *u10d, *ice, *d50, *psic;
    const uint *nflag;
    const double *par, *sig, *dden, *ecos, *esin, *sig2k, *dden2k, *fwtab, *erftab, *satw;
    const uint *sati, *dia_ip, *dia_ic;
    const double *af11, *taut, *tauhf, *cumw;
};

// the shared arrays of the value kernel
struct SrceSh {
    double *a, *vs, *vd, *dn, *ue, *b1, *b2, *kb, *sc, *dr, *red, *tp;
    uchar *ws;
};

__device__ SrceSh srce_shared(double *sh)
{
    SrceSh s;
    s.a = sh;
    s.vs = s.a + NSPEC;
    s.vd = s.vs + NSPEC;
    s.dn = s.vd + NSPEC;
    s.ue = s.dn + NSPEC;
    s.b1 = s.ue + NUE;
    s.b2 = s.b1 + NB1;
    s.kb = s.b2 + NB1;
    s.sc = s.kb + NKB * NK;
    s.dr = s.sc + NSC;
    s.red = s.dr + 3 * NTH;
    s.tp = s.red + NU;
    s.ws = (uchar *)(s.tp + 2 * NK);
    return s;
}

#define SRCE_SHARED_DOUBLES (4 * NSPEC + NUE + 2 * NB1 + NKB * NK + NSC + 3 * NTH + NU + 2 * NK)
#define SRCE_SHARED_BYTES (SRCE_SHARED_DOUBLES * 8 + NSPEC)

// the per-frequency constants of the node
__device__ void srce_init_kb(const SrceIn &in, double *kb, uint i, uint np)
{
    const uint t = threadIdx.x;
    const double *par = in.par;
    if (t < NK) {
        double w = in.wn[(t + 1) * np + i];
        kb[K_WN * NK + t] = w;
        kb[K_CG * NK + t] = in.cg[(t + 1) * np + i];
        kb[K_DAM * NK + t] = par[P_FACP] / (in.sig[t + 1] * (w * w * w));
        kb[K_SPW * NK + t] = pow(in.sig[t + 1], 2.0 * par[P_WWNMEANPTAIL]);
        kb[K_WNP * NK + t] = pow(w, par[P_WWNMEANP]);
        kb[K_DCG * NK + t] = in.dden[t] / kb[K_CG * NK + t];
    }
}

#define SRCE_IN_ARGS                                                                                                  \
    const double *wn, const double *cg, const double *dw, const double *u10, const double *u10d, const double *ice,   \
        const double *d50, const double *psic, const uint *nflag, const double *par, const double *sig,               \
        const double *dden, const double *ecos, const double *esin, const double *sig2k, const double *dden2k,        \
        const double *fwtab, const double *erftab, const double *satw, const uint *sati, const uint *dia_ip,          \
        const uint *dia_ic, const double *af11, const double *taut, const double *tauhf, const double *cumw
#define SRCE_IN_MAKE                                                                                                  \
    SrceIn in = {wn, cg, dw, u10, u10d, ice, d50, psic, nflag, par, sig, dden, ecos, esin, sig2k, dden2k, fwtab,       \
                 erftab, satw, sati, dia_ip, dia_ic, af11, taut, tauhf, cumw}

extern "C" __global__ void __launch_bounds__(NU, 1)
srce_fwd(double *va, SRCE_IN_ARGS, double *ust_out, double *dtdyn_out, double dtg, uint np, uint i0,
         double *tape, const unsigned long long *tape_off, const uint *tape_cap, uint *nsub)
{
    extern __shared__ double shm[];
    SRCE_IN_MAKE;
    const uint i = i0 + blockIdx.x, t = threadIdx.x;
    if (i >= np || (in.nflag[i] & 4u) == 0) return;
    SrceSh S = srce_shared(shm);
    double *a = S.a, *vs = S.vs, *vd = S.vd, *kb = S.kb, *sc = S.sc;
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is < NSPEC) { uint ik = is / NTH, th = is - ik * NTH; a[is] = va[(ik * np + i) * NTH + th]; S.ws[is] = 1; }
    }
    srce_init_kb(in, kb, i, np);
    __syncthreads();
    double d_inp = in.dw[i], depth = fmax(par[P_DMIN], d_inp);
    double u = in.u10[i], udir = in.u10d[i], c = in.ice[i], sd50 = in.d50[i], spsic = in.psic[i];
    double beta = par[P_BBETA] * exp(betau_ln(par, u));
    double isc_ln = 1.0, isc_in = 1.0, isc_nl = 1.0, isc_ds = 1.0;
    if (par[P_ICE_FORCING] > 0.0) {
        isc_ln = fmax(0.0, fmin(1.0, 1.0 - c * par[P_ICESCALES]));
        isc_in = fmax(0.0, fmin(1.0, 1.0 - c * par[P_ICESCALES + 1]));
        isc_nl = fmax(0.0, fmin(1.0, 1.0 - c * par[P_ICESCALES + 2]));
        isc_ds = fmax(0.0, fmin(1.0, 1.0 - c * par[P_ICESCALES + 3]));
    }
    // before the loop: W3SPR4, W3SIN4 (TAUWBUG), W3SPR4
    spr4(par, in.taut, in.sig, in.dden, a, S.ws, kb, sc, u, 0.0, 0.0);
    double ust0 = sc[S_USTAR], z00 = sc[S_Z0];
    sin4(par, in.tauhf, in.sig, in.dden, in.ecos, in.esin, in.sig2k, in.dden2k, in.fwtab, a, vs, vd, S.ws, kb, sc, S.dr, ust0, z00,
         udir, beta, 1.0, false, 0);
    double tauwx = sc[S_TAUWX], tauwy = sc[S_TAUWY];
    spr4(par, in.taut, in.sig, in.dden, a, S.ws, kb, sc, u, tauwx, tauwy);
    if (t == 0) {
        sc[S_FHIGH] = fmax(par[P_FFXFM] * fmax(sc[S_FMEAN1], sc[S_FMEANWS]), par[P_FFXPM] / sc[S_USTAR]);
        sc[S_DTTOT] = sc[S_DTDYN] = sc[S_NSTEPS] = sc[S_DONE] = 0.0;
    }
    __syncthreads();
    uint k = 0;
    while (true) {
        if (tape) {
            // the state at the start of sub-step k
            if (k < tape_cap[i]) {
                double *rec = tape + tape_off[i] + (unsigned long long)k * (NSPEC + NSC);
                for (uint r = 0; r < RS; r++) { uint is = r * NU + t; if (is < NSPEC) rec[is] = a[is]; }
                if (t < NSC) rec[NSPEC + t] = sc[t];
            }
        }
        double ustar = sc[S_USTAR], z0 = sc[S_Z0], fhigh = sc[S_FHIGH], wnmean = sc[S_WNMEAN], emean = sc[S_EMEAN], fmean = sc[S_FMEAN];
        for (uint r = 0; r < RS; r++) { uint is = r * NU + t; if (is < NSPEC) { vs[is] = 0.0; vd[is] = 0.0; } }
        __syncthreads();
        sln1(par, in.sig, in.ecos, in.esin, kb, vs, fhigh, ustar, udir, isc_ln);
        __syncthreads();
        sin4(par, in.tauhf, in.sig, in.dden, in.ecos, in.esin, in.sig2k, in.dden2k, in.fwtab, a, vs, vd, S.ws, kb, sc, S.dr, ustar, z0,
             udir, beta, isc_in, true, 0);
        double stauwx = sc[S_TAUWX], stauwy = sc[S_TAUWY];
        snl1(par, in.sig, in.dia_ip, in.dia_ic, in.af11, a, kb, S.ue, S.b1, S.b2, S.dn, vs, vd, wnmean * depth, isc_nl);
#if ROMERO
        sds4_romero(par, in.sig, in.ecos, in.esin, a, kb, S.b1, S.b2, S.dn, vs, vd, ustar, udir, isc_ds);
#else
        sds4(par, in.sig, in.ecos, in.esin, in.satw, in.sati, in.cumw, a, kb, S.b1, S.dn, vs, vd, depth, ustar, udir, isc_ds);
#endif
        double dbd = sdb1(par, emean, fmean, wnmean, depth);
        sbt4(par, in.sig, in.dden, in.ecos, in.esin, in.fwtab, in.erftab, a, kb, sc, vs, vd, depth, sd50, spsic);
        // dynamic step: NKH from the FHIGH of the previous step
        double nkh = fmin((double)NK, trunc(par[P_FACTI2] + par[P_FACTI1] * log(fmax(flt(1.0e-7f), fhigh))));
        double nkh1 = fmin((double)NK, nkh + 1.0);
        uint nspech = (uint)fmax(nkh1, 0.0) * NTH;
        double dttot = sc[S_DTTOT];
        double afilt = fmax(kb[K_DAM * NK + NK - 1], par[P_XFLT] * sc[S_AMAX]);
        double dtl = fmin(dtg - dttot, par[P_DTMAX]);
        for (uint r = 0; r < RS; r++) {
            uint is = r * NU + t;
            if (is < nspech) {
                uint ik = is / NTH, th = is - ik * NTH;
                double ainit = va[(ik * np + i) * NTH + th];
                double damax = fmin(kb[K_DAM * NK + ik], fmax(par[P_XREL] * ainit, afilt));
                double afac = 1.0 / fmax(flt(1.0e-10f), fabs(vs[is] / damax));
                dtl = fmin(dtl, afac / fmax(flt(1.0e-10f), 1.0 + afac * fmin(0.0, vd[is])));
            }
        }
        S.red[t] = dtl;
        __syncthreads();
        if (t == 0) {
            double dt = S.red[0];
            for (uint j = 1; j < NU; j++) dt = fmin(dt, S.red[j]);
            dt = fmax(0.5, dt);
            sc[S_DTDYN] += dt;
            sc[S_NSTEPS] += 1.0;
            double rem = dtg - dttot, idt = 1.0 + trunc(flt(0.99f) * rem / dt);
            dt = rem / idt;
            bool shave = dt < par[P_DTMIN] && dt < rem;
            dt = fmax(dt, fmin(par[P_DTMIN], rem));
            sc[S_DT] = dt;
            sc[S_SHAVE] = shave ? 1.0 : 0.0;
            sc[S_DTTOT] = dttot + dt;
            sc[S_DBD] = dbd;
        }
        __syncthreads();
        double dt = sc[S_DT];
        bool shave = sc[S_SHAVE] > 0.0;
        for (uint r = 0; r < RS; r++) {
            uint is = r * NU + t;
            if (is < nspech) {
                uint ik = is / NTH;
                double aold = a[is];
                double inc = vs[is] * dt / fmax(1.0, 1.0 - dt * vd[is]);
                if (shave) { double dam = kb[K_DAM * NK + ik], m = fmin(dam, fabs(inc)); inc = inc >= 0.0 ? m : -m; }
                double a1 = fmax(0.0, aold + inc);
                double incdb = (dbd * aold) * dt / fmax(1.0, 1.0 - dt * dbd);
                a[is] = fmax(0.0, a1 + incdb);
            }
        }
        __syncthreads();
        spr4(par, in.taut, in.sig, in.dden, a, S.ws, kb, sc, u, stauwx, stauwy);
        if (t == 0) {
            double fage = par[P_FFXFA] * tanh(flt(0.3f) * u * sc[S_FMEANWS] * TPI / par[P_GRAV]);
            double fh1 = (par[P_FFXFM] + fage) * sc[S_FMEAN1], fh2 = par[P_FFXPM] / sc[S_USTAR];
            double fh = fmin(in.sig[NK], fmax(fh1, fh2));
            sc[S_FHIGH] = fh;
            sc[S_NKH] = fmax(2.0, fmin(nkh1, trunc(par[P_FACTI2] + par[P_FACTI1] * log(fmax(flt(1.0e-7f), fh)))));
            bool last = sc[S_DTTOT] >= flt(0.9999f) * dtg;
            sc[S_DONE] = last ? 1.0 : 0.0;
            sc[S_SCALE] = 0.0;
            if (last) {
                double wnm = sc[S_WNMEAN];
                double hm = par[P_FHMAX] * tanh(wnm * fmax(0.0, d_inp)) / fmax(flt(0.0001f), wnm);
                double em = hm * hm / 16.0, e = sc[S_EMEAN];
                if (e > em && e > flt(1.0e-30f)) { sc[S_SCALE] = 1.0; sc[S_CONS] = em; }
            }
        }
        __syncthreads();
        if (sc[S_SCALE] > 0.0) {
            double fac = sc[S_CONS], e = sc[S_EMEAN];
            for (uint r = 0; r < RS; r++) { uint is = r * NU + t; if (is < NSPEC) a[is] = a[is] / e * fac; }
            __syncthreads();
        }
        uint nkt = (uint)sc[S_NKH];
        if (t < NTH)
            for (uint ik = nkt; ik < NK; ik++) a[ik * NTH + t] = a[(ik - 1) * NTH + t] * par[P_FACHFA];
        __syncthreads();
        k++;
        if (sc[S_DONE] > 0.0) break;
    }
    if (t == 0) {
        ust_out[i] = sc[S_USTAR];
        dtdyn_out[i] = sc[S_DTDYN] / fmax(1.0, sc[S_NSTEPS]);
        if (nsub) nsub[i] = k;
    }
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is < NSPEC) { uint ik = is / NTH, th = is - ik * NTH; va[(ik * np + i) * NTH + th] = fmax(0.0, a[is]); }
    }
}
