// The adjoint kernel of W3SRCE in CUDA C (milestone M4), one block of NU
// threads per node: the transpose of the tangent kernel srce_tl of
// hww3_wave::srce_tl over one global step. The forward kernel srce_fwd has
// written, for each node, the spectrum and the scalars at the start of each
// dynamic sub-step (the tape); the adjoint takes the sub-steps in reverse,
// recomputes each one's forward pass from its record (bit for bit the same
// sources, step and update as the forward kernel), then runs the transposes
// of the tail, the FHMAX limit, FHIGH, W3SPR4, the update and of each source
// term, which add the cotangents of the spectrum at the start of the
// sub-step and of the scalars it starts from (u*, z0, FHIGH, WNMEAN, EMEAN,
// FMEAN). Then the start of W3SRCE (W3SPR4, W3SIN4, W3SPR4) and BETAMAX.
// The freeze mask P_TLFREEZE leaves out the same contributions as in the
// tangent (each source term's cotangents of S and D, the u* and z0 after
// W3SPR4, the denominators of the update).
//
// Inputs: va the spectra at the start of the step, vout those at its end
// (from srce_fwd), ybar the cotangent of vout. Outputs: abar the cotangent
// of va, ubar and udbar those of the wind speed and direction, and per node
// pbar[i NPB + j] those of BETAMAX (j = 0, the derivative of the tail-stress
// table included through dtauhf, the table's derivative in BETAMAX), SDSC2
// (1), NLPROP (2), BJGAM (3) and the coefficients of BETAMAX(U) (4 + k).
// The block b handles node i0 + b (the nodes of a batch).

#define NPB (4 + BU_MAXK)

struct SrceAdjSh {
    double *a, *an, *dn, *vs, *vd, *ab, *vsb, *vdb, *ue, *b1, *b2, *kb, *sc, *scb, *dr, *red, *tp, *kbb;
    uchar *ws;
};

__device__ SrceAdjSh srce_adj_shared(double *sh)
{
    SrceAdjSh s;
    s.a = sh;
    s.an = s.a + NSPEC;
    s.dn = s.an + NSPEC; // an, dn contiguous: the DIA's ueb (NUE <= 2 NSPEC)
    s.vs = s.dn + NSPEC;
    s.vd = s.vs + NSPEC;
    s.ab = s.vd + NSPEC;
    s.vsb = s.ab + NSPEC;
    s.vdb = s.vsb + NSPEC;
    s.ue = s.vdb + NSPEC;
    s.b1 = s.ue + NUE;
    s.b2 = s.b1 + NB1;
    s.kb = s.b2 + NB1;
    s.sc = s.kb + NKB * NK;
    s.scb = s.sc + NSC;
    s.dr = s.scb + NSCB;
    s.red = s.dr + 3 * NTH;
    s.tp = s.red + NU + 1;
    s.kbb = s.tp + 2 * NK;
    s.ws = (uchar *)(s.kbb + 2 * NK);
    return s;
}

#define SRCE_ADJ_SHARED_DOUBLES (8 * NSPEC + NUE + 2 * NB1 + NKB * NK + NSC + NSCB + 3 * NTH + NU + 1 + 4 * NK)
#define SRCE_ADJ_SHARED_BYTES (SRCE_ADJ_SHARED_DOUBLES * 8 + NSPEC)

extern "C" __global__ void __launch_bounds__(NU, 1)
srce_adj(const double *va, const double *vout, const double *ybar, double *abar, SRCE_IN_ARGS, const double *dtauhf,
         const uint *tb1s, const uint *tb1e, const uint *tb2s, const uint *tb2e, const uint *tps, const uint *tpe,
         double dtg, uint np, uint i0, const double *tape, const unsigned long long *tape_off, const uint *nsub, double *ubar,
         double *udbar, double *pbar)
{
    extern __shared__ double shm[];
    SRCE_IN_MAKE;
    const uint i = i0 + blockIdx.x, t = threadIdx.x;
    if (i >= np) return;
    if ((in.nflag[i] & 4u) == 0) {
        for (uint r = 0; r < RS; r++) {
            uint is = r * NU + t;
            if (is < NSPEC) { uint ik = is / NTH, th = is - ik * NTH, g = (ik * np + i) * NTH + th; abar[g] = ybar[g]; }
        }
        if (t == 0) { ubar[i] = 0.0; udbar[i] = 0.0; }
        if (t < NPB) pbar[i * NPB + t] = 0.0;
        return;
    }
    SrceAdjSh S = srce_adj_shared(shm);
    double *a = S.a, *an = S.an, *vs = S.vs, *vd = S.vd, *ab = S.ab, *vsb = S.vsb, *vdb = S.vdb, *kb = S.kb, *sc = S.sc, *scb = S.scb;
    const uint fz = (uint)par[P_TLFREEZE];
    // the cotangent of the output max(0, a)
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is < NSPEC) {
            uint ik = is / NTH, th = is - ik * NTH, g = (ik * np + i) * NTH + th;
            ab[is] = vout[g] > 0.0 ? ybar[g] : 0.0;
        }
    }
    if (t < NSCB) scb[t] = 0.0;
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
    const uint nk_sub = nsub[i];
    const double *tape_i = tape + tape_off[i];
    for (int k = (int)nk_sub - 1; k >= 0; k--) {
        // ---- the forward pass of sub-step k from its record
        const double *rec = tape_i + (unsigned long long)k * (NSPEC + NSC);
        for (uint r = 0; r < RS; r++) { uint is = r * NU + t; if (is < NSPEC) { a[is] = rec[is]; vs[is] = 0.0; vd[is] = 0.0; } }
        if (t < NSC) sc[t] = rec[NSPEC + t];
        __syncthreads();
        double ustar = sc[S_USTAR], z0 = sc[S_Z0], fhigh = sc[S_FHIGH], wnmean = sc[S_WNMEAN], emean = sc[S_EMEAN], fmean = sc[S_FMEAN];
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
            double rem = dtg - dttot, idt = 1.0 + trunc(flt(0.99f) * rem / dt);
            dt = rem / idt;
            bool shave = dt < par[P_DTMIN] && dt < rem;
            dt = fmax(dt, fmin(par[P_DTMIN], rem));
            sc[S_DT] = dt;
            sc[S_SHAVE] = shave ? 1.0 : 0.0;
            sc[S_DTTOT] = dttot + dt;
        }
        __syncthreads();
        const double dt = sc[S_DT];
        const bool shave = sc[S_SHAVE] > 0.0;
        for (uint r = 0; r < RS; r++) {
            uint is = r * NU + t;
            if (is >= NSPEC) continue;
            if (is < nspech) {
                uint ik = is / NTH;
                double aold = a[is];
                double inc = vs[is] * dt / fmax(1.0, 1.0 - dt * vd[is]);
                if (shave) { double dam = kb[K_DAM * NK + ik], m = fmin(dam, fabs(inc)); inc = inc >= 0.0 ? m : -m; }
                double a1 = fmax(0.0, aold + inc);
                double incdb = (dbd * aold) * dt / fmax(1.0, 1.0 - dt * dbd);
                an[is] = fmax(0.0, a1 + incdb);
            } else an[is] = a[is];
        }
        __syncthreads();
        spr4(par, in.taut, in.sig, in.dden, an, S.ws, kb, sc, u, stauwx, stauwy);
        if (t == 0) {
            double fage = par[P_FFXFA] * tanh(flt(0.3f) * u * sc[S_FMEANWS] * TPI / par[P_GRAV]);
            double fh1 = (par[P_FFXFM] + fage) * sc[S_FMEAN1], fh2 = par[P_FFXPM] / sc[S_USTAR];
            double fh = fmin(in.sig[NK], fmax(fh1, fh2));
            sc[S_NKH] = fmax(2.0, fmin(nkh1, trunc(par[P_FACTI2] + par[P_FACTI1] * log(fmax(flt(1.0e-7f), fh)))));
            bool last = sc[S_DTTOT] >= flt(0.9999f) * dtg;
            sc[S_SCALE] = 0.0;
            if (last) {
                double wnm = sc[S_WNMEAN];
                double hm = par[P_FHMAX] * tanh(wnm * fmax(0.0, d_inp)) / fmax(flt(0.0001f), wnm);
                double em = hm * hm / 16.0, e = sc[S_EMEAN];
                if (e > em && e > flt(1.0e-30f)) { sc[S_SCALE] = 1.0; sc[S_CONS] = em; }
            }
        }
        __syncthreads();
        // ---- the transposes, last operation first
        // the tail above NKH
        uint nkt = (uint)sc[S_NKH];
        if (t < NTH)
            for (uint ik = NK - 1; ik >= nkt && ik >= 1; ik--) {
                ab[(ik - 1) * NTH + t] += par[P_FACHFA] * ab[ik * NTH + t];
                ab[ik * NTH + t] = 0.0;
            }
        __syncthreads();
        // the FHMAX limit: a = a E_max / EMEAN
        if (sc[S_SCALE] > 0.0) {
            double fac = sc[S_CONS], e = sc[S_EMEAN], pf = 0.0, pe = 0.0;
            for (uint r = 0; r < RS; r++) {
                uint is = r * NU + t;
                if (is < NSPEC) {
                    double x = an[is], g = ab[is];
                    pf += g * x / e;
                    pe += -g * x * fac / (e * e);
                    ab[is] = g * fac / e;
                }
            }
            pf = block_sum(pf, S.red);
            pe = block_sum(pe, S.red);
            if (t == 0) {
                scb[S_EMEAN] += pe;
                double wnm = sc[S_WNMEAN], dpt = fmax(0.0, d_inp), thw = tanh(wnm * dpt), wm = fmax(flt(0.0001f), wnm);
                double hm = par[P_FHMAX] * thw / wm, hmb = pf * hm / 8.0;
                scb[S_WNMEAN] += hmb * par[P_FHMAX] * ((1.0 - thw * thw) * dpt * wm - thw * (wnm > flt(0.0001f) ? 1.0 : 0.0)) / (wm * wm);
            }
        }
        // FHIGH at the end of the sub-step, then the freeze of u* and z0
        if (t == 0) {
            double fhb = scb[S_FHIGH];
            scb[S_FHIGH] = 0.0;
            const double g = par[P_GRAV], fmws = sc[S_FMEANWS], fm1 = sc[S_FMEAN1], ue_ = sc[S_USTAR];
            double arg = flt(0.3f) * u * fmws * TPI / g, th = tanh(arg), fage = par[P_FFXFA] * th;
            double fh1 = (par[P_FFXFM] + fage) * fm1, fh2 = par[P_FFXPM] / ue_, mx = fmax(fh1, fh2);
            double mxb = mx < in.sig[NK] ? fhb : 0.0;
            if (fh1 > fh2) {
                scb[S_FMEAN1] += mxb * (par[P_FFXFM] + fage);
                double argb = mxb * fm1 * par[P_FFXFA] * (1.0 - th * th);
                scb[B_U] += argb * flt(0.3f) * fmws * TPI / g;
                scb[S_FMEANWS] += argb * flt(0.3f) * u * TPI / g;
            } else scb[S_USTAR] += -mxb * fh2 / ue_;
            if (fz & 32u) { scb[S_USTAR] = 0.0; scb[S_Z0] = 0.0; }
        }
        __syncthreads();
        // W3SPR4 at the end of the sub-step
        spr4_adj(par, in.taut, in.sig, in.dden, an, S.ws, kb, S.kbb, scb, ab, u, stauwx, stauwy, scb + B_TWX);
        // the update
        double dbdp = 0.0;
        for (uint r = 0; r < RS; r++) {
            uint is = r * NU + t;
            if (is >= NSPEC) continue;
            if (is >= nspech) { vsb[is] = 0.0; vdb[is] = 0.0; continue; }
            uint ik = is / NTH;
            double aold = a[is], vsv = vs[is], vdv = vd[is];
            double denv = 1.0 - dt * vdv, den = fmax(1.0, denv);
            double inc = vsv * dt / den;
            bool clamped = false;
            if (shave) {
                double dam = kb[K_DAM * NK + ik];
                clamped = fabs(inc) > dam;
                double m = fmin(dam, fabs(inc));
                inc = inc >= 0.0 ? m : -m;
            }
            double s1 = aold + inc, a1 = fmax(0.0, s1);
            double den2v = 1.0 - dt * dbd, den2 = fmax(1.0, den2v);
            double incdb = (dbd * aold) * dt / den2, s2 = a1 + incdb;
            double g = s2 > 0.0 ? ab[is] : 0.0;
            bool c2 = den2v > 1.0 && (fz & 64u) == 0;
            dbdp += g * dt * (aold * den2 + (c2 ? dbd * aold * dt : 0.0)) / (den2 * den2);
            double daold = g * dt * dbd / den2, dincb = 0.0;
            if (s1 > 0.0) { daold += g; dincb = g; }
            if (shave && clamped) dincb = 0.0;
            bool c1 = denv > 1.0 && (fz & 64u) == 0;
            vsb[is] = dincb * dt / den;
            vdb[is] = c1 ? dincb * dt * vsv * dt / (den * den) : 0.0;
            ab[is] = daold;
        }
        double dbdb = block_sum(dbdp, S.red);
        // the source terms at the state of the start of the sub-step
        if ((fz & 16u) == 0)
            sbt4_adj(par, in.sig, in.dden, in.fwtab, in.erftab, a, kb, vsb, vdb, ab, S.kbb, sc, S.red, depth, sd50, spsic);
        if ((fz & 128u) == 0 && t == 0) sdb1_adj(par, emean, fmean, wnmean, depth, dbdb, scb);
        __syncthreads();
        if ((fz & 8u) == 0) {
#if ROMERO
            sds4_romero_adj(par, in.sig, in.ecos, in.esin, a, kb, S.b1, S.b2, S.dn, vsb, vdb, ab, vs, vd, an, S.ue, scb, S.red, ustar,
                            udir, isc_ds);
#else
            sds4_adj(par, in.sig, in.ecos, in.esin, in.satw, in.sati, in.cumw, a, kb, S.b1, S.dn, vsb, vdb, ab, vs, vd, an, S.kbb, scb,
                     S.red, depth, ustar, udir, isc_ds);
#endif
        }
        if ((fz & 4u) == 0)
            snl1_adj(par, in.sig, in.dia_ip, in.af11, a, kb, vsb, vdb, ab, scb, S.red, tb1s, tb1e, tb2s, tb2e, tps, tpe, S.ue, an, vs, vd,
                     S.b1, S.kbb, wnmean * depth, depth, isc_nl);
        sin4_adj(par, in.tauhf, dtauhf, in.sig, in.dden, in.ecos, in.esin, in.sig2k, in.dden2k, in.fwtab, a, kb, sc, S.dr, S.ws, S.tp,
                 vsb, vdb, ab, scb, S.kbb, S.red, ustar, z0, udir, beta, isc_in, true, (fz & 2u) == 0, scb + B_TWX);
        if (t == 0) { scb[B_TWX] = 0.0; scb[B_TWY] = 0.0; }
        __syncthreads();
        if ((fz & 1u) == 0) sln1_adj(par, in.sig, in.ecos, in.esin, kb, vsb, fhigh, ustar, udir, isc_ln, scb, S.red);
        __syncthreads();
    }
    // ---- the start of W3SRCE: W3SPR4, W3SIN4 (no sources), W3SPR4, FHIGH
    const double *rec0 = tape_i;
    for (uint r = 0; r < RS; r++) { uint is = r * NU + t; if (is < NSPEC) { a[is] = rec0[is]; S.ws[is] = 1; } }
    if (t < NSC) sc[t] = rec0[NSPEC + t];
    __syncthreads();
    if (t == 0) {
        double fhb = scb[S_FHIGH];
        scb[S_FHIGH] = 0.0;
        double fm1 = sc[S_FMEAN1], fmws = sc[S_FMEANWS], ust = sc[S_USTAR];
        double f1 = par[P_FFXFM] * fmax(fm1, fmws), f2 = par[P_FFXPM] / ust;
        if (f1 > f2) {
            if (fm1 > fmws) scb[S_FMEAN1] += par[P_FFXFM] * fhb;
            else scb[S_FMEANWS] += par[P_FFXFM] * fhb;
        } else scb[S_USTAR] += -fhb * f2 / ust;
    }
    __syncthreads();
    spr4(par, in.taut, in.sig, in.dden, a, S.ws, kb, sc, u, 0.0, 0.0);
    double ust0 = sc[S_USTAR], z00 = sc[S_Z0];
    sin4(par, in.tauhf, in.sig, in.dden, in.ecos, in.esin, in.sig2k, in.dden2k, in.fwtab, a, vs, vd, S.ws, kb, sc, S.dr, ust0, z00,
         udir, beta, 1.0, false, 0);
    double tauwx0 = sc[S_TAUWX], tauwy0 = sc[S_TAUWY];
    __syncthreads();
    spr4_adj(par, in.taut, in.sig, in.dden, a, S.ws, kb, S.kbb, scb, ab, u, tauwx0, tauwy0, scb + B_TWX);
    sin4_adj(par, in.tauhf, dtauhf, in.sig, in.dden, in.ecos, in.esin, in.sig2k, in.dden2k, in.fwtab, a, kb, sc, S.dr, S.ws, S.tp,
             vsb, vdb, ab, scb, S.kbb, S.red, ust0, z00, udir, beta, 1.0, false, false, scb + B_TWX);
    for (uint r = 0; r < RS; r++) { uint is = r * NU + t; if (is < NSPEC) S.ws[is] = 1; }
    if (t == 0) { scb[B_TWX] = 0.0; scb[B_TWY] = 0.0; }
    __syncthreads();
    spr4_adj(par, in.taut, in.sig, in.dden, a, S.ws, kb, S.kbb, scb, ab, u, 0.0, 0.0, scb + B_TWX);
    // the outputs: BETAMAX = P_BBETA exp(sum_k c_k phi_k(U))
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is < NSPEC) { uint ik = is / NTH, th = is - ik * NTH; abar[(ik * np + i) * NTH + th] = ab[is]; }
    }
    if (t < NPB) pbar[i * NPB + t] = 0.0;
    __syncthreads();
    if (t == 0) {
        uint nbk = (uint)par[P_BU_NK];
        double bb = scb[B_BETA], ub = scb[B_U];
        double *pb = pbar + (unsigned long long)i * NPB;
        pb[0] = (nbk > 0 ? bb * beta / par[P_BBETA] : bb) + scb[B_BETATAB];
        pb[1] = scb[B_SDSC2];
        pb[2] = scb[B_SNLC1];
        pb[3] = scb[B_SDBC2];
        for (uint kk = 0; kk < nbk && kk < BU_MAXK; kk++) {
            pb[4 + kk] = bb * beta * betau_phi(par, u, kk, nbk);
            ub += bb * beta * par[P_BU_COEF + kk] * betau_dphi(par, u, kk, nbk);
        }
        ubar[i] = ub;
        udbar[i] = scb[B_UDIR];
    }
}
