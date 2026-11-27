// Adjoints of the source terms (milestone M4), first part: CALC_USTAR,
// W3SPR4, W3SLN1, W3SDB1, W3SBT4, W3SDS4 (SDSBCHOICE = 1) and the DIA. Each
// function is the transpose of the tangent of hww3_wave::srce_tl (the M3
// reference) at the state given: it recomputes the forward intermediates it
// needs, then adds the cotangents of its inputs to ab (spectrum) and scb
// (scalars, slots S_* of sc plus the B_* slots below), from the cotangents
// of its outputs (vsb, vdb for the sums S and D of the sources, scb for the
// scalars). The discrete choices (MIN, MAX, branches, table cells) are those
// of the values, as in the tangent.

#define B_BETA (NSC + 0)
#define B_SDSC2 (NSC + 1)
#define B_SNLC1 (NSC + 2)
#define B_SDBC2 (NSC + 3)
#define B_U (NSC + 4)
#define B_UDIR (NSC + 5)
#define B_BETATAB (NSC + 6)
#define B_TWX (NSC + 8)
#define B_TWY (NSC + 9)
#define NSCB (NSC + 10)

// sum over the block in a fixed order (all threads get the result)
__device__ double block_sum(double v, double *red)
{
    const uint t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    if (t == 0) {
        double s = 0.0;
        for (uint j = 0; j < NU; j++) s += red[j];
        red[NU] = s;
    }
    __syncthreads();
    double s = red[NU];
    __syncthreads();
    return s;
}

// cotangent of the table interpolation table1 at x: the weight of dx
__device__ double table1_dx(const double *tab, uint n, double x)
{
    uint size = n - 1;
    uint xi = (uint)fmax(trunc(x), 0.0);
    uint ind = xi < size - 1 ? xi : size - 1;
    double f = x - (double)ind;
    return f < 1.0 ? tab[ind + 1] - tab[ind] : 0.0;
}

// CALC_USTAR (SINTABLE = 1): from the cotangents ustb, z0b of u*, z0, adds
// those of the wind speed (scb[B_U]) and returns that of tauw (thread 0)
__device__ double calc_ustar_adj(const double *par, const double *taut, double u, double tauw, double ustb, double z0b, double *scb)
{
    const double kappa = par[P_KAPPA], grav = par[P_GRAV];
    uint itm = (uint)par[P_ITAUMAX], jum = (uint)par[P_JUMAX];
    double tmax = par[P_TAUWMAX];
    double tl = fmax(fmin(tauw, tmax), 0.0);
    double xi = sqrt(tl) / par[P_DELTAUW];
    uint xif = (uint)trunc(xi), ind = xif < itm - 1 ? xif : itm - 1;
    double fi = xi - (double)ind, deli1 = fmin(1.0, fi), deli2 = 1.0 - deli1;
    double xj = u / par[P_DELU];
    uint xjf = (uint)trunc(xj), j = xjf < jum - 1 ? xjf : jum - 1;
    double fj = xj - (double)j, delj1 = fmin(1.0, fj), delj2 = 1.0 - delj1;
    uint n = itm + 1;
    double t00 = taut[ind + j * n], t10 = taut[ind + 1 + j * n], t01 = taut[ind + (j + 1) * n], t11 = taut[ind + 1 + (j + 1) * n];
    double lo = t00 * deli2 + t10 * deli1, hi = t01 * deli2 + t11 * deli1;
    double ust = lo * delj2 + hi * delj1;
    double ub = 0.0;
    if (ust > 0.0) {
        double r = u / ust, s = fmin(r, 100.0);
        double z0 = par[P_ZZWND] * exp(-kappa * s);
        double sb = -kappa * z0 * z0b;
        double rb = r < 100.0 ? sb : 0.0;
        ub += rb / ust;
        ustb += -rb * u / (ust * ust);
    }
    (void)grav;
    double deli1b = ustb * ((t10 - t00) * delj2 + (t11 - t01) * delj1);
    double delj1b = ustb * (hi - lo);
    ub += (fj < 1.0 ? delj1b : 0.0) / par[P_DELU];
    double xib = fi < 1.0 ? deli1b : 0.0;
    double tlb = tl > 0.0 ? xib * 0.5 / sqrt(tl) / par[P_DELTAUW] : 0.0;
    scb[B_U] += ub;
    return (tauw < tmax && tauw > 0.0) ? tlb : 0.0;
}

// W3SPR4: from the cotangents of its outputs in scb (S_EMEAN, S_FMEAN,
// S_FMEAN1, S_WNMEAN, S_FMEANWS, S_USTAR, S_Z0, consumed and zeroed), adds
// those of a (ab), of the wind speed (B_U) and of the wave-supported stress
// (tauwb[0], tauwb[1]). kbb holds NK + NK values of scratch.
__device__ void spr4_adj(const double *par, const double *taut, const double *sig, const double *dden, const double *a,
                         const uchar *ws, const double *kb, double *kbb, double *scb, double *ab, double u, double tauwx,
                         double tauwy, double *tauwb)
{
    const uint t = threadIdx.x;
    // the sums over the directions (as spr4), one frequency per thread
    if (t < NK) {
        double e = 0.0, e2 = 0.0;
        for (uint th = 0; th < NTH; th++) {
            double x = a[t * NTH + th];
            e += x;
            if (ws[t * NTH + th] != 0) e2 += x;
        }
        kbb[t] = e;
        kbb[NK + t] = e2;
    }
    __syncthreads();
    if (t == 0) {
        const double tw = par[P_WWNMEANPTAIL], wp = par[P_WWNMEANP], e7 = flt(1.0e-7f);
        double emean = 0.0, fmean = 0.0, fmean1 = 0.0, wnmean = 0.0, emeanws = 0.0, fmeanws = 0.0;
        const double *eb_ = kbb, *eb2_ = kbb + NK;
        for (uint ik = 0; ik < NK; ik++) {
            double f = kb[K_DCG * NK + ik], eb = eb_[ik] * f, eb2 = eb2_[ik] * f;
            double s = sig[ik + 1], sp = kb[K_SPW * NK + ik];
            emean += eb; fmean += eb / s; fmean1 += eb * sp; wnmean += eb * kb[K_WNP * NK + ik];
            emeanws += eb2; fmeanws += eb2 * sp;
        }
        double ddl = dden[NK - 1], cgl = kb[K_CG * NK + NK - 1];
        double tailf = (ddl / cgl) / ddl;
        double eband = eb_[NK - 1] * tailf;
        emean += eband * par[P_FTE]; fmean += eband * par[P_FTF];
        fmean1 += eband * par[P_SSTXFTFTAIL]; wnmean += eband * par[P_SSTXFTWN];
        double eband2 = eb2_[NK - 1] * tailf;
        emeanws += eband2 * par[P_FTE]; fmeanws += eband2 * par[P_SSTXFTFTAIL];
        // u* and z0 from tauw
        double tauw = sqrt(tauwx * tauwx + tauwy * tauwy);
        double tb = calc_ustar_adj(par, taut, u, tauw, scb[S_USTAR], scb[S_Z0], scb);
        if (tauw > 0.0) { tauwb[0] += tb * tauwx / tauw; tauwb[1] += tb * tauwy / tauw; }
        double q1 = 1.0 / (2.0 * tw);
        double em = fmax(e7, emean), emb = 0.0;
        double deb = scb[S_EMEAN], dfmb = 0.0, df1b = 0.0, dwnb = 0.0, dewsb = 0.0, dfwsb = 0.0;
        // FMEANWS
        if (fmeanws >= e7 && emeanws >= e7) {
            double r = fmeanws / emeanws;
            double g = scb[S_FMEANWS] * TPIINV * q1 * pow(r, q1 - 1.0);
            dfwsb += g / emeanws;
            dewsb += -g * fmeanws / (emeanws * emeanws);
        }
        // WNMEAN
        {
            double wv = fmax(e7, wnmean), r2 = wv / em;
            double g = scb[S_WNMEAN] / wp * pow(r2, 1.0 / wp - 1.0);
            double dwvb = g / em;
            emb += -g * wv / (em * em);
            dwnb += wnmean > e7 ? dwvb : 0.0;
        }
        // FMEAN1
        if (fmean1 >= e7) {
            double r = fmax(e7, fmean1) / em;
            double g = scb[S_FMEAN1] * TPIINV * q1 * pow(r, q1 - 1.0);
            df1b += g / em;
            emb += -g * fmean1 / (em * em);
        }
        // FMEAN
        {
            double fmv = fmax(e7, fmean);
            deb += scb[S_FMEAN] * TPIINV / fmv;
            double fmvb = -scb[S_FMEAN] * TPIINV * emean / (fmv * fmv);
            dfmb += fmean > e7 ? fmvb : 0.0;
        }
        deb += emean > e7 ? emb : 0.0;
        for (uint ik = 0; ik < NK; ik++) {
            double f = kb[K_DCG * NK + ik], s = sig[ik + 1], sp = kb[K_SPW * NK + ik], wq = kb[K_WNP * NK + ik];
            double g = f * (deb + dfmb / s + df1b * sp + dwnb * wq);
            double g2 = f * (dewsb + dfwsb * sp);
            if (ik == NK - 1) {
                g += tailf * (deb * par[P_FTE] + dfmb * par[P_FTF] + df1b * par[P_SSTXFTFTAIL] + dwnb * par[P_SSTXFTWN]);
                g2 += tailf * (dewsb * par[P_FTE] + dfwsb * par[P_SSTXFTFTAIL]);
            }
            kbb[ik] = g;
            kbb[NK + ik] = g2;
        }
        scb[S_EMEAN] = scb[S_FMEAN] = scb[S_FMEAN1] = scb[S_WNMEAN] = scb[S_FMEANWS] = 0.0;
        scb[S_USTAR] = scb[S_Z0] = 0.0;
    }
    __syncthreads();
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is >= NSPEC) continue;
        uint ik = is / NTH;
        ab[is] += kbb[ik] + (ws[is] != 0 ? kbb[NK + ik] : 0.0);
    }
    __syncthreads();
}

// W3SLN1: from vsb, adds the cotangents of fhigh, u*, the wind direction
__device__ void sln1_adj(const double *par, const double *sig, const double *ecos, const double *esin, const double *kb,
                         const double *vsb, double fhigh, double ustar, double usdir, double scale, double *scb, double *red)
{
    const uint t = threadIdx.x;
    double cosu = cos(usdir), sinu = sin(usdir), u2 = ustar * ustar;
    double fac = par[P_SLNC1] * (u2 * u2);
    double ff1 = par[P_FSPM] * par[P_GRAV] / (28.0 * ustar);
    double ff2 = par[P_FSHF] * fmin(sig[NK], fhigh);
    double mx = fmax(ff1, ff2), ffilt = fmin(mx, 2.0 * sig[NK]);
    double facb = 0.0, ffiltb = 0.0, udb = 0.0;
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is >= NSPEC) continue;
        uint ik = is / NTH, th = is - ik * NTH;
        double g = scale * vsb[is];
        double rfr = sig[ik + 1] / ffilt;
        double cwv = ecos[th] * cosu + esin[th] * sinu, d = fmax(cwv, 0.0), d2 = d * d;
        double wnf = 0.0, e = 0.0, k = kb[K_WN * NK + ik], r2 = rfr * rfr;
        if (rfr >= 0.5) { e = exp(-(1.0 / (r2 * r2))); wnf = fac / k * e; }
        double wnfb = g * (d2 * d2);
        double db = g * wnf * 4.0 * d2 * d;
        if (cwv > 0.0) udb += db * (-ecos[th] * sinu + esin[th] * cosu);
        if (rfr >= 0.5) {
            facb += wnfb * e / k;
            double rfrb = wnfb * wnf * 4.0 / (r2 * r2 * rfr);
            ffiltb += rfrb * (-rfr / ffilt);
        }
    }
    facb = block_sum(facb, red);
    ffiltb = block_sum(ffiltb, red);
    udb = block_sum(udb, red);
    if (t == 0) {
        double ustb = facb * par[P_SLNC1] * 4.0 * u2 * ustar;
        double mxb = mx < 2.0 * sig[NK] ? ffiltb : 0.0;
        double ff1b = ff1 > ff2 ? mxb : 0.0, ff2b = ff1 > ff2 ? 0.0 : mxb;
        ustb += ff1b * (-ff1 / ustar);
        scb[S_FHIGH] += fhigh < sig[NK] ? par[P_FSHF] * ff2b : 0.0;
        scb[S_USTAR] += ustb;
        scb[B_UDIR] += udb;
    }
    __syncthreads();
}

// W3SDB1 (thread 0): from the cotangent db of D, adds those of EMEAN, FMEAN,
// WNMEAN (scb at the start of the sub-step) and of BJGAM (B_SDBC2)
__device__ void sdb1_adj(const double *par, double emean, double fmean, double wnmean, double depth, double db, double *scb)
{
    if (emean < THR38 || db == 0.0) return;
    const double c1 = par[P_SDBC1], c2 = par[P_SDBC2];
    double maxdep = fmax(depth, 0.0), th = 0.0, hm = c2 * depth;
    bool fdo = par[P_FDONLY] != 0.0;
    if (!fdo) { th = tanh(wnmean * maxdep); hm = c2 / wnmean * th; }
    double hrms = sqrt(8.0 * emean), bb = 0.0, b = 0.0;
    if (hm > THR38) { bb = hrms * hrms / (hm * hm); b = sqrt(bb); }
    bool yb = b > 0.5 && b <= 1.0;
    double y = 2.0 * b - 1.0, q0 = yb ? y * y : 0.0;
    bool qbr = b > flt(0.2f) && b < 1.0;
    double arg = 0.0, num = 0.0, den = 0.0, qs[4] = {0.0, 0.0, 0.0, 0.0}, qb = 0.0;
    if (qbr) {
        arg = exp((q0 - 1.0) / bb);
        num = q0 - arg;
        den = bb - arg;
        qs[0] = q0 - bb * num / den;
        for (int it = 0; it < 3; it++) qs[it + 1] = exp((qs[it] - 1.0) / bb);
        qb = qs[3];
    } else if (b >= 1.0) qb = 1.0 - THR38;
    if (!(bb > THR38 && fabs(bb - qb) > THR38)) return;
    double qbb = 0.0, bbb = 0.0, fmb = 0.0;
    if (bb < 1.0) {
        qbb = db * (-2.0 * c1) * fmean / bb;
        fmb = db * (-2.0 * c1) * qb / bb;
        bbb = db * 2.0 * c1 * (qb > 0.0 ? qb * fmean / (bb * bb) : 0.0);
    } else {
        fmb = db * (-2.0 * c1) * bb;
        bbb = db * (-2.0 * c1) * fmean;
    }
    double q0b = 0.0;
    if (qbr) {
        // the three iterations, in reverse
        for (int it = 2; it >= 0; it--) {
            double q = qs[it], qn = qs[it + 1];
            double g = qbb;
            qbb = g * qn / bb;
            bbb += g * qn * (-(q - 1.0)) / (bb * bb);
        }
        double argb = qbb * (bb / den - bb * num / (den * den));
        q0b += qbb * (1.0 - bb / den) + argb * arg / bb;
        bbb += qbb * (-num / den + bb * num / (den * den)) + argb * (-arg * (q0 - 1.0) / (bb * bb));
    }
    double bbar = yb ? q0b * 4.0 * y : 0.0;
    bbb += b > 0.0 ? bbar / (2.0 * b) : 0.0;
    double hrmsb = 0.0, hmb = 0.0;
    if (hm > THR38) { hrmsb = bbb * 2.0 * hrms / (hm * hm); hmb = bbb * (-2.0 * hrms * hrms / (hm * hm * hm)); }
    scb[S_EMEAN] += hrmsb * 4.0 / hrms;
    scb[S_FMEAN] += fmb;
    if (!fdo) {
        scb[B_SDBC2] += hmb * th / wnmean;
        scb[S_WNMEAN] += hmb * (-c2 * th / (wnmean * wnmean) + c2 / wnmean * (1.0 - th * th) * maxdep);
    } else scb[B_SDBC2] += hmb * depth;
}

// W3SBT4: from vsb, vdb, adds the cotangents of a. kbb: 2 NK values of scratch
__device__ void sbt4_adj(const double *par, const double *sig, const double *dden, const double *fwtab, const double *erftab,
                         const double *a, const double *kb, const double *vsb, const double *vdb, double *ab, double *kbb,
                         double *sc, double *red, double depth, double d50_, double psic)
{
    const uint t = threadIdx.x;
    // the cotangent of FBT: -sum cb (vsb a + vdb), and its value
    double part = 0.0;
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is >= NSPEC) continue;
        part += -kb[K_CBETA * NK + is / NTH] * (vsb[is] * a[is] + vdb[is]);
    }
    double fbtb = block_sum(part, red);
    if (t < NK) {
        double e = 0.0;
        for (uint th = 0; th < NTH; th++) e += a[t * NTH + th];
        kbb[NK + t] = e;
        kbb[t] = 0.0;
    }
    __syncthreads();
    if (t == 0) {
        const double grav = par[P_GRAV], sg1 = par[P_SED_SG] - 1.0, e7 = flt(1.0e-7f);
        uint nfw = (uint)par[P_NFW];
        double d50 = fmin(fmax(d50_, flt(0.00001f)), 1.0);
        double background = fmax(par[P_SBTCX + 5], par[P_SBTCX + 6] * d50);
        const double *eb_ = kbb + NK;
        double sh[3], aos[3], uo2[3], aorbs[3], xis[3], fws[3], rvs[3], xivs[3];
        double cpsi = 2.0 * grav * sg1 * d50;
        for (uint isub = 0; isub < 3; isub++) {
            double xsub = isub == 0 ? flt(-0.001f) : (isub == 1 ? flt(0.001f) : 0.0);
            double dsub = depth * (1.0 + xsub), u = 0.0, ao = 0.0;
            for (uint ik = 0; ik < NK; ik++) {
                double k = kb[K_WN * NK + ik];
                if (k * dsub < 6.0) {
                    double s = sig[ik + 1], shh = sinh(k * dsub);
                    double cb = 0.5 * s * s / (grav * (shh * shh));
                    double factor = (dden[ik] / kb[K_CG * NK + ik]) * 2.0 * cb * grav;
                    u += eb_[ik] * factor;
                    ao += eb_[ik] * factor / (s * s);
                }
            }
            aos[isub] = ao;
            uo2[isub] = 2.0 * u;
            aorbs[isub] = sqrt(fmax(e7, 2.0 * ao));
            rvs[isub] = aorbs[isub] / d50;
            double rr = fmax(rvs[isub], flt(0.3f));
            xivs[isub] = (log10_ww3(rr) - par[P_ABMIN]) / par[P_DELAB];
            xis[isub] = fmax(xivs[isub], 1.0);
            fws[isub] = table1(fwtab, nfw, xis[isub]);
            sh[isub] = fws[isub] * uo2[isub] / cpsi / psic;
        }
        double uorb2 = uo2[2], uorb = sqrt(fmax(e7, uorb2)), aorb = aorbs[2];
        double sb0 = par[P_SBTCX], sb1 = par[P_SBTCX + 1], sb2 = par[P_SBTCX + 2], sb3 = par[P_SBTCX + 3];
        double xs = flt(0.001f) - flt(-0.001f);
        double dpsi = (sh[1] - sh[0]) / xs * par[P_SBTCX + 4];
        double rq = uorb2 / (sg1 * grav * aorb);
        double ripple = flt(0.0655f) * pow(rq, flt(1.4f));
        double mbv = aorb * sb3, mb = fmax(background, mbv);
        bool flat = fabs(dpsi) < flt(0.0001f) * sh[2] || fabs(dpsi) < flt(1.0e-8f);
        double ksubn = mb, p = 0.0, psix = 0.0, pxv = 0.0, psixt = 0.0, p2v = 0.0, proba2 = 0.0, proba1 = 0.0, gg = 0.0;
        double den = 0.0, term = 0.0, pnv = 0.0, psin2 = 0.0, pp = 0.0, inner = 0.0;
        uint nerf = (uint)par[P_NERF];
        if (flat) {
            if (sh[2] > sb2) { p = pow(sh[2], sb1); ksubn = aorb * sb0 * p + aorb * ripple; }
        } else {
            psix = (sb2 - sh[2]) / dpsi;
            pxv = (psix + par[P_XERFMAX]) / par[P_DELXERF];
            psixt = fmax(pxv, 0.0);
            p2v = table1(erftab, nerf, psixt);
            proba2 = fmin(fmax(p2v, 0.0), 1.0);
            proba1 = 1.0 - proba2;
            gg = exp(-(0.5 * psix * psix)) / sqrt(TPI);
            den = proba2 + flt(0.0001f);
            term = gg * dpsi / den;
            pnv = sh[2] + term;
            psin2 = fmax(pnv, sb2);
            pp = pow(psin2, sb1);
            inner = sb0 * pp + ripple;
            ksubn = proba1 * mb + proba2 * aorb * inner;
        }
        double rv = aorb / ksubn, rr = fmax(rv, flt(0.3f));
        double xiv = (log10_ww3(rr) - par[P_ABMIN]) / par[P_DELAB], xi = fmax(xiv, 1.0);
        double fw = table1(fwtab, nfw, xi);
        // reverse: FBT = FW UORB
        double fwb = fbtb * uorb, uorbb = fbtb * fw;
        double xib = fwb * table1_dx(fwtab, nfw, xi);
        double rb = xiv > 1.0 ? xib / (rr * flt(2.3025851f)) / par[P_DELAB] : 0.0;
        double rvb = rv > flt(0.3f) ? rb : 0.0;
        double aorbb = rvb / ksubn, ksubnb = -rvb * aorb / (ksubn * ksubn);
        double shb[3] = {0.0, 0.0, 0.0}, dpsib = 0.0, rippleb = 0.0, mbb = 0.0;
        if (flat) {
            if (sh[2] > sb2) {
                aorbb += ksubnb * (sb0 * p + ripple);
                double pb = ksubnb * aorb * sb0;
                rippleb += ksubnb * aorb;
                shb[2] += sh[2] > 0.0 ? pb * sb1 * p / sh[2] : 0.0;
            } else mbb += ksubnb;
        } else {
            double proba2b = -ksubnb * mb + ksubnb * aorb * inner;
            mbb += ksubnb * proba1;
            aorbb += ksubnb * proba2 * inner;
            double innerb = ksubnb * proba2 * aorb;
            double ppb = innerb * sb0;
            rippleb += innerb;
            double psin2b = psin2 > 0.0 ? ppb * sb1 * pp / psin2 : 0.0;
            double pnvb = pnv > sb2 ? psin2b : 0.0;
            shb[2] += pnvb;
            double termb = pnvb;
            double ggb = termb * dpsi / den;
            dpsib += termb * gg / den;
            double denb = -termb * gg * dpsi / (den * den);
            proba2b += denb;
            double psixb = ggb * (-psix * gg);
            double p2vb = (p2v > 0.0 && p2v < 1.0) ? proba2b : 0.0;
            double psixtb = p2vb * table1_dx(erftab, nerf, psixt);
            double pxvb = pxv > 0.0 ? psixtb / par[P_DELXERF] : 0.0;
            psixb += pxvb;
            // psix = (sb2 - sh2) / dpsi
            shb[2] += -psixb / dpsi;
            dpsib += -psixb * (sb2 - sh[2]) / (dpsi * dpsi);
        }
        aorbb += mbv > background ? mbb * sb3 : 0.0;
        // ripple = 0.0655 rq^1.4, rq = uorb2 / (sg1 g aorb)
        double rqb = rq > 0.0 ? rippleb * flt(1.4f) * ripple / rq : 0.0;
        double uo2b[3] = {0.0, 0.0, 0.0}, aorbsb[3] = {0.0, 0.0, 0.0};
        uo2b[2] += rqb / (sg1 * grav * aorb);
        aorbb += -rqb * uorb2 / (sg1 * grav * aorb * aorb);
        // dpsi = (sh1 - sh0) / xs * c4
        shb[1] += dpsib / xs * par[P_SBTCX + 4];
        shb[0] += -dpsib / xs * par[P_SBTCX + 4];
        // uorb = sqrt(max(e7, uorb2)) (last isub)
        uo2b[2] += uorb2 > e7 ? uorbb / (2.0 * uorb) : 0.0;
        aorbsb[2] += aorbb;
        for (uint isub = 0; isub < 3; isub++) {
            // shields = fw uorb2 / cpsi / psic
            double fwb2 = shb[isub] * uo2[isub] / cpsi / psic;
            uo2b[isub] += shb[isub] * fws[isub] / cpsi / psic;
            double xib2 = fwb2 * table1_dx(fwtab, nfw, xis[isub]);
            double rr2 = fmax(rvs[isub], flt(0.3f));
            double rb2 = xivs[isub] > 1.0 ? xib2 / (rr2 * flt(2.3025851f)) / par[P_DELAB] : 0.0;
            aorbsb[isub] += rvs[isub] > flt(0.3f) ? rb2 / d50 : 0.0;
            double aob = 2.0 * aos[isub] > e7 ? aorbsb[isub] * 2.0 / (2.0 * aorbs[isub]) : 0.0;
            double ub = 2.0 * uo2b[isub];
            double xsub = isub == 0 ? flt(-0.001f) : (isub == 1 ? flt(0.001f) : 0.0);
            double dsub = depth * (1.0 + xsub);
            for (uint ik = 0; ik < NK; ik++) {
                double k = kb[K_WN * NK + ik];
                if (k * dsub < 6.0) {
                    double s = sig[ik + 1], shh = sinh(k * dsub);
                    double cb = 0.5 * s * s / (grav * (shh * shh));
                    double factor = (dden[ik] / kb[K_CG * NK + ik]) * 2.0 * cb * grav;
                    kbb[ik] += ub * factor + aob * factor / (s * s);
                }
            }
        }
    }
    __syncthreads();
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is >= NSPEC) continue;
        ab[is] += -sc[S_FBT] * kb[K_CBETA * NK + is / NTH] * vsb[is] + kbb[is / NTH];
    }
    __syncthreads();
}
