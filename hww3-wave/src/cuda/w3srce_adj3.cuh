// Adjoint of W3SDS4 with SDSBCHOICE = 3 (Romero 2019) and the isotropic
// cumulative term (milestone M4): the transpose of sds4_romero_tl of
// hww3_wave::srce_tl. b1, b2 and dn hold the values of sds4_romero at the
// same state (b1 the breaking source SRHS, dn the crest lengths Lambda, b2
// the per-frequency sums R_* NK + ik and the direction DLWMEAN at R_N NK).
// From vsb, vdb, adds the cotangents of a (ab), of u*, the wind direction and
// SDSC2 (scb). Scratch: p1, p2, lb (NSPEC each), w (12 NK).
__device__ void sds4_romero_adj(const double *par, const double *sig, const double *ecos, const double *esin, const double *a,
                                const double *kb, const double *b1, const double *b2, const double *dn, const double *vsb,
                                const double *vdb, double *ab, double *p1, double *p2, double *lb, double *w, double *scb,
                                double *red, double ustar, double usdir, double scale)
{
    const uint t = threadIdx.x;
    const double dth = par[P_DTH], tiny = flt(1.0e-14f), tiny20 = flt(1.0e-20f);
    const double grav = par[P_GRAV], br = par[P_SSDSBR], c2 = par[P_SDSC2];
    const double sdsl = par[P_RO + 1], mwd = par[P_RO + 2], spmss = par[P_RO + 3], mwpow = par[P_RO + 4];
    const double kof = par[P_RO + 5], facmtf = par[P_RO + 6], nmtf = par[P_RO + 7], cump = par[P_RO + 8], strain2 = par[P_RO + 10];
    const double c3 = par[P_SDSC3], c5 = par[P_SDSC5], c21 = par[P_SDSC21];
    const uint dik = (uint)par[P_RO + 12];
    double kf = 28.0 / kof, ko = (grav / (ustar * ustar)) / (kf * kf);
    double facturb = c5 * ustar * ustar / grav * par[P_DRAT];
    double cu = cos(usdir), su = sin(usdir);
    double dlw = b2[R_N * NK];
    // the per-frequency arrays of w: 0 ISOB, 1 B0B, 2 SLB, 3 MSSB, 4 MSSDB,
    // 5 MSPB, 6 MC2B, 7 MS2B, 8 SC2 (cumulative), 9 SS2 (cumulative)
    double *isob = w, *b0b = w + NK, *slb = w + 2 * NK, *mssb = w + 3 * NK, *mssdb = w + 4 * NK;
    double *mspb = w + 5 * NK, *mc2b = w + 6 * NK, *ms2b = w + 7 * NK, *csc2 = w + 8 * NK, *css2 = w + 9 * NK;
    __shared__ double cel[2];
    // the output: S = scale (SRHS + a DIAG2), D = scale (DDIAG + DIAG2)
    double c2b = 0.0, ustb = 0.0, udb = 0.0;
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is >= NSPEC) continue;
        uint ik = is / NTH, th = is - ik * NTH;
        double gs = scale * vsb[is], gd = scale * vdb[is];
        double k = kb[K_WN * NK + ik], s = sig[ik + 1], k3 = k * k * k;
        double x = a[is], bv = x * s * k3, bth = fmax(bv, tiny);
        double sq0 = sqrt(b2[R_BTH0 * NK + ik]), btover = sq0 - sqrt(par[P_RO]), q = fmax(0.0, btover);
        double q25 = q * q * sqrt(q), brm12 = c2 * q25;
        double cph = s / k, c5p = cph * cph * cph * cph * cph / (grav * grav * s);
        double l = dn[is], srhs = b1[is];
        double bb = fmax(tiny20, bth), aa = fmax(tiny20, x);
        double facturb2 = -2.0 * s * k * facturb, dvisc = -4.0 * c21 * k * k;
        double cwv = ecos[th] * cu + esin[th] * su, cw = fmax(0.0, cwv);
        double diag2 = c3 * b2[R_ISO * NK + ik] + facturb2 * cw + dvisc;
        double g = gs * diag2;
        double srhsb = gs + gd * br / (bb * aa);
        if (bth > tiny20 && bv > tiny) g += -gd * br * srhs / (bb * bb * aa) * s * k3;
        if (x > tiny20) g += -gd * br * srhs / (bb * aa * aa);
        ab[is] += g;
        double diag2b = gs * x + gd;
        p1[is] = c3 * diag2b;
        ustb += diag2b * cw * (-2.0 * s * k) * c5 * 2.0 * ustar / grav * par[P_DRAT];
        if (cwv > 0.0) udb += diag2b * facturb2 * (-ecos[th] * su + esin[th] * cu);
        double brm12b = srhsb * l * c5p;
        lb[is] = srhsb * brm12 * c5p;
        c2b += brm12b * q25;
        double qb = brm12b * c2 * 2.5 * q * sqrt(q);
        p2[is] = btover > 0.0 ? qb / (2.0 * sq0) : 0.0;
    }
    __syncthreads();
    if (t < NK) {
        double si = 0.0, sb = 0.0;
        for (uint th = 0; th < NTH; th++) { si += p1[t * NTH + th]; sb += p2[t * NTH + th]; }
        isob[t] = si;
        b0b[t] = sb;
    }
    __syncthreads();
    // ISO(ik) = - sum_{ik2 < ik + 1 - DIK} W(ik, ik2) SL(ik2)
    if (t < NK) {
        double g = 0.0;
        if (c3 > 0.0) {
            double cc2 = sig[t + 1] / kb[K_WN * NK + t];
            for (uint ik = t + dik; ik < NK; ik++) {
                if (ik + 1 <= dik) continue;
                double cph = sig[ik + 1] / kb[K_WN * NK + ik];
                double wv = fabs(cc2 - cph) * par[P_DSIP + t] / (0.5 * cc2) * dth;
                g -= wv * isob[ik];
            }
        }
        slb[t] = g;
    }
    __syncthreads();
    // Lambda = sdsl exp(-br / bth) fk lmod where BTOVER >= 0
    double dlwb = 0.0;
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is >= NSPEC) continue;
        uint ik = is / NTH, th = is - ik * NTH;
        double lbv = lb[is] + slb[ik];
        double k = kb[K_WN * NK + ik], s = sig[ik + 1], k3 = k * k * k;
        double x = a[is], bv = x * s * k3, bth = fmax(bv, tiny);
        double btover = sqrt(b2[R_BTH0 * NK + ik]) - sqrt(par[P_RO]);
        double m1 = 0.0, m2 = 0.0;
        if (btover >= 0.0) {
            double lmod = 1.0, fs = 1.0, mp = 0.0, m = 0.0, kloc = 1.0, cosd = 0.0, cf = 1.0, sf = 0.0;
            if (facmtf > 0.0) {
                double dirf = strain2 > 0.0 ? b2[R_MSSD * NK + ik] : dlw;
                cf = cos(dirf);
                sf = sin(dirf);
                cosd = ecos[th] * cf + esin[th] * sf;
                kloc = pow(k, 2.0 - cump);
                m = b2[R_MSS * NK + ik] * kloc;
                mp = m > 0.0 ? pow(m, spmss) : 0.0;
                fs = 1.0 + facmtf * (mp * (cosd * cosd));
                lmod = pow(fs, nmtf);
            }
            double xk = k / ko, xp = pow(fmax(1.0, xk), mwpow);
            double fk = (1.0 + mwd * xp) / (mwd + 1.0);
            double e = sdsl * exp(-br / bth);
            double eb = lbv * fk * lmod, fkb = lbv * e * lmod, lmodb = lbv * e * fk;
            if (bv > tiny) ab[is] += eb * e * br / (bth * bth) * s * k3;
            if (xk > 1.0) ustb += fkb * mwd * mwpow * xp / xk / (mwd + 1.0) * 2.0 * xk / ustar;
            if (facmtf > 0.0) {
                double fsb = lmodb * nmtf * lmod / fs;
                double mpb = fsb * facmtf * (cosd * cosd), cosdb = fsb * facmtf * mp * 2.0 * cosd;
                m1 = (m > 0.0 ? mpb * spmss * mp / m : 0.0) * kloc;
                double dirfb = cosdb * (esin[th] * cf - ecos[th] * sf);
                if (strain2 > 0.0) m2 = dirfb; else dlwb += dirfb;
            }
        }
        p1[is] = m1;
        p2[is] = m2;
    }
    c2b = block_sum(c2b, red);
    ustb = block_sum(ustb, red);
    udb = block_sum(udb, red);
    dlwb = block_sum(dlwb, red);
    if (t < NK) {
        double s1 = 0.0, s2 = 0.0;
        for (uint th = 0; th < NTH; th++) { s1 += p1[t * NTH + th]; s2 += p2[t * NTH + th]; }
        mssb[t] = s1;
        mssdb[t] = s2;
    }
    __syncthreads();
    if (t == 0) {
        scb[B_SDSC2] += c2b;
        scb[S_USTAR] += ustb;
        scb[B_UDIR] += udb;
        // the cumulative sums, then their transpose
        double sc2 = 0.0, ss2 = 0.0, elcs = 0.0, elsn = 0.0;
        for (uint ik = 0; ik < NK; ik++) {
            sc2 += b2[R_MC2 * NK + ik]; ss2 += b2[R_MS2 * NK + ik];
            elcs += b2[R_ELC * NK + ik]; elsn += b2[R_ELS * NK + ik];
            csc2[ik] = sc2; css2[ik] = ss2;
        }
        double r2 = elcs * elcs + elsn * elsn;
        cel[0] = r2 > 0.0 ? -dlwb * elsn / r2 : 0.0;
        cel[1] = r2 > 0.0 ? dlwb * elcs / r2 : 0.0;
        double pb = 0.0, cb = 0.0, sb = 0.0;
        for (int ik = NK - 1; ik >= 0; ik--) {
            pb += mssb[ik];
            double rr = csc2[ik] * csc2[ik] + css2[ik] * css2[ik];
            if (rr > 0.0) { sb += mssdb[ik] * csc2[ik] / rr; cb += -mssdb[ik] * css2[ik] / rr; }
            mspb[ik] = pb; mc2b[ik] = cb; ms2b[ik] = sb;
        }
    }
    __syncthreads();
    const double elcb = cel[0], elsb = cel[1];
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is >= NSPEC) continue;
        uint ik = is / NTH, th = is - ik * NTH;
        double k = kb[K_WN * NK + ik], s = sig[ik + 1], k3 = k * k * k;
        double wml = pow(k, cump) * kb[K_DCG * NK + ik], sigfac = pow(s, par[P_RO + 11]) * kb[K_DCG * NK + ik];
        double bv = a[is] * s * k3;
        double g = wml * (mspb[ik] + mc2b[ik] * ecos[th] + ms2b[ik] * esin[th]) + sigfac * (elcb * ecos[th] + elsb * esin[th]);
        if (bv > tiny) g += b0b[ik] * s * k3 * dth;
        ab[is] += g;
    }
    __syncthreads();
}
