// Adjoints of the source terms (milestone M4), second part: W3SDS4 with
// SDSBCHOICE = 1 and the DIA (see w3srce_adj1.cuh for the conventions).

// W3SDS4 (SDSBCHOICE = 1): from vsb, vdb, adds the cotangents of a (ab), of
// u*, the wind direction and SDSC2 (scb). b1 and dn hold the forward
// saturation and crest lengths, kb[K_BTH0] the maxima (from sds4 at the same
// state). Scratch: b1b, d2b, qbb (NSPEC each, qbb then holding the
// cotangents of the crest lengths), kbb (NK).
__device__ void sds4_adj(const double *par, const double *sig, const double *ecos, const double *esin, const double *satw,
                         const uint *sati, const double *cumw, const double *a, const double *kb, const double *b1,
                         const double *dn, const double *vsb, const double *vdb, double *ab, double *b1b, double *d2b,
                         double *qbb, double *kbb, double *scb, double *red, double depth, double ustar, double usdir,
                         double scale)
{
    const uint t = threadIdx.x;
    const double br = par[P_SSDSBR], c3 = par[P_SDSC3], c5 = par[P_SDSC5], c21 = par[P_SDSC21];
    const double c2 = par[P_SDSC2], c6 = par[P_SDSC6], grav = par[P_GRAV];
    const bool extra = c3 != 0.0 || c5 != 0.0 || c21 != 0.0;
    const uint dik = (uint)par[P_DIKCUMUL];
    double facturb = c5 * ustar * ustar / grav * par[P_DRAT];
    double cu = cos(usdir), su = sin(usdir);
    double c2b = 0.0, ustb = 0.0, udb = 0.0;
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is >= NSPEC) continue;
        uint ik = is / NTH, th = is - ik * NTH;
        double gs = scale * vsb[is], gd = scale * vdb[is];
        double miche = 1.0;
        if (par[P_SSDSBM] != 1.0) {
            double x = tanh(fmin(kb[K_WN * NK + ik] * depth, 10.0));
            double p = x * (par[P_SSDSBM + 1] + x * (par[P_SSDSBM + 2] + x * (par[P_SSDSBM + 3] + x * par[P_SSDSBM + 4])));
            miche = p * p;
        }
        double coef1 = br * miche, s = sig[ik + 1];
        double qv = kb[K_BTH0 * NK + ik] / coef1 - 1.0, q = fmax(qv, 0.0);
        double sdiagiso = c2 * s * c6 * (q * q);
        double coef2 = c2 * s * (1.0 - c6) / (coef1 * coef1);
        double rv = b1[is] - coef1, rr = fmax(rv, 0.0), rp = pow(rr, par[P_SSDSP]);
        double d = sdiagiso + coef2 * rp;
        double x = a[is], k = kb[K_WN * NK + ik];
        double diag2 = 0.0, cwv = 0.0, facturb2 = 0.0;
        if (extra) {
            facturb2 = -2.0 * s * k * facturb;
            cwv = ecos[th] * cu + esin[th] * su;
            double renewal = 0.0;
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
            diag2 = c3 * renewal + facturb2 * fmax(0.0, cwv) - 4.0 * c21 * k * k;
        }
        ab[is] += gs * (d + diag2);
        double dbar = gs * x + gd;
        double rpb = dbar * coef2;
        double rrb = rr > 0.0 ? rpb * par[P_SSDSP] * rp / rr : 0.0;
        b1b[is] = rv > 0.0 ? rrb : 0.0;
        c2b += dbar * (s * c6 * (q * q) + rp * s * (1.0 - c6) / (coef1 * coef1));
        qbb[is] = qv > 0.0 ? dbar * c2 * s * c6 * 2.0 * q / coef1 : 0.0;
        double d2bar = extra ? dbar : 0.0;
        d2b[is] = c3 * d2bar;
        if (extra) {
            double cw = fmax(0.0, cwv);
            ustb += d2bar * cw * (-2.0 * s * k) * c5 * 2.0 * ustar / grav * par[P_DRAT];
            if (cwv > 0.0) udb += d2bar * facturb2 * (-ecos[th] * su + esin[th] * cu);
        }
    }
    c2b = block_sum(c2b, red);
    ustb = block_sum(ustb, red);
    udb = block_sum(udb, red);
    if (t == 0) { scb[B_SDSC2] += c2b; scb[S_USTAR] += ustb; scb[B_UDIR] += udb; }
    // the per-frequency sums of the cotangents of the maxima (qbb then free)
    if (t < NK) {
        double qs = 0.0;
        for (uint th = 0; th < NTH; th++) qs += qbb[t * NTH + th];
        kbb[t] = qs;
    }
    __syncthreads();
    double *dnb = qbb;
    // cotangent of the crest lengths: gather of the renewal sums
    for (uint r = 0; r < RS; r++) {
        uint is2 = r * NU + t;
        if (is2 >= NSPEC) continue;
        uint ik2 = is2 / NTH, th2 = is2 - ik2 * NTH;
        double g = 0.0;
        if (extra && c3 < 0.0 && kb[K_BTH0 * NK + ik2] > br)
            for (uint ik = ik2 + dik; ik < NK; ik++)
                for (uint th = 0; th < NTH; th++) {
                    uint dth = th2 >= th ? th2 - th : th - th2;
                    g += d2b[ik * NTH + th] * cumw[(ik2 * NK + ik) * NTH + dth];
                }
        dnb[is2] = g;
    }
    __syncthreads();
    if (c3 != 0.0) {
        double epsr = sqrt(br), pi = flt(3.1415927f), cb = flt(28.16f) / (2.0 * (pi * pi));
        for (uint r = 0; r < RS; r++) {
            uint is = r * NU + t;
            if (is >= NSPEC) continue;
            double sq = sqrt(b1[is]), qv = sq - epsr, q = fmax(qv, 0.0);
            if (qv > 0.0) b1b[is] += dnb[is] * 2.0 * q * cb / (2.0 * sq);
        }
    }
    // the maxima over the directions: their cotangent to the first maximum
    __syncthreads();
    if (t < NK) {
        double m = b1[t * NTH];
        uint im = 0;
        for (uint th = 1; th < NTH; th++)
            if (b1[t * NTH + th] > m) { m = b1[t * NTH + th]; im = th; }
        b1b[t * NTH + im] += kbb[t];
    }
    __syncthreads();
    // the saturation: B = FACSAT sum_j SATW A(SATI), gathered per bin
    for (uint r = 0; r < RS; r++) {
        uint is2 = r * NU + t;
        if (is2 >= NSPEC) continue;
        uint ik = is2 / NTH, th2 = is2 - ik * NTH;
        double k = kb[K_WN * NK + ik];
        double facsat = sig[ik + 1] * (k * k * k) * par[P_DTH];
        double g = 0.0;
        for (uint th = 0; th < NTH; th++)
            for (uint j = 0; j < NSAT; j++)
                if (sati[j + th * NSAT] == th2) g += b1b[ik * NTH + th] * satw[j + th * NSAT];
        ab[is2] += g * facsat;
    }
    __syncthreads();
}

// the DIA: from vsb, vdb, adds the cotangents of a (ab), of WNMEAN
// (kdmean = WNMEAN depth, scb[S_WNMEAN]) and of NLPROP (B_SNLC1); S and D
// being linear in CONS, the cotangent of CONS is the sum of the products of
// the cotangents of the B arrays with their values, over CONS. The
// transposed index tables are those of hww3_wave::adjoint::DiaAdjTables.
// Scratch: ue (the extended spectrum), ueb (NUE), gs, gd (NSPEC), eb
// (NSPECX), conx (NK).
__device__ void snl1_adj(const double *par, const double *sig, const uint *dia_ip, const double *af11, const double *a,
                         const double *kb, const double *vsb, const double *vdb,
                         double *ab, double *scb, double *red, const uint *tb1s, const uint *tb1e, const uint *tb2s,
                         const uint *tb2e, const uint *tps, const uint *tpe, double *ue, double *ueb, double *gs, double *gd,
                         double *eb, double *conx, double kdmean, double depth, double scale)
{
    const uint t = threadIdx.x;
    const double kdcon = par[P_KDCON], kdmn = par[P_KDMN], s1 = par[P_SNLS1], s2 = par[P_SNLS2], s3 = par[P_SNLS3];
    double xv = kdcon * kdmean, x = fmax(xv, kdmn);
    double x2v = s3 * x, x2 = fmax(-1.0e15, x2v), ex = exp(x2);
    double g1 = 1.0 + s1 / x * (1.0 - s2 * x) * ex;
    double cons = par[P_SNLC1] * g1;
    if (t < NK) conx[t] = TPIINV / sig[t + 1] * kb[K_CG * NK + t];
    __syncthreads();
    for (uint r = 0; r < (NUE + NU - 1) / NU; r++) {
        uint u = r * NU + t;
        if (u >= NUE) continue;
        double v = 0.0;
        if (u >= NTH && u < NSPEC + NTH) { uint is = u - NTH; v = a[is] / conx[is / NTH]; }
        ue[u] = v;
    }
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is < NSPEC) { gs[is] = scale * conx[is / NTH] * vsb[is]; gd[is] = scale * vdb[is]; }
    }
    __syncthreads();
    if (t < NTH)
        for (uint ifr = NK; ifr < NSPECY / NTH; ifr++) ue[NTH + ifr * NTH + t] = ue[ifr * NTH + t] * par[P_FACHFE];
    __syncthreads();
    const double dal1 = par[P_DAL1], dal2 = par[P_DAL2], dal3 = par[P_DAL3];
    const double w0 = par[P_AWG], w1 = par[P_AWG + 1], w2 = par[P_AWG + 2], w3 = par[P_AWG + 3];
    const double w4 = par[P_AWG + 4], w5 = par[P_AWG + 5], w6 = par[P_AWG + 6], w7 = par[P_AWG + 7];
    double e00b[(NSPECX + NU - 1) / NU], ep1b[(NSPECX + NU - 1) / NU], em1b[(NSPECX + NU - 1) / NU];
    double ep2b[(NSPECX + NU - 1) / NU], em2b[(NSPECX + NU - 1) / NU], acc[(NUE + NU - 1) / NU];
    double cpart = 0.0;
    for (uint r = 0; r < (NSPECX + NU - 1) / NU; r++) {
        uint isp = r * NU + t;
        e00b[r] = ep1b[r] = em1b[r] = ep2b[r] = em2b[r] = 0.0;
        if (isp >= NSPECX) continue;
        uint xb = isp + NTH;
        double b10 = 0.0, b11 = 0.0, b12 = 0.0, b20 = 0.0, b21 = 0.0, b22 = 0.0;
        if (isp < NSPEC) { b10 = -2.0 * gs[isp]; b20 = -2.0 * gs[isp]; }
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
            double gdn = -2.0 * caf * gd[isp];
            ge += gdn * 2.0 * (ep1 * dal1 + em1 * dal2 + ep2 * dal1 + em2 * dal2);
            gp1 += gdn * (2.0 * e00 * dal1 - em1 * dal3);
            gm1 += gdn * (2.0 * e00 * dal2 - ep1 * dal3);
            gp2 += gdn * (2.0 * e00 * dal1 - em2 * dal3);
            gm2 += gdn * (2.0 * e00 * dal2 - ep2 * dal3);
        }
        ge += b12 * (caf * m1 + factor * dal2) + b22 * (caf * m2 + factor * dal2);
        gp1 += -b12 * factor * dal3;
        gp2 += -b22 * factor * dal3;
        e00b[r] = ge; ep1b[r] = gp1; em1b[r] = gm1; ep2b[r] = gp2; em2b[r] = gm2;
        // the products with the values of the B arrays (linear in CONS)
        cpart += b10 * factor * sa1b + b20 * factor * sa2b + b11 * factor * p1 + b21 * factor * p2 + b12 * factor * m1 + b22 * factor * m2;
        if (isp < NSPEC) cpart += -2.0 * gd[isp] * (caf * (sa1a + sa1b) + caf * (sa2a + sa2b));
    }
    double consb = cons != 0.0 ? block_sum(cpart, red) / cons : block_sum(0.0, red);
    if (t == 0) {
        scb[B_SNLC1] += consb * g1;
        double g1b = consb * par[P_SNLC1];
        double xb = g1b * s1 * (-(1.0 - s2 * x) * ex / (x * x) + (-s2 * ex) / x);
        double x2b = g1b * s1 / x * (1.0 - s2 * x) * ex;
        xb += x2v > -1.0e15 ? x2b * s3 : 0.0;
        double kdb = xv > kdmn ? xb * kdcon : 0.0;
        scb[S_WNMEAN] += kdb * depth;
    }
    for (uint r = 0; r < (NUE + NU - 1) / NU; r++) acc[r] = 0.0;
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
                    uint e = tpe[q], isp = e / 16, kk = e - 16 * isp;
                    acc[r] += par[P_AWG + (kk % 8)] * eb[isp];
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
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is < NSPEC) ab[is] += ueb[is + NTH] / conx[is / NTH];
    }
    __syncthreads();
}
