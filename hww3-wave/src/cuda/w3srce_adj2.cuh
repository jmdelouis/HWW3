// Adjoint of W3SIN4 (SINTABLE = 1), milestone M4: the transpose of sin4_tl
// of hww3_wave::srce_tl. Conventions as in w3srce_adj1.cuh.

// cotangents of the weights ddeli1, ddelj1 of tauhf2_d (the table fixed)
__device__ void tauhf2_dw(const double *tab, uint base, uint ind, uint j, uint n1, double deli1, double deli2, double delj1, double &ci, double &cj)
{
    double delj2 = 1.0 - delj1;
    double a00 = tab[ind + j * n1 + base], a10 = tab[ind + 1 + j * n1 + base];
    double a01 = tab[ind + (j + 1) * n1 + base], a11 = tab[ind + 1 + (j + 1) * n1 + base];
    ci = (a10 - a00) * delj2 + (a11 - a01) * delj1;
    cj = (a01 * deli2 + a11 * deli1) - (a00 * deli2 + a10 * deli1);
}

// W3SIN4: from the cotangents vsb, vdb of S, D (with acc, scale applied
// here) and tauwb[0..1] of the wave-supported stress, adds the cotangents
// of a (ab), u*, z0, the wind direction, BETAMAX (scb[S_USTAR], scb[S_Z0],
// scb[B_UDIR], scb[B_BETA]) and of BETAMAX through the derivative of the
// tail-stress table (scb[B_BETATAB], with dtauhf the derivative of the table
// with respect to BETAMAX). spec = false leaves out the cotangents of S, D
// (the freeze mask of the tangent). With fresh, it recomputes sin4 (acc =
// false) into sc, dr, ws and the per-frequency stress tp[2 NK]; without, it
// uses those of the caller's last call of sin4 with tp. kbb: NK of scratch.
__device__ void sin4_adj(const double *par, const double *tauhf, const double *dtauhf, const double *sig, const double *dden,
                         const double *ecos, const double *esin, const double *sig2k, const double *dden2k,
                         const double *fwtab, const double *a, double *kb, double *sc, double *dr, uchar *ws, double *tp,
                         const double *vsb, const double *vdb, double *ab, double *scb, double *kbb, double *red,
                         double ustar, double z0, double usdir, double beta, double scale, bool acc, bool spec,
                         const double *tauwb, bool fresh)
{
    const uint t = threadIdx.x;
    // the forward pass (no accumulation): sc holds its scalars, tp the
    // stress; without fresh, the caller's last call of sin4 at this state,
    // with tp, left them (nothing in between writes those slots)
    if (fresh)
        sin4(par, tauhf, sig, dden, ecos, esin, sig2k, dden2k, fwtab, a, 0, 0, ws, kb, sc, dr, ustar, z0, usdir, beta, 1.0, false, tp);
    const double grav = par[P_GRAV], kappa = par[P_KAPPA], nu_air = par[P_NU_AIR], drat = par[P_DRAT];
    const double const0 = beta / (kappa * kappa) * drat;
    const double tsh = fabs(par[P_TTAUWSHELTER]);
    const double sw0 = par[P_SSWELLF], sw2 = par[P_SSWELLF + 2], sw4 = par[P_SSWELLF + 4];
    const double uorb = sc[S_UORB], fw = sc[S_FW], fu = sc[S_FU], fud = sc[S_FUD], pturb = sc[S_PTURB], pvisc = sc[S_PVISC];
    // carried cotangents (in sc slots of scratch: dr is reused per frequency)
    __shared__ double cs[16];
    // cs: 0 SXB, 1 SYB, 2 TAUXB, 3 TAUYB, 4 COSUB (last), 5 SINUB (last), 6 USTB, 7 Z0B, 8 BETAB, 9 TABB
    if (t == 0) {
        for (int j = 0; j < 16; j++) cs[j] = 0.0;
        // the end block, in reverse
        double sx = sc[S_SX], sy = sc[S_SY];
        double taupx = sc[S_TAUX] - tsh * sx, taupy = sc[S_TAUY] - tsh * sy;
        double p = taupx * taupx + taupy * taupy;
        double ust = pow(p, flt(0.25f)), usdirp = atan2(taupy, taupx);
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
        double cp = cos(usdirp), sp = sin(usdirp);
        double tx0 = sx + tauhf_v * cp, ty0 = sy + tauhf_v * sp;
        double tauw = sqrt(tx0 * tx0 + ty0 * ty0);
        const double eps2 = flt(0.000001f);
        double um2 = fmax(ustar, eps2), ust2 = um2 * um2;
        double limv = ust2 - flt(0.00001f), lim = fmax(limv, eps2 * eps2);
        double tauwbv = fmin(tauw, lim);
        double tx0b = tauwb[0], ty0b = tauwb[1], ustb = 0.0;
        if (tauwbv < tauw) {
            double r = tauwbv / tauw;
            double rtb = tauwb[0] * tx0 + tauwb[1] * ty0;
            tx0b = tauwb[0] * r;
            ty0b = tauwb[1] * r;
            double twbb = rtb / tauw, twb = -rtb * tauwbv / (tauw * tauw);
            if (tauw < lim) twb += twbb;
            else if (limv > eps2 * eps2 && ustar > eps2) ustb += twbb * 2.0 * um2;
            if (tauw > 0.0) { tx0b += twb * tx0 / tauw; ty0b += twb * ty0 / tauw; }
        }
        double sxb = tx0b, syb = ty0b;
        double thvb = tx0b * cp + ty0b * sp;
        double udpb = tx0b * (-tauhf_v * sp) + ty0b * (tauhf_v * cp);
        double levb = thvb * ust * ust * tau1;
        double ustpb = thvb * levtail0 * 2.0 * ust * tau1;
        double tau1b = thvb * levtail0 * ust * ust;
        // the interpolation of the tail stress
        uint n1 = ix.n1, n2 = ix.n2;
        uint o0 = ix.i * n1 * n2, o1 = (ix.i + 1) * n1 * n2;
        double d1b = 0.0, dj1b = 0.0, dk1b = 0.0, betab = 0.0, tabb = 0.0;
        bool shel = par[P_TTAUWSHELTER] > 0.0;
        if (ix.nbt > 1) {
            uint mb[4] = {ix.m * ix.stride, (ix.nb + ix.m) * ix.stride, (ix.m + 1) * ix.stride, (ix.nb + ix.m + 1) * ix.stride};
            double db = par[P_BT_DB];
            double xr = (beta - par[P_BT_B0]) / db, top = (double)(ix.nbt - 1);
            double xb = fmax(0.0, fmin(top, xr)), tt = xb - (double)ix.m, t2 = tt * tt;
            bool inside = xr >= 0.0 && xr <= top;
            double hd[4] = {inside ? (6.0 * t2 - 6.0 * tt) / db : 0.0, inside ? 3.0 * t2 - 4.0 * tt + 1.0 : 0.0,
                            inside ? (6.0 * tt - 6.0 * t2) / db : 0.0, inside ? 3.0 * t2 - 2.0 * tt : 0.0};
            for (int c = 0; c < 4; c++) {
                double ci, cj;
                if (shel) {
                    double la = tauhf2(tauhf, mb[c] + o0, ix.ind, ix.j, n1, ix.deli1, ix.deli2, ix.delj1, ix.delj2);
                    double lb = tauhf2(tauhf, mb[c] + o1, ix.ind, ix.j, n1, ix.deli1, ix.deli2, ix.delj1, ix.delj2);
                    double ci0, cj0, ci1, cj1;
                    tauhf2_dw(tauhf, mb[c] + o0, ix.ind, ix.j, n1, ix.deli1, ix.deli2, ix.delj1, ci0, cj0);
                    tauhf2_dw(tauhf, mb[c] + o1, ix.ind, ix.j, n1, ix.deli1, ix.deli2, ix.delj1, ci1, cj1);
                    d1b += tau1b * ix.hw[c] * (ci0 * ix.delk2 + ci1 * ix.delk1);
                    dj1b += tau1b * ix.hw[c] * (cj0 * ix.delk2 + cj1 * ix.delk1);
                    dk1b += tau1b * ix.hw[c] * (lb - la);
                    betab += tau1b * hd[c] * (la * ix.delk2 + lb * ix.delk1);
                } else {
                    double tc = tauhf2(tauhf, mb[c], ix.ind, ix.j, n1, ix.deli1, ix.deli2, ix.delj1, ix.delj2);
                    tauhf2_dw(tauhf, mb[c], ix.ind, ix.j, n1, ix.deli1, ix.deli2, ix.delj1, ci, cj);
                    d1b += tau1b * ix.hw[c] * ci;
                    dj1b += tau1b * ix.hw[c] * cj;
                    betab += tau1b * hd[c] * tc;
                }
            }
        } else if (shel) {
            double la = tauhf2(tauhf, o0, ix.ind, ix.j, n1, ix.deli1, ix.deli2, ix.delj1, ix.delj2);
            double lb = tauhf2(tauhf, o1, ix.ind, ix.j, n1, ix.deli1, ix.deli2, ix.delj1, ix.delj2);
            double ci0, cj0, ci1, cj1;
            tauhf2_dw(tauhf, o0, ix.ind, ix.j, n1, ix.deli1, ix.deli2, ix.delj1, ci0, cj0);
            tauhf2_dw(tauhf, o1, ix.ind, ix.j, n1, ix.deli1, ix.deli2, ix.delj1, ci1, cj1);
            d1b += tau1b * (ci0 * ix.delk2 + ci1 * ix.delk1);
            dj1b += tau1b * (cj0 * ix.delk2 + cj1 * ix.delk1);
            dk1b += tau1b * (lb - la);
            double ela = tauhf2(dtauhf, o0, ix.ind, ix.j, n1, ix.deli1, ix.deli2, ix.delj1, ix.delj2);
            double elb = tauhf2(dtauhf, o1, ix.ind, ix.j, n1, ix.deli1, ix.deli2, ix.delj1, ix.delj2);
            tabb += tau1b * (ela * ix.delk2 + elb * ix.delk1);
        } else {
            double ci, cj;
            tauhf2_dw(tauhf, 0, ix.ind, ix.j, n1, ix.deli1, ix.deli2, ix.delj1, ci, cj);
            d1b += tau1b * ci;
            dj1b += tau1b * cj;
            tabb += tau1b * tauhf2(dtauhf, 0, ix.ind, ix.j, n1, ix.deli1, ix.deli2, ix.delj1, ix.delj2);
        }
        // the weights: deli1 (ust), delj1 (z0, ust), delk1 (levtail0)
        double xi = ust / par[P_DELUST], fi = xi - (double)ix.ind;
        if (fi < 1.0 && fi > 0.0) ustpb += d1b / par[P_DELUST];
        double um = fmax(ust, flt(0.00001f));
        double q = grav * z0 / (um * um) - par[P_AALPHA];
        double xj = fmax(0.0, q / par[P_DELALP]), fj = xj - (double)ix.j;
        if (fj < 1.0 && fj > 0.0 && q > 0.0) {
            double qb = dj1b / par[P_DELALP];
            cs[7] += qb * grav / (um * um);
            if (ust > flt(0.00001f)) ustpb += qb * grav * (-2.0 * z0 / (um * um * um));
        }
        if (shel) {
            double xk = levtail0 / par[P_DELTAIL], fk = xk - (double)ix.i;
            if (fk < 1.0) levb += dk1b / par[P_DELTAIL];
        }
        // levtail0 = c0 sum a cw^3 (the last frequency, the direction of its loop)
        double cosub = 0.0, sinub = 0.0;
        for (uint th = 0; th < NTH; th++) {
            double cwv = ecos[th] * cosu + esin[th] * sinu, cw = fmax(cwv, 0.0);
            double x = a[(NK - 1) * NTH + th];
            ab[(NK - 1) * NTH + th] += levb * c0 * cw * cw * cw;
            if (cwv > 0.0) {
                double cwb = levb * c0 * x * 3.0 * cw * cw;
                cosub += cwb * ecos[th];
                sinub += cwb * esin[th];
            }
        }
        // ust = p^(1/4), usdirp = atan2(taupy, taupx)
        double taupxb = 0.0, taupyb = 0.0;
        if (p > 0.0) {
            double pb = ustpb * 0.25 * ust / p;
            taupxb = pb * 2.0 * taupx - udpb * taupy / p;
            taupyb = pb * 2.0 * taupy + udpb * taupx / p;
        }
        cs[0] = sxb - tsh * taupxb;
        cs[1] = syb - tsh * taupyb;
        cs[2] = taupxb;
        cs[3] = taupyb;
        cs[4] = cosub;
        cs[5] = sinub;
        cs[6] = ustb;
        cs[8] = betab;
        cs[9] = tabb;
    }
    __syncthreads();
    // the loop over the frequencies, in reverse
    double z0b = 0.0, betab = 0.0, fwb = 0.0, uorbb = 0.0, pturbb = 0.0;
    const double dzcn_dz0 = 1.0 / z0;
    for (int ikk = NK - 1; ikk >= 0; ikk--) {
        uint ik = (uint)ikk;
        double taupx = tp[2 * ik], taupy = tp[2 * ik + 1];
        double p = taupx * taupx + taupy * taupy, r = sqrt(p), r4 = sqrt(r);
        double m = fmax(ustar, flt(0.3f)), ustp = fmin(r4, m);
        double cosu = r > 0.0 ? taupx / r : 1.0, sinu = r > 0.0 ? taupy / r : 0.0;
        double sxb = cs[0], syb = cs[1];
        double ustpb_t = 0.0, cosub_t = 0.0, sinub_t = 0.0;
        if (t < NTH) {
            double k = kb[K_WN * NK + ik], c = kb[K_CG * NK + ik], s2 = sig2k[ik], s = sig[ik + 1];
            double cm = k / s2, ucn = ustp * cm + par[P_ZZALP];
            double const2 = dden2k[ik] / c * grav / (s / k * drat);
            double cnst = s2 * const0, zcn = log(k * z0);
            double swellcoefv = -sw4 * drat * 2.0 * k * sqrt(2.0 * nu_air * s2);
            double swellcoeft = -drat * sw0 * 16.0 * s2 * s2 / grav;
            uint is = ik * NTH + t;
            double coswind = ecos[t] * cosu + esin[t] * sinu;
            double dstw = 0.0, zlog = 0.0, xw = 0.0;
            bool wind = false;
            if (coswind > flt(0.01f)) {
                xw = coswind * ucn;
                zlog = zcn + kappa / xw;
                if (zlog < 0.0) {
                    double z2 = zlog * zlog;
                    dstw = cnst * exp(zlog) * (z2 * z2) * ucn * ucn * pow(coswind, par[P_SSINTHP]);
                    wind = true;
                }
            }
            bool swell = (sw0 != 0.0 && dstw < flt(1.0e-7f) * s2) || sw2 > 0.0;
            double dturb = swellcoeft * (fw * uorb + (fu + fud * coswind) * ustp);
            double dst = dstw + (swell ? pturb * dturb + pvisc * swellcoefv : 0.0);
            double x = a[is];
            // cotangents of DST: from S, D (acc) and from TEMP2 (SX, SY)
            double dstb = 0.0;
            if (acc && spec) { dstb += scale * (vsb[is] * x + vdb[is]); ab[is] += scale * vsb[is] * dst; }
            if (!(dst < 0.0)) {
                double temp2b = sxb * ecos[t] + syb * esin[t];
                dstb += temp2b * const2 * x;
                ab[is] += temp2b * const2 * dst;
            }
            double ucnb = 0.0, cwb = 0.0;
            if (wind) {
                double zlogb = dstb * dstw * (1.0 + 4.0 / zlog);
                z0b += zlogb * dzcn_dz0;
                double xwb = -kappa / (xw * xw) * zlogb;
                cwb += xwb * ucn;
                ucnb += xwb * coswind;
                ucnb += dstb * dstw * 2.0 / ucn;
                cwb += dstb * dstw * par[P_SSINTHP] / coswind;
                betab += dstb * dstw / cnst * s2 * drat / (kappa * kappa);
            }
            if (swell) {
                pturbb += dstb * (dturb - swellcoefv);
                double dturbb = dstb * pturb;
                fwb += dturbb * swellcoeft * uorb;
                uorbb += dturbb * swellcoeft * fw;
                cwb += dturbb * swellcoeft * fud * ustp;
                ustpb_t += dturbb * swellcoeft * (fu + fud * coswind);
            }
            ustpb_t += ucnb * cm;
            cosub_t = cwb * ecos[t];
            sinub_t = cwb * esin[t];
        }
        __syncthreads();
        if (t < NTH) { dr[t] = ustpb_t; dr[NTH + t] = cosub_t; dr[2 * NTH + t] = sinub_t; }
        __syncthreads();
        if (t == 0) {
            double ustpb = 0.0, cosub = 0.0, sinub = 0.0;
            for (uint th = 0; th < NTH; th++) { ustpb += dr[th]; cosub += dr[NTH + th]; sinub += dr[2 * NTH + th]; }
            if (ik == NK - 1) { cosub += cs[4]; sinub += cs[5]; }
            double taupxb = 0.0, taupyb = 0.0;
            if (r4 < m) {
                if (p > 0.0) {
                    double pb = ustpb * 0.25 * r4 / p;
                    taupxb += pb * 2.0 * taupx;
                    taupyb += pb * 2.0 * taupy;
                }
            } else if (ustar > flt(0.3f)) cs[6] += ustpb;
            if (p > 0.0) {
                double udpb = -sinu * cosub + cosu * sinub;
                taupxb += -udpb * taupy / p;
                taupyb += udpb * taupx / p;
            }
            cs[2] += taupxb;
            cs[3] += taupyb;
            cs[0] += -tsh * taupxb;
            cs[1] += -tsh * taupyb;
        }
        __syncthreads();
    }
    // the energy per frequency (kb[K_EB] may have been overwritten since)
    if (t < NK) {
        double e = 0.0;
        for (uint th = 0; th < NTH; th++) e += a[t * NTH + th];
        kbb[t] = e;
    }
    z0b = block_sum(z0b, red);
    betab = block_sum(betab, red);
    fwb = block_sum(fwb, red);
    uorbb = block_sum(uorbb, red);
    pturbb = block_sum(pturbb, red);
    // the start of sin4, in reverse: TAUX, TAUY, UORB, AORB, PTURB, FW
    if (t == 0) {
        double cu = cos(usdir), su = sin(usdir);
        double ustb = cs[6] + cs[2] * 2.0 * ustar * cu + cs[3] * 2.0 * ustar * su;
        double udb = -cs[2] * ustar * ustar * su + cs[3] * ustar * ustar * cu;
        double uo = 0.0, ao = 0.0;
        for (uint ik = 0; ik < NK; ik++) {
            double e = kbb[ik];
            double s = sig[ik + 1], c = kb[K_CG * NK + ik];
            uo += e * s * s * dden[ik] / c;
            ao += e * dden[ik] / c;
        }
        double ex = 1.0 - 0.5 * par[P_SSWELLF + 5];
        double aorb1 = 2.0 * pow(ao, ex);
        double re = 4.0 * uorb * aorb1 / nu_air;
        double aob = 0.0;
        if (par[P_SSWELLF + 3] > 0.0 && par[P_SSWELLF + 6] > 0.0) {
            double th = tanh((re - par[P_SSWELLF + 3]) / par[P_SSWELLF + 6]);
            double reb = pturbb * 0.5 * (1.0 - th * th) / par[P_SSWELLF + 6];
            uorbb += reb * 4.0 * aorb1 / nu_air;
            double aorb1b = reb * 4.0 * uorb / nu_air;
            aob += ao > 0.0 ? aorb1b * 2.0 * ex * pow(ao, ex - 1.0) : 0.0;
        }
        double z0bt = cs[7] + z0b;
        if (par[P_SSWELLF + 1] != 0.0) {
            double z0visc = flt(0.1f) * nu_air / fmax(ustar, flt(0.0001f));
            double z0r = par[P_ZZ0RAT] * z0, z0noz = fmax(z0visc, z0r);
            double aorb2 = 2.0 * sqrt(ao), rr0 = aorb2 / z0noz, rr = fmax(rr0, 3.0);
            double xi = (log10_ww3(rr) - par[P_ABMIN]) / par[P_DELAB];
            double xib = fwb * table1_dx(fwtab, (uint)par[P_NFW], xi);
            double rrb = rr0 > 3.0 ? xib / (rr * flt(2.3025851f)) / par[P_DELAB] : 0.0;
            double aorb2b = rrb / z0noz, z0nozb = -rrb * aorb2 / (z0noz * z0noz);
            if (z0visc > z0r) { if (ustar > flt(0.0001f)) ustb += z0nozb * (-z0visc / ustar); }
            else z0bt += z0nozb * par[P_ZZ0RAT];
            aob += ao > 0.0 ? aorb2b / sqrt(ao) : 0.0;
        }
        double uob = uo > 0.0 ? uorbb / sqrt(uo) : 0.0;
        for (uint ik = 0; ik < NK; ik++) {
            double s = sig[ik + 1], c = kb[K_CG * NK + ik];
            kbb[ik] = uob * s * s * dden[ik] / c + aob * dden[ik] / c;
        }
        scb[S_USTAR] += ustb;
        scb[S_Z0] += z0bt;
        scb[B_UDIR] += udb;
        scb[B_BETA] += cs[8] + betab;
        scb[B_BETATAB] += cs[9];
    }
    __syncthreads();
    for (uint r = 0; r < RS; r++) {
        uint is = r * NU + t;
        if (is < NSPEC) ab[is] += kbb[is / NTH];
    }
    __syncthreads();
}
