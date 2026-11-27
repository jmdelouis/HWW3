//! Milestone M4: the transposed index tables of the DIA, for the adjoints
//! that gather the cotangents of the B arrays and of the extended spectrum
//! instead of scattering them (no atomics).

/// The transposed index tables of the DIA adjoints (`hww3_wave::adjoint`
/// and the CUDA C source adjoint), built on the host from `DIA_IP`
/// (`16 nspecx`) and `DIA_IC` (`16 nspec`): for each bin of the B arrays and
/// of the extended spectrum, the entries that read it, so that the adjoints
/// gather instead of scattering.
pub struct DiaAdjTables {
    /// Start of the entries of each bin of B1 (`nspecx + nth + 1` values).
    pub tb1s: Vec<u32>,
    /// The entries reading B1: `8 is + j` for the weight `j` of bin `is`.
    pub tb1e: Vec<u32>,
    /// As [`DiaAdjTables::tb1s`] for B2.
    pub tb2s: Vec<u32>,
    /// As [`DiaAdjTables::tb1e`] for B2.
    pub tb2e: Vec<u32>,
    /// Start of the entries of each bin of the extended spectrum, for the
    /// four groups of four rows of `DIA_IP` (`4 (nspecy + nth + 1)` values).
    pub tps: Vec<u32>,
    /// The entries reading the extended spectrum: `16 isp + k` for row `k`.
    pub tpe: Vec<u32>,
}

impl DiaAdjTables {
    /// The tables for `nspec` bins, `nspecx`, `nspecy` and `nth` as in
    /// [`crate::sources::SrceTables`].
    pub fn new(dia_ip: &[u32], dia_ic: &[u32], nspec: usize, nspecx: usize, nspecy: usize, nth: usize) -> DiaAdjTables {
        let csr = |nx: usize, items: Vec<(usize, u32)>| -> (Vec<u32>, Vec<u32>) {
            let mut cnt = vec![0u32; nx + 1];
            for &(x, _) in &items {
                cnt[x + 1] += 1;
            }
            for x in 0..nx {
                cnt[x + 1] += cnt[x];
            }
            let mut pos = cnt.clone();
            let mut ent = vec![0u32; items.len()];
            for (x, e) in items {
                ent[pos[x] as usize] = e;
                pos[x] += 1;
            }
            (cnt, ent)
        };
        let nb = nspecx + nth;
        let mut i1 = vec![];
        let mut i2 = vec![];
        for j in 0..8 {
            for is in 0..nspec {
                i1.push((dia_ic[2 * j * nspec + is] as usize, (is * 8 + j) as u32));
                i2.push((dia_ic[(2 * j + 1) * nspec + is] as usize, (is * 8 + j) as u32));
            }
        }
        let (tb1s, tb1e) = csr(nb, i1);
        let (tb2s, tb2e) = csr(nb, i2);
        let nue = nspecy + nth;
        let mut tps = vec![];
        let mut tpe = vec![];
        for g in 0..4 {
            let mut items = vec![];
            for k in 4 * g..4 * g + 4 {
                for isp in 0..nspecx {
                    items.push((dia_ip[k * nspecx + isp] as usize, (isp * 16 + k) as u32));
                }
            }
            let (s, e) = csr(nue, items);
            let off = tpe.len() as u32;
            tps.extend(s.iter().map(|&v| v + off));
            tpe.extend(e);
        }
        DiaAdjTables { tb1s, tb1e, tb2s, tb2e, tps, tpe }
    }
}
