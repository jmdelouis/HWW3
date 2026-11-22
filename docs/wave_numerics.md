# Numerics of the GPU model

The GPU model follows the Fortran of WW3 (`develop`, commit `dea641c`,
HWW3 switches) operation by operation: same discretisation, same
sub-cycling, same order of the steps. What changes is how the loops are
mapped to the GPU, and the added transpose.

## Geometry and boundary flags

As in WW3 for a spherical grid, coordinates are in degrees. `geom` builds
what `PDLIB_INIT` and `SET_UG_IOBP_PDLIB_INIT` build:

- counter-clockwise triangles (WW3 swaps those of negative area);
- `PDLIB_IEN`, the inward normals of the edges opposite each vertex,
  $n_1 = (y_2 - y_3,\ x_3 - x_2)$;
- `PDLIB_SI`, a third of the area of the triangles around a node;
- `IOBP` = 0 on the coast, 1 inside, 2 on the open boundary; `IOBPA` = 1
  on the open boundary, where the scheme does not update;
- `IOBDP` = 0 where the depth is below `DMIN + 1e-5` (dry);
- `IOBPD(θ, i)` = 1 inside and on the open boundary; on the coast, 1 when
  $-e_\theta$ points into one of the triangles around the node, i.e. for
  waves coming from the sea (WW3's `eDet` test).

The advection velocity of component $(f, \theta)$ at node $i$ is

$$
C = \left(\frac{c_x}{\cos\varphi_i},\ c_y\right),\qquad
c_x = F\,(c_g \cos\theta + U),\quad c_y = F\,(c_g \sin\theta + V),\quad
F = \frac{1}{\mathrm{DERA}\ \mathrm{RADIUS}},
$$

in degrees per second, multiplied by `IOBDP`. The current $(U, V)$ is only
added where `IOBP > 0`.

## Explicit N scheme

One sub-step of one component (`PDLIB_W3XYPFSN2`) reads, in WW3's
notation:

$$
\begin{aligned}
K_{qT} &= \tfrac{1}{6}\Big(\textstyle\sum_{p \in T} C_p\Big)\cdot n_q, \qquad
\mathrm{NM}_T = \frac{-1}{\min\big(-\epsilon,\ \sum_q \min(0, K_{qT})\big)},\\
\tilde U_T &= \mathrm{NM}_T \sum_{q \in T} \mathrm{FLALL}_{qT}\, U_q,\\
U_i &\leftarrow \max\Big(0,\ U_i - \mathrm{DTSI}_i \sum_{T \ni i} K^+_{iT}\,(U_i - \tilde U_T)\,(1 - \mathrm{IOBPA}_i)\Big)\ \mathrm{IOBPD}(\theta, i)\ \mathrm{IOBDP}_i,
\end{aligned}
$$

with $K^+ = \max(0, K)$, $\mathrm{DTSI}_i = \Delta t / \mathrm{ITER} / \mathrm{SI}_i$,
and $\mathrm{FLALL}_{qT} = K^+_{qT}$ plus a Simpson integration of the
fluxes along the two edges of $T$ at vertex $q$. With a constant velocity,
$\mathrm{NM}_T \sum_q \mathrm{FLALL}_{qT} = 1$, so a uniform field is
unchanged at the interior nodes; a unit test checks it.

### Sub-cycling

Each component $(f, \theta)$ has its own number of sub-steps (the `LCALC`
branch of `PDLIB_W3XYPFSN2`):

$$
\mathrm{ITER}(f, \theta) = \Big\lceil \Delta t \max_i \frac{\sum_{T \ni i} K^+_{iT}}{\mathrm{SI}_i} \Big\rceil,
$$

the maximum being over every wet node, coast and open boundary included.
WW3 takes the interior nodes (`IOBP = 1`) only when `FSBCCFL` is false;
`FSBCCFL` comes from `UGBCCFL` in the `UNST` namelist of `ww3_grid`, which
is true by default and in the HWW3 runs. Counting the interior nodes only
(`--no-bccfl`) leaves the coast nodes in a single triangle at local CFL
numbers of 2.5–4.8 on grid v5 $K=4$, where the scheme is no longer
positive, and needs 3.5 times fewer sweeps per step on that grid. With
currents or varying water levels, `ITER` is recomputed at every step, as
WW3 does.

### From loops to kernels

WW3 loops over the triangles and scatters into the nodes. A GPU scatter
needs atomics, so every kernel here is a **gather**: one unit computes one
(node, direction) value from the triangles around its node, listed by a
node → triangle table. The state of one frequency is laid out as
`u[i nth + k]`, the 36 directions of a node being contiguous.

| Variant | What one sub-step does |
|---|---|
| `fused` | one unit per (node, direction); recomputes $K$, NM and FLALL of the about six triangles around the node |
| `two_phase` | for fixed velocities, $K$ and the Simpson part of FLALL are linear in $(\cos\theta, \sin\theta)$: twelve coefficients per triangle and frequency are computed once; one pass over (triangle, direction) gives $\tilde U_T$, then one gather over (node, direction) updates the state |
| `two_phase_x4` | the same with four directions per unit, so that the triangle decoding and its coefficients are loaded once for four directions |

`two_phase_x4` is the fastest: 0.93 ms per frequency sweep on grid v4 with
CubeCL. A launch is one sub-step of all directions of one frequency (a
*frequency sweep*); a direction whose `ITER` is used up is copied, so one
600 s step of frequency $f$ takes $\max_\theta \mathrm{ITER}(f, \theta)$
sweeps.

## Transpose and automatic differentiation

Without the clip, which is the identity on positive states, a sub-step is
linear: $y = m\,\big(x - d \sum_T K^+ (x_i - \sum_q W_q x_q)\big)$ with
$d = (1 - \mathrm{IOBPA})\,\mathrm{DTSI}$, $W = \mathrm{NM}\,\mathrm{FLALL}$
and the mask $m = \mathrm{IOBPD}\,\mathrm{IOBDP}$. Its transpose is again a
gather over the triangles around node $j$:

$$
z_j = m_j \bar y_j - \sum_{T \ni j} \Big[ K^+_{jT} v_j - W_{jT} \sum_{i \in T} K^+_{iT} v_i \Big],
\qquad v_i = m_i\, d_i\, \bar y_i .
$$

The transpose of a 600 s step applies the transposed sub-steps in reverse
order. Every backend passes the dot-product test
$\langle L\,\delta x, \bar y\rangle = \langle \delta x, L^\mathsf{T} \bar y\rangle$,
and so does a CPU implementation of the same formula in the unit tests.

In Burn, a sub-step is a custom operation of the autodiff tape
(`burn_ext::substep`) whose backward is the transpose kernel: for a loss
$\sum L(x)\,\bar y$, `backward()` returns $L^\mathsf{T} \bar y$, bit for bit
the output of the transpose kernel. The adjoint of the full step will also
need the clip mask, which can be rebuilt from the stored forward state
(milestone M3).

## Time step of the propagation-only model

`model` follows `W3WAVE` with the source terms off (`RUN%FLSOU = F`). One
global step of 600 s ending at `TIME`:

1. currents interpolated in magnitude and direction between their two
   records (`W3UCUR`), and their gradients (`UG_GRADIENTS`);
2. boundary spectra read when `TIME` passes the next record (`W3IOBC`,
   `W3UBPT`) and imposed (`APPLY_BOUNDARY_CONDITION_VA`);
3. ice (`W3UICE`) and water levels (`W3ULEV`) updated when their next
   field is within half a field interval; `W3ULEV` recomputes the depth,
   the wavenumbers and group speeds (`WAVNU3`) and remaps the spectra;
4. `ITER` of every component;
5. `W3KTP3` `ITLOCH` times, the geographic propagation of every component
   (`PDLIB_W3XYPUG`), then `W3KTP3` the remaining `NTLOC - ITLOCH` times,
   `NTLOC` being the number of spectral sub-steps of 300 s in the step.

The forcing files are read as `ww3_shel` reads them: currents are kept
between the two records around the time, levels and ice take the first
record after the start of the call. The Copernicus Marine hourly fields of
the HWW3 runs are stamped at hh:30, so the levels switch on the hour.

### Spectral propagation

`W3KTP3` advects the spectrum of each node in direction (refraction, from
depth gradients, currents and the great-circle term) and in wavenumber
(frequency shift, from currents and depth changes), with the ULTIMATE
QUICKEST schemes `W3QCK1` (periodic in direction) and `W3QCK2` (variable
grid in wavenumber). The parity of the step counter sets which of the two
comes first, as in WW3. On the GPU one unit handles one line of the
spectrum, staged through shared memory; `cpu::ktp3` keeps WW3's storage
(`VQ` with ghost bins, `MAPTH2`, `MAPWN2`) to check it.

### Layouts on the device

- action spectra `va[(ik np + i) nth + ith]`, frequency-major, so one
  frequency is the contiguous `[np, nth]` state of the N-scheme kernels;
- per-node and per-frequency arrays (`wn`, `cg`) as `[ik np + i]` for
  `ik = 0 .. NK+1`, ghost bins included;
- node flags: interior (`IOBP = 1`), wet (`IOBDP = 1`), sea point
  (`MAPSTA = 1`), not coast (`IOBP > 0`), output (`MAPSTA > 0`).

The propagated quantity of `PDLIB_W3XYPUG` is
$\mathrm{AC} = \mathrm{VA}\,\cos\varphi / c_g$, converted before and after
the geographic step.
