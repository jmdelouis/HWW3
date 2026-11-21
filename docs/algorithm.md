# Algorithm

`hww3-grid` turns a bathymetry and two mesh-size criteria into
a WW3 triangle mesh whose nodes are HEALPix cell centres. The steps below
are those of `pipeline::build` in the Rust crate; each one is timed and
reported in the JSON summary.

## 1. Required cell size from the bathymetry

The cell size is set by the depth $h$ and the slope $|\nabla h|$, not by
the distance to the coast, with the mesh-size functions of OceanMesh2D
(Roberts et al., 2019, GMD 12, 1847; options `wl`, `slp`, `fl` and `g` of
its `edgefx`), the reference generator of unstructured coastal and wave
meshes:

$$ \Delta(h, \nabla h) = \min\Big( \frac{T_{M2}}{\alpha_{wl}}\sqrt{g h},\;
   \frac{2\pi h}{\alpha_{slp}\,|\nabla \tilde h|} \Big), $$

* **wavelength** (`wl` $=\alpha_{wl}$): $\alpha_{wl}$ cells per
  shallow-water M2 wavelength ($T_{M2}$ = 12.42 h), so the resolution follows
  $\sqrt h$: finest in shallow water, coarsest over the abyssal plain
  (OceanMesh2D suggests 25 to 100; `wave_time_s` sets $T_{M2}/\alpha_{wl}$
  directly);
* **slope** (`slp` $=\alpha_{slp}$): $\alpha_{slp}$ cells per topographic
  length scale (OceanMesh2D suggests 10 to 30), optionally restricted to
  $h <$ `slope_max_depth_m` (OceanMesh2D depth bands): deeper than about half
  a wavelength, wind waves do not feel the bottom;
* **filter** (`fl`): $\tilde h$ is $h$ low-pass filtered before the
  gradient: `fl = 0` no filter (OceanMesh2D default), `fl > 0` a cut-off
  wavelength of `fl` km, `fl < 0` the barotropic Rossby radius
  $\sqrt{gh}/|f|$ divided by $|$`fl`$|$ ($\sqrt{\sqrt{gh}/2\beta}$ within 5° of
  the equator, at most 10,000 km);
* **grade** (`grade`): the size field is finally limited by
  $\Delta(x_j) \le \Delta(x_i) + g\,|x_i - x_j|$ (OceanMesh2D default
  $g = 0.2$), so fine areas are surrounded by a gradual transition.

$h$ is floored at `min_depth_m` (1 m, as in OceanMesh2D).

The bathymetry is averaged on blocks of `smooth_km` (by default half the
finest cell size, so each finest cell holds about four blocks); this is the
structured grid of spacing $h_0$ on which OceanMesh2D evaluates its size
functions. Land samples count as `min_depth_m`, so coastal blocks are
shallow and refined by the wavelength criterion. The low pass is a box mean
of width one cut-off wavelength (summed-area table); $\nabla\tilde h$ is the
centred difference between neighbouring blocks (metric
$R\cos\varphi\,\delta\lambda$, $R\,\delta\varphi$). The grade limit is applied
by forward and backward sweeps over the 8 neighbours of every block. A wet
block then requires the coarsest depth $d$ whose cell size
$R\sqrt{\pi/3}/2^d$ is at most $\Delta$, within $[d_{\min}, d_{\max}]$, and
every cell that contains the block centre (hashed on the authalic sphere) at
a depth below that one is marked for splitting.

The cell sizes are 50.9, 25.5, 12.7, 6.4 and 3.2 km at nside 128 to 2048.
Unlike OceanMesh2D, the sizes are quantised to powers of two and the 2:1
balance lets neighbouring cells differ by a factor of 2; the grade limit
widens the transition zones but cannot make a single jump smoother.

## 2. Refinement

The roots are the cells of depth $d_{\min}$ (`nside_min = 2^{d_min}`) whose
centre lies in the domain plus three cell widths. A cell of depth $d$ is
split into its four NESTED children when $d < d_{\max}$ and it was marked
in step 1.

Optionally (`refine` rules, empty by default), a cell of angular radius $r$
(largest centre–vertex angle) is also split when some rule $(D_k, n_k)$
satisfies

$$ \operatorname{dist}(\text{centre}, \text{coast}) - r < D_k / R
\quad\text{and}\quad \log_2 n_k > d, $$

with $R$ the radius of the HEALPix sphere (authalic radius, 6371.007 km for
WGS84). The coastline then comes from an ESRI shapefile (GSHHS), a native
GSHHG binary file or the zero contour of the bathymetry; it is densified to
half a finest cell, thinned to one point per cell of depth `dmax + 2` and
bucketed on cells of depth `dmax - 2` for nearest-point queries. The grids
v1 and v2 used these distance rules (10, 50 and 200 km for nside 2048, 1024
and 512); grid v3 and later use the bathymetry only.

## 3. 2:1 balance

Leaves are split until no leaf has an edge or corner neighbour more than
one level finer. This guarantees that, around every cell corner, the cells
differ by at most one level, which the triangulation relies on, and it
bounds the level jump across a mesh edge to 1.

## 4. Bathymetry and wet cells

Every bathymetry sample is hashed to its leaf (at depth $d_{\max}$, then
coarsened until a leaf is found). For each leaf the program accumulates the
number of samples, the number of wet samples (elevation < 0) and the mean
wet depth. Leaves without samples take the nearest sample at their centre.
A leaf is **wet** if its centre is in the domain, its wet fraction is at
least `wet_fraction` and its mean wet depth is at least `min_depth_m`.

## 5. Triangulation from the HEALPix adjacency

Every leaf registers the 8 points of its boundary that a neighbour can
share: its 4 corners and its 4 edge midpoints (corners of its children).
Identical points are merged with quantised keys. Around every point, the
registered leaves sorted by angle (in the gnomonic projection on the
tangent plane at the point) form a polygon of cell centres. Each maximal run
of consecutive wet leaves with at least three members is triangulated by
**ear clipping**, choosing at each step the ear with the largest minimum
angle.

Two leaves sharing a boundary segment $[p, q]$ appear consecutively around
both $p$ and $q$, so their connecting edge belongs to the polygons of $p$
and $q$ only. The mesh is therefore **conforming**, including at the
coarse/fine transitions, without any extra node.

```{figure} _static/fig/natl_brittany.png
:width: 80%

HEALPix cells (blue) and WW3 triangles (red) off Brittany, across the
transitions nside 512 → 1024 → 2048.
```

## 6. Pinch nodes

Where two wet regions touch at a single cell corner, a node can be shared
by two triangle fans that have no edge in common (a "bow-tie"). WW3
(`w3triamd`) stops on such nodes ("We have a dramatic inconsistency"). At
every node with several fans, the largest fan is kept and the triangles of
the others are dropped; this is repeated until every node has at most one
fan. On the North Atlantic grid, 671 triangles (0.07 %) are dropped.

## 7. Triangle quality

The nodes are fixed (they are the HEALPix centres), so the angles are
improved by changing the connectivity only:

1. **Edge flips** (Lawson's algorithm on the sphere). An interior edge
   shared by triangles $(u, v, w)$ and $(v, u, x)$ is replaced by $(w, x)$
   when the quadrilateral $u\,x\,v\,w$ is convex, $(w, x)$ is not already
   an edge, the level jump between $w$ and $x$ is at most
   `quality.max_level_jump`, and the smaller minimum angle of the two
   triangles increases. Passes are repeated until no edge flips. This
   max-min angle criterion converges to the Delaunay triangulation of the
   nodes constrained by the boundary, which maximises the smallest angle.
2. **Alternated diagonals** (`quality.alternate`). In a uniform area, the
   four cells around a HEALPix corner form a 2×2 block of pixels
   $(i_0, j_0)$…$(i_0+1, j_0+1)$ of one base face, and the two triangles
   of the block share one of its diagonals. Choosing everywhere the same
   diagonal gives triangles that all have the same orientation, which
   makes the numerical diffusion of the WW3 schemes anisotropic. The block
   diagonal is therefore taken as $(i_0, j_0)$–$(i_0+1, j_0+1)$ when $j_0$
   is even and $(i_0+1, j_0)$–$(i_0, j_0+1)$ when $j_0$ is odd: the
   diagonals alternate from one row of blocks to the next (chevrons) and
   every node keeps six neighbours. On a square lattice this pattern gives
   an explicit time step 27 % longer than a single diagonal direction
   ($0.895\,\Delta x/c$ against $0.707$), whereas a checkerboard
   ("union jack") would shorten it ($0.667$). Only nearly square blocks
   are changed, whose diagonals differ by at most `alternate_max_ratio`
   (1.2): HEALPix blocks are rhombi away from the equator, and their short
   diagonal already gives nearly equilateral triangles. The new triangles
   must keep their angles within the bounds.
3. **CFL flips** (`quality.cfl_flips`). For every interior node the
   largest explicit time step of the WW3 propagation schemes is computed
   as in `w3profsmd.F90` (section 10). Around each node whose CFL length is
   below `cfl_threshold` (0.8) cell sizes, an edge is flipped when the
   quadrilateral is convex, the angles stay within the bounds, the level
   jump across the new edge is at most `max_level_jump`, and the smallest
   CFL length of the four nodes increases by more than 0.1 %. This repairs
   the junctions between alternated and non-alternated blocks and the
   level transitions, where the explicit step is shortest.
4. **Boundary trimming.** A boundary triangle (one of its edges belongs to
   no other triangle) whose angles are outside
   [`min_angle_deg`, `max_angle_deg`] is removed; this exposes new boundary
   edges, so the test is repeated, and pinch nodes created on the way are
   removed. Interior triangles are never removed, so no hole is created.

None of these steps moves a node: every node stays the centre of its
HEALPix pixel, so the model output remains a HEALPix map. Flips do not
change the nodes, the boundary or the topology (Euler
characteristic). With `quality.enforce`, the program fails if an angle is
still outside the range; the bounds 20°–130° are the defaults.

On the North Atlantic grid the raw triangulation has angles from 14.1° to
139.7° (3,796 triangles below 20°, 3,506 above 120°). The quality step makes
14,750 flips and removes 1,275 boundary triangles (282 nodes); the final
angles are 20.7°–127.7°, with no triangle below 20° and 3 interior
triangles between 120° and 127.7°. These 3 cannot be improved by flips
(non-convex quadrilateral or no gain) and are not on the boundary; allowing
a level jump of 2 across flipped edges does not change the result. Moving
them below 120° would require moving nodes, which would break the identity
node = HEALPix pixel centre, hence the 130° default.

## 8. Connected components and open boundary

Only the largest connected set of triangles is kept (other basins, lakes
and isolated cells are dropped). Wet nodes that share a cell corner with a
cell outside the domain are **open-boundary nodes**, written to
`<name>.obc` and used by WW3 for boundary spectra.

## 9. Ellipsoid

HEALPix is defined on a sphere. With `"ellipsoid": "wgs84"`, every input
latitude $\varphi$ (bathymetry, coastline, domain) is converted to the
authalic latitude $\beta$, which maps the WGS84 ellipsoid to the sphere of
the same area with equal-area cells:

$$ \beta = \arcsin\frac{q(\varphi)}{q(\pi/2)}, \qquad
q(\varphi) = (1-e^2)\left[\frac{\sin\varphi}{1-e^2\sin^2\varphi}
 - \frac{1}{2e}\ln\frac{1-e\sin\varphi}{1+e\sin\varphi}\right]. $$

The node latitudes are converted back to geodetic latitudes (series to
$e^6$ plus two Newton iterations, round-trip error < $10^{-12}$ rad) for
WW3. The difference reaches 0.128° at 45°N.

## 10. Explicit time step

WW3 limits its explicit unstructured schemes (N, PSI, FCT; `w3profsmd.F90`,
routines `W3XYPFSN2` and `W3CFLUG`) with a CFL condition computed on the
triangles. For a triangle with vertices 1, 2, 3, the flux coefficients are
$K_k = \boldsymbol\lambda\cdot\mathbf n_k$, with $\boldsymbol\lambda$ half the
mean of the three nodal advection velocities and $\mathbf n_k$ the inward
normal of the edge opposite vertex $k$, whose length is that edge (`IEN`).
For each spectral component the largest stable step at node $i$ is

$$ \Delta t_i = S_i \Big/ \sum_{T\ni i}\max(0, K_{i,T}), $$

with $S_i$ a third of the area of the triangles around $i$ (`SI`). The
global step is the minimum over the interior nodes (open and coastal
boundary nodes are left out unless `FSBCCFL`), and each component is
sub-cycled `ceil(DT / min Δt_i)` times. For a speed $c$ in direction
$\theta$, $\Delta t_i = L_i(\theta)/c$ with a CFL length $L_i(\theta)$ that
depends only on the triangles. `hww3-grid` computes $L_i$, the minimum over
the model directions, and $\Delta t_i$ with the WW3 formula above for the
linear group speed of the lowest model frequency (`freq1`, the fastest
component), each triangle using the mean speed of its three vertices as
WW3 does, currents excluded (grid v4 and earlier reported $L_i/c_g$ with
the speed at the node; their triangles do not depend on it, since the CFL
flips use $L_i$ only). The summary (`cfl`) gives the smallest
step, the node that sets it, the number of sub-cycles at `dtxy` and, per
nside, the minimum, 1 % quantile and median of $\Delta t_i$ and of
$L_i$ divided by the cell size (the efficiency); `raw/dt_explicit.f64` and
`raw/cfl_length.f64` hold the values per node.

The fully implicit scheme used in the North Atlantic runs (`IMPTOTAL`) has
no such limit; these numbers matter for the explicit schemes and measure
the geometric quality of the triangles for the transport.

### Capping the sub-steps (`refine_bathy.cfl_substeps`)

Because WW3 sub-cycles every component with the smallest step of the whole
domain, a few fine coastal nodes set the cost of the explicit schemes. With
`cfl_substeps` $K > 0$, the grid is built so that the lowest frequency
needs at most $K$ sub-steps per `dtxy`, keeping the nodes on HEALPix
pixel centres:

1. **During the refinement**, a bathymetry block of depth $h$ is not
   refined below the finest level $d$ with
   $\eta\, L_d / c_g(f_1, h) \ge \mathrm{dtxy}/K$, where $L_d$ is the cell
   size and $\eta$ = `cfl_efficiency` (0.8) the expected CFL length in
   cell sizes. For the North Atlantic ($f_1=0.0373$ Hz, 600 s, nside 2048
   cells of 3.2 km) this forbids nside 2048 where $c_g > 21.2$ m/s for
   $K=5$ (depths of about 68 to 600 m) and where $c_g > 17.0$ m/s for
   $K=4$ (deeper than about 36 m).
2. **After each build**, the explicit step of every node is computed with
   the WW3 formula. For each interior node still above $K$ sub-steps the
   parent of its cell is marked as not splittable; when that parent was
   already marked (the 2:1 balance re-split it), the parents of its
   triangle neighbours of the same or a finer level are marked too. The
   grid is rebuilt from the start with these marks, at most `cfl_passes`
   times (3 by default), and the passes stop when no node is left above $K$.

The summary gives the number of capped blocks (`n_cfl_capped_blocks`),
the passes made, the merged cells (`n_cfl_merged`) and the nodes still above
$K$ (`n_cfl_above`). Each pass is a full build, so 10 passes take about
ten times as long as one.

## 11. Validation

The final mesh is checked (`validate::check`): non-manifold edges,
inconsistent orientation, clockwise triangles, pinch nodes, isolated nodes,
Euler characteristic $V - E + F$ (1 for a disc, $1 - h$ for $h$ holes),
angle statistics and the largest level jump across an edge. See
[Validation](validation.md).
