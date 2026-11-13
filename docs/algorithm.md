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
2. **Boundary trimming.** A boundary triangle (one of its edges belongs to
   no other triangle) whose angles are outside
   [`min_angle_deg`, `max_angle_deg`] is removed; this exposes new boundary
   edges, so the test is repeated, and pinch nodes created on the way are
   removed. Interior triangles are never removed, so no hole is created.

Flips do not change the nodes, the boundary or the topology (Euler
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

## 10. Validation

The final mesh is checked (`validate::check`): non-manifold edges,
inconsistent orientation, clockwise triangles, pinch nodes, isolated nodes,
Euler characteristic $V - E + F$ (1 for a disc, $1 - h$ for $h$ holes),
angle statistics and the largest level jump across an edge. See
[Validation](validation.md).
