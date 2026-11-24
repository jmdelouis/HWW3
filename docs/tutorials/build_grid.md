# Tutorial: build a grid

This tutorial builds a HEALPix grid of the Bay of Biscay with the
resolution of the North Atlantic grids (cells of 51 km over the deep ocean
down to 3.2 km on the coast), checks it and draws it.

## Prerequisites

- A Linux machine (or any machine with `bash`) with 8 cores and 8 GB of
  memory. The whole North Atlantic grid needs about 7 GB and a few
  minutes on 8 cores; this regional grid needs less.
- A Rust toolchain (stable), installed with [rustup](https://rustup.rs).
- Python 3.7 or later with `numpy`, `netCDF4` and `matplotlib`.
- About 15 GB of disk for the GEBCO 2024 bathymetry.

## 1. Get the code and build the generator

```bash
git clone git@github.com:jmdelouis/HWW3.git
cd HWW3
cargo build --release --manifest-path hww3-grid/Cargo.toml
cargo test --release --manifest-path hww3-grid/Cargo.toml
```

**Expected:** the program `hww3-grid/target/release/hww3-grid`, and every
test reported as `ok` (the tests take less than a second once built).

## 2. Get the bathymetry

```bash
bash scripts/fetch_sources.sh $PWD
```

The script (which needs `git`, `wget` and `unzip`) downloads GEBCO 2024
to `data/gebco/GEBCO_2024.nc`, and also the WW3 sources used by the next
tutorial. Skip it if you already have
`GEBCO_2024.nc`: put it, or a link to it, in `data/gebco/`.

Cut the region, with a margin of 3° around it:

```bash
python scripts/prep_gebco.py data/gebco/GEBCO_2024.nc data/gebco/biscay -10 0 43 49 --margin 3
```

**Expected:** `data/gebco/biscay/elev.f32` (the elevations) and
`data/gebco/biscay/meta.json`. Check the step of the grid in `meta.json`:
`"dlon": 0.004166666666662877`, that is 15 arc-seconds.

## 3. Write the configuration

Create `configs/biscay.json`:

```json
{
  "name": "biscay",
  "domain": { "lon_min": -10.0, "lon_max": 0.0, "lat_min": 43.0, "lat_max": 49.0 },
  "ellipsoid": "wgs84",
  "nside_min": 128,
  "nside_max": 2048,
  "max_level_span": 4,
  "refine_bathy": { "wl": 100.0, "slp": 10.0, "fl": -1.0, "grade": 0.2,
                    "slope_max_depth_m": 300.0, "min_depth_m": 1.0 },
  "min_depth_m": 5.0,
  "wet_fraction": 0.5,
  "keep_largest_component": true,
  "bathy": "../data/gebco/biscay/meta.json",
  "output_dir": "../grids/biscay"
}
```

The paths are relative to the directory of the configuration file. The
size criteria are those of the North Atlantic grid v3: 100 cells per
wavelength of the tide in shallow water (`wl`), 10 cells per length scale
of the slope above 300 m of depth (`slp`), and a growth of the cell size of
at most 0.2 per unit distance (`grade`). The keys left out (`quality`,
`ww3`) take their defaults. Every key is described in
[Configuration](../configuration.md), and how to choose them for another
region in [A grid for another region](../new_region.md).

## 4. Build the grid

```bash
mkdir -p grids
RAYON_NUM_THREADS=8 hww3-grid/target/release/hww3-grid configs/biscay.json > grids/biscay_summary.json
```

**Expected:** the command ends without an error message and writes to
`grids/biscay/`:

| File | Content |
|---|---|
| `biscay.msh` | the triangle mesh read by WW3 (Gmsh format) |
| `biscay.obc` | the open-boundary nodes |
| `ww3_grid.nml`, `namelists_biscay.nml` | the WW3 namelists |
| `raw/` | the nodes, levels, depths and triangles as binary arrays |
| `meta.json` | the configuration and the summary |

The program stops with an error if the mesh is invalid or if a triangle
angle stays outside 20°–130° (`quality.enforce`).

## 5. Check it

```bash
python - <<'EOF'
import json
s = json.load(open("grids/biscay_summary.json"))
m = s["mesh"]
print("nodes:", s["n_nodes"], " open boundary:", s["n_open_boundary"])
print("nodes per nside:", s["nodes_per_nside"])
print("angles: %.1f to %.1f deg" % (m["min_angle_deg"], m["max_angle_deg"]))
print("pinch nodes:", m["n_pinch_nodes"], " clockwise triangles:", m["n_negative"])
EOF
```

**What to look for:**

- the number of nodes grows from nside 128 to nside 2048: the finest cells
  are along the coast and on the shelf;
- open-boundary nodes exist: the western and northern edges of the box are
  at sea (edges on land give none);
- the angles lie between 20° and 130°, and there are no pinch nodes and no
  clockwise triangles (WW3 rejects pinch nodes).

For comparison, the North Atlantic grid v3, built with the same criteria
(`configs/natl_v3_datarmor.json`), has 794,613 nodes, 1,341 open-boundary
nodes and angles of 20.0°–125.9°, and takes 70 s on 8 cores.

## 6. Draw it

```bash
hww3-grid/target/release/hww3-grid cells grids/biscay
mkdir -p grids/biscay/plot
python scripts/extract_zoom.py grids/biscay grids/biscay/plot/nodes.npz --nodes
python scripts/plot_levels.py grids/biscay/plot/nodes.npz biscay_levels.png
python scripts/extract_zoom.py grids/biscay grids/biscay/plot/brittany.npz -6.5 -1.5 46.5 49
python scripts/plot_cells_tris.py grids/biscay/plot/brittany.npz biscay_brittany.png
```

**Expected:** `biscay_levels.png`, the nodes coloured by their nside, and
`biscay_brittany.png`, the HEALPix cells and the triangles around
Brittany. `grids/biscay/plot/nodes.npz` is also what the validation
scripts read (tutorial [Validate a run](validate_run.md)).

## Next

- To run WW3 on a grid: [Prepare the inputs and run WW3](run_ww3.md). The
  run scripts are written for the North Atlantic grids; for another
  region, change the forcing boxes and the boundary spectra as
  [A grid for another region](../new_region.md) explains in its section 6.
- What the files contain and how to read them in Python:
  [Outputs](../outputs.md).
