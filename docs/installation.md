# Installation

## Grid generator (`hww3-grid`, Rust)

Requirements: a Rust toolchain (stable, edition 2021; tested with 1.8x).
Install it with [rustup](https://rustup.rs) if needed.

```bash
git clone git@github.com:jmdelouis/HWW3.git
cd HWW3/hww3-grid
cargo build --release      # binary: target/release/hww3-grid
cargo test --release       # unit and end-to-end tests (< 1 s)
```

Dependencies (from crates.io): `cdshealpix` 0.7 (HEALPix), `rayon`
(threads), `serde`/`serde_json` (configuration), `anyhow` (errors).

The number of threads is set with `RAYON_NUM_THREADS`. The North Atlantic
grid (502,690 nodes) takes about 2.5 min and 6 GB of memory on 8 cores.

## Python scripts

The data-preparation and plotting scripts in `scripts/` need Python ≥ 3.7
with `numpy`, `netCDF4` and `matplotlib` (plots only). The forcing download
needs `cdsapi` (ERA5) and a `~/.cdsapirc` key. On Datarmor the scripts run
with `/appli/anaconda/versions/4.7.12/bin/python` (grid) or a Python 3.10
virtual environment (forcing):

```bash
/appli/anaconda/versions/miniconda3-py310_23.1.0/bin/python3 -m venv $R/venv
$R/venv/bin/pip install cdsapi numpy netCDF4
```

## WAVEWATCH III

WW3 is built from the NOAA-EMC `develop` branch with the switches of
`ww3/switch_hww3`; see [WW3 setup](ww3.md) and the build script
`scripts/build_ww3.pbs`.

## Documentation

```bash
pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html          # this site
cd hww3-grid && cargo doc --no-deps --open              # Rust API
```
