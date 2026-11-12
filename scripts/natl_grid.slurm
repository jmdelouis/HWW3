#!/bin/bash
#SBATCH -J hww3_grid
#SBATCH -p cpu
#SBATCH -c 32
#SBATCH --mem=100G
#SBATCH -t 2:00:00
#SBATCH -o logs/grid_%j.log
# Build the North Atlantic HEALPix grid on Datarmor.
# Run from /work/jmdeloui/HWW3: sbatch scripts/natl_grid.slurm
set -e
cd /work/jmdeloui/HWW3
mkdir -p grids logs
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
if [ ! -f data/gebco/natl/meta.json ]; then
  /usr/bin/time -v /work/jmdeloui/highway/venv/bin/python scripts/prep_gebco.py \
    data/gebco/GEBCO_2024.nc data/gebco/natl -100 45 0 80 --margin 3
fi
export RUSTUP_HOME=$HOME/stl-mpi/rustup CARGO_HOME=$HOME/stl-mpi/cargo
source $HOME/stl-mpi/cargo/env
export CARGO_TARGET_DIR=/work/jmdeloui/HWW3/target
cargo build --release --manifest-path hww3-grid/Cargo.toml
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
/usr/bin/time -v target/release/hww3-grid configs/natl.json > grids/natl_summary.json
# plotting subsets (numpy only; figures are made on the workstation)
target/release/hww3-grid cells grids/natl
bash scripts/natl_extract.sh
