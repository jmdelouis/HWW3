#!/bin/bash
#SBATCH -J hww3_allrust
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 00:25:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/pre_allrust_%j.log
#
# A GPU run of grid v5 K=4 whose inputs all come from Rust, with no WW3
# tool: the parameter file of the source terms from the namelists and the
# grid (hww3-src build-params), the forcing and boundary files of
# hww3-pre prnc and bounc ($S/pre/rust, scripts/pre_inputs_hpc.slurm), and
# the start from calm conditions on 1 January 00 h (hww3-prop --restart
# calm), for HOURS, hourly H_s and T_m02 in double precision to
# $S/pre/allrust. scripts/pre_allrust_cmp_hpc.slurm compares it with the
# calibration's start run ($S/cal/it0_value: WW3's restart after the first
# step, the harness parameter file).
#   sbatch scripts/pre_allrust_hpc.slurm
S=/scratch/jmdeloui/HWW3
B=${BIN:-$S/target_wave_pre/release}
G=$S/grids/natl_v5k4
P=$S/pre/rust
O=$S/pre/allrust
N=$S/runs/test_grid_v5k4
source /etc/profile.d/modules.sh
module load cuda/13.0.0
# CubeCL needs libdevice for sin, cos, exp: the toolkit of the module
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
mkdir -p $O
time $B/hww3-src build-params $G $N/namelists_natl.nml $P/full.params --grid-nml $N/ww3_grid.nml --check $S/m2/harness/src_8jan_full.bin.params | tail -4
time $B/hww3-prop run $G --restart calm --start 20260101000000 --current $P/current.ww3 --level $P/level.ww3 --ice $P/ice.ww3 \
  --nest $P/nest.ww3 --wind $P/wind.ww3 --sources $P/full.params --units 128 --hours ${HOURS:-216} --out64 --out $O
