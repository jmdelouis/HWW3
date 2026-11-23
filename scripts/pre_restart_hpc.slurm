#!/bin/bash
#SBATCH -J hww3_pre
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/pre_restart_%j.log
#
# Initial state in Rust (docs/wave_results.md, "Preprocessing in Rust"), on
# one A100 with the calibrated full ST4 on grid v5 K=4 and the forcing of
# $S/m2/forcing:
# 1. check-steps from calm conditions on 1 January 00 h (hww3-prop --restart
#    calm, WW3's start without a restart file) against the WW3 restart
#    written after the first step (runs/m2_week_start_full, 00:10);
# 2. a GPU restart: the run from calm to 01 h in one piece, and in two
#    pieces with a restart written by hww3-prop at 00:10 (--restart-out)
#    and read back (--itstep0 from the first piece); H_s and T_m02 at 01 h
#    of the two must be identical.
#   sbatch scripts/pre_restart_hpc.slurm
S=/scratch/jmdeloui/HWW3
D=${FORCING:-$S/m2/forcing}
G=${GRID:-$S/grids/natl_v5k4}
B=${BIN:-$S/target_wave_pre/release/hww3-prop}
P=$S/m2/harness/${PARAMS:-src_8jan_full.bin.params}
O=$S/pre/restart_$SLURM_JOB_ID
source /etc/profile.d/modules.sh
module load cuda/13.0.0
# CubeCL needs libdevice for sin, cos, exp: the toolkit of the module
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
mkdir -p $O
F="--current $D/current.ww3 --level $D/level.ww3 --ice $D/ice.ww3 --nest $D/nest.ww3 --wind $D/wind.ww3 --sources $P --units ${UNITS:-128}"
T0=20260101000000
echo "== 1. calm start against WW3"
time $B check-steps $G --restart calm --start $T0 $F --ref $S/runs/m2_week_start_full/restart001.ww3
echo "== 2. one piece"
time $B run $G --restart calm --start $T0 $F --stop 20260101010000 --out $O/one
echo "== 2. two pieces"
time $B run $G --restart calm --start $T0 $F --stop 20260101001000 --out $O/a --restart-out $O/r0010.ww3 | tee $O/a.log
IT=$(sed -n 's/.*continue with --itstep0 \([0-9]*\).*/\1/p' $O/a.log)
time $B run $G --restart $O/r0010.ww3 --itstep0 $IT $F --stop 20260101010000 --out $O/b
for f in hs_2026010101.f32 t02_2026010101.f32; do
  if cmp -s $O/one/$f $O/b/$f; then echo "$f: identical"; else
    $S/pyenv/bin/python -c "
import numpy as np, sys
a = np.fromfile('$O/one/$f', '<f4'); b = np.fromfile('$O/b/$f', '<f4')
d = np.abs(a - b); print('$f: %d nodes differ, max |diff| %.3e at node %d' % ((d > 0).sum(), d.max(), d.argmax()))"
  fi
done
