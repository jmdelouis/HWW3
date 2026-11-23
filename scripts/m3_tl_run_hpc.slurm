#!/bin/bash
#SBATCH -J hww3_tlrun
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/tlrun_%j.log
#
# Milestone M3 on one A100: the tangent of the GPU model in double
# precision (hww3-prop run --tangent NAME, NAME = WIND, BETAMAX, SDSC2,
# NLPROP or BJGAM, for a relative change) over HOURS from the restart of
# 8 January 2026 00 h of grid v5 K=4 with the calibrated full ST4 and the
# forcing of $S/m2/forcing, in $S/m3/tl_$SLURM_JOB_ID/tangent. With H, the
# same run with NAME multiplied by 1 + H and 1 - H (plus, minus) and
# scripts/m3_tl_compare.py comparing the tangent with their central
# difference (total energy and hourly H_s). BETAMAX enters the stress
# tables, so its perturbed runs read the parameter files PLUS_PARAMS and
# MINUS_PARAMS of perturbed mod_def.ww3 files (scripts/m2_moddef_hpc.slurm).
#   sbatch --export=ALL,NAME=WIND,HOURS=3,H=1e-3 scripts/m3_tl_run_hpc.slurm
#   sbatch --export=ALL,NAME=BETAMAX,HOURS=3,H=1e-3,PLUS_PARAMS=par_bp01.bin.params,MINUS_PARAMS=par_bm01.bin.params scripts/m3_tl_run_hpc.slurm
# Built on the login node with
#   module load cuda/13.0.0
#   CARGO_TARGET_DIR=$S/target_wave_m3 cargo build --release --bin hww3-prop

S=/scratch/jmdeloui/HWW3
D=${FORCING:-$S/m2/forcing}
G=${GRID:-$S/grids/natl_v5k4}
B=${BIN:-$S/target_wave_m3/release/hww3-prop}
P=$S/m2/harness/${PARAMS:-src_8jan_full.bin.params}
R=${RESTART:-$S/m1/v5k4/restart001.ww3}
O=$S/m3/tl_$SLURM_JOB_ID
source /etc/profile.d/modules.sh
module load cuda/13.0.0
# CubeCL needs libdevice for sin, cos, exp: the toolkit of the module
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
F="--current $D/current.ww3 --level $D/level.ww3 --ice $D/ice.ww3 --nest $D/nest.ww3 --wind $D/wind.ww3 --units ${UNITS:-128} --hours ${HOURS:-48} --f64"
set -e
mkdir -p $O
echo "tangent $NAME, restart $R, $F"
time $B run $G --restart $R $F --sources $P --tangent $NAME --out $O/tangent
if [ -n "$H" ]; then
  up=$(awk "BEGIN { printf \"%.12g\", 1 + $H }")
  dn=$(awk "BEGIN { printf \"%.12g\", 1 - $H }")
  case $NAME in
    WIND) PO="--sources $P --wind-factor $up"; MO="--sources $P --wind-factor $dn" ;;
    BETAMAX) PO="--sources $S/m2/harness/$PLUS_PARAMS"; MO="--sources $S/m2/harness/$MINUS_PARAMS" ;;
    *) PO="--sources $P --scale-par $NAME:$up"; MO="--sources $P --scale-par $NAME:$dn" ;;
  esac
  time $B run $G --restart $R $F $PO --out $O/plus
  time $B run $G --restart $R $F $MO --out $O/minus
  $S/pyenv/bin/python $S/scripts/m3_tl_compare.py $O/tangent $O/plus $O/minus $H $O/compare.json
fi
