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
# precision (hww3-prop run --tangent NAME --out64, NAME = WIND, BETAMAX,
# SDSC2, NLPROP, BJGAM or INIT, for a relative change) over HOURS from the
# restart of 8 January 2026 00 h of grid v5 K=4 with the calibrated full ST4
# (without the source terms with NOSRC=1, INIT only) and the forcing of
# $S/m2/forcing, in $S/m3/tl_$SLURM_JOB_ID/tangent. With H (relative steps
# separated by ':'), the same run with NAME multiplied by 1 + h and 1 - h
# for each step h (plus_h, minus_h), then scripts/m3_tl_compare.py compares
# the tangent with their central differences (total energy and hourly H_s,
# with two steps only on the nodes where they agree to 1e-4). BETAMAX
# enters the stress tables, so its perturbed runs read the parameter files
# PLUS_PARAMS and MINUS_PARAMS of perturbed mod_def.ww3 files
# (scripts/m2_moddef_hpc.slurm), for one step.
#   sbatch --export=ALL,NAME=WIND,HOURS=3,H=1e-4:1e-5 scripts/m3_tl_run_hpc.slurm
#   sbatch --export=ALL,NAME=INIT,NOSRC=1,HOURS=3,H=1e-4:1e-5 scripts/m3_tl_run_hpc.slurm
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
F="--current $D/current.ww3 --level $D/level.ww3 --ice $D/ice.ww3 --nest $D/nest.ww3 --units ${UNITS:-128} --hours ${HOURS:-48} --f64 --out64"
SRC="--wind $D/wind.ww3 --sources $P"
[ -n "$NOSRC" ] && SRC=""
set -e
mkdir -p $O
echo "tangent $NAME, restart $R, $F $SRC"
time $B run $G --restart $R $F $SRC --tangent $NAME --out $O/tangent
PAIRS=""
for h in ${H//:/ }; do
  up=$(awk "BEGIN { printf \"%.12g\", 1 + $h }")
  dn=$(awk "BEGIN { printf \"%.12g\", 1 - $h }")
  case $NAME in
    WIND) PO="$SRC --wind-factor $up"; MO="$SRC --wind-factor $dn" ;;
    INIT) PO="$SRC --init-factor $up"; MO="$SRC --init-factor $dn" ;;
    BETAMAX) PO="--wind $D/wind.ww3 --sources $S/m2/harness/$PLUS_PARAMS"; MO="--wind $D/wind.ww3 --sources $S/m2/harness/$MINUS_PARAMS" ;;
    *) PO="$SRC --scale-par $NAME:$up"; MO="$SRC --scale-par $NAME:$dn" ;;
  esac
  time $B run $G --restart $R $F $PO --out $O/plus_$h
  time $B run $G --restart $R $F $MO --out $O/minus_$h
  PAIRS="$PAIRS $h:$O/plus_$h:$O/minus_$h"
done
if [ -n "$PAIRS" ]; then
  $S/pyenv/bin/python $S/scripts/m3_tl_compare.py $O/tangent $PAIRS --json $O/compare.json
fi
