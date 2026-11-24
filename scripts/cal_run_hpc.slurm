#!/bin/bash
#SBATCH -J hww3_cal
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/cal_%j.log
#
# Calibration of the ST4 parameters (docs/differentiable.md, section 9.5):
# one run of the GPU model with the calibrated full ST4 on grid v5 K=4 and
# the forcing of $S/m2/forcing, with BETAMAX (stress tables computed by
# the Rust port; unset: those of the parameter file, from ww3_grid) and
# SDSC2, NLPROP, BJGAM multiplied by FSDSC2,
# FNLPROP, FBJGAM, from RESTART for HOURS, hourly H_s and T_m02 in double
# precision to $S/cal/$TAG; with TANGENT=NAME, the tangent run in double
# precision for a relative change of NAME (dhs_, dt02_ files); with F64=1
# a value run in double precision. BETAU=KNOTS:COEFS makes BETAMAX depend
# on the local wind speed (BETAMAX(U), hww3-prop --betau, tables in BETAMAX
# at BETAU_GRID=B0:DB:N), with / between the values since sbatch --export
# splits at commas, and TANGENT=BETAU<k> is the tangent for its
# coefficient k.
#   sbatch --export=ALL,TAG=bu0_BETAU3,BETAMAX=1.75,BETAU=2/5/8/11/14/17/20:,HOURS=156,TANGENT=BETAU3 scripts/cal_run_hpc.slurm
#   sbatch --export=ALL,TAG=it0_value,BETAMAX=1.75,HOURS=216 scripts/cal_run_hpc.slurm
#   sbatch --export=ALL,TAG=it0_SDSC2,BETAMAX=1.75,HOURS=120,TANGENT=SDSC2 scripts/cal_run_hpc.slurm
S=/scratch/jmdeloui/HWW3
D=${FORCING:-$S/m2/forcing}
G=${GRID:-$S/grids/natl_v5k4}
B=${BIN:-$S/target_wave_m3/release/hww3-prop}
P=$S/m2/harness/${PARAMS:-src_8jan_full.bin.params}
R=${RESTART:-$S/runs/m2_week_start_full/restart001.ww3}
O=$S/cal/$TAG
source /etc/profile.d/modules.sh
module load cuda/13.0.0
# CubeCL needs libdevice for sin, cos, exp: the toolkit of the module
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
PAR="${BETAMAX:+--betamax $BETAMAX} --scale-par SDSC2:${FSDSC2:-1}+NLPROP:${FNLPROP:-1}+BJGAM:${FBJGAM:-1}"
[ -n "$FWIND" ] && PAR="$PAR --wind-factor $FWIND"
[ -n "$BETAU" ] && PAR="$PAR --betau $(echo $BETAU | tr / ,) ${BETAU_GRID:+--betau-grid $BETAU_GRID}"
PREC=""
if [ -n "$TANGENT" ] || [ -n "$F64" ]; then PREC="--f64"; fi
mkdir -p $O
echo "$TAG: $PAR ${TANGENT:+tangent $TANGENT} $PREC, restart $R, $HOURS h"
time $B run $G --restart $R --current $D/current.ww3 --level $D/level.ww3 --ice $D/ice.ww3 --nest $D/nest.ww3 --wind $D/wind.ww3 \
  --sources $P --units ${UNITS:-128} --hours ${HOURS:-120} $PAR ${TANGENT:+--tangent $TANGENT} $PREC --out64 --out $O
