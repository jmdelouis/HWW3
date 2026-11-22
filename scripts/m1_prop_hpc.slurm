#!/bin/bash
#SBATCH -J hww3_prop
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/prop_%j.log
#
# Milestone M1 of docs/differentiable.md on one A100 of the Ifremer hpc
# cluster: the propagation-only model of hww3-wave (hww3-prop, CubeCL), the
# counterpart of the Fortran runs of scripts/m1_ww3_prop_hpc.slurm:
#   MODE=prop   from restart001.ww3 of the grid v5 K=4 explicit run (8 January
#               2026 00 h) with the currents, levels, ice and boundary spectra
#               of that run (copied to $D for 8-10 January)
#   MODE=geo    the same without refraction and frequency shift
#   MODE=ideal  from the COLD restart written by ww3_strt in $S/runs/m1_ideal
#               (idealised swell), no forcing and no boundary spectra
# Hourly H_s and T_m02 in $S/m1/out_$J.
#   sbatch [--export=ALL,MODE=geo,HOURS=6,CHECK=1,NORUN=1,EXTRA='--profile'] scripts/m1_prop_hpc.slurm
# Built on the login node with
#   module load cuda/13.0.0
#   CARGO_TARGET_DIR=$S/target_wave cargo build --release

S=/scratch/jmdeloui/HWW3
D=$S/m1/v5k4
G=$S/grids/natl_v5k4
B=${BIN:-$S/target_wave/release/hww3-prop}
J=$SLURM_JOB_ID
MODE=${MODE:-prop}
source /etc/profile.d/modules.sh
module load cuda/13.0.0
# CubeCL needs libdevice for sin, cos, exp: the toolkit of the module
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK

R=$D/restart001.ww3
F="--current $D/current.ww3 --level $D/level.ww3 --ice $D/ice.ww3 --nest $D/nest.ww3"
M=""
case $MODE in
  prop) ;;
  geo) M="--no-cth --no-ck" ;;
  ideal) R=$S/runs/m1_ideal/restart.ww3; F="--start 20260108000000" ;;
  *) echo "unknown MODE $MODE"; exit 1 ;;
esac
echo "mode $MODE: restart $R, $F $M"
time $B restart-hs $G --restart $R --out $S/m1/restart_hs_$MODE.f32
[ -n "$CHECK" ] && time $B check-ktp3 $G --restart $R $F
[ -n "$CHECK" ] && time $B check-geo $G --restart $R $F
[ -n "$NORUN" ] && exit 0
time $B run $G --restart $R $F $M --hours ${HOURS:-48} --out $S/m1/out_$J ${EXTRA}
