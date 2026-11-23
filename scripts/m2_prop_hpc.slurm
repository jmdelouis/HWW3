#!/bin/bash
#SBATCH -J hww3_m2gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/m2gpu_%j.log
#
# Milestone M2a on one A100 of the Ifremer hpc cluster: hww3-prop with the
# source terms of the simplified ST4 (SINTABLE = 0, SDSCUM = 0), the GPU
# counterpart of the WW3 references of scripts/m2_ww3_ref_hpc.slurm, from
# restart001.ww3 of the grid v5 K=4 explicit run (8 January 2026 00 h) with
# the wind, currents, levels, ice and boundary spectra of $S/m2/forcing and
# the source parameters written by the harness from the mod_def.ww3 of the
# reference (scripts/m2_src_harness_hpc.slurm, PARAMS).
# With STEPS_REF (WW3 restarts written at the end of some steps, separated
# by ':'), hww3-prop check-steps compares the spectra after those steps;
# otherwise HOURS of hourly H_s and T_m02 go to $S/m2/out_$J.
#   sbatch [--export=ALL,STEPS_REF=$S/runs/m2_dev/restart001.ww3:...,HOURS=48,UNITS=128] scripts/m2_prop_hpc.slurm
# Built on the login node with
#   module load cuda/13.0.0
#   CARGO_TARGET_DIR=$S/target_wave cargo build --release --bin hww3-prop

S=/scratch/jmdeloui/HWW3
# forcing and grid (FORCING=$S/m2/forcing_v4 GRID=$S/grids/natl_v4 for grid v4)
D=${FORCING:-$S/m2/forcing}
G=${GRID:-$S/grids/natl_v5k4}
B=${BIN:-$S/target_wave/release/hww3-prop}
J=$SLURM_JOB_ID
P=$S/m2/harness/${PARAMS:-src_8jan_c.bin.params}
R=${RESTART:-$S/m1/v5k4/restart001.ww3}
source /etc/profile.d/modules.sh
module load cuda/13.0.0
# CubeCL needs libdevice for sin, cos, exp: the toolkit of the module
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
F="--current $D/current.ww3 --level $D/level.ww3 --ice $D/ice.ww3 --nest $D/nest.ww3 --wind $D/wind.ww3 --sources $P --units ${UNITS:-128}"
echo "restart $R, $F"
if [ -n "$STEPS_REF" ]; then
  time $B check-steps $G --restart $R $F --ref ${STEPS_REF//:/,} ${NODES:+--nodes ${NODES//+/,}} $EXTRA
  exit 0
fi
time $B run $G --restart $R $F --hours ${HOURS:-48} --out $S/m2/out_$J $EXTRA
