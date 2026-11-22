#!/bin/bash
#SBATCH -J hww3_wave_m0
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/wave_m0_%j.log
#
# Milestone M0 of docs/differentiable.md on one A100 of the Ifremer hpc
# cluster: N-scheme sweep and transpose with the CubeCL kernels and Burn
# (hww3-wave, default feature), on the Channel-Biscay box of grid v4, on
# grid v4 and on grid v5 K=4. Built on the login node with
#   module load cuda/13.0.0
#   cargo build --release            (CARGO_TARGET_DIR=$S/target_wave)
# CubeCL 0.11 needs glibc >= 2.34 to build (cubecl-llvm), which Datarmor lacks.

S=/scratch/jmdeloui/HWW3
source /etc/profile.d/modules.sh
module load cuda/13.0.0
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.memory --format=csv
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
B=$S/target_wave/release/hww3-wave
G=$S/grids
R=$S/results
mkdir -p $R
J=$SLURM_JOB_ID

time $B bench $G/natl_v4 --box -13 2.5 42.5 51.5 --reps 400 --f64 --reference --burn \
    --out $R/wave_m0_box_v4_$J.json
time $B bench $G/natl_v4 --reps 200 --f64 --reference --burn --out $R/wave_m0_v4_$J.json
time $B bench $G/natl_v5k4 --reps 200 --f64 --reference --out $R/wave_m0_v5k4_$J.json
