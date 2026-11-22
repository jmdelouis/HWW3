#!/bin/bash
#SBATCH -J hww3_srcg
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/srcgpu_%j.log
#
# Milestone M2a: the CubeCL kernel of the source terms (hww3_wave::srce, the
# whole W3SRCE integration of a 600 s step) on the nodes of the WW3
# source-term harness (scripts/m2_src_harness_hpc.slurm), against WW3 and the
# CPU transcription, in single and double precision, then on REPLICATE
# copies of the nodes for the timing.
#   sbatch [--export=ALL,HARNESS=src_8jan_c.bin,REPLICATE=200,UNITS=128] scripts/m2_src_gpu_hpc.slurm
# Built on the login node with
#   module load cuda/13.0.0
#   CARGO_TARGET_DIR=$S/target_wave cargo build --release --bin hww3-src

S=/scratch/jmdeloui/HWW3
B=${BIN:-$S/target_wave/release/hww3-src}
H=$S/m2/harness/${HARNESS:-src_8jan_c.bin}
source /etc/profile.d/modules.sh
module load cuda/13.0.0
# CubeCL needs libdevice for exp, log, pow: the toolkit of the module
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
U=${UNITS:-128}
time $B check-gpu $H --units $U
time $B check-gpu $H --units $U --f64
time $B check-gpu $H --units $U --replicate ${REPLICATE:-200}
