#!/bin/bash
#SBATCH -J hww3_tlsrc
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -t 00:45:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/tlsrc_%j.log
#
# Milestone M3: the tangent kernel of the source terms (hww3_wave::srce_tl,
# the whole W3SRCE integration of a 600 s step differentiated in forward
# mode) on the nodes of the WW3 source-term harness, against central
# differences of the GPU source kernel in double precision, for the
# spectrum, the wind speed, BETAMAX, SDSC2, NLPROP and BJGAM.
#   sbatch [--export=ALL,HARNESS=src_8jan_full.bin,H=1e-4] scripts/m3_tl_src_hpc.slurm
# Built on the login node with
#   module load cuda/13.0.0
#   CARGO_TARGET_DIR=$S/target_wave_m3 cargo build --release --bin hww3-src

S=/scratch/jmdeloui/HWW3
B=${BIN:-$S/target_wave_m3/release/hww3-src}
F=$S/m2/harness/${HARNESS:-src_8jan_full.bin}
source /etc/profile.d/modules.sh
module load cuda/13.0.0
# CubeCL needs libdevice for exp, log, pow: the toolkit of the module
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export RAYON_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
time $B check-tl $F --units ${UNITS:-128} --h ${H:-1e-4}
