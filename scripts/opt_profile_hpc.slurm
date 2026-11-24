#!/bin/bash
#SBATCH -J hww3_prof
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/opt_profile_%j.log
#
# Optimisation, first step: where the time goes, on one A100, for grid v5
# K=4 with the calibrated full ST4 from the restart of 8 January 00 h
# (developed seas, so that the source terms work as in a storm) and the
# forcing of $S/m2/forcing:
#  1. HOURS (default 6) of the f32 value run, the f64 value run and the f64
#     tangent (BETAMAX) with --profile (the device is synchronised after
#     each part of a step, so the timers of summary.json are exact);
#  2. NSYS_HOURS (default 2) of the f32 value run under Nsight Systems
#     (CUDA trace), with the summary of the kernels and of the memory
#     copies.
#   sbatch scripts/opt_profile_hpc.slurm
S=/scratch/jmdeloui/HWW3
D=${FORCING:-$S/m2/forcing}
G=${GRID:-$S/grids/natl_v5k4}
B=${BIN:-$S/target_wave_pre/release/hww3-prop}
P=$S/m2/harness/${PARAMS:-src_8jan_full.bin.params}
R=${RESTART:-$S/m1/v5k4/restart001.ww3}
O=$S/opt/profile_$SLURM_JOB_ID
source /etc/profile.d/modules.sh
module load cuda/13.0.0
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv
mkdir -p $O
F="--restart $R --current $D/current.ww3 --level $D/level.ww3 --ice $D/ice.ww3 --nest $D/nest.ww3 --wind $D/wind.ww3 --sources $P --units ${UNITS:-128}"
H=${HOURS:-6}
for c in "f32:" "f64:--f64" "tangent:--f64 --tangent BETAMAX"; do
  tag=${c%%:*}; opts=${c#*:}
  echo "== $tag, $H h with --profile"
  ( time $B run $G $F --hours $H --profile $opts --out $O/$tag ) 2>&1 | grep -E "steps in|real|setup" | cut -c1-400
done
echo "== Nsight Systems, f32, ${NSYS_HOURS:-2} h"
nsys profile -t cuda --stats=false -o $O/f32_nsys -f true $B run $G $F --hours ${NSYS_HOURS:-2} --out $O/f32_nsys_out > $O/nsys.log 2>&1
nsys stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum --format table $O/f32_nsys.nsys-rep 2>&1 | head -80
