#!/bin/bash
#SBATCH -J hww3_m4
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/m4_proto_%j.log
#
# Milestone M4 prototypes (docs/differentiable.md, section 10.x): the
# hand-written adjoints and the Burn autodiff versions of the DIA and of one
# geographic sub-step, with their dot-product tests against the tangent
# kernels, the times against the value kernels and the peak of the GPU
# memory of each run (nvidia-smi every 100 ms, the memory before the run
# subtracted).
#   PARTS="dia geo" (default both)
#   dia: hww3-src adj-dia on the harness of grid v5 K=4 repeated REP times
#        (default 100) for the CubeCL kernels, RB times (default 4) for Burn;
#   geo: hww3-prop adj-geo on grid v5 K=4 from the restart of 1 January
#        00:10, one sub-step of frequency IK (default 12).
#   sbatch --export=ALL,BIN=$S/opt/bin_m4 scripts/m4_proto_hpc.slurm
S=/scratch/jmdeloui/HWW3
B=${BIN:-$S/target_wave_tl/release}
H=$S/m2/harness/src_8jan_full.bin
D=$S/m2/forcing
G=$S/grids/natl_v5k4
P=$S/m2/harness/src_8jan_full.bin.params
R=$S/runs/m2_week_start_full/restart001.ww3
O=$S/m4/proto/$SLURM_JOB_ID
mkdir -p $O
source /etc/profile.d/modules.sh
module load cuda/13.0.0
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
cd $S
peak() {
  # peak TAG CMD...: run CMD with the GPU memory sampled, print the peak above the start
  local tag=$1
  shift
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > $O/mem_$tag.txt &
  local mp=$!
  sleep 1
  "$@"
  sleep 1
  kill $mp
  awk -v t=$tag 'NR == 1 {b = $1} {if ($1 > m) m = $1} END {printf "== GPU memory %s: %d MiB at the start, peak %d MiB (+%d MiB)\n", t, b, m, m - b}' $O/mem_$tag.txt
}
for part in ${PARTS:-dia geo}; do
  case $part in
    dia)
      echo "== DIA: CubeCL value, tangent, hand-written adjoint"
      peak dia_cubecl $B/hww3-src adj-dia $H --replicate ${REP:-100} --reps ${REPS:-3} --only cubecl
      echo "== DIA: CubeCL (small) and Burn autodiff"
      peak dia_burn $B/hww3-src adj-dia $H --replicate ${RB:-4} --burn-replicate ${RB:-4} --reps ${REPS:-3}
      ;;
    geo)
      FORCE="--current $D/current.ww3 --level $D/level.ww3 --ice $D/ice.ww3 --nest $D/nest.ww3 --wind $D/wind.ww3 --sources $P"
      echo "== one geographic sub-step: CubeCL value, tangent, hand-written adjoint, Burn autodiff"
      peak geo $B/hww3-prop adj-geo $G --restart $R $FORCE --f64 --ik ${IK:-12} --reps ${REPS:-20} ${GEOOPT}
      ;;
  esac
done
