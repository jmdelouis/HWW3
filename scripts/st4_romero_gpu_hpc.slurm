#!/bin/bash
#SBATCH -J hww3_romgpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -o /scratch/jmdeloui/HWW3/struct/logs/romero_gpu_%j.log
#
# GPU checks of the ST4 whitecapping SDSBCHOICE = 3 with SDSCUM > 0:
#   1. the value source kernel against the CPU transcription and WW3 on the
#      nodes of the harness of scripts/st4_romero_ref_hpc.slurm (hww3-src
#      check-gpu, single and double precision, and the default kernel on
#      the M2 harness of the HWW3 settings);
#   2. the M3 check of the tangent kernel (hww3-src check-tl, h 1e-4) on
#      the same harness, and on the M2 harness with both builds;
#   3. the default path unchanged: 2 h of grid v5 K=4 (the case dev32 of
#      opt_bench_hpc.slurm) with the build of main (BASE, built from main
#      in struct/main into struct/target_main) and this build, compared hour
#      by hour in double-precision outputs (opt_compare.py).
#   sbatch --export=ALL,TAG=iri scripts/st4_romero_gpu_hpc.slurm
S=/scratch/jmdeloui/HWW3
B=${BIN:-$S/struct/target_gpu/release}
BASE=${BASE:-$S/struct/target_main/release}
H=$S/struct/romero/${TAG:-iri}/harness.bin
J=$SLURM_JOB_ID
O=$S/struct/romero/gpu_$J
PY=$S/pyenv/bin/python
source /etc/profile.d/modules.sh
module load cuda/13.0.0
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
mkdir -p $O
echo "== check-gpu f32 (Romero harness)"
$B/hww3-src check-gpu $H 2>&1 | tail -30
echo "== check-gpu f64 (Romero harness)"
$B/hww3-src check-gpu $H --f64 2>&1 | tail -30
echo "== check-gpu f32, M2 harness, this build then BASE"
$B/hww3-src check-gpu $S/m2/harness/src_8jan_full.bin 2>&1 | tail -12
$BASE/hww3-src check-gpu $S/m2/harness/src_8jan_full.bin 2>&1 | tail -12
echo "== check-tl (Romero harness)"
$B/hww3-src check-tl $H --units 128 --h 1e-4 2>&1 | grep -vE "^\s*$"
for b in $BASE $B; do
  echo "== check-tl M2 harness, $b"
  $b/hww3-src check-tl $S/m2/harness/src_8jan_full.bin --units 128 --h 1e-4 2>&1 | grep -vE "^\s*$"
done
D=$S/m2/forcing
ARGS="$S/grids/natl_v5k4 --restart $S/m1/v5k4/restart001.ww3 --current $D/current.ww3 --level $D/level.ww3 --ice $D/ice.ww3 --nest $D/nest.ww3 --wind $D/wind.ww3 --sources $S/m2/harness/src_8jan_full.bin.params --betamax 1.75 --scale-par SDSC2:1+NLPROP:1+BJGAM:1 --hours ${HOURS:-2} --out64 --units 128"
echo "== default path, f32, BASE then this build"
$BASE/hww3-prop run $ARGS --out $O/base32 2>&1 | grep -E "steps in|rror|panicked"
$B/hww3-prop run $ARGS --out $O/new32 2>&1 | grep -E "steps in|rror|panicked"
$PY $S/scripts/opt_compare.py $O/base32 $O/new32 --prefix hs,t02
echo "== default path, f64, BASE then this build"
$BASE/hww3-prop run $ARGS --f64 --out $O/base64 2>&1 | grep -E "steps in|rror|panicked"
$B/hww3-prop run $ARGS --f64 --out $O/new64 2>&1 | grep -E "steps in|rror|panicked"
$PY $S/scripts/opt_compare.py $O/base64 $O/new64 --prefix hs,t02
