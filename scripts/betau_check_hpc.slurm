#!/bin/bash
#SBATCH -J hww3_buchk
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 01:40:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/betau_check_%j.log
#
# Checks of BETAMAX(U) (hww3-prop --betau, docs/differentiable.md section
# 9.7) on 12 h from the start restart: (1) with all coefficients 0 the value
# run reproduces the run with BETAMAX 1.75 constant; (2) the tangent for all
# coefficients at once (BETAU, the hat functions adding up to 1) and the
# BETAMAX tangent with BETAMAX(U) reproduce the BETAMAX tangent of the
# constant run; (3) BETAMAX(U) constant at the step-3 value 1.37249382
# (all coefficients ln(1.37249382 / 1.75)) against the run with the tables
# computed for 1.37249382, the error of the interpolation in BETAMAX;
# (4) the tangent for the coefficient of the 11 m/s knot at c = 0.2 there
# against central differences of +-0.01 and +-0.02 (BETAMAX off the grid of
# the tables). Then scripts/betau_check.py and m3_tl_compare.py compare them.
# A run already finished is skipped, so the job can be submitted again after
# a time limit (12 h: about 2 min per value run in single precision, 9 min
# in double precision and 11 min per tangent, the 13 tables taking 2 min).
# With PART=tight, only the check of M3 over 3 h: the tangent for c_3 at
# 0.2 against central differences of steps 1e-4 and 1e-5, on the nodes
# where the two differences agree to 1e-6 of the node's scale. With
# PART=cut, check (2) again with the unstable interactions of the DIA left
# out of the tangents (--tl-nlcut CUT, docs/differentiable.md section 9.5,
# BIN of target_wave_tl), in $S/cal/betau_check_cutCUT.
#   sbatch scripts/betau_check_hpc.slurm
#   sbatch --export=ALL,PART=tight scripts/betau_check_hpc.slurm
#   sbatch --export=ALL,PART=cut,CUT=1,BIN=.../target_wave_tl/release/hww3-prop scripts/betau_check_hpc.slurm
S=/scratch/jmdeloui/HWW3
D=$S/m2/forcing
G=$S/grids/natl_v5k4
B=${BIN:-$S/target_wave_bu/release/hww3-prop}
P=$S/m2/harness/src_8jan_full.bin.params
R=$S/runs/m2_week_start_full/restart001.ww3
O=$S/cal/betau_check
H=${HOURS:-12}
K=2,5,8,11,14,17,20
source /etc/profile.d/modules.sh
module load cuda/13.0.0
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
run() {
  local tag=$1
  shift
  local t0=$SECONDS
  # a run that has finished (summary.json written) is not run again
  [ -f $O/$tag/summary.json ] && { echo "== $tag: done before"; return; }
  mkdir -p $O/$tag
  echo "== $tag: $*"
  $B run $G --restart $R --current $D/current.ww3 --level $D/level.ww3 --ice $D/ice.ww3 --nest $D/nest.ww3 \
    --wind $D/wind.ww3 --sources $P --units 128 --hours $H --out64 --out $O/$tag "$@" 2>&1 | grep -E "sources:|tangent:|rror|panicked"
  echo "$tag: $((SECONDS - t0)) s"
}
S3="--scale-par SDSC2:0.649237785+NLPROP:1.13718427+BJGAM:1"
if [ "$PART" = cut ]; then
  O=$S/cal/betau_check_cut$CUT
  run t_const --betamax 1.75 --tangent BETAMAX --f64 --tl-nlcut $CUT
  run t_c0_all --betamax 1.75 --betau $K: --tangent BETAU --f64 --tl-nlcut $CUT
  run t_c0_beta --betamax 1.75 --betau $K: --tangent BETAMAX --f64 --tl-nlcut $CUT
  cd $S
  $S/pyenv/bin/python scripts/betau_check.py $O
  exit 0
fi
if [ "$PART" = tight ]; then
  H=3
  run t_k3_3h --betamax 1.75 --betau $K:0,0,0,0.2,0,0,0 --tangent BETAU3 --f64
  for c in 0.2001 0.1999 0.20001 0.19999; do
    run v_k3_$c --betamax 1.75 --betau $K:0,0,0,$c,0,0,0 --f64
  done
  cd $S
  $S/pyenv/bin/python scripts/m3_tl_compare.py $O/t_k3_3h 0.0001:$O/v_k3_0.2001:$O/v_k3_0.1999 0.00001:$O/v_k3_0.20001:$O/v_k3_0.19999     --tol 1e-6 --target 1e-4 --json $O/fd_k3_tight.json
  exit 0
fi
c3=$($S/pyenv/bin/python -c "import math; print(repr(math.log(1.37249382 / 1.75)))")
C3=$c3,$c3,$c3,$c3,$c3,$c3,$c3
# (1) values, in single precision as the value runs of the calibration
run v_const --betamax 1.75
run v_c0 --betamax 1.75 --betau $K:
# (3) the interpolation in BETAMAX at the step-3 value
run v_s3_exact --betamax 1.37249382 $S3
run v_s3_interp --betamax 1.75 --betau $K:$C3 $S3
# (2) tangents, in double precision
run t_const --betamax 1.75 --tangent BETAMAX --f64
run t_c0_all --betamax 1.75 --betau $K: --tangent BETAU --f64
run t_c0_beta --betamax 1.75 --betau $K: --tangent BETAMAX --f64
# (4) the tangent for the coefficient of the 11 m/s knot off the grid, and central differences
run t_k3 --betamax 1.75 --betau $K:0,0,0,0.2,0,0,0 --tangent BETAU3 --f64
for c in 0.22 0.21 0.19 0.18; do
  run v_k3_$c --betamax 1.75 --betau $K:0,0,0,$c,0,0,0 --f64
done
cd $S
$S/pyenv/bin/python scripts/betau_check.py $O
$S/pyenv/bin/python scripts/m3_tl_compare.py $O/t_k3 0.02:$O/v_k3_0.22:$O/v_k3_0.18 0.01:$O/v_k3_0.21:$O/v_k3_0.19 --json $O/fd_k3.json
