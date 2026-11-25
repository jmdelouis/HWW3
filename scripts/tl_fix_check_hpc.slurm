#!/bin/bash
#SBATCH -J hww3_tlfix
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 01:40:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/tl_fix_%j.log
#
# Checks of the tangent with the unstable interactions of the DIA left out
# (hww3-prop --tl-nlcut CUT, docs/differentiable.md section 9.5), in
# $S/cal/tlfix/check_CUT:
#   PART=m3   the checks of M3 again: the source step on the harness nodes
#             against central differences (hww3-src check-tl), the 3 h
#             tangents of WIND and INIT (with the source terms) against the
#             stored central differences of steps 1e-4 and 1e-5 (Slurm jobs
#             4163936 and 4163935), and the 48 h tangents of BETAMAX and
#             NLPROP against the differences of M2b ($S/m2/fd/cases.txt);
#   PART=s3   the BETAMAX tangent at the parameters of step 3 over HOURS
#             (default 156 h; 216 h, with -t 02:15:00, covers both windows),
#             and its derivatives at the observations of the training window
#             (scripts/tl_diag.py), against those of the step-3 tangent
#             without the cut (job 4326004: 527 altimeter blocks above 10 m);
#   PART=bu3  the tangent of the 11 m/s coefficient of BETAMAX(U) at p0 over
#             156 h, against the run without the cut (job 4332614) at the
#             observations of the training window.
#   sbatch --export=ALL,PART=m3,CUT=1 scripts/tl_fix_check_hpc.slurm
S=/scratch/jmdeloui/HWW3
D=$S/m2/forcing
G=$S/grids/natl_v5k4
B=${BIN:-$S/target_wave_tl/release}
P=$S/m2/harness/src_8jan_full.bin.params
PY=$S/pyenv/bin/python
CUT=${CUT:-1}
O=$S/cal/tlfix/check_$CUT
C=$S/cal
mkdir -p $O
source /etc/profile.d/modules.sh
module load cuda/13.0.0
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
FORCE="--current $D/current.ww3 --level $D/level.ww3 --ice $D/ice.ww3 --nest $D/nest.ww3 --wind $D/wind.ww3 --sources $P --units 128"
cd $S
case $PART in
  m3)
    echo "== the source step on the harness nodes"
    $B/hww3-src check-tl $S/m2/harness/src_8jan_full.bin --units 128 --h 1e-4 --tl-nlcut $CUT
    R=$S/m1/v5k4/restart001.ww3
    for nj in WIND:4163936 INIT:4163935; do
      n=${nj%%:*}; j=${nj#*:}
      echo "== 3 h $n against the stored differences of job $j"
      $B/hww3-prop run $G --restart $R $FORCE --hours 3 --f64 --out64 --tangent $n --tl-nlcut $CUT --out $O/m3_$n 2>&1 | grep -E "tangent:|steps in|rror"
      $PY scripts/m3_tl_compare.py $O/m3_$n 1e-4:$S/m3/tl_$j/plus_1e-4:$S/m3/tl_$j/minus_1e-4 1e-5:$S/m3/tl_$j/plus_1e-5:$S/m3/tl_$j/minus_1e-5 \
        --tol 1e-6 --target 1e-4 | grep -E "^energy|^H_s 202601080[13]"
    done
    for n in BETAMAX NLPROP; do
      echo "== 48 h $n against the differences of M2b"
      $B/hww3-prop run $G --restart $R $FORCE --hours 48 --f64 --out64 --tangent $n --tl-nlcut $CUT --out $O/m3_48_$n 2>&1 | grep -E "tangent:|steps in|rror"
      PAIRS=""
      for s in 1:0.01 01:0.001; do
        t=${s%%:*}; h=${s#*:}
        jp=$(awk -v x=${n}_p$t '$1 == x { print $2 }' $S/m2/fd/cases.txt)
        jm=$(awk -v x=${n}_m$t '$1 == x { print $2 }' $S/m2/fd/cases.txt)
        if [ -n "$jp" ] && [ -n "$jm" ]; then PAIRS="$PAIRS $h:$S/m2/out_$jp:$S/m2/out_$jm"; fi
      done
      $PY scripts/m3_tl_compare.py $O/m3_48_$n $PAIRS --tol 1e-3 --json $O/cmp48_$n.json | grep -E "^energy|^H_s 2026011000"
    done
    ;;
  s3)
    $B/hww3-prop run $G --restart $S/runs/m2_week_start_full/restart001.ww3 $FORCE --betamax 1.37249382 \
      --scale-par SDSC2:0.649237785+NLPROP:1.13718427+BJGAM:1 --hours ${HOURS:-156} --f64 --out64 --tangent BETAMAX --tl-nlcut $CUT --out $O/s3_BETAMAX \
      2>&1 | grep -E "sources:|tangent:|steps in|rror"
    $PY scripts/tl_diag.py $O/s3_BETAMAX $C/it3_BETAMAX --op $C/op_train_cost3.npz --top 5 --json $O/tl_diag_s3.json | grep -v "^  2026"
    ;;
  bu3)
    $B/hww3-prop run $G --restart $S/runs/m2_week_start_full/restart001.ww3 $FORCE --betamax 1.75 --betau 2,5,8,11,14,17,20: \
      --hours 156 --f64 --out64 --tangent BETAU3 --tl-nlcut $CUT --out $O/bu0_BETAU3 2>&1 | grep -E "sources:|tangent:|steps in|rror"
    $PY scripts/tl_diag.py $O/bu0_BETAU3 $C/bu0_BETAU3 --op $C/op_train_cost3.npz --top 5 --json $O/tl_diag_bu3.json | grep -v "^  2026"
    $PY scripts/tl_cmp_obs.py $C/op_train_cost3.npz $C/bu0_BETAU3 $O/bu0_BETAU3
    ;;
  *) echo "PART m3, s3 or bu3"; exit 1 ;;
esac
