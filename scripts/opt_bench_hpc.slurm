#!/bin/bash
#SBATCH -J hww3_bench
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/opt_bench_%j.log
#
# Checks of an optimisation step against the present build (docs/wave_results.md,
# optimisation): runs of the binaries of NEW (default target_wave_tl) with
# the parameters of p0 (BETAMAX 1.75, as the calibration runs) from the
# restart of 1 January 00:10, in $S/opt/bench/$TAG, compared hour by hour
# (scripts/opt_compare.py) with the same run of the present build
# ($S/opt/bin_base, run once into $S/opt/bench/base/CASE):
#   f32w  216 h value run in single precision (the week);
#   f32   48 h value run in single precision;
#   f64   the same in double precision;
#   tl    24 h of the f64 BETAMAX tangent;
#   tl32  the same tangent in single precision, against the f64 tangent of
#         the present build (relative differences);
#   tl32w the f32 tangent over the training window of the calibration
#         (156 h) against its f64 tangent at p0 (cal/it0_BETAMAX), at the
#         observations and in the Jacobian of cost 3;
#   freeze32  the same with each diagnostic mask of P_TLFREEZE (MASKS),
#         the number of nodes with a non-finite tangent after each source step;
#   probe32  1 h of the f32 tangent with the nodes of NODES probed after
#         each part of every step (probe.csv), to find where it goes wrong;
#   src   the value source kernel alone on the nodes of the source harness
#         of grid v5 K=4 repeated REP times (default 200, about the size of
#         the grid; hww3-src check-gpu), f32 and f64, both builds: time of
#         one global step and the differences against WW3 and the CPU;
#   srcs  the same for each build of BINS (directories of $S/opt separated
#         by :, such as bin_base:bin_ws:bin_spw), one after the other on the
#         same GPU, REPEAT times (default 2);
#   m3    the M3 check of the source tangent (hww3-src check-tl against
#         central differences on the harness nodes of grids v5 K=4 and v4)
#         with both builds.
# The first run of a reference also compares it with the calibration run
# at p0 it repeats ($S/cal/it0_value, $S/cal/it0_BETAMAX).
#   sbatch --export=ALL,TAG=ws,CASES="f32w f64 tl" scripts/opt_bench_hpc.slurm
# UNITS sets the units per cube of the new runs (default 128).
S=/scratch/jmdeloui/HWW3
D=$S/m2/forcing
G=$S/grids/natl_v5k4
NEW=${NEW:-$S/target_wave_tl/release}
BASE=$S/opt/bin_base
P=$S/m2/harness/src_8jan_full.bin.params
R=$S/runs/m2_week_start_full/restart001.ww3
O=$S/opt/bench/${TAG:-new}
PY=$S/pyenv/bin/python
source /etc/profile.d/modules.sh
module load cuda/13.0.0
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
mkdir -p $O $S/opt/bench/base
cd $S
FORCE="--current $D/current.ww3 --level $D/level.ww3 --ice $D/ice.ww3 --nest $D/nest.ww3 --wind $D/wind.ww3 --sources $P"
PAR="--betamax 1.75 --scale-par SDSC2:1+NLPROP:1+BJGAM:1"
run() {
  # run BIN_DIR OUT HOURS [options]
  local b=$1 o=$2 h=$3
  shift 3
  echo "== $o: $h h $*"
  $b/hww3-prop run $G --restart $R $FORCE $PAR --hours $h --out64 --out $o "$@" 2>&1 | grep -E "steps in|rror|panicked|nonfinite"
}
ref() {
  # ref CASE HOURS [options]: the run of the present build, once
  local c=$1 h=$2
  shift 2
  [ -f $S/opt/bench/base/$c/summary.json ] || {
    run $BASE $S/opt/bench/base/$c $h --units 128 "$@"
    case $c in
      f32w) $PY scripts/opt_compare.py $S/cal/it0_value $S/opt/bench/base/f32w --prefix hs,t02 ;;
      tl) $PY scripts/opt_compare.py $S/cal/it0_BETAMAX $S/opt/bench/base/tl --prefix hs,dhs,dt02 --hours 24 ;;
    esac
  }
}
for c in ${CASES:-f32w f64 tl}; do
  case $c in
    f32w)
      ref f32w 216
      run $NEW $O/f32w 216 --units ${UNITS:-128}
      $PY scripts/opt_compare.py $S/opt/bench/base/f32w $O/f32w --prefix hs,t02 ;;
    f32|f64)
      opt=""; [ $c = f64 ] && opt="--f64"
      ref $c 48 $opt
      run $NEW $O/$c 48 --units ${UNITS:-128} $opt
      $PY scripts/opt_compare.py $S/opt/bench/base/$c $O/$c --prefix hs,t02 ;;
    tl)
      ref tl 24 --f64 --tangent BETAMAX
      run $NEW $O/tl 24 --units ${UNITS:-128} --f64 --tangent BETAMAX
      $PY scripts/opt_compare.py $S/opt/bench/base/tl $O/tl --prefix hs,dhs,dt02 ;;
    tl32)
      ref tl 24 --f64 --tangent BETAMAX
      run $NEW $O/tl32 24 --units ${UNITS:-128} --tangent BETAMAX
      $PY scripts/opt_compare.py $S/opt/bench/base/tl $O/tl32 --prefix hs,dhs,dt02 --rel ;;
    tl32w)
      # the f32 tangent over the training window of the calibration (156 h,
      # to 7 January 12:00) against the f64 tangent at p0 of the calibration
      # (cal/it0_BETAMAX): the fields, the derivatives at the observations
      # of cost 3, and the columns of the Jacobian and dJ (cal_jaccmp.py)
      C=$S/cal
      run $NEW $O/tl32w 156 --units ${UNITS:-128} --tangent BETAMAX
      $PY scripts/opt_compare.py $C/it0_BETAMAX $O/tl32w --prefix hs,dhs,dt02 --rel
      $PY scripts/tl_cmp_obs.py $C/op_train_cost3.npz $C/it0_BETAMAX $O/tl32w
      for r in $C/it0_BETAMAX $O/tl32w; do
        echo "== eval $r"
        $PY scripts/cal_obs.py eval $C/op_train_cost3.npz $C/it0_value BETAMAX:$r --json $O/eval_$(basename $r).json --npz $O/eval_$(basename $r).npz | grep -E "\"J\"|dJ/dln" | cut -c1-160
      done
      $PY scripts/cal_jaccmp.py $O/eval_it0_BETAMAX.npz $O/eval_tl32w.npz BETAMAX ;;
    freeze32)
      # the f32 tangent over 1 h with each mask of P_TLFREEZE (srce_tl) left
      # out, nodes with a tangent that is not finite after each source step
      for m in ${MASKS:-0 255 2 4 8 16 32 64 128 1}; do
        echo "== mask $m"
        HWW3_TLFREEZE=$m HWW3_NANCOUNT=srce run $NEW $O/freeze32 1 --units ${UNITS:-128} --tangent BETAMAX --probe 0 | grep -E "nonfinite" | cut -c1-140 | head -6
      done ;;
    probe32)
      run $NEW $O/probe32 ${HOURS:-1} --units ${UNITS:-128} --tangent BETAMAX --probe ${NODES//:/,}
      head -1 $O/probe32/probe.csv
      head -${ROWS:-80} $O/probe32/probe.csv | tail -n +2 ;;
    src)
      for opt in "" --f64; do
        for b in $BASE $NEW; do
          echo "== check-gpu $b $opt"
          $b/hww3-src check-gpu $S/m2/harness/src_8jan_full.bin $opt --units ${UNITS:-128} --replicate ${REP:-200} 2>&1 | grep -E "^GPU|after the step|dynamic step"
        done
      done ;;
    srcs)
      for k in $(seq ${REPEAT:-2}); do
        for opt in "" --f64; do
          for b in ${BINS//:/ }; do
            echo "== check-gpu $b $opt"
            $S/opt/$b/hww3-src check-gpu $S/m2/harness/src_8jan_full.bin $opt --units ${UNITS:-128} --replicate ${REP:-200} 2>&1 | grep -E "^GPU"
          done
        done
      done ;;
    m3)
      for hn in src_8jan_full.bin src_v4_full.bin; do
        for b in $BASE $NEW; do
          echo "== check-tl $b $hn"
          $b/hww3-src check-tl $S/m2/harness/$hn --units ${UNITS:-128} --h 1e-4 2>&1 | grep -vE "^\s*$"
        done
      done ;;
    *) echo "case $c: f32w, f32, f64, tl, tl32, tl32w, freeze32, probe32, src, srcs or m3" ;;
  esac
done
