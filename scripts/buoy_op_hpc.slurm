#!/bin/bash
#SBATCH -J hww3_buoyop
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/buoy_op_%j.log
#
# The depth-aware buoy operator (scripts/buoy_op.py) on the training and
# validation windows of the calibration, from the buoy positions
# ($S/coast/buoys.json) and their GEBCO depths ($S/coast/buoy_depths.json,
# computed where GEBCO is by scripts/buoy_depth.py):
#  - the degenerate operator (nearest node, flat kernel), whose cost and
#    scores must equal those of the operator of cal_obs.py exactly with
#    T_m02 linear, and to round-off with T_m02 through the moments, also
#    for the derivatives from the tangents of the accepted point;
#  - the depth-aware operator (T_m02 through m0 and m2), its weights in
#    $S/coast/weights_<window>.json;
#  - the cost and scores of the start run (it0_value) and of the run after
#    step 1 (it1_value) under these operators, T_m02 interpolated linearly
#    against through m0 and m2, and the coastal classes of
#    scripts/coast_diag.py under the old and the new operator;
#  - on the training window, the cost under the new operator at the
#    accepted point with the tangents there (frozen Jacobian A1) and the
#    Gauss-Newton step it gives ($S/cal/gn1D.json).
#   sbatch scripts/buoy_op_hpc.slurm
S=/scratch/jmdeloui/HWW3
PY=$S/pyenv/bin/python
G=$S/grids/natl_v5k4
C=$S/cal
X=$S/coast
BD=$S/obs/insitu_MO/202601
T="BETAMAX:$C/it1_BETAMAX SDSC2:$C/it1_SDSC2 NLPROP:$C/it1_NLPROP"
cd $S
show() {
  $PY -c "
import json; r = json.load(open('$1')); s = r['scores']
print('%-10s %-20s J %.6f  parts %s  buoy Hs %+.4f/%.4f (%d)  Tm02 %+.4f/%.4f (%d)' % ('$2', '$3', r['J'],
      {k: round(v, 3) for k, v in r['J_parts'].items()}, s['buoy_hs']['bias'], s['buoy_hs']['rmse'], s['buoy_hs']['n'],
      s['buoy_tm02']['bias'], s['buoy_tm02']['rmse'], s['buoy_tm02']['n']))
if 'dJ' in r: print('    dJ/dln', {k: '%.6e' % v for k, v in r['dJ'].items()})"
}
for w in train valid; do
  echo "################ $w"
  B="$C/op_$w.npz $G $C/it0_value $X/buoys.json $X/buoy_depths.json"
  $PY scripts/buoy_op.py build $B $C/op_${w}_nearest.npz --mode nearest --tm02 linear || exit 1
  $PY scripts/buoy_op.py build $B $C/op_${w}_nearestm.npz --mode nearest --tm02 moments || exit 1
  $PY scripts/buoy_op.py build $B $C/op_${w}_depth.npz --json $X/weights_$w.json || exit 1
  for run in it0_value it1_value; do
    for op in op_$w op_${w}_nearest op_${w}_nearestm op_${w}_depth; do
      $PY scripts/cal_obs.py eval $C/$op.npz $C/$run --json $X/${run}_$op.json > /dev/null || exit 1
      show $X/${run}_$op.json $run $op
    done
  done
  for run in it0_value it1_value; do
    echo "T_m02 through the weights, $run:"
    $PY scripts/buoy_op.py tm02 $C/op_${w}_depth.npz $C/$run
  done
  $PY scripts/coast_diag.py $C/op_$w.npz $G $BD $C/it0_value $C/it1_value --subgrid $X/buoy_subgrid.json \
    --op-new $C/op_${w}_depth.npz --fractions 0.25 --json $X/coast_$w.json
done
echo "################ calibration, training window, at the accepted point x1 with the tangents there (A1)"
for op in op_train op_train_nearestm op_train_depth; do
  $PY scripts/cal_obs.py eval $C/$op.npz $C/it1_trial $T --json $X/it1A1_$op.json --npz $X/it1A1_$op.npz > /dev/null || exit 1
  show $X/it1A1_$op.json it1_trial $op
done
cp $X/it1A1_op_train_depth.npz $C/it1_evalD.npz
$PY scripts/cal_gn.py $C/it1_evalD.npz $C/x1.json $C/gn1D.json --names BETAMAX+SDSC2+NLPROP
