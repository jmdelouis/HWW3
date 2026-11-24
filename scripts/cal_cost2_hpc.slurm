#!/bin/bash
#SBATCH -J hww3_cost2
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/cal_cost2_%j.log
#
# The second cost of the calibration (Jean-Marc, 24 November 2026): the
# depth-aware buoy operator (scripts/buoy_op.py) with representativeness
# errors per class of buoy fixed from the start run (scripts/buoy_sigma.py),
# then every stored point under the first cost (nearest node, sigma 0.30 m
# and 1.0 s) and the second: the start run, step 1, the WW3 run at step 1
# and the value run of step 2 (it2_trial, training window only), and the
# Gauss-Newton step of the frozen Jacobian A1 at step 1 under the second
# cost ($S/cal/gn1N.json).
#   sbatch scripts/cal_cost2_hpc.slurm
S=/scratch/jmdeloui/HWW3
PY=$S/pyenv/bin/python
C=$S/cal
X=$S/coast
cd $S
$PY scripts/buoy_sigma.py $C/op_train_depth.npz $C/op_valid_depth.npz $C/it0_value $X/coast_train.json $X/coast_valid.json \
  $X/buoy_subgrid_all.json $C/op_train_cost2.npz $C/op_valid_cost2.npz --json $X/sigma_repr.json || exit 1
show() {
  $PY -c "
import json; r = json.load(open('$1')); s = r['scores']
print('%-6s %-24s %-16s J %10.3f  alt %9.3f  bhs %9.3f  btm %9.3f   alt %+.4f/%.4f  buoy Hs %+.4f/%.4f (%d)  Tm02 %+.4f/%.4f (%d)' % (
      '$2', '$3', '$4', r['J'], r['J_parts']['alt'], r['J_parts']['bhs'], r['J_parts']['btm'], s['alt']['bias'], s['alt']['rmse'],
      s['buoy_hs']['bias'], s['buoy_hs']['rmse'], s['buoy_hs']['n'], s['buoy_tm02']['bias'], s['buoy_tm02']['rmse'], s['buoy_tm02']['n']))"
}
for w in train valid; do
  for run in $C/it0_value $C/it1_value $S/runs/m2_week_cal1/ww3.202601.nc $C/it2_trial; do
    [ "$w" = valid ] && [ "$run" = "$C/it2_trial" ] && continue
    tag=$(basename $(dirname $run))_$(basename $run .nc)
    for op in op_$w op_${w}_depth op_${w}_cost2; do
      $PY scripts/cal_obs.py eval $C/$op.npz $run --json $X/cost2_${tag}_$op.json > /dev/null || { echo "eval $run $op failed"; continue; }
      show $X/cost2_${tag}_$op.json $w $tag $op
    done
  done
done
echo "#### step of A1 at step 1 under the second cost"
$PY scripts/cal_obs.py eval $C/op_train_cost2.npz $C/it1_trial BETAMAX:$C/it1_BETAMAX SDSC2:$C/it1_SDSC2 NLPROP:$C/it1_NLPROP \
  --json $C/it1_evalN.json --npz $C/it1_evalN.npz > /dev/null || exit 1
$PY scripts/cal_gn.py $C/it1_evalN.npz $C/x1.json $C/gn1N.json --names BETAMAX+SDSC2+NLPROP
$PY -c "
import json, numpy as np
x1 = json.load(open('$C/x1.json')); x2 = json.load(open('$C/x2_trial.json'))
for n in x2['names']:
    a, b = x1['x'][x1['names'].index(n)], x2['x'][x2['names'].index(n)]
    print('prior check %s: x1 %+.5f  x2 %+.5f' % (n, a, b))
print('prior of x1 %.3f, of x2 %.3f' % (sum((np.array(x1['x']) / 0.25) ** 2), sum((np.array(x2['x']) / 0.25) ** 2)))"
