#!/bin/bash
#SBATCH -J hww3_calfin
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/cal_final_%j.log
#
# The end of the calibration (docs/differentiable.md, section 9.5), on a CPU
# node, at the accepted point K (default 1):
#  - the cost and its derivatives from the f64 tangents there
#    ($S/cal/itK_<NAME>), the posterior covariance and the step that this
#    Jacobian would take (cal_gn.py), and the change of the Jacobian from
#    that of iteration 0, which every step used (cal_jaccmp.py);
#  - the scores on the training and validation windows of the GPU runs at
#    the start (it0_value) and at the end (itK_value), and of the WW3 runs
#    with the parameters of the start ($S/refs/natl_v5k4_jan2026_exp) and
#    at the end ($S/runs/m2_week_cal$K).
#   sbatch [--export=ALL,K=1] scripts/cal_final_hpc.slurm
S=/scratch/jmdeloui/HWW3
PY=$S/pyenv/bin/python
K=${K:-1}
C=$S/cal
NAMES=BETAMAX+SDSC2+NLPROP
cd $S
$PY scripts/cal_obs.py eval $C/op_train.npz $C/it${K}_BETAMAX BETAMAX:$C/it${K}_BETAMAX SDSC2:$C/it${K}_SDSC2 NLPROP:$C/it${K}_NLPROP \
  --json $C/it${K}_eval.json --npz $C/it${K}_eval.npz | grep -E "\"J\"|dJ/dln" | cut -c1-160
$PY scripts/cal_gn.py $C/it${K}_eval.npz $C/x$K.json $C/gn${K}_final.json --names $NAMES
$PY scripts/cal_jaccmp.py $C/it0_eval.npz $C/it${K}_eval.npz $NAMES
for w in train valid; do
  for run in $C/it${K}_value $S/refs/natl_v5k4_jan2026_exp/ww3.202601.nc $S/runs/m2_week_cal$K/ww3.202601.nc; do
    tag=$(echo $run | sed -e "s|$S/||" -e 's|/ww3.202601.nc||' -e 's|/|_|g')
    $PY scripts/cal_obs.py eval $C/op_$w.npz $run --json $C/final_${tag}_$w.json > /dev/null || echo "eval of $run failed"
  done
  $PY - <<PY
import json, os
for name, p in (("GPU start", "$C/it0_$w.json"), ("GPU end", "$C/final_cal_it${K}_value_$w.json"),
                ("WW3 start", "$C/final_refs_natl_v5k4_jan2026_exp_$w.json"), ("WW3 end", "$C/final_runs_m2_week_cal${K}_$w.json")):
    if not os.path.exists(p):
        print("$w %-9s missing" % name); continue
    r = json.load(open(p)); s = r["scores"]
    print("$w %-9s J %9.1f  alt %+.4f/%.4f  buoy Hs %+.4f/%.4f  Tm02 %+.4f/%.4f" % (name, r["J"],
          s["alt"]["bias"], s["alt"]["rmse"], s["buoy_hs"]["bias"], s["buoy_hs"]["rmse"], s["buoy_tm02"]["bias"], s["buoy_tm02"]["rmse"]))
PY
done
