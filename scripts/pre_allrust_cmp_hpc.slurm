#!/bin/bash
#SBATCH -J hww3_allrcmp
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/pre_allrust_cmp_%j.log
#
# The all-Rust GPU run ($S/pre/allrust, scripts/pre_allrust_hpc.slurm)
# against the calibration's start run ($S/cal/it0_value): hourly H_s node
# by node, then the cost and scores of the calibration on the training and
# validation windows.
#   sbatch --dependency=afterok:JOB scripts/pre_allrust_cmp_hpc.slurm
S=/scratch/jmdeloui/HWW3
PY=$S/pyenv/bin/python
A=$S/cal/it0_value
R=$S/pre/allrust
cd $S
$PY scripts/pre_hs_cmp.py $A $R
for w in train valid; do
  $PY scripts/cal_obs.py eval $S/cal/op_$w.npz $R --json $S/pre/allrust_$w.json > /dev/null
  $PY - <<PY
import json
for name, p in (("WW3 inputs", "$S/cal/it0_$w.json"), ("Rust inputs", "$S/pre/allrust_$w.json")):
    r = json.load(open(p)); s = r["scores"]
    print("$w %-11s J %.1f  alt %+.4f/%.4f  buoy Hs %+.4f/%.4f  Tm02 %+.4f/%.4f" % (name, r["J"],
          s["alt"]["bias"], s["alt"]["rmse"], s["buoy_hs"]["bias"], s["buoy_hs"]["rmse"], s["buoy_tm02"]["bias"], s["buoy_tm02"]["rmse"]))
PY
done
