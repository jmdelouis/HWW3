#!/bin/bash
#SBATCH -J hww3_cost3
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/cal_cost3_%j.log
#
# The third cost of the calibration (Jean-Marc, 24 November 2026): the
# depth-aware buoy operator without the buoys the mesh cannot represent
# (configs/buoy_exclude.csv), representativeness errors per class fitted
# again from the start run without them, and a Huber loss with delta = 1.5
# observation errors for the altimeters and the buoys (scripts/buoy_sigma.py
# --exclude --huber); then the start run and step 3 under the four costs on
# both windows (scripts/cal_point.py).
#   sbatch scripts/cal_cost3_hpc.slurm
S=/scratch/jmdeloui/HWW3
PY=$S/pyenv/bin/python
C=$S/cal
X=$S/coast
cd $S
$PY scripts/buoy_sigma.py $C/op_train_depth.npz $C/op_valid_depth.npz $C/it0_value $X/coast_train.json $X/coast_valid.json \
  $X/buoy_subgrid_all.json $C/op_train_cost3.npz $C/op_valid_cost3.npz --json $X/sigma_repr3.json \
  --exclude configs/buoy_exclude.csv --huber 1.5 || exit 1
for p in it0_value:none it3_value:x3_trial.json; do
  run=${p%%:*}; x=${p#*:}; [ "$x" = none ] || x=$C/$x
  $PY scripts/cal_point.py $C/$run $x $C/point3_$run.json
done
