#!/bin/bash
#SBATCH -J hww3_calmon
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/cal_monitor_%j.log
#
# Monitoring of the calibration (scripts/cal_point.py): for each point of
# POINTS (RUN:X separated by +, RUN a directory of $S/cal and X its control
# vector file there, or none), J on the training and validation windows
# under the first cost, the depth-aware operator alone and the second cost,
# with its parts, the prior term and the unweighted scores.
#   sbatch --export=ALL,POINTS=it0_value:none+it2_value:x2.json scripts/cal_monitor_hpc.slurm
S=/scratch/jmdeloui/HWW3
C=$S/cal
cd $S
for p in ${POINTS//+/ }; do
  run=${p%%:*}; x=${p#*:}
  [ "$x" = none ] || x=$C/$x
  $S/pyenv/bin/python scripts/cal_point.py $C/$run $x $C/point_$run.json
done
