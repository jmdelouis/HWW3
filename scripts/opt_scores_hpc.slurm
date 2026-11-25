#!/bin/bash
#SBATCH -J hww3_optsc
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/opt_scores_%j.log
#
# Scores of the optimisation checks (docs/wave_results.md, optimisation):
# for each bench run of RUNS (directories of $S/opt/bench separated by +,
# such as base/f32w+spw/f32w), scripts/cal_point.py at p0 on the training
# and validation windows of the calibration: J under the four costs with
# its parts and the bias and RMSE of the altimeters and buoys, to compare
# with the run of the present build (base/...).
#   sbatch --export=ALL,RUNS=base/f32w+loc/f32w scripts/opt_scores_hpc.slurm
S=/scratch/jmdeloui/HWW3
B=$S/opt/bench
mkdir -p $B/scores
cd $S
for r in ${RUNS//+/ }; do
  echo "== $r"
  $S/pyenv/bin/python scripts/cal_point.py $B/$r none $B/scores/${r//\//_}.json | tail -n +2
done
