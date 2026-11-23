#!/bin/bash
#SBATCH -J hww3_fdgrad
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/fdgrad_%j.log
#
# Milestone M2b: finite-difference gradients of the GPU model in double
# precision with respect to BETAMAX, SDSC2, NLPROP, BJGAM and the wind
# speed, from the 48 h runs listed in $S/m2/fd/cases.txt (scripts/
# m2_fd_gradients.py: total energy, mean H_s and the altimeter and buoy
# scores of each run over 8-10 January, with the Python environment and the
# January 2026 observations copied to scratch, docs/reproducing.md).
#   sbatch --dependency=afterany:J1:J2:... [--export=ALL,CASES=FILE,OUT=FILE.json] scripts/m2_fd_hpc.slurm
S=/scratch/jmdeloui/HWW3
C=${CASES:-$S/m2/fd/cases.txt}
O=${OUT:-$S/m2/fd/gradients.json}
P=$S/pyenv/bin/python
export VAL_SPINUP_DAYS=${VAL_SPINUP_DAYS:-0}
export VAL_END=${VAL_END:-2026-01-10T00:00}
export VAL_MIN_HOURS=${VAL_MIN_HOURS:-24}
set -e
cd $S
time $P scripts/m2_fd_gradients.py $C $S/runs/m2_dev_full/ww3.202601.nc $S/grids/natl_v5k4/plot/nodes.npz $S/grids/natl_v5k4 $O
