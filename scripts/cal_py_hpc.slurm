#!/bin/bash
#SBATCH -J hww3_calpy
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/calpy_%j.log
#
# A step of the calibration in Python on a CPU node (the environment of
# docs/reproducing.md): ARGS are the arguments of scripts/$SCRIPT.
#   sbatch --export=ALL,SCRIPT=cal_obs.py,ARGS="build ..." scripts/cal_py_hpc.slurm
S=/scratch/jmdeloui/HWW3
cd $S
time $S/pyenv/bin/python scripts/$SCRIPT $ARGS
