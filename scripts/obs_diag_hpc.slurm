#!/bin/bash
#SBATCH -J hww3_obsdiag
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/obs_diag_%j.log
#
# Observation diagnostics of the calibration (scripts/obs_diag.py), on a
# CPU node, into $S/diag: first the operators of both windows rebuilt with
# the spread of the 7 samples of each altimeter block (op_<window>_sd.npz,
# the same blocks as op_<window>.npz), then the diagnostics of the buoys,
# the altimeters and the loss.
#   sbatch [--export=ALL,PARTS=A+B+C] scripts/obs_diag_hpc.slurm
S=/scratch/jmdeloui/HWW3
PY=$S/pyenv/bin/python
C=$S/cal
ICE=$S/refs/natl_v5k4_jan2026_exp/ww3.202601.nc+$S/runs/m2_dev_full/ww3.202601.nc
cd $S
[ -f $C/op_train_sd.npz ] || $PY scripts/cal_obs.py build $C/op_train_sd.npz $C/it0_value 2026010512 2026010712 $ICE $S/grids/natl_v5k4
[ -f $C/op_valid_sd.npz ] || $PY scripts/cal_obs.py build $C/op_valid_sd.npz $C/it0_value 2026010713 2026011000 $ICE $S/grids/natl_v5k4
time $PY scripts/obs_diag.py $S/diag --parts ${PARTS:-A+B+C}
ls -la $S/diag
