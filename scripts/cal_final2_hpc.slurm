#!/bin/bash
#SBATCH -J hww3_calfin2
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/cal_final2_%j.log
#
# The end of the calibration under the second cost (docs/differentiable.md,
# section 9.5), at the final point K (default 3), on a CPU node:
#  - the second cost on the training window with the residuals of the
#    216 h value run itK_value and the f64 tangents there (itK_<NAME>),
#    the posterior covariance and correlations and the step this Jacobian
#    would take (cal_gn.py), and the change of the Jacobian from A1, that
#    of step 1, which steps 2 to K used (cal_jaccmp.py);
#  - the WW3 run at the final point ($S/runs/m2_week_cal$K) and the GPU
#    run there under the three costs on both windows (cal_point.py).
#   sbatch [--export=ALL,K=3] scripts/cal_final2_hpc.slurm
S=/scratch/jmdeloui/HWW3
PY=$S/pyenv/bin/python
K=${K:-3}
C=$S/cal
NAMES=BETAMAX+SDSC2+NLPROP
cd $S
$PY scripts/cal_obs.py eval $C/op_train_cost2.npz $C/it${K}_value BETAMAX:$C/it${K}_BETAMAX SDSC2:$C/it${K}_SDSC2 NLPROP:$C/it${K}_NLPROP \
  --json $C/it${K}_evalN.json --npz $C/it${K}_evalN.npz | grep -E "\"J\"|dJ/dln" | cut -c1-120
$PY scripts/cal_gn.py $C/it${K}_evalN.npz $C/x$K.json $C/gn${K}_final.json --names $NAMES
$PY scripts/cal_jaccmp.py $C/it1_evalN.npz $C/it${K}_evalN.npz $NAMES
$PY scripts/cal_point.py $S/runs/m2_week_cal$K/ww3.202601.nc $C/x$K.json $C/point_ww3_cal$K.json
$PY scripts/cal_point.py $C/it${K}_value $C/x$K.json $C/point_it${K}_value.json
