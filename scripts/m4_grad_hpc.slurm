#!/bin/bash
#SBATCH -J hww3_m4grad
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH -t 24:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/m4_grad_%j.log
#
# Milestone M4, checkpoint d: the gradient of the training cost of the
# BETAMAX(U) test (third cost, op_train_cost3.npz, 5 January 12 h to
# 7 January 12 h) at p0 by the adjoint, against the derivatives of the same
# cost along the nine tangents of the calibration (BETAU0..6 of bu0_*,
# SDSC2 and NLPROP of it0_*), all true gradients:
#   1. scripts/cal_obs.py seed: dJ/dH_s and dJ/dT_m02 per hour from the
#      hourly fields of the value run at p0 (VALUE, default it0_value);
#   2. hww3-prop adj-grad over HOURS (default 156) from the restart of the
#      calibration, with BETAMAX(U) at zero coefficients (the p0 model) and
#      two levels of checkpoints in host memory (K1 steps per segment);
#   3. scripts/cal_obs.py eval: dJ along the nine tangents.
#   sbatch --export=ALL,BIN=$S/opt/bin_grad1 scripts/m4_grad_hpc.slurm
S=/scratch/jmdeloui/HWW3
B=${BIN:-$S/target_wave_tl/release}
D=$S/m2/forcing
G=$S/grids/natl_v5k4
P=$S/m2/harness/src_8jan_full.bin.params
R=$S/runs/m2_week_start_full/restart001.ww3
C=$S/cal
VALUE=${VALUE:-$C/it0_value}
OP=${OP:-$C/op_train_cost3.npz}
PY=$S/pyenv/bin/python
O=$S/m4/grad/${TAG:-$SLURM_JOB_ID}
mkdir -p $O
source /etc/profile.d/modules.sh
module load cuda/13.0.0
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
cd $S
echo "== seeds: dJ/dH_s and dJ/dT_m02 of $OP from $VALUE"
$PY scripts/cal_obs.py seed $OP $VALUE $O/seeds || exit 1
echo "== the adjoint over ${HOURS:-156} h"
FORCE="--current $D/current.ww3 --level $D/level.ww3 --ice $D/ice.ww3 --nest $D/nest.ww3 --wind $D/wind.ww3 --sources $P"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 60 > $O/mem.txt &
mp=$!
$B/hww3-prop adj-grad $G --restart $R $FORCE --betamax 1.75 --betau 2,5,8,11,14,17,20: --hours ${HOURS:-156} \
  --seeds $O/seeds ${K1:+--k1 $K1} | tee $O/adj_grad.txt
kill $mp
awk '{if ($1 > m) m = $1} END {printf "== GPU memory: peak %d MiB\n", m}' $O/mem.txt
echo "== dJ along the nine tangents"
T=""
for k in 0 1 2 3 4 5 6; do T="$T BETAU$k:$C/bu0_BETAU$k"; done
$PY scripts/cal_obs.py eval $OP $VALUE $T SDSC2:$C/it0_SDSC2 NLPROP:$C/it0_NLPROP --json $O/tl_eval.json | grep "dJ/dln"
$PY - $O/adj_grad.txt $O/tl_eval.json <<'EOF'
import json, sys
adj = json.loads([l for l in open(sys.argv[1]) if l.startswith("{")][-1])
tl = json.load(open(sys.argv[2]))["dJ"]
print("%-8s %14s %14s %10s" % ("", "adjoint", "tangent", "rel"))
for k in tl:
    a, t = adj.get(k, float("nan")), tl[k]
    print("%-8s %14.6e %14.6e %10.2e" % (k, a, t, abs(a - t) / max(abs(t), 1e-300)))
EOF
