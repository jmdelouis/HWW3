#!/bin/bash
#SBATCH -J hww3_tlprobe
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/tl_probe_%j.log
#
# Where the tangent grows (docs/differentiable.md, section 9.5, unstable
# tangents): a value run writes a restart a few hours before a blow-up seen
# in the f64 tangent runs of the calibration, then a tangent run in double
# precision from it probes the nodes of the blow-up after each part of every
# step (hww3-prop run --probe: probe.csv, probe_spec.f64), and
# scripts/tl_probe.py splits the growth by part of the step.
#   CASE=p0: parameters of p0, restart 2 January 18 h, tangent SDSC2, Gulf of
#            Finland nodes 66379, 66400, 66401 (it0 tangents: growth from
#            2 January 22 h)
#   CASE=s3: parameters of step 3, restart 6 January 06 h, tangent BETAMAX,
#            Ungava Bay nodes 198575, 198581 and 221189 south of Greenland
#            (it3 tangents: blow-up from 6 January 12 h)
#   CASE=s3b: parameters of step 3, restart 4 January 06 h, tangent BETAMAX,
#            nodes 238059, 238060 and 237131 on the coast of northern
#            Labrador (it3 tangents: from 5 m at 19 h to 1.3e6 m at 22 h, with
#            H_s 3 m, the source of the tangents that reach the altimeters
#            off Newfoundland on 7 January) and 221168 south of Greenland
#   sbatch --export=ALL,CASE=p0 scripts/tl_probe_hpc.slurm
S=/scratch/jmdeloui/HWW3
D=$S/m2/forcing
G=$S/grids/natl_v5k4
B=${BIN:-$S/target_wave_bu/release/hww3-prop}
P=$S/m2/harness/src_8jan_full.bin.params
R0=$S/runs/m2_week_start_full/restart001.ww3
O=$S/cal/tlfix
H=${HOURS:-12}
source /etc/profile.d/modules.sh
module load cuda/13.0.0
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
FORCE="--current $D/current.ww3 --level $D/level.ww3 --ice $D/ice.ww3 --nest $D/nest.ww3 --wind $D/wind.ww3 --sources $P --units 128"
case $CASE in
  p0) PAR="--betamax 1.75"; STOP="20260102 180000"; TAN=SDSC2; NODES=66379,66400,66401 ;;
  s3) PAR="--betamax 1.37249382 --scale-par SDSC2:0.649237785+NLPROP:1.13718427+BJGAM:1"; STOP="20260106 060000"; TAN=BETAMAX
      NODES=198575,198581,221189 ;;
  s3b) PAR="--betamax 1.37249382 --scale-par SDSC2:0.649237785+NLPROP:1.13718427+BJGAM:1"; STOP="20260104 060000"; TAN=BETAMAX
      NODES=238059,238060,237131,221168 ;;
  *) echo "CASE p0, s3 or s3b"; exit 1 ;;
esac
mkdir -p $O/$CASE
RST=$O/$CASE/restart.ww3
if [ ! -f $O/$CASE/itstep ]; then
  # the value run that writes the restart (single precision, as the value runs)
  $B run $G --restart $R0 $FORCE $PAR --stop "$STOP" --out $O/$CASE/value --restart-out $RST | tee $O/$CASE/value.txt | grep -E "restart|steps in"
  grep -o "continue with --itstep0 [0-9]*" $O/$CASE/value.txt | awk '{print $4}' > $O/$CASE/itstep
fi
IT=$(cat $O/$CASE/itstep)
echo "restart $RST, ITSTEP $IT"
time $B run $G --restart $RST --itstep0 $IT $FORCE $PAR --hours $H --tangent $TAN --f64 --out64 --probe $NODES ${EXTRA} --out $O/$CASE/tangent${SUF} \
  2>&1 | grep -E "sources:|tangent:|steps in|rror|panicked"
$S/pyenv/bin/python $S/scripts/tl_probe.py $O/$CASE/tangent${SUF}
