#!/bin/bash
#SBATCH -J hww3_sprobe
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/struct/logs/struct_probe_%j.log
#
# The number of dynamic source sub-steps per global step, from the mean
# dynamic step dtdyn of the source kernel at probed nodes (probe.csv of
# hww3-prop run --probe, every EVERY-th node, default 1000 and 250), with the timing of the same
# hours without probes, for
#   iri   IRIGLOB-7M (NK 36) from the restart of 4 January 2019 00 h of the
#         72 h run (job 4514626) with the FIELD_NC wind and currents;
#   v5k4  the HEALPix grid v5 K=4 (NK 32) from the restart of 8 January
#         2026 00 h with its forcing and the parameters of the calibration
#         at p0 (as the case dev32 of opt_bench_hpc.slurm).
#   sbatch --export=ALL,CASES="iri v5k4",HOURS=1 scripts/struct_probe_hpc.slurm
# NML=FILE gives another namelist file for iri.
# dtdyn is the mean of the raw limiter step, before the DTMIN floor of the
# step taken: the sub-steps per step are min(60, 600 / max(dtdyn, DTMIN)).
S=/scratch/jmdeloui/HWW3
B=${BIN:-$S/struct/target_gpu/release}
H=${HOURS:-1}
J=$SLURM_JOB_ID
PY=$S/pyenv/bin/python
source /etc/profile.d/modules.sh
module load cuda/13.0.0
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
for c in ${CASES:-iri v5k4}; do
  O=$S/struct/probe/${c}_$J
  mkdir -p $O
  case $c in
    iri)
      G=$S/struct/grids/IRIGLOB-7M
      F=$S/struct/forcing/IRIGLOB-7M
      $B/hww3-src build-params $G ${NML:-$S/struct/iri/namelists_hww3.nml} $O/src.params > /dev/null || exit 1
      ARGS="$G --restart $S/struct/restarts/IRIGLOB-7M_2019010400.ww3 --itstep0 433 --wind $F/wind.ww3 --current $F/current.ww3 --sources $O/src.params"
      NP=1189834; EV=${EVERY:-1000} ;;
    v5k4)
      D=$S/m2/forcing
      ARGS="$S/grids/natl_v5k4 --restart $S/m1/v5k4/restart001.ww3 --current $D/current.ww3 --level $D/level.ww3 --ice $D/ice.ww3 --nest $D/nest.ww3 --wind $D/wind.ww3 --sources $S/m2/harness/src_8jan_full.bin.params --betamax 1.75 --scale-par SDSC2:1+NLPROP:1+BJGAM:1"
      NP=$(( $(stat -c %s $S/grids/natl_v5k4/raw/lon.f64) / 8 )); EV=${EVERY:-250} ;;
  esac
  nodes=$(seq 0 $EV $((NP - 1)) | paste -sd,)
  echo "== $c: timing"
  $B/hww3-prop run $ARGS --hours $H --out $O/time 2>&1 | grep -E "steps in|rror|panicked"
  echo "== $c: probes"
  $B/hww3-prop run $ARGS --hours $H --out $O/probe --probe $nodes 2>&1 | grep -E "steps in|rror|panicked"
  rm -f $O/probe/probe_spec.f64
  $PY - $O/probe/probe.csv <<'P'
import csv, sys
import numpy as np
rows = list(csv.DictReader(open(sys.argv[1])))
ph = sorted({r["phase"] for r in rows})
dt = np.array([float(r["dtdyn"]) for r in rows if r["phase"] == "srce" and float(r["mapsta"]) == 1])
dt = dt[dt > 0]
n = np.minimum(60.0, 600.0 / np.maximum(dt, 10.0))
print(f"phases {ph}; {dt.size} node-steps after the sources: sub-steps per 600 s step mean {n.mean():.1f}, median {np.median(n):.1f}, p10 {np.percentile(n,10):.1f}, p90 {np.percentile(n,90):.1f}, max {n.max():.1f}; raw step below DTMIN at {100*(dt<10).mean():.1f} %, mean raw step {dt.mean():.1f} s")
P
done
