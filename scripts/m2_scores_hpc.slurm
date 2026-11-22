#!/bin/bash
#SBATCH -J hww3_score
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/scores_%j.log
#
# Altimeter and buoy scores of milestone M2 on the hpc cluster: the WW3 run
# WW3NC and, with GPU (an output directory of hww3-prop), the GPU run made
# into a netCDF file shaped like WW3NC (scripts/gpu_to_nc.py), both through
# validate_altimetry.py (window ending VAL_END) and validate_buoys.py (same
# buoys and hours for both runs, VAL_MIN_HOURS), with the Python environment
# and the January 2026 observations copied to scratch (docs/reproducing.md).
#   sbatch --export=ALL,WW3NC=$S/runs/m2_week/ww3.202601.nc,GPU=$S/m2/out_J,OUT=$S/m2/val_week scripts/m2_scores_hpc.slurm
S=/scratch/jmdeloui/HWW3
P=$S/pyenv/bin/python
G=${GRID:-$S/grids/natl_v5k4}
OUT=${OUT:-$S/m2/val}
export VAL_END=${VAL_END:-2026-01-07T21:00}
export VAL_MIN_HOURS=${VAL_MIN_HOURS:-48}
set -e
mkdir -p $OUT
cd $S
RUNS="ww3:$WW3NC"
if [ -n "$GPU" ]; then
  time $P scripts/gpu_to_nc.py $WW3NC $GPU $OUT/gpu.nc
  RUNS="$RUNS gpu:$OUT/gpu.nc"
fi
B=""
for r in $RUNS; do
  time $P scripts/validate_altimetry.py ${r#*:} $G/plot/nodes.npz $OUT/${r%%:*} $S/obs/l3
  B="$B ${r%%:*}:${r#*:}:$G"
done
time $P scripts/validate_buoys.py $OUT/buoys $B --buoys $S/obs/insitu_MO/202601
ls -la $OUT
