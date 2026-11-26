#!/bin/bash
#SBATCH -J hww3_struct
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -o /scratch/jmdeloui/HWW3/struct/logs/struct_run_%j.log
#
# hww3-prop on a WW3 structured grid triangulated by hww3-pre struct
# (hww3_wave::structgrid): the grids of the IRI global configuration
# (IRIGLOB-7M, RECT 0.125°; ARC-15KM and ANTARC-15KM, CURV 15 km), meshed
# into $S/struct/grids/<name>, from calm conditions with the idealised
# wind of hww3-pre ideal-wind (no forcing copied yet), the source
# parameters built from the namelists of the configuration (BETAMAX 1.57,
# SDSC2 -3.8, NLPROP 2.5e7) and the spectral grid of its ww3_grid.nml
# (NK 36, NTH 36, FREQ1 0.0339, XFR 1.1). The namelists are those of work/
# (the copies in data/ are templates with <betamax>), shared by the three
# grids, with the whitecapping of HWW3 (SDSBCHOICE 1, SDSCUM 0) in place of
# SDSBCHOICE 3 with the isotropic cumulative term SDSCUM 0.3, which the
# source kernels do not have (NML=... gives another namelist file).
#   sbatch --export=ALL,GRID=IRIGLOB-7M,HOURS=24,PREC=f32 scripts/struct_run_hpc.slurm
# FORCE=DIR takes wind.ww3 (and current.ww3, ice.ww3 when present) from DIR
# instead of the idealised wind.
# PROFILE=1 adds --profile (the time of each part, with synchronisations).
# The meshes are written on the login node with
#   hww3-pre struct $S/struct/iri/work/ww3_grid.nml.<name> --data $S/struct/iri/data --out $S/struct/grids/<name>
S=/scratch/jmdeloui/HWW3
B=${BIN:-$S/struct/target_gpu/release}
GN=${GRID:-IRIGLOB-7M}
G=$S/struct/grids/$GN
I=$S/struct/iri
H=${HOURS:-24}
T0=${START:-20190101000000}
J=$SLURM_JOB_ID
O=$S/struct/runs/${GN}_${PREC:-f32}_$J
source /etc/profile.d/modules.sh
module load cuda/13.0.0
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
mkdir -p $O
N=$I/namelists_hww3.nml
sed -e 's/SDSBCHOICE *= *3/SDSBCHOICE = 1/' -e 's/SDSCUM *= *0.300/SDSCUM = 0.0/' $I/work/namelists_IRIGLOB-7M.nml > $N
$B/hww3-src build-params $G ${NML:-$N} $O/src.params || exit 1
opt=""
if [ -n "$FORCE" ]; then
  # forcing files written for this mesh (struct_fieldnc.py, hww3-pre prnc)
  F="--wind $FORCE/wind.ww3"
  for f in current ice; do [ -f $FORCE/$f.ww3 ] && F="$F --$f $FORCE/$f.ww3"; done
else
  $B/hww3-pre ideal-wind $G --out $O/wind.ww3 --start $T0 --hours $H || exit 1
  F="--wind $O/wind.ww3"
fi
echo "forcing: $F"
[ "${PREC:-f32}" = f64 ] && opt="--f64"
[ -n "$PROFILE" ] && opt="$opt --profile"
$B/hww3-prop run $G --restart calm --start $T0 $F --sources $O/src.params \
  --hours $H --out $O/out $opt $XOPT 2>&1 | tail -80
nvidia-smi --query-gpu=memory.used --format=csv
rm -f $O/wind.ww3
echo "outputs in $O/out"
