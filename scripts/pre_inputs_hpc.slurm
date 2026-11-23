#!/bin/bash
#SBATCH -J hww3_preinp
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/pre_inputs_%j.log
#
# Forcing and boundary files of grid v5 K=4 built in Rust (hww3-pre prnc,
# bounc) from the netCDF inputs of the January 2026 WW3 runs (copied from
# Datarmor to $S/pre/natl_v5k4_jan2026_{cur,bnd}), and compared with the
# files of ww3_prnc and ww3_bounc used by the M2 runs ($S/m2/forcing),
# over the times of START to STOP; FIELDS (field:variables, separated by
# spaces) and BND=0 restrict the comparison.
#   sbatch [--export=ALL,START=20260101,STOP=20260111] scripts/pre_inputs_hpc.slurm
S=/scratch/jmdeloui/HWW3
B=${BIN:-$S/target_wave_test/release/hww3-pre}
PY=$S/pyenv/bin/python
G=$S/grids/natl_v5k4
MSH=$S/runs/m2_week_start_full/natl.msh
C=$S/pre/natl_v5k4_jan2026_cur
N=$S/pre/natl_v5k4_jan2026_bnd
R=${FORCING:-$S/m2/forcing}
O=$S/pre/rust
T="--start ${START:-20260101} --stop ${STOP:-20260111}"
mkdir -p $O
cd $S
for f in ${FIELDS:-wind:u10+v10 ice:siconc current:uo+vo level:zos}; do
  k=${f%%:*}; v=${f#*:}
  echo "== $k"
  time $PY scripts/prnc_nc.py $C/$k.nc $v $O/${k}_raw $T || exit 1
  time $B prnc $G --field $k --input $O/${k}_raw --out $O/$k.ww3 --msh $MSH || exit 1
  $B cmp-fields $O/$k.ww3 $R/$k.ww3
done
[ "${BND:-1}" = "0" ] && exit 0
echo "== boundary"
time $PY scripts/bounc_nc.py $N/spec.list $O/spec_raw || exit 1
time $B bounc $G --input $O/spec_raw --out $O/nest.ww3 --msh $MSH || exit 1
$B cmp-nest $O/nest.ww3 $R/nest.ww3
