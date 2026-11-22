#!/bin/bash
#SBATCH -J hww3_m2ref
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/m2ref_%j.log
#
# WW3 references of milestone M2a of docs/differentiable.md on the Ifremer
# hpc cluster (WW3 dea641c, HWW3 switches, oneAPI build in $S/opt/ww3): the
# grid v5 K=4 explicit run natl_v5k4_jan2026_exp with the simplified ST4,
# SINTABLE = 0 in &SIN4 (no stress tables: u* by Newton iterations, tail
# stress integrated in line) and SDSCUM = 0 in &SDS4 (no cumulative breaking
# term), everything else unchanged (BETAMAX = 1.75, UGBCCFL at its default):
#   MODE=dev   HOURS (default 48) from restart001.ww3 of natl_v5k4_jan2026_exp
#              (8 January 2026 00 h), with wind, currents, levels, ice and
#              boundary spectra
#   MODE=week  1-8 January 2026, cold start, as natl_v5k4_jan2026_exp
# With ST4=full the namelists of natl_v5k4_jan2026_exp are kept unchanged:
# the calibrated full ST4 of milestone M2b (stress tables, SINTABLE = 1, and
# the cumulative breaking term, SDSCUM = -0.40, both WW3 defaults).
# Forcing (1-10 January) in $S/m2/forcing. Hourly fields FLIST (default those
# of natl_v5k4_jan2026_exp) in $S/runs/m2_<MODE><TAG>/ww3.*.nc; with RSTRIDE,
# also RCOUNT restarts (full spectra) every RSTRIDE s from the start (or
# from RSTART s after the start),
# restart001.ww3 ... there.
#   sbatch --export=ALL,MODE=dev,RSTRIDE=600,RCOUNT=3 scripts/m2_ww3_ref_hpc.slurm
#   sbatch --export=ALL,MODE=week scripts/m2_ww3_ref_hpc.slurm
S=/scratch/jmdeloui/HWW3
MODE=${MODE:-dev}
ST4=${ST4:-simplified}
FLIST=${FLIST:-HS T02 FP DIR WND ICE CUR WLV}
# grid set-up (natl.msh, natl.obc, ww3_grid.nml, namelists_natl.nml) and forcing;
# SETUP=$S/runs/setup_v4 FORCING=$S/m2/forcing_v4 for grid v4
I=${SETUP:-$S/runs/test_grid_v5k4}
FO=${FORCING:-$S/m2/forcing}
D=$S/runs/m2_$MODE${TAG}
B=$S/opt/ww3/bin
source /etc/profile.d/modules.sh
module purge
module load oneapi/2025.2.1 hdf5/1.14.6 netcdf-c/4.9.3-parallel netcdf-fortran/4.6.2-parallel
export I_MPI_FABRICS=shm
# the PDLIB restart writer (UNST_PDLIB_WRITE_TO_FILE) holds a 460 MB
# automatic array: without this the ranks die on the first restart
ulimit -s unlimited
set -e
case $MODE in
  dev) T0='2026-01-08 00:00'; HOURS=${HOURS:-48} ;;
  week) T0='2026-01-01 00:00'; HOURS=${HOURS:-168} ;;
  *) echo "unknown MODE $MODE"; exit 1 ;;
esac
START=$(date -u -d "$T0 UTC" +'%Y%m%d %H%M%S')
STOP=$(date -u -d "$T0 UTC + $HOURS hours" +'%Y%m%d %H%M%S')
mkdir -p $D; cd $D
cp $I/natl.msh $I/natl.obc $I/ww3_grid.nml .
if [ "$ST4" = full ]; then
  cp $I/namelists_natl.nml .
else
  # simplified ST4
  sed -e 's|^&SIN4 BETAMAX = 1.75 /$|\&SIN4 BETAMAX = 1.75, SINTABLE = 0 /\n\&SDS4 SDSCUM = 0.0 /|' \
      $I/namelists_natl.nml > namelists_natl.nml
fi
grep -E "SIN4|SDS4|EXPFSN" namelists_natl.nml
$B/ww3_grid > ww3_grid.out 2>&1 || { tail -40 ww3_grid.out; exit 1; }
grep -E "SINTABLE|SDSCUM =" ww3_grid.out
for f in wind.ww3 current.ww3 level.ww3 ice.ww3 nest.ww3; do ln -sf $FO/$f .; done
rm -f restart.ww3
[ "$MODE" = dev ] && ln -sf $S/m1/v5k4/restart001.ww3 restart.ww3
cat > ww3_shel.nml <<NML
! HWW3 North Atlantic, set-up $(basename $I), explicit scheme, ST4 $ST4 (simplified:
! SINTABLE = 0, SDSCUM = 0; full: as calibrated): M2 reference, mode $MODE.
&DOMAIN_NML
  DOMAIN%START  = '$START'
  DOMAIN%STOP   = '$STOP'
/
&INPUT_NML
  INPUT%FORCING%WATER_LEVELS = 'T'
  INPUT%FORCING%CURRENTS     = 'T'
  INPUT%FORCING%WINDS        = 'T'
  INPUT%FORCING%ICE_CONC     = 'T'
/
&OUTPUT_TYPE_NML
  TYPE%FIELD%LIST = '$FLIST'
/
&OUTPUT_DATE_NML
  DATE%FIELD%START  = '$START'
  DATE%FIELD%STRIDE = '3600'
  DATE%FIELD%STOP   = '$STOP'
NML
if [ -n "$RSTRIDE" ]; then
  RS=${RSTART:-$RSTRIDE}
  R0=$(date -u -d "$T0 UTC + $RS seconds" +'%Y%m%d %H%M%S')
  R1=$(date -u -d "$T0 UTC + $((RS + RSTRIDE * (${RCOUNT:-1} - 1))) seconds" +'%Y%m%d %H%M%S')
  cat >> ww3_shel.nml <<NML
  DATE%RESTART%START  = '$R0'
  DATE%RESTART%STRIDE = '$RSTRIDE'
  DATE%RESTART%STOP   = '$R1'
NML
fi
echo "/" >> ww3_shel.nml
cat ww3_shel.nml
cat > ww3_ounf.nml <<NML
&FIELD_NML
  FIELD%TIMESTART  = '$START'
  FIELD%TIMESTRIDE = '3600'
  FIELD%TIMECOUNT  = '$((HOURS + 1))'
  FIELD%LIST       = '$FLIST'
  FIELD%SAMEFILE   = T
  FIELD%TYPE       = 4
/
&FILE_NML
  FILE%PREFIX  = 'ww3.'
  FILE%NETCDF  = 4
/
NML
SECONDS=0
mpirun -np $SLURM_NTASKS $B/ww3_shel > ww3_shel.out 2>&1
echo "ww3_shel rc=$? in $SECONDS s ($SLURM_NTASKS ranks, $((SECONDS * SLURM_NTASKS / 3600)) core-hours)"
tail -8 ww3_shel.out
SECONDS=0
$B/ww3_ounf > ww3_ounf.out 2>&1
echo "ww3_ounf rc=$? in $SECONDS s"
ls -la ww3.*.nc
ls -la restart[0-9]*.ww3 2>/dev/null || true
