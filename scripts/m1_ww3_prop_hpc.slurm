#!/bin/bash
#SBATCH -J hww3_ww3prop
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/ww3prop_%j.log
#
# Propagation-only WW3 runs on the Ifremer hpc cluster (WW3 dea641c, HWW3
# switches, oneAPI build in $S/opt/ww3), the Fortran references of milestone
# M1 of docs/differentiable.md, on grid v5 K=4 with the explicit scheme:
#   MODE=prop   from restart001.ww3 of natl_v5k4_jan2026_exp (8 January 2026
#               00 h), currents, levels, ice and boundary spectra of that
#               run, source terms off (RUN%FLSOU = F)
#   MODE=geo    the same with refraction and frequency shift off
#               (RUN%FLCTH = RUN%FLCK = F)
#   MODE=ideal  idealised swell from ww3_strt (Gaussian packet, 0.08 Hz,
#               centred at 25W 47N, 3 deg spread, H_max 3 m), no forcing and
#               no boundary spectra
# H_s and T_m02 (or the fields of FLIST) every FSTRIDE s (default 3600) for
# HOURS hours (default 48)
# in $S/runs/m1_<MODE><TAG>/ww3.*.nc; with RSTRIDE, also RCOUNT restarts
# (full spectra) every RSTRIDE s from the start, restart001.ww3 ... there.
#   sbatch --export=ALL,MODE=prop scripts/m1_ww3_prop_hpc.slurm
#   sbatch --export=ALL,MODE=prop,HOURS=1,FSTRIDE=600,RSTRIDE=600,RCOUNT=3,TAG=_diag scripts/m1_ww3_prop_hpc.slurm
S=/scratch/jmdeloui/HWW3
MODE=${MODE:-prop}
HOURS=${HOURS:-48}
FSTRIDE=${FSTRIDE:-3600}
FLIST=${FLIST:-HS T02}
I=$S/runs/test_grid_v5k4
F=$S/m1/v5k4
D=$S/runs/m1_$MODE${TAG}
B=$S/opt/ww3/bin
source /etc/profile.d/modules.sh
module purge
module load oneapi/2025.2.1 hdf5/1.14.6 netcdf-c/4.9.3-parallel netcdf-fortran/4.6.2-parallel
export I_MPI_FABRICS=shm
# the PDLIB restart writer (UNST_PDLIB_WRITE_TO_FILE) holds a 460 MB
# automatic array: without this the ranks die on the first restart
ulimit -s unlimited
set -e
START='20260108 000000'
STOP=$(date -u -d "2026-01-08 00:00 UTC + $HOURS hours" +'%Y%m%d %H%M%S')
mkdir -p $D; cd $D
cp $I/natl.msh $I/natl.obc $I/namelists_natl.nml $I/ST4TABUHF2.bin .
sed -e 's/^\( *RUN%FLSOU *=\).*/\1 F/' $I/ww3_grid.nml > ww3_grid.nml
if [ "$MODE" = geo ]; then
  sed -i -e 's/^\( *RUN%FLCTH *=\).*/\1 F/' -e 's/^\( *RUN%FLCK *=\).*/\1 F/' ww3_grid.nml
fi
grep -E "RUN%|EXPFSN" ww3_grid.nml namelists_natl.nml
$B/ww3_grid > ww3_grid.out 2>&1 || { tail -40 ww3_grid.out; exit 1; }
FORC=T
if [ "$MODE" = ideal ]; then
  FORC=F
  rm -f nest.ww3 current.ww3 level.ww3 ice.ww3
  printf '%s\n' '$ Idealised swell for the M1 propagation check' '  1' \
      '  0.08  0.01  270.  2  -25.  3.  47.  3.  3.0' > ww3_strt.inp
  # ww3_strt calls MPI_Init: start it through mpirun inside the job
  mpirun -np 1 $B/ww3_strt > ww3_strt.out 2>&1 || { tail -30 ww3_strt.out; exit 1; }
  tail -5 ww3_strt.out
else
  for f in current.ww3 level.ww3 ice.ww3 nest.ww3; do ln -sf $F/$f .; done
  ln -sf $F/restart001.ww3 restart.ww3
fi
cat > ww3_shel.nml <<NML
! Propagation only (RUN%FLSOU = F), M1 reference, mode $MODE.
&DOMAIN_NML
  DOMAIN%START  = '$START'
  DOMAIN%STOP   = '$STOP'
/
&INPUT_NML
  INPUT%FORCING%WATER_LEVELS = '$FORC'
  INPUT%FORCING%CURRENTS     = '$FORC'
  INPUT%FORCING%WINDS        = 'F'
  INPUT%FORCING%ICE_CONC     = '$FORC'
/
&OUTPUT_TYPE_NML
  TYPE%FIELD%LIST = '$FLIST'
/
&OUTPUT_DATE_NML
  DATE%FIELD%START  = '$START'
  DATE%FIELD%STRIDE = '$FSTRIDE'
  DATE%FIELD%STOP   = '$STOP'
NML
if [ -n "$RSTRIDE" ]; then
  R0=$(date -u -d "2026-01-08 00:00 UTC + $RSTRIDE seconds" +'%Y%m%d %H%M%S')
  R1=$(date -u -d "2026-01-08 00:00 UTC + $((RSTRIDE * ${RCOUNT:-1})) seconds" +'%Y%m%d %H%M%S')
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
  FIELD%TIMESTRIDE = '$FSTRIDE'
  FIELD%TIMECOUNT  = '$((HOURS * 3600 / FSTRIDE + 1))'
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
echo "ww3_shel rc=$? in $SECONDS s ($SLURM_NTASKS ranks)"
tail -8 ww3_shel.out
SECONDS=0
$B/ww3_ounf > ww3_ounf.out 2>&1
echo "ww3_ounf rc=$? in $SECONDS s"
ls -la ww3.*.nc
ls -la restart[0-9]*.ww3 2>/dev/null || true
