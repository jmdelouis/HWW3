#!/bin/bash
#SBATCH -J hww3_bccfl
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/bccfl_%j.log
#
# Sub-cycling of the explicit scheme with and without the boundary nodes, on
# the Ifremer hpc cluster (WW3 dea641c, HWW3 switches, oneAPI build in
# $S/opt/ww3): the grid v5 K=4 explicit run natl_v5k4_jan2026_exp with its
# sources (full ST4, BETAMAX = 1.75) from restart001.ww3 (8 January 2026
# 00 h) for HOURS hours (default 12), with wind, currents, levels, ice and
# boundary spectra, run twice in the same job, on the same node and ranks:
#   UGBCCFL = T  the WW3 default: PDLIB_W3XYPFSN2 takes the sub-cycling over
#                every wet node, coastal and open-boundary nodes included
#   UGBCCFL = F  over the interior nodes only
# in the UNST namelist of ww3_grid. Hourly fields FLIST in
# $S/runs/bccfl_{T,F}<TAG>/ww3.*.nc.
#   sbatch scripts/m2_ww3_bccfl_hpc.slurm
#   sbatch --export=ALL,HOURS=24,ORDER="F T",TAG=_24h scripts/m2_ww3_bccfl_hpc.slurm
S=/scratch/jmdeloui/HWW3
HOURS=${HOURS:-12}
FLIST=${FLIST:-HS T02 FP DIR}
ORDER=${ORDER:-T F}
I=$S/runs/test_grid_v5k4
FO=$S/m2/forcing
B=$S/opt/ww3/bin
source /etc/profile.d/modules.sh
module purge
module load oneapi/2025.2.1 hdf5/1.14.6 netcdf-c/4.9.3-parallel netcdf-fortran/4.6.2-parallel
export I_MPI_FABRICS=shm
# the PDLIB restart writer (UNST_PDLIB_WRITE_TO_FILE) holds a 460 MB
# automatic array: without this the ranks die on a restart write
ulimit -s unlimited
set -e
START='20260108 000000'
STOP=$(date -u -d "2026-01-08 00:00 UTC + $HOURS hours" +'%Y%m%d %H%M%S')
echo "node $(hostname), $SLURM_NTASKS ranks, $HOURS h from $START, order $ORDER"
for BC in $ORDER; do
  D=$S/runs/bccfl_$BC${TAG}
  mkdir -p $D; cd $D
  cp $I/natl.msh $I/natl.obc $I/ww3_grid.nml $I/ST4TABUHF2.bin .
  sed -e "s/^&UNST\$/\&UNST\n  UGBCCFL        = $BC,/" $I/namelists_natl.nml > namelists_natl.nml
  $B/ww3_grid > ww3_grid.out 2>&1 || { tail -40 ww3_grid.out; exit 1; }
  grep -o "UGBCCFL = *[TF]" ww3_grid.out
  grep -q "UGBCCFL = *$BC" ww3_grid.out || { echo "UGBCCFL is not $BC in ww3_grid.out"; exit 1; }
  for f in wind.ww3 current.ww3 level.ww3 ice.ww3 nest.ww3; do ln -sf $FO/$f .; done
  ln -sf $S/m1/v5k4/restart001.ww3 restart.ww3
  cat > ww3_shel.nml <<NML
! HWW3 North Atlantic, grid v5 K=4, explicit scheme with sources, UGBCCFL = $BC.
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
/
NML
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
  RC=0; mpirun -np $SLURM_NTASKS $B/ww3_shel > ww3_shel.out 2>&1 || RC=$?
  T_SHEL=$SECONDS
  echo "UGBCCFL = $BC: ww3_shel rc=$RC in $T_SHEL s ($SLURM_NTASKS ranks), $((T_SHEL * 24 / HOURS)) s per model day"
  tail -4 ww3_shel.out
  [ $RC -eq 0 ] || exit $RC
  SECONDS=0
  $B/ww3_ounf > ww3_ounf.out 2>&1
  echo "ww3_ounf rc=$? in $SECONDS s"
  ls -la ww3.*.nc
done
