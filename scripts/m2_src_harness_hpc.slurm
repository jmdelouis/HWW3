#!/bin/bash
#SBATCH -J hww3_srch
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/srcharness_%j.log
#
# Source-term harness of milestone M2a (hww3-wave/harness/src_harness.F90):
# compiled against the hpc WW3 build (libww3.a and its modules, same flags),
# then run in the directory of a run whose mod_def.ww3 holds the source-term
# settings (default the simplified-ST4 reference runs/m2_dev), on the spectra
# of RESTART and the wind of WIND, every STEP-th sea point plus NODES
# (separated by '+'). Output $S/m2/harness/$OUT.
#   sbatch --export=ALL,OUT=src_8jan.bin scripts/m2_src_harness_hpc.slurm
S=/scratch/jmdeloui/HWW3
RUN=${RUN:-$S/runs/m2_dev}
RESTART=${RESTART:-$S/m1/v5k4/restart001.ww3}
WIND=${WIND:-$S/m2/forcing/wind.ww3}
STEP=${STEP:-200}
OUT=${OUT:-src_harness.bin}
source /etc/profile.d/modules.sh
module purge
module load oneapi/2025.2.1 hdf5/1.14.6 netcdf-c/4.9.3-parallel netcdf-fortran/4.6.2-parallel
set -e
H=$S/m2/harness
mkdir -p $H
cd $H
mpiifx -no-fma -g -traceback -i4 -real-size 32 -fp-model precise -assume byterecl -O2 \
  -I$S/build/ww3/model/src/mod $S/hww3-wave/harness/src_harness.F90 -o ww3_srcharness \
  $S/opt/ww3/lib/libww3.a -L$S/opt/lib -lparmetis -lmetis $(nf-config --flibs)
cd $RUN
SECONDS=0
# an MPI program: start it through mpirun inside the job
mpirun -np 1 $H/ww3_srcharness $RESTART $WIND $H/$OUT $STEP ${NODES//+/ }
echo "harness done in $SECONDS s"
ls -la $H/$OUT
