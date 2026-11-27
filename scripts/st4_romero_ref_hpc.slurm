#!/bin/bash
#SBATCH -J hww3_romero
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH -t 00:40:00
#SBATCH -o /scratch/jmdeloui/HWW3/struct/logs/romero_ref_%j.log
#
# WW3 reference of the ST4 whitecapping of the IRI configuration
# (SDSBCHOICE = 3, Romero 2019, with the isotropic cumulative term
# SDSCUM = 0.3): the mod_def.ww3 of grid v5 K=4 built by ww3_grid with the
# namelists of the HWW3 runs whose &SIN4 is replaced by the &SIN4, &SDS4
# and &SNL1 groups of the IRI namelists (struct/iri/work), then the
# source-term harness of M2a (hww3-wave/harness/src_harness.F90, compiled
# against the hpc WW3 build into $D) on the restart of 8 January 2026 and
# the wind of the M2 forcing, every STEP-th sea point, and the check of
# the CPU transcriptions against it (hww3-src check, built without the GPU
# features in struct/target_cpu). NML=FILE takes the groups from another
# namelist file, TAG names the run directory $S/struct/romero/$TAG.
#   sbatch --export=ALL,TAG=iri scripts/st4_romero_ref_hpc.slurm
S=/scratch/jmdeloui/HWW3
I=$S/runs/test_grid_v5k4
TAG=${TAG:-iri}
D=$S/struct/romero/$TAG
NML=${NML:-$S/struct/iri/work/namelists_IRIGLOB-7M.nml}
B=$S/opt/ww3/bin
STEP=${STEP:-200}
source /etc/profile.d/modules.sh
module purge
module load oneapi/2025.2.1 hdf5/1.14.6 netcdf-c/4.9.3-parallel netcdf-fortran/4.6.2-parallel
set -e
mkdir -p $D
cd $D
cp $I/natl.msh $I/natl.obc $I/ww3_grid.nml .
# the &SIN4 line of the HWW3 namelists replaced by the source groups of NML
sed -n '/^&SIN4/,/^&SNL1/p' $NML > groups.nml
awk -v f=groups.nml '/^&SIN4 BETAMAX = 1.75 \/$/ { while ((getline l < f) > 0) print l; next } { print }' $I/namelists_natl.nml > namelists_natl.nml
grep -A8 "^&SIN4" namelists_natl.nml
$B/ww3_grid > ww3_grid.out 2>&1 || { tail -40 ww3_grid.out; exit 1; }
grep -E "SDSBCHOICE|SDSC2|SDSCUM|SDSBR|SDSFACMTF|BETAMAX|SINTABLE" ww3_grid.out | head
mpiifx -no-fma -g -traceback -i4 -real-size 32 -fp-model precise -assume byterecl -O2 \
  -I$S/build/ww3/model/src/mod $S/struct/hww3-wave/harness/src_harness.F90 -o $D/ww3_srcharness \
  $S/opt/ww3/lib/libww3.a -L$S/opt/lib -lparmetis -lmetis $(nf-config --flibs)
SECONDS=0
mpirun -np 1 $D/ww3_srcharness $S/m1/v5k4/restart001.ww3 $S/m2/forcing/wind.ww3 $D/harness.bin $STEP
echo "harness done in $SECONDS s"
$S/struct/target_cpu/release/hww3-src check $D/harness.bin
