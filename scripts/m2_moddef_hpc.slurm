#!/bin/bash
#SBATCH -J hww3_mdef
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/moddef_%j.log
#
# Milestone M2b: the mod_def.ww3 of the calibrated configuration of grid
# v5 K=4 with BETAMAX set to $BETAMAX (the stress tables of INSIN4 depend on
# it), in $S/runs/m2_moddef_$TAG, for the finite differences with respect to
# BETAMAX; scripts/m2_src_harness_hpc.slurm (RUN=that directory) then writes
# the parameter file the GPU model reads.
#   sbatch --export=ALL,BETAMAX=1.7675,TAG=bp1 scripts/m2_moddef_hpc.slurm
S=/scratch/jmdeloui/HWW3
I=${SETUP:-$S/runs/test_grid_v5k4}
D=$S/runs/m2_moddef_$TAG
B=$S/opt/ww3/bin
source /etc/profile.d/modules.sh
module purge
module load oneapi/2025.2.1 hdf5/1.14.6 netcdf-c/4.9.3-parallel netcdf-fortran/4.6.2-parallel
set -e
mkdir -p $D
cd $D
cp $I/natl.msh $I/natl.obc $I/ww3_grid.nml .
sed -e "s|^&SIN4 BETAMAX = 1.75 /\$|\&SIN4 BETAMAX = $BETAMAX /|" $I/namelists_natl.nml > namelists_natl.nml
grep "SIN4" namelists_natl.nml
$B/ww3_grid > ww3_grid.out 2>&1 || { tail -40 ww3_grid.out; exit 1; }
grep -E "BETAMAX|SINTABLE" ww3_grid.out
ls -la mod_def.ww3
