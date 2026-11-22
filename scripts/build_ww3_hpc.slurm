#!/bin/bash
#SBATCH -J hww3_build
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/build_ww3_hpc_%j.log
#
# Build ParMETIS 4.0.3 (with METIS 5.1.0) and WW3 develop (commit dea641c,
# same source and switches as on Datarmor) on the Ifremer hpc cluster
# (RHEL 9.4) with oneAPI 2025.2.1 (ifx, Intel MPI) and the system parallel
# netCDF modules. ifx is the IntelLLVM compiler, for which WW3's CMake sets no
# flags, so the ifort flags of the Datarmor build that ifx accepts are passed
# explicitly (-assume byterecl keeps the record lengths of the Datarmor
# restart files). METIS and ParMETIS are compiled with gcc (Intel MPI mpicc).
# Sources in $R/src: WW3 cloned at dea641c, parmetis-4.0.3.tar.gz from Datarmor.
# Run: sbatch scripts/build_ww3_hpc.slurm
set -e
R=${R:-/scratch/jmdeloui/HWW3}
SRC=$R/src
PREFIX=$R/opt
mkdir -p $PREFIX
source /etc/profile.d/modules.sh
module purge
module load oneapi/2025.2.1 hdf5/1.14.6 netcdf-c/4.9.3-parallel netcdf-fortran/4.6.2-parallel
module list 2>&1
which ifx mpiifx mpicc nc-config nf-config
ifx --version | head -1
NC=$(nc-config --prefix)
NF=$(nf-config --prefix)
echo "netCDF-C $NC, netCDF-Fortran $NF"

# --- ParMETIS + METIS --------------------------------------------------------
if [ ! -f $PREFIX/lib/libparmetis.a ]; then
  cd $SRC && rm -rf parmetis-4.0.3 && tar xzf parmetis-4.0.3.tar.gz
  cd parmetis-4.0.3
  (cd metis && make config cc=gcc prefix=$PREFIX && make -j8 && make install) > metis.out 2>&1 || { tail -40 metis.out; exit 1; }
  (make config cc=mpicc cxx=mpicxx prefix=$PREFIX && make -j8 && make install) > parmetis.out 2>&1 || { tail -40 parmetis.out; exit 1; }
fi
ls $PREFIX/lib

# --- WW3 ---------------------------------------------------------------------
# Intel fpp drops the quotes of -DENDIANNESS='big_endian'; hard-code the parameter.
(cd $SRC/WW3 && git checkout -- model/src/constants.F90)
sed -i "/FILE_ENDIAN *= *ENDIANNESS/s/ENDIANNESS/'big_endian'/" $SRC/WW3/model/src/constants.F90
grep -n "FILE_ENDIAN =" $SRC/WW3/model/src/constants.F90
export METIS_PATH=$PREFIX
B=$R/build/ww3
rm -rf $B && mkdir -p $B && cd $B
cmake $SRC/WW3 -DSWITCH=$R/ww3/switch_hww3 -DCMAKE_INSTALL_PREFIX=$PREFIX/ww3 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_COMPILER=mpiifx -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_Fortran_FLAGS="-no-fma -g -traceback -i4 -real-size 32 -fp-model precise -assume byterecl" \
  -DCMAKE_Fortran_FLAGS_RELEASE="-O3" \
  -DNetCDF_ROOT=$NF -DNetCDF_C_ROOT=$NC -DNetCDF_Fortran_ROOT=$NF
make -j8 2>&1 | grep -vE "warning #|^ *\^|^-+\^"
test ${PIPESTATUS[0]} -eq 0
make install > /dev/null
ls -la $PREFIX/ww3/bin
