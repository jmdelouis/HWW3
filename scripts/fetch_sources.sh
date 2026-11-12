#!/bin/bash
# Download the external sources and data used by HWW3 (run once, on a node
# with internet access, e.g. the hpc or Datarmor login node).
#
#   bash scripts/fetch_sources.sh /home1/datawork/$USER/HWW3
#
# - WAVEWATCH III develop (NOAA-EMC), pinned to the commit used in the note;
# - netCDF-Fortran 4.6.1 (Unidata) and ParMETIS 4.0.3 (with METIS 5.1.0);
# - GEBCO 2024 global bathymetry (netCDF, 7.5 GB zipped).
# The GSHHS coastline shapefile is taken from an existing copy
# (configs/natl_datarmor.json: /home1/datahome/marc/data/GSHHS/gshhs_land.shp);
# the GSHHG archive is at https://www.soest.hawaii.edu/pwessel/gshhg/.
set -e
R=${1:?usage: fetch_sources.sh <HWW3 work directory>}
mkdir -p $R/src $R/data/gebco
cd $R/src
if [ ! -d WW3 ]; then
  git clone https://github.com/NOAA-EMC/WW3.git
fi
(cd WW3 && git fetch -q && git checkout -q dea641c && git log --oneline -1)
[ -f netcdf-fortran-4.6.1.tar.gz ] || \
  wget -q -O netcdf-fortran-4.6.1.tar.gz \
    https://github.com/Unidata/netcdf-fortran/archive/refs/tags/v4.6.1.tar.gz
[ -f parmetis-4.0.3.tar.gz ] || \
  wget -q -O parmetis-4.0.3.tar.gz \
    https://karypis.github.io/glaros/files/sw/parmetis/parmetis-4.0.3.tar.gz || \
  wget -q -O parmetis-4.0.3.tar.gz \
    https://ftp.mcs.anl.gov/pub/petsc/externalpackages/parmetis-4.0.3-p9.tar.gz
cd $R/data/gebco
if [ ! -f GEBCO_2024.nc ]; then
  wget -q -O gebco_2024.zip https://www.bodc.ac.uk/data/open_download/gebco/gebco_2024/zip/
  unzip -o gebco_2024.zip GEBCO_2024.nc && rm gebco_2024.zip
fi
ls -la $R/src $R/data/gebco
