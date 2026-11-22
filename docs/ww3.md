# WW3 setup

## Build

WW3 `develop` (NOAA-EMC, commit `dea641c`) is built with CMake by
`scripts/build_ww3.pbs` on Datarmor:

- compilers: Intel 2020 (`intel-comp/20`, ifort 19.1.3) and Intel MPI
  2019.9 (`impi/2019.9.304`); Intel 17 lacks `mpi_f08`;
- netCDF-C 4.8.1 from the system
  (`/appli/netCDF/netcdf-4.8.1_impi-2017.2.174`); netCDF-Fortran 4.6.1
  built locally, because the system `netcdf.mod` was made by an old
  gfortran; its test suites are skipped (they do not compile with
  ifort 19.1);
- ParMETIS 4.0.3 and METIS 5.1.0 built locally (`METIS_PATH`);
- Intel's preprocessor drops the quotes of `-DENDIANNESS='big_endian'`,
  so `FILE_ENDIAN` is set in `model/src/constants.F90` before compiling.

The binaries are in `$R/opt/ww3/bin` (`ww3_grid`, `ww3_prnc`, `ww3_shel`,
`ww3_ounf`, ...).

### Ifremer hpc cluster

`scripts/build_ww3_hpc.slurm` builds the same source (commit `dea641c`,
cloned from GitHub) with the same switches on the Ifremer hpc cluster
(RHEL 9.4, Slurm), in `R=/scratch/jmdeloui/HWW3`. It was set up on
22 November 2026, when the Datarmor scratch refused every write.

- Compilers: oneAPI 2025.2.1, i.e. `ifx` 2025.2.1 with Intel MPI
  (`mpiifx`). netCDF comes from the system modules `hdf5/1.14.6`,
  `netcdf-c/4.9.3-parallel` and `netcdf-fortran/4.6.2-parallel`.
- Flags: CMake identifies `ifx` as IntelLLVM, for which WW3 sets no
  compiler flags. The script therefore passes the ifort flags of the
  Datarmor build that `ifx` accepts:
  `-no-fma -g -traceback -i4 -real-size 32 -fp-model precise -assume byterecl`,
  plus `-O3`.
- ParMETIS 4.0.3 and METIS 5.1.0 come from the same tarball as on
  Datarmor. They are compiled with gcc through Intel MPI's `mpicc`.
- `FILE_ENDIAN` is hard-coded, as on Datarmor.

The build takes 2 min 19 s on 8 cores (Slurm job 4115990).

**Check against Datarmor.** Two tests on the grid v5 $K=4$ explicit
configuration:

- `ww3_grid` (13 s, job 4116028):
  - `mapsta.ww3` and `mask.ww3` are identical to Datarmor's, byte for byte;
  - `mod_def.ww3` and the ST4 table `ST4TABUHF2.bin` keep their sizes but
    differ by rounding: 36 % of the table values differ, by at most
    8 × 10⁻⁷ relative.
- `ww3_shel`: the first 6 hours of run `natl_v5k4_jan2026_exp` from the
  same cold start, forcing and open-boundary spectra (job 4116085):
  - run time 939 s on 32 ranks of one node;
  - over the 371,278 wet nodes, at every hour up to 6 h, the Datarmor
    output is matched to within 1.4 mm in $H_s$ (RMS difference at most
    3.3 × 10⁻⁶ m) and 4.7 ms in $T_{m02}$;
  - wind, currents and sea level read from the same `ww3_prnc` files are
    identical.

## Switches (`ww3/switch_hww3`)

```text
NOGRB PDLIB METIS DIST MPI PR3 UQ FLX0 LN1 ST4 STAB0 NL1 BT4 DB1 MLIM TR0 BS0
IC0 IS0 REF0 WNT2 WNX1 RWND CRT1 CRX1 O0 O1 O2 O2a O2b O2c O3 O4 O5 O6 O7
```

PDLIB with METIS domain decomposition, ST4 source terms, DIA nonlinear
interactions, ice as an obstruction only (IC0, IS0), currents and winds
interpolated in time (WNT2, CRT1).

## Grid and numerics

`ww3_grid.nml` and `namelists_<name>.nml` are written by `hww3-grid`:
36 directions × 32 frequencies from 0.0373 Hz, time steps 600 / 600 /
300 / 10 s, fully implicit scheme (`IMPTOTAL = T`, refraction, frequency
shift and sources implicit, block Jacobi–Gauss–Seidel solver,
`JGS_PMIN = 3 %`, at most 100 iterations). Depths are written positive in
the Gmsh file and made negative by `UNST%SF = -1`.

## Forcing

| Field | Source | Script |
|---|---|---|
| 10 m wind (hourly) | ERA5 single levels, `cdsapi` | `scripts/get_era5.py`, `scripts/prep_era5.py` |
| sea-ice concentration | ERA5 January mean, fixed in time (obstruction) | same |
| surface currents, sea level (hourly) | Copernicus Marine `GLOBAL_ANALYSISFORECAST_PHY_001_024`, 1/12°, read from the Datarmor mirror | `scripts/prep_cmems.py` |

The Datarmor mirror is
`/home/ref-ocean-model-public/multiparameter/physic/global/cmems/global-analysisforecast-phy-001-024/cmems_mod_glo_phy_anfc_0.083deg_PT1H-m/<year>/<day-of-year>/`;
several forecast runs exist per day and the latest one (largest `R` date)
is used. No Copernicus Marine account is needed on Datarmor.

`ww3_prnc` interpolates the regular lat/lon fields onto the nodes. Its
input files must have ascending latitudes, a CF time axis and a
`_FillValue` attribute on every field (`ww3_prnc` stops otherwise); land
values are set to 0.

## Pitfalls met on Datarmor

| Symptom | Cause | Fix |
|---|---|---|
| `error #6592` on `big_endian` in `constants.F90` | Intel fpp drops the quotes | set `FILE_ENDIAN` in the source |
| `ww3_grid`: "We have a dramatic inconsistency" | pinch (bow-tie) nodes | `mesh::remove_pinches` |
| open boundary read as empty, no error | bare node list | `READMSHOBC` format |
| `ww3_prnc`: `OFI addrinfo() failed` | MPI build of a serial tool on a 1-core node | `export I_MPI_FABRICS=shm` |
| `ww3_prnc`: `_FillValue ATTRIBUTE NOT DEFINED` | missing attribute | written by `prep_era5.py` / `prep_cmems.py` |
| `qsub` rejects `-q omp` with 1 core | queue minimum | `-q sequentiel` |
| `scp` closes the connection | no SFTP subsystem | `ssh datarmor cat file > file` |
