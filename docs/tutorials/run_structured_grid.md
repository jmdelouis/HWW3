# Tutorial: run a WW3 structured-grid configuration on the GPU

This tutorial runs the GPU model `hww3-prop` on the three grids of an
existing WW3 configuration: the IRI global configuration, with a regular
grid of 0.125° (IRIGLOB-7M) and two curvilinear polar grids of 15 km
(ARC-15KM, ANTARC-15KM). Its steps are:

1. copy the inputs from Datarmor to the hpc cluster;
2. mesh the grids;
3. write the forcing on the meshes;
4. run one A100 per grid;
5. compare $H_s$ with WW3.

The design, the physics left out and the results are in
[WW3 structured grids](../structured_grids.md).

## Prerequisites

- Accounts on Datarmor (read access to `/home/datawork-WW3`) and on the hpc
  cluster (the `gpu` partition).
- On hpc, the working directory `S=/scratch/jmdeloui/HWW3`, holding:
  - a copy of the repository in `$S/struct/hww3-wave` and the scripts in
    `$S/struct/scripts`;
  - the Rust toolchain in `$S/rust`;
  - the Python environment `$S/pyenv`, with `netCDF4` and `numpy`.
- The job scripts read `S=/scratch/jmdeloui/HWW3` and write their logs to
  `$S/struct/logs`. Another user changes these lines at the top of
  `scripts/struct_run_hpc.slurm` and `scripts/struct_probe_hpc.slurm`.

The WW3 run used here lives in

```text
W=/home/datawork-WW3/PROJECT/IRI/IRI_15KM_L4SAT_11/bmax157/111901/IRIGLOB-7M/2019-01-01T00_2019-05-01T00
```

It holds:

| Directory | Contents |
|---|---|
| `work/` | `ww3_grid.nml.<grid>`, `namelists_<grid>.nml`, `ww3_multi.nml`, `log.mww3` (wall time) |
| `data/` | `<grid>.bot`, `.mask`, `.obst`, and `.lon`/`.lat` for the curvilinear grids (128 MB) |
| `FIELD_NC/` | the hourly outputs of IRIGLOB-7M, `WW3-IRIGLOB-7M_2019MM.nc`, with the wind and currents WW3 used |

## 1. Copy the inputs

The grid files and namelists are small. Copy `work/` and `data/` to
`$S/struct/iri/` on hpc.

The forcing files are large: a year of ERA5 wind is 21 GB, and a year of
CMEMS currents 25 GB. Cut the week you need on Datarmor scratch, never on
datawork:

```bash
source /usr/share/Modules/3.2.10/init/bash
module load nco/4.7.1_conda
D=/home1/scratch/$USER/HWW3/cut; mkdir -p $D; cd $D
F=/home/datawork-WW3/FORCING
R=$W/FIELD_NC
T="-d time,2019-01-01T00:00:00,2019-01-08T00:00:00"
nice ncks -O -4 -L 1 $T -v uwnd,vwnd,ucur,vcur,hs,MAPSTA $R/WW3-IRIGLOB-7M_201901.nc field_IRIGLOB-7M_20190101_08.nc
nice ncks -O -4 -L 1 $T $F/ERA5/NC4/wind_2019.nc wind_20190101_08.nc
nice ncks -O -4 -L 1 $T $F/ERA5/NC4/ice_2019.nc ice_20190101_08.nc
nice ncks -O -4 -L 1 $T $F/CMEMS_GLOBCUR_REP/NC4/cur_2019.nc cur_20190101_08.nc
md5sum *.nc > md5.txt
```

Copy the four files and `md5.txt` to `$S/struct/forcing_nc/` on hpc, then
check them there with `md5sum -c md5.txt`.

**Expected:** about 2.3 GB for the week:

| File | Size |
|---|---|
| `field_IRIGLOB-7M_20190101_08.nc` | 696 MB |
| `wind_20190101_08.nc` | 382 MB |
| `ice_20190101_08.nc` | 30 MB |
| `cur_20190101_08.nc` | 1.23 GB |

Copy whole files with `scp` or `rsync`, from a shell that runs `bash`. The
login shells of both clusters are `csh`, where an appending loop written
for `bash` can append the same file several times.

## 2. Build the programs

On the hpc login node:

```bash
S=/scratch/jmdeloui/HWW3
source /etc/profile.d/modules.sh
module load cuda/13.0.0
cd $S/struct/hww3-wave
export CARGO_HOME=$S/rust/cargo PATH=$S/rust/stable-x86_64-unknown-linux-gnu/bin:$PATH
export CARGO_TARGET_DIR=$S/struct/target_gpu CUDARC_CUDA_VERSION=13000
cargo build --release --bin hww3-src --bin hww3-prop --bin hww3-pre
```

**Expected:** `hww3-src`, `hww3-prop` and `hww3-pre` in
`$S/struct/target_gpu/release`.

## 3. Mesh the grids

```bash
B=$S/struct/target_gpu/release; I=$S/struct/iri
for g in IRIGLOB-7M ARC-15KM ANTARC-15KM; do
  $B/hww3-pre struct $I/work/ww3_grid.nml.$g --data $I/data --out $S/struct/grids/$g
done
```

`--rotate auto`, the default, meshes ARC-15KM in rotated coordinates, with
the pole moved to the equator.

**Expected:** the sea points of `ww3_grid`, less a few isolated points:

| Grid | Nodes |
|---|---|
| IRIGLOB-7M | 1,189,834 |
| ARC-15KM | 358,100 |
| ANTARC-15KM | 529,404 |

Check them against `output/ww3_grid.<grid>.out` of the WW3 run.

## 4. Write the forcing on the meshes

**IRIGLOB-7M:** take the wind and currents WW3 used, straight from its
output, with no interpolation:

```bash
PY=$S/pyenv/bin/python; cd $S/struct
$PY scripts/struct_fieldnc.py forcing_nc/field_IRIGLOB-7M_20190101_08.nc grids/IRIGLOB-7M forcing/IRIGLOB-7M
```

**The polar grids:** interpolate ERA5 and CMEMS with the port of
`ww3_prnc`. The wind and current vectors of ARC-15KM are turned into its
rotated frame.

```bash
mkdir -p raw_nc
$PY scripts/prnc_nc.py forcing_nc/wind_20190101_08.nc u10+v10 raw_nc/wind
$PY scripts/prnc_nc.py forcing_nc/cur_20190101_08.nc uo+vo raw_nc/cur
$PY scripts/prnc_nc.py forcing_nc/ice_20190101_08.nc siconc raw_nc/ice
for g in ARC-15KM ANTARC-15KM; do
  mkdir -p forcing/$g
  $B/hww3-pre prnc grids/$g --field wind --input raw_nc/wind --out forcing/$g/wind.ww3
  $B/hww3-pre prnc grids/$g --field current --input raw_nc/cur --out forcing/$g/current.ww3
  $B/hww3-pre prnc grids/$g --field ice --input raw_nc/ice --out forcing/$g/ice.ww3
done
```

**Expected:** `forcing/IRIGLOB-7M/{wind,current}.ww3`, and
`forcing/<polar grid>/{wind,current,ice}.ww3`.

## 5. Run

`scripts/struct_run_hpc.slurm` builds the source parameters from the
namelists of `work/`. It replaces `SDSBCHOICE = 3` and `SDSCUM = 0.3`,
which the GPU kernels do not have, by `SDSBCHOICE = 1` and `SDSCUM = 0`.
It then runs `hww3-prop` from calm conditions on one A100:

```bash
sbatch -t 04:00:00 --export=ALL,GRID=IRIGLOB-7M,HOURS=72,FORCE=$S/struct/forcing/IRIGLOB-7M scripts/struct_run_hpc.slurm
sbatch --export=ALL,GRID=ARC-15KM,HOURS=12,FORCE=$S/struct/forcing/ARC-15KM scripts/struct_run_hpc.slurm
sbatch --export=ALL,GRID=ANTARC-15KM,HOURS=12,FORCE=$S/struct/forcing/ANTARC-15KM scripts/struct_run_hpc.slurm
```

Options, set with `--export`:

| Option | Effect |
|---|---|
| `PREC=f64` | double precision; at 36 frequencies only one node fits per multiprocessor, so it costs more than twice single precision |
| `NML=FILE` | another namelist file, for example with `FXFM3 = 2.5` or the parameters of the North Atlantic runs |
| `PROFILE=1` | the time of each part of the step |
| without `FORCE` | an idealised wind from `hww3-pre ideal-wind` |

Submit the jobs one after the other (`--dependency=afterany:<job>`) to use
one GPU at a time. The script asks for 1:30, which covers 12 h of a polar
grid. A 72 h run of IRIGLOB-7M takes about 3 h with the IRI parameters,
hence the `-t 04:00:00` above.

**Expected:** in `$S/struct/logs/struct_run_<job>.log`, the time per model
day in single precision with the IRI parameters:

| Grid | s per model day |
|---|---|
| IRIGLOB-7M | about 3,500 |
| ARC-15KM | about 1,130 |
| ANTARC-15KM | about 1,800 |

For comparison, WW3 takes 985 s per model day for the three grids on 252
processes, from the wall time at the end of `work/log.mww3` divided by the
140 model days. With the North Atlantic parameters, IRIGLOB-7M takes about
340 s per model day.

To count the source sub-steps behind these times, run
`sbatch --export=ALL,CASES=iri,HOURS=1 scripts/struct_probe_hpc.slurm`
from the restart of 4 January. The runs write their hourly outputs to
`$S/struct/runs/<grid>_<prec>_<job>/out`.

## 6. Compare with WW3

```bash
$PY scripts/struct_compare.py runs/IRIGLOB-7M_f32_<job>/out grids/IRIGLOB-7M forcing_nc/field_IRIGLOB-7M_20190101_08.nc --every 12
```

The script gives the mean $H_s$ of both models, the bias, the RMS
difference, the scatter index and the correlation, overall and by
latitude band.

**Expected:** on 4 January 00 h, bias +0.03 m, RMSE 0.48 m, correlation
0.89. These still improve, since WW3 had started 20 days earlier.

The scheme is the first-order N scheme on triangles, not WW3's UQ, and the
ice, reflection and obstruction terms are not ported. Do not expect
agreement to the micrometre, as on the HEALPix grids. The FIELD_NC files
of the polar grids are not in this run directory, so they have no $H_s$
comparison.
