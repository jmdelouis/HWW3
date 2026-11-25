#!/bin/bash
#SBATCH -J hww3_ptx
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/opt_ptx_%j.log
#
# Registers and spills of the source kernels (docs/wave_results.md,
# optimisation): for each build of BINS (directories of $S/opt separated by
# :), the PTX of the kernels that hww3-src check-gpu (f32 and f64 value
# kernels) and check-tl (f64 tangent kernel) compile, written by CubeCL
# under CUBECL_CUDA_DUMP_PTX, then ptxas -v for sm_80 on the source kernel
# (the largest PTX of the run, NLARGE largest with NLARGE > 1):
# registers per unit, spill stores and loads, and the same with the
# registers capped at CAPS (default 128 and 255), which is what a launch
# bound of 4 nodes per SM in f32 would ask the compiler.
#   sbatch --export=ALL,BINS=bin_spw:bin_sin:bin_db1 scripts/opt_ptx_hpc.slurm
# With DUMPED=1 the PTX already written under $S/opt/ptx is analysed again
# without running the kernels (ptxas needs no GPU).
S=/scratch/jmdeloui/HWW3
H=$S/m2/harness/src_8jan_full.bin
O=$S/opt/ptx
source /etc/profile.d/modules.sh
module load cuda/13.0.0
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
[ -n "$DUMPED" ] || nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
mkdir -p $O
cd $S
for b in ${BINS//:/ }; do
  for k in f32 f64 tl; do
    d=$O/$b/$k
    [ -z "$DUMPED" ] && rm -rf $d
    mkdir -p $d
    [ -n "$DUMPED" ] || case $k in
      f32) CUBECL_CUDA_DUMP_PTX=$d $S/opt/$b/hww3-src check-gpu $H --replicate 1 > /dev/null 2>&1 ;;
      f64) CUBECL_CUDA_DUMP_PTX=$d $S/opt/$b/hww3-src check-gpu $H --f64 --replicate 1 > /dev/null 2>&1 ;;
      tl) CUBECL_CUDA_DUMP_PTX=$d $S/opt/$b/hww3-src check-tl $H --h 1e-4 > /dev/null 2>&1 ;;
    esac
    echo "$b $k: $(ls $d/*.ptx 2>/dev/null | wc -l) kernels"
    # the source kernel is the largest PTX of the run
    for f in $(ls -S $d/*.ptx 2>/dev/null | head -${NLARGE:-1}); do
      n=$(basename $f .ptx | tail -c 60)
      # the NVRTC that CubeCL loads writes PTX 9.2, newer than the ptxas of
      # the cuda/13.0.0 module (9.0): the version line is lowered, the
      # instructions of sm_80 being the same
      sed -E '1,30s/^\.version [0-9.]+/.version 9.0/' $f > $d/k.ptx9
      for cap in "" ${CAPS:-128 255}; do
        opt=""; [ -n "$cap" ] && opt="--maxrregcount=$cap"
        r=$(ptxas -arch=sm_80 -v $opt $d/k.ptx9 -o /dev/null 2>&1 | grep -E "registers|spill" | tr '\n' ' ' | sed 's/ptxas info *: *//g; s/  */ /g')
        echo "$b $k cap=${cap:-none} ...$n: $r"
      done
    done
  done
done
