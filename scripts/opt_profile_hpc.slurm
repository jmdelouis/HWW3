#!/bin/bash
#SBATCH -J hww3_opt
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/opt_profile_%j.log
#
# Profile of hww3-prop on grid v5 K=4 with the full ST4, from the restart of
# 1 January 00:10 and the forcing of the calibration runs (docs/wave_results.md,
# optimisation), in $S/opt/$TAG:
#   PART=nsys  short runs under Nsight Systems (value f32 over HOURS, value
#              f64 and the BETAMAX tangent over HOURS/2): time per kernel,
#              API calls, copies; then the same runs with --profile, which
#              synchronises at the end of each part of the step;
#   PART=ncu   Nsight Compute on a few launches of the kernels of NCU_LIST
#              (default: the source kernel in f32, f64 and the f64
#              tangent, the geographic sweeps, W3KTP3 and the sub-cycling
#              count) in runs of 1 h: occupancy, memory and compute
#              throughput, launch configuration, warp states.
#   PART=regs  registers per unit and shared memory of the source kernels
#              at run time (value f32 and f64, f64 and f32 tangents);
#   PART=units the units per cube of the source kernel (UNITS_LIST): time
#              and H_s against the first value, in f32 and f64.
#   sbatch --export=ALL,PART=nsys,TAG=base scripts/opt_profile_hpc.slurm
#   sbatch -t 01:00:00 --export=ALL,PART=ncu,TAG=base scripts/opt_profile_hpc.slurm
S=/scratch/jmdeloui/HWW3
D=$S/m2/forcing
G=$S/grids/natl_v5k4
B=${BIN:-$S/target_wave_tl/release/hww3-prop}
P=$S/m2/harness/src_8jan_full.bin.params
R=$S/runs/m2_week_start_full/restart001.ww3
O=$S/opt/${TAG:-base}
H=${HOURS:-6}
source /etc/profile.d/modules.sh
module load cuda/13.0.0
export CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,clocks.max.mem --format=csv
mkdir -p $O
FORCE="--current $D/current.ww3 --level $D/level.ww3 --ice $D/ice.ww3 --nest $D/nest.ww3 --wind $D/wind.ww3 --sources $P --units ${UNITS:-128}"
run() {
  # run TAG HOURS [options]: one run of hww3-prop, its summary line
  local t=$1 h=$2
  shift 2
  $B run $G --restart $R $FORCE --hours $h --out $O/$t "$@" 2>&1 | grep -E "steps in|rror|panicked"
}
case $PART in
  nsys)
    for c in "value_f32:$H:" "value_f64:$((H / 2)):--f64" "tangent_f64:$((H / 2)):--f64 --tangent BETAMAX"; do
      t=${c%%:*}; r=${c#*:}; h=${r%%:*}; opt=${r#*:}
      echo "== $t, $h h, nsys"
      nsys profile -t cuda,osrt --sample=none --cpuctxsw=none --force-overwrite true -o $O/$t \
        $B run $G --restart $R $FORCE --hours $h --out $O/run_$t $opt 2>&1 | grep -E "steps in|rror|panicked|Generated"
      nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
        --format csv --output $O/${t} $O/$t.nsys-rep > /dev/null 2>&1
      ls $O/${t}_*.csv
      echo "== $t, $h h, --profile"
      run prof_$t $h $opt --profile
    done
    ;;
  regs)
    # registers per unit and shared memory of the source kernels as the
    # driver compiles them at run time: runs of HOURS (default 1) under
    # Nsight Systems in f32, f64 and the f64 and f32 BETAMAX tangents, the
    # first two launches of the source kernel in the GPU trace (Reg/Trd,
    # DymSMem)
    for c in "value_f32:" "value_f64:--f64" "tangent_f64:--f64 --tangent BETAMAX" "tangent_f32:--tangent BETAMAX"; do
      t=${c%%:*}; opt=${c#*:}
      echo "== $t"
      nsys profile -t cuda --sample=none --cpuctxsw=none --force-overwrite true -o $O/regs_$t \
        $B run $G --restart $R $FORCE --hours ${HOURS:-1} --out $O/run_regs_$t $opt 2>&1 | grep -E "steps in|rror|panicked"
      nsys stats --force-export=true --report cuda_gpu_trace --format csv --output $O/regs_$t $O/regs_$t.nsys-rep > /dev/null 2>&1
      head -1 $O/regs_${t}_cuda_gpu_trace.csv
      grep -E "srce" $O/regs_${t}_cuda_gpu_trace.csv | head -2
    done
    ;;
  ncu)
    # NCU_LIST: TAG:REGEX:SKIP:COUNT:MODE separated by ;, MODE f32, f64 or
    # tl (BETAMAX tangent in f64); SKIP launches of the matching kernels are
    # skipped (start-up), then COUNT are profiled, in a run of 1 h
    L=${NCU_LIST:-"srce32:^srce_f_f32:2:1:f32;geo32:^geo_(node|tri)_x4_f_f32:600:4:f32;ktp32:^(ktp3_k|ktp3_theta|iter_partial)_f_f32:4:3:f32;srce64:^srce_f_f64:1:1:f64;srcetl:^srce_tl_f_f64:1:1:tl"}
    IFS=';' read -ra items <<< "$L"
    for it in "${items[@]}"; do
      IFS=':' read -r t rx sk cn md <<< "$it"
      case $md in f64) opt="--f64" ;; tl) opt="--f64 --tangent BETAMAX" ;; *) opt="" ;; esac
      echo "== ncu $t: $rx, skip $sk, count $cn, $md"
      ncu --kernel-name "regex:$rx" --launch-skip $sk --launch-count $cn \
        --section SpeedOfLight --section Occupancy --section LaunchStats --section MemoryWorkloadAnalysis \
        --section ComputeWorkloadAnalysis --section SchedulerStats --section WarpStateStats \
        --csv --page details --log-file $O/ncu_$t.csv \
        $B run $G --restart $R $FORCE --hours ${HOURS:-1} --out $O/run_ncu_$t $opt 2>&1 | grep -E "steps in|rror|panicked|==ERROR==|==WARNING==" | head -8
      wc -l $O/ncu_$t.csv
    done
    ;;
  units)
    # the units per cube of the source kernel: time over HOURS (f32) and
    # HOURS/2 (f64) for each value of UNITS_LIST, and H_s at the last hour
    # against the first value of the list
    FORCE=${FORCE% --units *}
    for md in f32 f64; do
      h=$H; opt=""
      [ $md = f64 ] && { h=$((H / 2)); opt="--f64"; }
      ref=""
      for u in ${UNITS_LIST:-128 256 512 64}; do
        echo "== $md, $u units per cube, $h h"
        $B run $G --restart $R $FORCE --units $u --hours $h --out $O/units_${md}_$u --out64 $opt 2>&1 | grep -E "steps in|rror|panicked"
        last=$(ls $O/units_${md}_$u/hs_*.f64 | tail -1)
        if [ -z "$ref" ]; then ref=$last; else
          $S/pyenv/bin/python -c "
import numpy as np, sys
a = np.fromfile('$ref', '<f8'); b = np.fromfile('$last', '<f8'); m = a > -900
d = np.abs(a - b)[m]
print('   H_s at %s against the first: max %.3e m, rms %.3e m, %d of %d nodes differ' % ('$last'.split('_')[-1][:10], d.max(), np.sqrt((d ** 2).mean()), (d > 0).sum(), m.sum()))"
        fi
      done
    done
    ;;
  *) echo "PART nsys, ncu or units"; exit 1 ;;
esac
