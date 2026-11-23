#!/bin/bash
#SBATCH -J hww3_tlcmp
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/jmdeloui/HWW3/logs/tlcmp_%j.log
#
# Milestone M3: the 48 h tangent runs listed in $S/m3/tl48.txt (lines
# "NAME JOBID" of scripts/m3_tl_run_hpc.slurm jobs) against the central
# differences of the 48 h finite-difference runs of $S/m2/fd/cases.txt
# (NAME_p1 and NAME_m1 for h = 0.01, NAME_p01 and NAME_m01 for h = 0.001
# where they exist), with scripts/m3_tl_compare.py (with both steps, per
# node on the nodes where the two differences agree to 1e-3).
#   sbatch --dependency=afterok:J1:J2:... scripts/m3_tl_cmp_hpc.slurm
S=/scratch/jmdeloui/HWW3
P=$S/pyenv/bin/python
C=$S/m2/fd/cases.txt
while read name job; do
  PAIRS=""
  for s in 1:0.01 01:0.001; do
    t=${s%%:*}
    h=${s#*:}
    jp=$(awk -v n=${name}_p$t '$1 == n { print $2 }' $C)
    jm=$(awk -v n=${name}_m$t '$1 == n { print $2 }' $C)
    if [ -n "$jp" ] && [ -n "$jm" ]; then PAIRS="$PAIRS $h:$S/m2/out_$jp:$S/m2/out_$jm"; fi
  done
  echo "=== $name: tangent $job, finite differences$PAIRS"
  # the H_s files of these runs are f32: per node, the two differences agree to about 1e-3 at best
  $P $S/scripts/m3_tl_compare.py $S/m3/tl_$job/tangent $PAIRS --tol 1e-3 --json $S/m3/cmp48_$name.json
done < $S/m3/tl48.txt
