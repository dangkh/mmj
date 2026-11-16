#!/bin/bash
# Submit the JEPA-Y E-A diagnosis: one job per dataset (scripts/jepay_diag.pbs), each running its steps
# sequentially. Re-running is safe: finished steps / runs are skipped inside the job.
#
#   bash scripts/submit_jepay.sh                         # baby ml1m dbbook
#   bash scripts/submit_jepay.sh baby                    # one dataset
#   SMOKE=1 bash scripts/submit_jepay.sh baby            # quick check (3 epochs, seed 0, mlp1), 1h walltime
#   QUEUE=med_gpuq bash scripts/submit_jepay.sh dbbook   # when small_gpuq is full
#   TEACHER_ARGS="--n_layers=3 --reg_weight=0.0001" bash scripts/submit_jepay.sh ml1m   # fixed teacher setting
#   DRY_RUN=1 bash scripts/submit_jepay.sh               # print the qsub commands only
#
# After all jobs:  python tools/summarize_jepay_diag.py --all baby ml1m dbbook   (Holm, gate, figures)
set -euo pipefail
cd "$(dirname "$0")/.."
DATASETS="${*:-baby ml1m dbbook}"
QUEUE="${QUEUE:-small_gpuq}"
MEM="${MEM:-16gb}"
NCPUS="${NCPUS:-8}"
WALLTIME="${WALLTIME:-24:00:00}"
SMOKE="${SMOKE:-0}"
SUF=""
if [ "$SMOKE" = "1" ]; then WALLTIME="${SMOKE_WALLTIME:-01:00:00}"; SUF="_smoke"; fi
mkdir -p results/jepay_diag/logs
for DS in $DATASETS; do
    [ -f "data/$DS/$DS.inter" ] || { echo "skip $DS: data/$DS/$DS.inter not found"; continue; }
    [ -f "src/configs/dataset/$DS.yaml" ] || { echo "skip $DS: src/configs/dataset/$DS.yaml not found"; continue; }
    VARS="DS=$DS,SMOKE=$SMOKE"
    if [ -n "${TEACHER_ARGS:-}" ]; then VARS="$VARS,TEACHER_ARGS=$TEACHER_ARGS"; fi
    CMD=(qsub -q "$QUEUE" -N "jepay_$DS" -l "select=1:ncpus=$NCPUS:ngpus=1:mem=$MEM" -l "walltime=$WALLTIME"
         -o "results/jepay_diag/logs/jepay_${DS}${SUF}.log" -j oe -v "$VARS"
         scripts/jepay_diag.pbs)
    echo "${CMD[*]}"
    if [ "${DRY_RUN:-0}" != "1" ]; then "${CMD[@]}"; fi
done
