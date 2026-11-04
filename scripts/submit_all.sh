#!/bin/bash
# Submit experiments as at most MAX_JOBS PBS jobs (default 2: every job waits in the queue, so
# fewer, fuller jobs start sooner). Experiments are spread round-robin over the jobs; each job
# runs PARALLEL of its experiments at the same time on its GPU (default 2).
#
#   bash scripts/submit_all.sh baby scripts/experiments_baby_repro.txt
#   PARALLEL=3 bash scripts/submit_all.sh ...       # 3 experiments at a time per GPU
#   MAX_JOBS=1 PARALLEL=4 bash scripts/submit_all.sh ...
#   ONLY="dragon_paper mgcn_grid" bash scripts/submit_all.sh baby scripts/experiments_baby_repro.txt
#   MAX_JOBS=2 bash scripts/submit_all.sh ...        # fewer jobs (e.g. 2 already running)
#   HOURS_PER_EXP=8 bash scripts/submit_all.sh ...   # walltime = HOURS_PER_EXP x ceil(#experiments / PARALLEL)
#   WALLTIME=24:00:00 bash scripts/submit_all.sh ... # or set the walltime directly
#   DRY_RUN=1 bash scripts/submit_all.sh ...         # print only
#
# Experiment file lines:  NAME  MODEL  [--key=value ...]   (# comments, blank lines ignored)
# Re-running skips experiments that already have a finished log in src/log/.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-baby}"
CFG_FILE="${2:-scripts/experiments_${DATASET}_repro.txt}"
CONDA_ENV="${CONDA_ENV:-Astar}"
MAX_JOBS="${MAX_JOBS:-2}"
PARALLEL="${PARALLEL:-2}"
HOURS_PER_EXP="${HOURS_PER_EXP:-6}"
WALLTIME="${WALLTIME:-}"
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-0}"

[ -f "$CFG_FILE" ] || { echo "Experiment file not found: $CFG_FILE"; exit 1; }
[ -f "data/${DATASET}/${DATASET}.inter" ] || { echo "data/${DATASET} not set up: bash scripts/setup_data.sh ${DATASET}"; exit 1; }

mapfile -t ALL < <(grep -v '^[[:space:]]*#' "$CFG_FILE" | tr -d '\r' | grep -v '^[[:space:]]*$')
TODO=()
for L in "${ALL[@]}"; do
    NAME=$(echo "$L" | awk '{print $1}')
    MODEL=$(echo "$L" | awk '{print $2}')
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $NAME "* ]]; then continue; fi
    if grep -l 'BEST ████' src/log/"${MODEL}-${DATASET}-${NAME}"-*.log >/dev/null 2>&1; then
        echo "  skip $NAME (finished log exists)"; continue
    fi
    TODO+=("$L")
done
N=${#TODO[@]}
[ "$N" -gt 0 ] || { echo "Nothing to run."; exit 0; }

if command -v qstat >/dev/null 2>&1; then
    MINE=$(qstat -u "$USER" 2>/dev/null | awk 'NR>5 && $10 ~ /^[QRHW]$/' | wc -l || true)
    if [ "${MINE:-0}" -gt 0 ]; then
        echo "Note: you already have $MINE job(s) queued/running; consider MAX_JOBS=$((MAX_JOBS - MINE))."
    fi
fi

W=$(( N < MAX_JOBS ? N : MAX_JOBS ))
STAMP=$(date '+%Y%m%d-%H%M%S')
QDIR="src/log/queue/${DATASET}-${STAMP}"
mkdir -p "$QDIR"
for ((w = 0; w < W; w++)); do : > "$QDIR/worker${w}.txt"; done
for i in "${!TODO[@]}"; do echo "${TODO[$i]}" >> "$QDIR/worker$((i % W)).txt"; done

echo "Dataset $DATASET | $N experiments from $CFG_FILE -> $W job(s) x $PARALLEL in parallel | task files in $QDIR"
for ((w = 0; w < W; w++)); do
    K=$(grep -c . "$QDIR/worker${w}.txt")
    if [ -n "$WALLTIME" ]; then WT="$WALLTIME"; else
        WT=$(printf '%02d:00:00' $(( (K + PARALLEL - 1) / PARALLEL * HOURS_PER_EXP + 1 ))); fi
    VARS="TASK_FILE=$QDIR/worker${w}.txt,DATASET=${DATASET},CONDA_ENV=${CONDA_ENV},PARALLEL=${PARALLEL}"
    CMD=(qsub -N "mm${DATASET:0:2}_w${w}_${STAMP: -4}" -l "walltime=${WT}" -v "$VARS" scripts/run_worker.pbs)
    if [ "$DRY_RUN" = "1" ]; then echo "[dry-run] ${CMD[*]}"; JID="DRYRUN.w${w}"; else JID=$("${CMD[@]}"); fi
    printf '  job %d (%s, walltime %s): %s\n' "$w" "$JID" "$WT" "$(awk '{print $1}' "$QDIR/worker${w}.txt" | xargs)"
done
echo "Progress: tail -f <jobname>.o<id>   Results: python scripts/collect_results.py -d ${DATASET}"
