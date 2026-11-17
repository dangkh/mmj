#!/bin/bash
# Submit the Text -> ID consolidation of one or more datasets as separate jobs (scripts/consolidate.pbs):
#   teacher_id, teacher_text             in parallel (skipped if their taps exist)
#   grid B2, grid B3 lam_J=0.1, grid B3 lam_J=1     in parallel, each starting when both teachers succeeded
# That is up to 5 jobs per dataset. Seeds: SEEDS (default 999) -- the grid uses its first seed.
#
#   bash scripts/submit_consolidate.sh dbbook
#   QUEUE=med_gpuq bash scripts/submit_consolidate.sh baby
#   DRY_RUN=1 bash scripts/submit_consolidate.sh dbbook
# Afterwards (summary only, a few seconds):  qsub -v DS=dbbook,STEP=summary scripts/consolidate.pbs
#   or directly: python tools/consolidate.py -d dbbook --summary --id_taps taps/LightGCNC-dbbook-teacher_id \
#                    --text_taps taps/LightGCNC-dbbook-teacher_text
# More seeds later:  qsub -v DS=dbbook,STEP=finals,SEEDS="999 1 2" scripts/consolidate.pbs
set -eu
cd "$(dirname "$0")/.."
QUEUE="${QUEUE:-small_gpuq}"
SEEDS="${SEEDS:-999}"
mkdir -p results/consolidate/logs

sub() {   # sub NAME DEPEND VARS -> prints the job id
    local NAME=$1 DEP=$2 VARS=$3 CMD
    CMD=(qsub -q "$QUEUE" -N "$NAME" -o "results/consolidate/logs/${NAME}.log" -j oe)
    if [ -n "$DEP" ]; then CMD+=(-W "depend=afterok:$DEP"); fi
    CMD+=(-v "$VARS" scripts/consolidate.pbs)
    echo "${CMD[*]}" >&2
    if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRYRUN_${NAME}"; else "${CMD[@]}"; fi
}

for DS in "$@"; do
    DEPS=()
    for SRC in id text; do
        if [ -f "taps/LightGCNC-${DS}-teacher_${SRC}/T3.npy" ]; then
            echo "[skip] teacher $SRC of $DS exists" >&2
        else
            DEPS+=("$(sub "ct_${DS}_t${SRC}" "" "DS=$DS,STEP=teacher,SRC=$SRC")")
        fi
    done
    DEP=$(IFS=:; echo "${DEPS[*]:-}")
    sub "ct_${DS}_B2" "$DEP" "DS=$DS,STEP=grid,MODES=B2,SEEDS=$SEEDS" > /dev/null
    sub "ct_${DS}_B3a" "$DEP" "DS=$DS,STEP=grid,MODES=B3,LAMJ=0.1,SEEDS=$SEEDS" > /dev/null
    sub "ct_${DS}_B3b" "$DEP" "DS=$DS,STEP=grid,MODES=B3,LAMJ=1,SEEDS=$SEEDS" > /dev/null
done
qstat -u "${USER:-$(whoami)}" 2>/dev/null || true
