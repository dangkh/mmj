#!/bin/bash
# Full content-vs-identity diagnosis for DRAGON and FREEDOM on one dataset, 3 seeds, in MAX_JOBS jobs.
#
#   bash scripts/submit_diag.sh                 # baby, seeds 999 1 2, 4 jobs x 2 tasks in parallel
#   DRY_RUN=1 bash scripts/submit_diag.sh       # print the task split only
#   SEEDS="999 1 2 3 4" MAX_JOBS=4 bash scripts/submit_diag.sh sports
#
# Conditions for the visual modality (data folders are created here if missing):
#   real    data/<ds>            real image features
#   perm    <ds>_vperm           image rows shuffled across items   (identity kept, content wrong)
#   noise   <ds>_vnoise          Gaussian rows                      (identity kept, no content)
#   const   <ds>_vconst          mean vector for every item         (no identity, no content)
#   id      <ds>_vnoise64        64-d noise; DRAGON: --learn_v_feat=True (learnable item-ID branch);
#                                FREEDOM fine-tunes its feature table anyway
# Text-graph versions: the item-item graph is built from TEXT ONLY (--mm_image_weight=0.0) in all of
# them, so only the modality branch differs (each has its own folder = its own graph cache):
#   real_tg  <ds>_vcopy        real image branch         perm_tg  <ds>_vperm_tg   wrong image
#   noise_tg <ds>_vnoise_tg    fixed random vectors      const_tg <ds>_vconst_tg  same vector for every item
#   id_tg    <ds>_vnoise64_tg  learnable 64-d item-ID table, random init with the usual Xavier scale
#            (std ~ sqrt(2/(n_items+64)) ~ 0.017), DRAGON --learn_v_feat=True
# Tasks of the same data folder go to the same job (they share the graph cache), started 2 min apart.
# Finished steps are skipped, so re-running this script only submits what is missing.
set -euo pipefail
cd "$(dirname "$0")/.."

DS="${1:-baby}"
SEEDS="${SEEDS:-999 1 2}"
MAX_JOBS="${MAX_JOBS:-4}"
PARALLEL="${PARALLEL:-2}"
HOURS_PER_TASK="${HOURS_PER_TASK:-1}"
CONDA_ENV="${CONDA_ENV:-Astar}"
DRY_RUN="${DRY_RUN:-0}"

[ -f "data/${DS}/${DS}.inter" ] || { echo "data/${DS} not set up"; exit 1; }

# ---- control data folders
mk() { [ -d "data/$1" ] && [ -f "src/configs/dataset/$1.yaml" ] || python tools/make_control_data.py -d "$DS" --modality v "${@:2}"; }
mk "${DS}_vperm"    --mode perm
mk "${DS}_vnoise"   --mode noise
mk "${DS}_vconst"   --mode const
mk "${DS}_vnoise64" --mode noise --dim 64
mk "${DS}_vcopy"    --mode copy
mk "${DS}_vperm_tg"    --mode perm  --suffix _tg
mk "${DS}_vnoise_tg"   --mode noise --suffix _tg
mk "${DS}_vconst_tg"   --mode const --suffix _tg
XAV=$(python -c "import numpy as np; n=np.load('data/${DS}/image_feat.npy', mmap_mode='r').shape[0]; print('%.4f' % (2.0 / (n + 64)) ** 0.5)")
mk "${DS}_vnoise64_tg" --mode noise --dim 64 --scale "$XAV" --suffix _tg

# ---- task list:  group  NAME MODEL DATASET SEED ARGS
DRAGON_ARGS="--learning_rate=0.0001 --reg_weight=0.001"
FREEDOM_ARGS="--dropout=0.8 --reg_weight=0.001"
TASKS=$(mktemp)
for s in $SEEDS; do
    # DRAGON keeps the names of the seed-999 runs already done (dragon_taps, dragon_vid, dragon_txtgraph)
    if [ "$s" = "999" ]; then dn=dragon_taps; dv=dragon_vid; dt=dragon_txtgraph; else dn=dragon_s$s; dv=dragon_vid_s$s; dt=dragon_txtgraph_s$s; fi
    fn=freedom_s$s; ft=freedom_txtgraph_s$s
    for d in "$DS" "${DS}_vperm" "${DS}_vnoise" "${DS}_vconst"; do
        [ -d "data/$d" ] || continue
        echo "$d $dn DRAGON $d $s $DRAGON_ARGS"   >> "$TASKS"
        echo "$d $fn FREEDOM $d $s $FREEDOM_ARGS" >> "$TASKS"
    done
    echo "${DS}_vnoise64 $dv DRAGON ${DS}_vnoise64 $s $DRAGON_ARGS --learn_v_feat=True" >> "$TASKS"
    echo "${DS}_vnoise64 $fn FREEDOM ${DS}_vnoise64 $s $FREEDOM_ARGS"                  >> "$TASKS"
    TG="--mm_image_weight=0.0"
    for d in "${DS}_vperm_tg" "${DS}_vnoise_tg" "${DS}_vconst_tg"; do
        echo "$d $dn DRAGON $d $s $DRAGON_ARGS $TG"   >> "$TASKS"
        echo "$d $fn FREEDOM $d $s $FREEDOM_ARGS $TG" >> "$TASKS"
    done
    echo "${DS}_vnoise64_tg $dv DRAGON ${DS}_vnoise64_tg $s $DRAGON_ARGS $TG --learn_v_feat=True" >> "$TASKS"
    echo "${DS}_vnoise64_tg $fn FREEDOM ${DS}_vnoise64_tg $s $FREEDOM_ARGS $TG"                  >> "$TASKS"
    echo "${DS}_vcopy $dt DRAGON ${DS}_vcopy $s $DRAGON_ARGS --mm_image_weight=0.0"    >> "$TASKS"
    echo "${DS}_vcopy $ft FREEDOM ${DS}_vcopy $s $FREEDOM_ARGS --mm_image_weight=0.0"  >> "$TASKS"
done

# drop tasks that are completely done
TODO=$(mktemp)
while read -r G NAME MODEL D SEED ARGS; do
    done_train=0; grep -l 'BEST ████' src/log/"${MODEL}-${D}-${NAME}"-*.log >/dev/null 2>&1 && done_train=1
    done_probe=0; [ -f "log/probe_knn_${D}_${MODEL}_${NAME}.json" ] && done_probe=1
    done_gc=1; [ "$MODEL" = "DRAGON" ] && [ ! -f "log/graph_control_${D}_${MODEL}-${D}-${NAME}.json" ] && done_gc=0
    if [ $done_train = 1 ] && [ $done_probe = 1 ] && [ $done_gc = 1 ]; then continue; fi
    echo "$G $NAME $MODEL $D $SEED $ARGS" >> "$TODO"
done < "$TASKS"
N=$(grep -c . "$TODO" || true)
[ "$N" -gt 0 ] || { echo "Everything is done."; exit 0; }

# ---- split by data folder (greedy: largest group to the least loaded job)
STAMP=$(date '+%Y%m%d-%H%M%S'); QDIR="src/log/queue/diag-${DS}-${STAMP}"; mkdir -p "$QDIR"
W=$(( N < MAX_JOBS ? N : MAX_JOBS ))
declare -a LOAD; for ((w = 0; w < W; w++)); do LOAD[$w]=0; : > "$QDIR/worker$w.txt"; done
for G in $(awk '{print $1}' "$TODO" | sort | uniq -c | sort -rn | awk '{print $2}'); do
    best=0; for ((w = 1; w < W; w++)); do [ "${LOAD[$w]}" -lt "${LOAD[$best]}" ] && best=$w; done
    awk -v g="$G" '$1 == g {$1 = ""; sub(/^ /, ""); print}' "$TODO" | sort -k4,4n >> "$QDIR/worker$best.txt"
    LOAD[$best]=$(( LOAD[$best] + $(awk -v g="$G" '$1 == g' "$TODO" | wc -l) ))
done
rm -f "$TASKS" "$TODO"

echo "Dataset $DS | seeds: $SEEDS | $N tasks -> $W job(s) x $PARALLEL in parallel | $QDIR"
for ((w = 0; w < W; w++)); do
    K=$(grep -c . "$QDIR/worker$w.txt" || true); [ "$K" -gt 0 ] || continue
    WT=$(printf '%02d:00:00' $(( (K + PARALLEL - 1) / PARALLEL * HOURS_PER_TASK + 2 )))
    CMD=(qsub -N "dg${DS:0:2}_w${w}" -l "walltime=${WT}" -v "TASK_FILE=$QDIR/worker$w.txt,CONDA_ENV=$CONDA_ENV,PARALLEL=$PARALLEL" scripts/diag_worker.pbs)
    if [ "$DRY_RUN" = "1" ]; then echo "[dry-run] ${CMD[*]}"; JID=DRYRUN; else JID=$("${CMD[@]}"); fi
    printf '  job %d (%s, %d tasks, walltime %s): %s\n' "$w" "$JID" "$K" "$WT" "$(awk '{printf "%s:%s:s%s ", $2, $3, $4}' "$QDIR/worker$w.txt")"
done
echo "Summary when done: python tools/summarize_diag.py -d $DS"
