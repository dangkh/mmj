#!/bin/bash
# Content-vs-identity diagnosis: MODELS x CONDS x SEEDS on one or more datasets, in at most MAX_JOBS jobs.
#
#   bash scripts/submit_diag.sh sports clothing baby          # default models / conditions / seeds
#   MODELS="MENTOR" bash scripts/submit_diag.sh baby
#   MODELS="COHESION LGMRec" bash scripts/submit_diag.sh baby   # Recall-only models (no tap adapter)
#   CONDS="real perm noise const real_tg const_tg" bash scripts/submit_diag.sh sports
#   DRY_RUN=1 bash scripts/submit_diag.sh sports              # show the split, submit nothing
#   MEM=32gb PARALLEL=1 bash scripts/submit_diag.sh clothing    # resources per job (default 16gb, 4 cpus)
#
# Two stages, both handled by re-running the same command:
#   1. FREEDOM / MENTOR are first tuned on the real data with their official grid (name freedom_grid /
#      mentor_grid, selection on valid). Their control runs are added once that tuning run has finished.
#      DRAGON uses the official per-dataset setting (lr 1e-4; reg 1e-3, Clothing 0.1).
#   2. Every condition x seed with the selected hyper-parameters.
# Tasks that are finished, or listed in the task file of one of your queued/running jobs, are skipped,
# so re-running is safe at any time.
#
# Conditions for the visual modality (data folders are created if missing):
#   real     data/<ds>             real image features
#   perm     <ds>_vperm            image rows shuffled across items   (identity kept, content wrong)
#   noise    <ds>_vnoise           Gaussian rows                      (identity kept, no content)
#   const    <ds>_vconst           mean vector for every item         (no identity, no content)
#   ones     <ds>_vones            all-ones rows   } knockout protocol of Ye et al. (ECIR'26), to compare
#   gauss    <ds>_vgauss           N(0,1) rows     } their conclusions with the perm / const decomposition
#   id       <ds>_vnoise64         64-d noise; DRAGON/MENTOR --learn_v_feat=True
#   real_tg  <ds>_vcopy            real images, item-item graph from text only (--mm_image_weight=0.0)
#   perm_tg / noise_tg / const_tg / id_tg   same with the text-only item-item graph
# Tasks of the same data folder go to the same job (they share the graph cache), started 2 min apart.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS="${*:-baby}"
MODELS="${MODELS:-DRAGON FREEDOM MENTOR}"
CONDS="${CONDS:-real perm noise const real_tg}"
SEEDS="${SEEDS:-999 1 2}"
MAX_JOBS="${MAX_JOBS:-4}"
PARALLEL="${PARALLEL:-2}"
HOURS_PER_TASK="${HOURS_PER_TASK:-2}"
CONDA_ENV="${CONDA_ENV:-Astar}"
DRY_RUN="${DRY_RUN:-0}"
MAX_WALLTIME_H="${MAX_WALLTIME_H:-48}"   # queue limit
MEM="${MEM:-16gb}"                       # host RAM per job (per-user limits count it; lower = easier to schedule)
JOB_CPUS="${JOB_CPUS:-4}"
TAPPED=" DRAGON DRAGONID FREEDOM MENTOR LightGCN "   # models with a tap adapter (src/taps.py); others: Recall only
# reduced tuning grid for models whose official grid is too large (COHESION: 64 combinations)
COHESION_GRID="${COHESION_GRID:---learning_rate=[0.001,0.0001] --reg_weight=[0.01,0.001] --num_layer=[2,3]}"

folder() {   # folder <ds> <cond>
    case "$2" in
        real) echo "$1" ;;              perm) echo "$1_vperm" ;;        noise) echo "$1_vnoise" ;;
        const) echo "$1_vconst" ;;      id) echo "$1_vnoise64" ;;       real_tg) echo "$1_vcopy" ;;
        perm_tg) echo "$1_vperm_tg" ;;  noise_tg) echo "$1_vnoise_tg" ;; const_tg) echo "$1_vconst_tg" ;;
        id_tg) echo "$1_vnoise64_tg" ;; ones) echo "$1_vones" ;;          gauss) echo "$1_vgauss" ;;
        *) echo "unknown condition $2" >&2; exit 1 ;;
    esac
}
make_folder() {   # make_folder <ds> <cond>
    local ds=$1 c=$2 f; f=$(folder "$ds" "$c")
    [ "$c" = real ] && return 0
    [ -d "data/$f" ] && [ -f "src/configs/dataset/$f.yaml" ] && return 0
    case "$c" in
        perm|noise|const|ones|gauss) python tools/make_control_data.py -d "$ds" --modality v --mode "$c" ;;
        id)       python tools/make_control_data.py -d "$ds" --modality v --mode noise --dim 64 ;;
        real_tg)  python tools/make_control_data.py -d "$ds" --modality v --mode copy ;;
        perm_tg|noise_tg|const_tg) python tools/make_control_data.py -d "$ds" --modality v --mode "${c%_tg}" --suffix _tg ;;
        id_tg)    local xav; xav=$(python -c "import numpy as np; n=np.load('data/$ds/image_feat.npy', mmap_mode='r').shape[0]; print('%.4f' % (2.0 / (n + 64)) ** 0.5)")
                  python tools/make_control_data.py -d "$ds" --modality v --mode noise --dim 64 --scale "$xav" --suffix _tg ;;
    esac > /dev/null
}
run_name() {   # run_name <model> <cond> <seed>  (DRAGON on baby keeps the names of the first seed-999 runs)
    local m=$1 c=$2 s=$3 low; low=$(echo "$m" | tr 'A-Z' 'a-z')
    if [ "$m" = DRAGON ] && [ "$s" = 999 ]; then
        case "$c" in id|id_tg) echo dragon_vid ;; real_tg) echo dragon_txtgraph ;; *) echo dragon_taps ;; esac; return
    fi
    case "$c" in
        real_tg) echo "${low}_txtgraph_s$s" ;;
        id|id_tg) [ "$m" = FREEDOM ] && echo "freedom_s$s" || echo "${low}_vid_s$s" ;;
        *) echo "${low}_s$s" ;;
    esac
}

# ---- tasks already handled by your queued / running jobs
BUSY=$(mktemp)
# Jobs without TASK_FILE (other projects) must not stop the script: under `set -e -o pipefail` a failing
# grep / [ -f ] at the end of this pipeline used to exit silently. qselect gives full job ids
# (qstat -u truncates them to e.g. 260219.hpc-hea*, which qstat -f does not accept).
if command -v qstat >/dev/null 2>&1; then
    JOBS=$(qselect -u "$USER" 2>/dev/null || qstat -w -u "$USER" 2>/dev/null | awk '$1 ~ /^[0-9]+\./ {print $1}' || true)
    for j in $JOBS; do
        qstat -f "$j" 2>/dev/null | tr -d '\n\t ' | grep -o 'TASK_FILE=[^,]*' | cut -d= -f2 || true
    done | sort -u | while read -r tf; do
        if [ -f "$tf" ]; then cat "$tf"; fi
    done | awk 'NF >= 3 {print $2, $3, $1}' | sort -u > "$BUSY" || true
fi
echo "tasks in your active jobs: $(wc -l < "$BUSY")"

done_task() {   # done_task <model> <ds> <name>
    grep -l 'BEST ████' src/log/"$1-$2-$3"-*.log >/dev/null 2>&1 || return 1
    case "$3" in *_grid) return 0 ;; esac
    [[ "$TAPPED" == *" $1 "* ]] || return 0
    [ -f "log/probe_knn_$2_$1_$3.json" ] || return 1
    if [ "$1" = DRAGON ] || [ "$1" = MENTOR ]; then [ -f "log/graph_control_$2_$1-$2-$3.json" ] || return 1; fi
    return 0
}

TODO=$(mktemp)
add() {   # add <group> <name> <model> <ds> <seed> <args...>
    local g=$1 n=$2 m=$3 d=$4 s=$5; shift 5
    done_task "$m" "$d" "$n" && return 0
    grep -qx "$m $d $n" "$BUSY" && return 0
    echo "$g $n $m $d $s $*" >> "$TODO"
}

for DS in $DATASETS; do
    [ -f "data/${DS}/${DS}.inter" ] || { echo "data/${DS} not set up (bash scripts/setup_data.sh ${DS})"; exit 1; }
    for c in $CONDS; do make_folder "$DS" "$c"; done
    for M in $MODELS; do
        case "$M" in
            DRAGON)  [[ "$DS" == cloth* ]] && ARGS="--learning_rate=0.0001 --reg_weight=0.1" || ARGS="--learning_rate=0.0001 --reg_weight=0.001" ;;
            *)  # FREEDOM, MENTOR, COHESION, LGMRec, PGL, ...: tune on the real data first (selection on valid)
                [ -f "src/configs/model/$M.yaml" ] || { echo "no config src/configs/model/$M.yaml"; exit 1; }
                TUNE=$(echo "$M" | tr 'A-Z' 'a-z')_grid
                TUNE_ARGS=""; [ "$M" = COHESION ] && TUNE_ARGS="$COHESION_GRID"
                if ! ARGS=$(python tools/best_params.py "$M" "${PARAMS_FROM:-$DS}" "$TUNE"); then
                    echo "  $M on $DS: tuning first ($TUNE ${TUNE_ARGS:-official grid}); re-run this script when it has finished"
                    add "$DS" "$TUNE" "$M" "$DS" 999 $TUNE_ARGS
                    continue
                fi ;;
        esac
        echo "  $M on $DS: $ARGS"
        for s in $SEEDS; do
            for c in $CONDS; do
                d=$(folder "$DS" "$c"); n=$(run_name "$M" "$c" "$s"); extra=""
                case "$c" in *_tg) extra="--mm_image_weight=0.0" ;; esac
                case "$c" in id|id_tg) [ "$M" != FREEDOM ] && extra="$extra --learn_v_feat=True" ;; esac
                add "$d" "$n" "$M" "$d" "$s" $ARGS $extra
            done
        done
    done
done
rm -f "$BUSY"
N=$(grep -c . "$TODO" || true)
[ "$N" -gt 0 ] || { echo "Nothing to submit (all done or already queued)."; rm -f "$TODO"; exit 0; }

# ---- split by data folder (largest group first, to the least loaded job)
STAMP=$(date '+%Y%m%d-%H%M%S'); QDIR="src/log/queue/diag-${STAMP}"; mkdir -p "$QDIR"
W=$(( N < MAX_JOBS ? N : MAX_JOBS ))
declare -a LOAD; for ((w = 0; w < W; w++)); do LOAD[$w]=0; : > "$QDIR/worker$w.txt"; done
for G in $(awk '{print $1}' "$TODO" | sort | uniq -c | sort -rn | awk '{print $2}'); do
    best=0; for ((w = 1; w < W; w++)); do [ "${LOAD[$w]}" -lt "${LOAD[$best]}" ] && best=$w; done
    # tuning runs first (they unblock the second stage), then by seed
    awk -v g="$G" '$1 == g {$1 = ""; sub(/^ /, ""); print}' "$TODO" | sort -k4,4n \
        | awk '$1 ~ /_grid$/ {print; next} {rest[++n] = $0} END {for (i = 1; i <= n; i++) print rest[i]}' >> "$QDIR/worker$best.txt"
    LOAD[$best]=$(( LOAD[$best] + $(awk -v g="$G" '$1 == g' "$TODO" | wc -l) ))
done
rm -f "$TODO"

echo "Datasets: $DATASETS | models: $MODELS | conditions: $CONDS | seeds: $SEEDS"
echo "$N tasks -> $W job(s) x $PARALLEL in parallel, ${JOB_CPUS} cpus / ${MEM} per job | $QDIR"
for ((w = 0; w < W; w++)); do
    K=$(grep -c . "$QDIR/worker$w.txt" || true); [ "$K" -gt 0 ] || continue
    H=$(( (K + PARALLEL - 1) / PARALLEL * HOURS_PER_TASK + 2 ))
    if [ "$H" -gt "$MAX_WALLTIME_H" ]; then
        echo "  job $w: estimated ${H}h > ${MAX_WALLTIME_H}h limit, capped; unfinished tasks are picked up by re-running this script"
        H=$MAX_WALLTIME_H
    fi
    WT=$(printf '%02d:00:00' "$H")
    CMD=(qsub -q "${QUEUE:-small_gpuq}" -N "diag_w${w}" -l "select=1:ncpus=${JOB_CPUS}:ngpus=1:mem=${MEM}" -l "walltime=${WT}" -v "TASK_FILE=$QDIR/worker$w.txt,CONDA_ENV=$CONDA_ENV,PARALLEL=$PARALLEL" scripts/diag_worker.pbs)
    if [ "$DRY_RUN" = "1" ]; then echo "[dry-run] ${CMD[*]}"; JID=DRYRUN
    elif ! JID=$("${CMD[@]}" 2>&1); then
        echo "${CMD[*]}" >> "$QDIR/pending.sh"
        echo "  job $w NOT submitted ($JID); later: bash $QDIR/pending.sh"
        continue
    fi
    printf '  job %d (%s, %d tasks, walltime %s): %s\n' "$w" "$JID" "$K" "$WT" "$(awk '{printf "%s:%s:s%s ", $2, $3, $4}' "$QDIR/worker$w.txt")"
done
[ -f "$QDIR/pending.sh" ] && echo "Some jobs are pending (queue limit). When slots are free: bash $QDIR/pending.sh"
echo "Summary when done: python tools/summarize_diag.py -d <dataset>"
