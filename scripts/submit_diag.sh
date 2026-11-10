#!/bin/bash
# Content-vs-identity diagnosis: MODELS x CONDS x SEEDS on one or more datasets, in at most MAX_JOBS jobs.
#
#   bash scripts/submit_diag.sh sports clothing baby          # default models / conditions / seeds
#   MODELS="MENTOR" bash scripts/submit_diag.sh baby
#   MODELS="COHESION LGMRec" bash scripts/submit_diag.sh baby   # Recall-only models (no tap adapter)
#   CONDS="real perm noise const real_tg const_tg" bash scripts/submit_diag.sh sports
#   DRY_RUN=1 bash scripts/submit_diag.sh sports              # show the split, submit nothing
#   MEM=32gb PARALLEL=1 bash scripts/submit_diag.sh clothing    # resources per job (default 16gb, 4 cpus)
#   QUEUE=medium_gpuq bash scripts/submit_diag.sh sports         # another GPU queue (default small_gpuq)
#   EXTRA_ARGS="--infonce_chunk=4096" MODELS=MENTOR bash scripts/submit_diag.sh cloth   # MENTOR on Clothing (GPU memory)
#   COHESION_ARGS="$(python tools/best_params.py COHESION sports cohesion_grid)" MODELS=COHESION bash scripts/submit_diag.sh sports_tconst
#       fixed hyper-parameters (no tuning) for a derived dataset, e.g. the text-off variant; DRAGON uses DRAGON_ARGS
#   Per-condition tuning (each control tuned on its own, a robustness check of the shared-HP protocol):
#   FORCE_TUNE=1 RUN_TAG=_ct CONDS=real MODELS=DRAGON bash scripts/submit_diag.sh cloth cloth_vperm cloth_vconst
#     (each folder is treated as a dataset: <model>_grid tuned on it, then <model>_ct_s<seed> runs)
#   GRID_TAG=_ct FORCE_TUNE=1 RUN_TAG=_ct CONDS=real MODELS=COHESION bash scripts/submit_diag.sh dbbook
#     (Real re-tuned on the same grid as the controls, as <model>_grid_ct, then <model>_ct_s<seed> on the real folder)
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
#   perm2..perm5  <ds>_vperm_p<k>  independent permutations (seed 2026+k), to measure permutation variance
#   constn   <ds>_vconstn          mean direction rescaled to the average feature norm
#   ones     <ds>_vones            all-ones rows   } knockout protocol of Ye et al. (ECIR'26), to compare
#   gauss    <ds>_vgauss           N(0,1) rows     } their conclusions with the perm / const decomposition
#   id       <ds>_vnoise64         64-d noise; DRAGON/MENTOR --learn_v_feat=True
#   real_tg  <ds>_vcopy            real images, item-item graph from text only (--mm_image_weight=0.0)
#   perm_tg / noise_tg / const_tg / id_tg   same with the text-only item-item graph
# Tasks of the same data folder go to the same job (they share the graph cache), started 2 min apart.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v python >/dev/null 2>&1 || {
    echo "python not found: activate the environment first (source ~/miniconda3/etc/profile.d/conda.sh && conda activate Astar)"
    exit 1; }

DATASETS="${*:-baby}"
MODELS="${MODELS:-DRAGON FREEDOM MENTOR}"
CONDS="${CONDS:-real perm noise const real_tg}"
SEEDS="${SEEDS:-999 1 2}"
MAX_JOBS="${MAX_JOBS:-4}"
PARALLEL="${PARALLEL:-2}"
HOURS_PER_TASK="${HOURS_PER_TASK:-2}"
CONDA_ENV="${CONDA_ENV:-Astar}"
DRY_RUN="${DRY_RUN:-0}"
MAX_WALLTIME_H="${MAX_WALLTIME_H:-48}"   # queue limit, whole hours (e.g. 24, not 24h)
MAX_WALLTIME_H="${MAX_WALLTIME_H%h}"
MEM="${MEM:-16gb}"                       # host RAM per job (per-user limits count it; lower = easier to schedule)
JOB_CPUS="${JOB_CPUS:-4}"
QUEUE="${QUEUE:-small_gpuq}"
FORCE_TUNE="${FORCE_TUNE:-0}"
GRID_TAG="${GRID_TAG:-}"      # e.g. GRID_TAG=_ct: tune Real again on the grid used for the controls (<model>_grid_ct)
EXTRA_ARGS="${EXTRA_ARGS:-}"             # appended to every task (tuning and runs), e.g. --infonce_chunk=4096            # 1: tune every model (DRAGON too) on each given data folder
RUN_TAG="${RUN_TAG:-}"                   # e.g. _ct: runs named <model>_ct_s<seed> (own series, no clash)             # e.g. QUEUE=<medium gpu queue> when it has free GPUs (qstat -Q lists queues)
TAPPED=" DRAGON DRAGONID FREEDOM MENTOR LightGCN "   # models with a tap adapter (src/taps.py); others: Recall only
# reduced tuning grid for models whose official grid is too large (COHESION: 64 combinations)
DRAGON_GRID="${DRAGON_GRID:---learning_rate=[0.001,0.0001] --reg_weight=[0.1,0.01,0.001]}"   # DRAGON outside Amazon
COHESION_GRID="${COHESION_GRID:---learning_rate=[0.001,0.0001] --reg_weight=[0.01,0.001] --num_layer=[2,3]}"

folder() {   # folder <ds> <cond>
    case "$2" in
        real) echo "$1" ;;              perm) echo "$1_vperm" ;;        noise) echo "$1_vnoise" ;;
        const) echo "$1_vconst" ;;      id) echo "$1_vnoise64" ;;       real_tg) echo "$1_vcopy" ;;
        perm_tg) echo "$1_vperm_tg" ;;  noise_tg) echo "$1_vnoise_tg" ;; const_tg) echo "$1_vconst_tg" ;;
        id_tg) echo "$1_vnoise64_tg" ;; ones) echo "$1_vones" ;;          gauss) echo "$1_vgauss" ;;
        constn) echo "$1_vconstn" ;;    constn_tg) echo "$1_vconstn_tg" ;;
        perm[2-9]) echo "$1_vperm_p${2#perm}" ;;
        perm[2-9]_tg) local k=${2#perm}; echo "$1_vperm_p${k%_tg}_tg" ;;
        *) echo "unknown condition $2" >&2; exit 1 ;;
    esac
}
make_folder() {   # make_folder <ds> <cond>
    local ds=$1 c=$2 f; f=$(folder "$ds" "$c")
    [ "$c" = real ] && return 0
    [ -d "data/$f" ] && [ -f "src/configs/dataset/$f.yaml" ] && return 0
    case "$c" in
        perm|noise|const|constn|ones|gauss) python tools/make_control_data.py -d "$ds" --modality v --mode "$c" ;;
        perm[2-9]|perm[2-9]_tg) local k=${c#perm}; k=${k%_tg}; local suf="_p$k"; [[ "$c" == *_tg ]] && suf="${suf}_tg"
                  python tools/make_control_data.py -d "$ds" --modality v --mode perm --seed $((2026 + k)) --suffix "$suf" ;;
        constn_tg) python tools/make_control_data.py -d "$ds" --modality v --mode constn --suffix _tg ;;
        id)       python tools/make_control_data.py -d "$ds" --modality v --mode noise --dim 64 ;;
        real_tg)  python tools/make_control_data.py -d "$ds" --modality v --mode copy ;;
        perm_tg|noise_tg|const_tg) python tools/make_control_data.py -d "$ds" --modality v --mode "${c%_tg}" --suffix _tg ;;
        id_tg)    local xav; xav=$(python -c "import numpy as np; n=np.load('data/$ds/image_feat.npy', mmap_mode='r').shape[0]; print('%.4f' % (2.0 / (n + 64)) ** 0.5)")
                  python tools/make_control_data.py -d "$ds" --modality v --mode noise --dim 64 --scale "$xav" --suffix _tg ;;
    esac > /dev/null
}
run_name() {   # run_name <model> <cond> <seed>  (DRAGON on baby keeps the names of the first seed-999 runs)
    local m=$1 c=$2 s=$3 low; low=$(echo "$m" | tr 'A-Z' 'a-z')
    if [ -n "$RUN_TAG" ]; then echo "${low}${RUN_TAG}_s$s"; return; fi   # separate series, e.g. per-condition tuning
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
        TUNE_DRAGON=0
        [ "$M" = DRAGON ] && [ "$FORCE_TUNE" = 1 ] && TUNE_DRAGON=1
        if [ "$M" = DRAGON ] && [ -z "${DRAGON_ARGS:-}" ] && [ "$TUNE_DRAGON" = 0 ]; then
            case "$DS" in baby*|sports*|cloth*) ;; *) TUNE_DRAGON=1 ;; esac   # no official setting outside Amazon
        fi
        case "$M" in
            DRAGON) [ "$TUNE_DRAGON" = 1 ] || {
                # official setting (Clothing reg 0.1); DRAGON_ARGS="..." overrides it, e.g. to stay consistent with earlier runs
                if [ -n "${DRAGON_ARGS:-}" ]; then ARGS="$DRAGON_ARGS"
                elif [[ "$DS" == cloth* ]]; then ARGS="--learning_rate=0.0001 --reg_weight=0.1"
                else ARGS="--learning_rate=0.0001 --reg_weight=0.001"; fi; } ;;
        esac
        FA="${M}_ARGS"
        if [ "$M" != DRAGON ] && [ -n "${!FA:-}" ]; then
            ARGS="${!FA}"; echo "  $M on $DS: fixed $FA = $ARGS"   # e.g. COHESION_ARGS=$(python tools/best_params.py COHESION sports cohesion_grid)
        else
        case "$M:$TUNE_DRAGON" in
            DRAGON:0) ;;
            *)  # FREEDOM, MENTOR, COHESION, ... and DRAGON on non-Amazon data: tune on the real data first (valid)
                [ -f "src/configs/model/$M.yaml" ] || { echo "no config src/configs/model/$M.yaml"; exit 1; }
                TUNE=$(echo "$M" | tr 'A-Z' 'a-z')_grid${GRID_TAG}   # GRID_TAG=_ct: a fresh grid run next to an existing one
                GV="${M}_GRID"; TUNE_ARGS="${!GV:-}"   # COHESION_GRID / DRAGON_GRID (defaults above), MENTOR_GRID, FREEDOM_GRID, ...
                if ! ARGS=$(python tools/best_params.py "$M" "$DS" "$TUNE"); then
                    echo "  $M on $DS: tuning first ($TUNE ${TUNE_ARGS:-official grid}); re-run this script when it has finished"
                    add "$DS" "$TUNE" "$M" "$DS" 999 $TUNE_ARGS $EXTRA_ARGS
                    continue
                fi ;;
        esac
        fi
        echo "  $M on $DS: $ARGS"
        for s in $SEEDS; do
            for c in $CONDS; do
                d=$(folder "$DS" "$c"); n=$(run_name "$M" "$c" "$s"); extra=""
                case "$c" in *_tg) extra="--mm_image_weight=0.0" ;; esac
                case "$c" in id|id_tg) [ "$M" != FREEDOM ] && extra="$extra --learn_v_feat=True" ;; esac
                add "$d" "$n" "$M" "$d" "$s" $ARGS $extra $EXTRA_ARGS
            done
        done
    done
done
rm -f "$BUSY"
N=$(grep -c . "$TODO" || true)
[ "$N" -gt 0 ] || { echo "Nothing to submit (all done or already queued)."; rm -f "$TODO"; exit 0; }

# ---- split by data folder (largest group first, to the least loaded job)
STAMP=$(date '+%Y%m%d-%H%M%S'); mkdir -p src/log/queue
QDIR=$(mktemp -d "src/log/queue/diag-${STAMP}-XXXX")   # unique even for two submissions in the same second
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
echo "$N tasks -> $W job(s) x $PARALLEL in parallel, ${JOB_CPUS} cpus / ${MEM} per job, queue $QUEUE | $QDIR"
for ((w = 0; w < W; w++)); do
    K=$(grep -c . "$QDIR/worker$w.txt" || true); [ "$K" -gt 0 ] || continue
    H=$(( (K + PARALLEL - 1) / PARALLEL * HOURS_PER_TASK + 2 ))
    if [ "$H" -gt "$MAX_WALLTIME_H" ]; then
        echo "  job $w: estimated ${H}h > ${MAX_WALLTIME_H}h limit, capped; unfinished tasks are picked up by re-running this script"
        H=$MAX_WALLTIME_H
    fi
    WT=$(printf '%02d:00:00' "$H")
    CMD=(qsub -q "$QUEUE" -N "diag_w${w}" -l "select=1:ncpus=${JOB_CPUS}:ngpus=1:mem=${MEM}" -l "walltime=${WT}" -v "TASK_FILE=$QDIR/worker$w.txt,CONDA_ENV=$CONDA_ENV,PARALLEL=$PARALLEL" scripts/diag_worker.pbs)
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
