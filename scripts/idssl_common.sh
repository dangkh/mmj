# Shared helpers of scripts/e0_smoke.pbs, e3_baselines.pbs, e4_jepa.pbs (sourced; needs DS).
# BB: LightGCN backbone of the dataset (n_layers, reg_weight selected by the LightGCNC id grid), used by every method.
BB=$(python tools/best_params.py LightGCNC "$DS" id_grid 2>/dev/null) || BB="--n_layers=3 --reg_weight=0.0001"
NL=$(echo "$BB" | grep -o -- '--n_layers=[0-9]*')
echo "backbone for $DS: $BB"

done_run() {   # done_run MODEL EXP: a finished log (BEST block) exists
    python tools/best_params.py "$1" "$DS" "$2" > /dev/null 2>&1
}

run() {        # run MODEL EXP ARGS...   (skipped if finished; taps of every combination are saved)
    local M=$1 E=$2; shift 2
    if done_run "$M" "$E"; then echo "[skip] $M $E finished"; return 0; fi
    echo "===== [$(date '+%F %T')] $M $E: $* ====="
    (cd src && python -u main.py -m "$M" -d "$DS" --exp_name="$E" --save_taps=True "$@")
}

best() {       # best MODEL EXP... : --key=value of the best seed-999 run (valid R@20) among these experiments
    python tools/taps_best.py "$1" "$DS" "${@:2}"
}

seeds() {      # seeds MODEL SEED_EXP GRID_EXP... : the selected setting again with seeds 1 and 2
    local M=$1 S=$2; shift 2
    local P
    P=$(best "$M" "$@") || { echo "no taps for $M $*"; return 1; }
    # shellcheck disable=SC2086
    run "$M" "$S" $BB $P --seed="[1,2]"
}
