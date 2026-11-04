#!/bin/bash
# Link the raw files of an existing MMRec data folder into mmj/data/<dataset>/.
#
#   bash scripts/setup_data.sh                       # baby, from ~/ihpc_data/hkieu/VIRAL/data
#   bash scripts/setup_data.sh sports
#   SRC=/path/to/data bash scripts/setup_data.sh clothing
#
# Only the raw files are linked (interactions, features, user graph). Do NOT link the whole
# folder: models cache their graphs there under names that do not encode every setting,
# e.g. DRAGON and VIRAL/VLIF both use mm_adj_<k>.pt (VLIF with image weight 0.5, DRAGON 0.1),
# so a shared folder silently gives DRAGON the wrong item-item graph.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-baby}"
SRC="${SRC:-$HOME/ihpc_data/hkieu/VIRAL/data}"
[ -d "$SRC/$DATASET" ] || { echo "Not found: $SRC/$DATASET"; exit 1; }
mkdir -p "data/$DATASET"

for f in "$DATASET.inter" image_feat.npy text_feat.npy user_graph_dict.npy; do
    if [ -f "$SRC/$DATASET/$f" ]; then
        ln -sfn "$(readlink -f "$SRC/$DATASET/$f")" "data/$DATASET/$f"
        echo "  linked data/$DATASET/$f"
    else
        echo "  missing $SRC/$DATASET/$f"
    fi
done

if [ ! -e "data/$DATASET/user_graph_dict.npy" ]; then
    echo "DRAGON needs data/$DATASET/user_graph_dict.npy:"
    echo "  (cd preprocessing && python dualgnn-gen-u-u-matrix.py)   # edit dataset name inside first"
fi
ls -la "data/$DATASET"
