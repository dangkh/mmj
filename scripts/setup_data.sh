#!/bin/bash
# Link the raw files of an existing MMRec data folder into mmj/data/<name>/.
#
#   bash scripts/setup_data.sh                       # baby, from ~/ihpc_data/hkieu/VIRAL/data
#   bash scripts/setup_data.sh sports
#   SRC=/path/to/data bash scripts/setup_data.sh clothing
#   TEXT=text_feat-v1.npy bash scripts/setup_data.sh baby      # source file has another name
#   NAME=baby_vlm IMAGE=en_image_feat.npy TEXT=text_feat-v1.npy bash scripts/setup_data.sh baby
#       -> data/baby_vlm with the VLM image features (VIRAL) and its own dataset config
#
# The links are always named image_feat.npy / text_feat.npy (the MMRec dataset configs).
# Only the raw files are linked (interactions, features, user graph). Do NOT link the whole
# folder: models cache their graphs there under names that do not encode every setting,
# e.g. DRAGON and VIRAL/VLIF both use mm_adj_<k>.pt (VLIF with image weight 0.5, DRAGON 0.1),
# so a shared folder silently gives DRAGON the wrong item-item graph.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-baby}"
NAME="${NAME:-$DATASET}"
SRC="${SRC:-$HOME/ihpc_data/hkieu/VIRAL/data}"
IMAGE="${IMAGE:-image_feat.npy}"
TEXT="${TEXT:-text_feat.npy}"
[ -d "$SRC/$DATASET" ] || { echo "Not found: $SRC/$DATASET"; exit 1; }
mkdir -p "data/$NAME"

link() {   # link <source name> <link name>
    if [ -f "$SRC/$DATASET/$1" ]; then
        ln -sfn "$(readlink -f "$SRC/$DATASET/$1")" "data/$NAME/$2"
        echo "  data/$NAME/$2 -> $1"
    else
        echo "  MISSING $SRC/$DATASET/$1"
    fi
}
link "$DATASET.inter" "$NAME.inter"
link "$IMAGE" image_feat.npy
link "$TEXT" text_feat.npy
link user_graph_dict.npy user_graph_dict.npy

if [ "$NAME" != "$DATASET" ]; then   # dataset config for the new name (same as the original, own inter file)
    sed "s/'$DATASET.inter'/'$NAME.inter'/" "src/configs/dataset/$DATASET.yaml" \
        | sed "1i # $NAME: copy of $DATASET with image=$IMAGE text=$TEXT (scripts/setup_data.sh)" \
        > "src/configs/dataset/$NAME.yaml"
    echo "  wrote src/configs/dataset/$NAME.yaml"
fi

echo
echo "Feature files available in $SRC/$DATASET:"
python - "$SRC/$DATASET" <<'EOF' || ls "$SRC/$DATASET"
import glob, os, sys
import numpy as np
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*feat*.npy'))):
    a = np.load(f, mmap_mode='r')
    print('  {:<28s} {}  {}'.format(os.path.basename(f), a.shape, a.dtype))
EOF
if [ ! -e "data/$NAME/user_graph_dict.npy" ]; then
    echo "DRAGON needs data/$NAME/user_graph_dict.npy:"
    echo "  (cd preprocessing && python dualgnn-gen-u-u-matrix.py)   # edit dataset name inside first"
fi
ls -la "data/$NAME"
