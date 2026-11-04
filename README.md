# mmj

Multimodal recommendation diagnostics on top of [MMRec](https://github.com/enoche/MMRec) (upstream copy, GPL-3.0).
Goal: measure how much of the image/text signal is **retained, transformed and used** inside
existing models (DRAGON, FREEDOM, MGCN, LATTICE, ...), then fix the bottleneck (JEPA-Y).

## Changes to upstream MMRec

| File | Change |
|---|---|
| `src/utils/quick_start.py` | best hyper-parameter combination chosen on **valid** Recall@20 (upstream: test); scalar CLI values accepted for grid keys |
| `src/main.py` | `--key=value` overrides for any config key, `--exp_name`, `--mg` |
| `src/utils/logger.py` | log name `src/log/<MODEL>-<dataset>-<exp_name>-<time>.log` |
| `src/common/trainer.py` | `--save_taps=True` saves representations at every best-valid epoch |
| `src/taps.py` | tap adapters for DRAGON, FREEDOM, MGCN, LATTICE, LightGCN |
| `tools/probe_knn.py` | main probe: parameter-free item-kNN on a frozen representation (CPU) |
| `tools/probe_rec.py` | optional learned probe (two-sided ridge), appendix only |
| `scripts/` | PBS jobs (max 4 at a time), data setup, result collection |

Model code in `src/models/` is unchanged except for library compatibility:
`dok_matrix._update` (removed in recent SciPy) replaced by an identical COO construction in
`common/encoders.py` and bm3/damrs/freedom/layergcn/lgmrec/lightgcn/pgl, and `np.float` (removed
in NumPy 1.24) replaced by `np.float64` in `utils/metrics.py`.

## Taps

| Tap | Meaning | DRAGON | MGCN | FREEDOM | LATTICE |
|---|---|---|---|---|---|
| T0 | raw features (`data/<ds>/image_feat.npy`, `text_feat.npy`) | x | x | x | x |
| T0ft | input features after training (MMRec fine-tunes them) | - | x | x | x |
| T1 | after the projection, before any graph | MLP output | Linear output (before ⊙ItemID) | img/txt_trs | img/txt_trs |
| T1g | ItemID ⊙ gate(T1) | - | x | - | - |
| T2 | after the graph, per modality | v_rep/t_rep | modal item after kNN | - | - |
| T3 | final item representation used for scoring | x | x | x | x |

Saved to `taps/<MODEL>-<dataset>-<exp_name>/` (`combo<i>/` when one run has several combinations) with `meta.json` (epoch, valid/test metrics).

## Server setup

```bash
cd ~/ihpc_data/hkieu
git clone <this repo> mmj && cd mmj
bash scripts/setup_data.sh baby          # links raw files from ~/ihpc_data/hkieu/VIRAL/data/baby
conda activate Astar
python -c "import torch_geometric"       # DRAGON needs it
```

Do not symlink the whole data folder: models cache item graphs there (`mm_adj_10.pt`, ...) under
names that do not encode every setting, so sharing the cache with VIRAL gives DRAGON the wrong graph.

## Stage 1: reproduce (Baby)

```bash
DRY_RUN=1 bash scripts/submit_all.sh baby scripts/experiments_baby_repro.txt   # check the split
bash scripts/submit_all.sh baby scripts/experiments_baby_repro.txt
python scripts/collect_results.py -d baby --combos                            # vs paper numbers
```

## Probe

```bash
qsub -v DATASET=baby scripts/probe.pbs                                       # T0 (raw), no training needed
qsub -v DATASET=baby,TAPS=taps/DRAGON-baby-dragon_paper scripts/probe.pbs     # taps of a trained model
python tools/probe_knn.py -d baby --taps taps/MGCN-baby-mgcn_grid/combo1      # interactively (~1 min)
```

score(u,i) = sum over the user's train items j of cosine-kNN(j, i): nothing is fitted (only the
neighbourhood size, chosen on valid), so there is no capacity/dimension difference between taps,
no loss over unobserved zeros and no memorisation (null64 is at chance). Same test protocol as MMRec.
The ridge probe (`tools/probe_rec.py`) learns a readout W and can learn CF even from noise in its
warm mode; keep it for the appendix only.
