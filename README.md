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
| `tools/probe_rec.py` | recommendation probe on frozen representations (CPU) |
| `scripts/` | PBS jobs (max 4 at a time), data setup, result collection |

Model code in `src/models/` is unchanged.

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
```

Warm-mode numbers are only comparable at the same dimension (`--warm_dim`, default 64): with d
dimensions the probe can learn a rank-d item-item model even from noise, so read them relative to
the `null64` row. Cold mode (unseen items) cannot memorise and is the cleaner content measure.
