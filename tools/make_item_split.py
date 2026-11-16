# coding: utf-8
"""
JEPA-Y diagnosis, step 2: item split 80/10/10 (train / val / eval), stratified by train degree.

    python tools/make_item_split.py -d baby                # -> data/baby/jepay_item_split.npz

Only items with at least one TRAIN interaction are split: an item without one has no collaborative
signal in the teacher (its LightGCN row stays at its random initialisation), so it is no target.
Train subsets for the data-fraction sweep are nested (25% within 50% within 100%) and stratified the
same way, and are fixed (seed 2026), so seed variance in the sweep comes from training only.

npz keys: train, val, eval, train_f25, train_f50 (sorted item indices), degree (train degree, all items),
n_items, seed.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from probe_rec import load_data   # noqa: E402


def stratified_split(items, degree, rng, n_bins=10, fracs=(0.8, 0.1, 0.1)):
    """Items sorted into degree-quantile bins; each bin shuffled and cut by fracs. Returns
    (train, val, eval, per-bin shuffled train lists) so nested train subsets can be cut the same way."""
    order = items[np.argsort(degree[items], kind='stable')]
    bins = np.array_split(order, n_bins)
    tr, va, ev, tr_bins = [], [], [], []
    for b in bins:
        b = rng.permutation(b)
        n_tr = int(round(fracs[0] * len(b)))
        n_va = int(round(fracs[1] * len(b)))
        tr_bins.append(b[:n_tr])
        tr.append(b[:n_tr]); va.append(b[n_tr:n_tr + n_va]); ev.append(b[n_tr + n_va:])
    cat = lambda xs: np.sort(np.concatenate(xs)).astype(np.int64)   # noqa: E731
    return cat(tr), cat(va), cat(ev), tr_bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True)
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--n_bins', type=int, default=10)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    dst = os.path.join(args.data_dir, args.dataset)
    out = os.path.join(dst, 'jepay_item_split.npz')
    if os.path.isfile(out) and not args.force:
        print('[split] exists: {} (use --force to redo)'.format(out))
        return
    n_feat = np.load(os.path.join(dst, 'image_feat.npy'), mmap_mode='r').shape[0]
    (R_tr, _, _), _, n_items = load_data(args.data_dir, args.dataset, min_items=n_feat)
    if n_items != n_feat:
        raise SystemExit('.inter has item ids up to {} but features have {} rows'.format(n_items - 1, n_feat))
    degree = np.asarray(R_tr.sum(0)).ravel().astype(np.int64)
    items = np.where(degree > 0)[0]
    rng = np.random.default_rng(args.seed)
    tr, va, ev, tr_bins = stratified_split(items, degree, rng, args.n_bins)
    subsets = {}
    for f in (25, 50):
        subsets['train_f{}'.format(f)] = np.sort(np.concatenate(
            [b[:int(round(len(b) * f / 100))] for b in tr_bins])).astype(np.int64)

    # checks
    assert len(np.intersect1d(tr, va)) == 0 and len(np.intersect1d(tr, ev)) == 0 and len(np.intersect1d(va, ev)) == 0
    assert len(tr) + len(va) + len(ev) == len(items)
    assert np.isin(subsets['train_f25'], subsets['train_f50']).all() and np.isin(subsets['train_f50'], tr).all()
    np.savez(out, train=tr, val=va, eval=ev, degree=degree, n_items=n_items, seed=args.seed, **subsets)
    stats = {'n_items': int(n_items), 'n_without_train_interaction': int(n_items - len(items)),
             'train': len(tr), 'val': len(va), 'eval': len(ev),
             'train_f25': len(subsets['train_f25']), 'train_f50': len(subsets['train_f50']),
             'median_degree': {k: float(np.median(degree[v])) for k, v in (('train', tr), ('val', va), ('eval', ev))}}
    print('[split]', json.dumps(stats))
    with open(os.path.join(dst, 'jepay_item_split.json'), 'w') as f:
        json.dump(stats, f, indent=1)


if __name__ == '__main__':
    main()
