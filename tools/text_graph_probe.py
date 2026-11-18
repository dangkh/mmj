# coding: utf-8
"""
Stage 2a: does graph propagation make the Text representation predictive of the ID collaborative preference?

Fixed target (no learned target encoder, so every condition is compared in the same space):
    Y_i = item embedding of the frozen ID teacher (LightGCNC item_input=id, taps T3), standardised per
    dimension with train-item statistics.
Inputs (all frozen), predictor P_theta = probe of controlled capacity (tools/jepay/probe.py: lin = ridge,
mlp1, mlp2; regularisation chosen on val items):
    J0   raw text feature (text_feat.npy, before any training)
    J1   T^(0) = E_T(text): layer-0 item embedding of the Text teacher (LightGCNC item_input=text)
    J2   T^G = mean_l T^(l): Text teacher after propagation (its final item embedding)
    J3   [T^(0) ; D],  D = T^G - T^(0)  (same information as (T^(0), T^G); differs only for non-linear heads)
    JD   D alone: the graph-induced change
    Jg   T^G of a teacher whose layer 0 is one shared vector for all items (item_input=const): graph and
         user embeddings only, no item content. Control for J2.
    L<l> T^(l) of the Text teacher at each layer l >= 1 (L0 = J1)
Source of L2 (same graph, only the layer-0 item vectors change; A_hat = LightGCN-normalised train graph, the item
rows of A_hat^2 x depend on the item block of x only, so L2 = A_hat^2 T^(0) exactly):
    L2perm<s>  A_hat^2 T^(0)[pi_s]: text vectors shuffled across items (s = 0, 1, 2); L2perm = mean over s
    L2const    A_hat^2 (1 c^T), c = mean_i T^(0)_i: rank 1, graph structure only (a degree-like scalar per item)
Protocol: the item split of E-A (data/<ds>/jepay_item_split.npz, 80/10/10, stratified by degree): probes are
fitted on train items, selected on val items, reported on eval items.
Metrics (eval items): MSE and R^2 to Y; ranking of eval items by U_id . Y_hat (Recall/NDCG@20; ceiling:
U_id . Y); paired item bootstrap (10 000) for J2-J1, J3-J2, J2-Jg, JD-J1, J1-J0, J3-J1.

    python tools/text_graph_probe.py -d dbbook                       # all available inputs, heads lin mlp1 mlp2
    python tools/text_graph_probe.py -d dbbook --summary             # table only
Output: results/text_graph/<ds>/{<set>_<head>_s<seed>.json,.npz}, summary.md
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from probe_rec import load_data                                                   # noqa: E402
from jepay.probe import fit_probe                                                  # noqa: E402
from jepay.metrics import per_item_mse, r2, recall_eval_items, paired_bootstrap   # noqa: E402

HEADS = ('lin', 'mlp1', 'mlp2')
PAIRS = [('J2', 'J1', 'graph on text'), ('J3', 'J2', 'raw + graph vs graph'), ('J2', 'Jg', 'text adds to graph'),
         ('JD', 'J1', 'graph change vs text'), ('J1', 'J0', 'learned E_T vs raw text'), ('J3', 'J1', 'raw + graph vs raw'),
         ('L2', 'L2perm', 'correct text-item match'), ('L2', 'L2const', 'text vs graph structure'),
         ('L2perm', 'L2const', 'any text vs structure'), ('L2', 'J1', 'neighbour text vs own text')]
N_PERM = 3


def item_adj(R_tr):
    """Item-side two-hop operator of the LightGCN graph: N = D_u^-1/2 R D_i^-1/2, A_hat^2 (items) = N^T N."""
    R = R_tr.tocsr().astype(np.float64)
    du = np.asarray(R.sum(1)).ravel() + 1e-7
    di = np.asarray(R.sum(0)).ravel() + 1e-7
    N = sp.diags(du ** -0.5) @ R @ sp.diags(di ** -0.5)
    return lambda X: np.asarray(N.T @ (N @ X.astype(np.float64)), dtype=np.float32)


def layers_of(path):
    files = sorted(glob.glob(os.path.join(path, 'T_item_l*.npy')), key=lambda f: int(f.split('_l')[-1][:-4]))
    if not files:
        raise SystemExit('{}: no T_item_l*.npy (re-run the teacher with the per-layer taps)'.format(path))
    return [np.load(f).astype(np.float32) for f in files]


def build_inputs(args, R_tr):
    sets, info = {}, {}
    T = np.load(os.path.join(args.data_dir, args.dataset, 'text_feat.npy')).astype(np.float32)
    sets['J0'] = T
    L = layers_of(args.text_taps)
    TG = np.mean(L, 0)
    t3 = os.path.join(args.text_taps, 'T3.npy')
    if os.path.isfile(t3):          # T^G must be the teacher's own final item embedding (no item-item graph)
        gap = float(np.abs(np.load(t3) - TG).max())
        info['TG_vs_T3_maxabs'] = gap
        print('[probe] max |mean_l T^(l) - T3| = {:.2e}'.format(gap), flush=True)
        if gap > 1e-3:
            print('[probe] WARNING: T^G differs from T3 (item-item graph on?)', flush=True)
    D = TG - L[0]
    sets.update({'J1': L[0], 'J2': TG, 'J3': np.concatenate([L[0], D], 1), 'JD': D})
    for l in range(1, len(L)):
        sets['L{}'.format(l)] = L[l]
    info['text_layers'] = len(L) - 1
    if len(L) > 2:
        hop2 = item_adj(R_tr)
        L2re = hop2(L[0])
        gap = float(np.abs(L2re - L[2]).max() / max(np.abs(L[2]).max(), 1e-12))
        info['L2_recomputed_relgap'] = gap
        print('[probe] A_hat^2 T^(0) vs saved T_item_l2: max rel. gap {:.2e}'.format(gap), flush=True)
        if gap > 1e-3:
            print('[probe] WARNING: recomputed L2 differs from the teacher (graph mismatch?)', flush=True)
        for s in range(N_PERM):
            sets['L2perm{}'.format(s)] = hop2(L[0][np.random.default_rng(100 + s).permutation(L[0].shape[0])])
        c = L[0].mean(0, keepdims=True)
        sets['L2const'] = hop2(np.repeat(c, L[0].shape[0], 0))
    if args.const_taps and os.path.isdir(args.const_taps):
        Lc = layers_of(args.const_taps)
        sets['Jg'] = np.mean(Lc, 0)
        info['const_layers'] = len(Lc) - 1
    else:
        print('[probe] no const teacher at {}: Jg skipped'.format(args.const_taps), flush=True)
    return sets, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True)
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--id_taps', default=None, help='default taps/LightGCNC-<ds>-teacher_id')
    ap.add_argument('--text_taps', default=None, help='default taps/LightGCNC-<ds>-teacher_text_layers')
    ap.add_argument('--const_taps', default=None, help='default taps/LightGCNC-<ds>-teacher_const_layers')
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'results', 'text_graph'))
    ap.add_argument('--sets', default='', help='subset, e.g. "J1 J2 J3"; default: all available')
    ap.add_argument('--heads', default=' '.join(HEADS))
    ap.add_argument('--seed', type=int, default=999)
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    taps = lambda name: os.path.join(ROOT, 'taps', 'LightGCNC-{}-{}'.format(args.dataset, name))  # noqa: E731
    args.id_taps = args.id_taps or taps('teacher_id')
    args.text_taps = args.text_taps or taps('teacher_text_layers')
    args.const_taps = args.const_taps or taps('teacher_const_layers')
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    out = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out, exist_ok=True)

    Y_raw = np.load(os.path.join(args.id_taps, 'T3.npy')).astype(np.float32)
    U_id = np.load(os.path.join(args.id_taps, 'T3_user.npy')).astype(np.float32)
    split = np.load(os.path.join(args.data_dir, args.dataset, 'jepay_item_split.npz'))
    tr, va, ev = split['train'], split['val'], split['eval']
    (R_tr, R_va, R_te), n_users, n_items = load_data(args.data_dir, args.dataset, min_items=Y_raw.shape[0])
    assert n_items == Y_raw.shape[0] and n_users == U_id.shape[0], 'teacher / data shapes differ'
    R_gt = (R_va + R_te).tocsr()
    R_gt.data[:] = 1.0
    mu, sd = Y_raw[tr].mean(0), Y_raw[tr].std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Y = (Y_raw - mu) / sd

    ceil_path = os.path.join(out, 'ceiling.json')
    if not os.path.isfile(ceil_path):
        json.dump(recall_eval_items(Y_raw[ev], U_id, R_tr, R_gt, ev), open(ceil_path, 'w'), indent=1)
    ceiling = json.load(open(ceil_path))

    if not args.summary:
        sets, info = build_inputs(args, R_tr)
        json.dump(info, open(os.path.join(out, 'inputs.json'), 'w'), indent=1)
        wanted = args.sets.split() or list(sets)
        for name in wanted:
            if name not in sets:
                print('[probe] {} not available'.format(name), flush=True)
                continue
            X = sets[name]
            assert X.shape[0] == n_items, '{}: {} rows, {} items'.format(name, X.shape[0], n_items)
            for head in args.heads.split():
                base = os.path.join(out, '{}_{}_s{}'.format(name, head, args.seed))
                if os.path.isfile(base + '.json'):
                    continue
                t0 = time.time()
                yhat, sel = fit_probe(X, Y, tr, va, head, args.seed, device)
                mse = per_item_mse(yhat, Y)
                res = {'set': name, 'head': head, 'seed': args.seed, 'in_dim': int(X.shape[1]), 'selection': sel,
                       'mse_train': float(mse[tr].mean()), 'mse_val': float(mse[va].mean()),
                       'mse_eval': float(mse[ev].mean()), 'r2_eval': r2(yhat[ev], Y[ev]),
                       'rank_eval': recall_eval_items(yhat[ev] * sd + mu, U_id, R_tr, R_gt, ev),
                       'seconds': time.time() - t0}
                np.savez_compressed(base + '.npz', mse=mse.astype(np.float32))
                json.dump(res, open(base + '.json', 'w'), indent=1)
                print('[probe] {:4s} {:5s}: eval MSE {:.4f} R2 {:.4f} | R@20 {:.4f} | {:.0f}s'.format(
                    name, head, res['mse_eval'], res['r2_eval'], res['rank_eval']['recall@20'], res['seconds']),
                    flush=True)
    summary(out, ev, ceiling, args)


def summary(out, ev, ceiling, args):
    runs = {}
    for f in glob.glob(os.path.join(out, '*_s{}.json'.format(args.seed))):
        r = json.load(open(f))
        r['_mse'] = np.load(f[:-5] + '.npz')['mse']
        runs[(r['set'], r['head'])] = r
    if not runs:
        print('no runs in', out)
        return
    for head in HEADS:                       # L2perm = mean of the permutations (per-item MSE averaged)
        rs = [runs[k] for k in runs if k[1] == head and k[0].startswith('L2perm') and k[0] != 'L2perm']
        if rs:
            m = np.mean([r['_mse'] for r in rs], 0)
            runs[('L2perm', head)] = {'set': 'L2perm', 'head': head, 'in_dim': rs[0]['in_dim'], '_mse': m,
                                      'mse_eval': float(m[ev].mean()),
                                      'r2_eval': float(np.mean([r['r2_eval'] for r in rs])),
                                      'mse_train': float(np.mean([r['mse_train'] for r in rs])),
                                      'rank_eval': {k: float(np.mean([r['rank_eval'][k] for r in rs]))
                                                    for k in ('recall@20', 'ndcg@20')}}
    info = json.load(open(os.path.join(out, 'inputs.json'))) if os.path.isfile(os.path.join(out, 'inputs.json')) else {}
    layers = sorted({k[0] for k in runs if k[0][0] == 'L' and k[0][1:].isdigit()})
    order = ['J0', 'J1', 'J2', 'J3', 'JD', 'Jg'] + layers + \
        sorted({k[0] for k in runs if k[0].startswith('L2perm') and k[0] != 'L2perm'}) + ['L2perm', 'L2const']
    lines = ['# Stage 2a: Text before / after graph -> ID preference ({})\n'.format(os.path.basename(out)),
             'Target: frozen ID teacher item embedding (standardised). Eval items: {}. Text teacher layers: {}; '
             'const teacher layers: {}. Ranking ceiling U_id . Y on eval items: R@20 {:.4f}.\n'.format(
                 len(ev), info.get('text_layers', '?'), info.get('const_layers', '-'), ceiling['recall@20']),
             '| input | head | dim | eval MSE | eval R2 | R@20 (eval items) | N@20 | train MSE |',
             '|---|---|---|---|---|---|---|---|']
    for name in order:
        for head in HEADS:
            r = runs.get((name, head))
            if r:
                lines.append('| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                    name, head, r['in_dim'], r['mse_eval'], r['r2_eval'], r['rank_eval']['recall@20'],
                    r['rank_eval']['ndcg@20'], r['mse_train']))
    lines += ['\n## Paired comparisons (eval items): dMSE = MSE(second) - MSE(first) > 0 means the first predicts Y better\n',
              '| comparison | meaning | head | dMSE | 95% CI | p (one-sided) |', '|---|---|---|---|---|---|']
    for a, b, meaning in PAIRS:
        for head in HEADS:
            ra, rb = runs.get((a, head)), runs.get((b, head))
            if ra and rb:
                bt = paired_bootstrap(rb['_mse'][ev] - ra['_mse'][ev], n_boot=args.n_boot)
                lines.append('| {} vs {} | {} | {} | {:+.4f} | [{:+.4f}, {:+.4f}] | {:.4f} |'.format(
                    a, b, meaning, head, bt['mean'], bt['ci_lo'], bt['ci_hi'], bt['p_one_sided']))
    text = '\n'.join(lines) + '\n'
    open(os.path.join(out, 'summary.md'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
