# coding: utf-8
"""
Where does DRAGON's T2 gain come from: the item content or the user-item graph?

DRAGON's modality GCN (models/dragon.py, class GCN) computes, for modality m,
    x   = row_normalise( [P_m ; T1_m] )          P_m: user preferences (user-ID parameters), T1_m: item MLP output
    h1  = A x,  h2 = A h1                        A = D^-1/2 A_UI D^-1/2 on the train user-item graph
    T2  = x + h1 + h2                            (item rows)
so T2 mixes the item's own content with the users' ID parameters and the 2-hop items.
This script rebuilds T2 offline from the saved taps and swaps the inputs (2 x 2):

    content  : T1_m (real)        or Gaussian noise with the same shape (structure only)
    users    : P_m (trained)      or zeros (no user-ID signal; only item -> user -> item paths)

and runs the item-kNN probe (tools/probe_knn.py) on every variant. The "real + P" variant
must reproduce the saved T2_m (checked and printed).

    python tools/graph_control.py -d baby --taps taps/DRAGON-baby-dragon_taps
Needs the taps T1_v/T1_t, T2_v/T2_t and P_v/P_t (saved by src/taps.py since P_* was added).
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_rec import load_data            # noqa: E402
from probe_knn import probe, probe_pop, paired_ci     # noqa: E402


def row_normalise(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)   # = F.normalize


def sym_adj(R):
    n_u, n_i = R.shape
    A = sp.bmat([[None, R], [R.T, None]], format='csr').astype(np.float64)
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    d = np.zeros_like(deg); d[deg > 0] = deg[deg > 0] ** -0.5
    D = sp.diags(d)
    return (D @ A @ D).tocsr()


def dragon_t2(A, P, items):
    x = row_normalise(np.vstack([P, items]))
    h1 = A @ x
    h2 = A @ h1
    return (x + h1 + h2)[P.shape[0]:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='baby')
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--taps', required=True)
    ap.add_argument('--ks', default='3,5,10,20,50,200')
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out_dir', default='log')
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(',')]

    ddir = os.path.join(args.data_dir, args.dataset)
    n_feat = np.load(os.path.join(ddir, 'image_feat.npy'), mmap_mode='r').shape[0]
    mats, n_users, n_items = load_data(args.data_dir, args.dataset, min_items=n_feat)
    A = sym_adj(mats[0])
    rng = np.random.default_rng(args.seed)
    ld = lambda n: np.load(os.path.join(args.taps, n + '.npy')).astype(np.float64)

    weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = {'pop': probe_pop(mats, args.topk)}
    store = {}
    print('{:<22s} R@{}={:.4f}'.format('pop', args.topk, results['pop']['recall']))
    for m in ('v', 't'):
        if not os.path.isfile(os.path.join(args.taps, 'P_{}.npy'.format(m))):
            sys.exit('P_{}.npy missing in {}: re-run DRAGON with the current src/taps.py'.format(m, args.taps))
        P, T1, T2 = ld('P_' + m), ld('T1_' + m), ld('T2_' + m)
        noise = rng.standard_normal(T1.shape)
        rebuilt = dragon_t2(A, P, T1)
        err = np.abs(rebuilt - T2).max() / max(np.abs(T2).max(), 1e-12)
        print('[check] modality {}: rebuilt T2 vs saved T2, max rel. diff = {:.2e}'.format(m, err))
        variants = {
            'content+users': rebuilt,
            'noise+users': dragon_t2(A, P, noise),            # structure + user IDs, no content
            'content+0users': dragon_t2(A, np.zeros_like(P), T1),   # content + item-user-item paths only
            'noise+0users': dragon_t2(A, np.zeros_like(P), noise),  # pure structure
            'T1 (no graph)': T1,
        }
        store[m] = variants
        for name, Z in variants.items():
            key = '{}:{}'.format(m, name)
            results[key] = probe([Z], mats, ks, args.topk)
            print('{:<22s} R@{}={:.4f}  [{:.4f},{:.4f}]  k_nb={}'.format(
                key, args.topk, results[key]['recall'], *results[key]['recall_ci'], results[key]['k_neighbours']))
        results['{}:check_rel_diff'.format(m)] = float(err)

    # both modalities together (weighted cosine, w chosen on valid): is the v+t gain content or two CF channels?
    for name in ('content+users', 'noise+users', 'content+0users', 'T1 (no graph)'):
        key = 'v+t:{}'.format(name)
        results[key] = probe([store['v'][name], store['t'][name]], mats, ks, args.topk, weights)
        print('{:<22s} R@{}={:.4f}  [{:.4f},{:.4f}]  k_nb={} w_v={}'.format(
            key, args.topk, results[key]['recall'], *results[key]['recall_ci'],
            results[key]['k_neighbours'], results[key].get('w_first')))
    print('\nPaired differences (95% bootstrap CI over users):')
    pairs = [('content+users', 'noise+users', 'content beyond structure+user IDs'),
             ('content+users', 'content+0users', 'user-ID signal'),
             ('content+users', 'T1 (no graph)', 'what the graph adds to the content')]
    diffs = {}
    for scope in ('v', 't', 'v+t'):
        for a_, b_, what in pairs:
            ka, kb = '{}:{}'.format(scope, a_), '{}:{}'.format(scope, b_)
            g = results[ka]['recall'] - results[kb]['recall']
            ci = paired_ci(results[ka]['_r'], results[kb]['_r'])
            diffs['{} | {} - {}'.format(scope, a_, b_)] = {'gain': g, 'ci': ci}
            print('  {:<4s} {:<34s} {:+.4f}  [{:+.4f},{:+.4f}]'.format(scope, what, g, *ci))
    results = {k: ({kk: vv for kk, vv in v.items() if kk != '_r'} if isinstance(v, dict) else v)
               for k, v in results.items()}
    results['_paired'] = diffs
    print('\nRead: content contribution after the graph = (content+users) - (noise+users);'
          ' user-ID contribution = (content+users) - (content+0users).')
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'graph_control_{}_{}.json'.format(
        args.dataset, os.path.basename(os.path.normpath(args.taps))))
    with open(out, 'w') as f:
        json.dump({'args': vars(args), 'results': results}, f, indent=1)
    print('Saved', out)


if __name__ == '__main__':
    main()
