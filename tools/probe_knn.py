# coding: utf-8
"""
Parameter-free recommendation probe: item-kNN on a frozen item representation Z.

    sim(j, i)   = cosine(z_j, z_i)                     (pair a+b: w*cos_a + (1-w)*cos_b, w chosen on valid)
    score(u, i) = sum_{j in history(u)} sim_k(j, i)    sim_k keeps the k nearest neighbours of j

Nothing is fitted, so:
  * no capacity difference between a 4096-d raw feature and a 64-d tap, no regularisation metric,
  * no squared loss over the unobserved zeros of the interaction matrix,
  * no memorisation: random vectors (null64) give chance-level ranking, seen or unseen items alike.
The only choice, the neighbourhood size k, is picked on VALID; Recall/NDCG@K are reported on
TEST with train items masked, i.e. exactly the MMRec protocol, so the numbers read as
"how good a recommender is this geometry on its own".

Why cosine kNN: FREEDOM, LATTICE, MGCN and DRAGON all build their item-item graph as a
cosine kNN (k=10) of the item features, so this probe measures directly whether the
neighbourhoods of a representation are behaviourally meaningful.

    python tools/probe_knn.py -d baby                                   # T0: raw v, t, v+t + baselines
    python tools/probe_knn.py -d baby --taps taps/DRAGON-baby-dragon_paper
    python tools/probe_knn.py -d baby --rep my=path/Z.npy --ks 10,50

Rows: every representation, pop (rank by train degree), null64 (Gaussian noise: chance).
Gain rows: G[a|b] = p(a+b) - p(b) (what a adds to b's neighbourhoods).
Output: printed table + log/probe_knn_<dataset>_<tag>.json
"""
import argparse
import glob
import json
import os
import sys
from time import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_rec import load_data, boot_ci   # same data loading / CI as the ridge probe


def cosine(P):
    P = P.astype(np.float32)
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    C = P @ P.T
    np.fill_diagonal(C, -np.inf)
    return C


def topk_graph(S, k):
    """Keep the k largest similarities of each row (negative ones dropped) as a sparse matrix."""
    n = S.shape[0]
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    val = np.take_along_axis(S, idx, 1)
    val = np.where(np.isfinite(val) & (val > 0), val, 0.0).astype(np.float32)
    G = sp.csr_matrix((val.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    G.eliminate_zeros()
    return G


def evaluate(score_batch, R_hist, R_gt, R_mask, k=20, batch=4096):
    """Per-user Recall@k / NDCG@k; score_batch(users) -> dense (len(users) x n_items)."""
    users = np.where((np.asarray(R_gt.sum(1)).ravel() > 0) & (np.asarray(R_hist.sum(1)).ravel() > 0))[0]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    rec, ndcg = np.zeros(len(users)), np.zeros(len(users))
    rng = np.random.default_rng(0)
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        S = score_batch(ub).astype(np.float64)
        S += 1e-9 * rng.random(S.shape)                 # random tie-break (many zero scores)
        mb = R_mask[ub].tocoo()
        S[mb.row, mb.col] = -np.inf
        top = np.argpartition(-S, k, axis=1)[:, :k]
        top = np.take_along_axis(top, np.take_along_axis(S, top, 1).argsort(1)[:, ::-1], 1)
        g = R_gt[ub]
        for j in range(len(ub)):
            gi = g.indices[g.indptr[j]:g.indptr[j + 1]]
            hits = np.isin(top[j], gi).astype(float)
            rec[s + j] = hits.sum() / len(gi)
            ndcg[s + j] = (hits * disc).sum() / disc[:min(len(gi), k)].sum()
    return rec, ndcg


def probe(parts, mats, ks, topk, weights=(0.5,)):
    """One part: cosine kNN. Two parts (v, t): sim = w*cos_v + (1-w)*cos_t, w and k chosen on valid."""
    R_tr, R_va, R_te = mats
    Cs = [cosine(P) for P in parts]
    ws = weights if len(Cs) == 2 else (None,)
    best = None
    for w in ws:
        S = Cs[0] if w is None else w * Cs[0] + (1 - w) * Cs[1]
        for k in ks:
            G = topk_graph(S, k)
            r, _ = evaluate(lambda ub: (R_tr[ub] @ G).toarray(), R_tr, R_va, R_tr, topk)
            if best is None or r.mean() > best[0]:
                best = (r.mean(), k, w, G)
    vr, k, w, G = best
    r, n = evaluate(lambda ub: (R_tr[ub] @ G).toarray(), R_tr, R_te, R_tr, topk)
    out = {'recall': float(r.mean()), 'ndcg': float(n.mean()), 'recall_ci': boot_ci(r), 'ndcg_ci': boot_ci(n),
           'k_neighbours': int(k), 'valid_recall': float(vr)}
    if w is not None:
        out['w_first'] = float(w)
    return out


def probe_pop(mats, topk):
    R_tr, _, R_te = mats
    deg = np.asarray(R_tr.sum(0)).ravel().astype(np.float64)
    r, n = evaluate(lambda ub: np.tile(deg, (len(ub), 1)), R_tr, R_te, R_tr, topk)
    return {'recall': float(r.mean()), 'ndcg': float(n.mean()), 'recall_ci': boot_ci(r), 'ndcg_ci': boot_ci(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='baby')
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--taps', default=None, help='taps folder written with --save_taps=True')
    ap.add_argument('--rep', action='append', default=[], help='name=a.npy or name=a.npy+b.npy')
    ap.add_argument('--no_raw', action='store_true', help='skip the raw-feature (T0) rows')
    ap.add_argument('--ks', default='3,5,10,20,50,200', help='neighbourhood sizes, chosen on valid')
    ap.add_argument('--weights', default='0.1,0.3,0.5,0.7,0.9',
                    help='for a pair a+b: weight of a in w*cos_a + (1-w)*cos_b, chosen on valid '
                         '(the models use 0.1 image + 0.9 text)')
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--out_dir', default='log')
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(',')]
    weights = [float(x) for x in args.weights.split(',')]

    t_start = time()
    ddir = os.path.join(args.data_dir, args.dataset)
    vf = os.path.join(ddir, 'image_feat.npy')
    n_feat = np.load(vf, mmap_mode='r').shape[0] if os.path.isfile(vf) else 0
    mats, n_users, n_items = load_data(args.data_dir, args.dataset, min_items=n_feat)
    print('[data] {} users={} items={} train={} valid={} test={}'.format(
        args.dataset, n_users, n_items, *[m.nnz for m in mats]))

    reps = {}                                   # name -> list of parts (a pair = two parts)
    if not args.no_raw:
        xv = np.load(os.path.join(ddir, 'image_feat.npy'))
        xt = np.load(os.path.join(ddir, 'text_feat.npy'))
        reps.update({'x_v': [xv], 'x_t': [xt], 'x_v+x_t': [xv, xt]})
    reps['null64'] = [np.random.default_rng(args.seed).standard_normal((n_items, 64))]
    if args.taps:
        files = {os.path.basename(f)[:-4]: f for f in glob.glob(os.path.join(args.taps, 'T*.npy'))}
        for name in sorted(files):
            if not name.endswith('_user'):
                reps[name] = [np.load(files[name])]
        for lvl in ('T0ft', 'T1', 'T1g', 'T2'):
            if lvl + '_v' in reps and lvl + '_t' in reps:
                reps['{0}_v+{0}_t'.format(lvl)] = reps[lvl + '_v'] + reps[lvl + '_t']
    for r in args.rep:
        name, spec = r.split('=', 1)
        reps[name] = [np.load(p) for p in spec.split('+')]
    for name, parts in reps.items():
        for P in parts:
            if P.shape[0] != n_items:
                sys.exit('{}: {} rows, expected {} items'.format(name, P.shape[0], n_items))

    results = {'pop': probe_pop(mats, args.topk)}
    print('[knn] {:<14s} R@{k}={:.4f} N@{k}={:.4f}'.format('pop', results['pop']['recall'],
                                                         results['pop']['ndcg'], k=args.topk))
    for name, parts in reps.items():
        t0 = time()
        res = probe(parts, mats, ks, args.topk, weights)
        res['dims'] = [int(P.shape[1]) for P in parts]
        res['seconds'] = round(time() - t0, 1)
        results[name] = res
        print('[knn] {:<14s} R@{k}={:.4f} N@{k}={:.4f}  k_nb={}{} ({:.0f}s)'.format(
            name, res['recall'], res['ndcg'], res['k_neighbours'],
            '  w_first={}'.format(res['w_first']) if 'w_first' in res else '', res['seconds'], k=args.topk))

    derived = {}
    for name in results:
        if '+' in name:
            a, b = name.split('+', 1)
            for x, y in ((a, b), (b, a)):
                if y in results:
                    derived['G[{}|{}]'.format(x, y)] = results[name]['recall'] - results[y]['recall']

    print('\nitem-kNN probe, Recall@{k} / NDCG@{k} on test (95% CI over users)'.format(k=args.topk))
    for name, r in results.items():
        print('  {:<16s} {:.4f} [{:.4f},{:.4f}]   {:.4f}'.format(name, r['recall'], *r['recall_ci'], r['ndcg']))
    for key in sorted(derived):
        print('  {:<30s} {:+.4f}'.format(key, derived[key]))

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or (os.path.basename(os.path.normpath(args.taps)) if args.taps else 'T0')
    out = os.path.join(args.out_dir, 'probe_knn_{}_{}.json'.format(args.dataset, tag))
    with open(out, 'w') as f:
        json.dump({'args': vars(args), 'results': results, 'derived': derived}, f, indent=1)
    print('\nSaved {} ({:.0f}s total)'.format(out, time() - t_start))


if __name__ == '__main__':
    main()
