# coding: utf-8
"""
Recommendation probe on frozen item representations (numpy/scipy, CPU only).

How much preference-relevant signal does an item representation Z carry? A probe that
sees items ONLY through Z is fitted on training interactions and scored on held-out ones:

    score(u, i) = (r_u Z) W z_i^T          r_u = user's (mean-normalised) history, W in R^{d x d}

There are no free user parameters, so all information must pass through Z. W solves a
two-sided ridge problem in closed form (unique, deterministic). Z is whitened first, so
the result does not depend on the coordinate system of Z (invertible linear maps).

Modes
  warm : fit on train edges (half of each user's history -> the other half), select the
         ridge strength on VALID edges, report Recall/NDCG@K on TEST edges (train items
         masked, as in MMRec).
  cold : items in K folds; fold f is never seen while fitting (neither as history nor as
         target), lambda chosen on fold f+1, then users rank the items of fold f from
         their history on the remaining items; ground truth = all their interactions with
         fold f. Measures generalisation from content to unseen items.

Representations (--rep name=spec, repeatable; spec = file.npy, or a+b to concatenate):
    python tools/probe_rec.py -d baby                                 # T0: raw v, t, v+t + baselines
    python tools/probe_rec.py -d baby --taps taps/FREEDOM-baby-freedom_s999
    python tools/probe_rec.py -d baby --rep my=path/to/Z.npy --modes warm

Baselines always included: pop (log train degree, warm only), null64 (Gaussian noise),
rp64_v / rp64_t (random 64-d projections of the raw features).

Important: in warm mode a d-dim representation lets the probe learn a rank-d item-item
CF model even when Z is pure noise (null64 is far above chance). Warm numbers are only
comparable at the same d (--warm_dim, default 64) and should be read as "minus null64".
Cold mode cannot memorise unseen items, so null64 is at chance there; it is the cleaner
measure of what content carries, and allows a larger PCA cap (--cold_dim).
Derived rows: incremental gain G(a|b) = p(a+b) - p(b) for every pair present, and the
residual gain p(T1_m+T3) - p(T3) when taps are given.

Output: printed table + log/probe_rec_<dataset>_<tag>.json
"""
import argparse
import glob
import json
import os
import sys
from time import time

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ----------------------------------------------------------------------------- data
def load_data(data_dir, dataset, uid='userID', iid='itemID', label='x_label', min_items=0):
    path = os.path.join(data_dir, dataset, '{}.inter'.format(dataset))
    df = pd.read_csv(path, sep='\t', usecols=[uid, iid, label])
    n_users = int(df[uid].max()) + 1
    n_items = max(int(df[iid].max()) + 1, min_items)     # same as MMRec unless features have extra rows
    mats = []
    train_users = set(df.loc[df[label] == 0, uid].values)
    for s in range(3):
        d = df[df[label] == s]
        if s > 0:                                   # MMRec: drop cold-start users from valid/test
            d = d[d[uid].isin(train_users)]
        m = sp.csr_matrix((np.ones(len(d), dtype=np.float32), (d[uid].values, d[iid].values)),
                          shape=(n_users, n_items))
        m.data[:] = 1.0
        mats.append(m)
    return mats, n_users, n_items


def load_rep(spec, base_dirs):
    parts = []
    for p in spec.split('+'):
        f = p if p.endswith('.npy') else p + '.npy'
        cand = [f] + [os.path.join(b, f) for b in base_dirs]
        hit = next((c for c in cand if os.path.isfile(c)), None)
        if hit is None:
            raise FileNotFoundError('{} (looked in {})'.format(f, cand))
        parts.append(np.load(hit).astype(np.float64))
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


def whiten(Z, fit_rows=None, max_dim=1024, eps=1e-6):
    """Centre + PCA-whiten (statistics from fit_rows); keep at most max_dim components."""
    X = Z if fit_rows is None else Z[fit_rows]
    mu = X.mean(0)
    Xc = X - mu
    if Xc.shape[1] > Xc.shape[0]:
        # more dims than rows: PCA through the Gram matrix
        G = Xc @ Xc.T
        ev, U = np.linalg.eigh(G)
        keep = ev > eps * ev.max()
        ev, U = ev[keep], U[:, keep]
        V = Xc.T @ U / np.sqrt(ev)
    else:
        C = Xc.T @ Xc
        ev, V = np.linalg.eigh(C)
        keep = ev > eps * ev.max()
        ev, V = ev[keep], V[:, keep]
    order = np.argsort(ev)[::-1][:max_dim]
    ev, V = ev[order], V[:, order]
    P = V / np.sqrt(ev / X.shape[0])
    return (Z - mu) @ P


def row_normalize(R):
    deg = np.asarray(R.sum(1)).ravel()
    inv = np.zeros_like(deg)
    inv[deg > 0] = 1.0 / deg[deg > 0]
    return sp.diags(inv.astype(np.float32)) @ R


def split_history(R, rng):
    """Each user's history -> random halves (input, target). Users with < 2 items are dropped."""
    R = R.tocsr()
    rin_r, rin_c, tg_r, tg_c = [], [], [], []
    for u in range(R.shape[0]):
        items = R.indices[R.indptr[u]:R.indptr[u + 1]]
        if len(items) < 2:
            continue
        items = rng.permutation(items)
        h = len(items) // 2
        rin_r += [u] * (len(items) - h); rin_c += list(items[h:])
        tg_r += [u] * h; tg_c += list(items[:h])
    mk = lambda r, c: sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=R.shape)
    return mk(rin_r, rin_c), mk(tg_r, tg_c)


# ----------------------------------------------------------------------------- probe
class TwoSidedRidge:
    """min_W ||T - A W B||^2 + lam ||W||^2  with A = R_in Z, B = Z_target^T (closed form)."""

    def __init__(self, Z, R_in, R_tg, tgt_cols):
        A = np.asarray(row_normalize(R_in) @ Z)               # n_users x d
        B = Z[tgt_cols].T                                      # d x n_tgt
        T = R_tg[:, tgt_cols]
        self.a, self.U = np.linalg.eigh(A.T @ A)
        self.b, self.V = np.linalg.eigh(B @ B.T)
        M = np.asarray((T.T @ A).T) @ B.T                     # A^T T B^T : d x d
        self.M = self.U.T @ M @ self.V
        self.scale = max(self.a.mean() * self.b.mean(), 1e-12)

    def W(self, alpha):
        D = np.outer(np.clip(self.a, 0, None), np.clip(self.b, 0, None)) + alpha * self.scale
        return self.U @ (self.M / D) @ self.V.T


def rank_metrics(Z, W, R_hist, R_gt, cand, R_mask=None, k=20, batch=2048):
    """Per-user Recall@k / NDCG@k of score = (r_u Z) W z_i^T over candidate items."""
    users = np.where((np.asarray(R_gt[:, cand].sum(1)).ravel() > 0) &
                     (np.asarray(R_hist.sum(1)).ravel() > 0))[0]
    WZ = W @ Z[cand].T                                         # d x n_cand
    Hn = row_normalize(R_hist)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    rec, ndcg = np.zeros(len(users)), np.zeros(len(users))
    gt_c = R_gt[:, cand].tocsr()
    mask_c = R_mask[:, cand].tocsr() if R_mask is not None else None
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        S = np.asarray(Hn[ub] @ Z) @ WZ
        if mask_c is not None:
            mb = mask_c[ub].tocoo()
            S[mb.row, mb.col] = -np.inf
        top = np.argpartition(-S, k, axis=1)[:, :k]
        order = np.take_along_axis(S, top, 1).argsort(1)[:, ::-1]
        top = np.take_along_axis(top, order, 1)
        g = gt_c[ub]
        for j in range(len(ub)):
            gi = g.indices[g.indptr[j]:g.indptr[j + 1]]
            hits = np.isin(top[j], gi).astype(float)
            n = len(gi)
            rec[s + j] = hits.sum() / n
            ndcg[s + j] = (hits * disc).sum() / disc[:min(n, k)].sum()
    return rec, ndcg


def boot_ci(x, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def probe_warm(Z_raw, mats, rng, k, max_dim, n_splits):
    R_tr, R_va, R_te = mats
    n_items = R_tr.shape[1]
    Z = whiten(Z_raw, max_dim=max_dim)
    Z = np.hstack([Z, np.ones((n_items, 1))])                 # item bias channel
    allc = np.arange(n_items)
    fits = []
    for _ in range(n_splits):
        R_in, R_tg = split_history(R_tr, rng)
        fits.append(TwoSidedRidge(Z, R_in, R_tg, allc))
    best = (-1, None)
    for al in ALPHAS:
        W = sum(f.W(al) for f in fits) / len(fits)
        r, _ = rank_metrics(Z, W, R_tr, R_va, allc, R_mask=R_tr, k=k)
        if r.mean() > best[0]:
            best = (r.mean(), al, W)
    _, al, W = best
    r, n = rank_metrics(Z, W, R_tr, R_te, allc, R_mask=R_tr, k=k)
    return {'recall': float(r.mean()), 'ndcg': float(n.mean()), 'recall_ci': boot_ci(r),
            'ndcg_ci': boot_ci(n), 'alpha': al, 'valid_recall': float(best[0]), 'dim': int(Z.shape[1] - 1),
            '_per_user': (r, n)}


def probe_cold(Z_raw, mats, rng, k, max_dim, n_folds):
    R_tr, R_va, R_te = mats
    R_all = (R_tr + R_va + R_te).tocsr()
    R_all.data[:] = 1.0
    n_items = R_tr.shape[1]
    fold = rng.permutation(n_items) % n_folds
    recs, ndcgs, alphas = [], [], []
    for f in range(n_folds):
        test_items = np.where(fold == f)[0]
        val_items = np.where(fold == (f + 1) % n_folds)[0]
        fit_items = np.where((fold != f) & (fold != (f + 1) % n_folds))[0]
        Z = whiten(Z_raw, fit_rows=fit_items, max_dim=max_dim)
        Z = np.hstack([Z, np.ones((n_items, 1))])
        keep = np.zeros(n_items, np.float32); keep[fit_items] = 1
        R_fit = R_tr @ sp.diags(keep)                        # history restricted to fit items
        R_fit.eliminate_zeros()
        R_in, R_tg = split_history(R_fit, rng)
        fit = TwoSidedRidge(Z, R_in, R_tg, fit_items)
        best = (-1, None, None)
        for al in ALPHAS:
            W = fit.W(al)
            r, _ = rank_metrics(Z, W, R_fit, R_all, val_items, k=k)
            if r.mean() > best[0]:
                best = (r.mean(), al, W)
        _, al, W = best
        r, n = rank_metrics(Z, W, R_fit, R_all, test_items, k=k)
        recs.append(r); ndcgs.append(n); alphas.append(al)
    r_fold = [x.mean() for x in recs]; n_fold = [x.mean() for x in ndcgs]
    return {'recall': float(np.mean(r_fold)), 'recall_std': float(np.std(r_fold, ddof=1)),
            'ndcg': float(np.mean(n_fold)), 'ndcg_std': float(np.std(n_fold, ddof=1)),
            'alphas': alphas, 'n_candidates': int(n_items / n_folds)}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='baby')
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--taps', default=None, help='taps folder written with --save_taps=True')
    ap.add_argument('--rep', action='append', default=[], help='name=spec (spec: a.npy or a+b)')
    ap.add_argument('--no_raw', action='store_true', help='skip the raw-feature (T0) rows')
    ap.add_argument('--modes', default='warm,cold')
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--warm_dim', type=int, default=64,
                    help='PCA cap in warm mode. Keep it equal for all rows: with d dims the probe can learn a '
                         'rank-d item-item CF model even from noise (see null64), so compare at matched d')
    ap.add_argument('--cold_dim', type=int, default=1024, help='PCA cap in cold mode (unseen items: no memorisation)')
    ap.add_argument('--n_splits', type=int, default=2, help='history splits averaged in the warm fit')
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--out_dir', default='log')
    args = ap.parse_args()

    t_start = time()
    ddir = os.path.join(args.data_dir, args.dataset)
    vf = os.path.join(ddir, 'image_feat.npy')
    n_feat = np.load(vf, mmap_mode='r').shape[0] if os.path.isfile(vf) else 0
    mats, n_users, n_items = load_data(args.data_dir, args.dataset, min_items=n_feat)
    print('[data] {} users={} items={} train={} valid={} test={}'.format(
        args.dataset, n_users, n_items, *[m.nnz for m in mats]))
    rng0 = np.random.default_rng(args.seed)

    reps = {}
    if not args.no_raw:
        xv = np.load(os.path.join(ddir, 'image_feat.npy')).astype(np.float64)
        xt = np.load(os.path.join(ddir, 'text_feat.npy')).astype(np.float64)
        reps['x_v'], reps['x_t'], reps['x_v+x_t'] = xv, xt, np.hstack([xv, xt])
        reps['rp64_v'] = xv @ rng0.standard_normal((xv.shape[1], 64)) / np.sqrt(xv.shape[1])
        reps['rp64_t'] = xt @ rng0.standard_normal((xt.shape[1], 64)) / np.sqrt(xt.shape[1])
    reps['null64'] = rng0.standard_normal((n_items, 64))      # warm: CF capacity of a rank-64 probe; cold: chance
    deg = np.asarray(mats[0].sum(0)).ravel()
    reps['pop'] = np.log1p(deg)[:, None]

    if args.taps:
        files = {os.path.basename(f)[:-4]: f for f in glob.glob(os.path.join(args.taps, 'T*.npy'))}
        for name in sorted(files):
            if name.endswith('_user'):
                continue
            reps[name] = np.load(files[name]).astype(np.float64)
        for lvl in ('T0ft', 'T1', 'T1g', 'T2'):
            if lvl + '_v' in reps and lvl + '_t' in reps:
                reps['{0}_v+{0}_t'.format(lvl)] = np.hstack([reps[lvl + '_v'], reps[lvl + '_t']])
        if 'T3' in reps:
            for m in ('v', 't'):
                if 'T1_' + m in reps:
                    reps['T1_{}+T3'.format(m)] = np.hstack([reps['T1_' + m], reps['T3']])
    for r in args.rep:
        name, spec = r.split('=', 1)
        reps[name] = load_rep(spec, [p for p in (args.taps, ddir) if p])

    for name, Z in reps.items():
        if Z.shape[0] != n_items:
            sys.exit('{}: {} rows, expected {} items'.format(name, Z.shape[0], n_items))

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    results = {}
    for name, Z in reps.items():
        results[name] = {'dim_in': int(Z.shape[1])}
        for mode in modes:
            if mode == 'cold' and name == 'pop':
                continue                                   # degree of an unseen item is unknown
            t0 = time()
            rng = np.random.default_rng(args.seed)          # same splits/folds for every representation
            if mode == 'warm':
                res = probe_warm(Z, mats, rng, args.k, args.warm_dim, args.n_splits)
                res.pop('_per_user')
            else:
                res = probe_cold(Z, mats, rng, args.k, args.cold_dim, args.n_folds)
            res['seconds'] = round(time() - t0, 1)
            results[name][mode] = res
            print('[{}] {:<14s} d={:<5d} R@{k}={:.4f} N@{k}={:.4f} ({:.0f}s)'.format(
                mode, name, Z.shape[1], res['recall'], res['ndcg'], res['seconds'], k=args.k))

    # derived: incremental and residual gains
    derived = {}
    for name in results:
        if '+' not in name:
            continue
        a, b = name.split('+', 1)
        for mode in modes:
            j = results[name].get(mode)
            if j is None:
                continue
            for x, y in ((a, b), (b, a)):
                if y in results and mode in results[y]:
                    derived['G[{}|{}]_{}'.format(x, y, mode)] = j['recall'] - results[y][mode]['recall']
    null = {m: results['null64'][m]['recall'] for m in modes if m in results.get('null64', {})}
    print('\nRecall@{} table  (gain rows: G[a|b] = p(a+b) - p(b); "-null" = minus the null64 row)'.format(args.k))
    print('{:<16s}'.format('rep') + ''.join('{:>12s}{:>10s}'.format(m, '-null') for m in modes))
    for name in results:
        cells = ''
        for m in modes:
            if m in results[name]:
                v = results[name][m]['recall']
                cells += '{:>12.4f}{:>+10.4f}'.format(v, v - null[m]) if m in null else '{:>12.4f}{:>10s}'.format(v, '')
            else:
                cells += '{:>12s}{:>10s}'.format('-', '')
        print('{:<16s}'.format(name) + cells)
    for key in sorted(derived):
        print('{:<34s} {:+.4f}'.format(key, derived[key]))

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or (os.path.basename(os.path.normpath(args.taps)) if args.taps else 'T0')
    out = os.path.join(args.out_dir, 'probe_rec_{}_{}.json'.format(args.dataset, tag))
    with open(out, 'w') as f:
        json.dump({'args': vars(args), 'results': results, 'derived': derived}, f, indent=1)
    print('\nSaved {} ({:.0f}s total)'.format(out, time() - t_start))


if __name__ == '__main__':
    main()
