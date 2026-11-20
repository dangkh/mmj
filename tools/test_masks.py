# coding: utf-8
"""
Unit tests of src/models/graph_aug.py (E0). On a random bipartite graph and, with -d, on the real training graph.
    mask ratio         uniform: masked fraction ~ rho; hub / tail: mean ~ rho (clipping can lower it), masking
                       probability increasing (hub) / decreasing (tail) with the item degree
    tail               every item with >= 1 training edge keeps >= 1 edge
    node               the chosen items have no kept edge; fraction of items ~ rho; the other items keep all edges
    norm_adj (torch)   with every edge kept = LightGCN.get_norm_adj_mat (same construction as models/lightgcn.py);
                       on a sub-graph: symmetric, values 1 / sqrt(d_u d_i) with the degrees of the sub-graph
    python tools/test_masks.py [-d baby]          exit code 1 if a test fails
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from models.graph_aug import EdgeSet          # noqa: E402

FAIL = []


def check(name, ok, info=''):
    print('[{}] {} {}'.format('ok  ' if ok else 'FAIL', name, info), flush=True)
    if not ok:
        FAIL.append(name)


def random_graph(n_users=3000, n_items=1500, seed=0):
    rng = np.random.default_rng(seed)
    pop = rng.pareto(1.2, n_items) + 1                      # long-tailed item popularity
    rows, cols = [], []
    for u in range(n_users):
        k = int(rng.integers(3, 30))
        it = rng.choice(n_items, size=k, replace=False, p=pop / pop.sum())
        rows += [u] * k
        cols += list(it)
    return sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n_users, n_items))


def test_masks(R, tag):
    E = EdgeSet(R.tocoo(), R.shape[0], R.shape[1])
    rng = np.random.default_rng(1)
    deg_e = E.deg_i[E.i]
    for rho in (0.03, 0.1, 0.15):
        keep = E.mask('uniform', rho, rng)
        f = 1 - keep.mean()
        check('{} uniform rho={}'.format(tag, rho), abs(f - rho) < 0.01 + 3 * np.sqrt(rho / E.n_edges), 'masked {:.4f}'.format(f))
        for mode in ('hub', 'tail'):
            fs, corr = [], []
            for _ in range(5):
                k = E.mask(mode, rho, rng, beta=1.0)
                fs.append(1 - k.mean())
                corr.append(np.corrcoef(np.log(deg_e), (~k).astype(float))[0, 1])
            f = np.mean(fs)
            c = np.mean(corr)
            sign_ok = c > 0 if mode == 'hub' else c < 0
            check('{} {} rho={}'.format(tag, mode, rho), sign_ok and f <= rho + 0.01 and f > 0.4 * rho,
                  'masked {:.4f}, corr(log deg, masked) {:+.3f}'.format(f, c))
        k = E.mask('tail', rho, rng, beta=1.0)
        kept = np.bincount(E.i[k], minlength=E.n_items)
        check('{} tail keeps >= 1 edge per item rho={}'.format(tag, rho), bool(np.all(kept[E.deg_i > 0] >= 1)))
        k = E.mask('node', rho, rng)
        kept = np.bincount(E.i[k], minlength=E.n_items)
        gone = (E.deg_i > 0) & (kept == 0)
        partial = (kept > 0) & (kept < E.deg_i)
        frac = gone.sum() / (E.deg_i > 0).sum()
        check('{} node rho={}'.format(tag, rho), abs(frac - rho) < 0.005 and not partial.any(),
              'items without edges {:.4f}, partially masked items {}'.format(frac, int(partial.sum())))
    k = E.node_mask(0.0, 0.1, rng)                      # SGL-ND, user side: a dropped user loses all edges
    ku = np.bincount(E.u[k], minlength=E.n_users)
    fu = ((E.deg_u > 0) & (ku == 0)).sum() / (E.deg_u > 0).sum()
    check('{} node_mask users'.format(tag), not ((ku > 0) & (ku < E.deg_u)).any() and abs(fu - 0.1) < 0.005,
          'users without edges {:.4f}'.format(fu))
    return E


def lightgcn_adj(R):
    """models/lightgcn.py get_norm_adj_mat, dense (small graphs only)."""
    nu, ni = R.shape
    A = sp.bmat([[None, R], [R.T, None]]).tocsr()
    d = np.asarray((A > 0).sum(1)).ravel() + 1e-7
    D = sp.diags(d ** -0.5)
    return (D @ A @ D).toarray()


def test_adj(R, tag):
    import torch
    E = EdgeSet(R.tocoo(), R.shape[0], R.shape[1])
    full = E.norm_adj(np.ones(E.n_edges, dtype=bool), 'cpu').to_dense().numpy()
    check('{} norm_adj(all kept) == LightGCN'.format(tag), np.allclose(full, lightgcn_adj(R.tocsr()), atol=1e-6))
    keep = E.mask('uniform', 0.3, np.random.default_rng(3))
    sub = sp.coo_matrix((np.ones(keep.sum(), np.float32), (E.u[keep], E.i[keep])), shape=R.shape)
    A = E.norm_adj(keep, 'cpu').to_dense().numpy()
    check('{} norm_adj(sub-graph) renormalised'.format(tag), np.allclose(A, lightgcn_adj(sub.tocsr()), atol=1e-6))
    check('{} norm_adj symmetric'.format(tag), np.allclose(A, A.T, atol=1e-7))
    del torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='')
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--no_torch', action='store_true')
    args = ap.parse_args()
    test_masks(random_graph(), 'random')
    if not args.no_torch:
        test_adj(random_graph(300, 150), 'random-small')
    if args.dataset:
        from probe_rec import load_data
        (R_tr, _, _), _, _ = load_data(args.data_dir, args.dataset)
        test_masks(R_tr, args.dataset)
    print('{} failed: {}'.format(len(FAIL), FAIL) if FAIL else 'all mask tests passed')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
