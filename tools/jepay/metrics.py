# coding: utf-8
"""
Metrics of the E-A diagnosis (diagnosis.md section 6). numpy / scipy only.

    M1, M2   per-item squared errors of the stage-B probes (compared in summarize_jepay_diag.py)
    M3       prediction change   mean||Y_T+X - Y_T||^2 / mean||Y_T||^2, same probe head (eval items)
    M4       cross-modal predictability (descriptive, not a criterion): R^2 of Z_v, S_v, h_v <- h_t
    M5       visual content kept: fresh linear decoders PCA_64(V) <- {h_t, h_v, S_v, Z_v, [S_v;Z_v],
             [P(h_t);Z_v], [h_t;Z_v]}, and the model's own decoder D(S_v,Z_v), D(P(h_t),Z_v) vs PCA_k(V)
    M6       collapse (effective rank, per-dim std) and decorrelation (mean |corr| of S_v-Z_v, Z_v-h_t)
    M7       memorisation gap    (MSE_eval - MSE_train) / MSE_eval of each probe
    M8       collaborative ranking: Recall@20 / NDCG@20 of eval items scored by U . Y_hat (teacher users)
Ridge probes: fitted on train items, ridge strength chosen on val items, R^2 reported on eval items.
"""
import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
from probe_knn import evaluate   # noqa: E402  (MMRec protocol: per-user Recall/NDCG, train items masked)

ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def per_item_mse(Yhat, Y):
    return ((Yhat - Y) ** 2).mean(1)


def r2(Yhat, Y):
    """Variance-weighted R^2 (1 - SSE / SST over all output dims)."""
    sse = ((Y - Yhat) ** 2).sum()
    sst = ((Y - Y.mean(0)) ** 2).sum()
    return float(1.0 - sse / max(sst, 1e-12))


def ridge_probe(X, Y, tr, va, ev, alphas=ALPHAS):
    """Fresh ridge Y <- X: fit on tr, alpha on va (R^2), R^2 on ev. X standardised with tr statistics."""
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = (X - mu) / sd
    ym = Y[tr].mean(0)
    Xt, Yt = Xs[tr], Y[tr] - ym
    G, XtY = Xt.T @ Xt, Xt.T @ Yt
    n = len(tr)
    best = None
    for a in alphas:
        W = np.linalg.solve(G + a * n * np.eye(G.shape[0]), XtY)
        s = r2(Xs[va] @ W + ym, Y[va])
        if best is None or s > best[0]:
            best = (s, a, W)
    _, a, W = best
    return {'r2': r2(Xs[ev] @ W + ym, Y[ev]), 'r2_val': float(best[0]), 'alpha': float(a)}


def effective_rank(Z):
    Zc = Z - Z.mean(0)
    s = np.linalg.svd(Zc.astype(np.float64), compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def collapse_stats(Z):
    return {'erank': effective_rank(Z), 'mean_std': float(Z.std(0).mean()), 'min_std': float(Z.std(0).min())}


def recall_eval_items(Yhat_ev, U, R_hist, R_gt, eval_idx, k=20):
    """Users rank the EVAL items only, by U . Y_hat (unstandardised). Ground truth: their valid + test
    interactions with eval items; their train interactions with eval items are masked."""
    gt = R_gt[:, eval_idx].tocsr()
    mask = R_hist[:, eval_idx].tocsr()
    Yt = Yhat_ev.astype(np.float32).T
    rec, ndcg = evaluate(lambda ub: U[ub] @ Yt, R_hist, gt, mask, k=k)
    return {'recall@{}'.format(k): float(rec.mean()), 'ndcg@{}'.format(k): float(ndcg.mean()), 'n_users': int(len(rec))}


def mean_abs_corr(a, b):
    """Mean |Pearson correlation| over all dimension pairs of a and b (rows = items)."""
    a = (a - a.mean(0)) / np.maximum(a.std(0), 1e-8)
    b = (b - b.mean(0)) / np.maximum(b.std(0), 1e-8)
    return float(np.abs(a.T @ b / a.shape[0]).mean())


def representation_metrics(p, arr, v_target=None):
    """M4, M5, M6 of one frozen stage-A model. arr: numpy outputs for ALL items (h_t, h_v, s, z, recon, ...)."""
    tr, va, ev = p.train, p.val, p.eval
    m = {'M6_ht': collapse_stats(arr['h_t'][ev])}
    if 'h_v' not in arr:
        return m
    ht = arr['h_t']
    # M4: cross-modal predictability (descriptive only)
    m['M4_hv_from_ht'] = ridge_probe(ht, arr['h_v'], tr, va, ev)
    m['M4_z_from_ht'] = ridge_probe(ht, arr['z'], tr, va, ev)
    if 's' in arr:
        m['M4_s_from_ht'] = ridge_probe(ht, arr['s'], tr, va, ev)
    # M5: visual content kept, fresh linear decoder probes -> PCA_64(V)
    V64 = p.pca64
    probes = {'ht': ht, 'hv': arr['h_v'], 'z': arr['z'], 'ht_z': np.concatenate([ht, arr['z']], 1)}
    if 's' in arr:
        probes.update({'s': arr['s'], 's_z': np.concatenate([arr['s'], arr['z']], 1)})
    if 's_hat' in arr:
        probes['shat_z'] = np.concatenate([arr['s_hat'], arr['z']], 1)
    m['M5_probe'] = {k: ridge_probe(X, V64, tr, va, ev) for k, X in probes.items()}
    # M5: the model's own decoder, D(S_v, Z_v) and D(P(h_t), Z_v), against its PCA_k target
    if 'recon' in arr and v_target is not None:
        m['M5_D_s_z'] = r2(arr['recon'][ev], v_target[ev])
        if 'recon_shat' in arr:
            m['M5_D_shat_z'] = r2(arr['recon_shat'][ev], v_target[ev])
    # M6: collapse and decorrelation
    m['M6_hv'] = collapse_stats(arr['h_v'][ev])
    m['M6_z'] = collapse_stats(arr['z'][ev])
    m['corr_z_ht'] = mean_abs_corr(arr['z'][ev], ht[ev])
    if 's' in arr:
        m['M6_s'] = collapse_stats(arr['s'][ev])
        m['corr_s_z'] = mean_abs_corr(arr['s'][ev], arr['z'][ev])
    return m


def probe_metrics(D, p, yhat, Y, rank=False):
    """M7 (memorisation gap) and, optionally, M8 for one probe's predictions on all items."""
    mse = per_item_mse(yhat, Y)
    m = {'mse_train': float(mse[p.train].mean()), 'mse_val': float(mse[p.val].mean()),
         'mse_eval': float(mse[p.eval].mean())}
    m['M7'] = (m['mse_eval'] - m['mse_train']) / max(m['mse_eval'], 1e-12)
    if rank:
        R_gt = (D.R_va + D.R_te).tocsr()
        R_gt.data[:] = 1.0
        m['M8'] = recall_eval_items(D.unstd_y(p, yhat[p.eval]), D.U, D.R_tr, R_gt, p.eval)
    return m, mse


def teacher_ceiling(D):
    """M8 with the teacher's own Y on eval items: the upper reference for U . Y_hat."""
    ev = D.split['eval']
    R_gt = (D.R_va + D.R_te).tocsr()
    R_gt.data[:] = 1.0
    return recall_eval_items(D.Y_raw[ev], D.U, D.R_tr, R_gt, ev)


# ---------------------------------------------------------------------------------- statistics
def paired_bootstrap(d, n_boot=10000, seed=0, chunk=500):
    """Bootstrap over items of mean(d). Returns mean, 95% CI, one-sided p for H0: mean <= 0."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        idx = rng.integers(0, n, size=(b, n), dtype=np.int32)
        means[s:s + b] = d[idx].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = max((means <= 0).mean(), 1.0 / n_boot)
    return {'mean': float(d.mean()), 'ci_lo': float(lo), 'ci_hi': float(hi), 'p_one_sided': float(p), 'n': int(n)}


def holm(pvals):
    """Holm-adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (m - r) * p[i])
        adj[i] = min(1.0, running)
    return adj
