# coding: utf-8
"""
Data for the JEPA-Y diagnosis (diagnosis.md). numpy only.

Every statistic (feature scalers, Y scaler, PCA, the OLS of A3) is fitted on the TRAIN items of the run
(the 25 / 50 / 100 % subset), never on val / eval items.

Visual targets and baselines:
  pca_target(k)   PCA_k(V): reconstruction target of the decoder D (k chosen on val items). Scores are
                  scaled by ONE global factor (mean variance per component = 1 on train rows), so the
                  reconstruction error weights components by their variance (= error of V within the
                  top-k subspace).
  pca64           the same with k = 64: target of the fresh decoder probes (M5) and the A1-raw input
  zlin            A3: PCA_64(V) - OLS(PCA_64(V) ~ T), standardised
"""
import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(TOOLS)
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
from probe_rec import load_data   # noqa: E402

PERM_SEED = 2026                  # permutation of the E-B-lite control (<ds>_jepay_zperm)
FRAC_KEY = {1.0: 'train', 0.5: 'train_f50', 0.25: 'train_f25'}
K_MAX = 256


def standardize(X, rows, eps=1e-6):
    mu = X[rows].mean(0)
    sd = X[rows].std(0)
    sd = np.where(sd < eps, 1.0, sd)
    return ((X - mu) / sd).astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def pca_fit(X, rows, k):
    """Top-k principal directions of X[rows] (X centred on rows): (components k x d, eigenvalues k, total var)."""
    Xr = X[rows].astype(np.float64)
    n = Xr.shape[0]
    k = min(k, n - 1, Xr.shape[1])
    total = float((Xr ** 2).sum() / max(n - 1, 1))
    if Xr.shape[1] > n:                                    # Gram trick when d > n
        ev, U = np.linalg.eigh(Xr @ Xr.T)
        idx = np.argsort(ev)[::-1][:k]
        ev, U = np.maximum(ev[idx], 1e-12), U[:, idx]
        comps = (Xr.T @ U / np.sqrt(ev)).T
    else:
        ev, V = np.linalg.eigh(Xr.T @ Xr)
        idx = np.argsort(ev)[::-1][:k]
        ev, comps = np.maximum(ev[idx], 0.0), V[:, idx].T
    return comps.astype(np.float32), (ev / max(n - 1, 1)).astype(np.float64), total


def ridge_solve(X, Y, alpha):
    """argmin_B ||Y - X B||^2 + alpha ||B||^2 (X, Y already centred)."""
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    if X.shape[1] <= X.shape[0]:
        return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)
    return X.T @ np.linalg.solve(X @ X.T + alpha * np.eye(X.shape[0]), Y)


class Prepared(object):
    """Inputs of one data fraction: T, V, Y standardised on its train rows, plus the visual targets."""
    pass


class DiagData(object):
    def __init__(self, dataset, data_dir=None):
        self.dataset = dataset
        self.data_dir = data_dir or os.path.join(ROOT, 'data')
        path = os.path.join(self.data_dir, dataset)
        self.T_raw = np.load(os.path.join(path, 'text_feat.npy')).astype(np.float32)
        self.V_raw = np.load(os.path.join(path, 'image_feat.npy')).astype(np.float32)
        self.Y_raw = np.load(os.path.join(path, 'teacher_Y.npy')).astype(np.float32)
        self.U = np.load(os.path.join(path, 'teacher_U.npy')).astype(np.float32)
        sp_ = np.load(os.path.join(path, 'jepay_item_split.npz'))
        self.split = {k: sp_[k] for k in ('train', 'val', 'eval', 'train_f25', 'train_f50')}
        self.n_items = self.Y_raw.shape[0]
        assert self.T_raw.shape[0] == self.V_raw.shape[0] == self.n_items, 'feature / teacher rows differ'
        (self.R_tr, self.R_va, self.R_te), self.n_users, n_items = load_data(self.data_dir, dataset,
                                                                             min_items=self.n_items)
        assert n_items == self.n_items and self.U.shape[0] == self.n_users, 'interaction / teacher shapes differ'
        self._cache = {}

    def train_rows(self, frac):
        return self.split[FRAC_KEY[float(frac)]]

    def prepare(self, frac=1.0):
        frac = float(frac)
        key = ('prep', frac)
        if key in self._cache:
            return self._cache[key]
        rows = self.train_rows(frac)
        p = Prepared()
        p.frac = frac
        p.train, p.val, p.eval = rows, self.split['val'], self.split['eval']
        p.T, _, _ = standardize(self.T_raw, rows)
        p.V, _, _ = standardize(self.V_raw, rows)
        p.Y, p.y_mu, p.y_sd = standardize(self.Y_raw, rows)
        comps, ev, total = pca_fit(p.V, rows, K_MAX)
        p._pca_scores = (p.V @ comps.T).astype(np.float32)          # n x K_MAX, centred on train rows
        p._pca_ev, p._pca_total = ev, total
        p.dt, p.dv, p.dy = p.T.shape[1], p.V.shape[1], p.Y.shape[1]
        p.pca64 = self.pca_target(p, 64)
        p.zlin = self._zlin(p)
        self._cache[key] = p
        return p

    def pca_target(self, p, k):
        """PCA_k(V) scores, scaled so that the mean per-component variance on train rows is 1."""
        k = min(int(k), p._pca_scores.shape[1])
        S = p._pca_scores[:, :k]
        scale = np.sqrt(S[p.train].var(0).mean())
        return (S / max(scale, 1e-12)).astype(np.float32)

    def explained_variance(self, p, ks=(64, 128, 256)):
        """Share of the variance of standardised V kept by PCA_k: on train rows (fit) and on eval rows."""
        Vc = p.V
        out = {}
        ev_den = float(((Vc[p.eval] - Vc[p.train].mean(0)) ** 2).sum())
        for k in ks:
            k = min(k, p._pca_scores.shape[1])
            tr = float(p._pca_ev[:k].sum() / max(p._pca_total, 1e-12))
            S = p._pca_scores[p.eval, :k]
            ev = float((S.astype(np.float64) ** 2).sum() / max(ev_den, 1e-12))   # orthonormal comps
            out[str(k)] = {'train': tr, 'eval': ev}
        return out

    def _zlin(self, p):
        """A3: Z_lin = PCA_64(V) - OLS(PCA_64(V) ~ T), fitted on train rows (tiny ridge for stability)."""
        P = p._pca_scores[:, :64].astype(np.float64)
        Tr = p.T[p.train]
        alpha = 1e-3 * float((Tr.astype(np.float64) ** 2).sum()) / Tr.shape[1]
        B = ridge_solve(Tr, P[p.train] - P[p.train].mean(0), alpha)
        Z = (P - p.T @ B).astype(np.float32)
        Z, _, _ = standardize(Z, p.train)
        return Z

    def unstd_y(self, p, Yhat):
        return Yhat * p.y_sd + p.y_mu
