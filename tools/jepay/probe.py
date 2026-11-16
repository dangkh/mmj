# coding: utf-8
"""
Stage B: probes on FROZEN stage-A representations (the same representation under three predictor capacities).

    lin    ridge regression (closed form), ridge strength chosen on val items
    mlp1   MLP, 1 hidden layer of 128   } Adam lr 1e-3, batch 1024, <= 300 epochs, early stopping on val MSE
    mlp2   MLP, 2 hidden layers of 128  } (patience 20); weight decay chosen on val items
Inputs are standardised on the run's train items. Fitted on train items only; reported on eval items.
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from .model import mlp, HEADS

RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
WD_GRID = (1e-5, 1e-4, 1e-3)


def _std(X, tr):
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((X - mu) / sd).astype(np.float32)


def ridge_fit(X, Y, tr, va, alphas=RIDGE_ALPHAS):
    X64, Y64 = X.astype(np.float64), Y.astype(np.float64)
    xm, ym = X64[tr].mean(0), Y64[tr].mean(0)
    Xc = X64 - xm
    G, XtY = Xc[tr].T @ Xc[tr], Xc[tr].T @ (Y64[tr] - ym)
    n = len(tr)
    best = None
    for a in alphas:
        W = np.linalg.solve(G + a * n * np.eye(G.shape[0]), XtY)
        pred = Xc @ W + ym
        v = float(((pred[va] - Y64[va]) ** 2).mean())
        if best is None or v < best[0]:
            best = (v, a, pred)
    return best[2].astype(np.float32), {'alpha': best[1], 'val_mse': best[0]}


def mlp_fit(X, Y, tr, va, head, seed, device, wd_grid=WD_GRID, hidden=128, lr=1e-3, batch=1024,
            epochs=300, patience=20):
    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)
    trt = torch.from_numpy(np.asarray(tr)).long().to(device)
    vat = torch.from_numpy(np.asarray(va)).long().to(device)
    best_all = None
    for wd in wd_grid:
        torch.manual_seed(seed)
        net = mlp(X.shape[1], hidden, Y.shape[1], HEADS[head]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        best, state, bad, best_ep = float('inf'), None, 0, -1
        for ep in range(epochs):
            net.train()
            perm = trt[torch.randperm(len(trt), device=device)]
            for s in range(0, len(perm), batch):
                b = perm[s:s + batch]
                loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(((net(Xt[vat]) - Yt[vat]) ** 2).mean())
            if v < best - 1e-7:
                best, state, bad, best_ep = v, copy.deepcopy(net.state_dict()), 0, ep
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_all is None or best < best_all[0]:
            best_all = (best, wd, state, best_ep)
    v, wd, state, ep = best_all
    net = mlp(X.shape[1], hidden, Y.shape[1], HEADS[head]).to(device)
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(Xt[s:s + 8192]) for s in range(0, Xt.shape[0], 8192)]).cpu().numpy()
    return pred.astype(np.float32), {'wd': wd, 'val_mse': v, 'best_epoch': ep}


def fit_probe(X, Y, tr, va, head, seed, device, wd_grid=WD_GRID):
    """Predictions for ALL items from a probe of the given capacity fitted on tr, selected on va."""
    Xs = _std(X, tr)
    if head == 'lin':
        return ridge_fit(Xs, Y, tr, va)
    return mlp_fit(Xs, Y, tr, va, head, seed, device, wd_grid=wd_grid)


def probe_sets(cond, arr, p, full=True):
    """Input sets of the probes of one stage-A run. 'T' is the text-only reference of the same h_t.
    full=False (data-fraction runs): only the sets needed for H2."""
    ht = arr['h_t']
    cat = lambda *xs: np.concatenate(xs, 1)   # noqa: E731
    sets = {'T': ht}
    if cond == 'A0':
        sets.update({'T+PCA': cat(ht, p.pca64), 'T+Zlin': cat(ht, p.zlin)})
    elif cond == 'A1':
        sets['T+hv'] = cat(ht, arr['h_v'])
    elif cond == 'A2':
        sets.update({'T+Z': cat(ht, arr['z']), 'T+hv': cat(ht, arr['h_v']), 'T+PCA': cat(ht, p.pca64)})
        if full:
            sets.update({'T+S': cat(ht, arr['s']), 'T+SZ': cat(ht, arr['s'], arr['z']), 'T+Zlin': cat(ht, p.zlin)})
    else:                                      # A2sub, A2noJ, A2noY
        sets['T+Z'] = cat(ht, arr['z'])
    return sets
