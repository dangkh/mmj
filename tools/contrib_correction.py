# coding: utf-8
"""
Conditional contribution: frozen score fusion (SF) + a zero-initialised correction from a Text residual.

Frozen teachers (LightGCNC taps at the best-valid epoch):
    ID    U_id, Y_id                         taps/LightGCNC-<ds>-teacher_id
    Text  U_t and item layers T^(0..L)       taps/LightGCNC-<ds>-teacher_text_layers
          Y_t = mean_l T^(l) (teacher output), TO = mean of even layers (text-origin), UO = mean of odd layers
          (user-origin), see tools/path_fusion.py
Base (A0):  s_SF(u, i) = (1 - a) z(s_ID) + a z(s_T),  a in {0, ..., 1} chosen on the selection interactions
Corrected:  s(u, i) = s_SF(u, i) + Delta(u, i),   Delta(u, i) = < G(U_id[u]), C([Y_id[i] ; X_i]) >
            G, C: 2-layer MLPs (same size in every config); last layer of C zero-initialised, so Delta = 0 and
            s = s_SF at epoch 0, which is part of the selection (no config can be below SF on valid).
            The scale lambda of the formulation is absorbed in C.  Loss: BPR + lam_D * E[Delta^2].
Item content X_i (the only difference between the configs):
    A1  z(TO_i)                                          correction from the text-origin pathway directly
    A2  R_B3 = z(Y_t) - P(z(Y_id))                       old B3 residual (full text predicted from ID)
    A3  R_TO = z(TO) - P([z(Y_id) ; z(UO)])             text-origin not predicted from ID and user-origin
    P: frozen predictor (stage 1), fitted out-of-fold over items (K folds; each item's residual comes from a
       predictor that never saw it), head mlp2 by default (lin = ridge). z = standardisation over all items.
Protocol (same for A1-A3): grid lr x lam_D with the first seed, selection on valid R@20 (eval every 2 epochs,
patience 10), the other seeds for the selected setting. --fit valid: the correction is fitted on half of the
valid interactions and selected on the other half (SF weight a is selected on the same half).
Paired bootstrap over test users (first seed): A1/A2/A3 vs A0, A3 vs A1, A3 vs A2, A2 vs A1.

    python tools/contrib_correction.py -d baby                       # residuals, SF, A1-A3
    python tools/contrib_correction.py -d baby --fit valid
    python tools/contrib_correction.py -d baby --summary
Output: results/contrib[_fitvalid]/<ds>/
"""
import argparse
import copy
import glob
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from probe_rec import load_data                                          # noqa: E402
from probe_knn import evaluate                                           # noqa: E402
from consolidate import mlp, zstats, make_fit, sample_triples, SF_GRID   # noqa: E402
from jepay.probe import fit_probe                                        # noqa: E402
from jepay.metrics import paired_bootstrap                               # noqa: E402

CONFIGS = {'A1': 'TO', 'A2': 'R_B3', 'A3': 'R_TO'}
LABEL = {'A0': 'SF (frozen)', 'A1': 'SF + correction from TO', 'A2': 'SF + correction from R_B3 (Y_t | ID)',
         'A3': 'SF + correction from R_TO (TO | ID, UO)'}
GRID = {'lr': [1e-3, 1e-4], 'lam_D': [1e-3, 1e-2, 1e-1]}
PAIRS = [('A1', 'A0'), ('A2', 'A0'), ('A3', 'A0'), ('A3', 'A1'), ('A3', 'A2'), ('A2', 'A1')]


def zs(X):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((X - mu) / sd).astype(np.float32)


def layers_of(path):
    files = sorted(glob.glob(os.path.join(path, 'T_item_l*.npy')), key=lambda f: int(f.split('_l')[-1][:-4]))
    if not files:
        raise SystemExit('{}: no T_item_l*.npy (run scripts/text_graph_probe.pbs first)'.format(path))
    return [np.load(f).astype(np.float32) for f in files]


def oof_residual(X, Y, k, head, seed, device):
    """Residual Y - P(X), P fitted out-of-fold: K item folds; inside the other folds 10% select the probe."""
    n = X.shape[0]
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    pred = np.zeros_like(Y)
    sel = []
    for f in range(k):
        held = folds[f]
        rest = np.random.default_rng(seed + 1 + f).permutation(np.concatenate([folds[j] for j in range(k) if j != f]))
        nv = max(1, len(rest) // 10)
        yhat, s = fit_probe(X, Y, np.sort(rest[nv:]), np.sort(rest[:nv]), head, seed, device)
        pred[held] = yhat[held]
        sel.append(s)
    R = (Y - pred).astype(np.float32)
    sst = float(((Y - Y.mean(0)) ** 2).sum())
    return R, {'r2_oof': 1.0 - float((R ** 2).sum()) / sst, 'res_energy': float((R ** 2).sum()) / float((Y ** 2).sum()),
               'selection': sel}


class SFBase:
    """Frozen s_SF = (1 - a) z(s_ID) + a z(s_T)."""

    def __init__(self, U_id, Y_id, U_t, Y_t, device):
        t = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
        self.U_id, self.Y_id, self.U_t, self.Y_t = t(U_id), t(Y_id), t(U_t), t(Y_t)
        self.st_id, self.st_t = zstats(self.U_id, self.Y_id), zstats(self.U_t, self.Y_t)
        self.a = 0.5

    def pair(self, u, i):
        s_id = ((self.U_id[u] * self.Y_id[i]).sum(1) - self.st_id[0]) / self.st_id[1]
        s_t = ((self.U_t[u] * self.Y_t[i]).sum(1) - self.st_t[0]) / self.st_t[1]
        return (1 - self.a) * s_id + self.a * s_t

    def full(self, u, a=None):
        a = self.a if a is None else a
        s_id = (self.U_id[u] @ self.Y_id.T - self.st_id[0]) / self.st_id[1]
        s_t = (self.U_t[u] @ self.Y_t.T - self.st_t[0]) / self.st_t[1]
        return (1 - a) * s_id + a * s_t


class Correction(nn.Module):
    def __init__(self, Uz, Xi, hidden, d):
        super().__init__()
        self.register_buffer('Uz', Uz)
        self.register_buffer('Xi', Xi)
        self.G = mlp(Uz.shape[1], hidden, d)
        self.C = mlp(Xi.shape[1], hidden, d)
        nn.init.zeros_(self.C[-1].weight)
        nn.init.zeros_(self.C[-1].bias)

    def pair(self, u, i):
        return (self.G(self.Uz[u]) * self.C(self.Xi[i])).sum(1)

    def full(self, u, Ci):
        return self.G(self.Uz[u]) @ Ci.T


def evaluate_split(score, fit, split):
    gt, mask = (fit['sel_gt'], fit['sel_mask']) if split == 'valid' else (fit['test_gt'], fit['test_mask'])
    r, n = evaluate(score, fit['hist'], gt, mask, k=20)
    return {'recall@20': float(r.mean()), 'ndcg@20': float(n.mean())}, r, n


def scorer(base, corr, device):
    Ci = None
    if corr is not None:
        corr.eval()
        with torch.no_grad():
            Ci = corr.C(corr.Xi)

    @torch.no_grad()
    def score(ub):
        u = torch.from_numpy(ub).long().to(device)
        s = base.full(u)
        if corr is not None:
            s = s + corr.full(u, Ci)
        return s.cpu().numpy()
    return score


def train(args, fit, base, Uz, Xi, cfg, seed, device, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Correction(Uz, Xi, args.hidden, args.d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    best = evaluate_split(scorer(base, model, device), fit, 'valid')[0]          # epoch 0 = SF
    best_ep, best_state, bad, hist = 0, copy.deepcopy(model.state_dict()), 0, []
    log('  epoch 0 valid R@20 {:.4f}'.format(best['recall@20']))
    n_per_epoch = fit['pos'].nnz if fit['how'] == 'train' else max(fit['pos'].nnz, 100000)
    for ep in range(1, args.epochs + 1):
        model.train()
        u, p, n = sample_triples(fit['pos'], n_per_epoch, rng, fit['excl'])
        tot = {'bpr': 0.0, 'D': 0.0}
        for s in range(0, n_per_epoch, args.batch):
            ub = torch.from_numpy(u[s:s + args.batch]).long().to(device)
            pb = torch.from_numpy(p[s:s + args.batch]).long().to(device)
            nb = torch.from_numpy(n[s:s + args.batch]).long().to(device)
            dp, dn = model.pair(ub, pb), model.pair(ub, nb)
            with torch.no_grad():
                bp, bn = base.pair(ub, pb), base.pair(ub, nb)
            bpr = -F.logsigmoid((bp + dp) - (bn + dn)).mean()
            pen = 0.5 * ((dp ** 2).mean() + (dn ** 2).mean())
            loss = bpr + cfg['lam_D'] * pen
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot['bpr'] += float(bpr)
            tot['D'] += float(pen)
        if ep % args.eval_every:
            continue
        va = evaluate_split(scorer(base, model, device), fit, 'valid')[0]
        hist.append({'epoch': ep, 'valid_recall@20': va['recall@20'], **tot})
        log('  epoch {} valid R@20 {:.4f} | bpr {:.4f} E[D^2] {:.4f}'.format(ep, va['recall@20'], tot['bpr'], tot['D']))
        if va['recall@20'] > best['recall@20'] + 1e-6:
            best, best_ep, best_state, bad = va, ep, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    te, r, n = evaluate_split(scorer(base, model, device), fit, 'test')
    with torch.no_grad():
        model.eval()
        d_abs = float(model.C(model.Xi).abs().mean())
    return {'cfg': cfg, 'seed': seed, 'best_epoch': best_ep, 'valid': best, 'test': te, 'history': hist,
            'item_side_abs_mean': d_abs}, r, n


def save_json(obj, f):
    tmp = '{}.{}.tmp'.format(f, os.getpid())
    json.dump(obj, open(tmp, 'w'), indent=1)
    os.replace(tmp, f)


def tag(c):
    return '_'.join('{}{:g}'.format(k, c[k]) for k in sorted(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True)
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--id_taps', default=None)
    ap.add_argument('--text_taps', default=None)
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'results', 'contrib'))
    ap.add_argument('--configs', default='A1 A2 A3')
    ap.add_argument('--fit', choices=['train', 'valid'], default='train')
    ap.add_argument('--seeds', default='999')
    ap.add_argument('--pred_head', default='mlp2', help='predictor P of stage 1: lin (ridge), mlp1, mlp2')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--eval_every', type=int, default=2)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    taps = lambda name: os.path.join(ROOT, 'taps', 'LightGCNC-{}-{}'.format(args.dataset, name))  # noqa: E731
    id_taps, text_taps = args.id_taps or taps('teacher_id'), args.text_taps or taps('teacher_text_layers')
    out = os.path.join(args.out_dir.rstrip('/') + ('' if args.fit == 'train' else '_fit' + args.fit), args.dataset)
    os.makedirs(out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split()]
    if args.summary:
        return write_summary(out, args)

    U_id = np.load(os.path.join(id_taps, 'T3_user.npy')).astype(np.float32)
    Y_id = np.load(os.path.join(id_taps, 'T3.npy')).astype(np.float32)
    U_t = np.load(os.path.join(text_taps, 'T3_user.npy')).astype(np.float32)
    L = layers_of(text_taps)
    Y_t, TO, UO = np.mean(L, 0), np.mean(L[0::2], 0), np.mean(L[1::2], 0)
    mats, n_users, n_items = load_data(args.data_dir, args.dataset, min_items=Y_id.shape[0])
    assert (n_users, n_items) == (U_id.shape[0], Y_id.shape[0]) == (U_t.shape[0], L[0].shape[0])
    fit = make_fit(mats, args.fit)
    print('[contrib] {} on {}: fit on {} ({} interactions), selection on {} interactions'.format(
        args.dataset, device, args.fit, fit['pos'].nnz, fit['sel_gt'].nnz), flush=True)

    # stage 1: out-of-fold residuals (shared by --fit train / valid: stored next to the train-fit results)
    res_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(res_dir, exist_ok=True)
    res_path = os.path.join(res_dir, 'residuals_{}_k{}.npz'.format(args.pred_head, args.folds))
    zY, zT, zTO, zUO = zs(Y_id), zs(Y_t), zs(TO), zs(UO)
    if not os.path.isfile(res_path):
        print('[contrib] stage 1: out-of-fold predictor {} with {} folds'.format(args.pred_head, args.folds), flush=True)
        R_B3, d_b3 = oof_residual(zY, zT, args.folds, args.pred_head, 2026, device)
        R_TO, d_to = oof_residual(np.concatenate([zY, zUO], 1), zTO, args.folds, args.pred_head, 2026, device)
        np.savez_compressed(res_path, R_B3=R_B3, R_TO=R_TO)
        save_json({'R_B3': d_b3, 'R_TO': d_to, 'pred_head': args.pred_head, 'folds': args.folds},
                  res_path[:-4] + '.json')
    R = np.load(res_path)
    diag = json.load(open(res_path[:-4] + '.json'))
    for k in ('R_B3', 'R_TO'):
        print('[contrib] {}: out-of-fold R2 of P {:.4f}, residual energy {:.4f}'.format(
            k, diag[k]['r2_oof'], diag[k]['res_energy']), flush=True)
    content = {'TO': zTO, 'R_B3': R['R_B3'], 'R_TO': R['R_TO']}

    # A0: frozen SF, a selected on the selection interactions
    base = SFBase(U_id, Y_id, U_t, Y_t, device)
    a0_path = os.path.join(out, 'A0.json')
    if not os.path.isfile(a0_path):
        curve = []
        for a in SF_GRID:
            va = evaluate_split(lambda ub, a=a: base.full(torch.from_numpy(ub).long().to(device), a).cpu().numpy(),
                                fit, 'valid')[0]
            curve.append({'a': a, 'valid': va})
        best = max(curve, key=lambda c: c['valid']['recall@20'])
        base.a = best['a']
        te, r, n = evaluate_split(scorer(base, None, device), fit, 'test')
        np.savez_compressed(os.path.join(out, 'A0_users.npz'), r=r, n=n)
        save_json({'config': 'A0', 'a': best['a'], 'valid': best['valid'], 'test': te, 'curve': curve}, a0_path)
    A0 = json.load(open(a0_path))
    base.a = A0['a']
    print('[A0] SF a = {}: valid R@20 {:.4f} | test R@20 {:.4f} N@20 {:.4f}'.format(
        A0['a'], A0['valid']['recall@20'], A0['test']['recall@20'], A0['test']['ndcg@20']), flush=True)

    Uz = torch.from_numpy(zs(U_id)).to(device)

    def run(cname, cfg, seed):
        f = os.path.join(out, '{}_{}_s{}.json'.format(cname, tag(cfg), seed))
        if os.path.isfile(f):
            return json.load(open(f))
        Xi = torch.from_numpy(np.concatenate([zY, content[CONFIGS[cname]]], 1)).to(device)
        print('[{}] {} seed {}'.format(cname, cfg, seed), flush=True)
        t0 = time.time()
        r, ru, nu = train(args, fit, base, Uz, Xi, cfg, seed, device)
        r.update({'config': cname, 'content': CONFIGS[cname], 'seconds': time.time() - t0})
        np.savez_compressed(f[:-5] + '_users.npz', r=ru, n=nu)
        save_json(r, f)
        print('[{}] {} seed {}: best epoch {} valid R@20 {:.4f} | test R@20 {:.4f} N@20 {:.4f}'.format(
            cname, cfg, seed, r['best_epoch'], r['valid']['recall@20'], r['test']['recall@20'],
            r['test']['ndcg@20']), flush=True)
        return r

    for cname in args.configs.split():
        grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
        best = max((run(cname, c, seeds[0]) for c in grid), key=lambda r: r['valid']['recall@20'])
        for s in seeds[1:]:
            run(cname, best['cfg'], s)
    write_summary(out, args)


def write_summary(out, args):
    A0 = json.load(open(os.path.join(out, 'A0.json')))
    seeds = [int(s) for s in args.seeds.split()]
    rows, users = {'A0': {'test': A0['test'], 'valid': A0['valid'], 'cfg': {'a': A0['a']}, 'n': 1}}, {}
    users['A0'] = np.load(os.path.join(out, 'A0_users.npz'))
    for cname in CONFIGS:
        runs = [json.load(open(f)) for f in glob.glob(os.path.join(out, '{}_*_s{}.json'.format(cname, seeds[0])))]
        if not runs:
            continue
        best = max(runs, key=lambda r: r['valid']['recall@20'])
        same = [json.load(open(f)) for s in seeds for f in
                glob.glob(os.path.join(out, '{}_{}_s{}.json'.format(cname, tag(best['cfg']), s)))]
        rows[cname] = {'cfg': best['cfg'], 'valid': best['valid'], 'best_epoch': best['best_epoch'], 'n': len(same),
                       'test': {k: float(np.mean([r['test'][k] for r in same])) for k in ('recall@20', 'ndcg@20')},
                       'item_abs': best.get('item_side_abs_mean')}
        users[cname] = np.load(os.path.join(out, '{}_{}_s{}_users.npz'.format(cname, tag(best['cfg']), seeds[0])))
    b0 = rows['A0']['test']['recall@20']
    lines = ['# Conditional contribution: frozen SF + correction ({}, fit on {})\n'.format(
        os.path.basename(out), args.fit)]
    res_json = os.path.join(args.out_dir, os.path.basename(out), 'residuals_{}_k{}.json'.format(args.pred_head, args.folds))
    if os.path.isfile(res_json):
        d = json.load(open(res_json))
        lines.append('Stage 1 predictor {} ({} folds, out-of-fold): R_B3 = z(Y_t) - P(z(Y_id)): R2 {:.4f}, residual '
                     'energy {:.4f}; R_TO = z(TO) - P(z(Y_id), z(UO)): R2 {:.4f}, residual energy {:.4f}.\n'.format(
                         d['pred_head'], d['folds'], d['R_B3']['r2_oof'], d['R_B3']['res_energy'],
                         d['R_TO']['r2_oof'], d['R_TO']['res_energy']))
    prev = os.path.join(ROOT, 'results', 'consolidate', os.path.basename(out), 'SF.json')
    if os.path.isfile(prev):
        p = json.load(open(prev))
        lines.append('Reference: SF of the consolidation runs (teacher_text taps): a = {}, test R@20 {:.4f}, N@20 '
                     '{:.4f}.\n'.format(p['cfg']['a'], p['test']['recall@20'], p['test']['ndcg@20']))
    lines += ['| config | setting | best epoch | valid R@20 | test R@20 | test N@20 | vs A0 | seeds |',
              '|---|---|---|---|---|---|---|---|']
    for k in ('A0', 'A1', 'A2', 'A3'):
        if k in rows:
            r = rows[k]
            lines.append('| {} {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:+.1f}% | {} |'.format(
                k, LABEL[k], tag(r['cfg']), r.get('best_epoch', '-'), r['valid']['recall@20'],
                r['test']['recall@20'], r['test']['ndcg@20'], 100 * (r['test']['recall@20'] / b0 - 1), r['n']))
    lines += ['\nbest epoch 0 = the correction stayed at zero (SF selected).\n',
              '## Paired bootstrap over test users (seed {}; first - second)\n'.format(seeds[0]),
              '| comparison | metric | diff | 95% CI | p (one-sided, first > second) |', '|---|---|---|---|---|']
    for a, b in PAIRS:
        if a in users and b in users:
            for m, key in (('recall@20', 'r'), ('ndcg@20', 'n')):
                bt = paired_bootstrap(users[a][key] - users[b][key], n_boot=args.n_boot)
                lines.append('| {} vs {} | {} | {:+.4f} | [{:+.4f}, {:+.4f}] | {:.4f} |'.format(
                    a, b, m, bt['mean'], bt['ci_lo'], bt['ci_hi'], bt['p_one_sided']))
    text = '\n'.join(lines) + '\n'
    open(os.path.join(out, 'summary.md'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
