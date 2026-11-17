# coding: utf-8
"""
Text -> ID consolidation on FROZEN teachers (B0-B3).

Teachers (LightGCNC runs with --save_taps=True; propagated embeddings at the best-valid epoch):
    ID teacher    U_id, Y_id   (item_input=id)        s_ID(u, i) = <U_id[u], Y_id[i]>
    Text teacher  U_t,  Y_t    (item_input=text)
Both are frozen. Only the correction Delta_T is trained, with BPR on the train interactions:

    s(u, i) = s_ID(u, i) + Delta_T(u, i),      Delta_T(u, i) = <g_u, Z_i>
    S_T     = E_T(Y_t[i])                      text latent (2-layer MLP)
    S_hat   = P(Y_id[i])                       ID-context prediction of the text latent (2-layer MLP)
    B2      Z_i = C([S_T ; Y_id[i]])                       conditional correction, no JEPA (lam_J = 0)
    B3      Z_i = C([S_T - sg(S_hat) ; Y_id[i]])           correction on the text part the ID context
                                                           does not predict;  L_J = ||S_hat - sg(S_T)||^2
    g_u     = G([U_id[u] ; U_t[u]])                        user side, no free user table
    L = BPR(s) + lam_J L_J + lam_D E[Delta_T^2]

B2 and B3 have the same modules and parameter count (P exists in B2 but receives no gradient). The last
layer of G is zero-initialised, so Delta_T = 0 at the start: the model starts exactly at B0, and epoch 0
(= B0) is part of model selection, so the selected model is never worse than B0 on VALID (no such
guarantee on test). B0 is the ID teacher itself, evaluated by the same code.
Evaluation: full ranking, Recall/NDCG@10/20 (tools/probe_knn.evaluate); valid masks train items, test
masks train + valid items. The checkpoint is chosen on valid Recall@20.

    python tools/consolidate.py -d dbbook --id_taps taps/LightGCNC-dbbook-teacher_id \
        --text_taps taps/LightGCNC-dbbook-teacher_text --mode B3 --lam_J 1 --lam_D 0.01 --seed 0
    python tools/consolidate.py -d dbbook ... --grid     # grid on valid (seed 0) for B2 and B3, then seeds 0,1,2
Output: results/consolidate/<ds>/<mode>_<cfg>_s<seed>.json and summary.md
"""
import argparse
import copy
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
from probe_rec import load_data        # noqa: E402
from probe_knn import evaluate          # noqa: E402

GRID = {'B2': {'lam_D': [0.0, 1e-3, 1e-2, 1e-1], 'lr': [1e-3, 1e-4]},
        'B3': {'lam_D': [0.0, 1e-3, 1e-2, 1e-1], 'lr': [1e-3, 1e-4], 'lam_J': [0.1, 1.0]}}


def mlp(d_in, d_hid, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hid), nn.LeakyReLU(), nn.Linear(d_hid, d_out))


class Consolidator(nn.Module):
    def __init__(self, U_id, Y_id, U_t, Y_t, mode, d=64, hidden=256):
        super(Consolidator, self).__init__()
        self.mode = mode
        for name, x in (('U_id', U_id), ('Y_id', Y_id), ('U_t', U_t), ('Y_t', Y_t)):
            self.register_buffer(name, x)                       # frozen teachers
        self.E_T = mlp(Y_t.shape[1], hidden, d)
        self.P = mlp(Y_id.shape[1], hidden, d)
        self.C = mlp(d + Y_id.shape[1], hidden, d)
        self.G = mlp(U_id.shape[1] + U_t.shape[1], hidden, d)
        nn.init.zeros_(self.G[-1].weight)                       # Delta_T = 0 at start (model = B0)
        nn.init.zeros_(self.G[-1].bias)

    def item_side(self, items=None):
        y_t = self.Y_t if items is None else self.Y_t[items]
        y_id = self.Y_id if items is None else self.Y_id[items]
        s_t = self.E_T(y_t)
        s_hat = self.P(y_id)
        x = s_t - s_hat.detach() if self.mode == 'B3' else s_t
        return self.C(torch.cat([x, y_id], 1)), s_t, s_hat

    def user_side(self, users=None):
        u_id = self.U_id if users is None else self.U_id[users]
        u_t = self.U_t if users is None else self.U_t[users]
        return self.G(torch.cat([u_id, u_t], 1))

    def scores(self, users, items):
        """Pairwise s_ID + Delta for aligned (users, items) vectors."""
        g = self.user_side(users)
        z, _, _ = self.item_side(items)
        delta = (g * z).sum(1)
        return (self.U_id[users] * self.Y_id[items]).sum(1) + delta, delta


def load_taps(path):
    U = np.load(os.path.join(path, 'T3_user.npy')).astype(np.float32)
    Y = np.load(os.path.join(path, 'T3.npy')).astype(np.float32)
    meta = json.load(open(os.path.join(path, 'meta.json'))) if os.path.isfile(os.path.join(path, 'meta.json')) else {}
    return U, Y, meta


def sample_triples(R_tr, n, rng):
    """(user, pos, neg) with neg uniform among items the user has no train interaction with."""
    coo = R_tr.tocoo()
    idx = rng.integers(0, coo.nnz, n)
    u, p = coo.row[idx], coo.col[idx]
    neg = rng.integers(0, R_tr.shape[1], n)
    bad = np.asarray(R_tr[u, neg]).ravel() > 0
    while bad.any():
        neg[bad] = rng.integers(0, R_tr.shape[1], bad.sum())
        bad = np.asarray(R_tr[u, neg]).ravel() > 0
    return u, p, neg


@torch.no_grad()
def rank_eval(model, R_tr, R_gt, R_mask, device, delta=True):
    model.eval()
    z = model.item_side()[0] if delta else None

    def score(ub):
        t = torch.from_numpy(ub).long().to(device)
        s = model.U_id[t] @ model.Y_id.T
        if delta:
            s = s + model.user_side(t) @ z.T
        return s.cpu().numpy()
    out = {}
    for k in (10, 20):
        r, n = evaluate(score, R_tr, R_gt, R_mask, k=k)
        out['recall@{}'.format(k)], out['ndcg@{}'.format(k)] = float(r.mean()), float(n.mean())
    return out


def train(args, mats, teachers, mode, cfg, seed, device, log=print):
    R_tr, R_va, R_te = mats
    R_trva = (R_tr + R_va).tocsr()
    R_trva.data[:] = 1.0
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Consolidator(*[torch.from_numpy(x).to(device) for x in teachers], mode=mode, d=args.d,
                         hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    best = rank_eval(model, R_tr, R_va, R_tr, device)          # epoch 0: Delta = 0, i.e. B0
    best_ep, best_state, bad, hist = 0, copy.deepcopy(model.state_dict()), 0, []
    log('  epoch 0 (= B0) valid R@20 {:.4f}'.format(best['recall@20']))
    n_per_epoch = R_tr.nnz
    for ep in range(1, args.epochs + 1):
        model.train()
        u, p, n = sample_triples(R_tr, n_per_epoch, rng)
        tot = {'bpr': 0.0, 'J': 0.0, 'D': 0.0}
        for s in range(0, n_per_epoch, args.batch):
            ub = torch.from_numpy(u[s:s + args.batch]).long().to(device)
            pb = torch.from_numpy(p[s:s + args.batch]).long().to(device)
            nb = torch.from_numpy(n[s:s + args.batch]).long().to(device)
            sp, dp = model.scores(ub, pb)
            sn, dn = model.scores(ub, nb)
            bpr = -F.logsigmoid(sp - sn).mean()
            loss = bpr + cfg['lam_D'] * 0.5 * (dp.pow(2).mean() + dn.pow(2).mean())
            if mode == 'B3' and cfg['lam_J'] > 0:
                items = torch.cat([pb, nb]).unique()
                _, s_t, s_hat = model.item_side(items)
                lj = F.mse_loss(s_hat, s_t.detach())
                loss = loss + cfg['lam_J'] * lj
                tot['J'] += float(lj)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot['bpr'] += float(bpr)
            tot['D'] += float(0.5 * (dp.pow(2).mean() + dn.pow(2).mean()))
        if ep % args.eval_every:
            continue
        va = rank_eval(model, R_tr, R_va, R_tr, device)
        hist.append({'epoch': ep, 'valid_recall@20': va['recall@20'], **tot})
        log('  epoch {} valid R@20 {:.4f} | bpr {:.4f} E[D^2] {:.4f} L_J {:.4f}'.format(
            ep, va['recall@20'], tot['bpr'], tot['D'], tot['J']))
        if va['recall@20'] > best['recall@20'] + 1e-6:
            best, best_ep, best_state, bad = va, ep, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    test = rank_eval(model, R_tr, R_te, R_trva, device)
    return {'mode': mode, 'cfg': cfg, 'seed': seed, 'best_epoch': best_ep, 'valid': best, 'test': test,
            'history': hist}


def cfg_tag(c):
    return '_'.join('{}{:g}'.format(k, c[k]) for k in sorted(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True)
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--id_taps', required=True)
    ap.add_argument('--text_taps', required=True)
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'results', 'consolidate'))
    ap.add_argument('--mode', choices=['B2', 'B3'], default='B3')
    ap.add_argument('--grid', action='store_true', help='grid (seed 0) for B2 and B3, then seeds 0 1 2 of the best')
    ap.add_argument('--lam_J', type=float, default=1.0)
    ap.add_argument('--lam_D', type=float, default=1e-2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--eval_every', type=int, default=2)
    ap.add_argument('--patience', type=int, default=10, help='in evaluations')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    out = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out, exist_ok=True)

    U_id, Y_id, m_id = load_taps(args.id_taps)
    U_t, Y_t, m_t = load_taps(args.text_taps)
    mats, n_users, n_items = load_data(args.data_dir, args.dataset, min_items=Y_id.shape[0])
    assert U_id.shape[0] == U_t.shape[0] == n_users and Y_id.shape[0] == Y_t.shape[0] == n_items, \
        'teacher shapes {} {} {} {} vs users {} items {}'.format(U_id.shape, Y_id.shape, U_t.shape, Y_t.shape,
                                                                 n_users, n_items)
    teachers = (U_id, Y_id, U_t, Y_t)
    print('[consolidate] {} on {} | ID teacher test R@20 {} | text teacher test R@20 {}'.format(
        args.dataset, device, m_id.get('test', {}).get('recall@20'), m_t.get('test', {}).get('recall@20')), flush=True)

    # B0: the frozen ID teacher under this evaluator (valid + test)
    b0_path = os.path.join(out, 'B0.json')
    if not os.path.isfile(b0_path):
        tmp = Consolidator(*[torch.from_numpy(x).to(device) for x in teachers], mode='B2').to(device)
        R_tr, R_va, R_te = mats
        R_trva = (R_tr + R_va).tocsr()
        R_trva.data[:] = 1.0
        b0 = {'mode': 'B0', 'valid': rank_eval(tmp, R_tr, R_va, R_tr, device, delta=False),
              'test': rank_eval(tmp, R_tr, R_te, R_trva, device, delta=False)}
        json.dump(b0, open(b0_path, 'w'), indent=1)
    b0 = json.load(open(b0_path))
    print('[B0] valid R@20 {:.4f} | test R@20 {:.4f} N@20 {:.4f}'.format(
        b0['valid']['recall@20'], b0['test']['recall@20'], b0['test']['ndcg@20']), flush=True)

    def run(mode, cfg, seed):
        f = os.path.join(out, '{}_{}_s{}.json'.format(mode, cfg_tag(cfg), seed))
        if os.path.isfile(f):
            return json.load(open(f))
        print('[{}] {} seed {}'.format(mode, cfg, seed), flush=True)
        t0 = time.time()
        r = train(args, mats, teachers, mode, cfg, seed, device)
        r['seconds'] = time.time() - t0
        json.dump(r, open(f, 'w'), indent=1)
        print('[{}] {} seed {}: best epoch {} valid R@20 {:.4f} | test R@20 {:.4f} N@20 {:.4f}'.format(
            mode, cfg, seed, r['best_epoch'], r['valid']['recall@20'], r['test']['recall@20'],
            r['test']['ndcg@20']), flush=True)
        return r

    if not args.grid:
        cfg = {'lr': args.lr, 'lam_D': args.lam_D}
        if args.mode == 'B3':
            cfg['lam_J'] = args.lam_J
        run(args.mode, cfg, args.seed)
        return

    seeds = [int(s) for s in args.seeds.split(',')]
    rows = []
    for mode in ('B2', 'B3'):
        keys = sorted(GRID[mode])
        res = [run(mode, dict(zip(keys, v)), seeds[0]) for v in itertools.product(*[GRID[mode][k] for k in keys])]
        top = max(res, key=lambda r: r['valid']['recall@20'])
        finals = [run(mode, top['cfg'], s) for s in seeds]
        rows.append((mode, top['cfg'], finals))
    lines = ['# Text -> ID consolidation: {}\n'.format(args.dataset),
             'B0 = frozen ID teacher. Test metrics: mean +- std over correction seeds {} (teachers fixed).\n'.format(seeds),
             '| config | setting | best epochs | valid R@20 | test R@20 | test N@20 | vs B0 (R@20) |',
             '|---|---|---|---|---|---|---|',
             '| B0 ID only | - | - | {:.4f} | {:.4f} | {:.4f} | - |'.format(
                 b0['valid']['recall@20'], b0['test']['recall@20'], b0['test']['ndcg@20'])]
    for mode, cfg, finals in rows:
        tr = np.array([r['test']['recall@20'] for r in finals])
        tn = np.array([r['test']['ndcg@20'] for r in finals])
        va = np.mean([r['valid']['recall@20'] for r in finals])
        lines.append('| {} | {} | {} | {:.4f} | {:.4f} ± {:.4f} | {:.4f} ± {:.4f} | {:+.2f}% |'.format(
            mode, cfg_tag(cfg), ','.join(str(r['best_epoch']) for r in finals), va, tr.mean(), tr.std(),
            tn.mean(), tn.std(), 100 * (tr.mean() / b0['test']['recall@20'] - 1)))
    lines.append('\nbest epoch 0 means the selected model is B0 itself (no useful correction found on valid).')
    text = '\n'.join(lines) + '\n'
    open(os.path.join(out, 'summary.md'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
