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
    B2d,B3d difference form: Z_i = C([x ; Y_id[i]]) - C([0 ; Y_id[i]])  (x as in B2 / B3), so the item side
            of the correction is 0 whenever the text input is 0: no ID-only residual inside C
    IDcorr  control without any text: the same modules on ID inputs only,
            S = E_T(Y_id[i]), Z_i = C([S ; Y_id[i]]), g_u = G([U_id[u] ; U_id[u]])  (same parameter count)
    g_u     = G([U_id[u] ; U_t[u]])                        user side, no free user table (IDcorr: U_id twice)
Fusion baselines on the same frozen teachers (no s_ID shortcut):
    CAT     concat -> MLP: e_u = F_u([U_id; U_t]), e_i = F_i([Y_id; Y_t]), s = <e_u, e_i>
    GATE    learned weighted fusion: e = g * A_id(id) + (1 - g) * A_t(text), g = sigmoid(MLP([id; text]))
            per dimension, separately for users and items; A_* linear initialised to the identity
    SF      score-level fusion: s = (1 - a) z(s_ID) + a z(s_T), a in {0, 0.05, ..., 1} chosen on valid
            (z = standardised scores); no training
    CAT and GATE use the grid of B2 (lam_D = L2 weight on the fused embeddings) x lr: same budget.
Preference-conditioned feature gating of the text channel (frozen teachers, only the gate is learned):
    FG      s = s_ID + <U_t[u], A_i * Y_t[i]>,  A_i = sigmoid(f([Y_id[i]; Y_t[i]])) in [0,1]^d per item
    FGu     A_ui = A_i * B_u,  B_u = sigmoid(h([U_id[u]; U_t[u]])): user conditioning in a separable form
            (full ranking stays one matrix product)
    f, h: 2-layer MLPs; the last bias of f is -4 (A ~ 0.02, the model starts close to B0), of h +4 (B ~ 1).
    Penalty lam_D E[Delta^2] with Delta = the gated text score; grid of B2. Gate statistics are saved.
    L = BPR(s) + lam_J L_J + lam_D E[Delta_T^2]

B2 and B3 have the same modules and parameter count (P exists in B2 but receives no gradient). The last
layer of G is zero-initialised, so Delta_T = 0 at the start: the model starts exactly at B0, and epoch 0
(= B0) is part of model selection, so the selected model is never worse than B0 on VALID (no such
guarantee on test). B0 is the ID teacher itself, evaluated by the same code.
Evaluation: full ranking, Recall/NDCG@10/20 (tools/probe_knn.evaluate); valid masks train items, test
masks train + valid items. The checkpoint is chosen on valid Recall@20.

    python tools/consolidate.py -d dbbook --id_taps taps/LightGCNC-dbbook-teacher_id \
        --text_taps taps/LightGCNC-dbbook-teacher_text --mode B3 --lam_J 1 --lam_D 0.01 --seed 0
    python tools/consolidate.py -d dbbook ... --grid --seeds 999   # grid on valid for B2 and B3 (seed 999)
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
GRID['B2d'], GRID['B3d'], GRID['IDcorr'] = GRID['B2'], GRID['B3'], GRID['B2']
GRID['CAT'], GRID['GATE'] = GRID['B2'], GRID['B2']      # lam_D = L2 weight on the fused embeddings here
GRID['FG'], GRID['FGu'] = GRID['B2'], GRID['B2']
MODES = ('B2', 'B3', 'B2d', 'B3d', 'IDcorr', 'CAT', 'GATE', 'FG', 'FGu')
CORR_MODES = ('B2', 'B3', 'B2d', 'B3d', 'IDcorr')
GATE_MODES = ('FG', 'FGu')
SF_GRID = [round(0.05 * k, 2) for k in range(21)]          # score fusion weight of the text teacher


def mlp(d_in, d_hid, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hid), nn.LeakyReLU(), nn.Linear(d_hid, d_out))


def eye_linear(d):
    lin = nn.Linear(d, d)
    nn.init.eye_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


class Consolidator(nn.Module):
    """Correction modes (B2, B3, B2d, B3d, IDcorr): s = s_ID + Delta, see the module docstring.
    Fusion baselines on the same frozen teachers (no s_ID shortcut, no Delta = 0 start):
      CAT   e_u = F_u([U_id; U_t]), e_i = F_i([Y_id; Y_t]) (2-layer MLPs), s = <e_u, e_i>
      GATE  e_i = g_i * A_id(Y_id) + (1 - g_i) * A_t(Y_t), g_i = sigmoid(MLP([Y_id; Y_t])) per dimension;
            same for users with their own gate; A_* linear, initialised to the identity; s = <e_u, e_i>
    pair_scores returns (score, penalty): penalty = Delta^2 (corrections) or ||e_u||^2 + ||e_i||^2 (CAT, GATE)."""

    def __init__(self, U_id, Y_id, U_t, Y_t, mode, d=64, hidden=256):
        super(Consolidator, self).__init__()
        self.mode = mode
        for name, x in (('U_id', U_id), ('Y_id', Y_id), ('U_t', U_t), ('Y_t', Y_t)):
            self.register_buffer(name, x)                       # frozen teachers
        du, di = U_id.shape[1] + U_t.shape[1], Y_id.shape[1] + Y_t.shape[1]
        if mode in CORR_MODES:
            self.E_T = mlp(Y_t.shape[1], hidden, d)
            self.P = mlp(Y_id.shape[1], hidden, d)
            self.C = mlp(d + Y_id.shape[1], hidden, d)
            self.G = mlp(du, hidden, d)
            nn.init.zeros_(self.G[-1].weight)                   # Delta_T = 0 at start (model = B0)
            nn.init.zeros_(self.G[-1].bias)
        elif mode == 'CAT':
            self.F_u = mlp(du, hidden, d)
            self.F_i = mlp(di, hidden, d)
        elif mode == 'GATE':
            assert U_id.shape[1] == U_t.shape[1] == Y_id.shape[1] == Y_t.shape[1], 'GATE needs equal teacher dims'
            k = Y_id.shape[1]
            self.A_uid, self.A_ut, self.A_iid, self.A_it = eye_linear(k), eye_linear(k), eye_linear(k), eye_linear(k)
            self.gate_u = mlp(du, hidden, k)
            self.gate_i = mlp(di, hidden, k)
        elif mode in GATE_MODES:
            assert Y_id.shape[1] == Y_t.shape[1], 'FG needs equal teacher dims'
            self.f_gate = mlp(di, hidden, Y_t.shape[1])
            nn.init.zeros_(self.f_gate[-1].weight)
            nn.init.constant_(self.f_gate[-1].bias, -4.0)          # A ~ 0.02 at start
            if mode == 'FGu':
                self.h_gate = mlp(du, hidden, U_t.shape[1])
                nn.init.zeros_(self.h_gate[-1].weight)
                nn.init.constant_(self.h_gate[-1].bias, 4.0)        # B ~ 0.98 at start
        else:
            raise ValueError(mode)

    # ---------------------------------------------------------------- feature gating
    def item_gate(self, items=None):
        y_id = self.Y_id if items is None else self.Y_id[items]
        y_t = self.Y_t if items is None else self.Y_t[items]
        return torch.sigmoid(self.f_gate(torch.cat([y_id, y_t], 1)))

    def user_gate(self, users=None):
        u_id = self.U_id if users is None else self.U_id[users]
        u_t = self.U_t if users is None else self.U_t[users]
        return torch.sigmoid(self.h_gate(torch.cat([u_id, u_t], 1)))

    def gated_user(self, users):
        u = self.U_t[users]
        return u * self.user_gate(users) if self.mode == 'FGu' else u

    # ---------------------------------------------------------------- corrections
    def item_side(self, items=None):
        y_t = self.Y_t if items is None else self.Y_t[items]
        y_id = self.Y_id if items is None else self.Y_id[items]
        s_t = self.E_T(y_id if self.mode == 'IDcorr' else y_t)
        s_hat = self.P(y_id)
        x = s_t - s_hat.detach() if self.mode in ('B3', 'B3d') else s_t
        z = self.C(torch.cat([x, y_id], 1))
        if self.mode in ('B2d', 'B3d'):
            z = z - self.C(torch.cat([torch.zeros_like(x), y_id], 1))
        return z, s_t, s_hat

    def user_side(self, users=None):
        u_id = self.U_id if users is None else self.U_id[users]
        if self.mode == 'IDcorr':
            return self.G(torch.cat([u_id, u_id], 1))
        u_t = self.U_t if users is None else self.U_t[users]
        return self.G(torch.cat([u_id, u_t], 1))

    # ---------------------------------------------------------------- fusion baselines
    def fused_user(self, users=None):
        u_id = self.U_id if users is None else self.U_id[users]
        u_t = self.U_t if users is None else self.U_t[users]
        x = torch.cat([u_id, u_t], 1)
        if self.mode == 'CAT':
            return self.F_u(x)
        g = torch.sigmoid(self.gate_u(x))
        return g * self.A_uid(u_id) + (1 - g) * self.A_ut(u_t)

    def fused_item(self, items=None):
        y_id = self.Y_id if items is None else self.Y_id[items]
        y_t = self.Y_t if items is None else self.Y_t[items]
        x = torch.cat([y_id, y_t], 1)
        if self.mode == 'CAT':
            return self.F_i(x)
        g = torch.sigmoid(self.gate_i(x))
        return g * self.A_iid(y_id) + (1 - g) * self.A_it(y_t)

    # ---------------------------------------------------------------- scores
    def pair_scores(self, users, items):
        if self.mode in GATE_MODES:
            delta = (self.gated_user(users) * self.item_gate(items) * self.Y_t[items]).sum(1)
            return (self.U_id[users] * self.Y_id[items]).sum(1) + delta, delta.pow(2)
        if self.mode in CORR_MODES:
            g = self.user_side(users)
            z, _, _ = self.item_side(items)
            delta = (g * z).sum(1)
            return (self.U_id[users] * self.Y_id[items]).sum(1) + delta, delta.pow(2)
        eu, ei = self.fused_user(users), self.fused_item(items)
        return (eu * ei).sum(1), eu.pow(2).sum(1) + ei.pow(2).sum(1)

    def eval_cache(self):
        if self.mode in GATE_MODES:
            return self.item_gate() * self.Y_t
        if self.mode in CORR_MODES:
            return self.item_side()[0]
        return self.fused_item()

    def full_scores(self, users, cache):
        if self.mode in GATE_MODES:
            return self.U_id[users] @ self.Y_id.T + self.gated_user(users) @ cache.T
        if self.mode in CORR_MODES:
            return self.U_id[users] @ self.Y_id.T + self.user_side(users) @ cache.T
        return self.fused_user(users) @ cache.T


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


def ranking(score_fn, R_tr, R_gt, R_mask):
    out = {}
    for k in (10, 20):
        r, n = evaluate(score_fn, R_tr, R_gt, R_mask, k=k)
        out['recall@{}'.format(k)], out['ndcg@{}'.format(k)] = float(r.mean()), float(n.mean())
    return out


@torch.no_grad()
def rank_eval(model, R_tr, R_gt, R_mask, device, delta=True):
    """delta=False: the frozen ID teacher alone (B0)."""
    model.eval()
    cache = model.eval_cache() if delta else None

    def score(ub):
        t = torch.from_numpy(ub).long().to(device)
        s = model.full_scores(t, cache) if delta else model.U_id[t] @ model.Y_id.T
        return s.cpu().numpy()
    return ranking(score, R_tr, R_gt, R_mask)


@torch.no_grad()
def score_fusion(mats, teachers, device, log=print):
    """Baseline SF: s = (1 - a) s_ID + a s_T with the two frozen teachers; a in SF_GRID chosen on valid R@20.
    Scores are z-normalised per teacher (mean / std over a sample of user-item pairs) so that a is comparable."""
    R_tr, R_va, R_te = mats
    R_trva = (R_tr + R_va).tocsr()
    R_trva.data[:] = 1.0
    U_id, Y_id, U_t, Y_t = [torch.from_numpy(x).to(device) for x in teachers]
    g = torch.Generator(device='cpu').manual_seed(0)
    su = torch.randint(0, U_id.shape[0], (20000,), generator=g).to(device)
    si = torch.randint(0, Y_id.shape[0], (20000,), generator=g).to(device)
    st = {}
    for name, U, Y in (('id', U_id, Y_id), ('t', U_t, Y_t)):
        v = (U[su] * Y[si]).sum(1)
        st[name] = (float(v.mean()), float(v.std()))

    def make(a):
        def score(ub):
            t = torch.from_numpy(ub).long().to(device)
            s_id = (U_id[t] @ Y_id.T - st['id'][0]) / st['id'][1]
            s_t = (U_t[t] @ Y_t.T - st['t'][0]) / st['t'][1]
            return ((1 - a) * s_id + a * s_t).cpu().numpy()
        return score
    curve = []
    for a in SF_GRID:
        va = ranking(make(a), R_tr, R_va, R_tr)
        curve.append({'a': a, 'valid': va})
        log('  SF a={:.2f} valid R@20 {:.4f}'.format(a, va['recall@20']))
    best = max(curve, key=lambda c: c['valid']['recall@20'])
    test = ranking(make(best['a']), R_tr, R_te, R_trva)
    return {'mode': 'SF', 'cfg': {'a': best['a']}, 'seed': 0, 'best_epoch': '-', 'valid': best['valid'],
            'test': test, 'curve': curve, 'norm': st}


@torch.no_grad()
def gate_stats(model, R_tr):
    """What the learned gate does: A over items (and B over users for FGu), per channel and by popularity."""
    model.eval()
    A = model.item_gate().cpu().numpy()
    deg = np.asarray(R_tr.sum(0)).ravel()
    q = np.quantile(deg, [1 / 3, 2 / 3])
    groups = {'tail': deg <= q[0], 'mid': (deg > q[0]) & (deg <= q[1]), 'head': deg > q[1]}
    out = {'gate_item': {'mean': float(A.mean()), 'std_over_items': float(A.mean(1).std()),
                         'frac_channels_open(>0.5)': float((A > 0.5).mean()),
                         'per_channel_mean': [float(x) for x in A.mean(0)],
                         'mean_by_popularity': {k: float(A[m].mean()) for k, m in groups.items()},
                         'quantiles_item_mean': [float(x) for x in np.quantile(A.mean(1), [0.05, 0.25, 0.5, 0.75, 0.95])]}}
    if model.mode == 'FGu':
        B = model.user_gate().cpu().numpy()
        out['gate_user'] = {'mean': float(B.mean()), 'std_over_users': float(B.mean(1).std()),
                            'quantiles_user_mean': [float(x) for x in np.quantile(B.mean(1), [0.05, 0.25, 0.5, 0.75, 0.95])]}
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
    best = rank_eval(model, R_tr, R_va, R_tr, device)          # epoch 0 (corrections: Delta = 0, i.e. B0)
    best_ep, best_state, bad, hist = 0, copy.deepcopy(model.state_dict()), 0, []
    log('  epoch 0 valid R@20 {:.4f}'.format(best['recall@20']))
    n_per_epoch = R_tr.nnz
    for ep in range(1, args.epochs + 1):
        model.train()
        u, p, n = sample_triples(R_tr, n_per_epoch, rng)
        tot = {'bpr': 0.0, 'J': 0.0, 'D': 0.0}
        for s in range(0, n_per_epoch, args.batch):
            ub = torch.from_numpy(u[s:s + args.batch]).long().to(device)
            pb = torch.from_numpy(p[s:s + args.batch]).long().to(device)
            nb = torch.from_numpy(n[s:s + args.batch]).long().to(device)
            sp, dp = model.pair_scores(ub, pb)
            sn, dn = model.pair_scores(ub, nb)
            bpr = -F.logsigmoid(sp - sn).mean()
            pen = 0.5 * (dp.mean() + dn.mean())
            loss = bpr + cfg['lam_D'] * pen
            if mode in ('B3', 'B3d') and cfg['lam_J'] > 0:
                items = torch.cat([pb, nb]).unique()
                _, s_t, s_hat = model.item_side(items)
                lj = F.mse_loss(s_hat, s_t.detach())
                loss = loss + cfg['lam_J'] * lj
                tot['J'] += float(lj)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot['bpr'] += float(bpr)
            tot['D'] += float(pen)
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
    extra = gate_stats(model, R_tr) if mode in GATE_MODES else {}
    return {'mode': mode, 'cfg': cfg, 'seed': seed, 'best_epoch': best_ep, 'valid': best, 'test': test,
            'history': hist, **extra}


def cfg_tag(c):
    return '_'.join('{}{:g}'.format(k, c[k]) for k in sorted(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True)
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--id_taps', required=True)
    ap.add_argument('--text_taps', required=True)
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'results', 'consolidate'))
    ap.add_argument('--mode', choices=list(MODES) + ['SF'], default='B3')
    ap.add_argument('--grid', action='store_true', help='grid on valid for B2 and B3 with the first of --seeds, then the other seeds for the best setting')
    ap.add_argument('--modes', default='B2 B3', help='modes of the grid, e.g. "B2", "CAT GATE SF" (separate jobs)')
    ap.add_argument('--fix', default='', help='restrict the grid, e.g. "lam_J=0.1" or "lam_J=1 lr=0.001"')
    ap.add_argument('--no_finals', action='store_true', help='grid only (seed = first of --seeds); no other seeds')
    ap.add_argument('--summary', action='store_true', help='no training: table from the finished runs in out_dir')
    ap.add_argument('--lam_J', type=float, default=1.0)
    ap.add_argument('--lam_D', type=float, default=1e-2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=999)
    ap.add_argument('--seeds', default='999', help='e.g. "999" or "999 1 2"; the first one is used for the grid')
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
        tmp_path = '{}.{}.tmp'.format(b0_path, os.getpid())
        json.dump(b0, open(tmp_path, 'w'), indent=1)
        os.replace(tmp_path, b0_path)
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
        tmp_path = '{}.{}.tmp'.format(f, os.getpid())
        json.dump(r, open(tmp_path, 'w'), indent=1)
        os.replace(tmp_path, f)
        print('[{}] {} seed {}: best epoch {} valid R@20 {:.4f} | test R@20 {:.4f} N@20 {:.4f}'.format(
            mode, cfg, seed, r['best_epoch'], r['valid']['recall@20'], r['test']['recall@20'],
            r['test']['ndcg@20']), flush=True)
        return r

    def run_sf():
        f = os.path.join(out, 'SF.json')
        if os.path.isfile(f):
            return json.load(open(f))
        print('[SF] score fusion of the two frozen teachers', flush=True)
        r = score_fusion(mats, teachers, device)
        tmp_path = '{}.{}.tmp'.format(f, os.getpid())
        json.dump(r, open(tmp_path, 'w'), indent=1)
        os.replace(tmp_path, f)
        print('[SF] a = {}: valid R@20 {:.4f} | test R@20 {:.4f} N@20 {:.4f}'.format(
            r['cfg']['a'], r['valid']['recall@20'], r['test']['recall@20'], r['test']['ndcg@20']), flush=True)
        return r

    if not (args.grid or args.summary):
        if args.mode == 'SF':
            run_sf()
            return
        cfg = {'lr': args.lr, 'lam_D': args.lam_D}
        if args.mode in ('B3', 'B3d'):
            cfg['lam_J'] = args.lam_J
        run(args.mode, cfg, args.seed)
        return

    seeds = [int(x) for x in args.seeds.replace(',', ' ').split()]
    if args.grid:
        fix = dict(kv.split('=', 1) for kv in args.fix.split())
        for mode in args.modes.replace(',', ' ').split():
            if mode == 'SF':
                run_sf()
                continue
            grid = {k: ([float(fix[k])] if k in fix else v) for k, v in GRID[mode].items()}
            keys = sorted(grid)
            res = [run(mode, dict(zip(keys, v)), seeds[0]) for v in itertools.product(*[grid[k] for k in keys])]
            if not args.no_finals:
                top = max(res, key=lambda r: r['valid']['recall@20'])
                for sd in seeds:
                    run(mode, top['cfg'], sd)
    write_summary(out, b0, seeds)


def write_summary(out, b0, seeds):
    """Best setting per mode on valid (first seed) over ALL finished grid runs in out, plus its other seeds."""
    rows = []
    for mode in MODES:
        runs = []
        for f in sorted(os.listdir(out)):
            if f.startswith(mode + '_') and f.endswith('.json'):
                runs.append(json.load(open(os.path.join(out, f))))
        grid = [r for r in runs if r['seed'] == seeds[0]]
        if not grid:
            continue
        top = max(grid, key=lambda r: r['valid']['recall@20'])
        finals = [r for r in runs if r['cfg'] == top['cfg'] and r['seed'] in seeds]
        rows.append((mode, top['cfg'], finals, len(grid)))
    lines = ['# Text -> ID consolidation: {}\n'.format(os.path.basename(out)),
             'B0 = frozen ID teacher. Setting chosen on valid R@20 with seed {}; test = mean +- std over the '
             'finished seeds of that setting (teachers fixed).\n'.format(seeds[0]),
             '| config | setting | grid runs | seeds | best epochs | valid R@20 | test R@20 | test N@20 | vs B0 (R@20) |',
             '|---|---|---|---|---|---|---|---|---|',
             '| B0 ID only | - | - | - | - | {:.4f} | {:.4f} | {:.4f} | - |'.format(
                 b0['valid']['recall@20'], b0['test']['recall@20'], b0['test']['ndcg@20'])]
    for mode, cfg, finals, n_grid in rows:
        tr = np.array([r['test']['recall@20'] for r in finals])
        tn = np.array([r['test']['ndcg@20'] for r in finals])
        va = np.mean([r['valid']['recall@20'] for r in finals])
        lines.append('| {} | {} | {} | {} | {} | {:.4f} | {:.4f} ± {:.4f} | {:.4f} ± {:.4f} | {:+.2f}% |'.format(
            mode, cfg_tag(cfg), n_grid, ','.join(str(r['seed']) for r in finals),
            ','.join(str(r['best_epoch']) for r in finals), va, tr.mean(), tr.std(), tn.mean(), tn.std(),
            100 * (tr.mean() / b0['test']['recall@20'] - 1)))
    sf_path = os.path.join(out, 'SF.json')
    if os.path.isfile(sf_path):
        r = json.load(open(sf_path))
        lines.append('| SF | a={:g} (text weight, z-scored scores) | {} | - | - | {:.4f} | {:.4f} | {:.4f} | {:+.2f}% |'.format(
            r['cfg']['a'], len(r['curve']), r['valid']['recall@20'], r['test']['recall@20'], r['test']['ndcg@20'],
            100 * (r['test']['recall@20'] / b0['test']['recall@20'] - 1)))
    for mode, cfg, finals, n_grid in rows:
        if mode in GATE_MODES and 'gate_item' in finals[0]:
            g = finals[0]['gate_item']
            lines.append('\n{} gate (seed {}): mean A {:.3f}, channels open (>0.5) {:.1%}, item-mean quantiles '
                         '5/25/50/75/95% {}, by popularity tail/mid/head {:.3f}/{:.3f}/{:.3f}'.format(
                             mode, finals[0]['seed'], g['mean'], g['frac_channels_open(>0.5)'],
                             ' '.join('{:.3f}'.format(x) for x in g['quantiles_item_mean']),
                             g['mean_by_popularity']['tail'], g['mean_by_popularity']['mid'],
                             g['mean_by_popularity']['head']))
    lines.append('\nCorrections (B2, B3, B2d, B3d, IDcorr) and gates (FG, FGu): best epoch 0 = B0 itself (nothing useful '
                 'learned on valid). CAT / GATE start from a random fusion; SF has no training.')
    text = '\n'.join(lines) + '\n'
    open(os.path.join(out, 'summary.md'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
