# coding: utf-8
"""
SCOPE-ID (plan.md 4c): are the two branches complementary, and is the concatenation a poor way to combine them?

For every two-branch SCOPEID run (taps/SCOPEID-<ds>-es_<exp>, seed 999; e.g. lf1_r05 = low + full, ff = two full):
    A, B   each branch alone: <U_b[u], Y_b[i]> (taps T3_user_<b>, T3_<b>)
    CAT    the model itself (concatenation = A + B with equal raw weight; taps T3_user, T3)
    SF     (1 - a) z(A) + a z(B), z-normalised scores, a in {0, 0.05, ..., 1} chosen on valid R@20
    ORACLE per-user max of the test recall of A and B (upper bound, uses test labels: NOT a method)
Cross-run control (independently trained): SF(f, l_rXX) from the single-branch runs es_f and es_l_rXX.
Reported on test (full catalog; train + valid masked): R@20 / N@20 overall and R@20 per item-degree group
(tail / mid / head = terciles of the items by training degree; users with >= 1 test item of the group).
Paired bootstrap over test users: SF - CAT (does re-weighting help?), CAT(lf) - CAT(ff), SF(lf) - SF(ff),
ORACLE(lf) - ORACLE(ff) (is low + full more complementary than full + full?).

    python tools/scope_fusion.py -d dbbook
    python tools/scope_fusion.py -d baby --runs "lf1_r05 ff lf1_r05_tg ff_tg"
    python tools/scope_fusion.py -d dbbook --model SCOPEKD --runs "kd_hard kd_soft kd_star kd0" --cross "" \
        --out_dir results/scopekd_fusion          # student (low) vs frozen teacher (full): SF and ORACLE
For a grid experiment the seed-999 combination with the best valid R@20 is used.
Output: results/scope_fusion/<ds>/{results.json, summary.md}
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from probe_rec import load_data                  # noqa: E402
from probe_knn import evaluate                   # noqa: E402
from consolidate import zstats, binary, SF_GRID  # noqa: E402
from jepay.metrics import paired_bootstrap       # noqa: E402
from taps_best import runs as taps_runs          # noqa: E402

GROUPS = ('tail', 'mid', 'head')


MODEL = 'SCOPEID'


def taps_dir(ds, exp, model=None):
    """Taps folder of the seed-999 run of an experiment (best valid R@20 if the experiment is a grid)."""
    r = taps_runs(model or MODEL, ds, [exp], taps_dir=os.path.join(ROOT, 'taps'), seed=999)
    return max(r, key=lambda x: x['valid'])['dir'] if r else None


def load(d, name):
    f = os.path.join(d, name + '.npy')
    return np.load(f).astype(np.float32) if os.path.isfile(f) else None


class Scorers:
    def __init__(self, device):
        self.dev, self.comp, self.st = device, {}, {}

    def add(self, key, U, Y):
        t = lambda x: torch.from_numpy(x).to(self.dev)  # noqa: E731
        self.comp[key] = (t(U), t(Y))
        self.st[key] = zstats(*self.comp[key])

    def fn(self, w, z=True):
        w = {k: v for k, v in w.items() if v != 0}

        @torch.no_grad()
        def score(ub):
            u = torch.from_numpy(ub).long().to(self.dev)
            s = 0
            for k, a in w.items():
                U, Y = self.comp[k]
                raw = U[u] @ Y.T
                if z:
                    m, sd = self.st[k]
                    raw = (raw - m) / sd
                s = s + a * raw
            return s.cpu().numpy()
        return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True)
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--runs', nargs='+',
                    default='lf1_r03 lf1_r05 lf1_r10 ff lf1_r03_tg lf1_r05_tg lf1_r10_tg ff_tg'.split())
    ap.add_argument('--cross', nargs='*', default='l_r03 l_r05 l_r10'.split(),
                    help='SF(f, <run>) of independently trained runs (give --cross with nothing to skip)')
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'results', 'scope_fusion'))
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--model', default='SCOPEID', help='SCOPEID, or SCOPEKD (branches: student low, teacher full)')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    global MODEL
    MODEL = args.model
    ds = args.dataset
    dev = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    out = os.path.join(args.out_dir, ds)
    os.makedirs(out, exist_ok=True)

    probe = taps_dir(ds, 'f', 'SCOPEID') or taps_dir(ds, args.runs[0])
    n_items = load(probe, 'T3').shape[0]
    (R_tr, R_va, R_te), n_users, n_items = load_data(args.data_dir, ds, min_items=n_items)
    sel = (R_tr, R_va, R_tr)
    tst = (R_tr, R_te, binary(R_tr, R_va))
    deg = np.asarray(R_tr.sum(0)).ravel()
    order = np.argsort(deg, kind='stable')
    grp = np.empty(n_items, dtype=int)
    for g, idx in enumerate(np.array_split(order, 3)):
        grp[idx] = g
    gts = {}
    for g, name in enumerate(GROUPS):
        keep = np.zeros(n_items, dtype=np.float32)
        keep[grp == g] = 1
        gts[name] = (R_te @ sp.diags(keep)).tocsr()
        gts[name].eliminate_zeros()

    S = Scorers(dev)
    res, per_user = {}, {}

    def measure(name, w, z=True):
        va, _ = evaluate(S.fn(w, z), *sel, k=20)
        r, n = evaluate(S.fn(w, z), *tst, k=20)
        groups = {g: float(evaluate(S.fn(w, z), R_tr, gts[g], tst[2], k=20)[0].mean()) for g in GROUPS}
        res[name] = {'w': w, 'z': z, 'valid_r20': float(va.mean()), 'test_r20': float(r.mean()),
                     'test_n20': float(n.mean()), 'test_r20_groups': groups}
        per_user[name] = (r, n)
        print('[scope] {:24s} valid {:.4f} | test R@20 {:.4f} N@20 {:.4f} | tail {:.4f} mid {:.4f} head {:.4f}'.format(
            name, va.mean(), r.mean(), n.mean(), groups['tail'], groups['mid'], groups['head']), flush=True)

    def sf(name, ka, kb):
        a = max(SF_GRID, key=lambda a: evaluate(S.fn({ka: 1 - a, kb: a}), *sel, k=20)[0].mean())
        measure(name, {ka: round(1 - a, 2), kb: a})

    def oracle(name, ka, kb):
        ra, na = per_user[ka]
        rb, nb = per_user[kb]
        pick = rb > ra
        per_user[name] = (np.where(pick, rb, ra), np.where(pick, nb, na))
        res[name] = {'test_r20': float(per_user[name][0].mean()), 'test_n20': float(per_user[name][1].mean()),
                     'share_b': float(pick.mean()), 'note': 'per-user max over test labels (upper bound)'}
        print('[scope] {:24s} test R@20 {:.4f} (upper bound; branch B chosen for {:.0%} of users)'.format(
            name, res[name]['test_r20'], pick.mean()), flush=True)

    done = []
    for exp in args.runs:
        d = taps_dir(ds, exp)
        if d is None:
            print('[scope] no taps for es_{}: skipped'.format(exp), flush=True)
            continue
        names = sorted(f[len('T3_user_'):-4] for f in os.listdir(d) if f.startswith('T3_user_'))
        if len(names) != 2:
            print('[scope] es_{}: branches {} (need 2): skipped'.format(exp, names), flush=True)
            continue
        if 'low' in names:
            names = ['low', 'full']
        A, B = exp + ':' + names[0], exp + ':' + names[1]
        S.add(A, load(d, 'T3_user_' + names[0]), load(d, 'T3_' + names[0]))
        S.add(B, load(d, 'T3_user_' + names[1]), load(d, 'T3_' + names[1]))
        S.add(exp + ':cat', load(d, 'T3_user'), load(d, 'T3'))
        measure(A, {A: 1.0})
        measure(B, {B: 1.0})
        measure(exp + ':CAT', {exp + ':cat': 1.0}, z=False)
        sf(exp + ':SF', A, B)
        oracle(exp + ':ORACLE', A, B)
        done.append(exp)

    fd = taps_dir(ds, 'f', 'SCOPEID')
    if fd is not None and args.cross:
        S.add('f', load(fd, 'T3_user'), load(fd, 'T3'))
        measure('f', {'f': 1.0}, z=False)
        for c in args.cross:
            d = taps_dir(ds, c, 'SCOPEID')
            if d is None:
                continue
            S.add(c, load(d, 'T3_user'), load(d, 'T3'))
            measure(c, {c: 1.0}, z=False)
            sf('SF(f,{})'.format(c), 'f', c)

    pairs = []
    for exp in done:
        pairs.append((exp + ':SF', exp + ':CAT', 're-weighting vs concatenation'))
    for sfx in ('', '_tg'):
        ff = 'ff' + sfx
        for exp in done:
            if exp.startswith('lf1_') and exp.endswith(sfx) and (sfx or not exp.endswith('_tg')) and ff in done:
                for kind in ('CAT', 'SF', 'ORACLE'):
                    pairs.append(('{}:{}'.format(exp, kind), '{}:{}'.format(ff, kind),
                                  'low + full vs full + full ({})'.format(kind)))
    boot = []
    for a, b, meaning in pairs:
        if a in per_user and b in per_user:
            for j, metric in enumerate(('recall@20', 'ndcg@20')):
                bt = paired_bootstrap(per_user[a][j] - per_user[b][j], n_boot=args.n_boot)
                boot.append({'a': a, 'b': b, 'meaning': meaning, 'metric': metric, **bt})
    json.dump({'results': res, 'bootstrap': boot}, open(os.path.join(out, 'results.json'), 'w'), indent=1)

    lines = ['# SCOPE-ID branch fusion ({}), seed 999\n'.format(ds),
             'A / B = one branch alone; CAT = the model (concatenation); SF = z-score fusion with the weight chosen on '
             'valid; ORACLE = per-user max of A and B on test labels (upper bound, not a method).\n',
             '| scorer | weights | valid R@20 | test R@20 | test N@20 | tail | mid | head |',
             '|---|---|---|---|---|---|---|---|']
    for k, v in res.items():
        if 'valid_r20' in v:
            g = v['test_r20_groups']
            lines.append('| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                k, {a: b for a, b in v['w'].items()}, v['valid_r20'], v['test_r20'], v['test_n20'],
                g['tail'], g['mid'], g['head']))
        else:
            lines.append('| {} | upper bound (B for {:.0%} of users) | | {:.4f} | {:.4f} | | | |'.format(
                k, v['share_b'], v['test_r20'], v['test_n20']))
    lines += ['', '## Paired bootstrap over test users', '', '| a - b | meaning | metric | diff | 95% CI | p (a > b) |',
              '|---|---|---|---|---|---|']
    for b in boot:
        lines.append('| {} - {} | {} | {} | {:+.4f} | [{:+.4f}, {:+.4f}] | {:.4f} |'.format(
            b['a'], b['b'], b['meaning'], b['metric'], b['mean'], b['ci_lo'], b['ci_hi'], b['p_one_sided']))
    open(os.path.join(out, 'summary.md'), 'w').write('\n'.join(lines) + '\n')
    print('[scope] -> {}'.format(os.path.join(out, 'summary.md')), flush=True)


if __name__ == '__main__':
    main()
