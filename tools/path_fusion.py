# coding: utf-8
"""
Stage 2a, step 2: incremental ranking contribution of the two pathways of the Text teacher.

Frozen teachers (LightGCNC, taps at the best-valid epoch):
    ID   U_id, Y_id                                        taps/LightGCNC-<ds>-teacher_id
    Text U_t and its item layers T^(0..L)                  taps/LightGCNC-<ds>-teacher_text_layers
On the bipartite user-item graph, even item layers are built from item vectors only (T^(0) = E_T(text),
T^(2) = text of co-interacted items) and odd item layers from the free user table only (T^(1), T^(3)). The
teacher's item embedding is the mean of all layers, so its score splits exactly into two pathways:
    full   Y_full = mean_l T^(l)                (= the teacher's T3)
    TO     Y_TO   = mean of the even layers     text-origin pathway (L0 + L2)
    UO     Y_UO   = mean of the odd layers      user-origin pathway (L1 + L3)
    s_p(u, i) = <U_t[u], Y_p[i]>   (the user side is the same for every pathway)
Score-level fusion with the frozen ID teacher, scores z-normalised per scorer as in consolidate.py SF:
    F0  s_ID                                         ID only
    F1  (1 - a) z_ID + a z_full                     = SF(ID, Text)
    F2  (1 - a) z_ID + a z_TO
    F3  (1 - a) z_ID + a z_UO
    F4  w_ID z_ID + w_TO z_TO + w_UO z_UO,  w on the simplex   selective fusion of the two pathways
Weights chosen on valid R@20 (F1-F3: a in {0, 0.05, ..., 1}; F4: simplex step 0.1, then 0.05 around the best),
reported on test (full catalog; valid masks train, test masks train + valid). Also each scorer alone.
Paired bootstrap over test users (R@20, N@20): F2-F3, F2-F1, F3-F1, F4-F1, F1-F0, F2-F0, F3-F0.

Extra teachers (--teachers "to pure teacher_id_s1"): a bare name means taps/LightGCNC-<ds>-teacher_text_<name>
(e.g. a Text teacher trained with layer_mode=to); a name starting with teacher_ is a taps folder as is (e.g.
teacher_id_s1 = the ID teacher with seed 1: SF(ID, ID') is the ensemble control of SF(ID, Text)). For each:
T_<name> alone and G_<name> = SF(ID, <name>), with bootstrap against F1, F2 and F0.

    python tools/path_fusion.py -d dbbook
    python tools/path_fusion.py -d dbbook --teachers "to pure"
Output: results/path_fusion/<ds>/{results.json, summary.md}
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from probe_rec import load_data                              # noqa: E402
from probe_knn import evaluate                               # noqa: E402
from consolidate import zstats, binary, SF_GRID              # noqa: E402
from jepay.metrics import paired_bootstrap                   # noqa: E402

PAIRS = [('F2', 'F3', 'text-origin vs user-origin'), ('F2', 'F1', 'text-origin vs full text'),
         ('F3', 'F1', 'user-origin vs full text'), ('F4', 'F1', 'selective vs full text'),
         ('F1', 'F0', 'SF(ID, Text) vs ID'), ('F2', 'F0', 'ID + text-origin vs ID'),
         ('F3', 'F0', 'ID + user-origin vs ID')]


def layers_of(path):
    files = sorted(glob.glob(os.path.join(path, 'T_item_l*.npy')), key=lambda f: int(f.split('_l')[-1][:-4]))
    if not files:
        raise SystemExit('{}: no T_item_l*.npy (run scripts/text_graph_probe.pbs first)'.format(path))
    return [np.load(f).astype(np.float32) for f in files]


class Scorer:
    """z-normalised score components on the device; combine(weights) -> score_batch for evaluate()."""

    def __init__(self, U_id, Y_id, U_t, items, device, extra=None):
        self.dev = device
        t = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
        self.comp = {'ID': (t(U_id), t(Y_id))}
        self.comp.update({k: (t(U_t), t(Y)) for k, Y in items.items()})
        self.comp.update({k: (t(U), t(Y)) for k, (U, Y) in (extra or {}).items()})
        self.st = {k: zstats(U, Y) for k, (U, Y) in self.comp.items()}

    @torch.no_grad()
    def combine(self, w):
        w = {k: v for k, v in w.items() if v != 0}

        def score(ub):
            u = torch.from_numpy(ub).long().to(self.dev)
            s = 0
            for k, a in w.items():
                U, Y = self.comp[k]
                m, sd = self.st[k]
                s = s + a * (U[u] @ Y.T - m) / sd
            return s.cpu().numpy()
        return score


def rank(scorer, w, hist, gt, mask):
    r, n = evaluate(scorer.combine(w), hist, gt, mask, k=20)
    return {'recall@20': float(r.mean()), 'ndcg@20': float(n.mean())}, r, n


def simplex(step):
    n = int(round(1 / step))
    return [(i * step, j * step, (n - i - j) * step) for i in range(n + 1) for j in range(n + 1 - i)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True)
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--id_taps', default=None)
    ap.add_argument('--text_taps', default=None)
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'results', 'path_fusion'))
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--teachers', default='', help='extra text teachers, e.g. "to pure" (taps ...-teacher_text_<name>)')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    taps = lambda name: os.path.join(ROOT, 'taps', 'LightGCNC-{}-{}'.format(args.dataset, name))  # noqa: E731
    id_taps = args.id_taps or taps('teacher_id')
    text_taps = args.text_taps or taps('teacher_text_layers')
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    out = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out, exist_ok=True)

    U_id = np.load(os.path.join(id_taps, 'T3_user.npy')).astype(np.float32)
    Y_id = np.load(os.path.join(id_taps, 'T3.npy')).astype(np.float32)
    U_t = np.load(os.path.join(text_taps, 'T3_user.npy')).astype(np.float32)
    L = layers_of(text_taps)
    items = {'full': np.mean(L, 0), 'TO': np.mean(L[0::2], 0), 'UO': np.mean(L[1::2], 0)}
    gap = float(np.abs(items['full'] - np.load(os.path.join(text_taps, 'T3.npy'))).max())
    print('[fusion] {} item layers; text-origin = layers {}, user-origin = layers {}; |mean - T3| = {:.2e}'.format(
        len(L), list(range(0, len(L), 2)), list(range(1, len(L), 2)), gap), flush=True)

    (R_tr, R_va, R_te), n_users, n_items = load_data(args.data_dir, args.dataset, min_items=Y_id.shape[0])
    assert (n_users, n_items) == (U_id.shape[0], Y_id.shape[0]) == (U_t.shape[0], L[0].shape[0])
    sel = (R_tr, R_va, R_tr)                    # hist, gt, mask (as SF in consolidate.py)
    tst = (R_tr, R_te, binary(R_tr, R_va))
    extra = {}
    for name in args.teachers.split():
        d = taps(name if name.startswith('teacher_') else 'teacher_text_' + name)
        if not os.path.isfile(os.path.join(d, 'T3.npy')):
            print('[fusion] no taps for teacher {} at {}: skipped'.format(name, d), flush=True)
            continue
        extra[name] = (np.load(os.path.join(d, 'T3_user.npy')).astype(np.float32),
                       np.load(os.path.join(d, 'T3.npy')).astype(np.float32))
        meta = os.path.join(d, 'meta.json')
        if os.path.isfile(meta):
            print('[fusion] teacher {}: {}'.format(name, json.load(open(meta)).get('config', '')), flush=True)
    S = Scorer(U_id, Y_id, U_t, items, device, extra)

    res, per_user = {}, {}

    def run(name, w):
        va = rank(S, w, *sel)[0]
        te, r, n = rank(S, w, *tst)
        res[name] = {'w': w, 'valid': va, 'test': te}
        per_user[name] = (r, n)
        print('[fusion] {:7s} {}: valid R@20 {:.4f} | test R@20 {:.4f} N@20 {:.4f}'.format(
            name, {k: round(v, 2) for k, v in w.items()}, va['recall@20'], te['recall@20'], te['ndcg@20']),
            flush=True)

    run('F0', {'ID': 1.0})
    for p in ('full', 'TO', 'UO'):
        run('T_' + p, {p: 1.0})                               # each text scorer alone
    for name, p in (('F1', 'full'), ('F2', 'TO'), ('F3', 'UO')):
        best = max(SF_GRID, key=lambda a: rank(S, {'ID': 1 - a, p: a}, *sel)[0]['recall@20'])
        run(name, {'ID': 1 - best, p: best})
    # F4: simplex over (ID, TO, UO), coarse 0.1 then 0.05 around the best
    cache = {}

    def val(wt):
        wt = tuple(round(x, 2) for x in wt)
        if wt not in cache:
            cache[wt] = rank(S, dict(zip(('ID', 'TO', 'UO'), wt)), *sel)[0]['recall@20']
        return cache[wt]
    best = max(simplex(0.1), key=val)
    near = [w for w in simplex(0.05) if max(abs(a - b) for a, b in zip(w, best)) <= 0.1 + 1e-9]
    best = max(near, key=val)
    run('F4', dict(zip(('ID', 'TO', 'UO'), (round(x, 2) for x in best))))
    res['F4']['grid_points'] = len(cache)
    pairs = list(PAIRS)
    for name in extra:
        run('T_' + name, {name: 1.0})
        a = max(SF_GRID, key=lambda a: rank(S, {'ID': 1 - a, name: a}, *sel)[0]['recall@20'])
        run('G_' + name, {'ID': 1 - a, name: a})
        pairs += [('G_' + name, 'F1', 'SF(ID, {}) vs SF(ID, Text)'.format(name)),
                  ('G_' + name, 'F2', 'SF(ID, {}) vs SF(ID, post-hoc TO)'.format(name)),
                  ('G_' + name, 'F0', 'SF(ID, {}) vs ID'.format(name))]

    boot = []
    for a, b, meaning in pairs:
        for j, metric in enumerate(('recall@20', 'ndcg@20')):
            bt = paired_bootstrap(per_user[a][j] - per_user[b][j], n_boot=args.n_boot)
            boot.append({'a': a, 'b': b, 'meaning': meaning, 'metric': metric, **bt})
    json.dump({'results': res, 'bootstrap': boot, 'norm': S.st, 'n_layers': len(L) - 1, 'T3_gap': gap},
              open(os.path.join(out, 'results.json'), 'w'), indent=1)
    write_summary(out, args.dataset, res, boot, len(L) - 1, list(extra))


def write_summary(out, ds, res, boot, n_layers, extra=()):
    label = {'F0': 'ID only', 'T_full': 'Text teacher alone (full)', 'T_TO': 'text-origin alone',
             'T_UO': 'user-origin alone', 'F1': 'SF(ID, full Text)', 'F2': 'SF(ID, text-origin)',
             'F3': 'SF(ID, user-origin)', 'F4': 'ID + TO + UO, selective'}
    b0 = res['F0']['test']['recall@20']
    lines = ['# Stage 2a step 2: pathway fusion ({}), Text teacher with {} layers\n'.format(ds, n_layers),
             'Full-catalog ranking; weights on valid R@20; z-normalised scores. text-origin = even item layers '
             '(T0, T2: item text), user-origin = odd item layers (T1, T3: free user table).\n',
             '| config | weights (ID, TO, UO / full) | valid R@20 | test R@20 | test N@20 | vs F0 |',
             '|---|---|---|---|---|---|']
    for name in extra:
        nm = name[len('teacher_'):] if name.startswith('teacher_') else 'Text teacher layer_mode=' + name
        label['T_' + name] = '{} alone'.format(nm)
        label['G_' + name] = 'SF(ID, {})'.format(nm)
    keys = ['F0', 'T_full', 'T_TO', 'T_UO', 'F1', 'F2', 'F3', 'F4'] + \
        [k for name in extra for k in ('T_' + name, 'G_' + name)]
    for k in keys:
        r = res[k]
        lines.append('| {} {} | {} | {:.4f} | {:.4f} | {:.4f} | {:+.1f}% |'.format(
            k, label[k], ', '.join('{}={:.2f}'.format(a, b) for a, b in r['w'].items()), r['valid']['recall@20'],
            r['test']['recall@20'], r['test']['ndcg@20'], 100 * (r['test']['recall@20'] / b0 - 1)))
    lines += ['\n## Paired bootstrap over test users (difference first - second)\n',
              '| comparison | meaning | metric | diff | 95% CI | p (one-sided, first > second) |',
              '|---|---|---|---|---|---|']
    for b in boot:
        lines.append('| {} vs {} | {} | {} | {:+.4f} | [{:+.4f}, {:+.4f}] | {:.4f} |'.format(
            b['a'], b['b'], b['meaning'], b['metric'], b['mean'], b['ci_lo'], b['ci_hi'], b['p_one_sided']))
    text = '\n'.join(lines) + '\n'
    open(os.path.join(out, 'summary.md'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
