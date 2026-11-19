# coding: utf-8
"""
Evaluation of LightGCNAlign runs (src/models/lightgcnalign.py): does InfoNCE learn ID-Text agreement, and does
the agreement help ranking?

For every run (taps/LightGCNAlign-<ds>-<exp>/ with T3, T3_user = ID branch; TT, TT_user = Text branch) and, as
a reference, the separately trained teachers (taps/LightGCNC-<ds>-teacher_id + teacher_text_layers):
  Ranking (full catalog; valid masks train, test masks train + valid), R@20 / N@20:
    ID alone, Text alone, SF(ID, Text) = (1 - a) z(s_ID) + a z(s_T), a in {0, ..., 1} chosen on valid R@20
  Matching (cosine between the Text and the ID representation, candidates = ALL items / users):
    item Recall@1 / @10 of the same item for queries in
        nce   items of the train part of data/<ds>/jepay_item_split.npz (the only items allowed in L_I)
        held  the other items (never in L_I; still in BPR)
    mean cosine of the same item and of random item pairs
    user Recall@1 / @10 on a fixed sample of users (all users can enter L_U)
  JEPA runs (taps TT_pred = P(T^G), TT_user_pred = P_u(U^T)): the same matching in the predicted ID space
    (P(T) -> Y^ID), R^2 of P(T) for Y^ID on nce / held items; raw Text -> ID matching is reported separately.
  Effective rank of T^G (collapse check).
  Paired bootstrap over test users (Text alone and SF) against the baseline of the same family: runs with a
    frozen ID branch (align_F*) against align_F0, the others against align_C0.

    python tools/align_eval.py -d baby                    # every taps/LightGCNAlign-<ds>-align_* folder
    python tools/align_eval.py -d baby --runs "align_C0 align_C1 align_C2"
Output: results/align/<ds>/{results.json, summary.md}
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from probe_rec import load_data                          # noqa: E402
from consolidate import binary, SF_GRID                  # noqa: E402
from path_fusion import Scorer                           # noqa: E402
from jepay.metrics import paired_bootstrap               # noqa: E402


def load_run(path, ref=None):
    if ref is not None:                                   # separately trained teachers
        id_dir, t_dir = ref
        g = lambda d, f: np.load(os.path.join(d, f)).astype(np.float32)  # noqa: E731
        return {'U_id': g(id_dir, 'T3_user.npy'), 'Y_id': g(id_dir, 'T3.npy'),
                'U_t': g(t_dir, 'T3_user.npy'), 'T': g(t_dir, 'T3.npy'), 'meta': {},
                'files': [os.path.join(id_dir, 'T3.npy'), os.path.join(t_dir, 'T3.npy')]}
    g = lambda f: np.load(os.path.join(path, f)).astype(np.float32)  # noqa: E731
    meta = os.path.join(path, 'meta.json')
    out = {'U_id': g('T3_user.npy'), 'Y_id': g('T3.npy'), 'U_t': g('TT_user.npy'), 'T': g('TT.npy'),
           'meta': json.load(open(meta)) if os.path.isfile(meta) else {},
           'files': [os.path.join(path, f) for f in ('T3.npy', 'TT.npy', 'TT_pred.npy')]}
    for k, f in (('P', 'TT_pred.npy'), ('P_u', 'TT_user_pred.npy')):
        if os.path.isfile(os.path.join(path, f)):
            out[k] = g(f)
    return out


def r2(P, Y, idx):
    res = ((P[idx] - Y[idx]) ** 2).sum()
    return float(1.0 - res / max(((Y[idx] - Y[idx].mean(0)) ** 2).sum(), 1e-12))


def erank(Z):
    s = np.linalg.svd((Z - Z.mean(0)).astype(np.float64), compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def baseline_of(name):
    return 'align_F0' if name.startswith('align_F') else 'align_C0'


@torch.no_grad()
def matching(Q, K, queries, device, ks=(1, 10), chunk=2048):
    """Rank of the true key K[q] for query Q[q] among all rows of K (cosine)."""
    Qn = torch.nn.functional.normalize(torch.from_numpy(Q).to(device), dim=1)
    Kn = torch.nn.functional.normalize(torch.from_numpy(K).to(device), dim=1)
    ranks, same = [], []
    for s in range(0, len(queries), chunk):
        q = torch.from_numpy(queries[s:s + chunk]).long().to(device)
        sims = Qn[q] @ Kn.T
        true = sims.gather(1, q.view(-1, 1))
        ranks.append((sims > true).sum(1).cpu().numpy())
        same.append(true.view(-1).cpu().numpy())
    r, c = np.concatenate(ranks), np.concatenate(same)
    out = {'R@{}'.format(k): float((r < k).mean()) for k in ks}
    out['cos_same'] = float(c.mean())
    return out


@torch.no_grad()
def random_cos(Q, K, n, device, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, Q.shape[0], n), rng.integers(0, K.shape[0], n)
    Qn = torch.nn.functional.normalize(torch.from_numpy(Q[a]).to(device), dim=1)
    Kn = torch.nn.functional.normalize(torch.from_numpy(K[b]).to(device), dim=1)
    return float((Qn * Kn).sum(1).mean())


@torch.no_grad()
def fast_rank(S, ws, hist, gt, mask, k=20, batch=2048):
    """Recall@k / NDCG@k per user for several weightings at once (same users and definitions as
    probe_knn.evaluate: users with history and ground truth; recall = hits / |gt|; NDCG with ideal DCG over
    min(|gt|, k)). S: path_fusion.Scorer (z-normalised components); ws: list of {component: weight}.
    Returns [(metrics, recall per user, ndcg per user)] in the order of ws."""
    users = np.where((np.asarray(gt.sum(1)).ravel() > 0) & (np.asarray(hist.sum(1)).ravel() > 0))[0]
    dev = S.dev
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, device=dev, dtype=torch.float32))
    idcg = disc.cumsum(0)
    keys = sorted({c for w in ws for c, a in w.items() if a != 0})
    R, N = np.zeros((len(ws), len(users))), np.zeros((len(ws), len(users)))
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    gt, mask = gt.tocsr(), mask.tocsr()
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        u = torch.from_numpy(ub).long().to(dev)
        comp = {}
        for c in keys:
            U, Y = S.comp[c]
            m, sd = S.st[c]
            comp[c] = (U[u] @ Y.T - m) / sd
        n_items = next(iter(comp.values())).shape[1]
        mb = mask[ub].tocoo()
        mr, mc = torch.from_numpy(mb.row).long().to(dev), torch.from_numpy(mb.col).long().to(dev)
        gb = gt[ub].tocoo()
        G = torch.zeros(len(ub), n_items, dtype=torch.bool, device=dev)
        G[torch.from_numpy(gb.row).long().to(dev), torch.from_numpy(gb.col).long().to(dev)] = True
        cnt = G.sum(1).float()
        best_dcg = idcg[(cnt.clamp(max=k) - 1).long()]
        noise = 1e-6 * torch.rand(len(ub), n_items, generator=gen, device=dev)      # random tie-break
        for wi, w in enumerate(ws):
            sc = noise.clone()
            for c, a in w.items():
                if a != 0:
                    sc += a * comp[c]
            sc[mr, mc] = -float('inf')
            hits = G.gather(1, sc.topk(k, dim=1).indices).float()
            R[wi, s:s + len(ub)] = (hits.sum(1) / cnt).cpu().numpy()
            N[wi, s:s + len(ub)] = ((hits * disc).sum(1) / best_dcg).cpu().numpy()
    return [({'recall@20': float(R[i].mean()), 'ndcg@20': float(N[i].mean())}, R[i], N[i]) for i in range(len(ws))]


def stamp(paths):
    return [int(os.path.getmtime(p)) if os.path.isfile(p) else 0 for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True)
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--runs', default='auto', help='exp names, or auto = every taps/LightGCNAlign-<ds>-align_*')
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'results', 'align'))
    ap.add_argument('--n_users', type=int, default=5000, help='user sample for user matching')
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--no_cache', action='store_true', help='re-evaluate runs whose taps did not change')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    taps = lambda m, e: os.path.join(ROOT, 'taps', '{}-{}-{}'.format(m, args.dataset, e))  # noqa: E731
    out = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out, exist_ok=True)

    runs = {}
    ref = (taps('LightGCNC', 'teacher_id'), taps('LightGCNC', 'teacher_text_layers'))
    if all(os.path.isfile(os.path.join(d, 'T3.npy')) for d in ref):
        runs['separate'] = load_run(None, ref)
    if args.runs == 'auto':
        pre = 'LightGCNAlign-{}-'.format(args.dataset)
        names = sorted(d[len(pre):] for d in os.listdir(os.path.join(ROOT, 'taps')) if d.startswith(pre + 'align_'))
        order = lambda n: (0 if n.startswith('align_C') else 1, n)  # noqa: E731
        args.runs = ' '.join(sorted(names, key=order))
    for e in args.runs.split():
        p = taps('LightGCNAlign', e)
        if os.path.isfile(os.path.join(p, 'TT.npy')):
            runs[e] = load_run(p)
        else:
            print('[align] no taps for {} at {}: skipped'.format(e, p), flush=True)
    if not runs:
        raise SystemExit('no runs')
    n_items = next(iter(runs.values()))['Y_id'].shape[0]
    (R_tr, R_va, R_te), n_users, _ = load_data(args.data_dir, args.dataset, min_items=n_items)
    sel, tst = (R_tr, R_va, R_tr), (R_tr, R_te, binary(R_tr, R_va))
    split = np.load(os.path.join(args.data_dir, args.dataset, 'jepay_item_split.npz'))
    nce_items = np.sort(split['train'])
    held = np.setdiff1d(np.arange(n_items), nce_items)
    users = np.sort(np.random.default_rng(0).choice(n_users, min(args.n_users, n_users), replace=False))

    cache = os.path.join(out, 'cache')
    os.makedirs(cache, exist_ok=True)
    res, per_user = {}, {}
    for name, R in runs.items():
        cj, cn = os.path.join(cache, name + '.json'), os.path.join(cache, name + '_users.npz')
        key = stamp(R['files'])
        if not args.no_cache and os.path.isfile(cj) and os.path.isfile(cn) and json.load(open(cj)).get('_stamp') == key:
            res[name] = json.load(open(cj))
            z = np.load(cn)
            for k in ('ID', 'Text', 'SF'):
                per_user[(name, k)] = (z[k + '_r'], z[k + '_n'])
            print('[align] {}: cached'.format(name), flush=True)
            continue
        S = Scorer(R['U_id'], R['Y_id'], R['U_t'], {'T': R['T']}, device)
        r = {'hyper': R['meta'].get('hyper_parameters', {}), 'epoch': R['meta'].get('epoch')}
        ws = [{'ID': 1.0}, {'T': 1.0}] + [{'ID': 1 - a, 'T': a} for a in SF_GRID]
        va = fast_rank(S, ws, *sel)
        a = SF_GRID[int(np.argmax([v[0]['recall@20'] for v in va[2:]]))]
        te = fast_rank(S, [ws[0], ws[1], {'ID': 1 - a, 'T': a}], *tst)
        for i, k in enumerate(('ID', 'Text')):
            r[k] = {'valid': va[i][0], 'test': te[i][0]}
            per_user[(name, k)] = te[i][1:]
        r['SF'] = {'a': a, 'valid': va[2 + SF_GRID.index(a)][0], 'test': te[2][0]}
        per_user[(name, 'SF')] = te[2][1:]
        r['match_item_nce'] = matching(R['T'], R['Y_id'], nce_items, device)
        r['match_item_held'] = matching(R['T'], R['Y_id'], held, device)
        r['cos_random_items'] = random_cos(R['T'], R['Y_id'], 20000, device)
        r['match_user'] = matching(R['U_t'], R['U_id'], users, device)
        r['cos_random_users'] = random_cos(R['U_t'], R['U_id'], 20000, device)
        r['erank_T'] = erank(R['T'])
        if 'P' in R:
            r['pred_item_nce'] = matching(R['P'], R['Y_id'], nce_items, device)
            r['pred_item_held'] = matching(R['P'], R['Y_id'], held, device)
            r['pred_r2_nce'], r['pred_r2_held'] = r2(R['P'], R['Y_id'], nce_items), r2(R['P'], R['Y_id'], held)
            r['erank_P'] = erank(R['P'])
        if 'P_u' in R:
            r['pred_user'] = matching(R['P_u'], R['U_id'], users, device)
            r['pred_r2_user'] = r2(R['P_u'], R['U_id'], users)
        r['_stamp'] = key
        res[name] = r
        json.dump(r, open(cj, 'w'), indent=1)
        np.savez_compressed(cn, **{k + suf: per_user[(name, k)][j] for k in ('ID', 'Text', 'SF')
                                   for j, suf in enumerate(('_r', '_n'))})
        print('[align] {}: ID {:.4f} | Text {:.4f} | SF {:.4f} (a {}) | item R@10 nce {:.3f} held {:.3f} | user R@10 '
              '{:.3f}'.format(name, r['ID']['test']['recall@20'], r['Text']['test']['recall@20'],
                              r['SF']['test']['recall@20'], a, r['match_item_nce']['R@10'],
                              r['match_item_held']['R@10'], r['match_user']['R@10']), flush=True)

    boot = []
    for name in res:
        base = baseline_of(name)
        if name in (base, 'separate') or base not in res:
            continue
        for k in ('Text', 'SF'):
            for j, m in enumerate(('recall@20', 'ndcg@20')):
                bt = paired_bootstrap(per_user[(name, k)][j] - per_user[(base, k)][j], n_boot=args.n_boot)
                boot.append({'a': name, 'b': base, 'score': k, 'metric': m, **bt})
    json.dump({'results': res, 'bootstrap': boot, 'n_items': n_items, 'n_nce_items': int(len(nce_items)),
               'n_held_items': int(len(held))}, open(os.path.join(out, 'results.json'), 'w'), indent=1)
    write_summary(out, args.dataset, res, boot, n_items, len(nce_items), len(held))


def write_summary(out, ds, res, boot, n_items, n_nce, n_held):
    L = ['# ID-Text alignment: InfoNCE and JEPA ({})\n'.format(ds),
         'separate = ID and Text teachers trained separately (LightGCNC teacher_id, teacher_text_layers). '
         'Ranking: full catalog, test. Matching: cosine Text -> ID over all {} items (chance R@10 = {:.4f}); '
         '"nce" = {} items allowed in L_I, "held" = {} items never in L_I.\n'.format(n_items, 10 / n_items, n_nce, n_held),
         '## Ranking (test)\n',
         '| run | setting | ID R@20 | ID N@20 | Text R@20 | Text N@20 | SF a | SF R@20 | SF N@20 |',
         '|---|---|---|---|---|---|---|---|---|']
    for n, r in res.items():
        L.append('| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} | {:.4f} | {:.4f} |'.format(
            n, ' '.join('{}={}'.format(k, v) for k, v in r['hyper'].items() if k != 'seed') or '-',
            r['ID']['test']['recall@20'], r['ID']['test']['ndcg@20'], r['Text']['test']['recall@20'],
            r['Text']['test']['ndcg@20'], r['SF']['a'], r['SF']['test']['recall@20'], r['SF']['test']['ndcg@20']))
    L += ['\n## Raw matching (Text -> ID, cosine)\n',
          '| run | item R@1 nce | item R@10 nce | item R@1 held | item R@10 held | cos same (nce / held) | '
          'cos random | user R@1 | user R@10 | user cos same / random | erank T^G |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for n, r in res.items():
        a, h, u = r['match_item_nce'], r['match_item_held'], r['match_user']
        L.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.3f} / {:.3f} | {:.3f} | {:.4f} | {:.4f} | {:.3f} / {:.3f} | '
                 '{:.1f} |'.format(n, a['R@1'], a['R@10'], h['R@1'], h['R@10'], a['cos_same'], h['cos_same'],
                                   r['cos_random_items'], u['R@1'], u['R@10'], u['cos_same'], r['cos_random_users'],
                                   r['erank_T']))
    jr = {n: r for n, r in res.items() if 'pred_item_nce' in r}
    if jr:
        L += ['\n## JEPA: predicted ID space (P(T^G) -> Y^ID, cosine matching; R^2 of P(T^G) for Y^ID)\n',
              '| run | item R@1 nce | item R@10 nce | item R@1 held | item R@10 held | R^2 nce | R^2 held | '
              'erank P(T) | user R@1 | user R@10 | user R^2 |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for n, r in jr.items():
            a, h, u = r['pred_item_nce'], r['pred_item_held'], r.get('pred_user')
            L.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.1f} | {} | {} | {} |'.format(
                n, a['R@1'], a['R@10'], h['R@1'], h['R@10'], r['pred_r2_nce'], r['pred_r2_held'], r['erank_P'],
                '{:.4f}'.format(u['R@1']) if u else '-', '{:.4f}'.format(u['R@10']) if u else '-',
                '{:.4f}'.format(r['pred_r2_user']) if u else '-'))
    if boot:
        L += ['\n## Paired bootstrap over test users (run - baseline: align_F0 for frozen-ID runs, else align_C0)\n',
              '| run vs | score | metric | diff | 95% CI | p (one-sided, run > baseline) |', '|---|---|---|---|---|---|']
        for b in boot:
            L.append('| {} vs {} | {} | {} | {:+.4f} | [{:+.4f}, {:+.4f}] | {:.4f} |'.format(
                b['a'], b['b'], b['score'], b['metric'], b['mean'], b['ci_lo'], b['ci_hi'], b['p_one_sided']))
    text = '\n'.join(L) + '\n'
    open(os.path.join(out, 'summary.md'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
