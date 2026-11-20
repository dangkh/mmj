# coding: utf-8
"""
E5: main table and geometry of the ID-only methods (LightGCN, SGL-ED/ND/RW, XSimGCL, DirectAU, SelfCF, JEPA-ID).

Every method = one or more grid experiments (taps saved with --save_taps=True, one folder per combination) and,
optionally, an experiment with the other seeds of its selected setting:
    selection: the seed-999 run with the best valid Recall@20 among the grid experiments
    seeds:     that run + the runs of the seed experiment (seeds 1, 2); mean +- sd over the available seeds
Measures, on the test split (full catalog; train + valid masked):
    Recall@20 / NDCG@20 overall, and Recall@20 per item-degree group (tail / mid / head = terciles of the items by
    training degree; recall of a user over his test items of the group, users with >= 1 such item)
    paired bootstrap over users (seed 999 runs) against L0 (LightGCN): overall and tail Recall@20
Geometry (seed 999 run; encoder embeddings G / G_user if the taps have them, else T3 / T3_user):
    alignment = mean ||f(u) - f(i)||^2 over training edges, uniformity (users, items) = log E exp(-2||x - y||^2),
    both on L2-normalised embeddings (Wang & Isola); effective rank; Spearman(norm, degree) of the items;
    over-smoothing: R^2 of a ridge map Y -> E_0 (layer 0), fitted on the train items of jepay_item_split.npz and
    measured on its eval items, per degree group (low R^2 = layer 0 hard to recover = more smoothing)

    python tools/geom_eval.py -d baby                                   # default method list (scripts/e3_*, e4_jepa)
    python tools/geom_eval.py -d baby --methods "jepa=LightGCNJEPA:smoke_jepa_u" --out_dir results/geom_smoke
    method spec: name=Model:exp1+exp2[:seed_exp]
Output: results/geom/<ds>/{results.json, summary.md}
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from probe_rec import load_data                  # noqa: E402
from consolidate import binary                   # noqa: E402
from jepay.metrics import paired_bootstrap       # noqa: E402

DEFAULT = [
    'L0=LightGCN:e3_lgn',
    'SGL-ED=SGL:e3_sgled+e3_sgled_drop:e3_sgled_s3',
    'SGL-ND=SGL:e3_sglnd:e3_sglnd_s3',
    'SGL-RW=SGL:e3_sglrw:e3_sglrw_s3',
    'XSimGCL=XSimGCL:e3_xsim:e3_xsim_s3',
    'DirectAU=DirectAU:e3_dau:e3_dau_s3',
    'SelfCF=SELFCFED_LGN:e3_selfcf:e3_selfcf_s3',
    'JEPA-U=LightGCNJEPA:e4u',
    'JEPA-H=LightGCNJEPA:e4h',
    'JEPA-T=LightGCNJEPA:e4t',
    'JEPA-N=LightGCNJEPA:e4n',
    'JEPA-edge=LightGCNJEPA:e4edge',
    'JEPA-VIC=LightGCNJEPA:e4vic',
    'JEPA-best=LightGCNJEPA:e4u+e4h+e4t+e4n+e4edge+e4vic:e4best_s3',
]
GROUPS = ('tail', 'mid', 'head')


def parse(spec):
    name, rest = spec.split('=', 1)
    parts = rest.split(':')
    return name, parts[0], parts[1].split('+'), (parts[2] if len(parts) > 2 else None)


def run_dirs(model, ds, exp):
    base = os.path.join(ROOT, 'taps', '{}-{}-{}'.format(model, ds, exp))
    out = []
    for f in [os.path.join(base, 'meta.json')] + sorted(glob.glob(os.path.join(base, 'combo*', 'meta.json'))):
        if os.path.isfile(f):
            m = json.load(open(f))
            out.append((os.path.dirname(f), m))
    return out


def pick(model, ds, exps, seed_exp):
    """(best seed-999 dir, meta), [dirs of every seed of that setting]."""
    cands = [(d, m) for e in exps for d, m in run_dirs(model, ds, e)
             if m.get('hyper_parameters', {}).get('seed') in (None, 999)]
    if not cands:
        return None, []
    best = max(cands, key=lambda dm: dm[1]['valid'].get('recall@20', -1))
    seeds = [best[0]]
    key = {k: v for k, v in best[1]['hyper_parameters'].items() if k != 'seed'}
    pool = [(d, m) for e in exps for d, m in run_dirs(model, ds, e)]          # seeds inside a grid exp (e.g. L0)
    if seed_exp:
        pool += run_dirs(model, ds, seed_exp)
    for d, m in pool:
        h = m.get('hyper_parameters', {})
        if d != best[0] and h.get('seed') not in (None, 999) and \
                all(str(h.get(k)) == str(v) for k, v in key.items()) and d not in seeds:
            seeds.append(d)
    return best, seeds


@torch.no_grad()
def rank_eval(U, Y, hist, gt, mask, group_of, device, k=20, batch=2048):
    users = np.where((np.asarray(gt.sum(1)).ravel() > 0) & (np.asarray(hist.sum(1)).ravel() > 0))[0]
    U, Y = torch.from_numpy(U).to(device), torch.from_numpy(Y).to(device)
    grp = torch.from_numpy(group_of).long().to(device)
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, device=device, dtype=torch.float32))
    idcg = disc.cumsum(0)
    rec, ndcg = np.zeros(len(users)), np.zeros(len(users))
    rec_g = np.full((3, len(users)), np.nan)
    gt, mask = gt.tocsr(), mask.tocsr()
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        sc = U[torch.from_numpy(ub).long().to(device)] @ Y.T
        mb = mask[ub].tocoo()
        sc[torch.from_numpy(mb.row).long().to(device), torch.from_numpy(mb.col).long().to(device)] = -float('inf')
        top = sc.topk(k, dim=1).indices
        gb = gt[ub].tocoo()
        G = torch.zeros_like(sc, dtype=torch.bool)
        G[torch.from_numpy(gb.row).long().to(device), torch.from_numpy(gb.col).long().to(device)] = True
        hits = G.gather(1, top).float()
        cnt = G.sum(1).float()
        rec[s:s + len(ub)] = (hits.sum(1) / cnt).cpu().numpy()
        ndcg[s:s + len(ub)] = ((hits * disc).sum(1) / idcg[(cnt.clamp(max=k) - 1).long()]).cpu().numpy()
        tg = grp[top]
        for g in range(3):
            cg = (G & (grp.view(1, -1) == g)).sum(1).float()
            hg = (hits * (tg == g).float()).sum(1)
            r = torch.where(cg > 0, hg / cg.clamp_min(1), torch.full_like(cg, float('nan')))
            rec_g[g, s:s + len(ub)] = r.cpu().numpy()
    return rec, ndcg, rec_g


def ridge_r2(X, Y, tr, ev, groups, alphas=(1e-3, 1e-2, 1e-1, 1, 10)):
    """R^2 of ridge X -> Y (fit on tr; alpha chosen on a fifth of tr) on ev, overall and per group."""
    mu_x, sd_x = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xs = (X - mu_x) / sd_x
    mu_y = Y[tr].mean(0)
    rng = np.random.default_rng(0)
    p = rng.permutation(tr)
    va, fit = p[:len(p) // 5], p[len(p) // 5:]

    def solve(idx, a):
        A = Xs[idx]
        return np.linalg.solve(A.T @ A + a * len(idx) * np.eye(A.shape[1]), A.T @ (Y[idx] - mu_y))

    def r2(idx, W):
        res = ((Xs[idx] @ W + mu_y - Y[idx]) ** 2).sum()
        tot = ((Y[idx] - Y[idx].mean(0)) ** 2).sum()
        return float(1 - res / max(tot, 1e-12))
    a = max(alphas, key=lambda a: r2(va, solve(fit, a)))
    W = solve(tr, a)
    out = {'all': r2(ev, W), 'alpha': a}
    for g, name in enumerate(GROUPS):
        idx = ev[groups[ev] == g]
        out[name] = r2(idx, W) if len(idx) > 5 else float('nan')
    return out


def geometry(R, R_tr, deg_i, split, groups):
    Yg = R.get('G', R['T3'])
    Ug = R.get('G_user', R['T3_user'])
    nz = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)  # noqa: E731
    rng = np.random.default_rng(0)
    coo = R_tr.tocoo()
    e = rng.choice(coo.nnz, min(coo.nnz, 50000), replace=False)
    align = float((np.linalg.norm(nz(Ug[coo.row[e]]) - nz(Yg[coo.col[e]]), axis=1) ** 2).mean())

    def unif(Z):
        z = nz(Z[rng.choice(len(Z), min(len(Z), 4096), replace=False)])
        d2 = np.maximum(2 - 2 * z @ z.T, 0)
        iu = np.triu_indices(len(z), 1)
        return float(np.log(np.exp(-2 * d2[iu]).mean()))

    def erank(Z):
        s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
        p = s / max(s.sum(), 1e-12)
        p = p[p > 0]
        return float(np.exp(-(p * np.log(p)).sum()))

    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    out = {'align': align, 'unif_user': unif(Ug), 'unif_item': unif(Yg), 'erank_item': erank(Yg),
           'erank_user': erank(Ug), 'spearman_norm_deg': spearman(np.linalg.norm(Yg, axis=1), deg_i)}
    if 'E0' in R:
        out['blur_r2'] = ridge_r2(Yg, R['E0'], np.sort(split['train']), np.sort(split['eval']), groups)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True)
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--methods', default='', help='";"-separated specs name=Model:exp1+exp2[:seed_exp]; default: E3 + E4')
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'results', 'geom'))
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    out = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out, exist_ok=True)
    specs = [s for s in args.methods.split(';') if s.strip()] or DEFAULT

    n_items = None
    for spec in specs:                                   # item count of the models (items with no interaction kept)
        name, model, exps, seed_exp = parse(spec.strip())
        best, _ = pick(model, args.dataset, exps, seed_exp)
        if best is not None:
            n_items = np.load(os.path.join(best[0], 'T3.npy'), mmap_mode='r').shape[0]
            break
    if n_items is None:
        raise SystemExit('no taps for any method')
    (R_tr, R_va, R_te), n_users, n_items = load_data(args.data_dir, args.dataset, min_items=n_items)
    deg_i = np.asarray(R_tr.sum(0)).ravel()
    order = np.argsort(deg_i, kind='stable')
    groups = np.zeros(n_items, dtype=np.int64)
    groups[order[n_items // 3: 2 * n_items // 3]] = 1
    groups[order[2 * n_items // 3:]] = 2
    cuts = [int(deg_i[order[n_items // 3]]), int(deg_i[order[2 * n_items // 3]])]
    split = np.load(os.path.join(args.data_dir, args.dataset, 'jepay_item_split.npz'))
    tst = (R_tr, R_te, binary(R_tr, R_va))

    res, per_user = {}, {}
    for spec in specs:
        name, model, exps, seed_exp = parse(spec.strip())
        best, dirs = pick(model, args.dataset, exps, seed_exp)
        if best is None:
            print('[geom] {}: no taps ({} {})'.format(name, model, exps), flush=True)
            continue
        rows = []
        for d in dirs:
            R = {f[:-4]: np.load(os.path.join(d, f)).astype(np.float32) for f in os.listdir(d) if f.endswith('.npy')}
            if R['T3'].shape[0] != n_items or R['T3_user'].shape[0] != n_users:
                print('[geom] {}: shape mismatch in {}'.format(name, d), flush=True)
                continue
            rec, ndcg, rec_g = rank_eval(R['T3_user'], R['T3'], *tst, groups, device)
            row = {'dir': os.path.relpath(d, ROOT), 'R@20': float(rec.mean()), 'N@20': float(ndcg.mean()),
                   'logged_test_R@20': json.load(open(os.path.join(d, 'meta.json')))['test'].get('recall@20')}
            for g, gn in enumerate(GROUPS):
                row['R@20_' + gn] = float(np.nanmean(rec_g[g]))
            if d == best[0]:
                row['geometry'] = geometry(R, R_tr, deg_i, split, groups)
                per_user[name] = (rec, rec_g[0])
            rows.append(row)
        if not rows:
            continue
        agg = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
               for k in ('R@20', 'N@20', 'R@20_tail', 'R@20_mid', 'R@20_head')}
        res[name] = {'model': model, 'setting': {k: v for k, v in best[1]['hyper_parameters'].items() if k != 'seed'},
                     'valid_R@20': best[1]['valid'].get('recall@20'), 'n_seeds': len(rows), 'mean_sd': agg,
                     'runs': rows, 'geometry': next((r['geometry'] for r in rows if 'geometry' in r), None)}
        print('[geom] {}: R@20 {:.4f} +- {:.4f} ({} seeds) | tail {:.4f}'.format(
            name, agg['R@20'][0], agg['R@20'][1], len(rows), agg['R@20_tail'][0]), flush=True)

    boot = []
    if 'L0' in per_user:
        r0, t0 = per_user['L0']
        for name, (r, t) in per_user.items():
            if name == 'L0':
                continue
            ok = ~np.isnan(t) & ~np.isnan(t0)
            boot.append({'method': name, 'scope': 'all', **paired_bootstrap(r - r0, n_boot=args.n_boot)})
            boot.append({'method': name, 'scope': 'tail', **paired_bootstrap(t[ok] - t0[ok], n_boot=args.n_boot)})
    json.dump({'results': res, 'bootstrap': boot, 'degree_cuts': cuts}, open(os.path.join(out, 'results.json'), 'w'),
              indent=1)
    write_summary(out, args.dataset, res, boot, cuts)


def write_summary(out, ds, res, boot, cuts):
    f = lambda ms: '{:.4f} ± {:.4f}'.format(*ms) if ms[1] > 0 else '{:.4f}'.format(ms[0])  # noqa: E731
    L = ['# E5: ID-only methods ({})\n'.format(ds),
         'Test, full catalog. Item degree groups (training degree): tail < {}, mid {}..{}, head >= {}. '
         'mean ± sd over seeds (999, 1, 2 when available); setting selected on valid (seed 999).\n'.format(
             cuts[0], cuts[0], cuts[1], cuts[1]),
         '| method | setting | seeds | R@20 | N@20 | R@20 tail | R@20 mid | R@20 head |', '|---|---|---|---|---|---|---|---|']
    for n, r in res.items():
        m = r['mean_sd']
        L.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            n, ' '.join('{}={}'.format(k, v) for k, v in r['setting'].items()), r['n_seeds'], f(m['R@20']),
            f(m['N@20']), f(m['R@20_tail']), f(m['R@20_mid']), f(m['R@20_head'])))
    L += ['\n## Geometry (seed 999 run)\n',
          '| method | align | unif user | unif item | erank item | erank user | Spearman(norm, deg) | '
          'R² Y→E0 all | tail | mid | head |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for n, r in res.items():
        g = r['geometry'] or {}
        b = g.get('blur_r2', {})
        L.append('| {} | {:.3f} | {:.3f} | {:.3f} | {:.1f} | {:.1f} | {:+.3f} | {} | {} | {} | {} |'.format(
            n, g.get('align', np.nan), g.get('unif_user', np.nan), g.get('unif_item', np.nan),
            g.get('erank_item', np.nan), g.get('erank_user', np.nan), g.get('spearman_norm_deg', np.nan),
            *('{:.3f}'.format(b[k]) if k in b else '-' for k in ('all', 'tail', 'mid', 'head'))))
    if boot:
        L += ['\n## Paired bootstrap over users vs L0 (seed 999; Recall@20)\n',
              '| method | scope | diff | 95% CI | p (one-sided, method > L0) |', '|---|---|---|---|---|']
        for b in boot:
            L.append('| {} | {} | {:+.4f} | [{:+.4f}, {:+.4f}] | {:.4f} |'.format(
                b['method'], b['scope'], b['mean'], b['ci_lo'], b['ci_hi'], b['p_one_sided']))
    text = '\n'.join(L) + '\n'
    open(os.path.join(out, 'summary.md'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
