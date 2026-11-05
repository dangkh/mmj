# coding: utf-8
"""
Summarise the content-vs-identity diagnosis written by scripts/submit_diag.sh.

    python tools/summarize_diag.py -d baby
    python tools/summarize_diag.py -d baby --seeds 999 1 2 --csv diag_baby.csv

For every model x condition: test Recall/NDCG@20 of the model (mean +/- std over seeds), the
difference to `real` paired by seed (mean, std, t), kNN-probe scores of the taps and the T2 graph
control (DRAGON). Missing runs are shown as '-'.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from collect_results import parse_log   # noqa: E402

CONDS = ['real', 'perm', 'noise', 'const', 'id',                       # full replacement (graph changes too)
         'real_tg', 'perm_tg', 'noise_tg', 'const_tg', 'id_tg']          # text-only item graph, branch varies
FOLDER = {'real': '{}', 'perm': '{}_vperm', 'noise': '{}_vnoise', 'const': '{}_vconst', 'id': '{}_vnoise64',
          'real_tg': '{}_vcopy', 'perm_tg': '{}_vperm_tg', 'noise_tg': '{}_vnoise_tg',
          'const_tg': '{}_vconst_tg', 'id_tg': '{}_vnoise64_tg'}
TAPS = ['T1_v', 'T1_t', 'T1_v+T1_t', 'T2_v', 'T2_t', 'T2_v+T2_t', 'T2_cat', 'T3']
GAINS = ['G[T1_v|T1_t]', 'G[T2_v|T2_t]']
GC = [('v | content+users - noise+users', 'T2 v content'), ('v | content+users - content+0users', 'T2 v user-ID'),
      ('t | content+users - noise+users', 'T2 t content'), ('t | content+users - content+0users', 'T2 t user-ID')]


def run_name(model, cond, seed):
    """Same naming as scripts/submit_diag.sh (DRAGON seed 999 keeps the names of the first runs)."""
    if model == 'DRAGON' and seed == 999:
        return {'id': 'dragon_vid', 'id_tg': 'dragon_vid', 'real_tg': 'dragon_txtgraph'}.get(cond, 'dragon_taps')
    low = model.lower()
    if cond == 'real_tg':
        return '{}_txtgraph_s{}'.format(low, seed)
    if cond in ('id', 'id_tg') and model != 'FREEDOM':
        return '{}_vid_s{}'.format(low, seed)
    return '{}_s{}'.format(low, seed)


def latest_log(model, ds, name):
    logs = sorted(glob.glob('src/log/{}-{}-{}-*.log'.format(model, ds, name)), key=os.path.getmtime)
    for p in reversed(logs):
        r = parse_log(p)
        if r.get('status') == 'done':
            return r
    return None


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, s, len(xs)


def fmt(st, sign=False):
    if st is None:
        return '-'
    m, s, n = st
    f = '{:+.4f}' if sign else '{:.4f}'
    return (f + '±{:.4f}').format(m, s) + ('' if n > 1 else '(n=1)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='baby')
    ap.add_argument('--seeds', type=int, nargs='+', default=[999, 1, 2])
    ap.add_argument('--models', nargs='+', default=['DRAGON', 'FREEDOM', 'MENTOR'])
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()

    rows = []
    for model in args.models:
        data = {}
        for cond in CONDS:
            ds = FOLDER[cond].format(args.dataset)
            for seed in args.seeds:
                name = run_name(model, cond, seed)
                rec = {'model': model, 'cond': cond, 'seed': seed, 'dataset': ds, 'name': name}
                lg = latest_log(model, ds, name)
                if lg:
                    rec['recall'] = lg.get('test_recall@20'); rec['ndcg'] = lg.get('test_ndcg@20')
                    rec['valid'] = lg.get('valid_recall@20')
                pj = 'log/probe_knn_{}_{}_{}.json'.format(ds, model, name)
                if os.path.exists(pj):
                    J = json.load(open(pj))
                    for t in TAPS:
                        if t in J['results']:
                            rec[t] = J['results'][t]['recall']
                    for g in GAINS:
                        if g in J['derived']:
                            rec[g] = J['derived'][g]['gain']
                gj = 'log/graph_control_{}_{}-{}-{}.json'.format(ds, model, ds, name)
                if os.path.exists(gj):
                    P = json.load(open(gj))['results'].get('_paired', {})
                    for k, lab in GC:
                        if k in P:
                            rec[lab] = P[k]['gain']
                data[(cond, seed)] = rec
                rows.append(rec)

        print('\n' + '=' * 100 + '\n{} on {}   (mean±std over seeds; Δ = paired by seed vs real)'.format(model, args.dataset))
        print('{:<9s} {:>18s} {:>18s} {:>18s} {:>8s}  {}'.format('cond', 'test R@20', 'Δ R@20 vs ref', 'test N@20', 't', 'seeds'))
        print('(ref = real for the first block and for real_tg; ref = real_tg for the other *_tg rows, same text graph)')
        for cond in CONDS:
            r = [data[(cond, s)].get('recall') for s in args.seeds]
            n = [data[(cond, s)].get('ndcg') for s in args.seeds]
            ref = 'real_tg' if cond.endswith('_tg') and cond != 'real_tg' else 'real'   # same item graph
            d = [data[(cond, s)].get('recall') - data[(ref, s)]['recall']
                 for s in args.seeds if data[(cond, s)].get('recall') is not None
                 and data[(ref, s)].get('recall') is not None]
            st = stats(d)
            t = st[0] / (st[1] / math.sqrt(st[2])) if st and st[2] > 1 and st[1] > 1e-9 else float('nan')
            have = [str(s) for s in args.seeds if data[(cond, s)].get('recall') is not None]
            print('{:<9s} {:>18s} {:>18s} {:>18s} {:>8s}  {}'.format(
                cond, fmt(stats(r)), fmt(st, True) if cond != 'real' else '', fmt(stats(n)),
                '' if cond == 'real' or math.isnan(t) else '{:+.1f}'.format(t), ','.join(have) or '-'))

        cols = [c for c in TAPS + GAINS + [lab for _, lab in GC]
                if any(data[(cd, s)].get(c) is not None for cd in CONDS for s in args.seeds)]
        if cols:
            print('\nkNN probe / graph control (mean over seeds)')
            print('{:<16s}'.format('') + ''.join('{:>10s}'.format(c) for c in CONDS))
            for c in cols:
                cells = ''
                for cond in CONDS:
                    st = stats([data[(cond, s)].get(c) for s in args.seeds])
                    cells += '{:>10s}'.format('-' if st is None else ('{:+.4f}' if c.startswith(('G[', 'T2 v', 'T2 t'))
                                                                    else '{:.4f}').format(st[0]))
                print('{:<16s}'.format(c) + cells)

    # DRAGONID (visual branch -> SCOPE-style item-ID branch), compared with DRAGON on the same folder/graph
    variants = [('dragonid (N(0,1))', 'dragonid_s{}'), ('dragonid (xavier)', 'dragonid_xav_s{}')]
    ds_tg = FOLDER['real_tg'].format(args.dataset)
    found = False
    for lab, pat in variants:
        rec = {s: latest_log('DRAGONID', ds_tg, pat.format(s)) for s in args.seeds}
        if not any(rec.values()):
            continue
        if not found:
            print('\n' + '=' * 100 + '\nDRAGONID on {} (text item graph); refs: DRAGON real_tg / real / const'.format(ds_tg))
            print('{:<20s} {:>18s} {:>18s} {:>18s} {:>18s}  {}'.format(
                'variant', 'test R@20', 'Δ vs real_tg', 'Δ vs real', 'Δ vs const', 'seeds'))
            found = True
        r = [rec[s].get('test_recall@20') if rec[s] else None for s in args.seeds]
        cells = []
        for ref in ('real_tg', 'real', 'const'):
            refd = {s: latest_log('DRAGON', FOLDER[ref].format(args.dataset), run_name('DRAGON', ref, s)) for s in args.seeds}
            d = [r[i] - refd[s]['test_recall@20'] for i, s in enumerate(args.seeds)
                 if r[i] is not None and refd[s] is not None]
            st = stats(d)
            t = st[0] / (st[1] / math.sqrt(st[2])) if st and st[2] > 1 and st[1] > 1e-9 else float('nan')
            cells.append(fmt(st, True) + ('' if math.isnan(t) else ' t={:+.1f}'.format(t)))
        print('{:<20s} {:>18s} {:>18s} {:>18s} {:>18s}  {}'.format(
            lab, fmt(stats(r)), *cells, ','.join(str(s) for i, s in enumerate(args.seeds) if r[i] is not None)))
        for i, s in enumerate(args.seeds):
            if r[i] is not None:
                rows.append({'model': 'DRAGONID', 'cond': lab, 'seed': s, 'dataset': ds_tg, 'name': pat.format(s),
                             'recall': r[i], 'ndcg': rec[s].get('test_ndcg@20'), 'valid': rec[s].get('valid_recall@20')})

    if args.csv:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print('\nSaved', args.csv)


if __name__ == '__main__':
    main()
