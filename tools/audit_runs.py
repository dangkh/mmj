# coding: utf-8
"""
Audit table of finished runs, so that every number in the paper can be traced to a comparable protocol.

    python tools/audit_runs.py                                   # all finished runs in src/log
    python tools/audit_runs.py --base cloth sports --models DRAGON COHESION --csv audit.csv

One row per (model, data folder, run name), latest finished log:
    base | cond | model | folder | run | train seed | perm seed | graph (mm_image_weight) | hyper-parameters |
    best epoch | valid R@20 | test R@20 | log
Checks printed at the end:
  * HP mismatch: within one (base dataset, model), control runs whose selected hyper-parameters (seed excluded)
    differ from each other or from the <model>_grid run on the base dataset (not comparable).
  * graph protocol: *_tg runs must have mm_image_weight = 0, the others the model default.
  * n per condition (paired comparisons need the same seeds).
Checkpoint selection is the same for every run (best valid recall@20, MMRec early stopping).
"""
import argparse
import ast
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from collect_results import parse_log, TIME_RE   # noqa: E402

NAME_RE = re.compile(r'^(?P<model>[^-]+)-(?P<ds>[^-]+)-(?P<name>.+)-(?P<t>' + TIME_RE + r')\.log$')
SUFFIX_RE = re.compile(r'^(?P<base>.+?)_v(?P<mode>perm|noise64|noise|constn|const|ones|gauss|copy)'
                       r'(?P<p>_p\d+)?(?P<tg>_tg)?$')


def condition(folder):
    m = SUFFIX_RE.match(folder)
    if not m:
        return folder, 'real'
    mode = m.group('mode')
    if mode == 'copy':
        return m.group('base'), 'real_tg'
    mode = {'noise64': 'id'}.get(mode, mode)
    if m.group('p'):
        mode = 'perm' + m.group('p')[2:]
    return m.group('base'), mode + ('_tg' if m.group('tg') else '')


def params(row):
    """'['seed', 'lr']=(999, 0.001)' -> ({'lr': 0.001}, 999)"""
    p = row.get('params')
    if not p or '=' not in p:
        return {}, None
    keys, vals = p.split('=', 1)
    try:
        keys, vals = ast.literal_eval(keys.strip()), ast.literal_eval(vals.strip().rstrip(','))
    except (ValueError, SyntaxError):
        return {'raw': p}, None
    if not isinstance(vals, tuple):
        vals = (vals,)
    d = dict(zip(keys, vals))
    return {k: v for k, v in d.items() if k != 'seed'}, d.get('seed')


def config_value(text, key):
    m = re.search(r'^{}=(.*)$'.format(re.escape(key)), text, re.M)
    return m.group(1).strip() if m else ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_dir', default='src/log')
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--base', nargs='*', default=None, help='base datasets to keep (default: all)')
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--csv', default='audit_runs.csv')
    args = ap.parse_args()

    latest = {}
    for path in glob.glob(os.path.join(args.log_dir, '*.log')):
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        if args.models and m.group('model') not in args.models:
            continue
        base, cond = condition(m.group('ds'))
        if args.base and base not in args.base:
            continue
        key = (m.group('model'), m.group('ds'), m.group('name'))
        if key not in latest or os.path.getmtime(path) > os.path.getmtime(latest[key]):
            latest[key] = path

    rows = []
    for (model, ds, name), path in sorted(latest.items()):
        r = parse_log(path)
        if r.get('status') != 'done':
            continue
        text = open(path, encoding='utf-8', errors='ignore').read()
        base, cond = condition(ds)
        hp, seed = params(r)
        meta_f = os.path.join(args.data_dir, ds, 'control_meta.json')
        perm_seed = ''
        if cond.startswith('perm'):
            perm_seed = json.load(open(meta_f)).get('seed', '') if os.path.exists(meta_f) else '2026 (default)'
        rows.append({
            'base': base, 'cond': 'grid' if name.endswith('_grid') else cond, 'model': model, 'folder': ds,
            'run': name, 'train_seed': seed, 'perm_seed': perm_seed,
            'mm_image_weight': config_value(text, 'mm_image_weight'),
            'learn_v_feat': config_value(text, 'learn_v_feat'),
            'hp': json.dumps(hp, sort_keys=True), 'best_epoch': r.get('best_epoch', ''),
            'valid_R@20': r.get('valid_recall@20', ''), 'test_R@20': r.get('test_recall@20', ''), 'log': path})

    if not rows:
        print('no finished runs found')
        return
    cols = list(rows[0].keys())
    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- checks
    print('{} finished runs -> {}'.format(len(rows), args.csv))
    by = defaultdict(list)
    for r in rows:
        by[(r['base'], r['model'])].append(r)
    for (base, model), rs in sorted(by.items()):
        grid = [r for r in rs if r['cond'] == 'grid']
        runs = [r for r in rs if r['cond'] != 'grid']
        print('\n== {} / {}'.format(base, model))
        n = defaultdict(list)
        for r in runs:
            n[r['cond']].append(str(r['train_seed']))
        print('   n per condition: ' + ', '.join('{}={}({})'.format(c, len(s), ','.join(sorted(s)))
                                                  for c, s in sorted(n.items())))
        hps = defaultdict(list)
        for r in runs:
            hps[r['hp']].append('{}:{}'.format(r['cond'], r['train_seed']))
        ref = grid[0]['hp'] if grid else None
        if len(hps) > 1 or (ref and set(hps) != {ref}):
            print('   !! HP MISMATCH' + (' (grid selected {})'.format(ref) if ref else ''))
            for h, who in hps.items():
                print('      {}  <- {}'.format(h, ' '.join(sorted(who))))
        else:
            print('   hyper-parameters consistent: {}{}'.format(next(iter(hps)) if hps else '-',
                                                              ' (= grid)' if ref else ''))
        bad = [r for r in runs if (r['cond'].endswith('_tg') and r['mm_image_weight'] not in ('0.0', '0'))
               or (not r['cond'].endswith('_tg') and r['mm_image_weight'] in ('0.0', '0') and model != 'DRAGONID')]
        if bad:
            print('   !! GRAPH PROTOCOL: ' + ' '.join('{}:{}(w={})'.format(r['cond'], r['train_seed'],
                                                                        r['mm_image_weight']) for r in bad))


if __name__ == '__main__':
    main()
