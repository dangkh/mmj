# coding: utf-8
"""
Table of the teacher study (scripts/teacher_modality.pbs): LightGCNC with the item embedding from ID, text,
visual or text+visual, with graph propagation (<src>_grid) and without (<src>_l0, n_layers = 0).

    python tools/summarize_teacher.py baby ml1m dbbook

Per (dataset, source, variant): the setting selected on valid (seed 999), its valid Recall@20, and test
Recall@20 / NDCG@20 as mean +- std over seeds {999, 1, 2} (seed 999 from the grid log, 1 and 2 from the
_s3 log). Output: paper/teacher_modality.md and .csv.
"""
import ast
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'src', 'log')
BASE_SOURCES = ('id', 'id128', 'text', 'visual', 'tv', 'idtext', 'idtext_free', 'idtext_none', 'idvisual', 'idtv')
SOURCES = tuple(b + g for b in BASE_SOURCES for g in ('', '_gtext', '_gvisual', '_gtv'))
GRAPH_LABEL = {'': '', '_gtext': ' + item graph (text)', '_gvisual': ' + item graph (visual)', '_gtv': ' + item graph (tv)'}
LABEL = {'id': 'ID only', 'id128': 'ID only (128-d)', 'text': 'Text only', 'visual': 'Visual only',
         'tv': 'Text + Visual', 'idtext': 'ID + Text (a_u)', 'idtext_free': 'ID + Text (DRAGON free w)',
         'idtext_none': 'ID + Text (no weights)', 'idvisual': 'ID + Visual (a_u)', 'idtv': 'ID + Text + Visual (a_u)'}
ALPHA = re.compile(r'user weight of the ID channel a_u: mean ([0-9.]+), sd ([0-9.]+)')
METRICS = ('recall@20', 'ndcg@20', 'recall@10', 'ndcg@10')

BLOCK = re.compile(r'Parameters:\s*(\[.*?\])=(\(.*?\)),\s*best valid:(.*?),\s*best test:(.*?)(?:\n|$)', re.S)
BEST = re.compile(r'BEST █+.*?Parameters:\s*(\[.*?\])=(\(.*?\)),\s*Valid:(.*?),\s*Test:(.*?)(?:\n|$)', re.S)


def metrics(s):
    return {k.lower(): float(v) for k, v in re.findall(r'([A-Za-z]+@\d+):\s*([0-9.]+)', s)}


def latest_log(ds, name):
    logs = sorted(glob.glob(os.path.join(LOG_DIR, 'LightGCNC-{}-{}-*.log'.format(ds, name))), key=os.path.getmtime)
    for path in reversed(logs):
        text = open(path, encoding='utf-8', errors='ignore').read()
        if 'All Over' in text:
            return text
    return None


def all_runs(text):
    """[(params dict, valid, test)] from the 'All Over' section."""
    i = text.rfind('All Over')
    out = []
    for keys, vals, va, te in BLOCK.findall(text[i:]):
        keys, vals = ast.literal_eval(keys), ast.literal_eval(vals)
        vals = vals if isinstance(vals, tuple) else (vals,)
        out.append((dict(zip(keys, vals)), metrics(va), metrics(te)))
    return out


def best_run(text):
    m = BEST.findall(text)
    if not m:
        return None
    keys, vals, va, te = m[-1]
    keys, vals = ast.literal_eval(keys), ast.literal_eval(vals)
    vals = vals if isinstance(vals, tuple) else (vals,)
    return dict(zip(keys, vals)), metrics(va), metrics(te)


def summarize(ds):
    rows = []
    for src in SOURCES:
        for var, vlabel in (('grid', 'graph (n_layers 1-3)'), ('l0', 'no graph (n_layers 0)')):
            g = latest_log(ds, '{}_{}'.format(src, var))
            if g is None:
                continue
            b = best_run(g)
            if b is None:
                continue
            params, va, te = b
            tests = [te]
            s3 = latest_log(ds, '{}_{}_s3'.format(src, var))
            if s3 is not None:
                tests += [t for _, _, t in all_runs(s3)]
            alphas = ALPHA.findall(g)
            base, g_suf = (src.split('_g')[0], '_g' + src.split('_g')[1]) if '_g' in src else (src, '')
            row = {'dataset': ds, 'source': LABEL[base] + GRAPH_LABEL[g_suf], 'variant': vlabel,
                   'a_u (last epoch logged)': '{} ± {}'.format(*alphas[-1]) if alphas else '-',
                   'setting': ' '.join('{}={}'.format(k, v) for k, v in params.items() if k != 'seed'),
                   'seeds': len(tests), 'valid R@20': va.get('recall@20', np.nan)}
            for m in METRICS:
                v = np.array([t.get(m, np.nan) for t in tests], dtype=float)
                row['test ' + m] = np.nanmean(v)
                row['sd ' + m] = np.nanstd(v) if len(v) > 1 else np.nan
            rows.append(row)
    return rows


def main():
    datasets = sys.argv[1:] or ['baby', 'ml1m', 'dbbook']
    rows = [r for ds in datasets for r in summarize(ds)]
    if not rows:
        print('no finished LightGCNC logs in', LOG_DIR)
        return
    df = pd.DataFrame(rows)
    out_dir = os.path.join(ROOT, 'paper')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'teacher_modality.csv'), index=False)
    lines = ['# Teacher study: item embedding from ID / text / visual (LightGCNC)\n',
             'Test metrics: mean +- std over seeds; setting selected on valid Recall@20 (seed 999).\n']
    for ds, g in df.groupby('dataset', sort=False):
        lines.append('\n## {}\n'.format(ds))
        lines.append('| source | variant | setting | seeds | valid R@20 | test R@20 | test N@20 | test R@10 | test N@10 | a_u (ID) |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|')
        for _, r in g.iterrows():
            def cell(m):
                mu, sd = r['test ' + m], r['sd ' + m]
                return '{:.4f} ± {:.4f}'.format(mu, sd) if np.isfinite(sd) else '{:.4f}'.format(mu)
            lines.append('| {} | {} | {} | {} | {:.4f} | {} | {} | {} | {} | {} |'.format(
                r['source'], r['variant'], r['setting'], r['seeds'], r['valid R@20'], cell('recall@20'),
                cell('ndcg@20'), cell('recall@10'), cell('ndcg@10'), r['a_u (last epoch logged)']))
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(out_dir, 'teacher_modality.md'), 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
