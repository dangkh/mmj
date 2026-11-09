# coding: utf-8
"""
Every number of the paper, regenerated from the run logs in one command (single source of truth).

    python tools/make_paper_tables.py                     # defaults below -> paper/
    python tools/make_paper_tables.py --datasets baby sports cloth ml1m dbbook --models DRAGON FREEDOM COHESION \
        --key DRAGON:cloth COHESION:sports --out paper

Writes to <out>/ (CSV = full precision + provenance; MD = readable; TEX = paste into the paper):
  table1_full.{csv,md,tex}     MAIN TABLE (table*, two columns): per model x dataset, Real / Perm / Const R@20 and
                               N@20 and A / D / T in % of Real (paired t); multirow + booktabs
  table1_compact.{tex,csv}     same numbers in the R | P | C | dA | dV | dK layout (x1e-3), dataset panels stacked
                               (--groups), deltas from the displayed means so they add up exactly
  table1_main.{csv,md,tex}     compact variant: A / D / T only, three panels
  table2_robust.{csv,md}       key cells: original, graph-fixed (_tg), independent permutations, ones / constn,
                               per-condition tuning (_ct series), each with A / D / T
  fig1_scatter.{csv,pdf,png}   A (x) vs T (y) per cell, 95% CI over seeds, y = x reference
  table_datasets.{csv,md}      users / items / interactions / density + raw-feature kNN probe (image, text, null)
  tableA_raw.csv               absolute R@20 / N@20 mean, std, n of every condition of every cell (appendix)
  table2a_modality.{csv,md}    visual R/P/C x text real/constant (<ds>_tconst): A with / without text, visual,
                               text and dual knockout, for --modality_cases
  tableA_textoff.{csv,md}      text-off settings (<ds>_tconst, <ds>_vlm_tconst): A / D / T
  provenance.csv               the log file behind every run used above
Runs are matched by the same names as scripts/submit_diag.sh (tools/summarize_diag.run_name); paired
differences use the seeds present in both conditions.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_diag import FOLDER, latest_log, run_name   # noqa: E402

PRETTY_DS = {'baby': 'Baby', 'sports': 'Sports', 'cloth': 'Clothing', 'clothing': 'Clothing',
             'ml1m': 'ML-1M', 'dbbook': 'DBbook'}
SEEDS = [999, 1, 2, 3, 4]
PROV = []


def tcrit(df):
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:
        return {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}.get(df, 2.0)


def runs(model, folder, name_fn, metric='test_recall@20'):
    """{seed: value} of finished runs; name_fn(seed) -> run name."""
    out = {}
    for s in SEEDS:
        r = latest_log(model, folder, name_fn(s))
        if r and r.get(metric) is not None:
            out[s] = r[metric]
            PROV.append({'model': model, 'folder': folder, 'run': name_fn(s), 'seed': s, metric: r[metric]})
    return out


def cond_runs(model, base, cond, tag=None, metric='test_recall@20'):
    folder = FOLDER[cond].format(base)
    if tag:
        return runs(model, folder, lambda s: '{}{}_s{}'.format(model.lower(), tag, s), metric)
    return runs(model, folder, lambda s: run_name(model, cond, s), metric)


def paired(a, b, ref):
    """a - b over common seeds; returns dict with abs / pct of mean(ref) / t / sig / n."""
    common = sorted(set(a) & set(b))
    if not common:
        return None
    d = [a[s] - b[s] for s in common]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else float('nan')
    t = m / (sd / math.sqrt(n)) if n > 1 and sd > 1e-12 else float('nan')
    half = tcrit(n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
    base = sum(ref.values()) / len(ref) if ref else float('nan')
    return {'abs': m, 'pct': 100 * m / base, 'ci_pct': 100 * half / base if n > 1 else float('nan'),
            't': t, 'n': n, 'sig': (not math.isnan(t)) and abs(t) > tcrit(n - 1)}


def sdiff(a, b):
    """per-seed contrast a - b over matched seeds (dict), for difference-in-contrasts tests."""
    return {x: a[x] - b[x] for x in sorted(set(a or {}) & set(b or {}))}


def decompose(real, perm, const):
    return {'A': paired(real, perm, real), 'D': paired(perm, const, real), 'T': paired(real, const, real)}


def mean_std(v):
    if not v:
        return None, None, 0
    x = list(v.values())
    m = sum(x) / len(x)
    s = math.sqrt(sum((y - m) ** 2 for y in x) / (len(x) - 1)) if len(x) > 1 else 0.0
    return m, s, len(x)


def cell(e, bold=False, tex=False):
    if not e:
        return '·' if not tex else '--'
    s = '{:+.1f}'.format(e['pct'])
    if tex and s.startswith('-'):
        s = '$-$' + s[1:]
    if e['sig']:
        s += '$^\\dagger$' if tex else ' †'
    if bold:
        s = ('\\textbf{%s}' % s) if tex else '**{}**'.format(s)
    return s


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def flat(prefix, e):
    if not e:
        return {}
    return {'{}_{}'.format(prefix, k): (round(v, 6) if isinstance(v, float) else v) for k, v in e.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['baby', 'sports', 'cloth', 'ml1m', 'dbbook'])
    ap.add_argument('--models', nargs='+', default=['DRAGON', 'FREEDOM', 'COHESION'])
    ap.add_argument('--appendix_models', nargs='+', default=['MENTOR'])
    ap.add_argument('--key', nargs='+', default=['DRAGON:cloth', 'COHESION:sports'])
    ap.add_argument('--modality_cases', nargs='+', default=['DRAGON:baby', 'MENTOR:baby', 'FREEDOM:baby', 'DRAGON:cloth', 'COHESION:sports', 'COHESION:dbbook'],
                    help='Table 2(a): model:dataset cells that also have <dataset>_tconst runs')
    ap.add_argument('--textoff', nargs='+', default=['baby_tconst', 'baby_vlm_tconst', 'sports_vlm_tconst'])
    ap.add_argument('--out', default='paper')
    ap.add_argument('--full_order', nargs='+', default=['DRAGON', 'MENTOR', 'FREEDOM', 'COHESION'],
                    help='row order of the full table (by architectural family)')
    ap.add_argument('--std', action='store_true', help='full table: add +-std to the raw values')
    ap.add_argument('--seeds', type=int, nargs='+', default=[999, 1, 2],
                    help='seeds used everywhere (same n in every cell); e.g. 999 1 2 3 4')
    ap.add_argument('--groups', default='baby,sports,cloth;ml1m,dbbook',
                    help='compact table: dataset panels, ";" between panels (e.g. Amazon; non-Amazon)')
    args = ap.parse_args()
    global SEEDS
    SEEDS = list(args.seeds)
    os.makedirs(args.out, exist_ok=True)

    # ---------------- Table 1 + Figure 1 data + raw appendix
    main_rows, raw_rows = [], []
    res = {}
    for model in args.models + args.appendix_models:
        for ds in args.datasets:
            r = {c: cond_runs(model, ds, c) for c in ('real', 'perm', 'const')}
            if not r['real']:
                continue
            dec = decompose(r['real'], r['perm'], r['const'])
            res[(model, ds)] = dec
            m, s, n = mean_std(r['real'])
            row = {'model': model, 'dataset': ds, 'real_R@20': round(m, 5), 'real_std': round(s, 5), 'n_real': n}
            for k in 'ADT':
                row.update(flat(k, dec[k]))
            row['in_main_table'] = model in args.models
            main_rows.append(row)
        for ds in args.datasets:
            for c in FOLDER:
                for metric in ('test_recall@20', 'test_ndcg@20'):
                    v = cond_runs(model, ds, c, metric=metric)
                    if v:
                        m, s, n = mean_std(v)
                        raw_rows.append({'model': model, 'dataset': ds, 'cond': c, 'metric': metric,
                                         'mean': round(m, 5), 'std': round(s, 5), 'n': n,
                                         'seeds': ' '.join(map(str, sorted(v)))})
    write_csv(os.path.join(args.out, 'table1_main.csv'), main_rows)
    write_csv(os.path.join(args.out, 'tableA_raw.csv'), raw_rows)

    key = {tuple(k.split(':')) for k in args.key}
    ds_cols = [d for d in args.datasets if any((m, d) in res for m in args.models)]
    md = ['Table 1. Change in R@20 (% of Real), mean over seeds; † paired t-test p < .05.', '',
          '| | ' + ' | '.join(PRETTY_DS.get(d, d) for d in ds_cols) + ' |',
          '|---|' + '---|' * len(ds_cols)]
    tex = ['\\begin{tabular}{l' + 'r' * len(ds_cols) + '}', '\\toprule',
           ' & ' + ' & '.join(PRETTY_DS.get(d, d) for d in ds_cols) + ' \\\\', '\\midrule']
    panels = [('A', '(a) Alignment $A$ = Real $-$ Perm', '(a) Alignment A = Real − Perm'),
              ('D', '(b) Distinctiveness $D$ = Perm $-$ Const', '(b) Distinctiveness D = Perm − Const'),
              ('T', '(c) Total $T = A + D$ (modality ablation)', '(c) Total T = A + D (ablation)')]
    for k, ttex, tmd in panels:
        md.append('| ***{}*** |'.format(tmd) + ' |' * len(ds_cols))
        tex.append('\\multicolumn{%d}{l}{\\textit{%s}} \\\\' % (len(ds_cols) + 1, ttex))
        for model in args.models:
            cells = [cell(res.get((model, d), {}).get(k), bold=(model, d) in key) for d in ds_cols]
            md.append('| {} | {} |'.format(model, ' | '.join(cells)))
            tex.append('{} & {} \\\\'.format(model, ' & '.join(
                cell(res.get((model, d), {}).get(k), bold=(model, d) in key, tex=True) for d in ds_cols)))
        if k != 'T':
            tex.append('\\midrule')
    tex += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(args.out, 'table1_main.md'), 'w').write('\n'.join(md) + '\n')
    open(os.path.join(args.out, 'table1_main.tex'), 'w').write('\n'.join(tex) + '\n')

    # ---------------- Table 1 (full, two-column table*): raw R@20 / N@20 + A / D / T for every model x dataset
    full_models = args.models + [m for m in args.appendix_models if m not in args.models]
    full_models = [m for m in args.full_order if m in full_models] + [m for m in full_models if m not in args.full_order]
    fds = [d for d in args.datasets if any((m, d) in res for m in full_models)]
    mets = [('test_recall@20', 'R@20'), ('test_ndcg@20', 'N@20')]
    F, full_csv = {}, []
    for model in full_models:
        for ds in fds:
            for met, _ in mets:
                r = {c: cond_runs(model, ds, c, metric=met) for c in ('real', 'perm', 'const')}
                if not r['real']:
                    continue
                F[(model, ds, met)] = (r, decompose(r['real'], r['perm'], r['const']))
                row = {'model': model, 'dataset': ds, 'metric': met}
                for c in ('real', 'perm', 'const'):
                    m, sd, n = mean_std(r[c])
                    row.update({c: m, c + '_std': sd, c + '_n': n})
                for kk in 'ADT':
                    row.update(flat(kk, F[(model, ds, met)][1][kk]))
                full_csv.append(row)
    write_csv(os.path.join(args.out, 'table1_full.csv'), full_csv)

    def raw_cell(model, ds, met, c, tex):
        if (model, ds, met) not in F or not F[(model, ds, met)][0][c]:
            return '--' if tex else '·'
        m, sd, n = mean_std(F[(model, ds, met)][0][c])
        if not tex:
            return '{:.4f}'.format(m) + ('' if n >= 3 else ' (n={})'.format(n))
        best = max((mean_std(F[(model, ds, met)][0][x])[0] or -1) for x in ('real', 'perm', 'const'))
        s = '{:.4f}'.format(m)
        if args.std:
            s = '${}_{{\\pm{:.4f}}}$'.format(s, sd).replace('0.', '.', 1) if n > 1 else s
        if abs(m - best) < 1e-12:
            s = '\\underline{%s}' % s
        return s + ('$^{n=%d}$' % n if n < 3 else '')

    def eff_cell(model, ds, met, k, tex):
        e = F.get((model, ds, met), (None, {}))[1].get(k) if (model, ds, met) in F else None
        return cell(e, bold=(model, ds) in key and k in 'AT', tex=tex)

    labels = [('real', 'Real'), ('perm', 'Perm'), ('const', 'Const'),
              ('A', '$A$ (\\%)'), ('D', '$D$ (\\%)'), ('T', '$T$ (\\%)')]
    ncol = 2 + 2 * len(fds)
    tex = ['\\begin{table*}[t]', '\\centering', '\\footnotesize', '\\setlength{\\tabcolsep}{3.2pt}',
           '\\caption{Recall@20 and NDCG@20 with real, permuted (Perm) and constant (Const) visual features '
           '(mean over %s seeds; best of the three underlined), and the derived effects in %% of Real: alignment '
           '$A$ = Real $-$ Perm, distinctiveness $D$ = Perm $-$ Const, total $T = A + D$ (modality ablation). '
           '$^\\dagger$ paired $t$-test over seeds, $p<.05$. Bold: cases where ablation hides ($T\\approx0$, $A>0$) '
           'or inflates ($T \\gg A$) alignment.}' % '3',
           '\\label{tab:main}',
           '\\begin{tabular}{ll' + 'rr' * len(fds) + '}', '\\toprule',
           ' & & ' + ' & '.join('\\multicolumn{2}{c}{%s}' % PRETTY_DS.get(d, d) for d in fds) + ' \\\\',
           ' '.join('\\cmidrule(lr){%d-%d}' % (3 + 2 * i, 4 + 2 * i) for i in range(len(fds))),
           'Model & & ' + ' & '.join('R@20 & N@20' for _ in fds) + ' \\\\', '\\midrule']
    md = ['| Model | | ' + ' | '.join('{} R@20 | {} N@20'.format(PRETTY_DS.get(d, d), PRETTY_DS.get(d, d)) for d in fds) + ' |',
          '|---|---|' + '---|---|' * len(fds)]
    for mi, model in enumerate(full_models):
        if not any((model, d, 'test_recall@20') in F for d in fds):
            continue
        for li, (k, lab) in enumerate(labels):
            first = '\\multirow{6}{*}{%s}' % model if li == 0 else ''
            if k in ('real', 'perm', 'const'):
                tcells = [raw_cell(model, d, met, k, True) for d in fds for met, _ in mets]
                mcells = [raw_cell(model, d, met, k, False) for d in fds for met, _ in mets]
            else:
                tcells = [eff_cell(model, d, met, k, True) for d in fds for met, _ in mets]
                mcells = [eff_cell(model, d, met, k, False) for d in fds for met, _ in mets]
            if k == 'A':
                tex.append('\\cmidrule(lr){2-%d}' % ncol)
            tex.append('{} & {} & {} \\\\'.format(first, lab, ' & '.join(tcells)))
            md.append('| {} | {} | {} |'.format(model if li == 0 else '', lab.replace('$', '').replace('\\%', '%'),
                                               ' | '.join(mcells)))
        tex.append('\\midrule' if mi < len(full_models) - 1 else '\\bottomrule')
    tex += ['\\end{tabular}', '\\end{table*}']
    open(os.path.join(args.out, 'table1_full.tex'), 'w').write('\n'.join(tex) + '\n')
    open(os.path.join(args.out, 'table1_full.md'), 'w').write('\n'.join(md) + '\n')

    # ---------------- Table 1, compact R/P/C layout (x 1e-3, one row per model, dataset panels stacked)
    groups = [[d for d in g.split(',') if d] for g in args.groups.split(';')]
    cmp_rows = []
    tex = ['\\begin{table*}[t]', '\\centering',
           '\\caption{Recall@20 ($\\times 10^{-3}$, mean over %d seeds) with real (R), permuted (P) and constant (C) '
           'visual features. $\\Delta_A = R-P$: alignment contrast; $\\Delta_V = P-C$: variation (distinctiveness) '
           'contrast; $\\Delta_K = R-C = \\Delta_A+\\Delta_V$: knockout effect. $^\\dagger$: paired $t$-test over seeds, '
           '$p<.05$. --: not run.}' % len(SEEDS), '\\label{tab:main}', '\\scriptsize', '\\setlength{\\tabcolsep}{2.2pt}']
    for gi, g in enumerate(groups):
        tex.append('\\begin{tabular}{l*{%d}{rrrrrr}}' % len(g))
        tex.append('\\toprule')
        tex.append(' & ' + ' & '.join('\\multicolumn{6}{c}{\\textbf{%s}}' % PRETTY_DS.get(d, d) for d in g) + ' \\\\')
        tex.append(' '.join('\\cmidrule(lr){%d-%d}' % (2 + 6 * i, 7 + 6 * i) for i in range(len(g))))
        tex.append('\\textbf{Model} & ' + ' & '.join('R & P & C & $\\Delta_A$ & $\\Delta_V$ & $\\Delta_K$' for _ in g) + ' \\\\')
        tex.append('\\midrule')
        for model in full_models:
            cells = []
            for d in g:
                key3 = (model, d, 'test_recall@20')
                if key3 not in F or not all(F[key3][0][c] for c in ('real', 'perm', 'const')):
                    cells += ['--'] * 6
                    continue
                r, dec = F[key3]
                # deltas from the displayed (rounded) means, so that R-P, P-C, R-C add up exactly in the table
                R_, P_, C_ = (round(1000 * mean_std(r[c])[0], 1) for c in ('real', 'perm', 'const'))
                vals = [(R_ - P_, dec['A']), (P_ - C_, dec['D']), (R_ - C_, dec['T'])]
                dc = []
                for (v, e), lab in zip(vals, 'ADT'):
                    t = '{:+.1f}'.format(v).replace('-', '$-$')
                    if e and e['sig']:
                        t += '$^\\dagger$'
                    if (model, d) in key and lab in 'AD':
                        t = '\\textbf{%s}' % t
                    dc.append(t)
                cells += ['{:.1f}'.format(R_), '{:.1f}'.format(P_), '{:.1f}'.format(C_)] + dc
                cmp_rows.append({'model': model, 'dataset': d, 'R': R_, 'P': P_, 'C': C_, 'dA': round(R_ - P_, 1),
                                 'dV': round(P_ - C_, 1), 'dK': round(R_ - C_, 1),
                                 'sigA': bool(dec['A'] and dec['A']['sig']), 'sigV': bool(dec['D'] and dec['D']['sig']),
                                 'sigK': bool(dec['T'] and dec['T']['sig']),
                                 'n': min(len(r[c]) for c in ('real', 'perm', 'const'))})
            if any(c != '--' for c in cells):
                tex.append('{} & {} \\\\'.format(model, ' & '.join(cells)))
        tex.append('\\bottomrule')
        tex.append('\\end{tabular}')
        if gi < len(groups) - 1:
            tex.append('\\\\[3pt]')
    tex.append('\\end{table*}')
    open(os.path.join(args.out, 'table1_compact.tex'), 'w').write('\n'.join(tex) + '\n')
    write_csv(os.path.join(args.out, 'table1_compact.csv'), cmp_rows)

    # ---------------- Figure 1: A vs T
    fig_rows = []
    for (model, ds), dec in res.items():
        if model in args.models and dec['A'] and dec['T']:
            fig_rows.append({'model': model, 'dataset': ds, 'A_pct': round(dec['A']['pct'], 3),
                             'A_ci': round(dec['A']['ci_pct'], 3), 'T_pct': round(dec['T']['pct'], 3),
                             'T_ci': round(dec['T']['ci_pct'], 3),
                             'D_pct': round(dec['D']['pct'], 3) if dec['D'] else ''})
    write_csv(os.path.join(args.out, 'fig1_scatter.csv'), fig_rows)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']          # validated categorical order
        markers = {'baby': 'o', 'sports': 's', 'cloth': '^', 'clothing': '^', 'ml1m': 'D', 'dbbook': 'P'}
        fig, ax = plt.subplots(figsize=(3.4, 3.6))
        lo = min([r['A_pct'] - (r['A_ci'] or 0) for r in fig_rows] + [r['T_pct'] - (r['T_ci'] or 0) for r in fig_rows] + [0]) - 1
        hi = max([r['A_pct'] + (r['A_ci'] or 0) for r in fig_rows] + [r['T_pct'] + (r['T_ci'] or 0) for r in fig_rows] + [0]) + 1
        ax.plot([lo, hi], [lo, hi], color='#9a9a92', lw=1, ls='--', zorder=1)
        ax.axhline(0, color='#d6d5cf', lw=0.8, zorder=0)
        ax.axvline(0, color='#d6d5cf', lw=0.8, zorder=0)
        for r in fig_rows:
            c = colors[args.models.index(r['model']) % len(colors)]
            ax.errorbar(r['A_pct'], r['T_pct'], xerr=r['A_ci'] if r['A_ci'] == r['A_ci'] else None,
                        yerr=r['T_ci'] if r['T_ci'] == r['T_ci'] else None, fmt=markers.get(r['dataset'], 'o'),
                        color=c, mec='#1a1a19', mew=0.6, ms=6, elinewidth=1, capsize=0, zorder=3)
            if (r['model'], r['dataset']) in key:
                ax.annotate('{} {}'.format(r['model'], PRETTY_DS.get(r['dataset'], r['dataset'])),
                            (r['A_pct'], r['T_pct']), textcoords='offset points', xytext=(5, 4), fontsize=7,
                            color='#3d3d3a')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel('Alignment A (% of Real R@20)', fontsize=8)
        ax.set_ylabel('Ablation T (% of Real R@20)', fontsize=8)
        ax.tick_params(labelsize=7)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker='o', ls='', color=colors[i % len(colors)], mec='#1a1a19', mew=0.6, label=m)
             for i, m in enumerate(args.models)]
        h += [Line2D([], [], marker=markers.get(d, 'o'), ls='', color='#c3c2b7', mec='#1a1a19', mew=0.6,
                     label=PRETTY_DS.get(d, d)) for d in ds_cols]
        ax.legend(handles=h, fontsize=6, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4,
                  handletextpad=0.2, columnspacing=0.8)
        ax.text(hi, hi, ' T = A', fontsize=6, color='#6b6b65', ha='right', va='bottom')
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(args.out, 'fig1_scatter.' + ext), dpi=300)
    except Exception as e:   # noqa: BLE001
        print('[fig1] matplotlib figure skipped:', e)

    # ---------------- Table 2: robustness of key cells
    rob = []
    for k in args.key:
        model, ds = k.split(':')
        real = cond_runs(model, ds, 'real')
        variants = [('original', real, cond_runs(model, ds, 'perm'), cond_runs(model, ds, 'const'))]
        rt = cond_runs(model, ds, 'real_tg')
        variants.append(('graph-fixed (text item graph)', rt, cond_runs(model, ds, 'perm_tg'), cond_runs(model, ds, 'const_tg')))
        perms = [('perm', cond_runs(model, ds, 'perm'))] + [('perm%d' % i, cond_runs(model, ds, 'perm%d' % i)) for i in range(2, 6)]
        perms = [(n, p) for n, p in perms if p]
        const = cond_runs(model, ds, 'const')
        for n, p in perms[1:]:
            variants.append(('permutation ' + n, real, p, const))
        nz = cond_runs(model, ds, 'noise')
        if nz:
            variants.append(('moment-matched noise instead of Perm', real, nz, const))
        for cname in ('ones', 'constn'):
            c = cond_runs(model, ds, cname)
            if c:
                variants.append(('constant = ' + cname, real, cond_runs(model, ds, 'perm'), c))
        # direct test of original vs graph-fixed: delta = contrast_original - contrast_fixed, paired over seeds
        pp, cc = cond_runs(model, ds, 'perm'), cond_runs(model, ds, 'const')
        pt, ctg = cond_runs(model, ds, 'perm_tg'), cond_runs(model, ds, 'const_tg')
        if rt and pt and ctg:
            dG = {'A': paired(sdiff(real, pp), sdiff(rt, pt), real), 'D': paired(sdiff(pp, cc), sdiff(pt, ctg), real),
                  'T': paired(sdiff(real, cc), sdiff(rt, ctg), real)}
            dG_row = ('delta_G = original - graph-fixed (direct test)', dG)
        else:
            dG_row = None
        ct = {c: cond_runs(model, ds, c, tag='_ct') for c in ('real', 'perm', 'const')}
        if ct['perm'] or ct['const']:
            variants.append(('per-condition tuning', ct['real'] or real, ct['perm'], ct['const']))
        for name, r_, p_, c_ in variants:
            if not r_:
                continue
            dec = decompose(r_, p_, c_)
            row = {'cell': k, 'variant': name}
            for kk in 'ADT':
                row.update(flat(kk, dec[kk]))
            rob.append(row)
            if name.startswith('graph-fixed') and dG_row:
                row = {'cell': k, 'variant': dG_row[0]}
                for kk in 'ADT':
                    row.update(flat(kk, dG_row[1][kk]))
                rob.append(row)
        if len(perms) >= 2:   # across-permutation summary (permutation-level inference)
            As = [paired(real, p, real) for _, p in perms]
            Ds = [paired(p, const, real) for _, p in perms]
            row = {'cell': k, 'variant': 'across {} permutations'.format(len(perms))}
            for lab, xs in (('A', As), ('D', Ds)):
                v = [x['pct'] for x in xs if x]
                if len(v) >= 2:
                    m = sum(v) / len(v)
                    sd = math.sqrt(sum((y - m) ** 2 for y in v) / (len(v) - 1))
                    row.update({'{}_pct'.format(lab): round(m, 3), '{}_sd_perm'.format(lab): round(sd, 3),
                                '{}_n_pos'.format(lab): sum(1 for y in v if y > 0)})
            rob.append(row)
    write_csv(os.path.join(args.out, 'table2_robust.csv'), rob)
    md = ['Table 2. Robustness of the key cells (% of Real R@20; † p < .05).', '',
          '| cell | variant | A | D | T |', '|---|---|---|---|---|']
    for r in rob:
        def g(kk):
            if '{}_pct'.format(kk) not in r:
                return '·'
            s = '{:+.1f}'.format(r['{}_pct'.format(kk)])
            if r.get('{}_sig'.format(kk)):
                s += ' †'
            if '{}_sd_perm'.format(kk) in r:
                s += ' ± {:.1f} ({} of {} > 0)'.format(r['{}_sd_perm'.format(kk)], r['{}_n_pos'.format(kk)],
                                                        r['variant'].split()[1])
            return s
        md.append('| {} | {} | {} | {} | {} |'.format(r['cell'], r['variant'], g('A'), g('D'), g('T')))
    open(os.path.join(args.out, 'table2_robust.md'), 'w').write('\n'.join(md) + '\n')

    # ---------------- Table 2(a): modality availability, visual R/P/C x text real / constant (<ds>_tconst)
    mav = []
    for k in args.modality_cases:
        model, ds = k.split(':')
        M = {}
        for tlab, tds in (('T', ds), ('noT', ds + '_tconst')):
            for c in ('real', 'perm', 'const'):
                M[(c, tlab)] = cond_runs(model, tds, c)
        if not M[('real', 'T')] or not M[('real', 'noT')]:
            continue
        ref = M[('real', 'T')]
        row = {'case': k}
        for c in ('real', 'perm', 'const'):
            for t in ('T', 'noT'):
                m_, sd_, n_ = mean_std(M[(c, t)])
                row['{}|{}'.format(c, t)] = round(m_, 5) if m_ is not None else ''
        row.update(flat('A_text', paired(M[('real', 'T')], M[('perm', 'T')], ref)))
        row.update(flat('A_notext', paired(M[('real', 'noT')], M[('perm', 'noT')], ref)))
        row.update(flat('visualKO', paired(M[('real', 'T')], M[('const', 'T')], ref)))
        row.update(flat('textKO', paired(M[('real', 'T')], M[('real', 'noT')], ref)))
        row.update(flat('dualKO', paired(M[('real', 'T')], M[('const', 'noT')], ref)))
        # difference in alignment contrasts, tested on matched seed-level differences
        row.update(flat('interaction', paired(sdiff(M[('real', 'noT')], M[('perm', 'noT')]),
                                              sdiff(M[('real', 'T')], M[('perm', 'T')]), ref)))
        mav.append(row)
    write_csv(os.path.join(args.out, 'table2a_modality.csv'), mav)
    md = ['Table 2(a). Modality availability (% of Real R@20 with text; † p < .05). A = R - P under each text condition;'
          ' KO = Real minus the knocked-out setting.', '',
          '| case | A (text real) | A (text constant) | interaction = A(text const) - A(text real) | visual KO | text KO | dual KO |',
          '|---|---|---|---|---|---|---|']
    for r in mav:
        f = lambda kk: '{:+.1f}{}'.format(r['{}_pct'.format(kk)], ' †' if r.get('{}_sig'.format(kk)) else '') \
            if '{}_pct'.format(kk) in r else '·'
        md.append('| {} | {} | {} | {} | {} | {} | {} |'.format(r['case'], f('A_text'), f('A_notext'), f('interaction'),
                                                              f('visualKO'), f('textKO'), f('dualKO')))
    open(os.path.join(args.out, 'table2a_modality.md'), 'w').write('\n'.join(md) + '\n')

    # ---------------- text-off appendix
    to = []
    for ds in args.textoff:
        for model in args.models + args.appendix_models:
            r = {c: cond_runs(model, ds, c) for c in ('real', 'perm', 'const')}
            if not r['real']:
                continue
            dec = decompose(r['real'], r['perm'], r['const'])
            row = {'setting': ds, 'model': model, 'real_R@20': round(mean_std(r['real'])[0], 5)}
            for kk in 'ADT':
                row.update(flat(kk, dec[kk]))
            to.append(row)
    write_csv(os.path.join(args.out, 'tableA_textoff.csv'), to)
    md = ['| setting | model | Real | A | D | T |', '|---|---|---|---|---|---|']
    for r in to:
        f = lambda kk: '{:+.1f}{}'.format(r['{}_pct'.format(kk)], ' †' if r.get('{}_sig'.format(kk)) else '') \
            if '{}_pct'.format(kk) in r else '·'
        md.append('| {} | {} | {:.4f} | {} | {} | {} |'.format(r['setting'], r['model'], r['real_R@20'], f('A'), f('D'), f('T')))
    open(os.path.join(args.out, 'tableA_textoff.md'), 'w').write('\n'.join(md) + '\n')

    # ---------------- dataset table
    dsr = []
    for ds in args.datasets:
        p = os.path.join('data', ds, ds + '.inter')
        if not os.path.exists(p):
            continue
        users, items, n = set(), set(), 0
        with open(p) as f:
            head = f.readline().rstrip('\n').split('\t')
            iu, ii = head.index('userID'), head.index('itemID')
            for line in f:
                x = line.rstrip('\n').split('\t')
                users.add(x[iu]); items.add(x[ii]); n += 1
        row = {'dataset': PRETTY_DS.get(ds, ds), 'users': len(users), 'items': len(items), 'interactions': n,
               'density_%': round(100 * n / (len(users) * len(items)), 3)}
        cands = sorted(glob.glob('log/probe_knn_{}_*.json'.format(ds)), key=os.path.getmtime, reverse=True)
        t0 = lambda p: p.lower().endswith('_t0.json')          # default tag is 'T0'
        cands = [p for p in cands if t0(p)] + [p for p in cands if not t0(p)]
        for pj in cands:
            J = json.load(open(pj))
            if J.get('args', {}).get('dataset') != ds:        # e.g. probe_knn_baby_vlm_* also matches baby_*
                continue
            R = J.get('results', {})
            if 'x_v' in R and 'x_t' in R:
                D = J.get('derived', {}) or {}
                G = D.get('G[x_v|x_t]')
                # current probe_knn: {'gain': g, 'ci': [lo, hi]}; older files stored a bare float (+ maybe '<key>_ci')
                if isinstance(G, dict):
                    g, ci = G.get('gain'), G.get('ci')
                    if G.get('p_boot') is not None:
                        row['_p_G'] = G['p_boot']
                elif isinstance(G, (int, float)):
                    g, ci = float(G), D.get('G[x_v|x_t]_ci') or D.get('G[x_v|x_t]_CI')
                else:
                    g, ci = None, None
                if g is None and 'x_v+x_t' in R:          # derive it from the results if needed (no CI then)
                    g = R['x_v+x_t']['recall'] - R['x_t']['recall']
                if g is not None:
                    sig = isinstance(ci, (list, tuple)) and len(ci) == 2 and (ci[0] > 0 or ci[1] < 0)
                    row['G_v|t'] = '{:+.4f}{}'.format(g, '*' if sig else ('' if ci else ' (no CI)'))
                row.update({'knn_image': R['x_v']['recall'], 'knn_text': R['x_t']['recall'],
                            'knn_null': R.get('null64', {}).get('recall', ''), 'knn_pop': R.get('pop', {}).get('recall', ''),
                            'probe_file': pj})
                break
        for pj in sorted(glob.glob('log/probe_knn_{}_*.json'.format(ds)), key=os.path.getmtime, reverse=True):
            J = json.load(open(pj))                             # permuted-image sanity probe (tag t0perm)
            if J.get('args', {}).get('dataset') == ds and 'vperm' in J.get('results', {}):
                row['knn_vperm'] = J['results']['vperm']['recall']
                break
        dsr.append(row)
    # Holm correction of the G[v|t] tests across datasets (only when every probe file carries a bootstrap p)
    ps = [(i, r['_p_G']) for i, r in enumerate(dsr) if '_p_G' in r]
    if ps and len(ps) == sum(1 for r in dsr if 'G_v|t' in r):
        m = len(ps)
        adj, run = {}, 0.0
        for j, (i, pv) in enumerate(sorted(ps, key=lambda x: x[1])):
            run = max(run, min(1.0, (m - j) * pv))
            adj[i] = run
        for i, r in enumerate(dsr):
            if i in adj:
                r['G_p_boot'] = round(r['_p_G'], 4)
                r['G_p_holm'] = round(adj[i], 4)
                r['G_v|t'] = r['G_v|t'].rstrip('*') + (' (Holm p={:.3f})'.format(adj[i]))
    for r in dsr:
        r.pop('_p_G', None)
    write_csv(os.path.join(args.out, 'table_datasets.csv'), dsr)
    md = ['| dataset | users | items | interactions | density % | kNN image | kNN perm-image | kNN text | kNN null | pop | G[v|t] |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in dsr:
        fmt = lambda v: '{:.4f}'.format(v) if isinstance(v, float) else (str(v) if v != '' else '·')
        md.append('| {} | {:,} | {:,} | {:,} | {} | {} | {} | {} | {} | {} | {} |'.format(
            r['dataset'], r['users'], r['items'], r['interactions'], r['density_%'], fmt(r.get('knn_image', '')),
            fmt(r.get('knn_vperm', '')), fmt(r.get('knn_text', '')), fmt(r.get('knn_null', '')), fmt(r.get('knn_pop', '')), r.get('G_v|t', '·')))
    open(os.path.join(args.out, 'table_datasets.md'), 'w').write('\n'.join(md) + '\n')

    write_csv(os.path.join(args.out, 'provenance.csv'), PROV)
    print('wrote', ', '.join(sorted(os.listdir(args.out))), 'to', args.out + '/')
    print(open(os.path.join(args.out, 'table1_main.md')).read())


if __name__ == '__main__':
    main()
