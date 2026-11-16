# coding: utf-8
"""
Summarise the JEPA-Y E-A probes (results/jepay_diag/<ds>/{runs,probes}) into the tables of diagnosis.md.

    python tools/summarize_jepay_diag.py -d baby                       # one dataset
    python tools/summarize_jepay_diag.py --all baby ml1m dbbook        # + Holm across datasets, gate, figures

All comparisons are on eval items: per-item squared errors averaged over seeds, paired bootstrap over items
(10 000), dMSE = MSE(reference) - MSE(candidate) (> 0: candidate better).

M1 (H1)   text-only probe vs text + visual probe, same head capacity. Reference: the probe on the run's own
          frozen h_t or on A0's h_t, whichever has the lower VAL MSE (model level). Conditions A2, A1, A2sub,
          A2noJ, A2noY (visual input: Z_v; A1: h_v).
Fixed-h_t comparison (A2 runs, same frozen h_t): T+Z_v vs T+h_v (M2-int), T+PCA_64(V) (M2-raw), T+Z_lin (A3),
          T+S_v, T+[S_v;Z_v].
M2-sup (H2) T+h_v of A1 (same supervision, no decomposition) vs T+Z_v of A2, per probe head and data fraction.
Ablations A2sub / A2noJ / A2noY vs A2 (T+Z_v probes).
Descriptive (not criteria): M3, M4, M5, M6, M7, M8.
Gate (--all): M1 of A2, head mlp1: CI lower bound > 0 and Holm p < .05 on >= --gate_k datasets (default 2 of 3).

Output: paper/jepay_diag_<ds>.md (+ csv), with --all paper/jepay_diag.md, paper/jepay_diag_m1.csv,
paper/fig_jepay_h2.pdf, paper/fig_jepay_h3.pdf.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from jepay.metrics import paired_bootstrap, holm   # noqa: E402

HEADS = ('lin', 'mlp1', 'mlp2')
VIS_SET = {'A1': 'T+hv', 'A2': 'T+Z', 'A2sub': 'T+Z', 'A2noJ': 'T+Z', 'A2noY': 'T+Z'}
FIXED_HT = [('T+Z', 'Z_v (JEPA-Y)'), ('T+hv', 'h_v (A1-int)'), ('T+PCA', 'PCA64(V) (A1-raw)'),
            ('T+Zlin', 'Z_lin (A3)'), ('T+S', 'S_v'), ('T+SZ', '[S_v;Z_v]')]


class Store(object):
    def __init__(self, res_dir):
        self.probes, self.runs = {}, {}
        for f in sorted(glob.glob(os.path.join(res_dir, 'probes', '*.json'))):
            r = json.load(open(f))
            z = np.load(f[:-5] + '.npz')
            r['_mse'] = {k: z[k] for k in z.files}
            self.probes[(r['cond'], r['frac'], r['seed'])] = r
        for f in sorted(glob.glob(os.path.join(res_dir, 'runs', '*.json'))):
            r = json.load(open(f))
            self.runs[(r['cond'], r['frac'], r['seed'])] = r

    def pick(self, cond, frac=1.0):
        return [self.probes[k] for k in sorted(self.probes) if k[0] == cond and abs(k[1] - frac) < 1e-9]

    def runs_of(self, cond, frac=1.0):
        return [self.runs[k] for k in sorted(self.runs) if k[0] == cond and abs(k[1] - frac) < 1e-9]


def has(rs, key):
    return bool(rs) and all(key in r['_mse'] for r in rs)


def item_mean(rs, key, rows):
    return np.mean([r['_mse'][key][rows] for r in rs], 0)


def val_mean(rs, key):
    return float(np.mean([r['probes'][key]['mse_val'] for r in rs]))


def ref_text(S, cond, head, frac):
    """(runs, key, source) of the stricter text-only reference, chosen on val MSE."""
    key = 'T|' + head
    cands = []
    own = S.pick(cond, frac)
    if has(own, key):
        cands.append((val_mean(own, key), own, cond))
    if frac == 1.0:
        a0 = S.pick('A0', 1.0)
        if has(a0, key):
            cands.append((val_mean(a0, key), a0, 'A0'))
    if not cands:
        return None, None, None
    _, rs, src = min(cands, key=lambda c: c[0])
    return rs, key, src


def compare(rs_a, key_a, rs_b, key_b, rows, n_boot):
    """dMSE = MSE(a) - MSE(b), bootstrap over items (b better when > 0)."""
    a, b = item_mean(rs_a, key_a, rows), item_mean(rs_b, key_b, rows)
    out = paired_bootstrap(a - b, n_boot=n_boot)
    out.update({'mse_ref': float(a.mean()), 'mse_cand': float(b.mean()),
                'pct': 100 * out['mean'] / max(float(a.mean()), 1e-12)})
    return out


def get(m, path):
    for p in path.split('.'):
        if not isinstance(m, dict) or p not in m:
            return np.nan
        m = m[p]
    return m


DESC = [('M4 R2(h_v<-h_t)', 'rep_metrics.M4_hv_from_ht.r2'), ('M4 R2(S_v<-h_t)', 'rep_metrics.M4_s_from_ht.r2'),
        ('M4 R2(Z_v<-h_t)', 'rep_metrics.M4_z_from_ht.r2'),
        ('M5 D(S,Z)', 'rep_metrics.M5_D_s_z'), ('M5 D(P(h_t),Z)', 'rep_metrics.M5_D_shat_z'),
        ('M5 V<-S', 'rep_metrics.M5_probe.s.r2'), ('M5 V<-Z', 'rep_metrics.M5_probe.z.r2'),
        ('M5 V<-[S;Z]', 'rep_metrics.M5_probe.s_z.r2'), ('M5 V<-[P(h_t);Z]', 'rep_metrics.M5_probe.shat_z.r2'),
        ('M5 V<-[h_t;Z]', 'rep_metrics.M5_probe.ht_z.r2'), ('M5 V<-h_t', 'rep_metrics.M5_probe.ht.r2'),
        ('M6 erank Z', 'rep_metrics.M6_z.erank'), ('M6 erank S', 'rep_metrics.M6_s.erank'),
        ('M6 std Z', 'rep_metrics.M6_z.mean_std'), ('|corr(S,Z)|', 'rep_metrics.corr_s_z'),
        ('|corr(Z,h_t)|', 'rep_metrics.corr_z_ht'), ('k', 'cfg.k')]


def summarize_dataset(ds, args):
    res_dir = os.path.join(args.out_dir, ds)
    S = Store(res_dir)
    if not S.probes:
        print('[summary] no probes for', ds)
        return None
    ev = np.load(os.path.join(args.data_dir, ds, 'jepay_item_split.npz'))['eval']
    rd = lambda f: json.load(open(os.path.join(res_dir, f))) if os.path.isfile(os.path.join(res_dir, f)) else {}  # noqa
    ceiling, pca_ex = rd('teacher_ceiling.json'), rd('pca_explained.json')
    nb = args.n_boot
    m1, fixed, m2, abl, desc, probe_desc = [], [], [], [], [], []

    for head in HEADS:
        # M1 for every visual condition
        for cond, vset in VIS_SET.items():
            rs = S.pick(cond)
            key = '{}|{}'.format(vset, head)
            ref_rs, ref_key, src = ref_text(S, cond, head, 1.0)
            if not has(rs, key) or ref_rs is None:
                continue
            c = compare(ref_rs, ref_key, rs, key, ev, nb)
            m1.append({'dataset': ds, 'head': head, 'cond': cond, 'ref_T': src, 'seeds': len(rs), **c})
        # fixed h_t (A2 runs)
        a2 = S.pick('A2')
        if has(a2, 'T|' + head):
            for vset, label in FIXED_HT:
                key = '{}|{}'.format(vset, head)
                if not has(a2, key):
                    continue
                g = compare(a2, 'T|' + head, a2, key, ev, nb)
                row = {'dataset': ds, 'head': head, 'input': label, 'gain_vs_T': g['mean'], 'gain_ci_lo': g['ci_lo'],
                       'gain_ci_hi': g['ci_hi'], 'mse': g['mse_cand']}
                if vset != 'T+Z' and has(a2, 'T+Z|' + head):
                    d = compare(a2, key, a2, 'T+Z|' + head, ev, nb)     # > 0: Z_v better than this input
                    row.update({'Z_better_by': d['mean'], 'Z_ci_lo': d['ci_lo'], 'Z_ci_hi': d['ci_hi'], 'Z_p': d['p_one_sided']})
                fixed.append(row)
        # ablations vs A2
        for cond in ('A2sub', 'A2noJ', 'A2noY'):
            rs, a2 = S.pick(cond), S.pick('A2')
            key = 'T+Z|' + head
            if has(rs, key) and has(a2, key):
                d = compare(rs, key, a2, key, ev, nb)
                abl.append({'dataset': ds, 'head': head, 'ablation': cond, 'A2_better_by': d['mean'],
                            'ci_lo': d['ci_lo'], 'ci_hi': d['ci_hi'], 'p': d['p_one_sided']})
    # M2-sup: A1 (T+h_v) vs A2 (T+Z_v), per head and fraction
    for frac in (1.0, 0.5, 0.25):
        a1, a2 = S.pick('A1', frac), S.pick('A2', frac)
        for head in HEADS:
            ka, kb = 'T+hv|' + head, 'T+Z|' + head
            if has(a1, ka) and has(a2, kb):
                d = compare(a1, ka, a2, kb, ev, nb)
                g = compare(a2, 'T|' + head, a2, kb, ev, nb) if has(a2, 'T|' + head) else {}
                m2.append({'dataset': ds, 'frac': frac, 'head': head, 'm2_sup': d['mean'], 'ci_lo': d['ci_lo'],
                           'ci_hi': d['ci_hi'], 'p': d['p_one_sided'], 'A2_gain_vs_T': g.get('mean', np.nan)})
    # descriptive
    for cond in sorted({k[0] for k in S.runs}):
        rs = S.runs_of(cond)
        if not rs:
            continue
        row = {'dataset': ds, 'cond': cond, 'seeds': len(rs)}
        for label, path in DESC:
            v = np.array([get(r, path) for r in rs], dtype=float)
            row[label] = np.nanmean(v) if np.isfinite(v).any() else np.nan
        desc.append(row)
    for cond in sorted({k[0] for k in S.probes}):
        rs = S.pick(cond)
        if not rs:
            continue
        for key in sorted(rs[0]['probes']):
            if not key.endswith('|mlp1') and not args.all_heads_desc:
                continue
            row = {'dataset': ds, 'cond': cond, 'probe': key}
            for label, path in (('MSE eval', 'mse_eval'), ('M3', 'M3'), ('M7', 'M7'), ('M8 R@20', 'M8.recall@20'),
                                ('M8 N@20', 'M8.ndcg@20')):
                v = np.array([get(r['probes'][key], path) for r in rs], dtype=float)
                row[label] = np.nanmean(v) if np.isfinite(v).any() else np.nan
            probe_desc.append(row)

    T = {n: pd.DataFrame(x) for n, x in (('m1', m1), ('fixed', fixed), ('m2', m2), ('abl', abl), ('desc', desc),
                                          ('probe_desc', probe_desc))}
    os.makedirs(args.paper_dir, exist_ok=True)
    base = os.path.join(args.paper_dir, 'jepay_diag_{}'.format(ds))
    for n, df in T.items():
        df.to_csv('{}_{}.csv'.format(base, n), index=False)
    with open(base + '.md', 'w') as f:
        f.write('# JEPA-Y E-A: {}\n\nEval items: {}. Teacher ceiling (U.Y on eval items): {}\n'.format(ds, len(ev), ceiling))
        f.write('PCA_k(V) explained variance (train / eval): {}\n\n'.format(
            ', '.join('k={} {:.3f}/{:.3f}'.format(k, v['train'], v['eval']) for k, v in pca_ex.items())))
        f.write('## M1: text-only probe vs text + visual probe (dMSE > 0: visual helps)\n\n')
        f.write(fmt(T['m1'], ['head', 'cond', 'ref_T', 'seeds', 'mse_ref', 'mse_cand', 'mean', 'ci_lo', 'ci_hi', 'pct',
                              'p_one_sided']) + '\n\n')
        f.write('## Same frozen h_t (A2 runs): which visual input helps most\n\n')
        f.write(fmt(T['fixed'], ['head', 'input', 'mse', 'gain_vs_T', 'gain_ci_lo', 'gain_ci_hi', 'Z_better_by',
                                 'Z_ci_lo', 'Z_ci_hi', 'Z_p']) + '\n\n')
        f.write('## M2-sup: MSE(A1, T+h_v) - MSE(A2, T+Z_v) (H2: > 0 at low capacity / little data, shrinking)\n\n')
        f.write(fmt(T['m2'], ['frac', 'head', 'm2_sup', 'ci_lo', 'ci_hi', 'p', 'A2_gain_vs_T']) + '\n\n')
        f.write('## Ablations: MSE(ablation) - MSE(A2), T+Z_v probes (> 0: A2 better)\n\n')
        f.write(fmt(T['abl'], ['head', 'ablation', 'A2_better_by', 'ci_lo', 'ci_hi', 'p']) + '\n\n')
        f.write('## Representations (descriptive: M4, M5, M6)\n\n')
        f.write(fmt(T['desc'], ['cond', 'seeds'] + [l for l, _ in DESC]) + '\n\n')
        f.write('## Probes (descriptive: M3, M7, M8)\n\n')
        f.write(fmt(T['probe_desc'], ['cond', 'probe', 'MSE eval', 'M3', 'M7', 'M8 R@20', 'M8 N@20']) + '\n')
    print('[summary] wrote {}.md'.format(base))
    print(fmt(T['m1'], ['head', 'cond', 'ref_T', 'mean', 'ci_lo', 'ci_hi', 'pct']))
    return T


def fmt(df, cols):
    if df is None or df.empty:
        return '(no runs)'
    cols = [c for c in cols if c in df.columns]
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df[cols].iterrows():
        out.append('| ' + ' | '.join(
            ('-' if not np.isfinite(r[c]) else '{:.4g}'.format(r[c])) if isinstance(r[c], (float, np.floating))
            else str(r[c]) for c in cols) + ' |')
    return '\n'.join(out)


def figures(res, path_h2, path_h3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dss = [d for d in res if res[d] is not None]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ds in dss:
        m2, fx = res[ds]['m2'], res[ds]['fixed']
        if not m2.empty:
            a = m2[m2.frac == 1.0].set_index('head').reindex(list(HEADS))
            axes[0].errorbar(range(3), a.m2_sup, yerr=[a.m2_sup - a.ci_lo, a.ci_hi - a.m2_sup], marker='o', capsize=3, label=ds)
            b = m2[m2['head'] == 'mlp1'].sort_values('frac')
            axes[1].errorbar(b.frac * 100, b.m2_sup, yerr=[b.m2_sup - b.ci_lo, b.ci_hi - b.m2_sup], marker='o', capsize=3, label=ds)
        if not fx.empty and 'Z_better_by' in fx:
            c = fx[fx.input == 'h_v (A1-int)'].set_index('head').reindex(list(HEADS))
            axes[2].errorbar(range(3), c.Z_better_by, yerr=[c.Z_better_by - c.Z_ci_lo, c.Z_ci_hi - c.Z_better_by],
                             marker='o', capsize=3, label=ds)
    for ax, xl, t in ((axes[0], 'probe head (100%)', 'M2-sup: A1 h_v vs A2 Z_v'),
                      (axes[1], '% train items (mlp1)', 'M2-sup vs data'),
                      (axes[2], 'probe head (100%)', 'M2-int: h_v vs Z_v, same h_t')):
        ax.axhline(0, color='grey', lw=0.8)
        ax.set_xlabel(xl)
        ax.set_title(t, fontsize=9)
    for ax in (axes[0], axes[2]):
        ax.set_xticks(range(3))
        ax.set_xticklabels(HEADS)
    axes[0].set_ylabel('dMSE (> 0: Z_v better)')
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path_h2)
    plt.close(fig)
    # H3 (descriptive): M4 and M5 for A2
    m4 = [('M4 R2(h_v<-h_t)', 'h_v'), ('M4 R2(S_v<-h_t)', 'S_v'), ('M4 R2(Z_v<-h_t)', 'Z_v')]
    m5 = [('M5 V<-h_t', 'h_t'), ('M5 V<-S', 'S_v'), ('M5 V<-Z', 'Z_v'), ('M5 V<-[S;Z]', '[S;Z]'),
          ('M5 V<-[P(h_t);Z]', '[P(h_t);Z]'), ('M5 D(S,Z)', 'D(S,Z)')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    w = 0.8 / max(len(dss), 1)
    for i, ds in enumerate(dss):
        d = res[ds]['desc']
        if d.empty or 'A2' not in set(d.cond):
            continue
        r = d[d.cond == 'A2'].iloc[0]
        axes[0].bar(np.arange(len(m4)) + i * w, [r.get(c, np.nan) for c, _ in m4], w, label=ds)
        axes[1].bar(np.arange(len(m5)) + i * w, [r.get(c, np.nan) for c, _ in m5], w, label=ds)
    for ax, labels, t in ((axes[0], m4, 'M4: R2(rep <- h_t), descriptive'),
                          (axes[1], m5, 'M5: R2 of PCA(V) (eval items)')):
        ax.set_xticks(np.arange(len(labels)) + w * (len(dss) - 1) / 2)
        ax.set_xticklabels([l for _, l in labels], fontsize=8)
        ax.set_title(t, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path_h3)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset')
    ap.add_argument('--all', nargs='*', help='datasets for the cross-dataset summary')
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'results', 'jepay_diag'))
    ap.add_argument('--paper_dir', default=os.path.join(ROOT, 'paper'))
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--gate_k', type=int, default=2, help='H1 must hold on at least this many datasets')
    ap.add_argument('--all_heads_desc', action='store_true', help='descriptive probe table for every head')
    args = ap.parse_args()
    if args.dataset:
        summarize_dataset(args.dataset, args)
    if not args.all:
        return
    res = {ds: summarize_dataset(ds, args) for ds in args.all}
    cat = lambda n: pd.concat([r[n] for r in res.values() if r is not None and not r[n].empty], ignore_index=True) \
        if any(r is not None and not r[n].empty for r in res.values()) else pd.DataFrame()   # noqa: E731
    m1, m2, fx, abl = cat('m1'), cat('m2'), cat('fixed'), cat('abl')
    lines = ['# JEPA-Y E-A: all datasets\n']
    if not m1.empty:
        m1['p_holm'] = np.nan
        for _, g in m1.groupby(['head', 'cond']):
            m1.loc[g.index, 'p_holm'] = holm(g['p_one_sided'].values)
        m1.to_csv(os.path.join(args.paper_dir, 'jepay_diag_m1.csv'), index=False)
        lines += ['## M1 with Holm over datasets\n',
                  fmt(m1, ['dataset', 'head', 'cond', 'ref_T', 'mse_ref', 'mse_cand', 'mean', 'ci_lo', 'ci_hi', 'pct',
                           'p_holm']) + '\n']
        g = m1[(m1.cond == 'A2') & (m1['head'] == 'mlp1')]
        ok = g[(g.ci_lo > 0) & (g.p_holm < 0.05)]
        lines.append('## Gate\n')
        lines.append('- H1 (A2, mlp1): CI lower bound > 0 and Holm p < .05 on {}/{} datasets ({}); needed {} -> **{}**'.format(
            len(ok), len(g), ', '.join(ok.dataset) or '-', args.gate_k, 'PASS' if len(ok) >= args.gate_k else 'FAIL'))
        if not m2.empty:
            low = m2[((m2['head'] == 'lin') & (m2.frac == 1.0)) | ((m2['head'] == 'mlp1') & (m2.frac == 0.25))]
            lines.append('- H2 (M2-sup > 0 at lin or 25%): ' + ', '.join(
                '{} {}/{:g}: {:+.5f} [{:+.5f}, {:+.5f}]'.format(r.dataset, r.head, r.frac, r.m2_sup, r.ci_lo, r.ci_hi)
                for r in low.itertuples()))
        if not fx.empty:
            a3 = fx[(fx.input == 'Z_lin (A3)') & (fx['head'] == 'mlp1')]
            lines.append('- A2 vs A3 (Z_v better than Z_lin, same h_t, mlp1): ' + ', '.join(
                '{} {:+.5f} [{:+.5f}, {:+.5f}]'.format(r.dataset, r.Z_better_by, r.Z_ci_lo, r.Z_ci_hi)
                for r in a3.itertuples()))
        if not abl.empty:
            lines += ['\n## Ablations vs A2 (> 0: A2 better)\n',
                      fmt(abl, ['dataset', 'head', 'ablation', 'A2_better_by', 'ci_lo', 'ci_hi', 'p']) + '\n']
        if not m2.empty:
            lines += ['\n## M2-sup\n', fmt(m2, ['dataset', 'frac', 'head', 'm2_sup', 'ci_lo', 'ci_hi', 'p']) + '\n']
    os.makedirs(args.paper_dir, exist_ok=True)
    with open(os.path.join(args.paper_dir, 'jepay_diag.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    figures(res, os.path.join(args.paper_dir, 'fig_jepay_h2.pdf'), os.path.join(args.paper_dir, 'fig_jepay_h3.pdf'))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
