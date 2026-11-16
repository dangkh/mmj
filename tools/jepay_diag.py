# coding: utf-8
"""
JEPA-Y diagnosis E-A (diagnosis.md; plan_diag.md). Needs data/<ds>/teacher_{Y,U}.npy (tools/export_teacher.py)
and data/<ds>/jepay_item_split.npz (tools/make_item_split.py).

Stage A  train each condition (tools/jepay/model.py) on the train items; checkpoint chosen on val items.
Stage B  freeze h_t, h_v, S_v, Z_v and fit fresh probes lin / mlp1 / mlp2 on the same train items
         (tools/jepay/probe.py); report on eval items. The text-only reference 'T' uses the same frozen h_t.

    python tools/jepay_diag.py -d baby --grid               # stage-A grid on val items (seed 0, 100 %)
    python tools/jepay_diag.py -d baby --matrix             # stage A + B for every run (finished parts skipped)
    python tools/jepay_diag.py -d baby --cond A2 --frac 1.0 --seed 0
    python tools/jepay_diag.py -d baby --grid --matrix --smoke     # quick check (3 epochs)

Matrix: A0, A1, A2, A2sub, A2noJ, A2noY x seeds {0,1,2} at 100 % (18) + A1, A2 x {25, 50} % x 3 seeds (12).
Grid (val criterion: MSE_TV of the stage-A head; A0: MSE_T; A2noY: L_JEPA + lam_R L_rec):
    A0 wd | A1 lam_C x wd | A2sub lam_J x lam_C x wd | A2 lam_J x lam_C x wd x lam_R x k | A2noJ lam_C x wd x lam_R x k
    lam_J {0.1, 1}, lam_C {0.01, 0.1}, wd {1e-4, 1e-3}, lam_R {0.1, 1}, k (PCA_k reconstruction target) {64, 128, 256}.
    A2noY reuses A2's configuration. The selected configuration is used for every seed and fraction.

Output: results/jepay_diag/<ds>/
    best_cfg.json, grid/*.json, pca_explained.json, teacher_ceiling.json
    runs/<cond>_f<pct>_s<seed>.{json,pt}      stage A: config, epochs, val terms, M4-M6, own-decoder M5
    probes/<cond>_f<pct>_s<seed>.{json,npz}   stage B: per probe (set|head) val/eval MSE, M3, M7, M8 (mlp1);
                                              npz: per-item squared error of every probe (all items)
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from jepay.data import DiagData                                                     # noqa: E402
from jepay.model import JEPAY, CONDS, loss_terms                                     # noqa: E402
from jepay.metrics import representation_metrics, probe_metrics, teacher_ceiling    # noqa: E402
from jepay.probe import fit_probe, probe_sets                                        # noqa: E402

DEFAULT = {'d': 64, 'hidden': 128, 'dec_hidden': 256, 'lam_T': 1.0, 'lam_TV': 1.0, 'lam_J': 1.0, 'lam_R': 1.0,
           'lam_C': 0.01, 'lam_zt': 0.0, 'k': 128, 'wd': 1e-4, 'lr': 1e-3, 'batch': 1024, 'epochs': 300,
           'patience': 20, 'ht_dropout': 0.0}
GRID = {'lam_J': [0.1, 1.0], 'lam_C': [0.01, 0.1], 'wd': [1e-4, 1e-3], 'lam_R': [0.1, 1.0], 'k': [64, 128, 256]}
GRID_KEYS = {'A0': ['wd'], 'A1': ['lam_C', 'wd'], 'A2sub': ['lam_J', 'lam_C', 'wd'],
             'A2': ['lam_J', 'lam_C', 'wd', 'lam_R', 'k'], 'A2noJ': ['lam_C', 'wd', 'lam_R', 'k']}
SEEDS = (0, 1, 2)
PROBE_HEADS = ('lin', 'mlp1', 'mlp2')


def run_name(cond, frac, seed):
    return '{}_f{}_s{}'.format(cond, int(round(float(frac) * 100)), seed)


def grid_configs(cond):
    keys = GRID_KEYS[cond]
    for vals in itertools.product(*[GRID[k] for k in keys]):
        yield dict(zip(keys, vals))


def cfg_tag(c):
    return '_'.join('{}{:g}'.format(k, c[k]) for k in sorted(c))


def criterion(cond, t, cfg):
    if cond == 'A0':
        return t['L_T']
    if cond == 'A2noY':
        return t['L_J'] + cfg['lam_R'] * t['L_rec']
    return t['L_TV']


@torch.no_grad()
def full_pass(model, T, V, chunk=8192):
    model.eval()
    parts = {}
    for s in range(0, T.shape[0], chunk):
        out = model(T[s:s + chunk], V[s:s + chunk] if model.has_v else None)
        for k, v in out.items():
            parts.setdefault(k, []).append(v.float().cpu())
    return {k: torch.cat(v).numpy() for k, v in parts.items()}


@torch.no_grad()
def val_terms(model, T, V, Y, VT, rows, cfg):
    model.eval()
    out = model(T[rows], V[rows] if model.has_v else None)
    _, terms = loss_terms(model, out, Y[rows], cfg, VT[rows] if VT is not None else None)
    return {k: float(v) for k, v in terms.items()}


def build_model(cond, p, cfg):
    return JEPAY(cond, p.dt, p.dv, p.dy, k_rec=cfg['k'], d=cfg['d'], hidden=cfg['hidden'],
                 dec_hidden=cfg['dec_hidden'], ht_dropout=cfg['ht_dropout'])


def train_stage_a(D, cond, frac, seed, cfg, device):
    p = D.prepare(frac)
    v_target = D.pca_target(p, cfg['k'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    T, V, Y = (torch.from_numpy(x).to(device) for x in (p.T, p.V, p.Y))
    VT = torch.from_numpy(v_target).to(device)
    tr = torch.from_numpy(p.train).long().to(device)
    va = torch.from_numpy(p.val).long().to(device)
    model = build_model(cond, p, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    best, best_state, best_ep, bad, ep = float('inf'), None, -1, 0, 0
    t0 = time.time()
    for ep in range(cfg['epochs']):
        model.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        for s in range(0, len(perm), cfg['batch']):
            b = perm[s:s + cfg['batch']]
            if len(b) < 2:             # VICReg needs a batch variance
                continue
            out = model(T[b], V[b] if model.has_v else None)
            loss, _ = loss_terms(model, out, Y[b], cfg, VT[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        crit = criterion(cond, val_terms(model, T, V, Y, VT, va, cfg), cfg)
        if not np.isfinite(crit):
            break
        if crit < best - 1e-7:
            best, best_ep, bad = crit, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg['patience']:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {'best_epoch': best_ep, 'epochs_run': ep + 1, 'val_criterion': best, 'train_seconds': time.time() - t0,
            'val_terms': val_terms(model, T, V, Y, VT, va, cfg)}
    return model, p, v_target, info


class Runner(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        self.out = os.path.join(args.out_dir, args.dataset)
        for sub in ('runs', 'probes', 'grid'):
            os.makedirs(os.path.join(self.out, sub), exist_ok=True)
        print('[jepay] loading {} on {}'.format(args.dataset, self.device), flush=True)
        self.D = DiagData(args.dataset, args.data_dir)
        p = self.D.prepare(1.0)
        print('[jepay] items {} (train {}, val {}, eval {}), users {}, T {}, V {}, Y {}'.format(
            self.D.n_items, len(p.train), len(p.val), len(p.eval), self.D.n_users, p.dt, p.dv, p.dy), flush=True)
        ex = self.D.explained_variance(p)
        json.dump(ex, open(os.path.join(self.out, 'pca_explained.json'), 'w'), indent=1)
        print('[jepay] PCA_k(V) explained variance (train / eval):',
              ', '.join('k={} {:.3f}/{:.3f}'.format(k, v['train'], v['eval']) for k, v in ex.items()), flush=True)
        tc = os.path.join(self.out, 'teacher_ceiling.json')
        if os.path.isfile(tc):
            self.ceiling = json.load(open(tc))
        else:
            self.ceiling = teacher_ceiling(self.D)
            json.dump(self.ceiling, open(tc, 'w'), indent=1)
        print('[jepay] M8 teacher ceiling (U . Y on eval items):', self.ceiling, flush=True)

    def base_cfg(self):
        c = dict(DEFAULT)
        if self.args.smoke:
            c['epochs'], c['patience'] = 3, 2
        for k in ('ht_dropout', 'lam_zt'):
            v = getattr(self.args, k)
            if v is not None:
                c[k] = v
        return c

    def best_cfg(self, cond):
        path = os.path.join(self.out, 'best_cfg.json')
        best = json.load(open(path)) if os.path.isfile(path) else {}
        key = 'A2' if cond == 'A2noY' else cond
        c = self.base_cfg()
        if key in best:
            c.update(best[key])
        else:
            print('  [warn] no grid result for {}: default configuration'.format(key), flush=True)
        return c

    # ----------------------------------------------------------------------------------- grid
    def grid(self):
        path = os.path.join(self.out, 'best_cfg.json')
        best = json.load(open(path)) if os.path.isfile(path) else {}
        for cond in [c for c in self.args.conds if c in GRID_KEYS]:
            if cond in best and not self.args.force:
                print('[grid] {} done: {}'.format(cond, best[cond]), flush=True)
                continue
            cfgs = list(grid_configs(cond))
            if self.args.smoke:
                cfgs = cfgs[:2]
            results = []
            for g in cfgs:
                f = os.path.join(self.out, 'grid', '{}_{}.json'.format(cond, cfg_tag(g)))
                if os.path.isfile(f) and not self.args.force:
                    r = json.load(open(f))
                else:
                    cfg = self.base_cfg()
                    cfg.update(g)
                    _, _, _, info = train_stage_a(self.D, cond, 1.0, 0, cfg, self.device)
                    r = {'cond': cond, 'grid': g, 'cfg': cfg, **info}
                    json.dump(r, open(f, 'w'), indent=1)
                print('[grid] {} {}: val {:.5f} (epoch {})'.format(cond, cfg_tag(g), r['val_criterion'],
                                                                  r['best_epoch']), flush=True)
                results.append(r)
            top = min(results, key=lambda r: r['val_criterion'])
            best[cond] = top['grid']
            json.dump(best, open(path, 'w'), indent=1)
            print('[grid] {} -> {}'.format(cond, top['grid']), flush=True)

    # ----------------------------------------------------------------------------------- stage A
    def stage_a(self, cond, frac, seed):
        name = run_name(cond, frac, seed)
        base = os.path.join(self.out, 'runs', name)
        if os.path.isfile(base + '.json') and os.path.isfile(base + '.pt') and not self.args.force:
            return False
        cfg = self.best_cfg(cond)
        model, p, v_target, info = train_stage_a(self.D, cond, frac, seed, cfg, self.device)
        arr = full_pass(model, *(torch.from_numpy(x).to(self.device) for x in (p.T, p.V)))
        rep = representation_metrics(p, arr, v_target)
        head = {}
        for br, key in (('T', 'y_t'), ('TV', 'y_tv')):
            if key in arr:
                e = ((arr[key] - p.Y) ** 2).mean(1)
                head[br] = {s: float(e[r].mean()) for s, r in (('train', p.train), ('val', p.val), ('eval', p.eval))}
        torch.save({'state_dict': model.state_dict(), 'cond': cond, 'frac': float(frac), 'seed': seed, 'cfg': cfg,
                    'dims': [p.dt, p.dv, p.dy]}, base + '.pt')
        res = {'dataset': self.args.dataset, 'name': name, 'cond': cond, 'frac': float(frac), 'seed': seed,
               'cfg': cfg, **info, 'stageA_head_mse': head, 'rep_metrics': rep, 'n_train': int(len(p.train))}
        with open(base + '.json', 'w') as f:
            json.dump(res, f, indent=1)
        msg = '[A] {}: epoch {} | val crit {:.4f}'.format(name, info['best_epoch'], info['val_criterion'])
        if 'M5_D_s_z' in rep:
            msg += ' | D(S,Z) R2 {:.3f}, D(P(h_t),Z) R2 {:.3f}, |corr(S,Z)| {:.3f}'.format(
                rep['M5_D_s_z'], rep.get('M5_D_shat_z', float('nan')), rep['corr_s_z'])
        print(msg + ' | {:.0f}s'.format(info['train_seconds']), flush=True)
        return True

    def load_reps(self, name):
        ck = torch.load(os.path.join(self.out, 'runs', name + '.pt'), map_location=self.device)
        p = self.D.prepare(ck['frac'])
        model = build_model(ck['cond'], p, ck['cfg']).to(self.device)
        model.load_state_dict(ck['state_dict'])
        arr = full_pass(model, *(torch.from_numpy(x).to(self.device) for x in (p.T, p.V)))
        return ck, p, arr

    # ----------------------------------------------------------------------------------- stage B
    def stage_b(self, cond, frac, seed):
        name = run_name(cond, frac, seed)
        base = os.path.join(self.out, 'probes', name)
        if os.path.isfile(base + '.json') and not self.args.force:
            return
        t0 = time.time()
        ck, p, arr = self.load_reps(name)
        sets = probe_sets(cond, arr, p, full=float(frac) == 1.0)
        wd_grid = (1e-4,) if self.args.smoke else None
        res, npz, yev = {}, {}, {}
        for sname, X in sets.items():
            for head in PROBE_HEADS:
                key = '{}|{}'.format(sname, head)
                kw = {'wd_grid': wd_grid} if wd_grid else {}
                yhat, info = fit_probe(X, p.Y, p.train, p.val, head, seed, self.device, **kw)
                m, mse = probe_metrics(self.D, p, yhat, p.Y, rank=(head == 'mlp1'))
                res[key] = {**m, **info, 'set': sname, 'head': head, 'in_dim': int(X.shape[1])}
                npz[key] = mse.astype(np.float32)
                yev[key] = yhat[p.eval]
        for key in res:                     # M3: change of the prediction when the visual input is added
            sname, head = key.split('|')
            ref = yev.get('T|' + head)
            if sname != 'T' and ref is not None:
                d = yev[key] - ref
                res[key]['M3'] = float((d ** 2).sum(1).mean() / max((ref ** 2).sum(1).mean(), 1e-12))
        np.savez_compressed(base + '.npz', **npz)
        out = {'dataset': self.args.dataset, 'name': name, 'cond': cond, 'frac': float(frac), 'seed': seed,
               'probes': res, 'seconds': time.time() - t0}
        with open(base + '.json', 'w') as f:
            json.dump(out, f, indent=1)
        line = ' '.join('{} {:.4f}'.format(k, v['mse_eval']) for k, v in res.items() if k.endswith('|mlp1'))
        print('[B] {} (mlp1 eval MSE) {} | {:.0f}s'.format(name, line, out['seconds']), flush=True)

    def one(self, cond, frac, seed):
        self.stage_a(cond, frac, seed)
        self.stage_b(cond, frac, seed)

    def matrix(self):
        jobs = [(c, 1.0, s) for c in self.args.conds for s in SEEDS]
        jobs += [(c, f, s) for c in ('A1', 'A2') if c in self.args.conds for f in (0.25, 0.5) for s in SEEDS]
        if self.args.smoke:
            jobs = [j for j in jobs if j[2] == 0]
        print('[matrix] {} runs (stage A + probes)'.format(len(jobs)), flush=True)
        failed = []
        for j in jobs:
            try:
                self.one(*j)
            except Exception as e:        # one failing run must not stop the job
                print('[run] FAILED {}: {!r}'.format(run_name(*j), e), flush=True)
                failed.append(run_name(*j))
        print('[matrix] done, failed: {}'.format(failed or 'none'), flush=True)
        return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True)
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'results', 'jepay_diag'))
    ap.add_argument('--grid', action='store_true')
    ap.add_argument('--matrix', action='store_true')
    ap.add_argument('--conds', default=','.join(CONDS), help='subset for --grid / --matrix, e.g. A0,A1,A2')
    ap.add_argument('--cond', choices=CONDS)
    ap.add_argument('--frac', type=float, default=1.0, choices=[0.25, 0.5, 1.0])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ht_dropout', type=float, help='dropout on h_t in the stage-A head')
    ap.add_argument('--lam_zt', type=float, help='ablation: CrossCov(Z_v, sg(h_t)) weight (default 0 = off)')
    ap.add_argument('--smoke', action='store_true', help='3 epochs, 2 grid configs, seed 0, one probe wd')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    args.conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    if args.smoke:
        args.out_dir = args.out_dir.rstrip('/') + '_smoke'
    if args.lam_zt:
        args.out_dir = args.out_dir.rstrip('/') + '_zt{:g}'.format(args.lam_zt)   # own series for the ablation
    if not (args.grid or args.matrix or args.cond):
        ap.error('give --grid, --matrix or --cond')
    r = Runner(args)
    if args.grid:
        r.grid()
    if args.matrix:
        if r.matrix():
            sys.exit(1)
    if args.cond:
        r.one(args.cond, args.frac, args.seed)


if __name__ == '__main__':
    main()
