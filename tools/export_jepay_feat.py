# coding: utf-8
"""
E-B-lite: freeze the JEPA-Y models of E-A and write their visual representations as image features of new
data folders (tools/make_control_data.py --mode file), to train DRAGON / COHESION on them unchanged.

    python tools/export_jepay_feat.py -d baby                 # seed-0 runs, head mlp1, 100 % train items
    python tools/export_jepay_feat.py -d baby --rg            # also *_rg copies that keep the REAL item-item graph

Folders (visual input; everything else linked from data/<ds>):
    <ds>_jepay_hv     h_v of A1      (same supervision, no decomposition)
    <ds>_jepay_z      Z_v of A2      (JEPA-Y)
    <ds>_jepay_znoy   Z_v of A2noY   (Y never shaped the representation)
    <ds>_jepay_zlin   Z_lin (A3)     (PCA_64(V) - OLS(PCA_64(V) ~ T), standardised on train items)
    <ds>_jepay_zperm  Z_v of A2, rows permuted across items (seed 2026)
  optional (--targets): s (S_v of A2), zsub (Z_v of A2sub), znoj (Z_v of A2noJ)
Every item gets a row (train, val and eval items, and items without train interactions), computed by the
frozen encoders; nothing is refitted here.

Graph caveat: DRAGON / COHESION build their item-item graph (mm_adj_<k>.pt) from the image features, so a
new folder changes the graph AND the visual input. --rg also writes <name>_rg folders with the real graph
cache copied from data/<ds> (needs a finished run of the model on <ds>; DRAGON and COHESION share the file
name mm_adj_10.pt, so keep one model per _rg folder as in the main diagnosis).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from jepay.data import DiagData, PERM_SEED   # noqa: E402
from jepay_diag import build_model            # noqa: E402

DEFAULT_TARGETS = 'hv,z,znoy,zlin,zperm'
TARGETS = {          # folder suffix: (condition, representation)
    'hv': ('A1', 'h_v'),
    'z': ('A2', 'z'),
    'znoy': ('A2noY', 'z'),
    'zlin': ('A2', 'zlin'),         # computed from the data (needs only the split), not from a model
    'zperm': ('A2', 'z_perm'),
    's': ('A2', 's'),
    'zsub': ('A2sub', 'z'),
    'znoj': ('A2noJ', 'z'),
}


@torch.no_grad()
def encode(D, run_path):
    ck = torch.load(run_path, map_location='cpu')
    p = D.prepare(ck['frac'])
    model = build_model(ck['cond'], p, ck['cfg'])
    model.load_state_dict(ck['state_dict'])
    model.eval()
    out = model.encode(torch.from_numpy(p.T), torch.from_numpy(p.V))
    return {k: v.numpy() for k, v in out.items()}, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True)
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'results', 'jepay_diag'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--targets', default=DEFAULT_TARGETS, help='any of ' + ','.join(TARGETS))
    ap.add_argument('--rg', action='store_true', help='also <name>_rg folders with the real item-item graph cache')
    args = ap.parse_args()

    D = DiagData(args.dataset, args.data_dir)
    feat_dir = os.path.join(args.out_dir, args.dataset, 'feat')
    os.makedirs(feat_dir, exist_ok=True)
    info = {}
    for tag in [t.strip() for t in args.targets.split(',') if t.strip()]:
        cond, rep = TARGETS[tag]
        run = os.path.join(args.out_dir, args.dataset, 'runs', '{}_f100_s{}.pt'.format(cond, args.seed))
        if rep == 'zlin':
            Z = D.prepare(1.0).zlin
        elif not os.path.isfile(run):
            print('[export] missing {}: skip {}'.format(run, tag))
            continue
        else:
            reps, _ = encode(D, run)
            Z = reps['z'] if rep == 'z_perm' else reps[rep]
            if rep == 'z_perm':
                Z = Z[np.random.default_rng(PERM_SEED).permutation(Z.shape[0])]
        Z = Z.astype(np.float32)
        assert Z.shape[0] == D.n_items and np.isfinite(Z).all(), tag
        path = os.path.join(feat_dir, '{}.npy'.format(tag))
        np.save(path, Z)
        info[tag] = {'run': run, 'shape': list(Z.shape), 'row_norm_mean': float(np.linalg.norm(Z, axis=1).mean())}
        names = [('{}_jepay_{}'.format(args.dataset, tag), [])]
        if args.rg:
            names.append(('{}_jepay_{}_rg'.format(args.dataset, tag), ['--reuse_graph']))
        for name, extra in names:
            cmd = [sys.executable, os.path.join(ROOT, 'tools', 'make_control_data.py'), '-d', args.dataset,
                   '--modality', 'v', '--mode', 'file', '--feat', path, '--name', name,
                   '--data_dir', args.data_dir, '--config_dir', os.path.join(ROOT, 'src', 'configs', 'dataset')] + extra
            print('[export]', ' '.join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=ROOT)
    with open(os.path.join(feat_dir, 'export_meta.json'), 'w') as f:
        json.dump(info, f, indent=1)
    print('[export] done:', json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
