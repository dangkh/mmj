# coding: utf-8
"""
JEPA-Y diagnosis, step 1: the frozen collaborative teacher Y (LightGCN on TRAIN interactions only).

    python tools/export_teacher.py -d baby
    python tools/export_teacher.py -d ml1m --teacher_args "--n_layers=3 --reg_weight=0.0001"   # fixed setting

1. Hyper-parameters, in this order of priority:
     --teacher_args "--key=value ..."                      used as given (no tuning)
     an existing finished grid log  src/log/LightGCN-<ds>-<grid_name>-*.log   (tools/best_params.py)
     otherwise a grid run is started first:  main.py -m LightGCN -d <ds> --exp_name=<grid_name> <--grid>
2. One LightGCN run with --save_taps=True (exp_name teacher_s<seed>): src/taps.py saves the item / user
   embeddings AFTER propagation (the ones used for scoring) at the best-valid epoch.
3. Copy them to data/<ds>/teacher_Y.npy (items x 64) and teacher_U.npy (users x 64), with teacher_meta.json.

LightGCN is trained by MMRec on x_label == 0 only, so valid/test interactions never reach the teacher.
Checks: the teacher has one row per feature row (image_feat.npy / text_feat.npy), finite values.
Skipped entirely if data/<ds>/teacher_Y.npy exists (use --force to redo).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from best_params import best_params   # noqa: E402


def run_main(args_list):
    cmd = [sys.executable, '-u', 'main.py'] + args_list
    print('[teacher] cd src &&', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=os.path.join(ROOT, 'src'), check=True, stdin=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True)
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--teacher_args', default='', help='fixed LightGCN overrides, e.g. "--n_layers=3 --reg_weight=1e-4"')
    ap.add_argument('--grid_name', default='lightgcn_grid')
    ap.add_argument('--grid', default='--n_layers=[2,3,4] --reg_weight=[0.001,0.0001]',
                    help='grid used only if no finished grid log exists (LightGCN.yaml: 4 x 5 combinations)')
    ap.add_argument('--seed', type=int, default=999)
    ap.add_argument('--embedding_size', type=int, default=64)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    dst = os.path.join(args.data_dir, args.dataset)
    out_y, out_u = os.path.join(dst, 'teacher_Y.npy'), os.path.join(dst, 'teacher_U.npy')
    if os.path.isfile(out_y) and os.path.isfile(out_u) and not args.force:
        print('[teacher] exists: {} (use --force to redo)'.format(out_y))
        return

    # 1. hyper-parameters
    if args.teacher_args.strip():
        hp = args.teacher_args.split()
        source = 'given'
    else:
        p = best_params('LightGCN', args.dataset, args.grid_name, log_dir=os.path.join(ROOT, 'src', 'log'))
        if p is None:
            run_main(['-m', 'LightGCN', '-d', args.dataset, '--exp_name=' + args.grid_name,
                      '--seed={}'.format(args.seed), '--embedding_size={}'.format(args.embedding_size)]
                     + args.grid.split())
            p = best_params('LightGCN', args.dataset, args.grid_name, log_dir=os.path.join(ROOT, 'src', 'log'))
            if p is None:
                raise SystemExit('grid run finished without a BEST block in src/log/LightGCN-{}-{}-*.log'.format(
                    args.dataset, args.grid_name))
        hp = ['--{}={}'.format(k, v) for k, v in p.items()]
        source = 'grid {}'.format(args.grid_name)
    print('[teacher] hyper-parameters ({}): {}'.format(source, ' '.join(hp)), flush=True)

    # 2. teacher run with taps
    name = 'teacher_s{}'.format(args.seed)
    taps = os.path.join(ROOT, 'taps', 'LightGCN-{}-{}'.format(args.dataset, name))
    if args.force and os.path.isdir(taps):
        shutil.rmtree(taps)
    if not os.path.isfile(os.path.join(taps, 'T3.npy')):
        run_main(['-m', 'LightGCN', '-d', args.dataset, '--exp_name=' + name, '--seed={}'.format(args.seed),
                  '--embedding_size={}'.format(args.embedding_size), '--save_taps=True'] + hp)
    Y = np.load(os.path.join(taps, 'T3.npy')).astype(np.float32)
    U = np.load(os.path.join(taps, 'T3_user.npy')).astype(np.float32)
    meta = json.load(open(os.path.join(taps, 'meta.json')))

    # 3. checks + copy
    n_feat = {f: np.load(os.path.join(dst, f), mmap_mode='r').shape[0] for f in ('image_feat.npy', 'text_feat.npy')}
    for f, n in n_feat.items():
        if n != Y.shape[0]:
            raise SystemExit('teacher has {} item rows but {} has {}: item ids of the .inter and the feature rows '
                             'do not match'.format(Y.shape[0], f, n))
    if not (np.isfinite(Y).all() and np.isfinite(U).all()):
        raise SystemExit('teacher embeddings contain NaN/inf')
    if Y.shape[1] != args.embedding_size:
        raise SystemExit('teacher dim {} != {}'.format(Y.shape[1], args.embedding_size))
    np.save(out_y, Y)
    np.save(out_u, U)
    info = {'source_taps': taps, 'hyper_parameters': hp, 'hp_source': source, 'seed': args.seed,
            'epoch': meta.get('epoch'), 'valid': meta.get('valid'), 'test': meta.get('test'),
            'n_items': int(Y.shape[0]), 'n_users': int(U.shape[0]), 'dim': int(Y.shape[1])}
    with open(os.path.join(dst, 'teacher_meta.json'), 'w') as f:
        json.dump(info, f, indent=1)
    print('[teacher] wrote {} {} and {} {}'.format(out_y, Y.shape, out_u, U.shape))
    print('[teacher] best-valid epoch {}: valid recall@20 {:.4f}, test recall@20 {:.4f}  <- compare with the '
          'reproduced LightGCN'.format(meta.get('epoch'), meta.get('valid', {}).get('recall@20', float('nan')),
                                       meta.get('test', {}).get('recall@20', float('nan'))))


if __name__ == '__main__':
    main()
