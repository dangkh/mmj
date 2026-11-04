# coding: utf-8
"""
Build a control copy of a dataset in which one modality carries no item-specific content:

    perm  : feature rows shuffled across items (same feature distribution, wrong item)
    noise : Gaussian rows with the same per-dimension mean / std (no real structure at all)

Everything else (interactions, the other modality, user graph) is linked unchanged, and the
copy gets its own folder, so its cached item-item graphs are rebuilt from the control features.

    python tools/make_control_data.py -d baby --modality v --mode perm     # -> data/baby_vperm
    python tools/make_control_data.py -d baby --modality v --mode noise    # -> data/baby_vnoise

Train any model on it with  -d baby_vperm  (a dataset config src/configs/dataset/baby_vperm.yaml
is written). Purpose: a model trained end-to-end can turn high-dimensional, item-unique features
into an item-ID lookup and store collaborative signal in its "visual" projection. If T1_v of the
control run probes as well as T1_v of the real run, the real T1_v gain is memorisation, not content.
"""
import argparse
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='baby')
    ap.add_argument('--modality', choices=['v', 't'], default='v')
    ap.add_argument('--mode', choices=['perm', 'noise'], default='perm')
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--config_dir', default='src/configs/dataset')
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    src = os.path.join(args.data_dir, args.dataset)
    name = '{}_{}{}'.format(args.dataset, args.modality, args.mode)
    dst = os.path.join(args.data_dir, name)
    os.makedirs(dst, exist_ok=True)
    feat = {'v': 'image_feat.npy', 't': 'text_feat.npy'}
    target = feat[args.modality]

    def link(src_name, dst_name):
        s = os.path.realpath(os.path.join(src, src_name))
        d = os.path.join(dst, dst_name)
        if os.path.lexists(d):
            os.remove(d)
        os.symlink(s, d)
        print('  link {} -> {}'.format(d, s))

    link('{}.inter'.format(args.dataset), '{}.inter'.format(name))
    for f in feat.values():
        if f != target:
            link(f, f)
    if os.path.exists(os.path.join(src, 'user_graph_dict.npy')):
        link('user_graph_dict.npy', 'user_graph_dict.npy')

    X = np.load(os.path.join(src, target)).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    if args.mode == 'perm':
        perm = rng.permutation(X.shape[0])
        Y = X[perm]
        np.save(os.path.join(dst, 'perm_index.npy'), perm)
    else:
        Y = (rng.standard_normal(X.shape) * X.std(0) + X.mean(0)).astype(np.float32)
    out = os.path.join(dst, target)
    if os.path.lexists(out):
        os.remove(out)
    np.save(out, Y)
    print('  wrote {} {} ({})'.format(out, Y.shape, args.mode))

    cfg_src = os.path.join(args.config_dir, '{}.yaml'.format(args.dataset))
    cfg_dst = os.path.join(args.config_dir, '{}.yaml'.format(name))
    text = open(cfg_src).read().replace("'{}.inter'".format(args.dataset), "'{}.inter'".format(name))
    with open(cfg_dst, 'w') as f:
        f.write('# control copy of {} ({} of modality {}), written by tools/make_control_data.py\n'.format(
            args.dataset, args.mode, args.modality) + text)
    print('  wrote', cfg_dst)
    print('Train with -d {} ; probe with tools/probe_knn.py -d {} --taps ...'.format(name, name))


if __name__ == '__main__':
    main()
