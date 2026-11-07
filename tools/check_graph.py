# coding: utf-8
"""
Check that the cached item-item graph really is the same across the conditions of a graph-fixed (*_tg) protocol.

    python tools/check_graph.py cloth_vcopy cloth_vperm_tg cloth_vconst_tg
    python tools/check_graph.py --file mm_adj_10.pt sports_vcopy sports_vperm_tg sports_vconst_tg

DRAGON / MENTOR / COHESION cache the graph as data/<folder>/mm_adj_<k>.pt; the file name does not encode
mm_image_weight, so a folder must only ever be trained with one protocol. With mm_image_weight = 0 the graph is
built from the text features alone, which are identical in these folders, so the cached graphs must be equal.
Also reports, for every folder, whether its image features differ (they should: real / perm / const).
"""
import argparse
import os

import numpy as np
import torch


def load(path):
    try:
        a = torch.load(path, map_location='cpu', weights_only=False)   # torch >= 2.6 defaults to weights_only=True
    except TypeError:
        a = torch.load(path, map_location='cpu')
    if a.is_sparse:
        a = a.coalesce()
        return a.indices().numpy(), a.values().numpy(), tuple(a.shape)
    a = a.numpy()
    idx = np.nonzero(a)
    return np.vstack(idx), a[idx], a.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folders', nargs='+')
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--file', default='mm_adj_10.pt')
    args = ap.parse_args()

    ref = None
    for f in args.folders:
        p = os.path.join(args.data_dir, f, args.file)
        img = np.load(os.path.join(args.data_dir, f, 'image_feat.npy'), mmap_mode='r')
        img_sig = float(np.asarray(img[:50]).sum())
        if not os.path.exists(p):
            print('{:<28s} no cached {} yet (built on the first run)'.format(f, args.file))
            continue
        idx, val, shape = load(p)
        line = '{:<28s} shape={} nnz={} image-feature checksum={:.3f}'.format(f, shape, val.size, img_sig)
        if ref is None:
            ref = (f, idx, val, shape)
            print(line + '  (reference)')
            continue
        same = (shape == ref[3] and idx.shape == ref[1].shape and np.array_equal(idx, ref[1])
                and np.allclose(val, ref[2], atol=1e-6))
        print(line + ('  SAME graph as ' + ref[0] if same else '  !! DIFFERENT graph from ' + ref[0]))


if __name__ == '__main__':
    main()
