# coding: utf-8
"""
Convert the multimodal ML-1M / DBbook / Last.fm-2K data of Spillo et al. (Zenodo 10.5281/zenodo.15403972,
code: github.com/giuspillo/extending_sota_data_mm_feat) into an MMRec data folder that follows the protocol of
the Amazon datasets used so far.

    python tools/convert_spillo.py --name ml1m \
        --inter  <repo>/3_data_processing/ml-1m_grouplens/processed_data/movielens_1m.inter \
        --feat_dir <repo>/3_data_processing/ml-1m_grouplens/multimodal_features/mmrec_npy \
        --image vgg --text all-MiniLM-L6-v2
    (DBbook: dbbook/processed_data/dbbook.inter, dbbook/multimodal_features/mmrec_npy)

Why not use their .inter as is: it contains NEGATIVE feedback (rating = 0: ML-1M ratings <= 3, DBbook dislikes)
next to the positives, and MMRec ignores the rating column, so every dislike would become a positive.
Steps:
  1. keep rating == 1 (their binarised positives);
  2. iterative 5-core on the positives (users and items >= 5);
  3. per-user random split 80/10/10 (seed 42), at least one valid and one test interaction per user; an item
     that ends up without any training interaction gets one of its valid/test interactions moved to train;
  4. re-index users and items 0..n-1; the feature rows are re-ordered accordingly (their npy row i = their
     itemID i, i.e. the item index of their .inter);
  5. write data/<name>/<name>.inter, image_feat.npy, text_feat.npy, user_graph_dict.npy (same construction as
     preprocessing/dualgnn-gen-u-u-matrix.py: co-interaction counts, top-200 per user, vectorised) and
     src/configs/dataset/<name>.yaml.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp


def core_filter(df, k):
    while True:
        uc, ic = df.userID.value_counts(), df.itemID.value_counts()
        keep = df.userID.map(uc).ge(k) & df.itemID.map(ic).ge(k)
        if keep.all():
            return df
        df = df[keep]


def split(df, seed, ratios=(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(seed)
    label = np.zeros(len(df), dtype=np.int64)
    pos = {u: idx for u, idx in df.groupby('userID').indices.items()}
    for u, idx in pos.items():
        idx = rng.permutation(idx)
        n = len(idx)
        n_test = max(1, int(round(n * ratios[2])))
        n_valid = max(1, int(round(n * ratios[1])))
        label[idx[n - n_test:]] = 2
        label[idx[n - n_test - n_valid:n - n_test]] = 1
    df = df.copy()
    df['x_label'] = label
    # items without any training interaction: move one of their interactions to train
    while True:
        tr_items = set(df.loc[df.x_label == 0, 'itemID'])
        missing = df[(df.x_label > 0) & ~df.itemID.isin(tr_items)]
        if missing.empty:
            break
        first = missing.groupby('itemID').head(1).index
        df.loc[first, 'x_label'] = 0
    return df


def user_graph(train, n_users, n_items, top=200):
    R = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (train.userID.values, train.itemID.values)),
                      shape=(n_users, n_items))
    R.data[:] = 1.0                                   # binary (duplicates count once, like the set-based original)
    C = (R @ R.T).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    out = {}
    for u in range(n_users):
        s, e = C.indptr[u], C.indptr[u + 1]
        cols, vals = C.indices[s:e], C.data[s:e]
        order = np.argsort(-vals, kind='stable')[:top]
        out[u] = [cols[order].tolist(), vals[order].astype(float).tolist()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='MMRec dataset name, e.g. ml1m, dbbook, ml1m_vit')
    ap.add_argument('--inter', required=True)
    ap.add_argument('--feat_dir', required=True)
    ap.add_argument('--image', default='vgg', help='vgg | resnet152 | vit_cls | vit_avg')
    ap.add_argument('--text', default='all-MiniLM-L6-v2', help='all-MiniLM-L6-v2 | all-mpnet-base-v2')
    ap.add_argument('--core', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--config_dir', default='src/configs/dataset')
    ap.add_argument('--template', default='baby', help='dataset config to copy (field names, feature file names)')
    args = ap.parse_args()

    df = pd.read_csv(args.inter, sep='\t')
    n0 = (len(df), df.userID.nunique(), df.itemID.nunique())
    pos = df[df.rating == 1].drop(columns=['x_label'])
    pos = pos.drop_duplicates(['userID', 'itemID'])
    pos = core_filter(pos, args.core)
    pos = split(pos, args.seed)

    users = np.sort(pos.userID.unique())
    items = np.sort(pos.itemID.unique())               # their item index == their feature row
    umap = {u: i for i, u in enumerate(users)}
    imap = {it: i for i, it in enumerate(items)}
    out = pos.copy()
    out['userID'] = out.userID.map(umap)
    out['itemID'] = out.itemID.map(imap)
    out = out.sort_values(['x_label', 'userID']).reset_index(drop=True)

    dst = os.path.join(args.data_dir, args.name)
    os.makedirs(dst, exist_ok=True)
    cols = ['userID', 'itemID', 'rating'] + (['timestamp'] if 'timestamp' in out.columns else []) + ['x_label']
    out[cols].to_csv(os.path.join(dst, args.name + '.inter'), sep='\t', index=False)

    for mod, fname in (('image', args.image), ('text', args.text)):
        X = np.load(os.path.join(args.feat_dir, fname + '.npy')).astype(np.float32)
        if X.shape[0] <= items.max():
            raise SystemExit('{} has {} rows but item index {} is used'.format(fname, X.shape[0], items.max()))
        np.save(os.path.join(dst, '{}_feat.npy'.format(mod)), X[items])

    train = out[out.x_label == 0]
    np.save(os.path.join(dst, 'user_graph_dict.npy'), user_graph(train, len(users), len(items)), allow_pickle=True)

    tpl = open(os.path.join(args.config_dir, args.template + '.yaml')).read()
    tpl = tpl.replace("'{}.inter'".format(args.template), "'{}.inter'".format(args.name))
    with open(os.path.join(args.config_dir, args.name + '.yaml'), 'w') as f:
        f.write('# {}: converted by tools/convert_spillo.py from {} (positives only, {}-core, 80/10/10 seed {}), '
                'image={} text={}\n'.format(args.name, os.path.basename(args.inter), args.core, args.seed,
                                             args.image, args.text) + tpl)

    stats = {'source_rows_users_items': n0, 'users': len(users), 'items': len(items), 'interactions': len(out),
             'split': out.x_label.value_counts().sort_index().to_dict(),
             'density': len(out) / (len(users) * len(items)), 'image': args.image, 'text': args.text,
             'min_user_deg': int(out.groupby('userID').size().min()),
             'min_item_deg': int(out.groupby('itemID').size().min())}
    json.dump(stats, open(os.path.join(dst, 'convert_meta.json'), 'w'), indent=1, default=int)
    print(json.dumps(stats, indent=1, default=int))


if __name__ == '__main__':
    main()
