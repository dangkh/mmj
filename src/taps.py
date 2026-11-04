# coding: utf-8
"""
mmj: extract item representations at fixed points ("taps") of a trained model,
for the frozen-representation probes in tools/probe_rec.py.

    T0     raw features x_v, x_t            -> read from data/<dataset>/ by the probe, not saved here
    T0ft_m input features as the model holds them after training (MMRec fine-tunes them,
           nn.Embedding.from_pretrained(..., freeze=False)) - only where the model reads them
    T1_m   modality representation after the projection, before any graph
    T1g_m  (MGCN only) ItemID * gate_m(T1_m): already mixed with the item ID
    T2_m   modality representation after the graph (only models that propagate it)
    T3     final item representation used for scoring (T3_user: final user representation)

Enabled with --save_taps=True. Taps are written at every best-validation epoch, so the
files on disk are always those of the epoch whose test result is reported.
Output: <taps_dir>/<model>-<dataset>-<exp_name>/{T*.npy, meta.json}
"""
import json
import os
from time import time

import numpy as np
import torch
import torch.nn.functional as F


def _spmm(adj, x, sparse=True):
    return torch.sparse.mm(adj, x) if sparse else torch.mm(adj, x)


def _dragon(m):
    out = {}
    for mod, feat, gcn in (('v', m.v_feat, getattr(m, 'v_gcn', None)), ('t', m.t_feat, getattr(m, 't_gcn', None))):
        if feat is None or gcn is None:
            continue
        # DRAGON feeds the raw features (m.v_feat) to its GCNs; image_embedding / image_trs are unused.
        if getattr(gcn, 'dim_latent', None):
            out['T1_' + mod] = gcn.MLP_1(F.leaky_relu(gcn.MLP(feat)))
        rep, _ = gcn(getattr(m, 'edge_index_drop' + mod), m.edge_index, feat)   # same call as DRAGON.forward
        out['T2_' + mod] = rep[m.n_users:]
    # result_embed is what full_sort_predict scores with (set by the last training forward)
    out['T3'] = m.result_embed[m.n_users:]
    out['T3_user'] = m.result_embed[:m.n_users]
    return out


def _projections(m, out):
    if m.v_feat is not None:
        out['T0ft_v'] = m.image_embedding.weight
        out['T1_v'] = m.image_trs(m.image_embedding.weight)
    if m.t_feat is not None:
        out['T0ft_t'] = m.text_embedding.weight
        out['T1_t'] = m.text_trs(m.text_embedding.weight)
    return out


def _freedom(m):
    # img_trs / txt_trs only enter the auxiliary BPR loss; scoring uses IDs + the frozen kNN graph
    out = _projections(m, {})
    u, i = m.forward(m.norm_adj)                      # as in full_sort_predict
    out['T3'], out['T3_user'] = i, u
    return out


def _lattice(m):
    # projections only shape the learned item-item graph; h propagates item IDs over it
    out = _projections(m, {})
    u, i = m.forward(m.norm_adj, build_item_graph=True)   # as in full_sort_predict
    out['T3'], out['T3_user'] = i, u
    return out


def _mgcn(m):
    out = _projections(m, {})
    ids = m.item_id_embedding.weight
    for mod, gate, adj in (('v', m.gate_v, getattr(m, 'image_original_adj', None)),
                           ('t', m.gate_t, getattr(m, 'text_original_adj', None))):
        if 'T1_' + mod not in out:
            continue
        x = torch.multiply(ids, gate(out['T1_' + mod]))       # behavior-guided purifier
        out['T1g_' + mod] = x
        for _ in range(m.n_layers):                           # item-item view, as in MGCN.forward
            x = _spmm(adj, x, m.sparse)
        out['T2_' + mod] = x
    u, i = m.forward(m.norm_adj)                              # as in full_sort_predict
    out['T3'], out['T3_user'] = i, u
    return out


def _lightgcn(m):
    u, i = m.forward()
    return {'T3': i, 'T3_user': u}


ADAPTERS = {
    'DRAGON': _dragon,
    'FREEDOM': _freedom,
    'LATTICE': _lattice,
    'MGCN': _mgcn,
    'LIGHTGCN': _lightgcn,
}


@torch.no_grad()
def extract_taps(model):
    fn = ADAPTERS.get(model.__class__.__name__.upper())
    if fn is None:
        raise KeyError('no tap adapter for {} (available: {})'.format(model.__class__.__name__, sorted(ADAPTERS)))
    was_training = model.training
    model.eval()
    try:
        taps = fn(model)
    finally:
        model.train(was_training)
    return {k: v.detach().float().cpu().numpy() for k, v in taps.items() if v is not None}


def save_taps(model, config, epoch, valid_result, test_result, logger=None):
    """Called by the trainer at every best-validation epoch. Never stops training."""
    t0 = time()
    path = config['taps_path'] or os.path.join('../taps', '{}-{}'.format(config['model'], config['dataset']))
    try:
        taps = extract_taps(model)
        os.makedirs(path, exist_ok=True)
        for k, v in taps.items():
            np.save(os.path.join(path, k + '.npy'), v.astype(np.float32))
        hyper = {k: config[k] for k in (config['hyper_parameters'] or [])}
        meta = {
            'model': config['model'], 'dataset': config['dataset'], 'exp_name': config['exp_name'],
            'epoch': int(epoch), 'hyper_parameters': {k: (v if isinstance(v, (int, float, str, bool)) or v is None
                                                          else str(v)) for k, v in hyper.items()},
            'valid': {k: float(v) for k, v in valid_result.items()},
            'test': {k: float(v) for k, v in test_result.items()},
            'shapes': {k: list(v.shape) for k, v in taps.items()},
        }
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        if logger is not None:
            logger.info('[taps] epoch {} -> {} ({}) {:.1f}s'.format(
                epoch, path, ', '.join('{}{}'.format(k, list(v.shape)) for k, v in sorted(taps.items())), time() - t0))
    except Exception as e:     # probing must never break a reproduction run
        if logger is not None:
            logger.warning('[taps] not saved: {!r}'.format(e))
