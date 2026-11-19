# coding: utf-8
"""
LightGCNAlign: an ID branch and a Text branch trained jointly, with an AlignRec-style InfoNCE between them
(Content-Category Alignment of AlignRec, CIKM 2024, applied to two existing branches; not a re-implementation
of AlignRec's encoders / regularisers).

Branches (separate parameters, same user-item graph, LightGCN propagation, mean of layers):
    ID    layer 0: user table u^ID, item table i^ID                 -> U^ID, Y^ID
    Text  layer 0: user table u^T,  item E_T(t_i) (frozen standardised text feature, MLP d -> 256 -> 64)
                                                                     -> U^T,  T^G
Loss (per batch of (u, i+, i-)):
    BPR(<U^ID_u, Y^ID_i>) + BPR(<U^T_u, T^G_i>) + reg_id * ||ID layer 0||^2 + reg_t * ||Text layer 0||^2
    + nce_weight * [ L_I  (+ L_U if nce_user) ]
    L_I = -1/B sum_i log exp(s(T^G_i, Y^ID_i)/tau) / sum_j exp(s(T^G_i, Y^ID_j)/tau)       (AlignRec Eq. 10)
        over the UNIQUE items of the batch (positives and negatives); s = cosine (nce_norm=True, default) or
        dot product; nce_sym=True also adds the ID -> Text direction (mean of the two).
        Items outside the train part of data/<ds>/jepay_item_split.npz are never in L_I (nce_holdout=True,
        default), so ID-Text matching can be measured on items the alignment never saw.
    L_U the same over the unique users of the batch: s(U^T_u, U^ID_u).
    No stop-gradient: L_I / L_U update both branches.
Configs: C0 nce_weight=0 (joint training, no alignment), C1 nce_weight>0, C2 nce_weight>0 and nce_user=True.
Ranking used by the trainer (model selection / logged test): eval_score = sum (<U^ID, Y^ID> + <U^T, T^G>,
i.e. the concatenated model), id, or text. ID alone, Text alone and SF(ID, Text) are computed afterwards from
the taps (tools/align_eval.py).
After every epoch: mean L_I / L_U, and the norm of the gradient of nce_weight * L_I on the text MLP, the text
user table and the ID item table (first batch of the epoch), so that one can check the alignment reaches them.

    cd src
    python main.py -m LightGCNAlign -d baby --nce_weight=0 --exp_name=align_C0
    python main.py -m LightGCNAlign -d baby --nce_weight="[0.01,0.1,0.5]" --nce_tau="[0.1,0.2]" --exp_name=align_C1
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from models.lightgcn import LightGCN


class LightGCNAlign(LightGCN):
    def __init__(self, config, dataset):
        GeneralRecommender.__init__(self, config, dataset)
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype('float32')
        self.latent_dim = d = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_id = config['reg_id'] if config['reg_id'] is not None else 1e-4
        self.reg_t = config['reg_t'] if config['reg_t'] is not None else 1e-2
        self.nce_weight = float(config['nce_weight'] or 0.0)
        self.nce_tau = float(config['nce_tau'] or 0.2)
        self.nce_user = bool(config['nce_user'])
        self.nce_sym = bool(config['nce_sym'])
        self.nce_norm = config['nce_norm'] is None or bool(config['nce_norm'])
        self.eval_score = config['eval_score'] or 'sum'
        assert self.eval_score in ('sum', 'id', 'text')
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        init = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict({
            'user_id': nn.Parameter(init(torch.empty(self.n_users, d))),
            'item_id': nn.Parameter(init(torch.empty(self.n_items, d))),
            'user_t': nn.Parameter(init(torch.empty(self.n_users, d))),
        })
        assert self.t_feat is not None, 'text_feat.npy not loaded'
        X = self._standardize(self.t_feat.detach())
        assert X.shape[0] == self.n_items, 'feature rows {} != items {}'.format(X.shape[0], self.n_items)
        self.register_buffer('item_feat', X)
        hidden = config['proj_hidden'] or 256
        self.item_proj = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.LeakyReLU(), nn.Linear(hidden, d))
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # items allowed in L_I: train part of the E-A item split (held-out items measure generalisation)
        ok = torch.ones(self.n_items, dtype=torch.bool)
        self.nce_holdout = config['nce_holdout'] is None or bool(config['nce_holdout'])
        if self.nce_holdout:
            path = os.path.join(os.path.abspath(config['data_path'] + config['dataset']), 'jepay_item_split.npz')
            if os.path.isfile(path):
                ok = torch.zeros(self.n_items, dtype=torch.bool)
                ok[torch.from_numpy(np.load(path)['train']).long()] = True
            else:
                print('[LightGCNAlign] {} not found: every item can enter L_I'.format(path))
        self.register_buffer('nce_item_ok', ok)
        self._stats = {'L_I': [], 'L_U': []}
        self._grad = None
        self._check_grad = True

    @staticmethod
    def _standardize(f):
        f = f.float()
        sd = f.std(0, keepdim=True)
        return (f - f.mean(0, keepdim=True)) / torch.where(sd < 1e-6, torch.ones_like(sd), sd)

    # ------------------------------------------------------------------ propagation
    def text_items0(self):
        return self.item_proj(self.item_feat)

    def _propagate(self, u0, i0):
        x = torch.cat([u0, i0], 0)
        layers = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj_matrix, x)
            layers.append(x)
        out = torch.stack(layers, 1).mean(1)
        return out[:self.n_users], out[self.n_users:]

    def branches(self):
        e = self.embedding_dict
        t0 = self.text_items0()
        U_id, Y_id = self._propagate(e['user_id'], e['item_id'])
        U_t, T_g = self._propagate(e['user_t'], t0)
        return U_id, Y_id, U_t, T_g, t0

    def forward(self):
        U_id, Y_id, U_t, T_g, _ = self.branches()
        if self.eval_score == 'id':
            return U_id, Y_id
        if self.eval_score == 'text':
            return U_t, T_g
        return torch.cat([U_id, U_t], 1), torch.cat([Y_id, T_g], 1)

    # ------------------------------------------------------------------ loss
    def _nce(self, q, k):
        if self.nce_norm:
            q, k = F.normalize(q, dim=1), F.normalize(k, dim=1)
        logits = q @ k.T / self.nce_tau
        labels = torch.arange(q.shape[0], device=q.device)
        loss = F.cross_entropy(logits, labels)
        if self.nce_sym:
            loss = 0.5 * (loss + F.cross_entropy(logits.T, labels))
        return loss

    def calculate_loss(self, interaction):
        user, pos, neg = interaction[0], interaction[1], interaction[2]
        e = self.embedding_dict
        U_id, Y_id, U_t, T_g, t0 = self.branches()
        u_id, u_t = U_id[user], U_t[user]
        bpr_id = self.mf_loss((u_id * Y_id[pos]).sum(1), (u_id * Y_id[neg]).sum(1))
        bpr_t = self.mf_loss((u_t * T_g[pos]).sum(1), (u_t * T_g[neg]).sum(1))
        reg = self.reg_id * self.reg_loss(e['user_id'][user], e['item_id'][pos], e['item_id'][neg]) + \
            self.reg_t * self.reg_loss(e['user_t'][user], t0[pos], t0[neg])
        loss = bpr_id + bpr_t + reg
        if self.nce_weight > 0:
            items = torch.unique(torch.cat([pos, neg]))
            items = items[self.nce_item_ok[items]]
            l_i = self._nce(T_g[items], Y_id[items]) if items.numel() > 1 else T_g.sum() * 0.0
            nce = l_i
            self._stats['L_I'].append(float(l_i))
            if self.nce_user:
                users = torch.unique(user)
                l_u = self._nce(U_t[users], U_id[users])
                nce = nce + l_u
                self._stats['L_U'].append(float(l_u))
            if self._check_grad and self.training:
                self._check_grad = False
                params = {'text_mlp': list(self.item_proj.parameters()), 'user_t': [e['user_t']],
                          'item_id': [e['item_id']], 'user_id': [e['user_id']]}
                flat = [p for ps in params.values() for p in ps]
                g = torch.autograd.grad(self.nce_weight * nce, flat, retain_graph=True, allow_unused=True)
                it, self._grad = iter(g), {}
                for k, ps in params.items():
                    sq = 0.0
                    for _ in ps:
                        gi = next(it)
                        sq += 0.0 if gi is None else float((gi ** 2).sum())
                    self._grad[k] = sq ** 0.5
            loss = loss + self.nce_weight * nce
        return loss

    def post_epoch_processing(self):
        self._check_grad = True
        if self.nce_weight <= 0:
            return None
        msg = 'L_I {:.4f}'.format(np.mean(self._stats['L_I'])) if self._stats['L_I'] else ''
        if self._stats['L_U']:
            msg += ' L_U {:.4f}'.format(np.mean(self._stats['L_U']))
        if self._grad:
            msg += ' | grad of nce: ' + ' '.join('{} {:.2e}'.format(k, v) for k, v in self._grad.items())
        self._stats = {'L_I': [], 'L_U': []}
        return msg

    def full_sort_predict(self, interaction):
        U, Y = self.forward()
        return U[interaction[0]] @ Y.T
