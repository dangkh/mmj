# coding: utf-8
"""
XSimGCL (Yu et al., "XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation", TKDE 2023).

    encoder     LightGCN propagation; during training a random noise is added after every layer:
                e <- e + sign(e) * normalize(noise) * eps
                final embedding = mean of layers 1..L (layer 0 excluded, as in the reference code)
    loss        BPR on the perturbed final embeddings + reg_weight * ||E_0||^2
                + cl_weight * [InfoNCE(final, layer layer_cl) for users and for items] over the unique users /
                positive items of the batch, cosine / cl_tau (cross-layer contrast)
    inference   the same encoder without noise
"""
import numpy as np
import torch
import torch.nn.functional as F

from models.lightgcn import LightGCN
from models.sgl import info_nce


class XSimGCL(LightGCN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        g = lambda k, d: d if config[k] is None else config[k]  # noqa: E731
        self.eps = float(g('eps', 0.2))
        self.cl_weight = float(g('cl_weight', 0.1))
        self.cl_tau = float(g('cl_tau', 0.2))
        self.layer_cl = int(g('layer_cl', 1))
        self._cl = []

    def _encode(self, perturbed):
        x = self.get_ego_embeddings()
        layers, cl = [], None
        for k in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj_matrix, x)
            if perturbed:
                x = x + torch.sign(x) * F.normalize(torch.rand_like(x), dim=-1) * self.eps
            layers.append(x)
            if k == self.layer_cl - 1:
                cl = x
        out = torch.stack(layers, 1).mean(1)
        return out[:self.n_users], out[self.n_users:], cl[:self.n_users], cl[self.n_users:]

    def forward(self):
        u, i, _, _ = self._encode(False)
        return u, i

    def calculate_loss(self, interaction):
        user, pos, neg = interaction[0], interaction[1], interaction[2]
        U, Y, Ucl, Ycl = self._encode(True)
        u = U[user]
        bpr = self.mf_loss((u * Y[pos]).sum(1), (u * Y[neg]).sum(1))
        e = self.embedding_dict
        reg = self.reg_loss(e['user_emb'][user], e['item_emb'][pos], e['item_emb'][neg])
        users, items = torch.unique(user), torch.unique(pos)
        cl = info_nce(U[users], Ucl[users], self.cl_tau) + info_nce(Y[items], Ycl[items], self.cl_tau)
        self._cl.append(float(cl))
        return bpr + self.reg_weight * reg + self.cl_weight * cl

    def post_epoch_processing(self):
        if not self._cl:
            return None
        msg = '[xsimgcl] cl {:.4f}'.format(np.mean(self._cl))
        self._cl = []
        return msg
