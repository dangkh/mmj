# coding: utf-8
"""
SGL (Wu et al., "Self-supervised Graph Learning for Recommendation", SIGIR 2021) on the LightGCN backbone.

    main task   BPR on the full training graph G (LightGCN) + reg_weight * ||E_0||^2
    views       two augmented graphs, regenerated every epoch:
                  ED  edge dropout (drop_ratio of the edges, each view independently)
                  ND  node dropout (drop_ratio of the users and of the items lose all their edges)
                  RW  edge dropout with a different sub-graph at every layer
    SSL         ssl_weight * [InfoNCE(U1, U2) + InfoNCE(Y1, Y2)] over the unique users / positive items of the batch,
                cosine similarity / ssl_tau, the other nodes of the batch as negatives (symmetric in the pair)
    inference   LightGCN on G
"""
import numpy as np
import torch
import torch.nn.functional as F

from models.lightgcn import LightGCN
from models.graph_aug import EdgeSet


def info_nce(a, b, tau):
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    logits = a @ b.T / tau
    labels = torch.arange(a.shape[0], device=a.device)
    return F.cross_entropy(logits, labels)


class SGL(LightGCN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        g = lambda k, d: d if config[k] is None else config[k]  # noqa: E731
        self.sgl_type = str(g('sgl_type', 'ED')).upper()
        assert self.sgl_type in ('ED', 'ND', 'RW')
        self.ssl_weight = float(g('ssl_weight', 0.1))
        self.ssl_tau = float(g('ssl_tau', 0.2))
        self.drop_ratio = float(g('drop_ratio', 0.1))
        self.edges = EdgeSet(self.interaction_matrix, self.n_users, self.n_items)
        self.rng = np.random.default_rng(int(config['seed'] or 0) + 23)
        self.views = None
        self._ssl = []

    def _view(self):
        """One augmented view: list of n_layers adjacencies (the same one repeated except for RW)."""
        E, r = self.edges, self.drop_ratio
        if self.sgl_type == 'ED':
            return [E.norm_adj(E.mask('uniform', r, self.rng), self.device)] * self.n_layers
        if self.sgl_type == 'ND':
            return [E.norm_adj(E.node_mask(r, r, self.rng), self.device)] * self.n_layers
        return [E.norm_adj(E.mask('uniform', r, self.rng), self.device) for _ in range(self.n_layers)]

    def pre_epoch_processing(self):
        if self.ssl_weight > 0:
            self.views = (self._view(), self._view())

    def _propagate(self, adjs):
        x = self.get_ego_embeddings()
        layers = [x]
        for adj in adjs:
            x = torch.sparse.mm(adj, x)
            layers.append(x)
        out = torch.stack(layers, 1).mean(1)
        return out[:self.n_users], out[self.n_users:]

    def calculate_loss(self, interaction):
        loss = super().calculate_loss(interaction)
        if self.ssl_weight <= 0:
            return loss
        user, pos = interaction[0], interaction[1]
        U1, Y1 = self._propagate(self.views[0])
        U2, Y2 = self._propagate(self.views[1])
        users, items = torch.unique(user), torch.unique(pos)
        ssl = info_nce(U1[users], U2[users], self.ssl_tau) + info_nce(Y1[items], Y2[items], self.ssl_tau)
        self._ssl.append(float(ssl))
        return loss + self.ssl_weight * ssl

    def post_epoch_processing(self):
        if not self._ssl:
            return None
        msg = '[sgl-{}] ssl {:.4f}'.format(self.sgl_type, np.mean(self._ssl))
        self._ssl = []
        return msg
