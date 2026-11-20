# coding: utf-8
"""
DirectAU (Wang et al., "Towards Representation Alignment and Uniformity in Collaborative Filtering", KDD 2022) with
the LightGCN encoder.

    loss        align + gamma * (uniform(users) + uniform(items)) / 2   (+ reg_weight * ||E_0||^2, default 0)
                align   = E_{(u, i) in batch} || f(u) - f(i) ||^2 on L2-normalised embeddings
                uniform = log E exp(-2 || f(x) - f(y) ||^2) over the pairs of the batch
                no negative sampling (use_neg_sampling: False)
    inference   dot product of the LightGCN embeddings, as LightGCN
"""
import torch
import torch.nn.functional as F

from models.lightgcn import LightGCN


def _uniform(x):
    return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()


class DirectAU(LightGCN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.gamma = float(1.0 if config['gamma'] is None else config['gamma'])

    def calculate_loss(self, interaction):
        user, pos = interaction[0], interaction[1]
        U, Y = self.forward()
        u, i = F.normalize(U[user], dim=1), F.normalize(Y[pos], dim=1)
        align = (u - i).norm(dim=1).pow(2).mean()
        uniform = (_uniform(u) + _uniform(i)) / 2
        loss = align + self.gamma * uniform
        if self.reg_weight:
            e = self.embedding_dict
            loss = loss + self.reg_weight * self.reg_loss(e['user_emb'][user], e['item_emb'][pos])
        return loss
