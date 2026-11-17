# coding: utf-8
"""
LightGCNC: LightGCN whose layer-0 ITEM embedding comes from a chosen source (mmj, JEPA-Y teacher study).

    --item_input=id       free item-ID table (identical to LightGCN)
    --item_input=text     E(t_i): 2-layer MLP on the standardised text feature, no item-ID parameters
    --item_input=visual   E(v_i): same on the visual feature
    --item_input=tv       E([t_i; v_i])

Users keep free ID embeddings; propagation, BPR loss and the L2 term on layer-0 embeddings are those of
LightGCN, so the sources are compared under the same model. With --n_layers=0 the score is u . E(x_i)
(MF with content items): item identity enters only through the content.
Features are standardised per dimension over all items (no labels involved).

    cd src
    python main.py -m LightGCNC -d baby --item_input=text --exp_name=text_grid
"""
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from models.lightgcn import LightGCN

SOURCES = ('id', 'text', 'visual', 'tv')


class LightGCNC(LightGCN):
    def __init__(self, config, dataset):
        GeneralRecommender.__init__(self, config, dataset)
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype('float32')
        self.latent_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.item_input = config['item_input'] or 'id'
        if self.item_input not in SOURCES:
            raise ValueError('item_input must be one of {}'.format(SOURCES))
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        init = nn.init.xavier_uniform_
        params = {'user_emb': nn.Parameter(init(torch.empty(self.n_users, self.latent_dim)))}
        if self.item_input == 'id':
            params['item_emb'] = nn.Parameter(init(torch.empty(self.n_items, self.latent_dim)))
        self.embedding_dict = nn.ParameterDict(params)
        if self.item_input != 'id':
            feats = []
            if self.item_input in ('text', 'tv'):
                assert self.t_feat is not None, 'text_feat.npy not loaded'
                feats.append(self.t_feat)
            if self.item_input in ('visual', 'tv'):
                assert self.v_feat is not None, 'image_feat.npy not loaded'
                feats.append(self.v_feat)
            X = torch.cat([self._standardize(f.detach()) for f in feats], 1)
            assert X.shape[0] == self.n_items, 'feature rows {} != items {}'.format(X.shape[0], self.n_items)
            self.register_buffer('item_feat', X)
            hidden = config['proj_hidden'] or 256
            if (config['proj_layers'] or 2) == 1:
                self.item_proj = nn.Linear(X.shape[1], self.latent_dim)
            else:
                self.item_proj = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.LeakyReLU(),
                                               nn.Linear(hidden, self.latent_dim))
            self.feat_dropout = nn.Dropout(config['feat_dropout'] or 0.0)
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

    @staticmethod
    def _standardize(f):
        f = f.float()
        sd = f.std(0, keepdim=True)
        return (f - f.mean(0, keepdim=True)) / torch.where(sd < 1e-6, torch.ones_like(sd), sd)

    def item_ego(self):
        if self.item_input == 'id':
            return self.embedding_dict['item_emb']
        return self.item_proj(self.feat_dropout(self.item_feat))

    def get_ego_embeddings(self):
        return torch.cat([self.embedding_dict['user_emb'], self.item_ego()], 0)

    def calculate_loss(self, interaction):
        user, pos_item, neg_item = interaction[0], interaction[1], interaction[2]
        ego_i = self.item_ego()
        all_embeddings = torch.cat([self.embedding_dict['user_emb'], ego_i], 0)
        layers = [all_embeddings]
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            layers.append(all_embeddings)
        out = torch.stack(layers, 1).mean(1)
        u_all, i_all = out[:self.n_users], out[self.n_users:]
        u, p, n = u_all[user], i_all[pos_item], i_all[neg_item]
        mf_loss = self.mf_loss((u * p).sum(1), (u * n).sum(1))
        reg_loss = self.reg_loss(self.embedding_dict['user_emb'][user], ego_i[pos_item], ego_i[neg_item])
        return mf_loss + self.reg_weight * reg_loss
