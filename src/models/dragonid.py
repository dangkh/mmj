# coding: utf-8
"""
DRAGONID (mmj): DRAGON with its visual branch replaced by an item-ID branch, SCOPE-style
(https://github.com/dangkh/SCOPE, models/scope.py).

Only the visual GCN changes; everything else (text GCN, user-user graph, item-item graph,
per-user modality weights, BPR + reg) is DRAGON's own code.

    visual branch (DRAGON)  : x_i = MLP(v_i)  (4096 -> 256 -> 64), user preference P_v, GCN
    ID branch (this model)  : x_i = E_id[i]   (nn.Embedding, no MLP), user preference P_id, same GCN

As in SCOPE the ID table is fed to the GCN directly (DRAGON's GCN with dim_latent=None, i.e. no
MLP), keeps nn.Embedding's default N(0, 1) init (--id_init=normal) and is not L2-regularised; GCN
rows are L2-normalised, so the init scale only changes the training dynamics. The item-item graph
should come from text only, as in SCOPE: train with --mm_image_weight=0.0 on a data folder whose
cached mm_adj_<k>.pt was built that way (e.g. data/baby_vcopy).

    cd src && python main.py -m DRAGONID -d baby_vcopy --mm_image_weight=0.0 --learning_rate=0.0001 --reg_weight=0.001
"""
import torch.nn as nn

from models.dragon import DRAGON, GCN


class DRAGONID(DRAGON):
    def __init__(self, config, dataset):
        super(DRAGONID, self).__init__(config, dataset)
        if self.v_feat is None:
            raise ValueError('DRAGONID replaces the visual branch: the dataset must have image features')
        id_dim = config['id_dim'] or 64
        self.id_embedding = nn.Embedding(self.n_items, id_dim)       # default init: N(0, 1), as in SCOPE
        if (config['id_init'] or 'normal') == 'xavier':
            nn.init.xavier_normal_(self.id_embedding.weight)
        self.id_embedding.to(self.device)
        # DRAGON's GCN without MLP (dim_latent=None): x = normalize([P_id ; E_id]), 2 hops, x + h + h1
        self.v_gcn = GCN(self.dataset, config['train_batch_size'], self.num_user, self.num_item, config['embedding_size'],
                         self.aggr_mode, num_layer=self.num_layer, has_id=True, dropout=self.drop_rate,
                         dim_latent=None, device=self.device, features=self.id_embedding.weight)
        # DRAGON.forward feeds self.v_feat to self.v_gcn: point it at the ID table (same Parameter)
        self.v_feat = self.id_embedding.weight
