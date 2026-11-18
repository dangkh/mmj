# coding: utf-8
"""
LightGCNC: LightGCN whose layer-0 ITEM embedding comes from a chosen source (mmj, JEPA-Y teacher study).

Single channel (64-d):
    --item_input=id       free item-ID table (identical to LightGCN)
    --item_input=text     E(t_i): 2-layer MLP on the standardised text feature, no item-ID parameters
    --item_input=visual   E(v_i): same on the visual feature
    --item_input=tv       E([t_i; v_i])
    --item_input=const    one learned vector shared by ALL items (no item information at layer 0): items
                          differ only through propagation, i.e. graph + user embeddings (control)
Two channels, ID + content, as DRAGON's construction='cat' (2 x 64-d):
    --item_input=id+text | id+visual | id+tv
    item  [i_id ; E(x_i)]                       (concatenated)
    user  [a_u * u_id ; (1 - a_u) * u_c]        u_id, u_c: separate user embeddings per channel
    score a_u <u_id, i_id> + (1 - a_u) <u_c, E(x_i)>    (after LightGCN propagation of each channel)
    --user_weight=softmax  a_u = softmax of 2 learned logits per user, init 0.5 (default)
    --user_weight=free     DRAGON's exact form: 2 free weights per user, initialised as a softmax of
                           random values and then unconstrained
    --user_weight=none     both weights 1 (plain concatenation, no learned mix)
    The mean / sd of a_u over users is logged after every epoch.
Item-item graph, as DRAGON (any source above):
    --item_graph=text | visual | tv     frozen kNN graph (cosine, top knn_k=10, symmetric normalisation) built
                                        from the RAW features; tv = mm_image_weight * A_v + (1 - w) * A_t (w=0.1)
    After the user-item propagation the (fused) item representation is smoothed on it:
        h = A^{n_mm_layers} i,   i <- i + h          (n_mm_layers = 1 by default)
    --item_graph=none (default): no item-item graph.

Content features are FROZEN (a buffer, standardised per dimension over all items): BPR updates only the
MLP E (Linear d -> 256, LeakyReLU, Linear 256 -> 64), as in DRAGON's GCN.MLP / MLP_1. Users keep free ID
embeddings; propagation, BPR and the L2 term on layer-0 embeddings are those of LightGCN. With
--n_layers=0 the score uses layer-0 embeddings only (no propagation).

    cd src
    python main.py -m LightGCNC -d baby --item_input=text --exp_name=text_grid
    python main.py -m LightGCNC -d baby --item_input=id+text --exp_name=id+text_grid
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from models.lightgcn import LightGCN

SOURCES = ('id', 'text', 'visual', 'tv', 'const', 'id+text', 'id+visual', 'id+tv')
USER_WEIGHTS = ('softmax', 'free', 'none')
ITEM_GRAPHS = ('none', 'text', 'visual', 'tv')
# Which propagation layers form the output (bipartite graph: even item layers come from item vectors, odd item
# layers from the user table; even user layers from the user table, odd user layers from item vectors):
#   all   items and users: mean of all layers (LightGCN)
#   to    items: mean of the even layers (text-origin pathway); users: mean of all layers
#   pure  items: even layers; users: odd layers -> the score uses item vectors only (no free user table)
LAYER_MODES = ('all', 'to', 'pure')


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
        self.has_id = self.item_input == 'id' or self.item_input.startswith('id+')
        self.content = self.item_input.split('+', 1)[1] if self.item_input.startswith('id+') else \
            (None if self.item_input in ('id', 'const') else self.item_input)
        self.two_channel = self.has_id and self.content is not None
        self.user_weight = config['user_weight'] or 'softmax'
        if self.user_weight not in USER_WEIGHTS:
            raise ValueError('user_weight must be one of {}'.format(USER_WEIGHTS))
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        init = nn.init.xavier_uniform_
        d = self.latent_dim
        params = {'user_emb': nn.Parameter(init(torch.empty(self.n_users, d)))}
        if self.has_id:
            params['item_emb'] = nn.Parameter(init(torch.empty(self.n_items, d)))
        if self.item_input == 'const':
            params['item_const'] = nn.Parameter(init(torch.empty(1, d)))
        if self.two_channel:
            params['user_emb_c'] = nn.Parameter(init(torch.empty(self.n_users, d)))
            if self.user_weight == 'softmax':
                params['user_logit'] = nn.Parameter(torch.zeros(self.n_users, 2))           # a_u = 0.5 at start
            elif self.user_weight == 'free':
                w = nn.init.xavier_normal_(torch.randn(self.n_users, 2, 1)).squeeze(-1)  # as DRAGON.weight_u
                params['user_w'] = nn.Parameter(F.softmax(w, dim=1))
        self.embedding_dict = nn.ParameterDict(params)

        if self.content is not None:
            feats = []
            if self.content in ('text', 'tv'):
                assert self.t_feat is not None, 'text_feat.npy not loaded'
                feats.append(self.t_feat)
            if self.content in ('visual', 'tv'):
                assert self.v_feat is not None, 'image_feat.npy not loaded'
                feats.append(self.v_feat)
            X = torch.cat([self._standardize(f.detach()) for f in feats], 1)
            assert X.shape[0] == self.n_items, 'feature rows {} != items {}'.format(X.shape[0], self.n_items)
            self.register_buffer('item_feat', X)                 # frozen: not a parameter
            hidden = config['proj_hidden'] or 256
            if (config['proj_layers'] or 2) == 1:
                self.item_proj = nn.Linear(X.shape[1], d)
            else:
                self.item_proj = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.LeakyReLU(), nn.Linear(hidden, d))
            self.feat_dropout = nn.Dropout(config['feat_dropout'] or 0.0)
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.layer_mode = config['layer_mode'] or 'all'
        if self.layer_mode not in LAYER_MODES:
            raise ValueError('layer_mode must be one of {}'.format(LAYER_MODES))
        if self.layer_mode == 'pure' and self.n_layers < 1:
            raise ValueError('layer_mode=pure needs n_layers >= 1 (users are built from odd layers)')

        self.item_graph = config['item_graph'] or 'none'
        if self.item_graph not in ITEM_GRAPHS:
            raise ValueError('item_graph must be one of {}'.format(ITEM_GRAPHS))
        self.mm_adj = None
        if self.item_graph != 'none':
            self.knn_k = config['knn_k'] or 10
            self.n_mm_layers = config['n_mm_layers'] if config['n_mm_layers'] is not None else 1
            w_img = config['mm_image_weight'] if config['mm_image_weight'] is not None else 0.1
            adj = None
            if self.item_graph in ('visual', 'tv'):
                assert self.v_feat is not None, 'image_feat.npy not loaded'
                adj = self.knn_adj(self.v_feat.detach()) * (w_img if self.item_graph == 'tv' else 1.0)
            if self.item_graph in ('text', 'tv'):
                assert self.t_feat is not None, 'text_feat.npy not loaded'
                a_t = self.knn_adj(self.t_feat.detach()) * ((1.0 - w_img) if self.item_graph == 'tv' else 1.0)
                adj = a_t if adj is None else (adj + a_t).coalesce()
            self.mm_adj = adj

    def knn_adj(self, feat):
        """DRAGON.get_knn_adj_mat + compute_normalized_laplacian: top-k cosine neighbours (self included),
        values 1 / sqrt(deg_row * deg_col)."""
        x = feat.float().to(self.device)
        x = x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        n = x.shape[0]
        knn = torch.cat([torch.topk(x[s:s + 4096] @ x.T, self.knn_k, dim=-1)[1] for s in range(0, n, 4096)], 0)
        rows = torch.arange(n, device=self.device).unsqueeze(1).expand(-1, self.knn_k).flatten()
        idx = torch.stack([rows, knn.flatten()], 0)
        deg = torch.zeros(n, device=self.device).index_add_(0, idx[0], torch.ones(idx.shape[1], device=self.device))
        r = (deg + 1e-7).pow(-0.5)
        return torch.sparse_coo_tensor(idx, r[idx[0]] * r[idx[1]], (n, n)).coalesce()

    @staticmethod
    def _standardize(f):
        f = f.float()
        sd = f.std(0, keepdim=True)
        return (f - f.mean(0, keepdim=True)) / torch.where(sd < 1e-6, torch.ones_like(sd), sd)

    # ------------------------------------------------------------------------- embeddings
    def ego(self):
        """Layer-0 user and item embeddings (both channels side by side when two_channel)."""
        users, items = [self.embedding_dict['user_emb']], []
        if self.has_id:
            items.append(self.embedding_dict['item_emb'])
        if self.content is not None:
            items.append(self.item_proj(self.feat_dropout(self.item_feat)))
        if self.item_input == 'const':
            items.append(self.embedding_dict['item_const'].expand(self.n_items, -1))
        if self.two_channel:
            users.append(self.embedding_dict['user_emb_c'])
        return torch.cat(users, 1), torch.cat(items, 1)

    @torch.no_grad()
    def item_layers(self):
        """Item embeddings at every propagation layer, [T^(0), T^(1), ..., T^(L)] (before averaging and
        before the optional item-item graph). T^(0) is the layer-0 item embedding (e.g. E_T(text))."""
        u0, i0 = self.ego()
        x = torch.cat([u0, i0], 0)
        layers = [x[self.n_users:]]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj_matrix, x)
            layers.append(x[self.n_users:])
        return layers

    def user_weights(self):
        """(n_users, 2): weight of the ID channel and of the content channel."""
        if self.user_weight == 'softmax':
            return F.softmax(self.embedding_dict['user_logit'], dim=1)
        if self.user_weight == 'free':
            return self.embedding_dict['user_w']
        return torch.ones(self.n_users, 2, device=self.device)

    def propagate(self, u0, i0):
        x = torch.cat([u0, i0], 0)
        layers = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj_matrix, x)
            layers.append(x)
        if self.layer_mode == 'all':
            out = torch.stack(layers, 1).mean(1)
            u, i = out[:self.n_users], out[self.n_users:]
        else:
            i = torch.stack(layers[0::2], 1).mean(1)[self.n_users:]
            u = torch.stack(layers if self.layer_mode == 'to' else layers[1::2], 1).mean(1)[:self.n_users]
        if self.two_channel:
            w = self.user_weights()
            d = self.latent_dim
            u = torch.cat([w[:, :1] * u[:, :d], w[:, 1:] * u[:, d:]], 1)
        if self.mm_adj is not None:                  # item-item graph on the fused item representation
            h = i
            for _ in range(self.n_mm_layers):
                h = torch.sparse.mm(self.mm_adj, h)
            i = i + h
        return u, i

    def get_ego_embeddings(self):
        u0, i0 = self.ego()
        return torch.cat([u0, i0], 0)

    def forward(self):
        return self.propagate(*self.ego())

    def calculate_loss(self, interaction):
        user, pos_item, neg_item = interaction[0], interaction[1], interaction[2]
        u0, i0 = self.ego()
        u_all, i_all = self.propagate(u0, i0)
        u, p, n = u_all[user], i_all[pos_item], i_all[neg_item]
        mf_loss = self.mf_loss((u * p).sum(1), (u * n).sum(1))
        reg_loss = self.reg_loss(u0[user], i0[pos_item], i0[neg_item])
        return mf_loss + self.reg_weight * reg_loss

    def post_epoch_processing(self):
        if not self.two_channel or self.user_weight == 'none':
            return None
        with torch.no_grad():
            w = self.user_weights()
            a = w[:, 0] / w.sum(1).clamp_min(1e-12)        # share of the ID channel
        return 'user weight of the ID channel a_u: mean {:.3f}, sd {:.3f}, min {:.3f}, max {:.3f}'.format(
            a.mean().item(), a.std().item(), a.min().item(), a.max().item())
