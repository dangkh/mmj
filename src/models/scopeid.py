# coding: utf-8
"""
SCOPEID: ID-only ablation of SCOPE (github.com/dangkh/scope, src/models/scope.py).

Kept exactly as in SCOPE:
  * GCN / Base_gcn: x = normalize([user_pref ; item_emb]); h1 = A x; h2 = A h1; out = x + h1 + h2
    (symmetric degree norm computed on the edge set passed in, no self loops).
  * low-exposure graph: drop ALL train edges of the top `low_high_ratio` (10%) items by train degree.
  * "full" branch runs on the full train graph (SCOPE calls it idh_gcn / id_embedding_high).
  * user/item representations are concatenated, score = dot product.
  * user gate on the full branch: scale = 0.5 + 0.5 * sigmoid(out_proj([userW, gate(LN(u_full - u_low))])).
  * loss = -mean(log2 sigmoid(pos - neg)) + 0.003 * sum_b mean(pref_b[users]^2).
  * optional item-item kNN graph (text) applied to the concatenated item rep: rep + A_knn^n rep.

Removed (needs text / LLM profiles):
  * text GCN branch (t_gcn) and user profile features;
  * semantic conflict c_u = 1 - cos(full_profile, local_profile) in the gate input.

Config (src/configs/model/SCOPEID.yaml; is_multimodal_model False, the text file is read only for 'text'):
  scope_branches : 'lf' (low + full, default) | 'ff' (two full-graph branches) | 'f' | 'l'
  scope_gate     : 'id' (gate without c_u) | 'none' (alpha = 1) | 'alpha' (one learned scalar on the LOW branch)
                   | 'alpha_u' (one learned scalar per user on the LOW branch)
                   | 'alpha_b' / 'alpha_bu' (learned, bounded: alpha = 2 sigmoid(theta) in (0, 2), init 1)
                   | 'fixed' (alpha = low_alpha_fixed, not learned; alpha = 0 gives the full branch alone)
                                                                                   [only used with 'lf']
  scope_combine  : 'cat' (SCOPE: concatenation, score = a * s_low + s_full for the alpha gates)
                   | 'sum' (z = z_full + a * z_low for users and items, then dot product)
  scope_item_graph: 'none' | 'text' (text kNN item-item graph, needs t_feat)
  low_high_ratio : share of top-degree items whose edges are dropped in the low view (SCOPE: 0.10)
Run: scripts/scopeid.pbs. Taps (--save_taps=True): T3 / T3_user = concatenated scoring vectors, plus per-branch
item / user representations (T3_<branch>, T3_user_<branch>).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree

from common.abstract_recommender import GeneralRecommender


class SCOPEID(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SCOPEID, self).__init__(config, dataset)
        self.num_user, self.num_item = self.n_users, self.n_items
        self.dim = config['feat_embed_dim'] if config['feat_embed_dim'] else 64
        self.knn_k = config['knn_k'] if config['knn_k'] else 10
        self.n_mm_layers = config['n_mm_layers'] if config['n_mm_layers'] else 1
        self.branches = config['scope_branches'] or 'lf'
        self.gate_mode = config['scope_gate'] or 'id'
        self.combine = config['scope_combine'] or 'cat'
        assert self.gate_mode in ('id', 'none', 'alpha', 'alpha_u', 'alpha_b', 'alpha_bu', 'fixed') \
            and self.combine in ('cat', 'sum')
        self.alpha_fixed = float(config['low_alpha_fixed'] if config['low_alpha_fixed'] is not None else 1.0)
        self.item_graph = config['scope_item_graph'] or 'none'
        self.high_ratio = config['low_high_ratio'] if config['low_high_ratio'] is not None else 0.10
        self.pref_reg = 0.003  # hard-coded in SCOPE

        # ---- graphs (same construction as SCOPE) ----
        inter = dataset.inter_matrix(form='coo').astype(np.float32)
        rows = inter.row.astype(np.int64)
        cols = inter.col.astype(np.int64)                     # item index (0..n_items-1)
        item_degree = np.bincount(cols, minlength=self.num_item)
        num_high = int(self.num_item * self.high_ratio)
        high_items = np.argsort(item_degree)[-num_high:] if num_high > 0 else np.array([], dtype=np.int64)
        is_high = np.zeros(self.num_item, dtype=bool)
        is_high[high_items] = True
        keep_low = ~is_high[cols]

        def to_undirected(r, c):
            e = torch.tensor(np.stack([r, c + self.num_user]), dtype=torch.long)
            return torch.cat((e, e[[1, 0]]), dim=1).contiguous().to(self.device)

        self.edge_index_full = to_undirected(rows, cols)
        self.edge_index_low = to_undirected(rows[keep_low], cols[keep_low])
        self.n_high_items = int(is_high.sum())
        self.frac_low_edges = float(keep_low.mean())

        # ---- branches ----
        spec = {'lf': ['low', 'full'], 'ff': ['full', 'full2'], 'f': ['full'], 'l': ['low']}[self.branches]
        self.branch_names = spec
        self.item_emb = nn.ModuleDict()
        self.gcn = nn.ModuleDict()
        for b in spec:
            emb = nn.Embedding(self.num_item, self.dim)
            self.item_emb[b] = emb
            self.gcn[b] = GCN(self.num_user, self.dim)

        # ---- gate on the full branch (SCOPE, without semantic conflict c_u) ----
        self.use_gate = (self.branches == 'lf' and self.gate_mode == 'id')
        if self.use_gate:
            self.gate = nn.Sequential(nn.Linear(self.dim, 32), nn.ReLU(), nn.Linear(32, 1))
            self.out_proj = nn.Linear(2, 1)   # SCOPE: Linear(3, 1) over [userW, c_u, gate]
            self.userW = nn.Parameter(torch.empty(self.num_user, 1))
            nn.init.xavier_normal_(self.userW)
        # learned weight of the LOW branch (baseline): one scalar, or one scalar per user; initialised at 1
        self.low_alpha = None
        if self.branches == 'lf' and self.gate_mode in ('alpha', 'alpha_u', 'alpha_b', 'alpha_bu'):
            n = self.num_user if self.gate_mode in ('alpha_u', 'alpha_bu') else 1
            # unbounded: alpha itself (init 1); bounded: theta with alpha = 2 sigmoid(theta) (init theta 0 -> alpha 1)
            init = torch.zeros(n, 1) if self.gate_mode in ('alpha_b', 'alpha_bu') else torch.ones(n, 1)
            self.low_alpha = nn.Parameter(init)
        assert self.combine == 'cat' or self.branches in ('lf', 'ff'), 'scope_combine=sum needs two branches'

        # ---- optional text kNN item-item graph (as in SCOPE) ----
        self.mm_adj = None
        if self.item_graph == 'text':
            t_feat = self.t_feat
            if t_feat is None:   # is_multimodal_model is False for SCOPEID: read the text file only for the kNN graph
                path = os.path.join(os.path.abspath(config['data_path'] + config['dataset']),
                                    config['text_feature_file'])
                t_feat = torch.from_numpy(np.load(path, allow_pickle=True)).float().to(self.device)
            self.mm_adj = self._knn_adj(t_feat)

        self.result_embed = None

    # ---------- helpers copied from SCOPE ----------
    def _knn_adj(self, emb):
        x = emb.div(torch.norm(emb, p=2, dim=-1, keepdim=True))
        sim = torch.mm(x, x.t())
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        size = sim.size()
        del sim
        i0 = torch.arange(knn_ind.shape[0], device=self.device).unsqueeze(1).expand(-1, self.knn_k)
        idx = torch.stack((i0.flatten(), knn_ind.flatten()), 0)
        adj = torch.sparse_coo_tensor(idx, torch.ones_like(idx[0], dtype=torch.float), size)
        r = torch.pow(1e-7 + torch.sparse.sum(adj, -1).to_dense(), -0.5)
        return torch.sparse_coo_tensor(idx, r[idx[0]] * r[idx[1]], size).coalesce()

    def _item_item(self, rep):
        h = rep
        for _ in range(self.n_mm_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return rep + h

    def alpha_value(self):
        """Weight of the low branch (None when the model has none)."""
        if self.branches == 'lf' and self.gate_mode == 'fixed':
            return self.alpha_fixed
        if self.low_alpha is None:
            return None
        if self.gate_mode in ('alpha_b', 'alpha_bu'):
            return 2.0 * torch.sigmoid(self.low_alpha)
        return self.low_alpha

    def post_epoch_processing(self):
        a = self.alpha_value()
        if not torch.is_tensor(a):
            return None
        a = a.detach().flatten()
        return '[SCOPEID] low-branch alpha: mean {:.3f} | min {:.3f} | max {:.3f}'.format(
            float(a.mean()), float(a.min()), float(a.max()))

    # ---------- forward ----------
    def _compute(self):
        users, items, prefs = {}, {}, {}
        for b in self.branch_names:
            edges = self.edge_index_low if b == 'low' else self.edge_index_full
            rep, pref = self.gcn[b](edges, self.item_emb[b].weight)
            users[b], items[b], prefs[b] = rep[:self.num_user], rep[self.num_user:], pref

        if self.use_gate:
            g_in = F.layer_norm(users['full'] - users['low'], [self.dim])
            a = torch.sigmoid(self.out_proj(torch.cat([self.userW, self.gate(g_in)], dim=1)))
            users['full'] = (0.5 + 0.5 * a) * users['full']

        a = self.alpha_value()
        if a is not None:
            users['low'] = a * users['low']
            if self.combine == 'sum':
                items['low'] = (a.mean() if torch.is_tensor(a) and a.numel() > 1 else a) * items['low']
        if self.combine == 'sum':
            user_rep = sum(users[b] for b in self.branch_names)
            item_rep = sum(items[b] for b in self.branch_names)
        else:
            user_rep = torch.cat([users[b] for b in self.branch_names], dim=1)
            item_rep = torch.cat([items[b] for b in self.branch_names], dim=1)
        if self.mm_adj is not None:
            item_rep = self._item_item(item_rep)
        return user_rep, item_rep, prefs

    @torch.no_grad()
    def branch_reps(self):
        """Per-branch user / item representations after propagation (for taps)."""
        out = {}
        for b in self.branch_names:
            edges = self.edge_index_low if b == 'low' else self.edge_index_full
            rep, _ = self.gcn[b](edges, self.item_emb[b].weight)
            out[b] = (rep[:self.num_user], rep[self.num_user:])
        return out

    def forward(self, interaction):
        user_nodes, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        user_rep, item_rep, prefs = self._compute()
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        u = user_rep[user_nodes]
        pos = (u * item_rep[pos_items]).sum(1)
        neg = (u * item_rep[neg_items]).sum(1)
        return pos, neg, prefs

    def calculate_loss(self, interaction):
        pos, neg, prefs = self.forward(interaction)
        users = interaction[0]
        reg = self.pref_reg * sum((prefs[b][users] ** 2).mean() for b in self.branch_names)
        return -torch.mean(torch.log2(torch.sigmoid(pos - neg))) + reg

    def full_sort_predict(self, interaction):
        if self.result_embed is None or not self.training:
            with torch.no_grad():
                u, i, _ = self._compute()
                self.result_embed = torch.cat((u, i), dim=0)
        user_tensor = self.result_embed[:self.n_users][interaction[0], :]
        item_tensor = self.result_embed[self.n_users:]
        return torch.matmul(user_tensor, item_tensor.t())


class GCN(nn.Module):
    """SCOPE GCN with has_feature=False: free user preference table + 2 propagation steps, out = x + h1 + h2."""
    def __init__(self, num_user, dim):
        super(GCN, self).__init__()
        self.preference = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(num_user, dim), dtype=torch.float32), gain=1))
        self.conv = Base_gcn(aggr='add')

    def forward(self, edge_index, features):
        x = F.normalize(torch.cat((self.preference, features), dim=0))
        h = self.conv(x, edge_index)
        h_1 = self.conv(h, edge_index)
        return h + x + h_1, self.preference


class Base_gcn(MessagePassing):
    def __init__(self, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        dis = deg.pow(-0.5)
        return (dis[row] * dis[col]).view(-1, 1) * x_j
