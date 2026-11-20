# coding: utf-8
"""
Edge / node masking of the user-item training graph and the LightGCN normalised adjacency of a sub-graph.
Shared by LightGCNJEPA, SGL (and the unit tests in tools/test_masks.py). Not a model (get_model never loads it).

    E = EdgeSet(interaction_matrix_coo, n_users, n_items)
    keep = E.mask('uniform', 0.1, rng)          # boolean over the training edges, True = kept
    A = E.norm_adj(keep, device)                 # D^-1/2 A D^-1/2 of the kept sub-graph (torch sparse, n+m square)

Mask modes (ratio rho = expected fraction of masked EDGES, except 'node'):
    uniform  every edge masked with probability rho
    hub      p_e proportional to deg(item)^beta (beta > 0), rescaled so that the mean is rho (clipped at 0.95)
    tail     same with beta < 0; every item with >= 1 training edge keeps at least one edge
    node     rho of the items (with >= 1 edge) lose all their edges; item_ratio / user_ratio for SGL-ND
The adjacency of a sub-graph is renormalised with the degrees of the sub-graph (+1e-7, as LightGCN).
"""
import numpy as np


class EdgeSet:
    def __init__(self, inter_coo, n_users, n_items):
        self.u = np.asarray(inter_coo.row, dtype=np.int64)
        self.i = np.asarray(inter_coo.col, dtype=np.int64)
        self.n_users, self.n_items = int(n_users), int(n_items)
        self.n_edges = len(self.u)
        self.deg_u = np.bincount(self.u, minlength=self.n_users)
        self.deg_i = np.bincount(self.i, minlength=self.n_items)

    # ------------------------------------------------------------------ masks
    def mask(self, mode, ratio, rng, beta=1.0):
        """Boolean array over the edges, True = kept."""
        if ratio <= 0:
            return np.ones(self.n_edges, dtype=bool)
        if mode == 'uniform':
            return rng.random(self.n_edges) >= ratio
        if mode in ('hub', 'tail'):
            b = abs(beta) if mode == 'hub' else -abs(beta)
            w = self.deg_i[self.i].astype(np.float64) ** b
            p = np.clip(ratio * w / w.mean(), 0.0, 0.95)
            keep = rng.random(self.n_edges) >= p
            if mode == 'tail':
                keep = self._keep_one_per_item(keep, rng)
            return keep
        if mode == 'node':
            return self.node_mask(item_ratio=ratio, user_ratio=0.0, rng=rng)
        raise ValueError('unknown mask mode {}'.format(mode))

    def node_mask(self, item_ratio, user_ratio, rng):
        keep = np.ones(self.n_edges, dtype=bool)
        if item_ratio > 0:
            items = np.where(self.deg_i > 0)[0]
            drop = rng.choice(items, int(round(item_ratio * len(items))), replace=False)
            gone = np.zeros(self.n_items, dtype=bool)
            gone[drop] = True
            keep &= ~gone[self.i]
        if user_ratio > 0:
            users = np.where(self.deg_u > 0)[0]
            drop = rng.choice(users, int(round(user_ratio * len(users))), replace=False)
            gone = np.zeros(self.n_users, dtype=bool)
            gone[drop] = True
            keep &= ~gone[self.u]
        return keep

    def _keep_one_per_item(self, keep, rng):
        kept = np.bincount(self.i[keep], minlength=self.n_items)
        lost = np.where((kept == 0) & (self.deg_i > 0))[0]
        if len(lost):
            idx = np.where(np.isin(self.i, lost))[0]
            order = idx[rng.permutation(len(idx))]
            _, first = np.unique(self.i[order], return_index=True)     # one random edge per lost item
            keep = keep.copy()
            keep[order[first]] = True
        return keep

    # ------------------------------------------------------------------ adjacency
    def norm_adj(self, keep, device):
        import torch
        u, i = self.u[keep], self.i[keep] + self.n_users
        n = self.n_users + self.n_items
        rows = np.concatenate([u, i])
        cols = np.concatenate([i, u])
        deg = np.bincount(rows, minlength=n).astype(np.float64) + 1e-7
        d = deg ** -0.5
        vals = (d[rows] * d[cols]).astype(np.float32)
        idx = torch.from_numpy(np.stack([rows, cols])).long()
        return torch.sparse_coo_tensor(idx, torch.from_numpy(vals), (n, n)).coalesce().to(device)

    def masked_edges(self, keep):
        """(users, items) of the masked edges."""
        return self.u[~keep], self.i[~keep]
