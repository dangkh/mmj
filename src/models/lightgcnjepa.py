# coding: utf-8
"""
LightGCNJEPA: LightGCN (BPR on the full training graph G) + a JEPA term between two views of the SAME graph.
ID only: no text / visual feature.

    context  U', Y' = LightGCN(G', E_0)            G' = G with a fresh mask every epoch (mask_mode, mask_ratio)
    target   computed with the EMA copy E_0^xi of the online layer-0 embeddings (no gradient):
               target_type = agg   U, Y = LightGCN(G, E_0^xi)                 predict the full-graph node
               target_type = edge  U, Y = LightGCN(G', E_0^xi); for a MASKED edge (u, i) predict the other
                                   endpoint: P(Y'_i) -> U_u, P_u(U'_u) -> Y_i  (the edge appears in no view)
    loss     L = BPR(G) + reg * ||E_0||^2 + (jepa_weight / mask_ratio) * (L_item + L_user) [+ vic_weight * VIC]
             agg:  L_item = mean_{i in batch} ||P(Y'_i) - sg(Y_i)||^2 / Var(Y), L_user the same for users
             edge: over a sample of masked edges (as many as the batch), same normalisation
             Var(.) = mean squared distance of the target to its batch mean (detached), so jepa_weight is in units of
             relative error (jepa_scale = var; jepa_scale = none: raw squared error).
             VIC (optional): per-dimension std of the context batch kept >= the target's (hinge) + off-diagonal
             covariance of the context batch (scale-free).
    P, P_u   residual MLPs x + MLP(x), last layer zero-initialised (identity at the start).
    EMA      E_0^xi <- tau E_0^xi + (1 - tau) E_0 before every step; tau goes from ema_tau_start to ema_tau_end
             (cosine) over ema_anneal epochs; E_0^xi is set to E_0 when the JEPA term starts (after warmup epochs of
             BPR only).
    Inference: LightGCN(G, E_0)  (predictors unused).
After every epoch (log): actual mask fraction, tau, mean L_item / L_user, effective rank / mean per-dimension std /
uniformity of Y (full graph) and Y' (context), so that collapse or over-smoothing is visible during training.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lightgcn import LightGCN
from models.graph_aug import EdgeSet


class Residual(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.LeakyReLU(), nn.Linear(hidden, d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


def _erank(z):
    z = z - z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(z.double())
    p = s / s.sum().clamp_min(1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def _uniformity(z, n=2048, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    idx = torch.randperm(z.shape[0], generator=g)[:n].to(z.device)
    x = F.normalize(z[idx], dim=1)
    return float(torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log())


class LightGCNJEPA(LightGCN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.edges = EdgeSet(self.interaction_matrix, self.n_users, self.n_items)
        g = lambda k, d: d if config[k] is None else config[k]  # noqa: E731
        self.mask_mode = g('mask_mode', 'uniform')
        self.mask_ratio = float(g('mask_ratio', 0.1))
        self.mask_beta = float(g('mask_beta', 1.0))
        self.target_type = g('target_type', 'agg')
        assert self.target_type in ('agg', 'edge')
        self.jepa_weight = float(g('jepa_weight', 0.1))
        self.jepa_scale = g('jepa_scale', 'var')
        self.vic_weight = float(g('vic_weight', 0.0))
        self.warmup = int(g('warmup', 5))
        self.tau0, self.tau1 = float(g('ema_tau_start', 0.99)), float(g('ema_tau_end', 0.999))
        self.ema_anneal = int(g('ema_anneal', 100))
        d, ph = self.latent_dim, int(g('pred_hidden', 128))
        self.pred_item = Residual(d, ph)
        self.pred_user = Residual(d, ph)
        self.register_buffer('tgt_user', self.embedding_dict['user_emb'].detach().clone())
        self.register_buffer('tgt_item', self.embedding_dict['item_emb'].detach().clone())
        self.rng = np.random.default_rng(int(config['seed'] or 0) + 17)
        self.epoch = -1
        self.tau = self.tau0
        self.adj_ctx = None
        self.m_u = self.m_i = None
        self.mask_frac = 0.0
        self._stats = {'L_item': [], 'L_user': [], 'vic': []}

    # ------------------------------------------------------------------ helpers
    def active(self):
        return self.jepa_weight > 0 and self.epoch >= self.warmup

    def _propagate(self, adj, u0, i0):
        x = torch.cat([u0, i0], 0)
        layers = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            layers.append(x)
        out = torch.stack(layers, 1).mean(1)
        return out[:self.n_users], out[self.n_users:]

    @torch.no_grad()
    def _ema(self):
        e = self.embedding_dict
        self.tgt_user.mul_(self.tau).add_(e['user_emb'].detach(), alpha=1 - self.tau)
        self.tgt_item.mul_(self.tau).add_(e['item_emb'].detach(), alpha=1 - self.tau)

    def _rel_mse(self, pred, tgt):
        tgt = tgt.detach()
        err = ((pred - tgt) ** 2).sum(1).mean()
        if self.jepa_scale == 'var':
            err = err / ((tgt - tgt.mean(0, keepdim=True)) ** 2).sum(1).mean().clamp_min(1e-12)
        return err

    @staticmethod
    def _vic(ctx, tgt):
        std_c = ctx.std(0)
        std_t = tgt.detach().std(0)
        var = F.relu(std_t - std_c).mean() / std_t.mean().clamp_min(1e-12)
        z = ctx - ctx.mean(0, keepdim=True)
        cov = z.T @ z / max(z.shape[0] - 1, 1)
        diag = torch.diagonal(cov)
        off = cov - torch.diag(diag)
        cov_term = (off ** 2).sum() / z.shape[1] / diag.mean().clamp_min(1e-12) ** 2
        return var + cov_term

    # ------------------------------------------------------------------ epoch hooks
    def pre_epoch_processing(self):
        self.epoch += 1
        if not self.active():
            return
        if self.epoch == self.warmup:                   # target starts as a copy of the online embeddings
            with torch.no_grad():
                self.tgt_user.copy_(self.embedding_dict['user_emb'])
                self.tgt_item.copy_(self.embedding_dict['item_emb'])
        keep = self.edges.mask(self.mask_mode, self.mask_ratio, self.rng, beta=self.mask_beta)
        self.mask_frac = 1.0 - keep.mean()
        self.adj_ctx = self.edges.norm_adj(keep, self.device)
        mu, mi = self.edges.masked_edges(keep)
        self.m_u = torch.from_numpy(mu).long().to(self.device)
        self.m_i = torch.from_numpy(mi).long().to(self.device)
        prog = min(max(self.epoch - self.warmup, 0) / max(self.ema_anneal, 1), 1.0)
        self.tau = self.tau1 - (self.tau1 - self.tau0) * (math.cos(math.pi * prog) + 1) / 2

    def calculate_loss(self, interaction):
        loss = super().calculate_loss(interaction)          # BPR + reg on the full graph
        if not self.active():
            return loss
        self._ema()
        user, pos, neg = interaction[0], interaction[1], interaction[2]
        e = self.embedding_dict
        Uc, Yc = self._propagate(self.adj_ctx, e['user_emb'], e['item_emb'])
        with torch.no_grad():
            adj_t = self.norm_adj_matrix if self.target_type == 'agg' else self.adj_ctx
            Ut, Yt = self._propagate(adj_t, self.tgt_user, self.tgt_item)
        if self.target_type == 'agg':
            users, items = torch.unique(user), torch.unique(torch.cat([pos, neg]))
            ci, ti, cu, tu = Yc[items], Yt[items], Uc[users], Ut[users]
        else:
            if self.m_u.numel() == 0:
                return loss
            k = torch.randint(0, self.m_u.numel(), (user.numel(),), device=self.device)
            mu, mi = self.m_u[k], self.m_i[k]
            ci, ti, cu, tu = Yc[mi], Ut[mu], Uc[mu], Yt[mi]
        l_i = self._rel_mse(self.pred_item(ci), ti)
        l_u = self._rel_mse(self.pred_user(cu), tu)
        self._stats['L_item'].append(float(l_i))
        self._stats['L_user'].append(float(l_u))
        loss = loss + self.jepa_weight / max(self.mask_ratio, 1e-6) * (l_i + l_u)
        if self.vic_weight > 0:
            v = self._vic(ci, ti) + self._vic(cu, tu)
            self._stats['vic'].append(float(v))
            loss = loss + self.vic_weight * v
        return loss

    @torch.no_grad()
    def post_epoch_processing(self):
        if not self.active():
            return None
        e = self.embedding_dict
        U, Y = self.forward()
        Uc, Yc = self._propagate(self.adj_ctx, e['user_emb'], e['item_emb'])
        m = lambda k: np.mean(self._stats[k]) if self._stats[k] else float('nan')  # noqa: E731
        msg = ('[jepa] mask {:.3f} tau {:.4f} | L_item {:.4f} L_user {:.4f}{} | item erank {:.1f} / ctx {:.1f}, '
               'std {:.4f} / ctx {:.4f}, unif {:.3f} / ctx {:.3f}').format(
            self.mask_frac, self.tau, m('L_item'), m('L_user'),
            ' vic {:.4f}'.format(m('vic')) if self._stats['vic'] else '',
            _erank(Y), _erank(Yc), float(Y.std(0).mean()), float(Yc.std(0).mean()), _uniformity(Y), _uniformity(Yc))
        self._stats = {'L_item': [], 'L_user': [], 'vic': []}
        return msg
