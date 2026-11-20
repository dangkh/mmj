# coding: utf-8
"""
SCOPEKD: selective preference correction of the LOW view, guided by a frozen FULL-view teacher.

    Low encoder   z^L = SCOPE GCN on the low-exposure graph (edges of the top low_high_ratio items dropped),
                  same GCN as SCOPEID (src/models/scopeid.py); trained from scratch.
    Correction    h_u = z_u^L + a * P_u(z_u^L),  h_i = z_i^L + a * P_i(z_i^L)
                  P_u, P_i: MLP d -> corr_hidden -> d (last layer zero-initialised); a: learned scalar,
                  initialised at corr_alpha_init.
    Teacher       frozen FULL view: U^F, Y^F read from taps (teacher_taps, default ../taps/SCOPEID-<ds>-es_f),
                  Delta^F_t = <U^F_u, Y^F_i+> - <U^F_u, Y^F_i->  for a training triplet t = (u, i+, i-).
    Loss          L = BPR(h) + 0.003 * mean ||pref||^2 + kd_weight * L_corr
                  L_corr = sum_t w_t * max(0, kd_delta - Delta^{h,sg}_t)^2 / (sum_t w_t + eps)
                  Delta^{h,sg}: score margin of h computed from sg(z^L), so L_corr updates only P_u, P_i and a
                  (the Low encoder is trained by BPR only).
    Weights w_t   kd_wmode = hard : 1[Delta^F > kd_tmargin]                              (teacher correct)
                             soft : 1[Delta^F > 0] * sigmoid(Delta^F / std_batch(Delta^F))   (teacher confidence)
                             star : 1[Delta^F > kd_tmargin] * 1[Delta^L <= 0]            (teacher right, Low wrong)
    Scoring       <h_u, h_i>  (the student alone). The teacher is saved in the taps (T3_full / T3_user_full) so that
                  tools/scope_fusion.py can also report the score fusion student + teacher.
Logged every epoch: share of triplets with w > 0, mean Delta^F, mean Delta^h, a.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from models.scopeid import GCN


class SCOPEKD(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SCOPEKD, self).__init__(config, dataset)
        self.num_user, self.num_item = self.n_users, self.n_items
        self.dim = config['feat_embed_dim'] if config['feat_embed_dim'] else 64
        self.high_ratio = config['low_high_ratio'] if config['low_high_ratio'] is not None else 0.05
        self.kd_weight = float(config['kd_weight'] if config['kd_weight'] is not None else 1.0)
        self.kd_delta = float(config['kd_delta'] if config['kd_delta'] is not None else 0.5)
        self.kd_tmargin = float(config['kd_tmargin'] or 0.0)
        self.kd_wmode = config['kd_wmode'] or 'hard'
        assert self.kd_wmode in ('hard', 'soft', 'star')
        self.pref_reg = 0.003

        # ---- low-exposure graph (as SCOPE) ----
        inter = dataset.inter_matrix(form='coo').astype(np.float32)
        rows, cols = inter.row.astype(np.int64), inter.col.astype(np.int64)
        deg = np.bincount(cols, minlength=self.num_item)
        n_high = int(self.num_item * self.high_ratio)
        is_high = np.zeros(self.num_item, dtype=bool)
        if n_high > 0:
            is_high[np.argsort(deg)[-n_high:]] = True
        keep = ~is_high[cols]
        e = torch.tensor(np.stack([rows[keep], cols[keep] + self.num_user]), dtype=torch.long)
        self.edge_index_low = torch.cat((e, e[[1, 0]]), dim=1).contiguous().to(self.device)

        # ---- low encoder + correction heads ----
        self.item_emb = nn.Embedding(self.num_item, self.dim)
        self.gcn = GCN(self.num_user, self.dim)
        hid = config['corr_hidden'] or 64

        def head():
            m = nn.Sequential(nn.Linear(self.dim, hid), nn.LeakyReLU(), nn.Linear(hid, self.dim))
            nn.init.zeros_(m[2].weight)
            nn.init.zeros_(m[2].bias)
            return m
        self.P_u, self.P_i = head(), head()
        self.alpha = nn.Parameter(torch.tensor(float(config['corr_alpha_init'] if config['corr_alpha_init']
                                                     is not None else 0.1)))

        # ---- frozen FULL teacher ----
        path = config['teacher_taps'] or os.path.join('../taps', 'SCOPEID-{}-es_f'.format(config['dataset']))
        U = np.load(os.path.join(path, 'T3_user.npy')).astype(np.float32)
        Y = np.load(os.path.join(path, 'T3.npy')).astype(np.float32)
        assert U.shape[0] == self.num_user and Y.shape[0] == self.num_item, 'teacher shapes {} {}'.format(U.shape, Y.shape)
        self.register_buffer('U_full', torch.from_numpy(U))
        self.register_buffer('Y_full', torch.from_numpy(Y))
        print('[SCOPEKD] teacher {} | low view keeps {:.1%} of the edges | wmode {}'.format(
            os.path.abspath(path), keep.mean(), self.kd_wmode))
        self.stats, self.result_embed = [], None

    # ---------- representations ----------
    def low(self):
        rep, pref = self.gcn(self.edge_index_low, self.item_emb.weight)
        return rep[:self.num_user], rep[self.num_user:], pref

    def correct(self, zu, zi):
        return zu + self.alpha * self.P_u(zu), zi + self.alpha * self.P_i(zi)

    def calculate_loss(self, interaction):
        u, ip, ineg = interaction[0], interaction[1], interaction[2]
        zu, zi, pref = self.low()
        hu, hi = self.correct(zu, zi)
        self.result_embed = torch.cat((hu, hi), 0).detach()
        pos = (hu[u] * hi[ip]).sum(1)
        neg = (hu[u] * hi[ineg]).sum(1)
        loss = -torch.mean(torch.log2(torch.sigmoid(pos - neg))) + self.pref_reg * (pref[u] ** 2).mean()

        with torch.no_grad():
            dF = (self.U_full[u] * (self.Y_full[ip] - self.Y_full[ineg])).sum(1)
            if self.kd_wmode == 'soft':
                w = (dF > 0).float() * torch.sigmoid(dF / (dF.std() + 1e-8))
            else:
                w = (dF > self.kd_tmargin).float()
            if self.kd_wmode == 'star':
                dL = (zu[u] * (zi[ip] - zi[ineg])).sum(1)
                w = w * (dL <= 0).float()
        su, si = self.correct(zu.detach(), zi.detach())          # correction branch: Low encoder detached
        dH = (su[u] * (si[ip] - si[ineg])).sum(1)
        l_corr = (w * F.relu(self.kd_delta - dH) ** 2).sum() / (w.sum() + 1e-8)
        self.stats.append((float(w.gt(0).float().mean()), float(dF.mean()), float((pos - neg).mean().detach()),
                           float(self.alpha.detach())))
        return loss + self.kd_weight * l_corr

    def post_epoch_processing(self):
        if not self.stats:
            return None
        s = np.asarray(self.stats).mean(0)
        self.stats = []
        return '[SCOPEKD] share w>0 {:.3f} | mean dF {:.3f} | mean dH {:.3f} | alpha {:.3f}'.format(*s)

    @torch.no_grad()
    def full_sort_predict(self, interaction):
        zu, zi, _ = self.low()
        hu, hi = self.correct(zu, zi)
        self.result_embed = torch.cat((hu, hi), 0)
        return torch.matmul(hu[interaction[0]], hi.t())

    @torch.no_grad()
    def branch_reps(self):
        zu, zi, _ = self.low()
        hu, hi = self.correct(zu, zi)
        return {'low': (hu, hi), 'full': (self.U_full, self.Y_full)}
