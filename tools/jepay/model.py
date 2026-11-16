# coding: utf-8
"""
JEPA-Y (stage A of the diagnosis, diagnosis.md section 2).

    h_t = E_t(T),  h_v = LN(E_v(V))                     LN without affine: E_v cannot shrink h_v
    (S_v, Z_v) = F_phi(h_v)                             two linear branches, each followed by LN
    L_JEPA = || P_theta(h_t) - sg(S_v) ||^2             S_v: the visual part text can predict
    L_rec  = || D([S_v; Z_v]) - PCA_k(V) ||^2           the two branches together keep the visual content
    L_Y    = lam_T || Y - H(h_t, m) ||^2 + lam_TV || Y - H(h_t, Z_v) ||^2      m: learned "no visual" token
    L_anti = VICReg(h_t) + VICReg(S_v) + VICReg(Z_v) + CrossCov(S_v, Z_v)  [+ lam_zt CrossCov(Z_v, sg(h_t))]
    L      = lam_J L_JEPA + lam_R L_rec + L_Y + lam_C L_anti

Conditions:
    A0     text only (own model): H(h_t, m), L_Y on the T branch, VICReg(h_t)
    A1     T + h_v, same supervision, no decomposition: z = h_v           (A1-sup)
    A2     JEPA-Y with F_phi (above)
    A2sub  earlier design: Z_v = h_v - sg(P_v(h_t)),  L_JEPA = || P_v(h_t) - sg(h_v) ||^2, no F_phi / D
    A2noJ  A2 without L_JEPA (no predictor): the split comes from L_rec, L_Y and the cross-covariance only
    A2noY  A2 whose representation never sees Y: the head is trained on detached inputs (for E-B-lite)
The head H of stage A is fixed (mlp1); capacity is varied in stage B (tools/jepay/probe.py) on the frozen
representations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CONDS = ('A0', 'A1', 'A2', 'A2sub', 'A2noJ', 'A2noY')
FPHI_CONDS = ('A2', 'A2noJ', 'A2noY')
HEADS = {'lin': 0, 'mlp1': 1, 'mlp2': 2}          # number of hidden layers


def mlp(d_in, d_hidden, d_out, n_hidden):
    layers, d = [], d_in
    for _ in range(n_hidden):
        layers += [nn.Linear(d, d_hidden), nn.ReLU()]
        d = d_hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class JEPAY(nn.Module):
    def __init__(self, cond, dt, dv, dy, k_rec=128, d=64, hidden=128, dec_hidden=256, head='mlp1', ht_dropout=0.0):
        super(JEPAY, self).__init__()
        assert cond in CONDS, cond
        self.cond = cond
        self.has_v = cond != 'A0'
        self.fphi = cond in FPHI_CONDS
        self.E_t = nn.Linear(dt, d)
        if self.has_v:
            self.E_v = nn.Linear(dv, d)
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        if self.fphi:
            self.F_s = nn.Linear(d, d)
            self.F_z = nn.Linear(d, d)
            self.D = mlp(2 * d, dec_hidden, k_rec, 1)
        self.P = mlp(d, d, d, 1) if cond in ('A2', 'A2noY', 'A2sub') else None     # 2-layer MLP d -> d
        self.m = nn.Parameter(torch.randn(d))
        self.H = mlp(2 * d, hidden, dy, HEADS[head])
        self.drop = nn.Dropout(ht_dropout)
        self.detach_head = cond == 'A2noY'

    def encode(self, t, v=None):
        out = {'h_t': self.E_t(t)}
        if not self.has_v:
            return out
        h_v = self.ln(self.E_v(v))
        out['h_v'] = h_v
        if self.fphi:
            s, z = self.ln(self.F_s(h_v)), self.ln(self.F_z(h_v))
            out['s'], out['z'] = s, z
            out['recon'] = self.D(torch.cat([s, z], 1))
            if self.P is not None:
                out['s_hat'] = self.P(out['h_t'])
                out['recon_shat'] = self.D(torch.cat([out['s_hat'], z], 1))   # D(P(h_t), Z_v), for M5 only
        elif self.cond == 'A2sub':
            out['hv_hat'] = self.P(out['h_t'])
            out['z'] = h_v - out['hv_hat'].detach()
        else:                                                                    # A1
            out['z'] = h_v
        return out

    def forward(self, t, v=None):
        out = self.encode(t, v)
        h_t = out['h_t'].detach() if self.detach_head else out['h_t']
        h_t = self.drop(h_t)
        m = self.m.unsqueeze(0).expand(t.shape[0], -1)
        out['y_t'] = self.H(torch.cat([h_t, m], 1))
        if self.has_v:
            z = out['z'].detach() if self.detach_head else out['z']
            out['y_tv'] = self.H(torch.cat([h_t, z], 1))
        return out


def vicreg(x, eps=1e-4):
    """Variance + covariance terms of VICReg (no invariance term)."""
    n, d = x.shape
    x = x - x.mean(0)
    std = torch.sqrt(x.var(0) + eps)
    var_term = F.relu(1.0 - std).mean()
    cov = (x.T @ x) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_term + off.pow(2).sum() / d


def cross_cov(a, b):
    """|| Cov(a, b) ||_F^2 / d over the batch (0 when a and b are linearly decorrelated)."""
    n, d = a.shape
    a = a - a.mean(0)
    b = b - b.mean(0)
    c = (a.T @ b) / max(n - 1, 1)
    return c.pow(2).sum() / d


def loss_terms(model, out, y, cfg, v_target=None):
    """Returns (total, dict of scalar terms). cfg: lam_T, lam_TV, lam_J, lam_R, lam_C, lam_zt."""
    terms = {}
    terms['L_T'] = F.mse_loss(out['y_t'], y)
    total = cfg['lam_T'] * terms['L_T']
    if model.has_v:
        terms['L_TV'] = F.mse_loss(out['y_tv'], y)
        total = total + cfg['lam_TV'] * terms['L_TV']
    if 's_hat' in out:
        terms['L_J'] = F.mse_loss(out['s_hat'], out['s'].detach())
        total = total + cfg['lam_J'] * terms['L_J']
    elif 'hv_hat' in out:
        terms['L_J'] = F.mse_loss(out['hv_hat'], out['h_v'].detach())
        total = total + cfg['lam_J'] * terms['L_J']
    if model.fphi:
        terms['L_rec'] = F.mse_loss(out['recon'], v_target)
        total = total + cfg['lam_R'] * terms['L_rec']
    c = vicreg(out['h_t'])
    if model.fphi:
        c = c + vicreg(out['s']) + vicreg(out['z'])
        terms['X_sz'] = cross_cov(out['s'], out['z'])
        c = c + terms['X_sz']
        if cfg.get('lam_zt', 0.0) > 0:
            terms['X_zt'] = cross_cov(out['z'], out['h_t'].detach())
            c = c + cfg['lam_zt'] * terms['X_zt']
    elif model.has_v:
        c = c + vicreg(out['h_v'])
    terms['L_C'] = c
    total = total + cfg['lam_C'] * c
    return total, terms
