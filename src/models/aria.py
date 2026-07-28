"""ARIA model family. Three underlying architectures, used to build the
five models this benchmark actually searches over:

  - ARIALayer / ARIA           -- the original per-node belief gate: a
    single softmax over 4 channels (self / 1-hop / 2-hop / low-rank
    global). Used bare for "ARIA (baseline)", and with the PE +
    homophily-auxiliary-loss training recipe (see aria_augmentations.py)
    for "ARIA-PH".

  - ARIALayerBounded / ARIABounded -- a 3-way softmax over {self, 1-hop,
    2-hop} PLUS a separate scalar correction gate alpha for the global
    channel, structurally bounded below 50% influence via the
    "/(1+alpha)" mixing denominator. Used with the same PE +
    homophily-auxiliary-loss recipe for "ARIA-PHG" (the "G" is for the
    separate Gated/bounded global-channel scalar -- this is the only
    architectural difference from ARIA-PH; both use identical PE +
    homophily-loss training add-ons).

  - ARIALayerMoE / ARIAMoE      -- the original 4-way gate, unchanged, but
    the shared output projection is replaced by K "expert" projections
    mixed by a FIXED (non-trainable) per-node soft cluster membership
    (see aria_augmentations.compute_regime_membership). Used for
    "ARIA-MoE".

"ARIA-Ensemble" (the 5th model) is not a fourth architecture -- it's a
post-hoc average of ARIA-PH + ARIA-PHG + ARIA-MoE's predictions, built in
ensemble.py.

All three model classes accept num_layers (search hyperparameter here,
unlike earlier stages of this project which fixed it at 2).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm_adj(edge_index, num_nodes):
    """Symmetrically normalized adjacency with self-loops, as (edge_index,
    edge_weight) ready for sparse propagation."""
    from torch_geometric.utils import add_self_loops
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index, norm


def spmm_propagate(edge_index, norm, H):
    """out = A_hat @ H, via scatter-add (no torch.sparse.* calls -- keeps
    this portable across CUDA/MPS/CPU without relying on sparse-tensor
    kernels that may not be implemented on every backend)."""
    row, col = edge_index
    out = torch.zeros_like(H)
    msg = H[col] * norm.unsqueeze(-1)
    out.index_add_(0, row, msg)
    return out


# ---------------------------------------------------------------------------
# ARIALayer / ARIA -- the original unbounded 4-way gate
# ---------------------------------------------------------------------------

class ARIALayer(nn.Module):
    def __init__(self, dim, num_nodes, rank=16, struct_dim=2, dropout=0.5):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.U = nn.Parameter(torch.empty(num_nodes, rank))
        self.V = nn.Parameter(torch.empty(num_nodes, rank))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        self.W = nn.Linear(dim, dim)
        self.W_beta = nn.Linear(dim + struct_dim, 4)
        self.ln = nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, H, edge_index, norm, struct_feat):
        Z0 = H
        Z1 = spmm_propagate(edge_index, norm, H)
        Z2 = spmm_propagate(edge_index, norm, Z1)
        Zg = (self.U @ (self.V.t() @ H)) / math.sqrt(self.rank)

        gate_in = torch.cat([H, struct_feat], dim=-1)
        beta = F.softmax(self.W_beta(gate_in), dim=-1)  # [n, 4]

        H_tilde = (beta[:, 0:1] * Z0 + beta[:, 1:2] * Z1 +
                   beta[:, 2:3] * Z2 + beta[:, 3:4] * Zg)
        H_tilde = F.dropout(H_tilde, p=self.dropout, training=self.training)

        out = F.relu(self.W(H_tilde))
        H_next = self.ln(H + out)
        return H_next, beta


class ARIA(nn.Module):
    """L-layer ARIA. Used bare for the "ARIA (baseline)" model, and with
    input PE-augmentation + a homophily-supervised auxiliary loss (both
    applied at training time, see aria_augmentations.py) for "ARIA-PH"."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_nodes, num_layers=2,
                 rank=16, struct_dim=2, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            ARIALayer(hidden_dim, num_nodes, rank=rank, struct_dim=struct_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        self._cached_adj = None

    def forward(self, x, edge_index, struct_feat, return_beta=False, **kwargs):
        n = x.size(0)
        if self._cached_adj is None or self._cached_adj[0].device != x.device:
            self._cached_adj = build_norm_adj(edge_index, n)
        norm_edge_index, norm_weight = self._cached_adj

        H = F.dropout(x, p=self.dropout, training=self.training)
        H = F.relu(self.input_proj(H))

        betas = []
        for layer in self.layers:
            H, beta = layer(H, norm_edge_index, norm_weight, struct_feat)
            betas.append(beta)

        H = F.dropout(H, p=self.dropout, training=self.training)
        logits = self.out_lin(H)
        if return_beta:
            return logits, betas
        return logits


# ---------------------------------------------------------------------------
# ARIALayerBounded / ARIABounded -- 3-way softmax + separate bounded alpha
# ---------------------------------------------------------------------------

class ARIALayerBounded(nn.Module):
    """gate_logits beta_i = softmax(gamma + W_beta[h_i || R1_i || s_i]) over
    {Z0, Z1, Z2} only; alpha_i = sigmoid(b_alpha + w_alpha^T[h_i || s_i])
    is a separate scalar controlling how much of the low-rank global
    channel blends in, initialized so alpha ~ 0.05 at the start of
    training. Per-channel LayerNorm R_r = LN(Z_r) always applied. Mixing:
    h~_i = (sum_r beta_r R_r + alpha Rg) / (1 + alpha) -- the "/(1+alpha)"
    denominator structurally bounds the global channel's influence below
    0.5, so it can only ever correct the topology mixture, never dominate
    it.
    """

    def __init__(self, dim, num_nodes, rank=16, struct_dim=3, dropout=0.5):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.U = nn.Parameter(torch.empty(num_nodes, rank))
        self.V = nn.Parameter(torch.empty(num_nodes, rank))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

        self.ln0 = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.lng = nn.LayerNorm(dim)

        self.gamma = nn.Parameter(torch.zeros(3))
        self.W_beta = nn.Linear(2 * dim + struct_dim, 3, bias=False)

        self.b_alpha = nn.Parameter(torch.tensor(-3.0))
        self.w_alpha = nn.Linear(dim + struct_dim, 1, bias=False)

        nn.init.xavier_uniform_(self.W_beta.weight, gain=0.01)
        nn.init.xavier_uniform_(self.w_alpha.weight, gain=0.01)

        self.W = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, H, edge_index, norm, struct_feat):
        Z0 = H
        Z1 = spmm_propagate(edge_index, norm, H)
        Z2 = spmm_propagate(edge_index, norm, Z1)
        Zg = (self.U @ (self.V.t() @ H)) / math.sqrt(self.rank)

        R0 = self.ln0(Z0)
        R1 = self.ln1(Z1)
        R2 = self.ln2(Z2)
        Rg = self.lng(Zg)

        beta_in = torch.cat([H, R1, struct_feat], dim=-1)
        beta = F.softmax(self.gamma + self.W_beta(beta_in), dim=-1)  # [n, 3]

        alpha_in = torch.cat([H, struct_feat], dim=-1)
        alpha = torch.sigmoid(self.b_alpha + self.w_alpha(alpha_in))  # [n, 1]

        H_topo = (beta[:, 0:1] * R0 + beta[:, 1:2] * R1 + beta[:, 2:3] * R2)
        H_tilde = (H_topo + alpha * Rg) / (1.0 + alpha)
        H_tilde = F.dropout(H_tilde, p=self.dropout, training=self.training)

        out = F.relu(self.W(H_tilde))
        H_next = self.ln(H + out)
        return H_next, beta, alpha


class ARIABounded(nn.Module):
    """L-layer bounded-gate ARIA. Always used with the PE + homophily-aux
    training recipe in this benchmark ("ARIA-PHG"); struct_dim defaults to
    3 to match datasets.compute_structural_features_v3."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_nodes, num_layers=2,
                 rank=16, struct_dim=3, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            ARIALayerBounded(hidden_dim, num_nodes, rank=rank, struct_dim=struct_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        self._cached_adj = None

    def forward(self, x, edge_index, struct_feat, return_gates=False, **kwargs):
        n = x.size(0)
        if self._cached_adj is None or self._cached_adj[0].device != x.device:
            self._cached_adj = build_norm_adj(edge_index, n)
        norm_edge_index, norm_weight = self._cached_adj

        H = F.dropout(x, p=self.dropout, training=self.training)
        H = F.relu(self.input_proj(H))

        betas, alphas = [], []
        for layer in self.layers:
            H, beta, alpha = layer(H, norm_edge_index, norm_weight, struct_feat)
            betas.append(beta)
            alphas.append(alpha)

        H = F.dropout(H, p=self.dropout, training=self.training)
        logits = self.out_lin(H)
        if return_gates:
            return logits, betas, alphas
        return logits


# ---------------------------------------------------------------------------
# ARIALayerMoE / ARIAMoE -- original gate, expert-mixed output projection
# ---------------------------------------------------------------------------

class ARIALayerMoE(nn.Module):
    """Same 4-way softmax gate as ARIALayer. The post-mix representation is
    projected by n_experts output matrices instead of one shared W, mixed
    by a fixed per-node soft regime membership passed in at forward time
    (see aria_augmentations.compute_regime_membership)."""

    def __init__(self, dim, num_nodes, rank=16, struct_dim=2, dropout=0.5, n_experts=4):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.n_experts = n_experts
        self.U = nn.Parameter(torch.empty(num_nodes, rank))
        self.V = nn.Parameter(torch.empty(num_nodes, rank))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        self.W_beta = nn.Linear(dim + struct_dim, 4)
        self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_experts)])
        self.ln = nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, H, edge_index, norm, struct_feat, regime_membership):
        Z0 = H
        Z1 = spmm_propagate(edge_index, norm, H)
        Z2 = spmm_propagate(edge_index, norm, Z1)
        Zg = (self.U @ (self.V.t() @ H)) / math.sqrt(self.rank)

        gate_in = torch.cat([H, struct_feat], dim=-1)
        beta = F.softmax(self.W_beta(gate_in), dim=-1)

        H_tilde = (beta[:, 0:1] * Z0 + beta[:, 1:2] * Z1 +
                   beta[:, 2:3] * Z2 + beta[:, 3:4] * Zg)
        H_tilde = F.dropout(H_tilde, p=self.dropout, training=self.training)

        expert_outs = torch.stack([F.relu(e(H_tilde)) for e in self.experts], dim=-1)  # [n, dim, K]
        out = (expert_outs * regime_membership.unsqueeze(1)).sum(-1)  # [n, dim]

        H_next = self.ln(H + out)
        return H_next, beta


class ARIAMoE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_nodes, num_layers=2,
                 rank=16, struct_dim=2, dropout=0.5, n_experts=4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            ARIALayerMoE(hidden_dim, num_nodes, rank=rank, struct_dim=struct_dim,
                         dropout=dropout, n_experts=n_experts)
            for _ in range(num_layers)
        ])
        self.out_lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        self._cached_adj = None

    def forward(self, x, edge_index, struct_feat, regime_membership, return_beta=False, **kwargs):
        n = x.size(0)
        if self._cached_adj is None or self._cached_adj[0].device != x.device:
            self._cached_adj = build_norm_adj(edge_index, n)
        norm_edge_index, norm_weight = self._cached_adj

        H = F.dropout(x, p=self.dropout, training=self.training)
        H = F.relu(self.input_proj(H))

        betas = []
        for layer in self.layers:
            H, beta = layer(H, norm_edge_index, norm_weight, struct_feat, regime_membership)
            betas.append(beta)

        H = F.dropout(H, p=self.dropout, training=self.training)
        logits = self.out_lin(H)
        if return_beta:
            return logits, betas
        return logits


ARIA_MODEL_NAMES = ["aria", "aria_ph", "aria_phg", "aria_moe"]  # + "aria_ensemble" (see ensemble.py)

# Human-readable names for tables/figures/papers.
DISPLAY_NAMES = {
    "mlp": "MLP",
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "gat": "GAT",
    "gps": "GPS",
    "aria": "ARIA (baseline)",
    "aria_ph": "ARIA-PH",       # Positional-encoding + Homophily-aux-loss
    "aria_phg": "ARIA-PHG",     # same + separate Gated/bounded global channel
    "aria_moe": "ARIA-MoE",
    "aria_ensemble": "ARIA-Ensemble",
}
