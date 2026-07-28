"""Training-time add-ons used by ARIA-PH, ARIA-PHG, and ARIA-MoE. These are
data/loss-level, not architectural -- they don't change models/aria.py at
all.

  - compute_rwpe / augment_with_pe: the positional-encoding (PE) input
    augmentation used by ARIA-PH and ARIA-PHG.
  - compute_train_homophily_target / gate_aux_loss: the homophily-
    supervised auxiliary loss used by ARIA-PH and ARIA-PHG.
  - compute_regime_membership: the fixed (non-trainable) soft cluster
    assignment used by ARIA-MoE.
"""
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

from .models.aria import build_norm_adj, spmm_propagate
from .datasets import compute_structural_features


# ---------------------------------------------------------------------------
# PE (positional encoding) -- used by ARIA-PH, ARIA-PHG, and the GPS baseline
# ---------------------------------------------------------------------------

def compute_rwpe(data, k=8, batch_size=1024):
    """RWPE-style structural descriptor: [diag(A_hat), diag(A_hat^2), ...,
    diag(A_hat^k)], using the exact symmetric normalized adjacency A_hat
    (with self-loops) the ARIA gates use internally. Each node's own
    "return probability" profile under repeated propagation -- a fixed,
    cheap, structure-only signal that survives even when raw node features
    carry none (e.g. the Airports datasets' one-hot IDs).

    Computed in column-batches of the identity matrix rather than
    materializing a dense [n, n] matrix up front: peak memory is O(n *
    batch_size) instead of O(n^2), numerically identical either way. The
    dense-n×n version OOM'd on PubMed (n=19,717, ~1.5GB) and Roman-Empire
    (n~22,662, ~2GB) under a 16GB job -- small graphs (Cora/CiteSeer/WebKB/
    Airports) never triggered this since n^2 stayed small there."""
    n = data.num_nodes
    edge_index, norm = build_norm_adj(data.edge_index, n)
    pe = torch.zeros(n, k)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        width = end - start
        idx = torch.arange(start, end)
        H = torch.zeros(n, width)
        H[idx, torch.arange(width)] = 1.0
        for step in range(k):
            H = spmm_propagate(edge_index, norm, H)
            pe[idx, step] = H[idx, torch.arange(width)]
    mean, std = pe.mean(0, keepdim=True), pe.std(0, keepdim=True)
    std = std.clamp_min(1e-8)
    return (pe - mean) / std


def augment_with_pe(data, k=8):
    """Returns x' = [x || RWPE(k)], leaving `data` itself untouched."""
    pe = compute_rwpe(data, k=k)
    return torch.cat([data.x, pe], dim=1)


# ---------------------------------------------------------------------------
# Homophily-supervised auxiliary loss -- used by ARIA-PH, ARIA-PHG
# ---------------------------------------------------------------------------

def compute_train_homophily_target(data, train_mask):
    """Per-node target propagation mass: fraction of a node's neighbors
    (restricted to edges where BOTH endpoints are train nodes, so no val/
    test label leakage) that share its label. Returns (target [n], valid
    [n] bool) -- valid is True only for nodes with >=1 train-train edge."""
    n = data.num_nodes
    edge_index = data.edge_index  # no self-loops: real neighbors only
    row, col = edge_index
    y = data.y
    same = (y[row] == y[col]).float()
    train_edge = train_mask[row] & train_mask[col]

    num = torch.zeros(n)
    den = torch.zeros(n)
    num.scatter_add_(0, row[train_edge], same[train_edge])
    den.scatter_add_(0, row[train_edge], torch.ones_like(same[train_edge]))

    valid = den > 0
    target = torch.zeros(n)
    target[valid] = num[valid] / den[valid]
    valid = valid & train_mask
    return target, valid


def gate_aux_loss(betas, target, valid):
    """betas: list of [n, C] softmax gate outputs (C=4 for ARIA-PH's gate,
    C=3 for ARIA-PHG's topology-only gate -- indices 1, 2 are the 1-hop/
    2-hop channels in both cases, so this works unmodified for either).
    Averages the propagation mass (beta_1 + beta_2) across layers and
    pushes it toward `target` via MSE, over `valid` nodes only."""
    if valid.sum() == 0:
        return torch.tensor(0.0, device=betas[0].device)
    prop_mass = torch.stack([b[:, 1] + b[:, 2] for b in betas], dim=0).mean(0)
    return F.mse_loss(prop_mass[valid], target[valid])


# ---------------------------------------------------------------------------
# Fixed regime clustering -- used by ARIA-MoE
# ---------------------------------------------------------------------------

def compute_regime_membership(data, n_experts=4, seed=0):
    """Fixed (non-trainable) soft cluster membership over each node's
    richer structural descriptor, via a Gaussian mixture fit once before
    training. Returns [n, n_experts] posterior probabilities (rows sum to
    1). Uses diagonal covariance + a reg_covar floor: several datasets have
    many nodes with near-identical/duplicate structural descriptors (e.g.
    leaf nodes all at degree 1), which collapses a full-covariance GMM
    component to a singular matrix otherwise (hit this on CiteSeer during
    development)."""
    struct = compute_structural_features(data, richer=True).double().numpy()
    gmm = GaussianMixture(n_components=n_experts, random_state=seed, n_init=3,
                           covariance_type="diag", reg_covar=1e-3)
    gmm.fit(struct)
    membership = gmm.predict_proba(struct)
    return torch.tensor(membership, dtype=torch.float32)
