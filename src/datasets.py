"""Unified dataset loading for the ARIA benchmark suite (13 datasets):

  Citation networks   : Cora, CiteSeer, PubMed         (geom-gcn splits)
  Co-purchase graphs  : Amazon-Photo, Amazon-Computers  (generated splits)
  WebKB               : Texas, Wisconsin, Cornell       (geom-gcn splits)
  Actor                                                 (geom-gcn splits)
  Heterophily bench.  : Roman-Empire                    (own 10 splits)
  struc2vec Airports  : USA/Brazil/Europe-Airports      (generated splits)

Every returned Data object exposes .train_mask/.val_mask/.test_mask as
[num_nodes, n_splits] boolean tensors (n_splits=10 for everything except
Roman-Empire, which ships its own split count) and .num_classes, so every
downstream script can treat all 13 datasets identically.
"""
import os

import torch
import torch.nn.functional as F
from torch_geometric.datasets import (WebKB, Actor, Airports, Planetoid,
                                       Amazon, HeterophilousGraphDataset)
from torch_geometric.utils import add_self_loops

_DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")

_PLANETOID = ("cora", "citeseer", "pubmed")
_WEBKB = ("texas", "wisconsin", "cornell")
_AMAZON = {"amazon-photo": "Photo", "amazon-computers": "Computers"}
_AIRPORTS_PYG_NAME = {"usa-airports": "usa", "brazil-airports": "brazil",
                       "europe-airports": "europe"}

ALL_DATASETS = (list(_PLANETOID) + list(_WEBKB) + list(_AMAZON.keys()) +
                ["actor", "roman-empire"] + list(_AIRPORTS_PYG_NAME.keys()))


def _generate_stratified_splits(y, n_splits=10, train_ratio=0.6, val_ratio=0.2):
    """n_splits random stratified 60/20/20 splits (per class), seeded by
    split index -- same convention used for every generated-split dataset
    in the earlier stages of this project (data.py)."""
    n = y.size(0)
    train_mask = torch.zeros(n, n_splits, dtype=torch.bool)
    val_mask = torch.zeros(n, n_splits, dtype=torch.bool)
    test_mask = torch.zeros(n, n_splits, dtype=torch.bool)
    classes = y.unique().tolist()

    for split_idx in range(n_splits):
        g = torch.Generator().manual_seed(split_idx)
        for c in classes:
            idx = (y == c).nonzero(as_tuple=True)[0]
            perm = idx[torch.randperm(idx.size(0), generator=g)]
            n_train = max(1, int(round(train_ratio * perm.size(0))))
            n_val = max(1, int(round(val_ratio * perm.size(0))))
            train_mask[perm[:n_train], split_idx] = True
            val_mask[perm[n_train:n_train + n_val], split_idx] = True
            test_mask[perm[n_train + n_val:], split_idx] = True

    return train_mask, val_mask, test_mask


def load_dataset(name: str, root: str = None, n_splits: int = 10):
    """Load a dataset by name -- see ALL_DATASETS for the valid list.
    Returns a torch_geometric.data.Data object with num_classes set and
    train/val/test masks of shape [num_nodes, n_splits_available]."""
    root = root or _DEFAULT_ROOT
    key = name.lower()

    if key in _PLANETOID:
        # NOTE: .capitalize() gives "Citeseer"/"Pubmed" (not "CiteSeer"/
        # "PubMed") -- PyG's Planetoid accepts this fine (proven in the
        # earlier stage of this project across many runs), so this is
        # intentional, not a typo.
        ds = Planetoid(root=f"{root}/Planetoid", name=key.capitalize(), split="geom-gcn")
        data = ds[0]
        data.train_mask = data.train_mask.bool()
        data.val_mask = data.val_mask.bool()
        data.test_mask = data.test_mask.bool()

    elif key in _WEBKB:
        ds = WebKB(root=f"{root}/WebKB", name=key.capitalize())
        data = ds[0]

    elif key == "actor":
        ds = Actor(root=f"{root}/Actor")
        data = ds[0]

    elif key in _AMAZON:
        ds = Amazon(root=f"{root}/Amazon", name=_AMAZON[key])
        data = ds[0]
        train_mask, val_mask, test_mask = _generate_stratified_splits(data.y, n_splits=n_splits)
        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

    elif key == "roman-empire":
        ds = HeterophilousGraphDataset(root=f"{root}/HeterophilousGraphDataset", name="Roman-empire")
        data = ds[0]
        # Ships with its own splits (10, per Platonov et al. 2023). NOT
        # network-verified in this repo's dev environment --
        # raw.githubusercontent.com was blocked by the sandbox's network
        # allowlist, so this download has never actually completed here.
        # The defensive reshape below handles the case where PyG returns
        # 1-D masks (a single split) instead of the expected [n, n_splits];
        # double check data.train_mask.shape on first real run.
        if data.train_mask.dim() == 1:
            data.train_mask = data.train_mask.unsqueeze(-1)
            data.val_mask = data.val_mask.unsqueeze(-1)
            data.test_mask = data.test_mask.unsqueeze(-1)

    elif key in _AIRPORTS_PYG_NAME:
        ds = Airports(root=f"{root}/Airports", name=_AIRPORTS_PYG_NAME[key])
        data = ds[0]
        train_mask, val_mask, test_mask = _generate_stratified_splits(data.y, n_splits=n_splits)
        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

    else:
        raise ValueError(f"Unknown dataset '{name}'. Valid names: {ALL_DATASETS}")

    data.num_classes = int(data.y.max().item()) + 1
    return data


def num_available_splits(data) -> int:
    """How many splits this dataset actually has (Roman-Empire may differ
    from the standard 10 used everywhere else)."""
    return data.train_mask.size(1)


# ---------------------------------------------------------------------------
# Static structural descriptors used by the ARIA belief gate(s). Unchanged
# from the original project's data.py.
# ---------------------------------------------------------------------------

def compute_structural_features(data, richer: bool = False):
    """s_i for the ARIA / ARIA-PH / ARIA-MoE gate.
    richer=False: normalized [degree, pagerank] -> dim 2 (used by the gate).
    richer=True: adds local feature-homophily + 2-hop degree profile -> dim 4
    (used only as clustering input for ARIA-MoE's fixed regime membership,
    never fed to the gate itself).
    """
    n = data.num_nodes
    edge_index, _ = add_self_loops(data.edge_index, num_nodes=n)
    row, col = edge_index

    deg = torch.zeros(n)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))

    out_deg = deg.clone()
    out_deg[out_deg == 0] = 1.0
    damping = 0.85
    pr = torch.full((n,), 1.0 / n)
    for _ in range(50):
        msg = pr[row] / out_deg[row]
        new_pr = torch.zeros(n)
        new_pr.scatter_add_(0, col, msg)
        pr = (1 - damping) / n + damping * new_pr
        pr = pr / pr.sum()

    def normalize(t):
        t = t.float()
        mean, std = t.mean(), t.std()
        if std < 1e-8:
            return torch.zeros_like(t)
        return (t - mean) / std

    deg_n = normalize(deg)
    pr_n = normalize(pr)
    if not richer:
        return torch.stack([deg_n, pr_n], dim=1)

    deg2 = torch.zeros(n)
    deg2.scatter_add_(0, row, deg[col])
    deg2_n = normalize(deg2)

    x_norm = F.normalize(data.x.float(), dim=-1, eps=1e-8)
    sim = (x_norm[row] * x_norm[col]).sum(-1)
    homophily = torch.zeros(n)
    homophily.scatter_add_(0, row, sim)
    cnt = torch.zeros(n)
    cnt.scatter_add_(0, row, torch.ones_like(sim))
    cnt[cnt == 0] = 1.0
    homophily = homophily / cnt
    homophily_n = normalize(homophily)

    return torch.stack([deg_n, pr_n, homophily_n, deg2_n], dim=1)


def compute_structural_features_v3(data, eps: float = 1e-8):
    """s_i = [deg(i), PageRank(i), feature-homophily(i)], dim 3, each column
    min-max normalized to [0, 1]. Used by ARIA-PHG's bounded gate."""
    n = data.num_nodes
    edge_index = data.edge_index
    row, col = edge_index

    deg = torch.zeros(n)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))

    deg_for_pr = deg.clone().clamp(min=1.0)
    damping = 0.85
    pr = torch.full((n,), 1.0 / n)
    for _ in range(30):
        msg = pr[row] / deg_for_pr[row]
        new_pr = torch.zeros(n)
        new_pr.scatter_add_(0, col, msg)
        pr = damping * new_pr + (1.0 - damping) / n

    x_norm = F.normalize(data.x.float(), p=2, dim=1)
    sim = (x_norm[row] * x_norm[col]).sum(dim=1)
    homophily = torch.zeros(n)
    count = torch.zeros(n)
    homophily.scatter_add_(0, col, sim)
    count.scatter_add_(0, col, torch.ones_like(sim))
    valid = count > 0
    homophily[valid] = homophily[valid] / count[valid]

    s = torch.stack([deg, pr, homophily], dim=1)
    s_min = s.min(dim=0, keepdim=True).values
    s_max = s.max(dim=0, keepdim=True).values
    return (s - s_min) / (s_max - s_min + eps)
