"""Priority-1 appendix experiment: channel ablation, isolating the value of
ARIA's adaptive per-node routing itself.

    Model          Self  1-hop  2-hop  Global  Node-adaptive
    MLP             X
    GCN                    X
    2-hop-only                    X
    Fixed-mixture   X      X      X      X
    ARIA            X      X      X      X         X

MLP, GCN, and ARIA (baseline) already have real final-phase results from
the main sweep (results/baseline_search.json, results/aria_search.json --
see results/FINAL_RESULTS.md) and are NOT retrained here. This script adds
the two missing rows:

  - "2-hop only": propagate exactly 2 hops (Z2 = A_hat(A_hat H)) at every
    layer, no self/1-hop/global channels, no gate -- tests whether 2-hop
    propagation alone, with no routing choice at all, is competitive.
  - "Fixed mixture": the exact same 4 channels as ARIA (self/1-hop/2-hop/
    low-rank global) mixed with FIXED, non-learned, uniform weights
    (0.25 each) instead of ARIA's per-node adaptive softmax gate -- holds
    the channel set fixed and isolates the value of adaptive routing,
    which is the point of this ablation.

Both variants deliberately reuse the winning ARIA hyperparameters per
dataset from results/aria_search.json (hidden_dim, lr, weight_decay,
dropout, num_layers, rank) rather than an independent grid search, since
they share ARIA's architecture -- consistent with how this project reused
existing tuned configs for other appendix-only analyses.

Deliberately self-contained (new model classes + training loop defined
here, not added to src/models/aria.py or src/training.py) so this
one-off ablation carries zero risk to the already-validated main benchmark
pipeline, following the project's existing convention of keeping
exploratory/appendix-only additions isolated (see aria_cnn_pilot/).

Usage:
    PYTHONPATH=/tmp/pylibs:. python3 scripts/run_channel_ablation.py
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import load_dataset
from src.models.aria import build_norm_adj, spmm_propagate
from src.search_space import DATASET_TRAIN_CONFIG

PAPER_11_DATASETS = [
    "cora", "citeseer", "pubmed", "texas", "wisconsin", "cornell",
    "amazon-photo", "actor", "usa-airports", "brazil-airports", "europe-airports",
]


# ---------------------------------------------------------------------------
# "2-hop only": pure 2-hop propagation, no gate, no other channels
# ---------------------------------------------------------------------------

class TwoHopLayer(nn.Module):
    def __init__(self, dim, dropout=0.5):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, H, edge_index, norm):
        Z1 = spmm_propagate(edge_index, norm, H)
        Z2 = spmm_propagate(edge_index, norm, Z1)
        H_tilde = F.dropout(Z2, p=self.dropout, training=self.training)
        out = F.relu(self.W(H_tilde))
        return self.ln(H + out)


class TwoHopOnly(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([TwoHopLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.out_lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        self._cached_adj = None

    def forward(self, x, edge_index, **kwargs):
        n = x.size(0)
        if self._cached_adj is None or self._cached_adj[0].device != x.device:
            self._cached_adj = build_norm_adj(edge_index, n)
        norm_edge_index, norm_weight = self._cached_adj
        H = F.dropout(x, p=self.dropout, training=self.training)
        H = F.relu(self.input_proj(H))
        for layer in self.layers:
            H = layer(H, norm_edge_index, norm_weight)
        H = F.dropout(H, p=self.dropout, training=self.training)
        return self.out_lin(H)


# ---------------------------------------------------------------------------
# "Fixed mixture": ARIA's exact 4 channels, fixed uniform (non-learned)
# mixture weights instead of the per-node adaptive gate
# ---------------------------------------------------------------------------

class FixedMixLayer(nn.Module):
    def __init__(self, dim, num_nodes, rank=16, dropout=0.5):
        super().__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.empty(num_nodes, rank))
        self.V = nn.Parameter(torch.empty(num_nodes, rank))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        self.W = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.dropout = dropout
        self.register_buffer("fixed_beta", torch.full((4,), 0.25))

    def forward(self, H, edge_index, norm):
        Z0 = H
        Z1 = spmm_propagate(edge_index, norm, H)
        Z2 = spmm_propagate(edge_index, norm, Z1)
        Zg = (self.U @ (self.V.t() @ H)) / math.sqrt(self.rank)
        b = self.fixed_beta
        H_tilde = b[0] * Z0 + b[1] * Z1 + b[2] * Z2 + b[3] * Zg
        H_tilde = F.dropout(H_tilde, p=self.dropout, training=self.training)
        out = F.relu(self.W(H_tilde))
        return self.ln(H + out)


class FixedMixture(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_nodes, num_layers=2, rank=16, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            FixedMixLayer(hidden_dim, num_nodes, rank=rank, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        self._cached_adj = None

    def forward(self, x, edge_index, **kwargs):
        n = x.size(0)
        if self._cached_adj is None or self._cached_adj[0].device != x.device:
            self._cached_adj = build_norm_adj(edge_index, n)
        norm_edge_index, norm_weight = self._cached_adj
        H = F.dropout(x, p=self.dropout, training=self.training)
        H = F.relu(self.input_proj(H))
        for layer in self.layers:
            H = layer(H, norm_edge_index, norm_weight)
        H = F.dropout(H, p=self.dropout, training=self.training)
        return self.out_lin(H)


# ---------------------------------------------------------------------------
# Training loop (mirrors src/training.py's train_aria protocol exactly:
# same optimizer, same early-stopping rule, same epoch budget/patience per
# dataset from search_space.DATASET_TRAIN_CONFIG)
# ---------------------------------------------------------------------------

def train_one_split(model_name, data, split_idx, cfg, max_epochs, patience, seed=0):
    torch.manual_seed(seed)
    x, y, edge_index = data.x, data.y, data.edge_index
    train_mask = data.train_mask[:, split_idx]
    val_mask = data.val_mask[:, split_idx]
    test_mask = data.test_mask[:, split_idx]

    if model_name == "two_hop_only":
        model = TwoHopOnly(x.size(1), cfg["hidden_dim"], data.num_classes,
                            num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    elif model_name == "fixed_mixture":
        model = FixedMixture(x.size(1), cfg["hidden_dim"], data.num_classes, x.size(0),
                              num_layers=cfg["num_layers"], rank=cfg["rank"], dropout=cfg["dropout"])
    else:
        raise ValueError(model_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_val, best_test, no_improve = -1.0, 0.0, 0
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index)
            val_acc = (logits[val_mask].argmax(-1) == y[val_mask]).float().mean().item()
            test_acc = (logits[test_mask].argmax(-1) == y[test_mask]).float().mean().item()

        if val_acc > best_val:
            best_val, best_test, no_improve = val_acc, test_acc, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return best_test, time.time() - t0


def run_dataset(name, aria_configs, split_indices, model_name, existing=None):
    """Trains `model_name` on `name` for exactly the given split indices
    (allows chunking a 10-split run across multiple short calls). Merges
    into `existing` (a per_split dict: {str(split_idx): {"acc":.., "time":..}})
    if given, so repeated partial calls accumulate rather than overwrite."""
    data = load_dataset(name)
    cfg = dict(aria_configs[name]["aria"]["best_config"])
    train_cfg = DATASET_TRAIN_CONFIG[name]

    per_split = dict(existing) if existing else {}
    for split_idx in split_indices:
        acc, t = train_one_split(model_name, data, split_idx, cfg,
                                  train_cfg["max_epochs"], train_cfg["patience"],
                                  seed=split_idx)
        per_split[str(split_idx)] = {"acc": acc * 100, "time": t}

    accs = [v["acc"] for v in per_split.values()]
    times = [v["time"] for v in per_split.values()]
    import statistics
    summary = {
        "mean": statistics.mean(accs), "std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
        "n_splits": len(accs), "per_split": per_split, "config": cfg,
        "train_time_mean_sec": statistics.mean(times),
        "train_time_std_sec": statistics.pstdev(times) if len(times) > 1 else 0.0,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=PAPER_11_DATASETS)
    ap.add_argument("--models", nargs="+", default=["two_hop_only", "fixed_mixture"])
    ap.add_argument("--splits", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--out", default="results/channel_ablation.json")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "results/aria_search.json")) as f:
        aria_configs = json.load(f)

    out_path = os.path.join(root, args.out)
    all_results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)

    import statistics
    for name in args.datasets:
        data = None
        cfg = None
        train_cfg = DATASET_TRAIN_CONFIG[name]
        for model_name in args.models:
            existing = all_results.get(name, {}).get(model_name, {}).get("per_split", {})
            todo = [s for s in args.splits if str(s) not in existing]
            if not todo:
                print(f"=== {name} / {model_name}: all requested splits already done, skipping ===", flush=True)
                continue
            if data is None:
                data = load_dataset(name)
                cfg = dict(aria_configs[name]["aria"]["best_config"])
            per_split = dict(existing)
            for split_idx in todo:
                acc, t = train_one_split(model_name, data, split_idx, cfg,
                                          train_cfg["max_epochs"], train_cfg["patience"],
                                          seed=split_idx)
                per_split[str(split_idx)] = {"acc": acc * 100, "time": t}
                accs = [v["acc"] for v in per_split.values()]
                times = [v["time"] for v in per_split.values()]
                summary = {
                    "mean": statistics.mean(accs),
                    "std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                    "n_splits": len(accs), "per_split": per_split, "config": cfg,
                    "train_time_mean_sec": statistics.mean(times),
                    "train_time_std_sec": statistics.pstdev(times) if len(times) > 1 else 0.0,
                }
                all_results.setdefault(name, {})[model_name] = summary
                with open(out_path, "w") as f:
                    json.dump(all_results, f, indent=2)
                print(f"  {name}/{model_name} split {split_idx}: acc={acc*100:.2f} "
                      f"(time {t:.2f}s) -- running mean {summary['mean']:.2f}±{summary['std']:.2f}",
                      flush=True)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
