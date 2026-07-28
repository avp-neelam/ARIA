"""Priority-2 appendix analysis: does the ARIA belief gate's per-node
weighting actually track which channel helps that node?

For the base ARIA model (bare 4-way softmax gate: self / 1-hop / 2-hop /
low-rank global), this script reuses the already-trained, already-saved
final-phase weights in artifacts/<dataset>/aria/split_<i>.pt (no retraining)
and asks a causal question per node i and per channel c in {2-hop, global}:

    delta_c(i) = P_full(y_i) - P_ablate_c(y_i)

i.e. how much true-class probability node i loses when channel c's
representation is zeroed out at every ARIA layer, holding the model's own
gate (beta, computed from the UN-ablated forward pass) fixed. beta is left
untouched by the ablation deliberately: we want to test whether the belief
the model already assigned to a channel predicts how much that channel's
information actually mattered, not let the ablation change the gate's own
assessment.

The hypothesis (from the paper's interpretability claim, Section on
belief-vector diagnostics): nodes with high beta_{i,2} (2-hop reliance)
should show larger delta_2hop(i), and nodes with high beta_{i,3} (global
reliance) should show larger delta_global(i). Evaluated as a per-node
Pearson/Spearman correlation, on test-split nodes, per dataset per split,
then aggregated across the 10 final splits and across the 11-dataset paper
suite. A node-shuffle permutation null is included for calibration (see
--n_perm).

Usage:
    PYTHONPATH=/tmp/pylibs:. python3 scripts/validate_belief_vectors.py \
        --datasets cora citeseer ... --out results/belief_validation.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import load_dataset, compute_structural_features
from src.models.aria import ARIA, build_norm_adj, spmm_propagate
from src.artifact_logger import load_artifact

PAPER_11_DATASETS = [
    "cora", "citeseer", "pubmed", "texas", "wisconsin", "cornell",
    "amazon-photo", "actor", "usa-airports", "brazil-airports", "europe-airports",
]

CHANNEL_NAMES = {0: "self", 1: "1-hop", 2: "2-hop", 3: "global"}


def rebuild_model(art, in_dim, num_classes, num_nodes, struct_dim):
    cfg = art["config"]
    model = ARIA(in_dim, cfg["hidden_dim"], num_classes, num_nodes,
                 num_layers=cfg["num_layers"], rank=cfg["rank"],
                 struct_dim=struct_dim, dropout=cfg["dropout"])
    model.load_state_dict(art["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def forward_full_and_ablated(model, x, edge_index, struct_feat, ablate_channel):
    """Returns (logits_full, beta_full [n,4], logits_ablated). beta_full is
    from the UN-ablated run (mean over layers) -- see module docstring for
    why the gate itself is never touched by the ablation."""
    n = x.size(0)
    norm_edge_index, norm_weight = build_norm_adj(edge_index, n)

    # --- full (un-ablated) pass, also gives us beta_full ---
    H = F.relu(model.input_proj(x))
    betas_full = []
    for layer in model.layers:
        Z0 = H
        Z1 = spmm_propagate(norm_edge_index, norm_weight, H)
        Z2 = spmm_propagate(norm_edge_index, norm_weight, Z1)
        Zg = (layer.U @ (layer.V.t() @ H)) / math.sqrt(layer.rank)
        gate_in = torch.cat([H, struct_feat], dim=-1)
        beta = F.softmax(layer.W_beta(gate_in), dim=-1)
        betas_full.append(beta)
        H_tilde = (beta[:, 0:1] * Z0 + beta[:, 1:2] * Z1 +
                   beta[:, 2:3] * Z2 + beta[:, 3:4] * Zg)
        out = F.relu(layer.W(H_tilde))
        H = layer.ln(H + out)
    logits_full = model.out_lin(H)
    beta_full = torch.stack(betas_full, dim=0).mean(0)

    # --- ablated pass: zero channel `ablate_channel`'s representation at
    # every layer, but reuse each layer's OWN freshly-computed beta (i.e.
    # the gate reacts to the ablated H as it flows forward -- only the
    # channel's information is denied, nothing about the gate mechanism
    # itself is frozen) ---
    H = F.relu(model.input_proj(x))
    for layer in model.layers:
        Z0 = H
        Z1 = spmm_propagate(norm_edge_index, norm_weight, H)
        Z2 = spmm_propagate(norm_edge_index, norm_weight, Z1)
        Zg = (layer.U @ (layer.V.t() @ H)) / math.sqrt(layer.rank)
        if ablate_channel == 0:
            Z0 = torch.zeros_like(Z0)
        elif ablate_channel == 1:
            Z1 = torch.zeros_like(Z1)
        elif ablate_channel == 2:
            Z2 = torch.zeros_like(Z2)
        elif ablate_channel == 3:
            Zg = torch.zeros_like(Zg)
        gate_in = torch.cat([H, struct_feat], dim=-1)
        beta = F.softmax(layer.W_beta(gate_in), dim=-1)
        H_tilde = (beta[:, 0:1] * Z0 + beta[:, 1:2] * Z1 +
                   beta[:, 2:3] * Z2 + beta[:, 3:4] * Zg)
        out = F.relu(layer.W(H_tilde))
        H = layer.ln(H + out)
    logits_ablated = model.out_lin(H)

    return logits_full, beta_full, logits_ablated


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return pearson(ra, rb)


def permutation_null(a, b, n_perm, rng):
    """Null distribution of Pearson r under random shuffles of b, for a
    quick calibration check (task: 'is the observed correlation bigger
    than chance node-index alignment would produce')."""
    obs = pearson(a, b)
    null = np.empty(n_perm)
    b = b.copy()
    for k in range(n_perm):
        rng.shuffle(b)
        null[k] = pearson(a, b)
    p_two_sided = float((np.abs(null) >= abs(obs)).mean())
    return obs, null, p_two_sided


def run_dataset(name, n_perm=200, seed=0, channels=(2, 3)):
    data = load_dataset(name)
    struct = compute_structural_features(data)
    x, edge_index, y = data.x, data.edge_index, data.y
    n_splits = data.train_mask.size(1)
    rng = np.random.default_rng(seed)

    per_split = []
    for split_idx in range(min(10, n_splits)):
        path = f"artifacts/{name}/aria/split_{split_idx}.pt"
        if not os.path.exists(path):
            continue
        art = load_artifact(path)
        if art.get("model_state_dict") is None:
            continue
        model = rebuild_model(art, x.size(1), data.num_classes, x.size(0), struct.size(1))

        test_mask = data.test_mask[:, split_idx]
        test_idx = test_mask.nonzero(as_tuple=True)[0]

        split_result = {"split": split_idx, "n_test": int(test_idx.numel()), "channels": {}}
        for c in channels:
            logits_full, beta_full, logits_ablated = forward_full_and_ablated(
                model, x, edge_index, struct, ablate_channel=c)
            p_full = F.softmax(logits_full, dim=-1)
            p_abl = F.softmax(logits_ablated, dim=-1)
            true_p_full = p_full.gather(1, y.unsqueeze(1)).squeeze(1)
            true_p_abl = p_abl.gather(1, y.unsqueeze(1)).squeeze(1)
            delta = (true_p_full - true_p_abl)  # positive = channel c helped

            beta_c_test = beta_full[test_idx, c].numpy()
            delta_test = delta[test_idx].numpy()

            r = pearson(beta_c_test, delta_test)
            rho = spearman(beta_c_test, delta_test)
            obs, null, p_val = permutation_null(beta_c_test, delta_test, n_perm, rng)

            # top/bottom quartile split by beta_c on test nodes
            q75, q25 = np.percentile(beta_c_test, [75, 25])
            top = delta_test[beta_c_test >= q75]
            bot = delta_test[beta_c_test <= q25]

            split_result["channels"][str(c)] = {
                "pearson_r": r, "spearman_rho": rho, "perm_p_value": p_val,
                "n_perm": n_perm,
                "mean_beta_c": float(beta_c_test.mean()),
                "mean_delta": float(delta_test.mean()),
                "top_quartile_mean_delta": float(top.mean()) if top.size else float("nan"),
                "bottom_quartile_mean_delta": float(bot.mean()) if bot.size else float("nan"),
                "top_quartile_n": int(top.size), "bottom_quartile_n": int(bot.size),
            }
        per_split.append(split_result)

    agg = {"channels": {}}
    for c in channels:
        rs = np.array([s["channels"][str(c)]["pearson_r"] for s in per_split
                        if not math.isnan(s["channels"][str(c)]["pearson_r"])])
        rhos = np.array([s["channels"][str(c)]["spearman_rho"] for s in per_split
                          if not math.isnan(s["channels"][str(c)]["spearman_rho"])])
        top = np.array([s["channels"][str(c)]["top_quartile_mean_delta"] for s in per_split])
        bot = np.array([s["channels"][str(c)]["bottom_quartile_mean_delta"] for s in per_split])
        agg["channels"][str(c)] = {
            "mean_pearson_r": float(rs.mean()) if rs.size else float("nan"),
            "std_pearson_r": float(rs.std()) if rs.size else float("nan"),
            "mean_spearman_rho": float(rhos.mean()) if rhos.size else float("nan"),
            "mean_top_quartile_delta": float(np.nanmean(top)),
            "mean_bottom_quartile_delta": float(np.nanmean(bot)),
            "n_splits_used": int(len(per_split)),
        }

    return {"dataset": name, "n_splits_used": len(per_split),
            "per_split": per_split, "aggregate": agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=PAPER_11_DATASETS)
    ap.add_argument("--out", default="results/belief_validation.json")
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--channels", nargs="+", type=int, default=[2, 3])
    args = ap.parse_args()

    all_results = {}
    for name in args.datasets:
        print(f"=== {name} ===", flush=True)
        res = run_dataset(name, n_perm=args.n_perm, channels=args.channels)
        all_results[name] = res
        for c in args.channels:
            a = res["aggregate"]["channels"][str(c)]
            print(f"  channel={CHANNEL_NAMES[c]:6s} mean_r={a['mean_pearson_r']:+.3f} "
                  f"mean_rho={a['mean_spearman_rho']:+.3f} "
                  f"top_q_delta={a['mean_top_quartile_delta']:+.4f} "
                  f"bot_q_delta={a['mean_bottom_quartile_delta']:+.4f} "
                  f"(n_splits={a['n_splits_used']})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
