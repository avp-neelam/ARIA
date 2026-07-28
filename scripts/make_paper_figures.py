#!/usr/bin/env python3
"""Regenerates every figure the rewritten (11-dataset) paper needs, as PDF:

  efficiency.pdf              -- mean accuracy vs mean training time (log-x)
  summary_avg_rank.pdf        -- (a) mean accuracy bars (b) mean rank bars
  belief_homophily_all11.pdf  -- per-dataset scatter: propagation mass vs
                                  true label homophily, base ARIA gate
  belief_vector_composition.pdf -- per-dataset stacked-bar belief composition

Reads results/paper_stats_11ds.json (built by build_paper_stats.py) for the
first two, and artifacts/<dataset>/aria/split_0.pt (real saved belief
vectors from the final HPC run) + a fresh load_dataset() call (for true
label homophily, which isn't stored in the artifact) for the last two.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import load_dataset

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIRS = [
    os.path.join(os.path.dirname(BASE), "paper", "figures"),
    os.path.join(os.path.dirname(BASE), "paper-draft", "figures"),
]

DATASETS = ["cora", "citeseer", "pubmed", "texas", "wisconsin", "cornell",
            "amazon-photo", "actor", "usa-airports", "brazil-airports", "europe-airports"]
DATASET_DISPLAY = {
    "cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed", "texas": "Texas",
    "wisconsin": "Wisconsin", "cornell": "Cornell", "amazon-photo": "Amazon-Photo",
    "actor": "Actor", "usa-airports": "USA-Airports", "brazil-airports": "Brazil-Airports",
    "europe-airports": "Europe-Airports",
}
MODELS = ["mlp", "gcn", "graphsage", "gat", "gps",
          "aria", "aria_ph", "aria_phg", "aria_moe", "aria_ensemble", "aria_ensemble2"]
DISPLAY = {
    "mlp": "MLP", "gcn": "GCN", "graphsage": "GraphSAGE", "gat": "GAT", "gps": "GPS",
    "aria": "ARIA", "aria_ph": "ARIA-PH", "aria_phg": "ARIA-PHG",
    "aria_moe": "ARIA-MoE", "aria_ensemble": "ARIA-Ensemble", "aria_ensemble2": "ARIA-Ensemble2",
}
COLORS = {
    "mlp": "#7f7f7f", "gcn": "#1f77b4", "graphsage": "#1f77b4", "gat": "#2ca02c",
    "gps": "#9467bd", "aria": "#e6c619", "aria_ph": "#ff7f0e", "aria_phg": "#d62728",
    "aria_moe": "#e377c2", "aria_ensemble": "#000000", "aria_ensemble2": "#17becf",
}


def save_fig(fig, name):
    fig.tight_layout()
    for d in FIG_DIRS:
        if os.path.isdir(d):
            path = os.path.join(d, name)
            fig.savefig(path, bbox_inches="tight")
            print("wrote", path)


def fig_efficiency(stats):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    # manual (dx_mult, dy_abs, ha) label offsets, tuned for this exact point
    # cloud so nothing overlaps -- several models land within ~1s / ~1pt of
    # each other (aria_ph/aria_moe/graphsage/gat; aria_phg/gps).
    offsets = {
        "mlp": (1.15, 0.0, "left"),
        "gcn": (1.0, -1.0, "center"),
        "graphsage": (0.55, 0.9, "right"),
        "gat": (1.0, -1.1, "center"),
        "gps": (1.0, -1.2, "center"),
        "aria": (1.15, -0.3, "left"),
        "aria_ph": (1.0, -1.1, "center"),
        "aria_phg": (1.15, 0.6, "left"),
        "aria_moe": (1.15, 0.9, "left"),
        "aria_ensemble": (0.85, 1.6, "right"),
        "aria_ensemble2": (1.05, -1.6, "left"),
    }
    for m in MODELS:
        t = stats["mean_time_sec"][m]
        a = stats["overall_mean_acc"][m]
        marker = "*" if m == "aria_ensemble" else ("D" if m == "aria_ensemble2" else "o")
        size = 170 if m in ("aria_ensemble", "aria_ensemble2") else 55
        ax.scatter(t, a, color=COLORS[m], marker=marker, s=size, zorder=3,
                   edgecolor="black" if m in ("aria_ensemble", "aria_ensemble2") else "none", linewidth=0.8)
        dx, dy, ha = offsets[m]
        ax.annotate(DISPLAY[m], (t, a), xytext=(t * dx, a + dy), fontsize=7, ha=ha, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.12, 40)
    ax.set_ylim(58.5, 73)
    ax.set_xlabel("Mean training time per run (s, log scale)")
    ax.set_ylabel("Mean accuracy (%), 11 datasets", fontsize=8)
    ax.set_title("Accuracy vs.\\ compute cost")
    save_fig(fig, "efficiency.pdf")
    plt.close(fig)


def fig_summary_avg_rank(stats):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    order_acc = sorted(MODELS, key=lambda m: -stats["overall_mean_acc"][m])
    ax = axes[0]
    ax.bar(range(len(order_acc)), [stats["overall_mean_acc"][m] for m in order_acc],
           color=[COLORS[m] for m in order_acc])
    ax.set_xticks(range(len(order_acc)))
    ax.set_xticklabels([DISPLAY[m] for m in order_acc], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Mean accuracy (%), 11 datasets", fontsize=8)
    ax.set_ylim(min(stats["overall_mean_acc"].values()) - 3, max(stats["overall_mean_acc"].values()) + 2)
    ax.set_title("(a) Average accuracy")

    order_rank = sorted(MODELS, key=lambda m: stats["mean_rank"][m])
    ax = axes[1]
    ax.bar(range(len(order_rank)), [stats["mean_rank"][m] for m in order_rank],
           color=[COLORS[m] for m in order_rank])
    ax.set_xticks(range(len(order_rank)))
    ax.set_xticklabels([DISPLAY[m] for m in order_rank], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Mean rank (1=best), 11 datasets", fontsize=8)
    ax.invert_yaxis()
    ax.set_title("(b) Mean rank")

    save_fig(fig, "summary_avg_rank.pdf")
    plt.close(fig)


def label_homophily_per_node(data):
    row, col = data.edge_index
    same = (data.y[row] == data.y[col]).float()
    n = data.num_nodes
    num = torch.zeros(n)
    den = torch.zeros(n)
    num.scatter_add_(0, row, same)
    den.scatter_add_(0, row, torch.ones_like(same))
    den[den == 0] = 1.0
    return (num / den).numpy()


def load_belief(dataset):
    path = os.path.join(BASE, "artifacts", dataset, "aria", "split_0.pt")
    art = torch.load(path, map_location="cpu", weights_only=False)
    return art["belief"]["beta"].numpy()  # [n, 4] : self, 1-hop, 2-hop, global


def fig_belief_homophily():
    n_ds = len(DATASETS)
    ncols = 4
    nrows = int(np.ceil(n_ds / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 1.9 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)

    for i, ds in enumerate(DATASETS):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        data = load_dataset(ds)
        homophily = label_homophily_per_node(data)
        beta = load_belief(ds)
        prop_mass = beta[:, 1] + beta[:, 2]
        corr = np.corrcoef(prop_mass, homophily)[0, 1]
        ax.scatter(homophily, prop_mass, s=3, alpha=0.35, color="#1f77b4", linewidths=0)
        ax.set_title(f"{DATASET_DISPLAY[ds]} (r={corr:.2f})", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=6)

    for j in range(n_ds, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.supxlabel("True per-node label homophily", fontsize=9)
    fig.supylabel(r"GraphReliance ($\beta_1+\beta_2$)", fontsize=9)
    save_fig(fig, "belief_homophily_all11.pdf")
    plt.close(fig)
    return {ds: None for ds in DATASETS}  # correlations printed above, saved separately below


def fig_belief_composition():
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    labels = ["Self", "1-hop", "2-hop", "Global"]
    colors = ["#7f7f7f", "#9ecae1", "#3182bd", "#e6550d"]

    means = []
    for ds in DATASETS:
        beta = load_belief(ds)
        means.append(beta.mean(0))
    means = np.array(means)  # [n_ds, 4]

    y = np.arange(len(DATASETS))
    left = np.zeros(len(DATASETS))
    for k in range(4):
        ax.barh(y, means[:, k], left=left, color=colors[k], label=labels[k])
        left += means[:, k]

    ax.set_yticks(y)
    ax.set_yticklabels([DATASET_DISPLAY[ds] for ds in DATASETS])
    ax.invert_yaxis()
    ax.set_xlabel("Mean belief weight")
    ax.set_xlim(0, 1)
    ax.set_title("Belief-vector composition by dataset (base ARIA gate)", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=8)
    save_fig(fig, "belief_vector_composition.pdf")
    plt.close(fig)


def print_correlations():
    print("=== belief/homophily correlations (11 datasets) ===")
    corrs = {}
    for ds in DATASETS:
        data = load_dataset(ds)
        homophily = label_homophily_per_node(data)
        beta = load_belief(ds)
        prop_mass = beta[:, 1] + beta[:, 2]
        corr = float(np.corrcoef(prop_mass, homophily)[0, 1])
        corrs[ds] = corr
        print(f"  {DATASET_DISPLAY[ds]:18s} r={corr:.3f}")
    with open(os.path.join(BASE, "results", "belief_homophily_corr_11ds.json"), "w") as f:
        json.dump(corrs, f, indent=2)


def main():
    with open(os.path.join(BASE, "results", "paper_stats_11ds.json")) as f:
        stats = json.load(f)
    fig_efficiency(stats)
    fig_summary_avg_rank(stats)
    fig_belief_homophily()
    fig_belief_composition()
    print_correlations()


if __name__ == "__main__":
    main()
