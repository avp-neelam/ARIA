#!/usr/bin/env python3
"""Regenerates the two figures that depend on which models are in the
lineup (efficiency.pdf, summary_avg_rank.pdf) for the trimmed 9-model
paper (5 baselines + ARIA, ARIA-PHG, ARIA-MoE, ARIA-Ensemble), per advisor
feedback (2026-07-23) to cut ARIA-PH and ARIA-Ensemble2 as standalone rows.

belief_homophily_all11.pdf and belief_vector_composition.pdf are untouched
by this trim (they only ever depended on the base ARIA gate's artifacts,
not on which family members are reported), so this script does not
regenerate them -- see make_paper_figures.py for those.

Reads results/paper_stats_9model.json (built by build_paper_stats_4model.py).
Writes efficiency.pdf and summary_avg_rank.pdf directly (overwriting the
11-model versions) into paper/figures and paper-draft/figures.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIRS = [
    os.path.join(os.path.dirname(BASE), "paper", "figures"),
]

MODELS = ["mlp", "gcn", "graphsage", "gat", "gps",
          "aria", "aria_phg", "aria_moe", "aria_ensemble"]
DISPLAY = {
    "mlp": "MLP", "gcn": "GCN", "graphsage": "GraphSAGE", "gat": "GAT", "gps": "GPS",
    "aria": "ARIA", "aria_phg": "ARIA-PHG",
    "aria_moe": "ARIA-MoE", "aria_ensemble": "ARIA-Ensemble",
}
COLORS = {
    "mlp": "#7f7f7f", "gcn": "#1f77b4", "graphsage": "#1f77b4", "gat": "#2ca02c",
    "gps": "#9467bd", "aria": "#e6c619", "aria_phg": "#d62728",
    "aria_moe": "#e377c2", "aria_ensemble": "#000000",
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
    offsets = {
        "mlp": (1.15, 0.0, "left"),
        "gcn": (1.0, -1.0, "center"),
        "graphsage": (0.55, 0.9, "right"),
        "gat": (1.0, -1.1, "center"),
        "gps": (1.0, -1.2, "center"),
        "aria": (1.15, -0.3, "left"),
        "aria_phg": (1.15, 0.6, "left"),
        "aria_moe": (1.15, 0.9, "left"),
        "aria_ensemble": (0.85, 1.6, "right"),
    }
    for m in MODELS:
        t = stats["mean_time_sec"][m]
        a = stats["overall_mean_acc"][m]
        marker = "*" if m == "aria_ensemble" else "o"
        size = 170 if m == "aria_ensemble" else 55
        ax.scatter(t, a, color=COLORS[m], marker=marker, s=size, zorder=3,
                   edgecolor="black" if m == "aria_ensemble" else "none", linewidth=0.8)
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


def main():
    with open(os.path.join(BASE, "results", "paper_stats_9model.json")) as f:
        stats = json.load(f)
    fig_efficiency(stats)
    fig_summary_avg_rank(stats)


if __name__ == "__main__":
    main()
