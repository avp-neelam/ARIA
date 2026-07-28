#!/usr/bin/env python3
"""Trimmed variant of build_paper_stats.py, per advisor feedback (2026-07-23):
"5 variations are too much. One base (ARIA) + 3 variations are enough...
remove one ensemble and one PH model (PH or PHG)."

Keeps: ARIA (baseline), ARIA-PHG, ARIA-MoE, ARIA-Ensemble (still PH+PHG+MoE
under the hood -- ARIA-PH is retained only as an ensemble ingredient, not as
a standalone reported row). Drops: ARIA-PH as a standalone model,
ARIA-Ensemble2 entirely.

Reads the same underlying result files as build_paper_stats.py -- ARIA-PH's
raw runs are NOT read here at all (it never appears as its own column), only
the recipe survives, embedded inside aria_ensemble's existing eval.

Writes results/paper_stats_9model.json.
"""
import json
import os

import numpy as np
from scipy import stats

KEEP_DATASETS = ["cora", "citeseer", "pubmed", "texas", "wisconsin", "cornell",
                  "amazon-photo", "actor", "usa-airports", "brazil-airports",
                  "europe-airports"]

BASELINES = ["mlp", "gcn", "graphsage", "gat", "gps"]
ARIA_FAMILY = ["aria", "aria_phg", "aria_moe", "aria_ensemble"]
ALL_MODELS = BASELINES + ARIA_FAMILY

DISPLAY = {
    "mlp": "MLP", "gcn": "GCN", "graphsage": "GraphSAGE", "gat": "GAT", "gps": "GPS",
    "aria": "ARIA (baseline)", "aria_phg": "ARIA-PHG",
    "aria_moe": "ARIA-MoE", "aria_ensemble": "ARIA-Ensemble",
}

DATASET_DISPLAY = {
    "cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed", "texas": "Texas",
    "wisconsin": "Wisconsin", "cornell": "Cornell", "amazon-photo": "Amazon-Photo",
    "actor": "Actor", "usa-airports": "USA-Airports", "brazil-airports": "Brazil-Airports",
    "europe-airports": "Europe-Airports",
}


def load_all(base_dir):
    with open(os.path.join(base_dir, "results/baseline_search.json")) as f:
        baseline = json.load(f)
    with open(os.path.join(base_dir, "results/aria_search.json")) as f:
        aria = json.load(f)
    with open(os.path.join(base_dir, "results/ensemble_eval.json")) as f:
        ens1 = json.load(f)

    runs = {}
    times = {}
    for ds in KEEP_DATASETS:
        runs[ds] = {}
        times[ds] = {}
        for m in BASELINES:
            runs[ds][m] = baseline[ds][m]["runs"]
            times[ds][m] = baseline[ds][m].get("train_time_mean_sec")
        for m in ["aria", "aria_phg", "aria_moe"]:
            runs[ds][m] = aria[ds][m]["runs"]
            times[ds][m] = aria[ds][m].get("train_time_mean_sec")
        runs[ds]["aria_ensemble"] = ens1[ds]["aria_ensemble"]["runs"]
        times[ds]["aria_ensemble"] = ens1[ds]["aria_ensemble"].get("train_time_mean_sec")
    return runs, times


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs, times = load_all(base_dir)

    mean_acc = {ds: {m: float(np.mean(runs[ds][m])) for m in ALL_MODELS} for ds in KEEP_DATASETS}
    std_acc = {ds: {m: float(np.std(runs[ds][m])) for m in ALL_MODELS} for ds in KEEP_DATASETS}
    n_runs = {ds: {m: len(runs[ds][m]) for m in ALL_MODELS} for ds in KEEP_DATASETS}

    best_model = {}
    for ds in KEEP_DATASETS:
        best_model[ds] = max(ALL_MODELS, key=lambda m: mean_acc[ds][m])

    overall_mean_acc = {m: float(np.mean([mean_acc[ds][m] for ds in KEEP_DATASETS])) for m in ALL_MODELS}

    ranks = {ds: {} for ds in KEEP_DATASETS}
    for ds in KEEP_DATASETS:
        ordered = sorted(ALL_MODELS, key=lambda m: -mean_acc[ds][m])
        for i, m in enumerate(ordered):
            ranks[ds][m] = i + 1
    mean_rank = {m: float(np.mean([ranks[ds][m] for ds in KEEP_DATASETS])) for m in ALL_MODELS}
    worst_count = {m: sum(1 for ds in KEEP_DATASETS if ranks[ds][m] == len(ALL_MODELS)) for m in ALL_MODELS}
    never_worst = {m: worst_count[m] == 0 for m in ALL_MODELS}

    def paired_test(a_runs, b_runs):
        n = min(len(a_runs), len(b_runs))
        a = np.array(a_runs[:n]); b = np.array(b_runs[:n])
        diff = a - b
        if np.allclose(diff, 0):
            return 1.0, 0.0
        t, p = stats.ttest_rel(a, b)
        return float(p), float(t)

    significance = {}
    for ds in KEEP_DATASETS:
        best_base = max(BASELINES, key=lambda m: mean_acc[ds][m])
        significance[ds] = {"best_baseline": best_base, "best_baseline_acc": mean_acc[ds][best_base]}
        for fam in ARIA_FAMILY:
            p_vs_best, t_vs_best = paired_test(runs[ds][fam], runs[ds][best_base])
            p_vs_gps, t_vs_gps = paired_test(runs[ds][fam], runs[ds]["gps"])
            significance[ds][fam] = {
                "p_vs_best_baseline": p_vs_best, "t_vs_best_baseline": t_vs_best,
                "p_vs_gps": p_vs_gps, "t_vs_gps": t_vs_gps,
                "margin_vs_best_baseline": mean_acc[ds][fam] - mean_acc[ds][best_base],
                "margin_vs_gps": mean_acc[ds][fam] - mean_acc[ds]["gps"],
            }
        # best ARIA-family model on this dataset (for the significance table)
        best_fam = max(ARIA_FAMILY, key=lambda m: mean_acc[ds][m])
        significance[ds]["best_family_model"] = best_fam
        significance[ds]["best_family_acc"] = mean_acc[ds][best_fam]

    mean_time = {m: float(np.mean([times[ds][m] for ds in KEEP_DATASETS if times[ds][m] is not None]))
                 for m in ALL_MODELS}

    out = {
        "datasets": KEEP_DATASETS,
        "models": ALL_MODELS,
        "mean_acc": mean_acc, "std_acc": std_acc, "n_runs": n_runs,
        "best_model_per_dataset": best_model,
        "overall_mean_acc": overall_mean_acc,
        "ranks": ranks, "mean_rank": mean_rank, "never_worst": never_worst, "worst_count": worst_count,
        "significance": significance,
        "mean_time_sec": mean_time,
        "time_by_dataset": times,
    }
    out_path = os.path.join(base_dir, "results/paper_stats_9model.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("=== Overall mean accuracy across 11 datasets (sorted) ===")
    for m in sorted(ALL_MODELS, key=lambda m: -overall_mean_acc[m]):
        print(f"  {DISPLAY[m]:20s} {overall_mean_acc[m]:.2f}")
    print()
    print("=== Mean rank across 11 datasets, 9 models (sorted, lower=better) ===")
    for m in sorted(ALL_MODELS, key=lambda m: mean_rank[m]):
        print(f"  {DISPLAY[m]:20s} rank={mean_rank[m]:.2f}  never_worst={never_worst[m]}  worst_count={worst_count[m]}")
    print()
    print("=== Per-dataset winner (9-model set) ===")
    for ds in KEEP_DATASETS:
        print(f"  {DATASET_DISPLAY[ds]:18s} {DISPLAY[best_model[ds]]:20s} {mean_acc[ds][best_model[ds]]:.2f}   best_family={DISPLAY[significance[ds]['best_family_model']]}")
    print()
    print("=== Significant results (p<0.05) vs best-baseline-per-dataset ===")
    for ds in KEEP_DATASETS:
        for fam in ARIA_FAMILY:
            p = significance[ds][fam]["p_vs_best_baseline"]
            if p < 0.05:
                print(f"  {DATASET_DISPLAY[ds]:18s} {DISPLAY[fam]:20s} p={p:.4f} margin={significance[ds][fam]['margin_vs_best_baseline']:+.2f}")
    print()
    print("=== Significant results (p<0.05) vs GPS specifically ===")
    for ds in KEEP_DATASETS:
        for fam in ARIA_FAMILY:
            p = significance[ds][fam]["p_vs_gps"]
            if p < 0.05:
                print(f"  {DATASET_DISPLAY[ds]:18s} {DISPLAY[fam]:20s} p={p:.4f} margin={significance[ds][fam]['margin_vs_gps']:+.2f}")
    print()
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
