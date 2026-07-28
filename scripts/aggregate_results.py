#!/usr/bin/env python3
"""Combines results/baseline_search.json + results/aria_search.json +
results/ensemble_eval.json into one final markdown table (mean+-std test
accuracy, bold = best per dataset column), mirroring the style of
RESULTS.md / NEW_RESULTS.md / NEW_RESULTS2.md from the earlier stages of
this project.

Usage:
    python scripts/aggregate_results.py --out results/FINAL_RESULTS.md
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import ALL_DATASETS
from src.models.aria import DISPLAY_NAMES

MODEL_ORDER = ["mlp", "gcn", "graphsage", "gat", "gps",
               "aria", "aria_ph", "aria_phg", "aria_moe", "aria_ensemble"]


def _load(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def display_name(model):
    return DISPLAY_NAMES.get(model, model.upper())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_results", default="results/baseline_search.json")
    ap.add_argument("--aria_results", default="results/aria_search.json")
    ap.add_argument("--ensemble_results", default="results/ensemble_eval.json")
    ap.add_argument("--out", default="results/FINAL_RESULTS.md")
    args = ap.parse_args()

    baseline = _load(args.baseline_results)
    aria = _load(args.aria_results)
    ensemble = _load(args.ensemble_results)

    def get(dataset, model):
        for source in (baseline, aria, ensemble):
            if dataset in source and model in source[dataset]:
                return source[dataset][model]
        return None

    header = "| Model | " + " | ".join(d for d in ALL_DATASETS) + " |"
    sep = "|---|" + "---|" * len(ALL_DATASETS)
    lines = [header, sep]

    # find per-column best mean for bolding
    best_per_dataset = {}
    for ds in ALL_DATASETS:
        best_mean, best_model = -1, None
        for model in MODEL_ORDER:
            entry = get(ds, model)
            if entry and entry["mean"] > best_mean:
                best_mean, best_model = entry["mean"], model
        best_per_dataset[ds] = best_model

    for model in MODEL_ORDER:
        row = [display_name(model)]
        for ds in ALL_DATASETS:
            entry = get(ds, model)
            if entry is None:
                row.append("-")
            else:
                cell = f"{entry['mean']:.2f}±{entry['std']:.2f}"
                if best_per_dataset[ds] == model:
                    cell = f"**{cell}**"
                row.append(cell)
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("Bold = best mean per dataset column, across every model with results present. "
                 "'-' = no results yet for that (dataset, model) pair.")

    # --- Training time table (only if at least one entry has timing data --
    # older results files predating train_time_* logging simply won't have it) ---
    has_any_time = any(
        get(ds, m) and "train_time_mean_sec" in get(ds, m)
        for ds in ALL_DATASETS for m in MODEL_ORDER
    )
    if has_any_time:
        lines.append("")
        lines.append("## Training time (seconds, mean +- std per final-phase run, wall-clock)")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for model in MODEL_ORDER:
            row = [display_name(model)]
            for ds in ALL_DATASETS:
                entry = get(ds, model)
                if entry is None or "train_time_mean_sec" not in entry:
                    row.append("-")
                else:
                    row.append(f"{entry['train_time_mean_sec']:.1f}±{entry['train_time_std_sec']:.1f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("Wall-clock seconds per training run (one early-stopped run to convergence), "
                     "measured during the final phase only (not the hyperparameter search). For "
                     "ARIA-Ensemble this is the combined time to train all 3 members sequentially.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
