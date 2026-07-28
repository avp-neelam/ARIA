#!/usr/bin/env python3
"""Evaluates ARIA-Ensemble: reads the winning hyperparameter configs found
by run_aria_search.py for aria_ph, aria_phg, and aria_moe (per dataset),
and evaluates the 3-member logit-averaging ensemble on the full 10 splits.

Must be run AFTER run_aria_search.py has produced final results for
aria_ph/aria_phg/aria_moe on whatever datasets you want an ensemble number
for -- this script has nothing to search itself, it only assembles.

*** THIS SCRIPT IS NOT INTENDED TO BE RUN YET (nothing is, in this repo). ***

Usage (once run_aria_search.py has produced results/aria_search.json):
    python scripts/run_ensemble_eval.py --time_budget 300 \\
        --aria_results results/aria_search.json \\
        --results_file results/ensemble_eval.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import ALL_DATASETS, load_dataset
from src.search_space import DATASET_TRAIN_CONFIG
from src.ensemble import train_ensemble_one_split
from src.device import get_device, device_description
from src.artifact_logger import artifact_path, save_artifact

REQUIRED_MEMBERS = ["aria_ph", "aria_phg", "aria_moe"]


def _state_path(state_dir, dataset):
    return os.path.join(state_dir, f"{dataset}__aria_ensemble.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time_budget", type=float, default=300.0)
    ap.add_argument("--state_dir", default="state/ensemble_eval")
    ap.add_argument("--results_file", default="results/ensemble_eval.json")
    ap.add_argument("--aria_results", default="results/aria_search.json",
                     help="Output of run_aria_search.py -- must contain "
                          "'best_config' for aria_ph/aria_phg/aria_moe on "
                          "every dataset you list in --datasets.")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--save_artifacts", action="store_true",
                     help="Save belief vectors/probs/curves for each ensemble member plus the "
                          "averaged prediction, per (dataset, split), to --artifacts_dir.")
    ap.add_argument("--save_weights", action="store_true",
                     help="Also include full model state_dicts for each member (bulkier). "
                          "Ignored unless --save_artifacts is set.")
    ap.add_argument("--artifacts_dir", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.state_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    device = get_device(args.device)
    print(f"Device: {device_description(device)}", flush=True)

    with open(args.aria_results) as f:
        aria_results = json.load(f)

    datasets = args.datasets or ALL_DATASETS
    start_time = time.time()

    for dataset in datasets:
        if dataset not in aria_results:
            print(f"SKIP {dataset}: not present in {args.aria_results} "
                  f"(run run_aria_search.py for it first)", flush=True)
            continue
        missing = [m for m in REQUIRED_MEMBERS if m not in aria_results[dataset]]
        if missing:
            print(f"SKIP {dataset}: missing member results {missing} in "
                  f"{args.aria_results}", flush=True)
            continue

        ph_config = aria_results[dataset]["aria_ph"]["best_config"]
        phg_config = aria_results[dataset]["aria_phg"]["best_config"]
        moe_config = aria_results[dataset]["aria_moe"]["best_config"]

        train_cfg = DATASET_TRAIN_CONFIG[dataset]
        for cfg in (ph_config, phg_config, moe_config):
            cfg["max_epochs"] = train_cfg["max_epochs"]
            cfg["patience"] = train_cfg["patience"]

        path = _state_path(args.state_dir, dataset)
        state = {"finished_accs": [], "finished_times": []}
        if os.path.exists(path):
            with open(path) as f:
                state = json.load(f)
            state.setdefault("finished_times", [])

        if len(state["finished_accs"]) >= train_cfg["n_final_splits"]:
            continue

        data = (load_dataset(dataset, root=args.data_root) if args.data_root
                 else load_dataset(dataset))

        while len(state["finished_accs"]) < train_cfg["n_final_splits"]:
            if time.time() - start_time > args.time_budget:
                print(f"TIME_UP after {time.time()-start_time:.1f}s", flush=True)
                return
            split = len(state["finished_accs"])
            t0 = time.time()
            test_acc, _, artifacts = train_ensemble_one_split(
                data, split, device, ph_config, phg_config, moe_config, seed=split,
                capture_artifacts=args.save_artifacts, save_weights=args.save_weights)
            dt = time.time() - t0
            if args.save_artifacts:
                path_art = artifact_path(args.artifacts_dir, dataset, "aria_ensemble", split)
                save_artifact(path_art, artifacts)
            state["finished_accs"].append(test_acc)
            state["finished_times"].append(dt)
            with open(path, "w") as f:
                json.dump(state, f)
            print(f"SPLIT_DONE {dataset}/aria_ensemble split={split} "
                  f"acc={test_acc:.4f} time={dt:.1f}s done={len(state['finished_accs'])}/"
                  f"{train_cfg['n_final_splits']}", flush=True)

        results = {}
        if os.path.exists(args.results_file):
            with open(args.results_file) as f:
                results = json.load(f)
        accs_pct = (np.array(state["finished_accs"]) * 100).tolist()
        entry = {
            "mean": float(np.mean(accs_pct)),
            "std": float(np.std(accs_pct)),
            "runs": accs_pct,
            "member_configs": {"aria_ph": ph_config, "aria_phg": phg_config, "aria_moe": moe_config},
        }
        times = state.get("finished_times")
        if times:
            entry["train_time_mean_sec"] = float(np.mean(times))
            entry["train_time_std_sec"] = float(np.std(times))
            entry["train_time_runs_sec"] = [float(t) for t in times]
        results.setdefault(dataset, {})["aria_ensemble"] = entry
        with open(args.results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"COMBO_DONE {dataset}/aria_ensemble mean={np.mean(accs_pct):.2f} "
              f"std={np.std(accs_pct):.2f}", flush=True)

    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
