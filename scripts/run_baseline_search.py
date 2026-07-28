#!/usr/bin/env python3
"""Resumable hyperparameter search for the baseline models (MLP, GCN,
GraphSAGE, GAT, GPS) across all 13 datasets.

For each (dataset, model): trains every config in search_space.py's grid on
a small number of splits (DATASET_TRAIN_CONFIG[dataset]["n_search_splits"]),
picks the config with the best mean validation accuracy, then re-trains
that single winning config on the full n_final_splits (10) splits for the
number that actually goes in a results table.

Time-boxed and resumable (mirrors runner.py/runner_new.py/runner_v2.py's
design from earlier in this project): state is checkpointed to disk after
every single training run, so this can be safely interrupted and re-invoked
with the same arguments.

*** THIS SCRIPT IS NOT INTENDED TO BE RUN YET. ***
Run `python run_baseline_search.py --dry_run` first to see exactly how many
training runs the current grid implies (also see `python -m
src.search_space` from the repo root) and trim search_space.py's grids if
that number is larger than your compute budget.

Usage (once you're ready):
    python scripts/run_baseline_search.py --time_budget 300 \\
        --state_dir state/baseline_search --results_file results/baseline_search.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import ALL_DATASETS, load_dataset, num_available_splits
from src.search_space import BASELINE_SEARCH_SPACE, DATASET_TRAIN_CONFIG, enumerate_configs
from src.training import train_one_split
from src.device import get_device, device_description
from src.artifact_logger import artifact_path, save_artifact

MODELS = list(BASELINE_SEARCH_SPACE.keys())  # mlp, gcn, graphsage, gat, gps


def _state_path(state_dir, dataset, model):
    return os.path.join(state_dir, f"{dataset}__{model}.json")


def _load_state(path, configs):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "configs": configs,
        "search_results": {},   # {config_idx (str): [val_acc, ...]}
        "best_config_idx": None,
        "final_test_accs": None,
        "final_train_times": None,  # wall-clock seconds per final-phase split, parallel to final_test_accs
    }


def _save_state(path, state):
    with open(path, "w") as f:
        json.dump(state, f)


def run_search_phase(model, data, dataset, train_cfg, state, device, start_time, time_budget):
    n_search = train_cfg["n_search_splits"]
    for cfg_idx, cfg in enumerate(state["configs"]):
        key = str(cfg_idx)
        done = state["search_results"].get(key, [])
        while len(done) < n_search:
            if time.time() - start_time > time_budget:
                return False  # ran out of time, not finished with search
            split = len(done)
            full_cfg = {**cfg, "max_epochs": train_cfg["max_epochs"], "patience": train_cfg["patience"]}
            t0 = time.time()
            _, val_acc, _ = train_one_split(model, data, split, device, full_cfg, seed=split)
            dt = time.time() - t0
            done.append(val_acc)
            state["search_results"][key] = done
            print(f"SEARCH {dataset}/{model} cfg={cfg_idx}/{len(state['configs'])} "
                  f"split={split} val_acc={val_acc:.4f} time={dt:.1f}s", flush=True)
    return True  # every config has n_search splits' worth of results


def run_final_phase(model, data, dataset, train_cfg, state, device, start_time, time_budget,
                     save_artifacts=False, save_weights=False, artifacts_dir="artifacts"):
    if state["best_config_idx"] is None:
        means = {int(k): np.mean(v) for k, v in state["search_results"].items()}
        state["best_config_idx"] = max(means, key=means.get)
        print(f"BEST_CONFIG {dataset}/{model} -> config #{state['best_config_idx']} "
              f"= {state['configs'][state['best_config_idx']]} "
              f"(mean val_acc={means[state['best_config_idx']]:.4f})", flush=True)

    best_cfg = state["configs"][state["best_config_idx"]]
    n_final = train_cfg["n_final_splits"]
    accs = state["final_test_accs"] or []
    times = state.get("final_train_times") or []
    while len(accs) < n_final:
        if time.time() - start_time > time_budget:
            state["final_test_accs"] = accs
            state["final_train_times"] = times
            return False
        split = len(accs)
        full_cfg = {**best_cfg, "max_epochs": train_cfg["max_epochs"], "patience": train_cfg["patience"]}
        t0 = time.time()
        test_acc, _, artifacts = train_one_split(
            model, data, split, device, full_cfg, seed=split,
            capture_artifacts=save_artifacts, save_weights=save_weights)
        dt = time.time() - t0
        if save_artifacts:
            path = artifact_path(artifacts_dir, dataset, model, split)
            save_artifact(path, artifacts)
        accs.append(test_acc)
        times.append(dt)
        state["final_test_accs"] = accs
        state["final_train_times"] = times
        print(f"FINAL {dataset}/{model} split={split} test_acc={test_acc:.4f} "
              f"time={dt:.1f}s done={len(accs)}/{n_final}", flush=True)
    return True


def save_summary(results_file, dataset, model, state):
    results = {}
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
    accs_pct = (np.array(state["final_test_accs"]) * 100).tolist()
    entry = {
        "mean": float(np.mean(accs_pct)),
        "std": float(np.std(accs_pct)),
        "runs": accs_pct,
        "best_config": state["configs"][state["best_config_idx"]],
    }
    times = state.get("final_train_times")
    if times:
        entry["train_time_mean_sec"] = float(np.mean(times))
        entry["train_time_std_sec"] = float(np.std(times))
        entry["train_time_runs_sec"] = [float(t) for t in times]
    results.setdefault(dataset, {})[model] = entry
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time_budget", type=float, default=300.0,
                     help="Wall-clock seconds this invocation is allowed to run before checkpointing and exiting.")
    ap.add_argument("--state_dir", default="state/baseline_search")
    ap.add_argument("--results_file", default="results/baseline_search.json")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--datasets", nargs="*", default=None, help="Subset of ALL_DATASETS; default = all 13.")
    ap.add_argument("--models", nargs="*", default=None, help=f"Subset of {MODELS}; default = all.")
    ap.add_argument("--dry_run", action="store_true",
                     help="Print grid sizes / total implied run counts and exit -- trains nothing.")
    ap.add_argument("--save_artifacts", action="store_true",
                     help="During the FINAL phase only, save belief vectors/probs/curves per "
                          "(dataset, model, split) to --artifacts_dir. Not used during search.")
    ap.add_argument("--save_weights", action="store_true",
                     help="Also include full model state_dict in saved artifacts (bulkier). "
                          "Ignored unless --save_artifacts is set.")
    ap.add_argument("--artifacts_dir", default="artifacts")
    args = ap.parse_args()

    if args.dry_run:
        from src.search_space import report_total_combos
        report_total_combos()
        return

    os.makedirs(args.state_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    device = get_device(args.device)
    print(f"Device: {device_description(device)}", flush=True)

    datasets = args.datasets or ALL_DATASETS
    models = args.models or MODELS
    start_time = time.time()

    for dataset in datasets:
        train_cfg = DATASET_TRAIN_CONFIG[dataset]
        data = None
        for model in models:
            configs = enumerate_configs(BASELINE_SEARCH_SPACE[model])
            path = _state_path(args.state_dir, dataset, model)
            state = _load_state(path, configs)

            if state["final_test_accs"] is not None and len(state["final_test_accs"]) >= train_cfg["n_final_splits"]:
                continue  # fully done already

            if data is None:
                data = (load_dataset(dataset, root=args.data_root) if args.data_root
                         else load_dataset(dataset))

            if time.time() - start_time > args.time_budget:
                print(f"TIME_UP after {time.time()-start_time:.1f}s", flush=True)
                return

            search_done = run_search_phase(model, data, dataset, train_cfg, state, device,
                                            start_time, args.time_budget)
            _save_state(path, state)
            if not search_done:
                print(f"TIME_UP after {time.time()-start_time:.1f}s", flush=True)
                return

            final_done = run_final_phase(model, data, dataset, train_cfg, state, device,
                                          start_time, args.time_budget,
                                          save_artifacts=args.save_artifacts,
                                          save_weights=args.save_weights,
                                          artifacts_dir=args.artifacts_dir)
            _save_state(path, state)
            if not final_done:
                print(f"TIME_UP after {time.time()-start_time:.1f}s", flush=True)
                return

            save_summary(args.results_file, dataset, model, state)
            print(f"COMBO_DONE {dataset}/{model} mean={np.mean(state['final_test_accs'])*100:.2f} "
                  f"std={np.std(state['final_test_accs'])*100:.2f}", flush=True)

    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
