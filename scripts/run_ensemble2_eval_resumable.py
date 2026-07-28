#!/usr/bin/env python3
"""Epoch-resumable ARIA-Ensemble2 (bare ARIA + ARIA-PHG + ARIA-MoE) sweep.

Same target as run_ensemble2_eval.py, but checkpoints at the epoch level
(via src/ensemble2_resumable.py) so it can be re-invoked many times with a
short wall-clock budget each time and still make guaranteed forward
progress -- needed because some (dataset, member) configs run close to
their full max_epochs budget before early stopping triggers, which can
exceed a single short process invocation.

Idempotent / resumable: safe to call over and over with the same
arguments; it always resumes from whatever is on disk under --state_dir.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import ALL_DATASETS, load_dataset
from src.search_space import DATASET_TRAIN_CONFIG
from src.ensemble2_resumable import MEMBER_TRAIN_FN
from src.device import get_device, device_description

MEMBERS = ["aria", "aria_phg", "aria_moe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time_budget", type=float, default=38.0,
                     help="Soft wall-clock budget for this invocation (seconds).")
    ap.add_argument("--state_dir", default="state/ensemble2_eval_r")
    ap.add_argument("--results_file", default="results/ensemble2_eval.json")
    ap.add_argument("--aria_results", default="results/aria_search.json")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--n_splits_override", type=int, default=None,
                     help="If set, cap n_final_splits at this value for every dataset "
                          "in --datasets this invocation (compute-budget escape hatch; "
                          "must be footnoted in the paper wherever used).")
    args = ap.parse_args()

    ckpt_dir = os.path.join(args.state_dir, "ckpt")
    probs_dir = os.path.join(args.state_dir, "probs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(probs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)

    device = get_device(args.device)
    print(f"Device: {device_description(device)}", flush=True)

    with open(args.aria_results) as f:
        aria_results = json.load(f)

    datasets = args.datasets or ALL_DATASETS
    deadline = time.time() + args.time_budget

    for dataset in datasets:
        if dataset not in aria_results:
            print(f"SKIP {dataset}: not in {args.aria_results}", flush=True)
            continue
        missing = [m for m in MEMBERS if m not in aria_results[dataset]]
        if missing:
            print(f"SKIP {dataset}: missing {missing}", flush=True)
            continue

        configs = {}
        for m in MEMBERS:
            cfg = dict(aria_results[dataset][m]["best_config"])
            train_cfg = DATASET_TRAIN_CONFIG[dataset]
            cfg["max_epochs"] = train_cfg["max_epochs"]
            cfg["patience"] = train_cfg["patience"]
            configs[m] = cfg
        n_final = DATASET_TRAIN_CONFIG[dataset]["n_final_splits"]
        if args.n_splits_override is not None:
            n_final = min(n_final, args.n_splits_override)

        progress_path = os.path.join(args.state_dir, f"{dataset}.json")
        progress = {"finished": []}
        if os.path.exists(progress_path):
            with open(progress_path) as f:
                progress = json.load(f)

        if len(progress["finished"]) >= n_final:
            continue

        data = load_dataset(dataset, root=args.data_root) if args.data_root else load_dataset(dataset)
        y = data.y.to(device)
        edge_index = data.edge_index.to(device)

        for split in range(len(progress["finished"]), n_final):
            if time.time() >= deadline:
                print("TIME_UP (between splits)", flush=True)
                return

            train_mask = data.train_mask[:, split].to(device)
            val_mask = data.val_mask[:, split].to(device)
            test_mask = data.test_mask[:, split].to(device)

            member_probs = {}
            all_members_done = True
            for member in MEMBERS:
                probs_path = os.path.join(probs_dir, f"{dataset}__s{split}__{member}.pt")
                if os.path.exists(probs_path):
                    saved = torch.load(probs_path, map_location="cpu", weights_only=False)
                    member_probs[member] = saved["probs"]
                    continue

                ckpt_path = os.path.join(ckpt_dir, f"{dataset}__s{split}__{member}.pt")
                if time.time() >= deadline:
                    print("TIME_UP (before member)", flush=True)
                    return
                fn = MEMBER_TRAIN_FN[member]
                if member == "aria_moe":
                    probs, val_acc, done = fn(data, split, device, configs[member],
                                               train_mask, val_mask, y, edge_index, split,
                                               ckpt_path, deadline, dataset_key=dataset)
                elif member == "aria_phg":
                    probs, val_acc, done = fn(data, split, device, configs[member],
                                               train_mask, val_mask, y, edge_index,
                                               ckpt_path, deadline, dataset_key=dataset)
                else:
                    probs, val_acc, done = fn(data, split, device, configs[member],
                                               train_mask, val_mask, y, edge_index,
                                               ckpt_path, deadline)
                if not done:
                    print(f"CHECKPOINTED {dataset} split={split} member={member} (mid-training)", flush=True)
                    all_members_done = False
                    break
                torch.save({"probs": probs, "val_acc": val_acc}, probs_path)
                member_probs[member] = probs
                print(f"MEMBER_DONE {dataset} split={split} member={member} val_acc={val_acc:.4f}", flush=True)

            if not all_members_done:
                return

            avg_probs = torch.stack([member_probs[m] for m in MEMBERS], dim=0).mean(0)
            test_acc = (avg_probs[test_mask].argmax(-1) == y[test_mask]).float().mean().item()
            progress["finished"].append({"split": split, "acc": test_acc})
            with open(progress_path, "w") as f:
                json.dump(progress, f)
            # NOTE: deliberately not deleting the now-unneeded probs_path files here --
            # this sandbox's mounted filesystem disallows unlink on some paths once
            # written. Leftover .pt files are harmless (probs existing on disk is only
            # ever used as a "this member is done" check keyed by dataset/split/member,
            # so stale files past a completed split are just inert leftovers).
            print(f"SPLIT_DONE {dataset}/aria_ensemble2 split={split} acc={test_acc:.4f} "
                  f"done={len(progress['finished'])}/{n_final}", flush=True)

        if len(progress["finished"]) >= n_final:
            results = {}
            if os.path.exists(args.results_file):
                with open(args.results_file) as f:
                    results = json.load(f)
            accs_pct = [x["acc"] * 100 for x in progress["finished"]]
            entry = {
                "mean": float(np.mean(accs_pct)),
                "std": float(np.std(accs_pct)),
                "runs": accs_pct,
                "member_configs": configs,
            }
            results.setdefault(dataset, {})["aria_ensemble2"] = entry
            with open(args.results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"COMBO_DONE {dataset}/aria_ensemble2 mean={np.mean(accs_pct):.2f} "
                  f"std={np.std(accs_pct):.2f}", flush=True)

    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
