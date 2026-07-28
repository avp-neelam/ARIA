#!/usr/bin/env python3
"""Pre-flight smoke test -- run this BEFORE launching the full sweep
(run_baseline_search.py / run_aria_search.py / run_ensemble_eval.py, ~98,000
runs combined) to catch dataset-download issues, shape mismatches, or API
incompatibilities early, in minutes instead of hours into the real run.

For every dataset in ALL_DATASETS (13) x every model (5 baselines + 4 ARIA
variants + aria_ensemble = 10, 130 combos total by default), this:
  1. Loads the dataset once (catches download/parsing errors, wrong mask
     shapes, etc. -- shared across all models for that dataset).
  2. Runs a tiny (3-epoch, single-split) training pass with a fixed,
     minimal hyperparameter config for every model (catches shape
     mismatches, missing attributes, device-placement bugs, GPSConv/RWPE
     issues, GaussianMixture clustering failures, etc.).
  3. Prints a PASS/FAIL summary table plus a list of exactly which combos
     failed and why.

Nothing here produces meaningful accuracy numbers -- 3 epochs is only
enough to prove a model trains and evaluates end-to-end without crashing.
Exits with status 1 if anything failed (so it's safe to use as a gate in a
shell script, e.g. `python scripts/smoke_test.py && python
scripts/run_baseline_search.py ...`).

Usage:
    python scripts/smoke_test.py --device auto
    python scripts/smoke_test.py --datasets cora citeseer --models gcn aria_moe
    SMOKE_TEST_VERBOSE=1 python scripts/smoke_test.py   # full tracebacks on failure
"""
import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import ALL_DATASETS, load_dataset
from src.training import train_one_split
from src.ensemble import train_ensemble_one_split
from src.device import get_device, device_description

ALL_MODELS = ["mlp", "gcn", "graphsage", "gat", "gps",
              "aria", "aria_ph", "aria_phg", "aria_moe", "aria_ensemble"]

# Deliberately tiny/fixed -- this is not a hyperparameter search, just enough
# of every knob (rank, heads, pe_k, aux_lambda, n_experts, num_layers) to
# exercise every code path each model touches.
TINY_CFG = dict(
    hidden_dim=16, lr=1e-2, weight_decay=5e-4, dropout=0.5, num_layers=2,
    heads=4, pe_dim=8, rank=8, aux_lambda=0.5, pe_k=8, n_experts=4,
    max_epochs=3, patience=5,
)


def run_one(model, data, device):
    """Runs a tiny training pass for one model on split 0. Raises on failure."""
    if model == "aria_ensemble":
        train_ensemble_one_split(data, 0, device, TINY_CFG, TINY_CFG, TINY_CFG, seed=0)
    else:
        train_one_split(model, data, 0, device, TINY_CFG, seed=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--datasets", nargs="*", default=None,
                     help=f"Subset of {ALL_DATASETS}; default = all 13.")
    ap.add_argument("--models", nargs="*", default=None,
                     help=f"Subset of {ALL_MODELS}; default = all 10.")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"Device: {device_description(device)}\n", flush=True)

    datasets = args.datasets or ALL_DATASETS
    models = args.models or ALL_MODELS
    verbose = bool(os.environ.get("SMOKE_TEST_VERBOSE"))

    results = {}  # (dataset, model) -> (status, detail)
    for dataset in datasets:
        t0 = time.time()
        try:
            data = (load_dataset(dataset, root=args.data_root) if args.data_root
                     else load_dataset(dataset))
            n_splits = data.train_mask.size(1)
            print(f"[{dataset}] loaded: x={tuple(data.x.shape)} classes={data.num_classes} "
                  f"splits={n_splits} ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[{dataset}] LOAD FAILED: {type(e).__name__}: {e}", flush=True)
            if verbose:
                traceback.print_exc()
            for model in models:
                results[(dataset, model)] = ("FAIL_LOAD", f"{type(e).__name__}: {e}")
            continue

        for model in models:
            t1 = time.time()
            try:
                run_one(model, data, device)
                dt = time.time() - t1
                results[(dataset, model)] = ("PASS", f"{dt:.1f}s")
                print(f"  {model:14s} PASS ({dt:.1f}s)", flush=True)
            except Exception as e:
                results[(dataset, model)] = ("FAIL_TRAIN", f"{type(e).__name__}: {e}")
                print(f"  {model:14s} FAIL: {type(e).__name__}: {e}", flush=True)
                if verbose:
                    traceback.print_exc()

    # --- Summary table ---
    print("\n" + "=" * 100)
    print("SMOKE TEST SUMMARY")
    print("=" * 100)
    col_w = 11
    header = f"{'dataset':18s}" + "".join(f"{m[:col_w]:>{col_w+1}s}" for m in models)
    print(header)
    symbol_map = {"PASS": "ok", "FAIL_LOAD": "FAIL(load)", "FAIL_TRAIN": "FAIL(train)", "SKIP": "-"}
    n_pass, n_fail = 0, 0
    for dataset in datasets:
        row = f"{dataset:18s}"
        for model in models:
            status, _ = results.get((dataset, model), ("SKIP", ""))
            row += f"{symbol_map[status]:>{col_w+1}s}"
            if status == "PASS":
                n_pass += 1
            elif status.startswith("FAIL"):
                n_fail += 1
        print(row)

    print("\nFailures:")
    any_fail = False
    for dataset in datasets:
        for model in models:
            status, detail = results.get((dataset, model), ("SKIP", ""))
            if status.startswith("FAIL"):
                any_fail = True
                print(f"  {dataset}/{model}: {status} -- {detail}")
    if not any_fail:
        print("  (none)")

    total = len(datasets) * len(models)
    print(f"\n{n_pass} passed, {n_fail} failed, out of {total} combos.")
    if n_fail > 0:
        print("Fix the failures above (set SMOKE_TEST_VERBOSE=1 for full tracebacks) "
              "before launching the full sweep.")
        sys.exit(1)
    print("All combos OK -- safe to launch the full sweep "
          "(run_baseline_search.py, run_aria_search.py, run_ensemble_eval.py).")


if __name__ == "__main__":
    main()
