"""Hyperparameter search space (small grid: 2-3 values per hyperparameter)
and per-dataset training protocol (epoch budget, patience, split counts --
these are fixed, not searched, matching the earlier stages of this
project's convention of a fixed training protocol per dataset).

Grid sizes are intentionally visible here (not hidden in a runner script)
since the combinatorics matter: some of these, especially the ARIA
variants with 4 rank values, multiply out to several hundred configs per
(dataset, model). See scripts/run_*_search.py's --dry_run flag to print
exact combo counts before running anything for real.
"""
import itertools


# ---------------------------------------------------------------------------
# Per-dataset training protocol (NOT searched -- fixed budget/patience/splits)
# ---------------------------------------------------------------------------
# n_search_splits: how many splits to average over when comparing configs
# during the search phase (kept small since the grid itself is what's
# expensive -- accuracy differences between configs are used to pick a
# winner, not to report final numbers).
# n_final_splits: how many splits the WINNING config is re-evaluated on for
# the number that actually goes in a results table (10 everywhere, matching
# every prior stage of this project, except Roman-Empire which uses
# whatever split count datasets.num_available_splits() reports).

DATASET_TRAIN_CONFIG = {
    "cora":             dict(max_epochs=300, patience=50,  n_search_splits=3, n_final_splits=10),
    "citeseer":         dict(max_epochs=300, patience=50,  n_search_splits=3, n_final_splits=10),
    "pubmed":           dict(max_epochs=300, patience=50,  n_search_splits=3, n_final_splits=10),
    "amazon-photo":     dict(max_epochs=300, patience=50,  n_search_splits=3, n_final_splits=10),
    "amazon-computers": dict(max_epochs=300, patience=50,  n_search_splits=3, n_final_splits=10),
    "texas":            dict(max_epochs=500, patience=100, n_search_splits=3, n_final_splits=10),
    "wisconsin":        dict(max_epochs=500, patience=100, n_search_splits=3, n_final_splits=10),
    "cornell":          dict(max_epochs=500, patience=100, n_search_splits=3, n_final_splits=10),
    "actor":            dict(max_epochs=300, patience=50,  n_search_splits=3, n_final_splits=10),
    "roman-empire":     dict(max_epochs=300, patience=50,  n_search_splits=3, n_final_splits=10),
    "usa-airports":     dict(max_epochs=300, patience=50,  n_search_splits=3, n_final_splits=10),
    "brazil-airports":  dict(max_epochs=500, patience=100, n_search_splits=3, n_final_splits=10),
    "europe-airports":  dict(max_epochs=500, patience=100, n_search_splits=3, n_final_splits=10),
}


# ---------------------------------------------------------------------------
# Baseline hyperparameter grids
# ---------------------------------------------------------------------------

_BASE_GRID = dict(
    hidden_dim=[32, 64],
    lr=[1e-2, 5e-3, 1e-3],
    weight_decay=[5e-4, 5e-5],
    dropout=[0.3, 0.5],
    num_layers=[2, 3],
)

BASELINE_SEARCH_SPACE = {
    "mlp": dict(_BASE_GRID),
    "gcn": dict(_BASE_GRID),
    "graphsage": dict(_BASE_GRID),
    "gat": {**_BASE_GRID, "heads": [4, 8]},
    "gps": {
        "hidden_dim": [32, 64],
        "lr": [5e-3, 1e-3],
        "weight_decay": [5e-4, 5e-5],
        "dropout": [0.3, 0.5],
        "num_layers": [2, 3],
        "heads": [4, 8],
        "pe_dim": [8, 16],
    },
}


# ---------------------------------------------------------------------------
# ARIA hyperparameter grids (rank values match what was actually tested in
# the earlier stages of this project: 8/16 as the original defaults, 32 from
# the follow-up rank sweep that showed it helping the Airports datasets for
# ARIA-Ensemble. 64 was also swept and didn't improve further over 32, so
# it's excluded here to keep the grid smaller.)
# ---------------------------------------------------------------------------

_ARIA_BASE_GRID = dict(
    hidden_dim=[32, 64],
    lr=[1e-2, 5e-3, 1e-3],
    weight_decay=[5e-4, 5e-5],
    dropout=[0.3, 0.5],
    num_layers=[2, 3],
    rank=[8, 16, 32],
)

ARIA_SEARCH_SPACE = {
    "aria": dict(_ARIA_BASE_GRID),
    "aria_ph": {**_ARIA_BASE_GRID, "aux_lambda": [0.5, 1.0], "pe_k": [8, 16]},
    "aria_phg": {**_ARIA_BASE_GRID, "aux_lambda": [0.5, 1.0], "pe_k": [8, 16]},
    "aria_moe": {**_ARIA_BASE_GRID, "n_experts": [4, 8]},
}
# ARIA-Ensemble has no independent grid: it's assembled from whichever
# configs win the aria_ph / aria_phg / aria_moe searches (see
# scripts/run_ensemble_eval.py).


def enumerate_configs(grid: dict):
    """Cartesian product of a {name: [values]} grid -> list of {name: value}
    dicts, in a stable (deterministic) order."""
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*[grid[k] for k in keys]):
        combos.append(dict(zip(keys, values)))
    return combos


def grid_size(grid: dict) -> int:
    n = 1
    for values in grid.values():
        n *= len(values)
    return n


def report_total_combos():
    """Prints how many (config) combinations each model's grid produces,
    and the total number of individual training runs implied once you
    multiply by 13 datasets x n_search_splits (the search phase) plus 13 x
    n_final_splits (the final confirmation of the winning config).
    Intended for `python -m src.search_space` / the --dry_run flags on the
    search scripts -- NOT to launch anything."""
    from .datasets import ALL_DATASETS

    print(f"{len(ALL_DATASETS)} datasets: {ALL_DATASETS}\n")

    grand_total_runs = 0
    for label, space in [("BASELINES", BASELINE_SEARCH_SPACE), ("ARIA", ARIA_SEARCH_SPACE)]:
        print(f"--- {label} ---")
        for model, grid in space.items():
            n_configs = grid_size(grid)
            # worst case: search-phase splits for every config, every dataset,
            # plus a final full re-run of just the winning config per dataset.
            search_runs = n_configs * 3 * len(ALL_DATASETS)  # 3 ~ typical n_search_splits
            final_runs = 10 * len(ALL_DATASETS)
            total = search_runs + final_runs
            grand_total_runs += total
            print(f"  {model:12s} grid={n_configs:5d} configs   "
                  f"~search_runs={search_runs:7d}   final_runs={final_runs:4d}   "
                  f"~total={total:7d}")
        print()
    print(f"GRAND TOTAL (approx, across all datasets/models): ~{grand_total_runs:,} individual "
          f"train-to-convergence runs. This is a design report, not a launch -- "
          f"trim grids in this file before actually running scripts/run_*_search.py "
          f"if that number is bigger than your compute budget.")


if __name__ == "__main__":
    report_total_combos()
