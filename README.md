# ARIA

Benchmark suite for the ARIA graph-learning architecture family: a
per-node belief gate that mixes self / 1-hop / 2-hop / low-rank-global
channels, evaluated against MLP/GCN/GraphSAGE/GAT/GraphGPS baselines
across 13 node-classification datasets. Includes resumable, time-boxed
hyperparameter search with CUDA / Apple Silicon (MPS) / CPU
auto-detection, opt-in artifact logging (belief vectors, attention/gate
weights, training curves, per-node predictions), and the scripts used to
produce the project's reported results.

## Usage

```bash
pip install -r requirements.txt

# 1. Baselines
python scripts/run_baseline_search.py --time_budget 3600

# 2. ARIA family (aria, aria_ph, aria_phg, aria_moe)
python scripts/run_aria_search.py --time_budget 3600

# 3. ARIA-Ensemble (needs #2's output for aria_ph/aria_phg/aria_moo)
python scripts/run_ensemble_eval.py --time_budget 3600

# 4. Final table
python scripts/aggregate_results.py --out results/FINAL_RESULTS.md
```

Each script also accepts `--datasets` and `--models` to restrict scope
(e.g. `--datasets cora citeseer --models mlp gcn` to pilot a small corner
of the grid before committing to the full sweep).
