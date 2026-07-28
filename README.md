# ARIA

Benchmark suite for the ARIA graph-learning architecture family: a
per-node belief gate that mixes self / 1-hop / 2-hop / low-rank-global
channels, evaluated against MLP/GCN/GraphSAGE/GAT/GraphGPS baselines
across 13 node-classification datasets. Includes resumable, time-boxed
hyperparameter search with CUDA / Apple Silicon (MPS) / CPU
auto-detection, opt-in artifact logging (belief vectors, attention/gate
weights, training curves, per-node predictions), and the scripts used to
produce the project's reported results.

## Results

`results/FINAL_RESULTS.md` has the full head-to-head accuracy table (5
baselines + ARIA family, 13 datasets, 10 splits each). `results/
BELIEF_VALIDATION.md` has a causal check of the belief vectors themselves
(does a high per-channel gate weight actually predict that channel
mattering, measured by ablating it and watching the accuracy drop). The
underlying JSON (`results/*.json`) backs both.

Raw datasets, trained model weights, and full search-checkpoint state are
**not** included in this repo (see `.gitignore`) — they're either
auto-downloaded on first run or regenerable via the scripts below with
`--save_artifacts` / `--save_weights`.

## What's being compared

**Baselines:** MLP, GCN, GraphSAGE, GAT, GPS (GraphGPS -- local MPNN +
global transformer attention, via PyG's `GPSConv`).

**ARIA family** (5 models, all in `src/models/aria.py` + `src/ensemble.py`):

| Name | What it is |
|---|---|
| `ARIA` (baseline) | The original per-node belief gate: one softmax over 4 channels (self / 1-hop / 2-hop / low-rank global). No extra training-time add-ons. |
| `ARIA-PH` | Same architecture as `ARIA`, plus two training-time add-ons: a positional-encoding (**P**E) input augmentation and a **H**omophily-supervised auxiliary loss on the gate's propagation weight. (This was called "ARIA-Combined" in earlier project notes.) |
| `ARIA-PHG` | Same two add-ons (**P**E + **H**omophily-aux-loss) as ARIA-PH, but on a different gate: a 3-way softmax over {self, 1-hop, 2-hop} plus a separate, bounded scalar correction gate for the global channel (the "**G**" -- Gated/bounded global channel). (This was called "ARIA-V4" in earlier project notes.) ARIA-PH and ARIA-PHG differ *only* in this gate mechanism -- everything else about them is identical. |
| `ARIA-MoE` | Original gate, unchanged, but the shared output projection becomes a mixture of 4 (or more) "expert" projections, routed by a fixed, non-trainable structural clustering computed once before training. |
| `ARIA-Ensemble` | Not a fourth architecture -- a post-hoc average of ARIA-PH + ARIA-PHG + ARIA-MoE's predicted class probabilities, each member trained and early-stopped completely independently. See `src/ensemble.py`. |

The `ARIA-PH` / `ARIA-PHG` naming is a suggestion (easy to change -- see
`DISPLAY_NAMES` in `src/models/aria.py`, one dict, used everywhere
including `scripts/aggregate_results.py`'s table headers).

## Datasets (13)

| Category | Datasets | Splits |
|---|---|---|
| Citation networks | Cora, CiteSeer, PubMed | geom-gcn (10 standard splits) |
| Co-purchase graphs | Amazon-Photo, Amazon-Computers | generated (10 stratified 60/20/20) |
| WebKB | Texas, Wisconsin, Cornell | geom-gcn (10 standard splits) |
| Actor | Actor | geom-gcn (10 standard splits) |
| Heterophily benchmark | Roman-Empire | own splits (Platonov et al. 2023) |
| struc2vec Airports | USA-Airports, Brazil-Airports, Europe-Airports | generated (10 stratified 60/20/20) |

See `src/datasets.py` for the loader (`load_dataset(name)`) and
`ALL_DATASETS` for the exact string keys.

**One unverified item:** Roman-Empire's dataset download
(`raw.githubusercontent.com`) is blocked by this development sandbox's
network allowlist, so `HeterophilousGraphDataset` has never actually
finished downloading here. The loader code is written defensively (handles
either a 1-D or [n, n_splits] mask shape), but double-check
`data.train_mask.shape` the first time this actually runs somewhere with
normal network access.

## Device support

`src/device.py`'s `get_device()` auto-selects CUDA > Apple Silicon (MPS) >
CPU. Every script accepts `--device {auto,cpu,cuda,mps}` to override.

If you hit an "operator not implemented for MPS" error on Apple Silicon (a
handful of PyTorch ops still lack MPS kernels depending on your torch
version), set `PYTORCH_ENABLE_MPS_FALLBACK=1` in your environment to let
those specific ops silently fall back to CPU rather than crashing:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_baseline_search.py ...
```

## Repository layout

```
├── src/
│   ├── device.py              # CUDA/MPS/CPU auto-detection
│   ├── datasets.py             # all 13 dataset loaders + structural descriptors
│   ├── aria_augmentations.py   # PE, homophily-aux loss, MoE clustering
│   ├── search_space.py         # hyperparameter grids + per-dataset training protocol
│   ├── training.py              # single-split train/eval for baselines + ARIA/PH/PHG/MoE
│   ├── ensemble.py              # ARIA-Ensemble (3-member logit averaging)
│   ├── artifact_logger.py       # save/load helpers for --save_artifacts output
│   └── models/
│       ├── baselines.py         # MLP, GCN, GraphSAGE, GAT, GPS
│       └── aria.py              # ARIA, ARIABounded, ARIAMoE layer + model classes
├── scripts/
│   ├── smoke_test.py            # pre-flight check -- run this FIRST, see below
│   ├── run_baseline_search.py   # resumable HPO for the 5 baselines
│   ├── run_aria_search.py       # resumable HPO for ARIA/ARIA-PH/ARIA-PHG/ARIA-MoE
│   ├── run_ensemble_eval.py     # assembles + evaluates ARIA-Ensemble from the above
│   └── aggregate_results.py     # combines everything into one final markdown table
├── slurm/                       # ganymede2 SLURM job scripts -- see "Running on ganymede2" below
│   ├── common.sh                 # shared env setup, sourced by every .slurm script
│   ├── setup_env.sh              # one-time conda env + dataset pre-download (run on login node)
│   ├── smoke_test.slurm          # dev partition correctness check
│   ├── gpu_check.slurm           # optional CUDA sanity check on a GPU node
│   └── 01-04_*.slurm             # the real sweep, self-chaining phase by phase
├── results/                     # reported results (JSON + markdown tables), committed
├── state/                       # resumable checkpoints -- gitignored, populated when run
├── artifacts/                   # belief vectors/weights/curves -- gitignored, only if --save_artifacts
├── logs/                        # SLURM job stdout/stderr -- gitignored, populated when run
└── requirements.txt
```

## Before the real sweep: smoke test

`scripts/smoke_test.py` loads all 13 datasets and runs a tiny (3-epoch,
single-split) training pass for every model (5 baselines + ARIA/ARIA-PH/
ARIA-PHG/ARIA-MoE + ARIA-Ensemble -- 10 total, 130 combos by default) to
catch dataset-download issues, shape mismatches, or device-placement bugs
before committing to the full sweep. It prints a PASS/FAIL table and exits
non-zero if anything failed:

```bash
python scripts/smoke_test.py --device auto

# narrow it down while debugging a specific failure:
python scripts/smoke_test.py --datasets roman-empire --models aria_moe
SMOKE_TEST_VERBOSE=1 python scripts/smoke_test.py --datasets roman-empire --models aria_moe
```

Run this first on the actual machine you'll do the real sweep on -- it's
the fastest way to confirm network access, dataset caching, and the CUDA/MPS
device path all work there before spending hours on the full grid.

## Artifact logging (belief vectors, weights, curves, probabilities)

Both search scripts and the ensemble script accept `--save_artifacts`,
which saves one `.pt` file per `(dataset, model, split)` under
`--artifacts_dir` (default `artifacts/`) -- but **only during the final
phase** (the winning config's re-evaluation on the full 10 splits), never
during the hyperparameter search itself, since dumping artifacts for every
candidate config would be both enormous and useless.

Each artifact (load with `src/artifact_logger.load_artifact(path)`) is a
dict with:

- `config` -- the winning hyperparameter config used for that run
- `best_epoch`, `train_loss_curve`, `val_acc_curve` -- training dynamics
- `test_probs` -- softmax class probabilities for every node (not just the
  test split), at the best-validation-accuracy epoch
- `belief` -- `{"beta": ..., "alpha": ..., "regime_membership": ...}`,
  whichever apply to that model (baselines: all `None`; `aria`/`aria_ph`:
  `beta` only; `aria_phg`: `beta` + `alpha`; `aria_moe`: `beta` +
  `regime_membership`)
- `model_state_dict` -- full weights, only if you also pass `--save_weights`
  (kept opt-in since it's the bulkiest piece)

ARIA-Ensemble artifacts are shaped slightly differently (it isn't a single
model): top-level `test_probs`/`test_probs_by_member` for the averaged and
per-member predictions, plus a `members` dict with one regular
artifact-shaped entry each for `ph`/`phg`/`moe`.

```bash
python scripts/run_baseline_search.py --time_budget 3600 --save_artifacts
python scripts/run_aria_search.py --time_budget 3600 --save_artifacts --save_weights
python scripts/run_ensemble_eval.py --time_budget 3600 --save_artifacts
```

Hand the resulting `artifacts/` directory back for downstream figure-making
(belief-vector composition charts, homophily-vs-propagation-mass plots,
error analysis, ensemble-member correlation) without re-training anything.

## Before you run anything

The hyperparameter grids in `src/search_space.py` are intentionally kept
small (2-3 values per hyperparameter, per the scoping decision behind this
repo), but they still multiply out to a lot of individual training runs
once you cross 13 datasets x several models x a grid x multiple splits.
Check the actual numbers before launching anything:

```bash
cd aria_benchmark
python -m src.search_space
# or equivalently:
python scripts/run_baseline_search.py --dry_run
python scripts/run_aria_search.py --dry_run
```

As designed right now, this reports roughly **77,000 individual
train-to-convergence runs** across everything (dominated by ARIA-PH and
ARIA-PHG's 576-config grids each, mostly from the `rank` dimension --
`[8, 16, 32]`, matching the earlier rank-sweep finding that 64 didn't
improve further over 32 -- crossed with `aux_lambda`/`pe_k`). If that's more
than your compute budget, trim `BASELINE_SEARCH_SPACE` / `ARIA_SEARCH_SPACE`
in `src/search_space.py` further -- e.g. dropping `aux_lambda` to a single
fixed value cuts the ARIA-PH/PHG grids by another 2x.

## How the search works

For each `(dataset, model)` pair:

1. **Search phase**: every hyperparameter config in the grid is trained on
   a small number of splits (`n_search_splits`, 3 by default --
   `src/search_space.DATASET_TRAIN_CONFIG`), and its mean validation
   accuracy is recorded.
2. **Final phase**: whichever config had the best mean validation accuracy
   is re-trained on the full `n_final_splits` (10) splits -- that's the
   number that goes in the results table.
3. **ARIA-Ensemble** (`run_ensemble_eval.py`) doesn't get its own search --
   it reads the winning configs for ARIA-PH/ARIA-PHG/ARIA-MoE from step 2's
   output and evaluates the 3-member ensemble using each member's own
   winning hyperparameters.

Everything is resumable: state is checkpointed to disk (`--state_dir`)
after every single training run, so any script can be killed and
re-invoked with the same arguments to pick up where it left off. Each
script takes `--time_budget <seconds>` to bound how long a single
invocation runs before checkpointing and exiting -- this project's earlier
stages were built in a sandbox with a hard ~45-second wall-clock limit per
shell invocation, so the scripts are written to be safe to call in a tight
loop; on a normal machine you can just pass a large `--time_budget` (or
loop the same command in a shell `while` loop until it prints `ALL_DONE`).

## Usage (once you've reviewed the grid sizes above)

```bash
pip install -r requirements.txt

# 0. Pre-flight check (do this first, especially on a new machine)
python scripts/smoke_test.py

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
