"""Lightweight artifact logging for the FINAL (winning-config) evaluation
phase only -- deliberately not wired into the hyperparameter search phase,
where the sheer number of configs (hundreds per model) would make
per-config artifact dumps impractical and mostly useless (you only care
about the winning config's internals, not every candidate's).

One .pt file per (dataset, model, split), containing whatever the training
function captured: belief vectors / alpha gates / regime membership (ARIA
family only), per-node predicted class probabilities (all models), training
curves, and -- opt-in separately, since it's the bulkiest piece -- full
model weights.

Hand the resulting artifacts/ directory back for downstream figure-making:
belief-vector composition charts, homophily-vs-propagation-mass plots,
error analysis, ensemble-member correlation analysis -- the same kind of
by-hand analysis done earlier in this project for a handful of datasets,
now available for all 13 without re-training anything.
"""
import os

import torch


def artifact_path(artifacts_dir: str, dataset: str, model: str, split_idx: int) -> str:
    d = os.path.join(artifacts_dir, dataset, model)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"split_{split_idx}.pt")


def save_artifact(path: str, payload: dict):
    torch.save(payload, path)


def load_artifact(path: str, map_location: str = "cpu"):
    return torch.load(path, map_location=map_location)


def artifact_summary(payload: dict) -> str:
    """One-line human-readable summary of what's in a saved artifact, for
    quick sanity-checking without writing a full analysis script."""
    parts = [f"best_epoch={payload.get('best_epoch')}"]
    if payload.get("test_probs") is not None:
        parts.append(f"test_probs_shape={tuple(payload['test_probs'].shape)}")
    belief = payload.get("belief")
    if belief:
        parts.append("belief_keys=" + ",".join(k for k in belief if belief[k] is not None))
    if payload.get("model_state_dict") is not None:
        n_params = sum(v.numel() for v in payload["model_state_dict"].values())
        parts.append(f"weights_saved (n_params={n_params:,})")
    return ", ".join(parts)
