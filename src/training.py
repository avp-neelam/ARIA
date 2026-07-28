"""Single-split train/eval loops, one per model family, all device-aware
(everything moved onto whatever torch.device is passed in -- see
device.py). Every function returns (test_acc, val_acc, artifacts) for the
best-validation-accuracy epoch, early-stopped on patience. `artifacts` is
None unless capture_artifacts=True (see artifact_logger.py) -- callers that
don't care can just ignore the third element.

capture_artifacts is meant to be used ONLY for the final (winning-config)
evaluation phase, not the hyperparameter search phase -- see
scripts/run_*_search.py's --save_artifacts flag. When True, `artifacts` is
a dict with:
  config              : the hyperparameter config used
  best_epoch          : which epoch produced the returned test_acc
  train_loss_curve     : list of per-epoch training loss
  val_acc_curve        : list of per-epoch validation accuracy
  test_probs           : [n, num_classes] softmax probabilities for EVERY
                         node (not just the test split) at the best epoch
  belief               : dict of belief-gate internals (ARIA family only,
                         None for baselines) -- beta / alpha / regime_membership,
                         each averaged across layers where applicable
  model_state_dict     : full weights, only if save_weights=True (this is
                         the bulkiest piece, kept as a separate opt-in)

These are intentionally plain functions (not a class hierarchy), mirroring
the style of the earlier stages of this project.
"""
import torch
import torch.nn.functional as F

from .models.baselines import build_baseline
from .models.aria import ARIA, ARIABounded, ARIAMoE
from .aria_augmentations import (augment_with_pe, compute_rwpe,
                                  compute_train_homophily_target, gate_aux_loss,
                                  compute_regime_membership)
from .datasets import compute_structural_features, compute_structural_features_v3


def _snapshot_weights(model, save_weights):
    if not save_weights:
        return None
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def train_baseline(model_name, data, split_idx, device, config,
                    capture_artifacts=False, save_weights=False):
    """config: dict with hidden_dim, lr, weight_decay, dropout, num_layers,
    and (gat only) heads, (gps only) heads/pe_dim."""
    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    train_mask = data.train_mask[:, split_idx].to(device)
    val_mask = data.val_mask[:, split_idx].to(device)
    test_mask = data.test_mask[:, split_idx].to(device)

    pe = None
    if model_name == "gps":
        pe = compute_rwpe(data, k=config.get("pe_dim", 8)).to(device)

    model = build_baseline(
        model_name, x.size(1), config["hidden_dim"], data.num_classes,
        num_layers=config["num_layers"], dropout=config["dropout"],
        heads=config.get("heads", 8), pe_dim=config.get("pe_dim", 8),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])

    def forward():
        if model_name == "gps":
            return model(x, edge_index, pe=pe)
        return model(x, edge_index)

    best_val, best_test, no_improve, best_epoch = -1.0, 0.0, 0, 0
    loss_curve, val_curve = [], []
    best_probs, best_state = None, None

    for epoch in range(config["max_epochs"]):
        model.train()
        optimizer.zero_grad()
        logits = forward()
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = forward()
            val_acc = (logits[val_mask].argmax(-1) == y[val_mask]).float().mean().item()
            test_acc = (logits[test_mask].argmax(-1) == y[test_mask]).float().mean().item()

        if capture_artifacts:
            loss_curve.append(loss.item())
            val_curve.append(val_acc)

        if val_acc > best_val:
            best_val, best_test, no_improve, best_epoch = val_acc, test_acc, 0, epoch
            if capture_artifacts:
                best_probs = F.softmax(logits, dim=-1).detach().cpu()
                best_state = _snapshot_weights(model, save_weights)
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                break

    if not capture_artifacts:
        return best_test, best_val, None
    artifacts = {
        "config": config, "best_epoch": best_epoch,
        "train_loss_curve": loss_curve, "val_acc_curve": val_curve,
        "test_probs": best_probs, "belief": None, "model_state_dict": best_state,
    }
    return best_test, best_val, artifacts


# ---------------------------------------------------------------------------
# ARIA (baseline, bare -- no PE, no auxiliary loss)
# ---------------------------------------------------------------------------

def train_aria(data, split_idx, device, config, capture_artifacts=False, save_weights=False):
    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    struct = compute_structural_features(data).to(device)
    train_mask = data.train_mask[:, split_idx].to(device)
    val_mask = data.val_mask[:, split_idx].to(device)
    test_mask = data.test_mask[:, split_idx].to(device)

    model = ARIA(x.size(1), config["hidden_dim"], data.num_classes, x.size(0),
                 num_layers=config["num_layers"], rank=config["rank"],
                 struct_dim=struct.size(1), dropout=config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])

    best_val, best_test, no_improve, best_epoch = -1.0, 0.0, 0, 0
    loss_curve, val_curve = [], []
    best_probs, best_beta, best_state = None, None, None

    for epoch in range(config["max_epochs"]):
        model.train()
        optimizer.zero_grad()
        logits, betas = model(x, edge_index, struct, return_beta=True)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, betas = model(x, edge_index, struct, return_beta=True)
            val_acc = (logits[val_mask].argmax(-1) == y[val_mask]).float().mean().item()
            test_acc = (logits[test_mask].argmax(-1) == y[test_mask]).float().mean().item()

        if capture_artifacts:
            loss_curve.append(loss.item())
            val_curve.append(val_acc)

        if val_acc > best_val:
            best_val, best_test, no_improve, best_epoch = val_acc, test_acc, 0, epoch
            if capture_artifacts:
                best_probs = F.softmax(logits, dim=-1).detach().cpu()
                best_beta = torch.stack(betas, dim=0).mean(0).detach().cpu()
                best_state = _snapshot_weights(model, save_weights)
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                break

    if not capture_artifacts:
        return best_test, best_val, None
    artifacts = {
        "config": config, "best_epoch": best_epoch,
        "train_loss_curve": loss_curve, "val_acc_curve": val_curve,
        "test_probs": best_probs,
        "belief": {"beta": best_beta, "alpha": None, "regime_membership": None},
        "model_state_dict": best_state,
    }
    return best_test, best_val, artifacts


# ---------------------------------------------------------------------------
# ARIA-PH (ARIA + PE + homophily-auxiliary loss)
# ---------------------------------------------------------------------------

def train_aria_ph(data, split_idx, device, config, capture_artifacts=False, save_weights=False):
    x = augment_with_pe(data, k=config["pe_k"]).to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    struct = compute_structural_features(data).to(device)
    train_mask = data.train_mask[:, split_idx].to(device)
    val_mask = data.val_mask[:, split_idx].to(device)
    test_mask = data.test_mask[:, split_idx].to(device)
    target, valid = compute_train_homophily_target(data, train_mask.cpu())
    target, valid = target.to(device), valid.to(device)

    model = ARIA(x.size(1), config["hidden_dim"], data.num_classes, x.size(0),
                 num_layers=config["num_layers"], rank=config["rank"],
                 struct_dim=struct.size(1), dropout=config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])

    best_val, best_test, no_improve, best_epoch = -1.0, 0.0, 0, 0
    loss_curve, val_curve = [], []
    best_probs, best_beta, best_state = None, None, None

    for epoch in range(config["max_epochs"]):
        model.train()
        optimizer.zero_grad()
        logits, betas = model(x, edge_index, struct, return_beta=True)
        loss = (F.cross_entropy(logits[train_mask], y[train_mask]) +
                config["aux_lambda"] * gate_aux_loss(betas, target, valid))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, betas = model(x, edge_index, struct, return_beta=True)
            val_acc = (logits[val_mask].argmax(-1) == y[val_mask]).float().mean().item()
            test_acc = (logits[test_mask].argmax(-1) == y[test_mask]).float().mean().item()

        if capture_artifacts:
            loss_curve.append(loss.item())
            val_curve.append(val_acc)

        if val_acc > best_val:
            best_val, best_test, no_improve, best_epoch = val_acc, test_acc, 0, epoch
            if capture_artifacts:
                best_probs = F.softmax(logits, dim=-1).detach().cpu()
                best_beta = torch.stack(betas, dim=0).mean(0).detach().cpu()
                best_state = _snapshot_weights(model, save_weights)
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                break

    if not capture_artifacts:
        return best_test, best_val, None
    artifacts = {
        "config": config, "best_epoch": best_epoch,
        "train_loss_curve": loss_curve, "val_acc_curve": val_curve,
        "test_probs": best_probs,
        "belief": {"beta": best_beta, "alpha": None, "regime_membership": None},
        "model_state_dict": best_state,
    }
    return best_test, best_val, artifacts


# ---------------------------------------------------------------------------
# ARIA-PHG (ARIABounded + PE + homophily-auxiliary loss)
# ---------------------------------------------------------------------------

def train_aria_phg(data, split_idx, device, config, capture_artifacts=False, save_weights=False):
    x = augment_with_pe(data, k=config["pe_k"]).to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    struct = compute_structural_features_v3(data).to(device)
    train_mask = data.train_mask[:, split_idx].to(device)
    val_mask = data.val_mask[:, split_idx].to(device)
    test_mask = data.test_mask[:, split_idx].to(device)
    target, valid = compute_train_homophily_target(data, train_mask.cpu())
    target, valid = target.to(device), valid.to(device)

    model = ARIABounded(x.size(1), config["hidden_dim"], data.num_classes, x.size(0),
                         num_layers=config["num_layers"], rank=config["rank"],
                         struct_dim=struct.size(1), dropout=config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])

    best_val, best_test, no_improve, best_epoch = -1.0, 0.0, 0, 0
    loss_curve, val_curve = [], []
    best_probs, best_beta, best_alpha, best_state = None, None, None, None

    for epoch in range(config["max_epochs"]):
        model.train()
        optimizer.zero_grad()
        logits, betas, alphas = model(x, edge_index, struct, return_gates=True)
        loss = (F.cross_entropy(logits[train_mask], y[train_mask]) +
                config["aux_lambda"] * gate_aux_loss(betas, target, valid))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, betas, alphas = model(x, edge_index, struct, return_gates=True)
            val_acc = (logits[val_mask].argmax(-1) == y[val_mask]).float().mean().item()
            test_acc = (logits[test_mask].argmax(-1) == y[test_mask]).float().mean().item()

        if capture_artifacts:
            loss_curve.append(loss.item())
            val_curve.append(val_acc)

        if val_acc > best_val:
            best_val, best_test, no_improve, best_epoch = val_acc, test_acc, 0, epoch
            if capture_artifacts:
                best_probs = F.softmax(logits, dim=-1).detach().cpu()
                best_beta = torch.stack(betas, dim=0).mean(0).detach().cpu()
                best_alpha = torch.stack(alphas, dim=0).mean(0).detach().cpu()
                best_state = _snapshot_weights(model, save_weights)
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                break

    if not capture_artifacts:
        return best_test, best_val, None
    artifacts = {
        "config": config, "best_epoch": best_epoch,
        "train_loss_curve": loss_curve, "val_acc_curve": val_curve,
        "test_probs": best_probs,
        "belief": {"beta": best_beta, "alpha": best_alpha, "regime_membership": None},
        "model_state_dict": best_state,
    }
    return best_test, best_val, artifacts


# ---------------------------------------------------------------------------
# ARIA-MoE
# ---------------------------------------------------------------------------

def train_aria_moe(data, split_idx, device, config, seed=0,
                    capture_artifacts=False, save_weights=False):
    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    struct = compute_structural_features(data).to(device)
    regime = compute_regime_membership(data, n_experts=config["n_experts"], seed=seed).to(device)
    train_mask = data.train_mask[:, split_idx].to(device)
    val_mask = data.val_mask[:, split_idx].to(device)
    test_mask = data.test_mask[:, split_idx].to(device)

    model = ARIAMoE(x.size(1), config["hidden_dim"], data.num_classes, x.size(0),
                     num_layers=config["num_layers"], rank=config["rank"],
                     struct_dim=struct.size(1), dropout=config["dropout"],
                     n_experts=config["n_experts"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])

    best_val, best_test, no_improve, best_epoch = -1.0, 0.0, 0, 0
    loss_curve, val_curve = [], []
    best_probs, best_beta, best_state = None, None, None

    for epoch in range(config["max_epochs"]):
        model.train()
        optimizer.zero_grad()
        logits, betas = model(x, edge_index, struct, regime, return_beta=True)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, betas = model(x, edge_index, struct, regime, return_beta=True)
            val_acc = (logits[val_mask].argmax(-1) == y[val_mask]).float().mean().item()
            test_acc = (logits[test_mask].argmax(-1) == y[test_mask]).float().mean().item()

        if capture_artifacts:
            loss_curve.append(loss.item())
            val_curve.append(val_acc)

        if val_acc > best_val:
            best_val, best_test, no_improve, best_epoch = val_acc, test_acc, 0, epoch
            if capture_artifacts:
                best_probs = F.softmax(logits, dim=-1).detach().cpu()
                best_beta = torch.stack(betas, dim=0).mean(0).detach().cpu()
                best_state = _snapshot_weights(model, save_weights)
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                break

    if not capture_artifacts:
        return best_test, best_val, None
    artifacts = {
        "config": config, "best_epoch": best_epoch,
        "train_loss_curve": loss_curve, "val_acc_curve": val_curve,
        "test_probs": best_probs,
        "belief": {"beta": best_beta, "alpha": None, "regime_membership": regime.detach().cpu()},
        "model_state_dict": best_state,
    }
    return best_test, best_val, artifacts


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_ARIA_TRAIN_FN = {
    "aria": train_aria,
    "aria_ph": train_aria_ph,
    "aria_phg": train_aria_phg,
    "aria_moe": train_aria_moe,
}


def train_one_split(model_name, data, split_idx, device, config, seed=0,
                     capture_artifacts=False, save_weights=False):
    """Unified entry point. model_name in BASELINE_NAMES + ARIA_MODEL_NAMES
    (aria_ensemble is NOT handled here -- see ensemble.py). Always returns
    (test_acc, val_acc, artifacts); artifacts is None unless
    capture_artifacts=True."""
    torch.manual_seed(seed)
    if model_name in _ARIA_TRAIN_FN:
        fn = _ARIA_TRAIN_FN[model_name]
        if model_name == "aria_moe":
            return fn(data, split_idx, device, config, seed=seed,
                       capture_artifacts=capture_artifacts, save_weights=save_weights)
        return fn(data, split_idx, device, config,
                   capture_artifacts=capture_artifacts, save_weights=save_weights)
    return train_baseline(model_name, data, split_idx, device, config,
                           capture_artifacts=capture_artifacts, save_weights=save_weights)
