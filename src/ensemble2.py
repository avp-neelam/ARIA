"""ARIA-Ensemble2: a second post-hoc ensemble construction, requested as an
alternative to ARIA-Ensemble (which averages ARIA-PH + ARIA-PHG + ARIA-MoE).

ARIA-Ensemble2 instead averages ARIA (bare baseline, no PE, no homophily-aux
loss) + ARIA-PHG + ARIA-MoE -- i.e. it swaps out the PH member for the plain
baseline gate. Mirrors ensemble.py's structure/contract exactly (same
per-member independent training, same best-validation-checkpoint softmax
averaging), just with a different member set.
"""
import torch
import torch.nn.functional as F

from .models.aria import ARIA, ARIABounded, ARIAMoE
from .aria_augmentations import (augment_with_pe, compute_train_homophily_target,
                                  gate_aux_loss, compute_regime_membership)
from .datasets import compute_structural_features, compute_structural_features_v3


def _snapshot_weights(model, save_weights):
    if not save_weights:
        return None
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _train_member_base(data, split_idx, device, config, train_mask, val_mask, y, edge_index,
                        capture_artifacts=False, save_weights=False):
    """Bare ARIA baseline member: no PE augmentation, no homophily-aux loss
    (mirrors training.py's train_aria exactly)."""
    x = data.x.to(device)
    struct = compute_structural_features(data).to(device)

    model = ARIA(x.size(1), config["hidden_dim"], data.num_classes, x.size(0),
                 num_layers=config["num_layers"], rank=config["rank"],
                 struct_dim=struct.size(1), dropout=config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])

    best_val, best_probs, no_improve, best_epoch = -1.0, None, 0, 0
    loss_curve, val_curve = [], []
    best_beta, best_state = None, None
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
        if capture_artifacts:
            loss_curve.append(loss.item())
            val_curve.append(val_acc)
        if val_acc > best_val:
            best_val, no_improve, best_epoch = val_acc, 0, epoch
            best_probs = F.softmax(logits, dim=-1).detach()
            if capture_artifacts:
                best_beta = torch.stack(betas, dim=0).mean(0).detach().cpu()
                best_state = _snapshot_weights(model, save_weights)
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                break

    artifacts = None
    if capture_artifacts:
        artifacts = {
            "config": config, "best_epoch": best_epoch,
            "train_loss_curve": loss_curve, "val_acc_curve": val_curve,
            "test_probs": best_probs.cpu(),
            "belief": {"beta": best_beta, "alpha": None, "regime_membership": None},
            "model_state_dict": best_state,
        }
    return best_probs, best_val, artifacts


def _train_member_phg(data, split_idx, device, config, train_mask, val_mask, y, edge_index,
                       capture_artifacts=False, save_weights=False):
    x = augment_with_pe(data, k=config["pe_k"]).to(device)
    struct = compute_structural_features_v3(data).to(device)
    target, valid = compute_train_homophily_target(data, train_mask.cpu())
    target, valid = target.to(device), valid.to(device)

    model = ARIABounded(x.size(1), config["hidden_dim"], data.num_classes, x.size(0),
                         num_layers=config["num_layers"], rank=config["rank"],
                         struct_dim=struct.size(1), dropout=config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])

    best_val, best_probs, no_improve, best_epoch = -1.0, None, 0, 0
    loss_curve, val_curve = [], []
    best_beta, best_alpha, best_state = None, None, None
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
        if capture_artifacts:
            loss_curve.append(loss.item())
            val_curve.append(val_acc)
        if val_acc > best_val:
            best_val, no_improve, best_epoch = val_acc, 0, epoch
            best_probs = F.softmax(logits, dim=-1).detach()
            if capture_artifacts:
                best_beta = torch.stack(betas, dim=0).mean(0).detach().cpu()
                best_alpha = torch.stack(alphas, dim=0).mean(0).detach().cpu()
                best_state = _snapshot_weights(model, save_weights)
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                break

    artifacts = None
    if capture_artifacts:
        artifacts = {
            "config": config, "best_epoch": best_epoch,
            "train_loss_curve": loss_curve, "val_acc_curve": val_curve,
            "test_probs": best_probs.cpu(),
            "belief": {"beta": best_beta, "alpha": best_alpha, "regime_membership": None},
            "model_state_dict": best_state,
        }
    return best_probs, best_val, artifacts


def _train_member_moe(data, split_idx, device, config, train_mask, val_mask, y, edge_index, seed,
                       capture_artifacts=False, save_weights=False):
    x = data.x.to(device)
    struct = compute_structural_features(data).to(device)
    regime = compute_regime_membership(data, n_experts=config["n_experts"], seed=seed).to(device)

    model = ARIAMoE(x.size(1), config["hidden_dim"], data.num_classes, x.size(0),
                     num_layers=config["num_layers"], rank=config["rank"],
                     struct_dim=struct.size(1), dropout=config["dropout"],
                     n_experts=config["n_experts"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])

    best_val, best_probs, no_improve, best_epoch = -1.0, None, 0, 0
    loss_curve, val_curve = [], []
    best_beta, best_state = None, None
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
        if capture_artifacts:
            loss_curve.append(loss.item())
            val_curve.append(val_acc)
        if val_acc > best_val:
            best_val, no_improve, best_epoch = val_acc, 0, epoch
            best_probs = F.softmax(logits, dim=-1).detach()
            if capture_artifacts:
                best_beta = torch.stack(betas, dim=0).mean(0).detach().cpu()
                best_state = _snapshot_weights(model, save_weights)
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                break

    artifacts = None
    if capture_artifacts:
        artifacts = {
            "config": config, "best_epoch": best_epoch,
            "train_loss_curve": loss_curve, "val_acc_curve": val_curve,
            "test_probs": best_probs.cpu(),
            "belief": {"beta": best_beta, "alpha": None, "regime_membership": regime.detach().cpu()},
            "model_state_dict": best_state,
        }
    return best_probs, best_val, artifacts


def train_ensemble2_one_split(data, split_idx, device, base_config, phg_config,
                               moe_config, seed=0, capture_artifacts=False, save_weights=False):
    """ARIA-Ensemble2: post-hoc average of bare ARIA + ARIA-PHG + ARIA-MoE
    (swaps ARIA-Ensemble's PH member for the plain baseline). Same contract
    as ensemble.train_ensemble_one_split."""
    torch.manual_seed(seed)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    train_mask = data.train_mask[:, split_idx].to(device)
    val_mask = data.val_mask[:, split_idx].to(device)
    test_mask = data.test_mask[:, split_idx].to(device)

    probs_base, val_base, art_base = _train_member_base(
        data, split_idx, device, base_config, train_mask, val_mask, y, edge_index,
        capture_artifacts=capture_artifacts, save_weights=save_weights)
    probs_phg, val_phg, art_phg = _train_member_phg(
        data, split_idx, device, phg_config, train_mask, val_mask, y, edge_index,
        capture_artifacts=capture_artifacts, save_weights=save_weights)
    probs_moe, val_moe, art_moe = _train_member_moe(
        data, split_idx, device, moe_config, train_mask, val_mask, y, edge_index, seed,
        capture_artifacts=capture_artifacts, save_weights=save_weights)

    avg_probs = torch.stack([probs_base, probs_phg, probs_moe], dim=0).mean(0)
    test_acc = (avg_probs[test_mask].argmax(-1) == y[test_mask]).float().mean().item()
    avg_val_acc = (val_base + val_phg + val_moe) / 3.0

    if not capture_artifacts:
        return test_acc, avg_val_acc, None

    artifacts = {
        "config": {"base": base_config, "phg": phg_config, "moe": moe_config},
        "test_probs": avg_probs.cpu(),
        "test_probs_by_member": {
            "base": probs_base.cpu(), "phg": probs_phg.cpu(), "moe": probs_moe.cpu(),
        },
        "members": {"base": art_base, "phg": art_phg, "moe": art_moe},
    }
    return test_acc, avg_val_acc, artifacts
