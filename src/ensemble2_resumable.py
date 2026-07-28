"""Epoch-level resumable version of ensemble2.py's member trainers.

Needed because this benchmark is being run in a sandbox where each process
invocation has a hard wall-clock ceiling well under the time some
(dataset, member) combinations need to finish training (low-lr configs can
run close to their full max_epochs budget before patience triggers). This
module checkpoints model/optimizer state + best-so-far val/probs to disk
every epoch and can resume exactly where it left off, so forward progress
survives being killed mid-training. Numerically identical to
ensemble2.py's members (same architectures, same optimizer, same
early-stopping rule) -- this is purely an infra adaptation, not a change to
the method.
"""
import os
import time

import torch
import torch.nn.functional as F

from .models.aria import ARIA, ARIABounded, ARIAMoE
from .aria_augmentations import (augment_with_pe, compute_train_homophily_target,
                                  gate_aux_loss, compute_regime_membership)
from .datasets import compute_structural_features, compute_structural_features_v3


def _save_ckpt(path, state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_ckpt(path, map_location="cpu"):
    if os.path.exists(path):
        return torch.load(path, map_location=map_location, weights_only=False)
    return None


def train_member_base(data, split_idx, device, config, train_mask, val_mask, y, edge_index,
                       ckpt_path, deadline):
    x = data.x.to(device)
    struct = compute_structural_features(data).to(device)
    model = ARIA(x.size(1), config["hidden_dim"], data.num_classes, x.size(0),
                 num_layers=config["num_layers"], rank=config["rank"],
                 struct_dim=struct.size(1), dropout=config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])

    start_epoch, best_val, best_probs, no_improve, best_epoch = 0, -1.0, None, 0, 0
    ck = _load_ckpt(ckpt_path)
    if ck is not None:
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        start_epoch, best_val, best_probs = ck["epoch"], ck["best_val"], ck["best_probs"]
        no_improve, best_epoch = ck["no_improve"], ck["best_epoch"]

    for epoch in range(start_epoch, config["max_epochs"]):
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
        if val_acc > best_val:
            best_val, no_improve, best_epoch = val_acc, 0, epoch
            best_probs = F.softmax(logits, dim=-1).detach()
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                return best_probs, best_val, True
        if time.time() >= deadline:
            _save_ckpt(ckpt_path, {"epoch": epoch + 1, "model": model.state_dict(),
                                    "optim": optimizer.state_dict(), "best_val": best_val,
                                    "best_probs": best_probs, "no_improve": no_improve,
                                    "best_epoch": best_epoch})
            return None, None, False
    return best_probs, best_val, True


_PE_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "state", "pe_cache")


def _cached_pe(data, k, dataset_key):
    """augment_with_pe is expensive on larger graphs (RWPE does k rounds of
    sparse propagation over batches of the identity matrix) and gets
    recomputed from scratch at the start of every resumed process
    invocation otherwise -- caching it to disk turns that from "paid every
    single resumed call" into "paid once"."""
    os.makedirs(_PE_CACHE_DIR, exist_ok=True)
    path = os.path.join(_PE_CACHE_DIR, f"{dataset_key}__k{k}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    pe_x = augment_with_pe(data, k=k)
    tmp = path + ".tmp"
    torch.save(pe_x, tmp)
    os.replace(tmp, path)
    return pe_x


def train_member_phg(data, split_idx, device, config, train_mask, val_mask, y, edge_index,
                      ckpt_path, deadline, dataset_key="unknown"):
    x = _cached_pe(data, config["pe_k"], dataset_key).to(device)
    struct = compute_structural_features_v3(data).to(device)
    target, valid = compute_train_homophily_target(data, train_mask.cpu())
    target, valid = target.to(device), valid.to(device)
    model = ARIABounded(x.size(1), config["hidden_dim"], data.num_classes, x.size(0),
                         num_layers=config["num_layers"], rank=config["rank"],
                         struct_dim=struct.size(1), dropout=config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])

    start_epoch, best_val, best_probs, no_improve, best_epoch = 0, -1.0, None, 0, 0
    ck = _load_ckpt(ckpt_path)
    if ck is not None:
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        start_epoch, best_val, best_probs = ck["epoch"], ck["best_val"], ck["best_probs"]
        no_improve, best_epoch = ck["no_improve"], ck["best_epoch"]

    for epoch in range(start_epoch, config["max_epochs"]):
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
        if val_acc > best_val:
            best_val, no_improve, best_epoch = val_acc, 0, epoch
            best_probs = F.softmax(logits, dim=-1).detach()
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                return best_probs, best_val, True
        if time.time() >= deadline:
            _save_ckpt(ckpt_path, {"epoch": epoch + 1, "model": model.state_dict(),
                                    "optim": optimizer.state_dict(), "best_val": best_val,
                                    "best_probs": best_probs, "no_improve": no_improve,
                                    "best_epoch": best_epoch})
            return None, None, False
    return best_probs, best_val, True


def _cached_regime(data, n_experts, seed, dataset_key):
    os.makedirs(_PE_CACHE_DIR, exist_ok=True)
    path = os.path.join(_PE_CACHE_DIR, f"{dataset_key}__regime_e{n_experts}_s{seed}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    regime = compute_regime_membership(data, n_experts=n_experts, seed=seed)
    tmp = path + ".tmp"
    torch.save(regime, tmp)
    os.replace(tmp, path)
    return regime


def train_member_moe(data, split_idx, device, config, train_mask, val_mask, y, edge_index, seed,
                      ckpt_path, deadline, dataset_key="unknown"):
    x = data.x.to(device)
    struct = compute_structural_features(data).to(device)
    regime = _cached_regime(data, config["n_experts"], seed, dataset_key).to(device)
    model = ARIAMoE(x.size(1), config["hidden_dim"], data.num_classes, x.size(0),
                     num_layers=config["num_layers"], rank=config["rank"],
                     struct_dim=struct.size(1), dropout=config["dropout"],
                     n_experts=config["n_experts"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])

    start_epoch, best_val, best_probs, no_improve, best_epoch = 0, -1.0, None, 0, 0
    ck = _load_ckpt(ckpt_path)
    if ck is not None:
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        start_epoch, best_val, best_probs = ck["epoch"], ck["best_val"], ck["best_probs"]
        no_improve, best_epoch = ck["no_improve"], ck["best_epoch"]

    for epoch in range(start_epoch, config["max_epochs"]):
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
        if val_acc > best_val:
            best_val, no_improve, best_epoch = val_acc, 0, epoch
            best_probs = F.softmax(logits, dim=-1).detach()
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                return best_probs, best_val, True
        if time.time() >= deadline:
            _save_ckpt(ckpt_path, {"epoch": epoch + 1, "model": model.state_dict(),
                                    "optim": optimizer.state_dict(), "best_val": best_val,
                                    "best_probs": best_probs, "no_improve": no_improve,
                                    "best_epoch": best_epoch})
            return None, None, False
    return best_probs, best_val, True


MEMBER_TRAIN_FN = {"aria": train_member_base, "aria_phg": train_member_phg, "aria_moe": train_member_moe}
