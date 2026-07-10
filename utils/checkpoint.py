"""
Checkpoint utilities for DNNet.

save_checkpoint : saves model, optimisers, schedulers, and training state
load_checkpoint : restores all of the above
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch


def save_checkpoint(
    checkpoint_dir: str,
    epoch:          int,
    model:          torch.nn.Module,
    optimizer_adam: torch.optim.Optimizer,
    optimizer_sgd:  torch.optim.Optimizer,
    scheduler_adam,
    scheduler_sgd,
    metrics:        Dict[str, float],
    is_best:        bool = False,
    filename:       str  = "checkpoint.pth",
) -> str:
    """Save a full training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    state = {
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "adam_state":     optimizer_adam.state_dict(),
        "sgd_state":      optimizer_sgd.state_dict(),
        "sched_adam":     scheduler_adam.state_dict(),
        "sched_sgd":      scheduler_sgd.state_dict(),
        "metrics":        metrics,
    }

    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)
        print(f"  ✓ New best model saved → {best_path}  (rank_1={metrics.get('rank_1', 0):.3f}%)")

    return path


def load_checkpoint(
    checkpoint_path: str,
    model:           torch.nn.Module,
    optimizer_adam:  Optional[torch.optim.Optimizer] = None,
    optimizer_sgd:   Optional[torch.optim.Optimizer] = None,
    scheduler_adam=None,
    scheduler_sgd=None,
    device:          torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Load a checkpoint.  Returns the stored metadata dict.
    Optimisers and schedulers are restored in-place if provided.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model_state"])

    if optimizer_adam and "adam_state" in ckpt:
        optimizer_adam.load_state_dict(ckpt["adam_state"])
    if optimizer_sgd and "sgd_state" in ckpt:
        optimizer_sgd.load_state_dict(ckpt["sgd_state"])
    if scheduler_adam and "sched_adam" in ckpt:
        scheduler_adam.load_state_dict(ckpt["sched_adam"])
    if scheduler_sgd and "sched_sgd" in ckpt:
        scheduler_sgd.load_state_dict(ckpt["sched_sgd"])

    print(
        f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
        f"({checkpoint_path})"
    )
    return ckpt
