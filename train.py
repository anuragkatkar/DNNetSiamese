"""
Training loop for DNNet.

Features:
  - Dual optimisers: Adam (contrastive/feature branch) + SGD (ArcFace head)
  - Both use the paper's LR schedule: constant → linear decay
  - W&B logging (toggled via config.USE_WANDB)
  - Best-model checkpoint saved when Rank-1 val accuracy improves
  - Fixed seed for reproducibility
  - Gradient clipping for stability
"""

import os
import sys
import time
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# ── Local imports ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from configs import config
from data.dataset import build_kfold_loaders
from models.dnnet import build_model
from models.losses import TotalLoss
from utils.schedulers import get_dnnet_scheduler, sync_lambda_scheduler_param_groups
from utils.evaluation import evaluate
from utils.checkpoint import save_checkpoint, load_checkpoint


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Single training epoch ─────────────────────────────────────────────────────

def train_one_epoch(
    model:          nn.Module,
    loader:         DataLoader,
    criterion:      TotalLoss,
    opt_adam:       torch.optim.Optimizer,
    opt_sgd:        torch.optim.Optimizer,
    device:         torch.device,
    epoch:          int,
    scaler,
    wandb_run=None,
) -> dict:
    model.train()

    total_loss = con_loss = arc_a_loss = arc_p_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (anchor_img, pair_img, anchor_cls, pair_cls, pair_bin) in enumerate(loader):
        anchor_img  = anchor_img.to(device, non_blocking=True)
        pair_img    = pair_img.to(device,   non_blocking=True)
        anchor_cls  = anchor_cls.to(device, non_blocking=True)
        pair_cls    = pair_cls.to(device,   non_blocking=True)
        pair_bin    = pair_bin.to(device,   non_blocking=True)

        # ── Forward ───────────────────────────────────────────────────────
        with autocast(device_type='cuda'):
            anchor_emb, pair_emb, anchor_logits, pair_logits = model(anchor_img, pair_img)

            l_total, l_con, l_arc_a, l_arc_p = criterion(
                anchor_emb    = anchor_emb,
                pair_emb      = pair_emb,
                pair_labels   = pair_bin,
                anchor_logits = anchor_logits,
                pair_logits   = pair_logits,
                anchor_class  = anchor_cls,
                pair_class    = pair_cls,
            )

        opt_adam.zero_grad()
        opt_sgd.zero_grad()
        scaler.scale(l_total).backward()
        scaler.unscale_(opt_adam)
        scaler.unscale_(opt_sgd)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(opt_adam)
        scaler.step(opt_sgd)
        scaler.update()

        # ── Accumulate ────────────────────────────────────────────────────
        total_loss  += l_total.item()
        con_loss    += l_con.item()
        arc_a_loss  += l_arc_a.item()
        arc_p_loss  += l_arc_p.item()

        if (batch_idx + 1) % max(1, n_batches // 5) == 0:
            log.info(
                f"  Epoch {epoch:03d} [{batch_idx+1:4d}/{n_batches}] "
                f"loss={l_total.item():.4f}  "
                f"con={l_con.item():.4f}  "
                f"arc_a={l_arc_a.item():.4f}  "
                f"arc_p={l_arc_p.item():.4f}"
            )

    avg = lambda x: x / n_batches
    metrics = {
        "train/loss_total":   avg(total_loss),
        "train/loss_con":     avg(con_loss),
        "train/loss_arc_anc": avg(arc_a_loss),
        "train/loss_arc_pair":avg(arc_p_loss),
        "train/lr_adam":      opt_adam.param_groups[0]["lr"],
        "train/lr_sgd":       opt_sgd.param_groups[0]["lr"],
        "epoch":              epoch,
    }

    if wandb_run:
        wandb_run.log(metrics)

    return metrics


# ── Main training function ────────────────────────────────────────────────────

def train(
    fold:            int  = config.VAL_FOLD,
    resume_ckpt:     Optional[str] = None,
):
    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    log.info(f"Loading dataset from: {config.DATA_ROOT}  (fold {fold}/{config.NUM_FOLDS-1})")
    train_loader, val_loader, num_classes = build_kfold_loaders(
        root       = config.DATA_ROOT,
        fold       = fold,
        num_folds  = config.NUM_FOLDS,
        batch_size = config.BATCH_SIZE,
        num_workers= config.NUM_WORKERS,
        seed       = config.SEED,
    )
    log.info(f"Number of classes (dog IDs): {num_classes}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(num_classes=num_classes, cfg=config).to(device)
    scaler = GradScaler() 

    if config.FREEZE_BACKBONE_EPOCHS > 0:
        model.dnnet.feature_extractor.freeze_backbone()
        log.info(f"Backbone frozen for first {config.FREEZE_BACKBONE_EPOCHS} epochs.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {total_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = TotalLoss(
        contrastive_margin = config.CONTRASTIVE_MARGIN,
        arcface_scale      = config.ARCFACE_SCALE,
        arcface_margin     = config.ARCFACE_MARGIN,
    )

    # ── Optimisers (paper: Adam for feature branch, SGD for ArcFace head) ─
    # Parameters excluding ArcFace head
    feat_params = [
        p for n, p in model.named_parameters()
        if "arcface_head" not in n and p.requires_grad
    ]
    arc_params  = list(model.arcface_head.parameters())

    opt_adam = torch.optim.Adam(
        feat_params,
        lr     = config.LR_CONTRASTIVE,
        betas  = (config.ADAM_BETA1, config.ADAM_BETA2),
    )

    opt_sgd = torch.optim.SGD(
        arc_params,
        lr           = config.LR_ARCFACE,
        momentum     = config.SGD_MOMENTUM,
        weight_decay = config.SGD_WEIGHT_DECAY,
    )

    # ── Schedulers ────────────────────────────────────────────────────────
    sched_adam = get_dnnet_scheduler(opt_adam, config.LR_WARMUP_EPOCHS, config.NUM_EPOCHS)
    sched_sgd  = get_dnnet_scheduler(opt_sgd,  config.LR_WARMUP_EPOCHS, config.NUM_EPOCHS)

    # ── W&B ───────────────────────────────────────────────────────────────
    wandb_run = None
    if config.USE_WANDB:
        try:
            import wandb
            wandb_run = wandb.init(
                project = config.PROJECT_NAME,
                name    = f"fold_{fold}",
                config  = {
                    k: v for k, v in vars(config).items()
                    if not k.startswith("_") and isinstance(v, (int, float, str, bool, list))
                },
            )
            wandb_run.watch(model, log="gradients", log_freq=100)
            log.info("W&B run initialised.")
        except Exception as e:
            log.warning(f"W&B init failed ({e}). Continuing without logging.")
            wandb_run = None

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_rank1  = 0.0
    best_loss = 9999999999
    patience = max(0, int(getattr(config, "PATIENCE", 0)))
    no_improve_validations = 0
    no_improve_training = 0
    

    if resume_ckpt:
        ckpt = load_checkpoint(
            resume_ckpt, model, opt_adam, opt_sgd, sched_adam, sched_sgd, device
        )
        start_epoch = ckpt["epoch"] + 1
        best_rank1  = ckpt["metrics"].get("val/rank_1", 0.0)
        log.info(f"Resuming from epoch {start_epoch}  (best rank-1: {best_rank1:.3f}%)")

    # ── Training loop ─────────────────────────────────────────────────────
    log.info(f"Starting training for {config.NUM_EPOCHS} epochs …")
    if patience > 0:
        log.info(f"Early stopping enabled: patience={patience} validation(s) without Rank-1 improvement.")
    ckpt_dir = os.path.join(config.CHECKPOINT_DIR, f"fold_{fold}")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch += 1
        t0 = time.time()

        if epoch == config.FREEZE_BACKBONE_EPOCHS and config.FREEZE_BACKBONE_EPOCHS > 0:
            log.info(f"Epoch {epoch}: unfreezing backbone for end-to-end fine-tuning …")
            model.dnnet.feature_extractor.unfreeze_backbone()

            backbone_params = list(model.dnnet.feature_extractor.backbone.parameters())
            opt_adam.add_param_group({
                "params": backbone_params,
                "lr":     config.LR_CONTRASTIVE * 0.1,  # 10x lower LR for pretrained weights
            })
            sync_lambda_scheduler_param_groups(sched_adam)
            log.info(f"  Added backbone params to Adam at LR={config.LR_CONTRASTIVE * 0.1:.1e}")


        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion,
            opt_adam, opt_sgd, device, epoch, scaler, wandb_run
        )

        # LR step
        sched_adam.step()
        sched_sgd.step()

        # Validate (every 5 epochs to save time)
        val_metrics = {}
        if (epoch) % 5 == 0 or epoch == config.NUM_EPOCHS:
            val_metrics = evaluate(
                model, val_loader, device,
                rank_k        = config.RANK_K,
                far_thresholds= config.FAR_THRESHOLDS,
            )
            # Prefix keys for W&B
            val_metrics_log = {f"val/{k}": v for k, v in val_metrics.items()}
            val_metrics_log["epoch"] = epoch

            rank1 = val_metrics.get("rank_1", 0.0)
            loss = train_metrics["train/loss_total"]

            log.info(
                f"Epoch {epoch:03d}/{config.NUM_EPOCHS}  "
                f"[{time.time()-t0:.1f}s]  "
                f"rank-1={rank1:.3f}%  "
                + "  ".join(f"{k}={v:.3f}" for k, v in val_metrics.items() if k != "rank_1")
            )

            if wandb_run:
                wandb_run.log(val_metrics_log)

            # ── Checkpoint ────────────────────────────────────────────────
            is_best_rank = rank1 > best_rank1
            is_best_loss = loss < best_loss

            if is_best_rank:
                best_rank1 = rank1
                no_improve_validations = 0
            else:
                no_improve_validations += 1
            
            if is_best_loss:
                best_loss = loss
                no_improve_training = 0
            else:
                no_improve_training += 1

            save_checkpoint(
                checkpoint_dir = ckpt_dir,
                epoch          = epoch,
                model          = model,
                optimizer_adam = opt_adam,
                optimizer_sgd  = opt_sgd,
                scheduler_adam = sched_adam,
                scheduler_sgd  = sched_sgd,
                metrics        = {**train_metrics, **val_metrics_log},
                is_best        = is_best_rank,
                filename       = f"epoch_{epoch:04d}.pth",
            )

            if patience > 0 and no_improve_validations >= patience:
                log.info(
                    f"Early stopping triggered at epoch {epoch:03d}: "
                    f"no Rank-1 improvement for {no_improve_validations} validation(s)."
                )
                break

    log.info(f"Training complete.  Best Rank-1: {best_rank1:.3f}%")

    if wandb_run:
        wandb_run.summary["best_rank_1"] = best_rank1
        wandb_run.finish()

    return best_rank1


# ── Cross-validation runner ───────────────────────────────────────────────────

def run_cross_validation():
    """Run all 5 folds and report average Rank-1 (paper: 98.972%)."""
    fold_results = []
    for fold in range(config.NUM_FOLDS):
        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold + 1} / {config.NUM_FOLDS}")
        log.info(f"{'='*60}")
        best = train(fold=fold)
        fold_results.append(best)
        log.info(f"Fold {fold}: best Rank-1 = {best:.3f}%")

    avg = np.mean(fold_results)
    std = np.std(fold_results)
    log.info(f"\nCross-validation complete.")
    log.info(f"Rank-1 per fold: {[f'{r:.3f}' for r in fold_results]}")
    log.info(f"Average Rank-1:  {avg:.3f}% ± {std:.3f}%")
    return fold_results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DNNet (Dog Nose-Print Network)")
    parser.add_argument("--fold",  type=int,  default=0,     help="Fold index (0-4)")
    parser.add_argument("--cv",    action="store_true",       help="Run full 5-fold CV")
    parser.add_argument("--resume",type=str,  default=None,  help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if args.cv:
        run_cross_validation()
    else:
        train(fold=args.fold, resume_ckpt=args.resume)
