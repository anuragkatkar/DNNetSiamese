"""
Training loop for DNNetV2.

Differences from train.py (DNNet):
  - Uses DNNetV2Siamese instead of DNNetSiamese
  - Uses TotalLossV2 (MagFace + Focal) instead of TotalLoss (ArcFace)
  - Uses HardNegativePairDataset — hard negative index refreshed every epoch
  - Two-phase backbone training (freeze → unfreeze) enabled by default
  - MagFace scale is larger (64 vs 30) so SGD LR is adjusted accordingly

Usage
-----
# Train fold 0
python train_v2.py --fold 0

# Full 5-fold cross-validation
python train_v2.py --cv

# Resume
python train_v2.py --fold 0 --resume checkpoints_v2/fold_0/epoch_0049.pth
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
from torch.amp import GradScaler, autocast


ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from configs import config
from data.dataset import build_kfold_loaders, DogNosePrintDataset, get_train_transforms
from data.dataset_v2 import HardNegativePairDataset
from models.dnnet_v2 import build_model_v2
from models.losses_v2 import TotalLossV2
from utils.schedulers import get_dnnet_scheduler, sync_lambda_scheduler_param_groups
from utils.evaluation import evaluate
from utils.checkpoint import save_checkpoint, load_checkpoint


# ── V2-specific config overrides (no need to change config.py) ───────────────

V2_CHECKPOINT_DIR    = "./checkpoints_v2"
V2_PROJECT_NAME      = "DNNetV2-Pet-NosePrint"
FREEZE_BACKBONE_EPOCHS = getattr(config, "FREEZE_BACKBONE_EPOCHS", 30)
TINYVIT_MODEL          = getattr(config, "TINYVIT_MODEL", "tiny_vit_21m_224")
HARD_NEG_WARMUP        = 5     # epochs before hard mining starts
HARD_NEG_TOPK          = 10    # top-K hard negatives per anchor
MINING_REFRESH_EVERY   = 3     # refresh hard negative index every N epochs

# MagFace hyperparameters
MAGFACE_SCALE   = 64.0
MAGFACE_L_M     = 0.45
MAGFACE_U_M     = 0.80
MAGFACE_L_A     = 5.0
MAGFACE_U_A     = 150.0
MAGFACE_LAMBDA  = 0.01
FOCAL_GAMMA     = 2.0


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # allow benchmark
    torch.backends.cudnn.benchmark     = True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Single training epoch ─────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    criterion: TotalLossV2,
    opt_adam,
    opt_sgd,
    device,
    epoch: int,
    scaler,
    wandb_run=None,
) -> dict:
    model.train()

    total_loss = con_loss = mag_a_loss = mag_p_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (anchor_img, pair_img, anchor_cls, pair_cls, pair_bin) in enumerate(loader):
        anchor_img = anchor_img.to(device, non_blocking=True)
        pair_img   = pair_img.to(device,   non_blocking=True)
        anchor_cls = anchor_cls.to(device, non_blocking=True)
        pair_cls   = pair_cls.to(device,   non_blocking=True)
        pair_bin   = pair_bin.to(device,   non_blocking=True)

        # ── Forward ───────────────────────────────────────────────────────
        with autocast():
            (anchor_emb, pair_emb,
             anchor_cos, pair_cos,
             anchor_mag, pair_mag) = model(anchor_img, pair_img)

        # ── Loss ──────────────────────────────────────────────────────────
            l_total, l_con, l_mag_a, l_mag_p = criterion(
                anchor_emb    = anchor_emb,
                pair_emb      = pair_emb,
                pair_labels   = pair_bin,
                anchor_cosine = anchor_cos,
                pair_cosine   = pair_cos,
                anchor_mag    = anchor_mag,
                pair_mag      = pair_mag,
                anchor_class  = anchor_cls,
                pair_class    = pair_cls,
            )

        # ── Backward ──────────────────────────────────────────────────────
        opt_adam.zero_grad()
        opt_sgd.zero_grad()
        scaler.scale(l_total).backward()
        scaler.unscale_(opt_adam)
        scaler.unscale_(opt_sgd)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(opt_adam)
        scaler.step(opt_sgd)
        scaler.update()

        total_loss += l_total.item()
        con_loss   += l_con.item()
        mag_a_loss += l_mag_a.item()
        mag_p_loss += l_mag_p.item()

        if (batch_idx + 1) % max(1, n_batches // 5) == 0:
            log.info(
                f"  Epoch {epoch:03d} [{batch_idx+1:4d}/{n_batches}] "
                f"loss={l_total.item():.4f}  con={l_con.item():.4f}  "
                f"mag_a={l_mag_a.item():.4f}  mag_p={l_mag_p.item():.4f}"
            )
    print(f"  mag range: [{anchor_mag.min():.2f}, {anchor_mag.max():.2f}]  mean={anchor_mag.mean():.2f}")
    
    avg = lambda x: x / n_batches
    metrics = {
        "train/loss_total":    avg(total_loss),
        "train/loss_con":      avg(con_loss),
        "train/loss_mag_anc":  avg(mag_a_loss),
        "train/loss_mag_pair": avg(mag_p_loss),
        "train/lr_adam":       opt_adam.param_groups[0]["lr"],
        "train/lr_sgd":        opt_sgd.param_groups[0]["lr"],
        "epoch":               epoch,
    }

    if wandb_run:
        wandb_run.log(metrics)

    return metrics


# ── Main training function ────────────────────────────────────────────────────

def train(fold: int = 0, resume_ckpt: Optional[str] = None):
    set_seed(config.SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  DNNetV2  |  Fold: {fold}")

    # ── Data ──────────────────────────────────────────────────────────────
    log.info(f"Loading dataset from: {config.DATA_ROOT}")
    train_loader, val_loader, num_classes = build_kfold_loaders(
        root        = config.DATA_ROOT,
        fold        = fold,
        num_folds   = config.NUM_FOLDS,
        batch_size  = config.BATCH_SIZE,
        num_workers = 2,
        seed        = config.SEED,
    )
    log.info(f"Number of classes: {num_classes}")

    # ── Build hard-negative dataset (wraps the train base dataset) ────────
    # Extract the base DogNosePrintDataset from the siamese loader
    train_base: DogNosePrintDataset = train_loader.dataset.base
    train_base_with_train_transform = DogNosePrintDataset(
        samples        = train_base.samples,
        label_to_name  = train_base.label_to_name,
        transform      = get_train_transforms(),
    )
    hard_dataset = HardNegativePairDataset(
        base_dataset   = train_base_with_train_transform,
        positive_ratio = 0.5,
        num_hard_neg   = HARD_NEG_TOPK,
        warmup_epochs  = HARD_NEG_WARMUP,
    )
    hard_loader = DataLoader(
        hard_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        num_workers = 2,
        drop_last   = True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    log.info(f"Building DNNetV2 with {TINYVIT_MODEL} backbone …")
    model = build_model_v2(num_classes=num_classes, cfg=config).to(device)

    scaler = GradScaler()

    # Phase 1: freeze backbone
    if FREEZE_BACKBONE_EPOCHS > 0:
        model.freeze_backbone()
        log.info(f"Backbone frozen for first {FREEZE_BACKBONE_EPOCHS} epochs.")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {total_params:,} total / {trainable_params:,} trainable")

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = TotalLossV2(
        contrastive_margin = config.CONTRASTIVE_MARGIN,
        magface_scale      = MAGFACE_SCALE,
        l_m                = MAGFACE_L_M,
        u_m                = MAGFACE_U_M,
        l_a                = MAGFACE_L_A,
        u_a                = MAGFACE_U_A,
        lambda_g           = MAGFACE_LAMBDA,
        focal_gamma        = FOCAL_GAMMA,
    )

    # ── Optimisers ────────────────────────────────────────────────────────
    # Adam for everything except MagFace head (only trainable params)
    feat_params = [
        p for n, p in model.named_parameters()
        if "magface_head" not in n and p.requires_grad
    ]
    mag_params = list(model.magface_head.parameters())

    opt_adam = torch.optim.Adam(
        feat_params,
        lr    = config.LR_CONTRASTIVE,
        betas = (config.ADAM_BETA1, config.ADAM_BETA2),
    )
    opt_sgd = torch.optim.SGD(
        mag_params,
        lr           = config.LR_ARCFACE,
        momentum     = config.SGD_MOMENTUM,
        weight_decay = config.SGD_WEIGHT_DECAY,
    )

    sched_adam = get_dnnet_scheduler(opt_adam, config.LR_WARMUP_EPOCHS, config.NUM_EPOCHS)
    sched_sgd  = get_dnnet_scheduler(opt_sgd,  config.LR_WARMUP_EPOCHS, config.NUM_EPOCHS)

    # ── W&B ───────────────────────────────────────────────────────────────
    wandb_run = None
    if config.USE_WANDB:
        try:
            import wandb
            wandb_run = wandb.init(
                project = V2_PROJECT_NAME,
                name    = f"v2_fold_{fold}",
                config  = {
                    "tinyvit_model":      TINYVIT_MODEL,
                    "embedding_dim":      config.EMBEDDING_DIM,
                    "freeze_epochs":      FREEZE_BACKBONE_EPOCHS,
                    "hard_neg_topk":      HARD_NEG_TOPK,
                    "hard_neg_warmup":    HARD_NEG_WARMUP,
                    "magface_scale":      MAGFACE_SCALE,
                    "focal_gamma":        FOCAL_GAMMA,
                    "batch_size":         config.BATCH_SIZE,
                    "num_epochs":         config.NUM_EPOCHS,
                    "fold":               fold,
                },
            )
        except Exception as e:
            log.warning(f"W&B init failed ({e}). Continuing without logging.")

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_rank1  = 0.0

    if resume_ckpt:
        ckpt = load_checkpoint(
            resume_ckpt, model, opt_adam, opt_sgd, sched_adam, sched_sgd, device
        )
        start_epoch = ckpt["epoch"] + 1
        best_rank1  = ckpt["metrics"].get("val/rank_1", 0.0)
        log.info(f"Resuming from epoch {start_epoch}  (best rank-1: {best_rank1:.3f}%)")

    # ── Training loop ─────────────────────────────────────────────────────
    ckpt_dir = os.path.join(V2_CHECKPOINT_DIR, f"fold_{fold}")
    log.info(f"Starting training for {config.NUM_EPOCHS} epochs …")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        t0 = time.time()

        # ── Phase 2: unfreeze backbone ────────────────────────────────────
        if epoch == FREEZE_BACKBONE_EPOCHS and FREEZE_BACKBONE_EPOCHS > 0:
            log.info(f"Epoch {epoch}: unfreezing backbone for end-to-end fine-tuning …")
            model.unfreeze_backbone()
            backbone_params = [
                p for p in model.dnnet_v2.backbone.parameters()
            ]
            opt_adam.add_param_group({
                "params": backbone_params,
                "lr":     config.LR_CONTRASTIVE * 0.1,   # lower LR for pretrained
            })
            sync_lambda_scheduler_param_groups(sched_adam)
            log.info(f"  Added backbone to Adam at LR={config.LR_CONTRASTIVE * 0.1:.1e}")

        # ── Hard negative mining refresh ──────────────────────────────────
        if epoch % MINING_REFRESH_EVERY == 0:
            hard_dataset.refresh(model, device)
            log.info(
                f"  Hard negative index refreshed "
                f"({'warmup – random negatives' if epoch < HARD_NEG_WARMUP else 'hard negatives active'})"
            )

        # ── Train ─────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, hard_loader, criterion,
            opt_adam, opt_sgd, device, epoch, scaler, wandb_run,
        )

        sched_adam.step()
        sched_sgd.step()
        hard_dataset.step_epoch()

        # ── Validate ──────────────────────────────────────────────────────
        val_metrics = {}
        if (epoch + 1) % 5 == 0 or epoch == config.NUM_EPOCHS - 1:
            val_metrics = evaluate(
                model, val_loader, device,
                rank_k         = config.RANK_K,
                far_thresholds = config.FAR_THRESHOLDS,
            )
            val_metrics_log = {f"val/{k}": v for k, v in val_metrics.items()}
            val_metrics_log["epoch"] = epoch

            rank1 = val_metrics.get("rank_1", 0.0)
            elapsed = time.time() - t0
            log.info(
                f"Epoch {epoch:03d}/{config.NUM_EPOCHS-1}  [{elapsed:.1f}s]  "
                f"rank-1={rank1:.3f}%  "
                + "  ".join(f"{k}={v:.3f}" for k, v in val_metrics.items() if k != "rank_1")
            )

            if wandb_run:
                wandb_run.log(val_metrics_log)

            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1

            save_checkpoint(
                checkpoint_dir = ckpt_dir,
                epoch          = epoch,
                model          = model,
                optimizer_adam = opt_adam,
                optimizer_sgd  = opt_sgd,
                scheduler_adam = sched_adam,
                scheduler_sgd  = sched_sgd,
                metrics        = {**train_metrics, **val_metrics_log},
                is_best        = is_best,
                filename       = f"epoch_{epoch:04d}.pth",
            )

    log.info(f"Training complete.  Best Rank-1: {best_rank1:.3f}%")
    if wandb_run:
        wandb_run.summary["best_rank_1"] = best_rank1
        wandb_run.finish()

    return best_rank1


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cross_validation():
    results = []
    for fold in range(config.NUM_FOLDS):
        log.info(f"\n{'='*60}\nFOLD {fold + 1} / {config.NUM_FOLDS}\n{'='*60}")
        best = train(fold=fold)
        results.append(best)
        log.info(f"Fold {fold}: best Rank-1 = {best:.3f}%")

    avg = np.mean(results)
    std = np.std(results)
    log.info(f"\nCross-validation complete.")
    log.info(f"Rank-1 per fold : {[f'{r:.3f}' for r in results]}")
    log.info(f"Average Rank-1  : {avg:.3f}% ± {std:.3f}%")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DNNetV2")
    parser.add_argument("--fold",   type=int, default=0)
    parser.add_argument("--cv",     action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.cv:
        run_cross_validation()
    else:
        train(fold=args.fold, resume_ckpt=args.resume)
