"""
Inference / Evaluation script for DNNet.

Usage examples:
  # Evaluate a trained checkpoint on a test folder
  python inference.py --checkpoint checkpoints/fold_0/best_model.pth \\
                      --data_root  ./dataset \\
                      --fold       0

  # Identify a single query image against a gallery
  python inference.py --checkpoint checkpoints/fold_0/best_model.pth \\
                      --gallery    ./dataset \\
                      --query      ./query_nose.jpg \\
                      --top_k      5
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from configs import config
from data.dataset import build_kfold_loaders, DogNosePrintDataset, get_val_transforms, _scan_dataset
from models.dnnet import build_model
from utils.evaluation import evaluate, extract_embeddings, pairwise_l2_distance
from utils.checkpoint import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def load_model_for_inference(
    checkpoint_path: str,
    num_classes:     int,
    device:          torch.device,
) -> torch.nn.Module:
    model = build_model(num_classes=num_classes, cfg=config).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    return model


# ── Evaluate on a fold ────────────────────────────────────────────────────────

def eval_fold(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, num_classes = build_kfold_loaders(
        root       = args.data_root,
        fold       = args.fold,
        batch_size = config.BATCH_SIZE,
        seed       = config.SEED,
    )

    model = load_model_for_inference(args.checkpoint, num_classes, device)

    metrics = evaluate(
        model, val_loader, device,
        rank_k         = [1, 5],
        far_thresholds = [0.001, 0.0001],
    )

    log.info("Evaluation results:")
    for k, v in metrics.items():
        log.info(f"  {k:30s}: {v:.4f}%")
    return metrics


# ── Single-query identification ───────────────────────────────────────────────

@torch.no_grad()
def identify_query(args):
    """
    Given a query nose-print image, find the top-K matching identities
    from a gallery folder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build gallery ──────────────────────────────────────────────────────
    samples, label_to_name = _scan_dataset(args.gallery)
    num_classes = len(label_to_name)
    gallery_ds  = DogNosePrintDataset(samples, label_to_name, transform=get_val_transforms())

    from torch.utils.data import DataLoader
    gallery_loader = DataLoader(gallery_ds, batch_size=config.BATCH_SIZE,
                                shuffle=False, num_workers=0)

    # ── Load model ─────────────────────────────────────────────────────────
    model = load_model_for_inference(args.checkpoint, num_classes, device)

    # ── Extract gallery embeddings ─────────────────────────────────────────
    log.info("Extracting gallery embeddings …")
    gallery_embs, gallery_labels = extract_embeddings(model, gallery_loader, device)

    # ── Embed query ────────────────────────────────────────────────────────
    tfm   = get_val_transforms()
    query = Image.open(args.query).convert("RGB")
    query_tensor = tfm(query).unsqueeze(0).to(device)
    query_emb    = model.get_embedding(query_tensor).cpu().numpy()  # (1, D)

    # ── Find nearest neighbours ────────────────────────────────────────────
    dists = np.linalg.norm(gallery_embs - query_emb, axis=1)       # (N,)
    top_k = min(args.top_k, len(dists))
    top_k_idx = np.argsort(dists)[:top_k]

    log.info(f"\nTop-{top_k} matches for query: {args.query}")
    log.info(f"{'Rank':<6} {'Dog ID':<20} {'Distance':<12} {'Image path'}")
    log.info("-" * 70)

    seen_ids = set()
    rank = 1
    for idx in top_k_idx:
        dog_id   = label_to_name[gallery_labels[idx]]
        distance = dists[idx]
        img_path = samples[idx][0]
        if dog_id not in seen_ids:
            log.info(f"{rank:<6} {dog_id:<20} {distance:<12.4f} {img_path}")
            seen_ids.add(dog_id)
            rank += 1
        if rank > top_k:
            break


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNNet Inference")
    sub = parser.add_subparsers(dest="mode")

    # eval subcommand
    eval_p = sub.add_parser("eval", help="Evaluate on a validation fold")
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--data_root",  default=config.DATA_ROOT)
    eval_p.add_argument("--fold",       type=int, default=0)

    # identify subcommand
    id_p = sub.add_parser("identify", help="Identify a single query image")
    id_p.add_argument("--checkpoint", required=True)
    id_p.add_argument("--gallery",    required=True, help="Gallery root directory")
    id_p.add_argument("--query",      required=True, help="Path to query nose-print image")
    id_p.add_argument("--top_k",      type=int, default=5)

    args = parser.parse_args()

    if args.mode == "eval":
        eval_fold(args)
    elif args.mode == "identify":
        identify_query(args)
    else:
        parser.print_help()
