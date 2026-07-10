"""
Evaluation utilities for DNNet.

Provides:
  - extract_embeddings   : runs the model over a DataLoader → embeddings + labels
  - compute_rank_k       : Rank-K identification accuracy
  - compute_vr_at_far    : Verification Rate at specified False Acceptance Rate
  - compute_roc          : ROC curve (FAR, TAR arrays)
  - compute_confusion    : build confusion matrix for Rank-1 predictions
  - run_tsne             : 2-D t-SNE projection of embeddings
  - evaluate             : master function that returns a metrics dict
"""

import numpy as np
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve, auc


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs the model's get_embedding() over all batches in *loader*.

    Returns:
        embeddings : (N, D) float32 numpy array
        labels     : (N,)   int64   numpy array
    """
    model.eval()
    all_embs, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        embs = model.get_embedding(imgs)          # (B, D)
        all_embs.append(embs.cpu().numpy())
        all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embs,   axis=0).astype(np.float32)
    labels     = np.concatenate(all_labels, axis=0).astype(np.int64)
    return embeddings, labels


# ── Distance matrix ───────────────────────────────────────────────────────────

def pairwise_l2_distance(embs: np.ndarray) -> np.ndarray:
    """Compute N×N pairwise Euclidean distance matrix."""
    # ||a-b||² = ||a||² + ||b||² - 2·a·bᵀ
    sq = np.sum(embs ** 2, axis=1, keepdims=True)   # (N,1)
    dist_sq = sq + sq.T - 2.0 * (embs @ embs.T)
    dist_sq = np.clip(dist_sq, 0.0, None)           # numerical safety
    return np.sqrt(dist_sq)                         # (N, N)


# ── Rank-K accuracy ───────────────────────────────────────────────────────────

def compute_rank_k(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    k_list:     List[int] = (1, 5),
) -> Dict[str, float]:
    """
    Gallery = all embeddings.  Each sample is queried against all others.
    The query's own slot is excluded from the ranking (leave-one-out).

    Returns dict: {'rank_1': float, 'rank_5': float, ...}
    """
    N = len(labels)
    dist = pairwise_l2_distance(embeddings)   # (N, N)

    # For each query, exclude itself and rank the rest
    np.fill_diagonal(dist, np.inf)

    results = {}
    for k in k_list:
        top_k_indices = np.argsort(dist, axis=1)[:, :k]  # (N, k)
        top_k_labels  = labels[top_k_indices]             # (N, k)
        correct = np.any(top_k_labels == labels[:, None], axis=1)
        results[f"rank_{k}"] = float(correct.mean()) * 100.0

    return results


# ── Verification metrics (VR@FAR, ROC) ───────────────────────────────────────

def _build_verification_pairs(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    max_pairs:  int = 50_000,
    seed:       int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample genuine (same-class) and impostor (diff-class) pairs.
    Returns distances (M,) and binary ground-truth (M,) [1=genuine, 0=impostor].

    We cap at max_pairs per type to keep evaluation tractable.
    """
    rng = np.random.default_rng(seed)
    N   = len(labels)

    # Build same-class pairs
    genuine_pairs, impostor_pairs = [], []
    for i in range(N):
        same = np.where(labels == labels[i])[0]
        diff = np.where(labels != labels[i])[0]
        same = same[same > i]   # upper triangle only
        if len(same):
            genuine_pairs.extend([(i, j) for j in same])
        # Only a few impostors per probe to keep balance
        if len(diff):
            chosen = rng.choice(diff, size=min(2, len(diff)), replace=False)
            impostor_pairs.extend([(i, j) for j in chosen])

    # Subsample if too large
    if len(genuine_pairs) > max_pairs:
        idx = rng.choice(len(genuine_pairs), max_pairs, replace=False)
        genuine_pairs = [genuine_pairs[i] for i in idx]
    if len(impostor_pairs) > max_pairs:
        idx = rng.choice(len(impostor_pairs), max_pairs, replace=False)
        impostor_pairs = [impostor_pairs[i] for i in idx]

    pairs  = genuine_pairs  + impostor_pairs
    gt     = [1] * len(genuine_pairs) + [0] * len(impostor_pairs)
    pairs  = np.array(pairs,  dtype=np.int64)
    gt     = np.array(gt,     dtype=np.int8)

    dist_mat = pairwise_l2_distance(embeddings)
    distances = dist_mat[pairs[:, 0], pairs[:, 1]]
    return distances, gt


def compute_vr_at_far(
    embeddings:      np.ndarray,
    labels:          np.ndarray,
    far_thresholds:  List[float] = (0.001, 0.0001),
) -> Dict[str, float]:
    """
    Compute Verification Rate (TAR) at specific False Acceptance Rates.

    Returns dict: {'vr_far_0.001': float, 'vr_far_0.0001': float, ...}
    """
    distances, gt = _build_verification_pairs(embeddings, labels)

    # sklearn roc_curve works on similarity (higher = more similar).
    # Distances are inverted: similarity = -distance.
    similarity = -distances
    fpr, tpr, _ = roc_curve(gt, similarity)

    results = {}
    for far in far_thresholds:
        # Find the largest TAR where FPR ≤ FAR
        idx = np.searchsorted(fpr, far, side='right') - 1
        idx = max(0, min(idx, len(tpr) - 1))
        key = f"vr_far_{far}"
        results[key] = float(tpr[idx]) * 100.0

    return results


def compute_roc(
    embeddings: np.ndarray,
    labels:     np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Returns fpr, tpr, auc_score."""
    distances, gt = _build_verification_pairs(embeddings, labels)
    similarity    = -distances
    fpr, tpr, _   = roc_curve(gt, similarity)
    auc_score     = auc(fpr, tpr)
    return fpr, tpr, auc_score


# ── Confusion matrix ──────────────────────────────────────────────────────────

def compute_confusion(
    embeddings: np.ndarray,
    labels:     np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (true_labels, pred_labels) for Rank-1 classification.
    Suitable for plotting with seaborn/matplotlib.
    """
    N    = len(labels)
    dist = pairwise_l2_distance(embeddings)
    np.fill_diagonal(dist, np.inf)
    pred_indices = np.argmin(dist, axis=1)   # nearest neighbour
    pred_labels  = labels[pred_indices]
    return labels, pred_labels


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def run_tsne(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    max_samples: int = 2000,
    seed:        int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2-D t-SNE projection (paper Figure 9).
    Subsamples to max_samples if the dataset is large.

    Returns tsne_2d (N, 2), labels (N,).
    """
    from sklearn.manifold import TSNE

    if len(embeddings) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]
        labels     = labels[idx]

    perplexity = min(30.0, max(2.0, len(embeddings) - 1))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)
    proj = tsne.fit_transform(embeddings)
    return proj, labels


# ── Master evaluate function ──────────────────────────────────────────────────

def evaluate(
    model,
    val_loader:    DataLoader,
    device:        torch.device,
    rank_k:        List[int]   = (1, 5),
    far_thresholds: List[float] = (0.001, 0.0001),
) -> Dict[str, float]:
    """
    Full evaluation pipeline.

    Returns a flat metrics dictionary ready for logging.
    """
    embeddings, labels = extract_embeddings(model, val_loader, device)

    metrics = {}
    metrics.update(compute_rank_k(embeddings, labels, k_list=rank_k))
    metrics.update(compute_vr_at_far(embeddings, labels, far_thresholds=far_thresholds))

    return metrics
