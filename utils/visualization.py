"""
Visualization utilities for DNNet evaluation.

Generates:
  - t-SNE embedding plot  (paper Figure 9)
  - ROC curve             (paper Figure 11)
  - Confusion matrix      (paper Figure 10)
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

log = logging.getLogger(__name__)


# ── t-SNE plot ────────────────────────────────────────────────────────────────

def plot_tsne(
    proj:     np.ndarray,       # (N, 2)  t-SNE coordinates
    labels:   np.ndarray,       # (N,)    class indices
    save_path: Optional[str] = None,
    title:    str = "t-SNE Embedding Space",
    max_classes: int = 50,
):
    unique_labels = np.unique(labels)
    n_classes     = len(unique_labels)

    # Use a large colour palette; truncate if needed
    cmap = plt.cm.get_cmap("tab20", min(n_classes, max_classes))
    colours = {lbl: cmap(i % max_classes) for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 10))
    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=[colours[lbl]], s=20, alpha=0.7, linewidths=0,
            label=str(lbl) if n_classes <= 20 else None,
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    if n_classes <= 20:
        ax.legend(fontsize=7, ncol=2, loc="best")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        log.info(f"t-SNE plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── ROC curve ─────────────────────────────────────────────────────────────────

def plot_roc(
    fpr:      np.ndarray,
    tpr:      np.ndarray,
    auc_score: float,
    label:    str = "DNNet (Ours)",
    save_path: Optional[str] = None,
    title:    str = "ROC Curve",
):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.semilogx(fpr, tpr, lw=2, label=f"{label} (AUC={auc_score:.4f})")
    ax.axhline(y=1.0, color="grey", linestyle="--", lw=0.8)
    ax.set_xlim([1e-4, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Acceptance Rate", fontsize=12)
    ax.set_ylabel("True Acceptance Rate",  fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        log.info(f"ROC curve saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    save_path:   Optional[str] = None,
    title:       str = "Confusion Matrix",
    max_classes: int = 60,
):
    from sklearn.metrics import confusion_matrix

    classes = np.unique(true_labels)
    if len(classes) > max_classes:
        log.warning(
            f"Too many classes ({len(classes)}) for confusion matrix. "
            f"Showing first {max_classes}."
        )
        mask    = true_labels < max_classes
        true_labels = true_labels[mask]
        pred_labels = pred_labels[mask]
        classes = np.unique(true_labels)

    cm = confusion_matrix(true_labels, pred_labels, labels=classes)

    fig_size = max(6, len(classes) // 4)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label",      fontsize=10)

    # Only show tick labels if classes fit
    if len(classes) <= 30:
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_yticklabels(classes, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        log.info(f"Confusion matrix saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Master viz runner ─────────────────────────────────────────────────────────

def generate_all_plots(
    model,
    val_loader,
    device,
    output_dir: str = "./plots",
    fold:       int = 0,
):
    """
    Convenience function: extract embeddings once and generate all three plots.
    """
    import os
    from utils.evaluation import (
        extract_embeddings, run_tsne, compute_roc, compute_confusion
    )

    os.makedirs(output_dir, exist_ok=True)

    log.info("Extracting embeddings for visualization …")
    embeddings, labels = extract_embeddings(model, val_loader, device)

    # ── t-SNE ─────────────────────────────────────────────────────────────
    log.info("Computing t-SNE …")
    proj, proj_labels = run_tsne(embeddings, labels)
    plot_tsne(
        proj, proj_labels,
        save_path = os.path.join(output_dir, f"tsne_fold{fold}.png"),
        title     = f"t-SNE – Fold {fold}",
    )

    # ── ROC ───────────────────────────────────────────────────────────────
    log.info("Computing ROC …")
    fpr, tpr, auc_score = compute_roc(embeddings, labels)
    plot_roc(
        fpr, tpr, auc_score,
        save_path = os.path.join(output_dir, f"roc_fold{fold}.png"),
        title     = f"ROC Curve – Fold {fold}",
    )

    # ── Confusion matrix ──────────────────────────────────────────────────
    log.info("Computing confusion matrix …")
    true_lbl, pred_lbl = compute_confusion(embeddings, labels)
    plot_confusion_matrix(
        true_lbl, pred_lbl,
        save_path = os.path.join(output_dir, f"confusion_fold{fold}.png"),
        title     = f"Confusion Matrix – Fold {fold}",
    )

    log.info(f"All plots saved to {output_dir}/")
