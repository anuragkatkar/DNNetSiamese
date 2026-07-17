"""
analyse.py  –  Post-training analysis for DNNet

Usage
-----
# Full analysis on both train and val splits for a fold
python analyse.py eval \
    --checkpoint checkpoints/fold_0/best_model.pth \
    --data_root  ./dataset \
    --fold       0 \
    --output_dir ./analysis

# Get embeddings for every image in a folder
python analyse.py embed \
    --checkpoint checkpoints/fold_0/best_model.pth \
    --image_dir  ./my_nose_images \
    --output_dir ./analysis
"""

import sys
import os
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from configs import config
from data.dataset import (
    build_kfold_loaders,
    DogNosePrintDataset,
    get_val_transforms,
    _scan_dataset,
)
from models.dnnet import build_model
from utils.checkpoint import load_checkpoint
from utils.evaluation import (
    extract_embeddings,
    compute_rank_k,
    compute_vr_at_far,
    compute_roc,
    compute_confusion,
    run_tsne,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ── Similarity matrix (kept exactly as in original code) ─────────────────────

def plot_similarity_matrix(embeddings, labels, label_to_name, save_path, title):
    """
    Cosine similarity matrix using one embedding per identity
    (first image of each identity, same logic as original code).
    """
    # One representative embedding per unique identity (in label order)
    seen      = set()
    uniq_embs = []
    uniq_names = []
    for emb, lbl in zip(embeddings, labels):
        if lbl not in seen:
            seen.add(lbl)
            uniq_embs.append(emb)
            uniq_names.append(label_to_name.get(int(lbl), str(lbl)))

    sim_matrix = cosine_similarity(np.array(uniq_embs))

    plt.figure(figsize=(12, 10))
    plt.imshow(sim_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Cosine Similarity')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(uniq_names)), uniq_names, rotation=90, fontsize=8)
    plt.yticks(range(len(uniq_names)), uniq_names, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved similarity matrix → {save_path}")


def plot_similarity_matrix_all(embeddings, labels, matrix_save_path, distribution_save_path, title):
    """
    Cosine similarity matrix using all embeddings per identity
    """

    sim_matrix = cosine_similarity(np.array(embeddings))

    plt.figure(figsize=(12, 10))
    plt.imshow(sim_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Cosine Similarity')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.tight_layout()
    plt.savefig(matrix_save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved similarity matrix → {matrix_save_path}")

    bins = np.linspace(0, 1.5, 100)
    same_similarities = []
    for i in range(len(labels)):
        for j in range(i):
            if labels[j] == labels[i]:
                same_similarities.append(float(sim_matrix[j][i]))
    same_distance = np.array([1]) - np.array(same_similarities)

    different_similarities = []
    for i in range(len(labels)):
        for j in range(i):
            if labels[j] != labels[i]:
                different_similarities.append(float(sim_matrix[j][i]))
    different_distance = np.array([1]) - np.array(different_similarities)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(same_distance, bins=bins, alpha=0.5, label='Same')
    ax.hist(different_distance, bins=bins, alpha=0.5, label='Different')

    major_positions = np.linspace(0, 1.5, 16)
    minor_positions = np.linspace(0, 1.5, 151)  # Skip major spots to keep it clean

    # 2. Tell the axis exactly where to place them
    ax.set_xticks(major_positions, minor=False)
    ax.set_xticks(minor_positions, minor=True)

    # 3. Apply your custom styling
    ax.tick_params(axis='x', which='major', length=10, width=2, labelsize=12)
    ax.tick_params(axis='x', which='minor', length=5, width=1, labelrotation=90)

    ax.axvline(x=0.15, color='red', linestyle=':', linewidth=1.5, label='Threshold')
    ax.text(x=0.16, y=270, s='Threshold', color='red', rotation=90, va='center')
    ax.text(x=0.15, y=-20, s='0.15', color='red', rotation=90, va='center', ha='center')

    ax.legend()
    plt.title("Similarity Distance Distribution\nSiamese Model")
    plt.tight_layout()
    plt.savefig(distribution_save_path, dpi=150, bbox_inches="tight")
    plt.close()

    log.info(f"    Highest Distance of two different IDs: {different_distance.max():.2f}")
    log.info(f"    Lowest Distance of two different IDs: {different_distance.min():.2f}")
    log.info(f"    Highest Distance of same IDs: {same_distance.max():.2f}")
    log.info(f"    Lowest Distance of same IDs: {same_distance.min():.2f}")
    log.info(f"  Saved similarity distribution → {distribution_save_path}")


# ── t-SNE + PCA side-by-side ─────────────────────────────────────────────────

def plot_tsne_pca(embeddings, labels, label_to_name, save_path, title_prefix):
    """
    Side-by-side t-SNE and PCA scatter plots with per-identity colours and
    annotated identity names.
    """
    str_labels = np.array([label_to_name.get(int(l), str(l)) for l in labels])
    unique_names = list(dict.fromkeys(str_labels))   # ordered unique
    n = len(unique_names)
    cmap = matplotlib.pyplot.get_cmap("tab20", min(n, 20))
    colour_map = {name: cmap(i % 20) for i, name in enumerate(unique_names)}

    perplexity = min(30, len(embeddings) - 1)
    proj_tsne  = TSNE(n_components=2, random_state=42,
                      perplexity=perplexity).fit_transform(embeddings)

    pca        = PCA(n_components=2, random_state=42)
    proj_pca   = pca.fit_transform(embeddings)
    var_exp    = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, proj, subtitle, xlabel, ylabel in [
        (axes[0], proj_tsne,
         f"{title_prefix} — t-SNE",
         "t-SNE Component 1", "t-SNE Component 2"),
        (axes[1], proj_pca,
         f"{title_prefix} — PCA  ({var_exp.sum():.1f}% var explained)",
         f"PC1 ({var_exp[0]:.1f}%)", f"PC2 ({var_exp[1]:.1f}%)"),
    ]:
        colours = [colour_map[name] for name in str_labels]
        ax.scatter(proj[:, 0], proj[:, 1],
                   c=colours, s=80, alpha=0.7, linewidths=0)
        for i, name in enumerate(str_labels):
            ax.annotate(name, (proj[i, 0], proj[i, 1]),
                        fontsize=7, alpha=0.65,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_title(subtitle, fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved t-SNE + PCA → {save_path}")


# ── ROC curve ─────────────────────────────────────────────────────────────────

def plot_roc(fpr, tpr, auc_score, save_path, title="ROC Curve"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogx(fpr, tpr, lw=2.5, color="steelblue",
                label=f"DNNet  (AUC = {auc_score:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.10, color="steelblue")
    ax.axhline(1.0, color="grey", lw=0.8, linestyle="--")
    ax.set_xlim([1e-4, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Acceptance Rate (FAR)", fontsize=12)
    ax.set_ylabel("True Acceptance Rate (TAR)",  fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    for target_far, colour in [(0.001, "tomato"), (0.0001, "darkorange")]:
        idx = np.searchsorted(fpr, target_far, side="right") - 1
        idx = max(0, min(idx, len(tpr) - 1))
        ax.axvline(target_far, color=colour, lw=1.2, linestyle=":",
                   label=f"FAR={target_far*100:.2f}%  TAR={tpr[idx]*100:.1f}%")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved ROC → {save_path}")


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion(true_lbl, pred_lbl, label_to_name, save_path,
                   title="Confusion Matrix", max_classes=60):
    classes = np.unique(true_lbl)
    if len(classes) > max_classes:
        log.warning(f"  {len(classes)} classes — showing first {max_classes}.")
        mask     = true_lbl < max_classes
        true_lbl = true_lbl[mask]
        pred_lbl = pred_lbl[mask]
        classes  = np.unique(true_lbl)

    cm  = sk_confusion_matrix(true_lbl, pred_lbl, labels=classes)
    acc = np.diag(cm).sum() / cm.sum() * 100

    sz  = max(7, len(classes) // 4)
    fig, ax = plt.subplots(figsize=(sz, sz))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title(f"{title}\nRank-1 Accuracy = {acc:.2f}%",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label",      fontsize=10)
    if len(classes) <= 30:
        tick_labels = [label_to_name.get(int(c), str(c)) for c in classes]
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax.set_yticklabels(tick_labels, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved confusion matrix → {save_path}")


# ── TensorBoard Embedding Projector export ───────────────────────────────────

def save_tsv_projector(
    embeddings:   np.ndarray,
    labels:       np.ndarray,
    paths:        list,
    label_to_name: dict,
    out_dir:      str,
):
    """
    Save embeddings and metadata in the two-file TSV format expected by
    TensorBoard Embedding Projector (https://projector.tensorflow.org):

      embeddings_projector.tsv  – (N × D) values, tab-separated, no header
      metadata_projector.tsv   – header row + one row per point

    Load in TensorBoard:
      tensorboard --logdir <out_dir>
    Or upload both files at https://projector.tensorflow.org.
    """
    os.makedirs(out_dir, exist_ok=True)
    emb_path  = os.path.join(out_dir, "embeddings_projector.tsv")
    meta_path = os.path.join(out_dir, "metadata_projector.tsv")

    # Embedding vectors — no header, values tab-separated, no quoting
    np.savetxt(emb_path, embeddings, delimiter="\t", fmt="%.6f")

    # Metadata — first row is column headers (triggers multi-column mode in
    # TensorBoard Projector), then one data row per embedding.
    # Written directly (no csv.writer) to guarantee no quoting is added.
    lines = ["label\tfilename"]
    for lbl, path in zip(labels, paths):
        name  = label_to_name.get(int(lbl), str(lbl))
        fname = Path(path).name if path else ""
        lines.append(f"{name}\t{fname}")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    log.info(f"  Saved TensorBoard embeddings TSV  → {emb_path}")
    log.info(f"  Saved TensorBoard metadata TSV   → {meta_path}")


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, num_classes: int, device: torch.device):
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]

    embedding_dim    = state["dnnet.attention.fc.weight"].shape[0]
    num_classes_ckpt = state["arcface_head.weight"].shape[0]

    _orig = config.EMBEDDING_DIM
    config.EMBEDDING_DIM = embedding_dim
    model = build_model(num_classes=num_classes_ckpt, cfg=config).to(device)
    config.EMBEDDING_DIM = _orig

    model.load_state_dict(state)
    model.eval()
    log.info(f"  Checkpoint: embedding_dim={embedding_dim}, num_classes={num_classes_ckpt}")
    return model


# ── Core analysis for one split ───────────────────────────────────────────────

def analyse_split(
    model,
    samples,          # [(path, label), ...]  for this split
    label_to_name,
    device,
    fold,
    split_name,       # "train" or "val"
    output_dir,
):
    """
    Runs all plots + metrics for a single split (train or val).
    Saves everything under output_dir/fold_{fold}/{split_name}/
    """
    out = os.path.join(output_dir, f"fold_{fold}", split_name)
    os.makedirs(out, exist_ok=True)

    # Build a plain DataLoader for this split
    ds = DogNosePrintDataset(samples, label_to_name,
                             transform=get_val_transforms())
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE,
                        shuffle=False, num_workers=0)

    # ── Embeddings ────────────────────────────────────────────────────────
    log.info(f"  [{split_name}] Extracting embeddings ({len(samples)} images) …")
    embeddings, labels = extract_embeddings(model, loader, device)

    np.savez(
        os.path.join(out, "embeddings.npz"),
        embeddings=embeddings, labels=labels,
    )

    # ── TensorBoard TSV export ────────────────────────────────────────────
    split_paths = [s[0] for s in samples]
    save_tsv_projector(embeddings, labels, split_paths, label_to_name, out)

    # ── Silhouette score ──────────────────────────────────────────────────
    if len(np.unique(labels)) > 1:
        sil = silhouette_score(embeddings, labels, metric="cosine")
        log.info(f"  [{split_name}] Silhouette score (cosine): {sil:.4f}")
    else:
        sil = float("nan")
        log.warning(f"  [{split_name}] Only one class — silhouette score skipped.")

    # ── Rank-K + VR@FAR ──────────────────────────────────────────────────
    rank_metrics = compute_rank_k(embeddings, labels, k_list=[1, 5])
    far_metrics  = compute_vr_at_far(embeddings, labels,
                                     far_thresholds=[0.001, 0.0001])
    log.info(f"  [{split_name}] ── Metrics ──")
    log.info(f"  [{split_name}]   silhouette       : {sil:.4f}")
    for k, v in {**rank_metrics, **far_metrics}.items():
        log.info(f"  [{split_name}]   {k:28s}: {v:.4f}%")

    # ── Similarity matrix ─────────────────────────────────────────────────
    plot_similarity_matrix(
        embeddings, labels, label_to_name,
        save_path = os.path.join(out, "similarity_matrix.png"),
        title     = f"Embedding Similarity Matrix — Fold {fold} [{split_name}]",
    )

    plot_similarity_matrix_all(
        embeddings, labels,
        matrix_save_path = os.path.join(out, "similarity_matrix_all.png"),
        distribution_save_path = os.path.join(out, "similarity_distribution.png"),
        title     = f"Embedding Similarity Matrix — Fold {fold} [{split_name}]",
    )

    # ── t-SNE + PCA ───────────────────────────────────────────────────────
    plot_tsne_pca(
        embeddings, labels, label_to_name,
        save_path    = os.path.join(out, "tsne_pca.png"),
        title_prefix = f"Fold {fold} [{split_name}]",
    )

    # ── ROC ───────────────────────────────────────────────────────────────
    fpr, tpr, auc_score = compute_roc(embeddings, labels)
    plot_roc(
        fpr, tpr, auc_score,
        save_path = os.path.join(out, "roc.png"),
        title     = f"ROC Curve — Fold {fold} [{split_name}]",
    )

    # ── Confusion matrix ──────────────────────────────────────────────────
    true_lbl, pred_lbl = compute_confusion(embeddings, labels)
    plot_confusion(
        true_lbl, pred_lbl, label_to_name,
        save_path = os.path.join(out, "confusion_matrix.png"),
        title     = f"Confusion Matrix — Fold {fold} [{split_name}]",
    )

    log.info(f"  [{split_name}] All plots saved to {out}/")
    return {
        "silhouette":  sil,
        **rank_metrics,
        **far_metrics,
    }


# ── Sub-command: eval ─────────────────────────────────────────────────────────

def cmd_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  Fold: {args.fold}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Reconstruct the exact train/val split for this fold ───────────────
    from sklearn.model_selection import KFold

    all_samples, label_to_name = _scan_dataset(args.data_root)
    num_classes = len(label_to_name)

    label_to_samples = {}
    for path, label in all_samples:
        label_to_samples.setdefault(label, []).append((path, label))

    classes      = sorted(label_to_samples.keys())
    class_array  = np.array(classes)
    kf           = KFold(n_splits=config.NUM_FOLDS, shuffle=True,
                         random_state=config.SEED)
    splits       = list(kf.split(class_array))
    train_idx, val_idx = splits[args.fold]

    train_classes = set(class_array[train_idx].tolist())
    val_classes   = set(class_array[val_idx].tolist())

    train_samples = [s for s in all_samples if s[1] in train_classes]
    val_samples   = [s for s in all_samples if s[1] in val_classes]

    log.info(f"Train: {len(train_samples)} images / {len(train_classes)} identities")
    log.info(f"Val:   {len(val_samples)}   images / {len(val_classes)}   identities")

    # ── Load model ────────────────────────────────────────────────────────
    log.info(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, num_classes, device)

    # ── Run analysis on both splits ───────────────────────────────────────
    log.info("\n── Training split ───────────────────────────────────────────")
    train_metrics = analyse_split(
        model, train_samples, label_to_name, device,
        fold=args.fold, split_name="train", output_dir=args.output_dir,
    )

    log.info("\n── Validation split ─────────────────────────────────────────")
    val_metrics = analyse_split(
        model, val_samples, label_to_name, device,
        fold=args.fold, split_name="val", output_dir=args.output_dir,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("\n══ Summary ══════════════════════════════════════════════════")
    log.info(f"  {'Metric':<30} {'Train':>10} {'Val':>10}")
    log.info(f"  {'─'*52}")
    all_keys = sorted(set(train_metrics) | set(val_metrics))
    for k in all_keys:
        tv = train_metrics.get(k, float("nan"))
        vv = val_metrics.get(k, float("nan"))
        log.info(f"  {k:<30} {tv:>10.4f} {vv:>10.4f}")
    log.info(f"\nAll outputs saved to: {args.output_dir}/fold_{args.fold}/")


# ── Sub-command: embed ────────────────────────────────────────────────────────

@torch.no_grad()
def cmd_embed(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    tfm        = get_val_transforms()
    image_dir  = Path(args.image_dir)

    subdirs = [d for d in image_dir.iterdir() if d.is_dir()]
    if subdirs:
        log.info("Detected ID-structured folder layout.")
        samples, label_to_name = _scan_dataset(str(image_dir))
        num_classes = len(label_to_name)
        paths  = [s[0] for s in samples]
        labels = [s[1] for s in samples]
    else:
        log.info("Detected flat folder layout.")
        paths  = sorted([str(f) for f in image_dir.iterdir()
                         if f.suffix.lower() in IMAGE_EXTS])
        labels        = [0] * len(paths)
        label_to_name = {0: "unknown"}
        num_classes   = 1

    if not paths:
        log.error(f"No images found in {image_dir}")
        return

    log.info(f"Found {len(paths)} images across {num_classes} "
             f"{'identity' if num_classes == 1 else 'identities'}.")
    model = load_model(args.checkpoint, num_classes, device)

    all_embeddings, all_labels, all_paths = [], [], []
    for path, label in zip(paths, labels):
        try:
            img    = Image.open(path).convert("RGB")
            tensor = tfm(img).unsqueeze(0).to(device)
            emb    = model.get_embedding(tensor)
            all_embeddings.append(emb.squeeze(0).cpu().numpy())
            all_labels.append(label)
            all_paths.append(path)
        except Exception as e:
            log.warning(f"  Skipping {path}: {e}")

    embeddings = np.stack(all_embeddings, axis=0)
    labels_arr = np.array(all_labels, dtype=np.int64)

    # Print table
    print(f"\n{'─'*90}")
    print(f"{'Index':<6}  {'ID':<20}  {'Image':<45}  {'Norm':>9}")
    print(f"{'─'*90}")
    for i, (path, lbl, emb) in enumerate(zip(all_paths, labels_arr, embeddings)):
        identity = label_to_name.get(int(lbl), str(lbl))
        print(f"{i:<6}  {identity:<20}  {Path(path).name:<45}  {np.linalg.norm(emb):>9.4f}")
    print(f"{'─'*90}\n")

    np.savez(os.path.join(args.output_dir, "embeddings.npz"),
             embeddings=embeddings, labels=labels_arr,
             paths=np.array(all_paths))
    log.info(f"Saved embeddings {embeddings.shape} → {args.output_dir}/embeddings.npz")

    save_tsv_projector(embeddings, labels_arr, all_paths, label_to_name, args.output_dir)

    if num_classes > 1 and len(np.unique(labels_arr)) > 1:
        sil = silhouette_score(embeddings, labels_arr, metric="cosine")
        log.info(f"Silhouette score (cosine): {sil:.4f}")

        plot_similarity_matrix(
            embeddings, labels_arr, label_to_name,
            save_path = os.path.join(args.output_dir, "similarity_matrix.png"),
            title     = "Embedding Similarity Matrix",
        )

        plot_similarity_matrix_all(
            embeddings, labels,
            matrix_save_path = os.path.join(args.output_dir, "similarity_matrix_all.png"),
            distribution_save_path = os.path.join(args.output_dir, "similarity_distribution.png"),
            title     = f"Embedding Similarity Matrix",
        )

        plot_tsne_pca(
            embeddings, labels_arr, label_to_name,
            save_path    = os.path.join(args.output_dir, "tsne_pca.png"),
            title_prefix = "Embedded Images",
        )
        fpr, tpr, auc_score = compute_roc(embeddings, labels_arr)
        plot_roc(fpr, tpr, auc_score,
                 save_path=os.path.join(args.output_dir, "roc.png"))
        true_lbl, pred_lbl = compute_confusion(embeddings, labels_arr)
        plot_confusion(true_lbl, pred_lbl, label_to_name,
                       save_path=os.path.join(args.output_dir, "confusion_matrix.png"))

    log.info(f"All outputs saved to: {args.output_dir}/")
    return embeddings, labels_arr, all_paths


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DNNet post-training analysis",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    ep = sub.add_parser("eval",
        help="Full analysis on train + val splits for a fold")
    ep.add_argument("--checkpoint", required=True)
    ep.add_argument("--data_root",  default=config.DATA_ROOT)
    ep.add_argument("--fold",       type=int, default=0)
    ep.add_argument("--output_dir", default="./analysis")

    ip = sub.add_parser("embed",
        help="Embed all images in a folder")
    ip.add_argument("--checkpoint", required=True)
    ip.add_argument("--image_dir",  required=True)
    ip.add_argument("--output_dir", default="./analysis")

    args = parser.parse_args()
    if args.mode == "eval":
        cmd_eval(args)
    elif args.mode == "embed":
        cmd_embed(args)