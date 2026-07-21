"""
Dataset utilities for DNNet.

Provides:
  - DogNosePrintDataset  : loads images + labels, applies transforms
  - SiamesePairDataset   : wraps the above to produce (anchor, positive/negative, label) triplets
  - build_kfold_loaders  : returns train/val DataLoaders for a given fold with a fixed seed
"""

import os
import csv
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config

# Add this near the top of dataset.py, at module level

def _worker_init_fn(worker_id):
    np.random.seed(config.SEED + worker_id)

# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transforms(image_size: int = config.IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = config.IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ── Base Dataset ──────────────────────────────────────────────────────────────

class DogNosePrintDataset(Dataset):
    """
    Flat image dataset.
    root/
      dog_001/  img1.jpg img2.jpg ...
      dog_002/  ...
    Each class (dog identity) must have at least 2 images to allow positive
    pair creation.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],   # [(path, label), ...]
        label_to_name: Dict[int, str],
        transform=None,
    ):
        self.samples       = samples
        self.label_to_name = label_to_name
        self.transform     = transform

        # Build an index: label -> [indices in self.samples]
        self.label_to_indices: Dict[int, List[int]] = {}
        for idx, (_, label) in enumerate(samples):
            self.label_to_indices.setdefault(label, []).append(idx)

        self.num_classes = len(label_to_name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    # ── helpers ───────────────────────────────────────────────────────────────
    def get_random_positive(self, label: int, exclude_idx: int) -> int:
        """Return a random index from the same class, different from exclude_idx."""
        candidates = [i for i in self.label_to_indices[label] if i != exclude_idx]
        if not candidates:
            return exclude_idx          # fallback – only one sample for this class
        return random.choice(candidates)

    def get_random_negative(self, label: int) -> int:
        """Return a random index from a *different* class."""
        neg_label = random.choice(
            [l for l in self.label_to_indices if l != label]
        )
        return random.choice(self.label_to_indices[neg_label])


# ── Siamese Pair Dataset ───────────────────────────────────────────────────────

class SiamesePairDataset(Dataset):
    """
    On every __getitem__ call, returns:
        anchor_img, pair_img, is_same_label (int 1=positive, 0=negative)

    Pairs are generated dynamically (online) so that each epoch sees
    different pairs – this matches the paper's use of contrastive loss.
    """

    def __init__(self, base_dataset: DogNosePrintDataset, positive_ratio: float = 0.5):
        self.base            = base_dataset
        self.positive_ratio  = positive_ratio

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, anchor_idx: int):
        anchor_img, anchor_label = self.base[anchor_idx]

        is_positive = random.random() < self.positive_ratio

        if is_positive:
            pair_idx = self.base.get_random_positive(anchor_label, anchor_idx)
        else:
            pair_idx = self.base.get_random_negative(anchor_label)

        pair_img, pair_label = self.base[pair_idx]

        # Binary label: i=1 → positive pair, i=0 → negative pair  (paper eq. 7)
        pair_label_bin = torch.tensor(
            1 if anchor_label == pair_label else 0, dtype=torch.long
        )

        return anchor_img, pair_img, anchor_label, pair_label, pair_label_bin


# ── Dataset Builder / K-Fold Split ────────────────────────────────────────────

def _scan_dataset(root: str) -> Tuple[List[Tuple[str, int]], Dict[int, str]]:
    """
    Walk *root* and collect (image_path, class_idx) pairs.
    Skips classes with fewer than 2 images (can't form a positive pair).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if len(class_dirs) == 0:
        raise RuntimeError(f"No class subdirectories found under {root}")

    samples: List[Tuple[str, int]] = []
    label_to_name: Dict[int, str] = {}
    label_idx = 0

    for class_dir in class_dirs:
        imgs = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTS
        ]
        if len(imgs) < 2:
            print(f"  [SKIP] {class_dir.name}: only {len(imgs)} image(s), need ≥2")
            continue
        for img_path in sorted(imgs):
            samples.append((str(img_path), label_idx))
        label_to_name[label_idx] = class_dir.name
        label_idx += 1

    print(f"Loaded {len(samples)} images across {label_idx} dog identities.")
    return samples, label_to_name

def save_split_csv(
    train_samples: list,
    val_samples:   list,
    label_to_name: dict,
    fold:          int,
    output_dir:    str = "./checkpoints",
):
    """
    Saves two CSVs for a given fold:
      fold_0_train.csv  — all images used for training
      fold_0_val.csv    — all images used for validation

    Columns: dog_id (folder name), image_path
    """
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        csv_path = os.path.join(output_dir, f"fold_{fold}_{split_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dog_id", "image_path"])
            for path, label in split_samples:
                writer.writerow([label_to_name[label], path])
        print(f"  Saved {split_name} split ({len(split_samples)} images) → {csv_path}")

def build_kfold_loaders(
    root: str        = config.DATA_ROOT,
    fold: int        = config.VAL_FOLD,
    num_folds: int   = config.NUM_FOLDS,
    batch_size: int  = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    seed: int        = config.SEED,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Builds train and validation DataLoaders for a specific fold.

    The split is done at the *class* level (all images of a dog go entirely
    into train or val) to prevent data leakage.  A fixed seed ensures the
    same split on every run.

    Returns:
        train_loader, val_loader, num_classes
    """
    samples, label_to_name = _scan_dataset(root)
    num_classes = len(label_to_name)

    # ── Group samples by class ─────────────────────────────────────────────
    label_to_samples: Dict[int, List[Tuple[str, int]]] = {}
    for path, label in samples:
        label_to_samples.setdefault(label, []).append((path, label))

    classes     = sorted(label_to_samples.keys())
    class_array = np.array(classes)

    # ── KFold on class IDs (we want equal numbers of dogs in each split) ─
    # StratifiedKFold needs ≥ n_splits members per class; since each class
    # is itself a unique entity here, we use plain KFold instead.
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(class_array))
    train_class_idx, val_class_idx = splits[fold]

    train_classes = set(class_array[train_class_idx].tolist())
    val_classes   = set(class_array[val_class_idx].tolist())

    train_samples = [s for s in samples if s[1] in train_classes]
    val_samples   = [s for s in samples if s[1] in val_classes]

    save_split_csv(train_samples, val_samples, label_to_name, fold, output_dir=os.path.join("checkpoints", f"fold_{fold}"))
    
    print(
        f"Fold {fold}/{num_folds-1} | "
        f"Train: {len(train_samples)} imgs / {len(train_classes)} dogs | "
        f"Val:   {len(val_samples)} imgs / {len(val_classes)} dogs"
    )

    # ── Build Dataset objects ──────────────────────────────────────────────
    train_base = DogNosePrintDataset(
        train_samples, label_to_name, transform=get_train_transforms()
    )
    val_base = DogNosePrintDataset(
        val_samples, label_to_name, transform=get_val_transforms()
    )

    train_siamese = SiamesePairDataset(train_base, positive_ratio=0.5)

    # Val dataset: plain (no pairing needed; we do gallery-probe matching)
    # We keep val as a plain dataset for embedding extraction during evaluation

    train_loader = DataLoader(
        train_siamese,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn,   # ← named function, not lambda
    )
    val_loader = DataLoader(
        val_base,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, num_classes
