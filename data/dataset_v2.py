"""
Hard Negative Mining pair dataset for DNNetV2.

DNNetV2 replaces random negative sampling with a *hard negative selector*:
instead of picking any random different-class sample, it picks the most
similar (hardest) one based on current embedding distances.

Two strategies are provided:

1. HardNegativePairDataset (offline / semi-hard)
   - At the start of each epoch, a mining step embeds all training images
     and builds a per-anchor hard-negative index using cosine similarity.
   - During __getitem__ the hard negative is drawn from the pre-built index.
   - Fast per-batch, but requires one full forward pass per epoch to refresh.

2. The mining step is called from the training loop via:
       hard_dataset.refresh(model, device)
   Call this at the start of each epoch (or every N epochs).
"""

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .dataset import DogNosePrintDataset, get_val_transforms


class HardNegativePairDataset(Dataset):
    """
    Siamese pair dataset with hard negative mining.

    On each __getitem__:
      - Anchor and positive  : same as SiamesePairDataset (random positive)
      - Negative             : chosen from a pre-built hard-negative index
                               (the K most similar samples from different classes)

    Call refresh(model, device) at the start of each epoch to update
    the hard-negative index with current model embeddings.

    Args:
        base_dataset    : DogNosePrintDataset
        positive_ratio  : fraction of pairs that are positive (default 0.5)
        num_hard_neg    : how many hard negatives to keep per anchor (default 10)
                          The actual negative used is sampled from these K.
        mining_batch    : batch size for the embedding extraction pass
        warmup_epochs   : for the first N epochs use random negatives
                          (model not yet stable enough for meaningful mining)
    """

    def __init__(
        self,
        base_dataset:   DogNosePrintDataset,
        positive_ratio: float = 0.5,
        num_hard_neg:   int   = 10,
        mining_batch:   int   = 32,
        warmup_epochs:  int   = 5,
    ):
        self.base           = base_dataset
        self.positive_ratio = positive_ratio
        self.num_hard_neg   = num_hard_neg
        self.mining_batch   = mining_batch
        self.warmup_epochs  = warmup_epochs

        self._epoch          = 0
        # hard_neg_index[i] = list of dataset indices that are hard negatives for i
        self._hard_neg_index: Dict[int, List[int]] = {}

    # ── Public: call from training loop ──────────────────────────────────────

    def step_epoch(self):
        """Increment internal epoch counter. Call at the end of each epoch."""
        self._epoch += 1

    @torch.no_grad()
    def refresh(self, model, device: torch.device):
        """
        Re-embed all training images and rebuild the hard-negative index.
        Call this at the START of each epoch (after step_epoch for epoch > 0).

        If still in warmup, skips mining and uses random negatives instead.
        """
        if self._epoch < self.warmup_epochs:
            self._hard_neg_index = {}
            return

        model.eval()
        tfm    = get_val_transforms()
        loader = DataLoader(
            self.base,
            batch_size  = self.mining_batch,
            shuffle     = False,
            num_workers = 0,
        )

        # ── Extract all embeddings ────────────────────────────────────────
        all_embs   = []
        all_labels = []
        for imgs, labels in loader:
            imgs = imgs.to(device)
            # get_embedding returns normalised vectors
            emb = model.get_embedding(imgs).cpu()
            all_embs.append(emb)
            all_labels.append(labels)

        all_embs   = torch.cat(all_embs,   dim=0)   # (N, D)
        all_labels = torch.cat(all_labels, dim=0)   # (N,)

        # ── Build hard-negative index ─────────────────────────────────────
        # Cosine similarity matrix: (N, N)  (embeddings already normalised)
        sim = all_embs @ all_embs.T                 # (N, N)

        self._hard_neg_index = {}
        N = len(all_labels)
        for i in range(N):
            anchor_label = all_labels[i].item()

            # Mask out same-class samples and self
            diff_class_mask = (all_labels != anchor_label)   # (N,)
            diff_class_mask[i] = False

            diff_indices = torch.where(diff_class_mask)[0]  # indices of neg samples

            if len(diff_indices) == 0:
                self._hard_neg_index[i] = []
                continue

            # Sort by similarity descending → hardest negatives first
            neg_sims    = sim[i][diff_indices]
            sorted_idx  = torch.argsort(neg_sims, descending=True)
            top_k       = sorted_idx[: self.num_hard_neg]
            hard_neg_ds_indices = diff_indices[top_k].tolist()

            self._hard_neg_index[i] = hard_neg_ds_indices

        model.train()

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, anchor_idx: int):
        anchor_img, anchor_label = self.base[anchor_idx]

        is_positive = random.random() < self.positive_ratio

        if is_positive:
            pair_idx = self.base.get_random_positive(anchor_label, anchor_idx)
        else:
            pair_idx = self._get_hard_negative(anchor_idx, anchor_label)

        pair_img, pair_label = self.base[pair_idx]

        pair_label_bin = torch.tensor(
            1 if anchor_label == pair_label else 0, dtype=torch.long
        )

        return anchor_img, pair_img, anchor_label, pair_label, pair_label_bin

    def _get_hard_negative(self, anchor_idx: int, anchor_label: int) -> int:
        """Return a hard negative index, falling back to random if not mined yet."""
        hard_negs = self._hard_neg_index.get(anchor_idx, [])

        if hard_negs:
            return random.choice(hard_negs)

        # Fallback: random negative (used during warmup or if index missing)
        return self.base.get_random_negative(anchor_label)
