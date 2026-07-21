"""
Loss functions for DNNet.

1. ContrastiveLoss  – eq. (7) in the paper
   L_con(i, x1, x2) = (1-i)[max(0, m-d)]² + i·d²
   where i=1 for positive pairs, i=0 for negatives, d = Euclidean distance.

2. ArcFaceLoss      – eq. (8) in the paper  (Deng et al., CVPR 2019)
   L_arc = -1/N Σ log [ e^{s·cos(θ_yi + m)} /
                        (e^{s·cos(θ_yi + m)} + Σ_{j≠yi} e^{s·cos θ_j}) ]

3. TotalLoss        – eq. (9)
   L_total = L_con + ½(L_arc(anchor) + L_arc(pair))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss (eq. 7).

    Args:
        margin: m in the paper (default 2.0)
    """

    def __init__(self, margin: float = 2.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1: torch.Tensor,   # anchor embeddings    (B, D)
        emb2: torch.Tensor,   # pair   embeddings    (B, D)
        labels: torch.Tensor, # binary labels        (B,)  1=positive 0=negative
    ) -> torch.Tensor:
        # Euclidean distance (paper uses l2 distance, not squared)
        d = F.pairwise_distance(emb1, emb2, p=2)   # (B,)

        labels = labels.float()

        # Positive pair loss: i·d²
        pos_loss = labels * d.pow(2)

        # Negative pair loss: (1-i)·[max(0, m-d)]²
        neg_loss = (1 - labels) * F.relu(self.margin - d).pow(2)

        loss = torch.mean(pos_loss + neg_loss)
        return loss


class ArcFaceLinear(nn.Module):
    """
    ArcFace projection layer (eq. 8).

    Maintains a learnable weight matrix W ∈ R^{num_classes × embedding_dim}.
    The forward pass returns the raw cosine logits *before* applying the
    angular margin – the margin is added inside ArcFaceLoss.forward.

    This is kept as a separate nn.Module so its parameters can be optimised
    with a different optimiser (SGD) as specified in the paper.
    """

    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return normalised cosine logits (B, num_classes)."""
        # L2-normalise both embeddings and weight rows
        norm_emb = F.normalize(embeddings, p=2, dim=1)
        norm_w   = F.normalize(self.weight, p=2, dim=1)
        cosine   = F.linear(norm_emb, norm_w)          # (B, num_classes)
        return cosine


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss (eq. 8).

    Args:
        scale  : s in the paper (default 30.0)
        margin : m in the paper, angular margin penalty (default 0.5 rad)
    """

    def __init__(self, scale: float = 30.0, margin: float = 0.5):
        super().__init__()
        self.scale  = scale
        self.margin = margin
        self.ce     = nn.CrossEntropyLoss()

    def forward(
        self,
        cosine_logits: torch.Tensor,  # (B, num_classes) from ArcFaceLinear
        labels: torch.Tensor,          # (B,) class indices (long)
    ) -> torch.Tensor:
        # ── Add angular margin to the target class angle ──────────────────
        # cos(θ_yi + m) = cos θ_yi · cos m  − sin θ_yi · sin m
        cos_theta = cosine_logits.float().clamp(-1 + 1e-4, 1 - 1e-4)
        theta        = torch.acos(cos_theta)
        cos_m        = math.cos(self.margin)
        sin_m        = math.sin(self.margin)
        sin_theta    = torch.sqrt(1.0 - cos_theta.pow(2))

        # Modified cosine for target class
        cos_theta_m  = cos_theta * cos_m - sin_theta * sin_m

        # Only apply margin to the ground-truth class
        one_hot = torch.zeros_like(cosine_logits)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta

        # Scale and cross-entropy
        output = output * self.scale
        loss   = self.ce(output, labels)
        return loss


class TotalLoss(nn.Module):
    """
    Combined loss (eq. 9):
        L_total = L_con + ½(L_arc(anchor) + L_arc(pair))
    """

    def __init__(
        self,
        contrastive_margin: float = 2.0,
        arcface_scale:      float = 30.0,
        arcface_margin:     float = 0.5,
    ):
        super().__init__()
        self.contrastive = ContrastiveLoss(margin=contrastive_margin)
        self.arcface     = ArcFaceLoss(scale=arcface_scale, margin=arcface_margin)

    def forward(
        self,
        anchor_emb:    torch.Tensor,   # (B, D)
        pair_emb:      torch.Tensor,   # (B, D)
        pair_labels:   torch.Tensor,   # (B,)  binary: 1=same, 0=diff
        anchor_logits: torch.Tensor,   # (B, num_classes) cosine logits for anchor
        pair_logits:   torch.Tensor,   # (B, num_classes) cosine logits for pair
        anchor_class:  torch.Tensor,   # (B,) true class index for anchor
        pair_class:    torch.Tensor,   # (B,) true class index for pair
    ) -> tuple:
        l_con = self.contrastive(anchor_emb, pair_emb, pair_labels)
        l_arc_anchor = self.arcface(anchor_logits, anchor_class)
        l_arc_pair   = self.arcface(pair_logits,   pair_class)
        l_total      = 0.5 * l_con + 0.5 * (l_arc_anchor + l_arc_pair)
        return l_total, l_con, l_arc_anchor, l_arc_pair
