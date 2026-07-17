"""
Loss functions for DNNetV2.

1. MagFaceLinear   – classifier head (same weight structure as ArcFaceLinear)
2. MagFaceLoss     – adaptive margin based on embedding magnitude
                     (Meng et al., CVPR 2021)
3. FocalLoss       – down-weights easy examples, focuses on hard ones
                     (Lin et al., ICCV 2017)
4. MagFaceFocalLoss – MagFace loss with Focal Loss applied on top
5. TotalLossV2     – L_con + ½(L_mag(anchor) + L_mag(pair))

MagFace key idea
----------------
ArcFace uses a *fixed* angular margin m for every sample.
MagFace makes the margin *adaptive*:
  - High magnitude embedding (model is confident) → larger margin (harder constraint)
  - Low  magnitude embedding (model is uncertain)  → smaller margin (easier constraint)
  - Additionally a regularisation term pulls magnitudes into a valid range [l_a, u_a]

  m(a) = l_m + (u_m - l_m) / (u_a - l_a) * (a - l_a)

  where a = ||embedding||, [l_a, u_a] is the valid magnitude range,
  and [l_m, u_m] is the margin range.

Focal Loss on MagFace logits
-----------------------------
After applying the adaptive margin, the scaled logits are passed through
Focal Loss instead of plain CrossEntropyLoss. This down-weights identities
the model already classifies correctly, focusing training budget on hard ones.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── MagFace classifier head ───────────────────────────────────────────────────

class MagFaceLinear(nn.Module):
    """
    Classifier head for MagFace.
    Identical structure to ArcFaceLinear — a normalised weight matrix.
    Returns raw (un-normalised) embeddings AND cosine logits so that
    MagFaceLoss can access the embedding magnitude.
    """

    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor):
        """
        Args:
            embeddings: (B, D)  raw (NOT L2-normalised) embeddings

        Returns:
            cosine  : (B, num_classes)  cosine similarity logits
            mag     : (B,)              embedding magnitude ||x||
        """
        mag    = torch.norm(embeddings, p=2, dim=1)            # (B,)
        norm_e = F.normalize(embeddings, p=2, dim=1)           # (B, D)
        norm_w = F.normalize(self.weight,   p=2, dim=1)        # (C, D)
        cosine = F.linear(norm_e, norm_w)                      # (B, C)
        return cosine, mag


# ── MagFace loss ──────────────────────────────────────────────────────────────

class MagFaceLoss(nn.Module):
    """
    MagFace loss (Meng et al., CVPR 2021).

    Hyperparameters (paper defaults):
        scale   : feature scale s (default 64)
        l_m     : lower margin bound  (default 0.45)
        u_m     : upper margin bound  (default 0.8)
        l_a     : lower magnitude bound (default 10)
        u_a     : upper magnitude bound (default 110)
        lambda_g: weight of magnitude regularisation (default 35)
    """

    def __init__(
        self,
        scale:    float = 64.0,
        l_m:      float = 0.45,
        u_m:      float = 0.80,
        l_a:      float = 10.0,
        u_a:      float = 110.0,
        lambda_g: float = 35.0,
    ):
        super().__init__()
        self.scale    = scale
        self.l_m      = l_m
        self.u_m      = u_m
        self.l_a      = l_a
        self.u_a      = u_a
        self.lambda_g = lambda_g

    def _adaptive_margin(self, mag: torch.Tensor) -> torch.Tensor:
        """Compute per-sample margin from embedding magnitude."""
        mag   = mag.clamp(self.l_a, self.u_a)
        slope = (self.u_m - self.l_m) / (self.u_a - self.l_a)
        return self.l_m + slope * (mag - self.l_a)              # (B,)

    def _magnitude_regulariser(self, mag: torch.Tensor) -> torch.Tensor:
        """
        Pull magnitudes into [l_a, u_a].
        g(a) = (1/u_a) * a + (1/a)   — convex, minimum at a = sqrt(u_a)
        Regularisation: lambda_g * mean(g(a))
        """
        mag   = mag.clamp(self.l_a, self.u_a)
        g     = (1.0 / self.u_a) * mag + 1.0 / mag
        return self.lambda_g * g.mean()

    def forward(
        self,
        cosine: torch.Tensor,   # (B, num_classes)  from MagFaceLinear
        mag:    torch.Tensor,   # (B,)               from MagFaceLinear
        labels: torch.Tensor,   # (B,)  class indices
    ) -> torch.Tensor:
        # Per-sample adaptive margin
        m = self._adaptive_margin(mag)                          # (B,)

        cos_theta = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)

        # cos(θ + m_i) for each sample i — only for the target class
        cos_m = torch.cos(m).unsqueeze(1)                      # (B, 1)
        sin_m = torch.sin(m).unsqueeze(1)                      # (B, 1)
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m   # (B, C)

        # Apply margin only to the ground-truth class column
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cosine

        # Scale and cross-entropy
        output = output * self.scale
        ce_loss = F.cross_entropy(output, labels)

        # Magnitude regularisation
        mag_loss = self._magnitude_regulariser(mag)

        return ce_loss + mag_loss


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., ICCV 2017).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Applied on top of the MagFace scaled logits to down-weight
    easy (well-classified) examples and focus on hard ones.

    Args:
        gamma : focusing parameter (default 2.0)
        alpha : class weight scalar or None  (default 1.0 = no weighting)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Standard CE gives log(p_t)
        ce   = F.cross_entropy(logits, labels, reduction="none")   # (B,)
        p_t  = torch.exp(-ce)                                       # (B,)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce
        return loss.mean()


# ── MagFace + Focal combined ──────────────────────────────────────────────────

class MagFaceFocalLoss(nn.Module):
    """
    MagFace with Focal Loss on the classifier outputs.

    The MagFace margin is applied first (exactly as in MagFaceLoss),
    then the resulting scaled logits are passed through Focal Loss
    instead of standard CrossEntropyLoss.
    """

    def __init__(
        self,
        scale:    float = 64.0,
        l_m:      float = 0.45,
        u_m:      float = 0.80,
        l_a:      float = 10.0,
        u_a:      float = 110.0,
        lambda_g: float = 35.0,
        gamma:    float = 2.0,
        alpha:    float = 1.0,
    ):
        super().__init__()
        # Reuse MagFaceLoss internals for margin + regularisation
        self._mag  = MagFaceLoss(scale, l_m, u_m, l_a, u_a, lambda_g)
        self._focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.scale  = scale

    def forward(
        self,
        cosine: torch.Tensor,
        mag:    torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # ── Apply adaptive margin ──────────────────────────────────────────
        m = self._mag._adaptive_margin(mag)

        cos_theta   = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        sin_theta   = torch.sqrt(1.0 - cos_theta ** 2)
        cos_m       = torch.cos(m).unsqueeze(1)
        sin_m       = torch.sin(m).unsqueeze(1)
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        output  = one_hot * cos_theta_m + (1.0 - one_hot) * cosine
        output  = output * self.scale                     # scaled logits

        # ── Focal loss instead of CE ───────────────────────────────────────
        focal_loss = self._focal(output, labels)

        # ── Magnitude regularisation (same as MagFace) ────────────────────
        mag_loss   = self._mag._magnitude_regulariser(mag)

        return focal_loss + mag_loss


# ── Total loss for DNNetV2 ────────────────────────────────────────────────────

class TotalLossV2(nn.Module):
    """
    Combined loss for DNNetV2:
        L_total = L_con + ½(L_mag(anchor) + L_mag(pair))

    where L_mag is MagFaceFocalLoss.
    The contrastive loss is kept from DNNet for consistency.
    """

    def __init__(
        self,
        contrastive_margin: float = 2.0,
        magface_scale:      float = 64.0,
        l_m:                float = 0.45,
        u_m:                float = 0.80,
        l_a:                float = 10.0,
        u_a:                float = 110.0,
        lambda_g:           float = 35.0,
        focal_gamma:        float = 2.0,
        focal_alpha:        float = 1.0,
    ):
        super().__init__()
        from .losses import ContrastiveLoss
        self.contrastive = ContrastiveLoss(margin=contrastive_margin)
        self.magface     = MagFaceFocalLoss(
            scale    = magface_scale,
            l_m      = l_m,
            u_m      = u_m,
            l_a      = l_a,
            u_a      = u_a,
            lambda_g = lambda_g,
            gamma    = focal_gamma,
            alpha    = focal_alpha,
        )

    def forward(
        self,
        anchor_emb:     torch.Tensor,   # (B, D)  raw (not normalised)
        pair_emb:       torch.Tensor,   # (B, D)  raw
        pair_labels:    torch.Tensor,   # (B,)    binary 1=same 0=diff
        anchor_cosine:  torch.Tensor,   # (B, C)  from MagFaceLinear
        pair_cosine:    torch.Tensor,   # (B, C)
        anchor_mag:     torch.Tensor,   # (B,)    from MagFaceLinear
        pair_mag:       torch.Tensor,   # (B,)
        anchor_class:   torch.Tensor,   # (B,)    true class index
        pair_class:     torch.Tensor,   # (B,)
    ) -> tuple:
        # Normalise embeddings for contrastive loss
        anc_norm  = F.normalize(anchor_emb, p=2, dim=1)
        pair_norm = F.normalize(pair_emb,   p=2, dim=1)

        l_con      = self.contrastive(anc_norm, pair_norm, pair_labels)
        l_mag_anc  = self.magface(anchor_cosine, anchor_mag, anchor_class)
        l_mag_pair = self.magface(pair_cosine,   pair_mag,   pair_class)
        l_total    = l_con + 0.5 * (l_mag_anc + l_mag_pair)

        return l_total, l_con, l_mag_anc, l_mag_pair
