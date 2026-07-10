"""
DNNetV2: Dog Nose Network Version 2.

Improvements over DNNet (Boulaouane et al., IEEE Access 2024):
  1. Backbone   : ResNet-152       → TinyViT-21M (Vision Transformer)
  2. Loss       : ArcFace          → MagFace + Focal Loss
  3. Pair mining: random negatives → hard negative mining

Architecture:
  Input (B, 3, 224, 224)
      │
      ▼  TinyViTFeatureExtractor
      TinyViT-21M (no head, avg pool)      → (B, 576)
      │
      ▼  TinyViTEmbeddingHead
      FC(576 → 256) + BN + ReLU + Dropout
      FC(256 → embedding_dim) + BN         → (B, embedding_dim)  [raw, not normalised]
      │
      ▼  Siamese Network (shared weights)
      anchor branch ──┐
                       ├──► ContrastiveLoss  (on L2-normalised embeddings)
      pair   branch ──┘
      both branches → MagFaceLinear → MagFaceFocalLoss
      Total Loss = L_con + ½(L_mag_anchor + L_mag_pair)

Key design note
---------------
MagFaceLoss needs the raw (un-normalised) embedding magnitude to compute
the adaptive margin. Therefore DNNetV2Single.forward() returns the raw
embedding, unlike DNNet which returns the L2-normalised version.
L2 normalisation for contrastive loss is applied inside TotalLossV2.
For inference (get_embedding), L2 normalisation IS applied so that
cosine distance can be used for nearest-neighbour search.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extraction_v2 import TinyViTFeatureExtractor, TinyViTEmbeddingHead
from .losses_v2 import MagFaceLinear


class DNNetV2Single(nn.Module):
    """
    Single branch of the DNNetV2 Siamese network.

    Returns raw (un-normalised) embeddings so that MagFaceLoss can
    access the embedding magnitude. Call get_embedding() for inference
    which returns the L2-normalised version.

    Args:
        model_name    : TinyViT variant (default 'tiny_vit_21m_224')
        embedding_dim : size of the output embedding vector (default 128)
        pretrained    : load pretrained TinyViT weights (default True)
        freeze_epochs : handled externally via freeze/unfreeze methods
    """

    def __init__(
        self,
        model_name:    str = "tiny_vit_21m_224",
        embedding_dim: int = 128,
        pretrained:    bool = True,
    ):
        super().__init__()

        self.backbone = TinyViTFeatureExtractor(
            model_name = model_name,
            pretrained = pretrained,
            freeze     = False,   # freeze/unfreeze managed externally
        )
        self.head = TinyViTEmbeddingHead(
            in_features   = self.backbone.out_channels,
            embedding_dim = embedding_dim,
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw (un-normalised) embeddings (B, embedding_dim)."""
        features  = self.backbone(x)    # (B, 576)
        embedding = self.head(features) # (B, embedding_dim)  — raw, not normalised
        return embedding

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: returns L2-normalised embeddings for distance computation."""
        with torch.no_grad():
            raw = self.forward(x)
        return F.normalize(raw, p=2, dim=1)

    def freeze_backbone(self):
        self.backbone.freeze_backbone()

    def unfreeze_backbone(self):
        self.backbone.unfreeze_backbone()


class DNNetV2Siamese(nn.Module):
    """
    Full Siamese DNNetV2 with shared weights.

    Owns the MagFaceLinear classifier head, which is optimised separately
    with SGD (same convention as DNNetSiamese + ArcFaceLinear).

    Args:
        num_classes   : number of identities in the training set
        model_name    : TinyViT variant
        embedding_dim : embedding size
        pretrained    : load pretrained TinyViT weights
    """

    def __init__(
        self,
        num_classes:   int,
        model_name:    str  = "tiny_vit_21m_224",
        embedding_dim: int  = 128,
        pretrained:    bool = True,
    ):
        super().__init__()
        self.dnnet_v2      = DNNetV2Single(
            model_name    = model_name,
            embedding_dim = embedding_dim,
            pretrained    = pretrained,
        )
        self.magface_head = MagFaceLinear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim

    def forward(
        self,
        anchor: torch.Tensor,    # (B, 3, H, W)
        pair:   torch.Tensor,    # (B, 3, H, W)
    ):
        """
        Returns:
            anchor_emb    : (B, D)  raw embeddings for anchor
            pair_emb      : (B, D)  raw embeddings for pair
            anchor_cosine : (B, C)  cosine logits for anchor
            pair_cosine   : (B, C)  cosine logits for pair
            anchor_mag    : (B,)    embedding magnitudes for anchor
            pair_mag      : (B,)    embedding magnitudes for pair
        """
        anchor_emb = self.dnnet_v2(anchor)   # raw, not normalised
        pair_emb   = self.dnnet_v2(pair)

        anchor_cosine, anchor_mag = self.magface_head(anchor_emb)
        pair_cosine,   pair_mag   = self.magface_head(pair_emb)

        return (
            anchor_emb, pair_emb,
            anchor_cosine, pair_cosine,
            anchor_mag, pair_mag,
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: L2-normalised embeddings for nearest-neighbour search."""
        raw = self.dnnet_v2(x)
        return F.normalize(raw, p=2, dim=1)

    def freeze_backbone(self):
        self.dnnet_v2.freeze_backbone()

    def unfreeze_backbone(self):
        self.dnnet_v2.unfreeze_backbone()


def build_model_v2(num_classes: int, cfg=None) -> DNNetV2Siamese:
    """Factory function — reads from config if cfg not provided."""
    if cfg is None:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from configs import config as cfg

    model = DNNetV2Siamese(
        num_classes   = num_classes,
        model_name    = getattr(cfg, "TINYVIT_MODEL", "tiny_vit_21m_224"),
        embedding_dim = cfg.EMBEDDING_DIM,
        pretrained    = True,
    )
    return model
