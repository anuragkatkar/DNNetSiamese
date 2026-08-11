"""
DNNet: Dog Nose Network (full Siamese model).

Architecture (Figure 3 & 4 of the paper):
  Siamese Network
  ├── Branch A (anchor)  ──┐
  │   DNNet (shared weights)│──> embeddings ──> Contrastive Loss
  └── Branch B (pair)   ──┘           │
                                        └─> ArcFace Loss (per branch)

Each DNNet consists of:
  1. FeatureExtractionModule  (ResNet-152 backbone + extra conv blocks)
  2. AttentionModule          (DAN: channel + spatial attention → GAP → FC → 1024-d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extraction import FeatureExtractionModule
from .attention import AttentionModule
from .losses import ArcFaceLinear
from .mask import CircularMask

from configs.config import IMAGE_SIZE

class DNNet(nn.Module):
    """
    Single branch of the Siamese network.

    Input  : (B, 3, H, W)
    Output : (B, embedding_dim)  L2-normalised embedding vector
    """

    def __init__(
        self,
        extra_channels: list  = None,
        embedding_dim:  int   = 1024,
    ):
        super().__init__()
        if extra_channels is None:
            extra_channels = [512, 256]
        self.mask = CircularMask(IMAGE_SIZE, IMAGE_SIZE - 12)
        self.feature_extractor = FeatureExtractionModule(extra_channels=extra_channels)
        self.attention          = AttentionModule(
            in_channels   = self.feature_extractor.out_channels,
            embedding_dim = embedding_dim,
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked = self.mask(x)
        features  = self.feature_extractor(masked)     # (B, 256, H', W')
        embedding = self.attention(features)       # (B, 1024)
        # L2-normalise before distance computation (standard practice)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class DNNetSiamese(nn.Module):
    """
    Full Siamese DNNet with shared weights (Figure 4).

    The single DNNet instance is called twice (once for anchor, once for
    pair) so all weights are naturally shared.

    Also owns the ArcFace linear classifier head, which needs to be
    optimised with a separate SGD optimiser (as in the paper).
    """

    def __init__(
        self,
        num_classes:    int,
        extra_channels: list = None,
        embedding_dim:  int  = 1024,
    ):
        super().__init__()
        self.dnnet       = DNNet(extra_channels=extra_channels,
                                 embedding_dim=embedding_dim)
        self.arcface_head = ArcFaceLinear(embedding_dim, num_classes)

    def forward(
        self,
        anchor: torch.Tensor,          # (B, 3, H, W)
        pair:   torch.Tensor,          # (B, 3, H, W)
    ):
        """
        Returns:
            anchor_emb     : (B, D)  L2-normalised anchor embeddings
            pair_emb       : (B, D)  L2-normalised pair   embeddings
            anchor_logits  : (B, num_classes) cosine logits for anchor
            pair_logits    : (B, num_classes) cosine logits for pair
        """
        anchor_emb = self.dnnet(anchor)
        pair_emb   = self.dnnet(pair)

        anchor_logits = self.arcface_head(anchor_emb)
        pair_logits   = self.arcface_head(pair_emb)

        return anchor_emb, pair_emb, anchor_logits, pair_logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding for a single batch of images (inference/eval)."""
        return self.dnnet(x)


def build_model(num_classes: int, cfg=None) -> "DNNetSiamese":
    """Convenience factory that reads from config if cfg is not provided."""
    if cfg is None:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from configs import config as cfg

    model = DNNetSiamese(
        num_classes    = num_classes,
        extra_channels = cfg.EXTRA_CHANNELS,
        embedding_dim  = cfg.EMBEDDING_DIM,
    )
    return model
