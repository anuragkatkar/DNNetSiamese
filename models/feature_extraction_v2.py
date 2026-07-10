"""
Feature Extraction Module for DNNetV2.

Replaces the ResNet-152 CNN backbone with TinyViT — a lightweight Vision
Transformer pretrained on ImageNet-21k via knowledge distillation.

TinyViT variants (from timm):
    tiny_vit_5m_224    ~5M  params
    tiny_vit_11m_224   ~11M params
    tiny_vit_21m_224   ~21M params  ← paper uses this one
    tiny_vit_21m_384   ~21M params, higher resolution

Unlike ResNet which outputs a spatial feature map (B, C, H, W), TinyViT
outputs a sequence of patch tokens. We take the average of all patch tokens
as the feature vector, which plays the same role as the GAP output in DNNet.

The extra conv blocks from DNNet are removed because TinyViT's final stage
already produces a compact 576-d feature (21M model), which is directly
usable by the AttentionModule or a simple FC.

Requires: pip install timm
"""

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# ── Supported model names and their output feature dimensions ─────────────────
TINYVIT_FEATURE_DIMS = {
    "tiny_vit_5m_224":  320,
    "tiny_vit_11m_224": 448,
    "tiny_vit_21m_224": 576,
    "tiny_vit_21m_384": 576,
}


class TinyViTFeatureExtractor(nn.Module):
    """
    TinyViT backbone for DNNetV2.

    Loads a pretrained TinyViT from timm, removes its classification head,
    and returns a (B, feature_dim) vector per image — ready to feed into
    the AttentionModule (which expects a 2-D feature, not a spatial map).

    Because the output is already 1-D (pooled), the AttentionModule's GAP
    is bypassed. A simple FC inside the AttentionModule still produces the
    final embedding_dim vector.

    Args:
        model_name  : timm model name (default 'tiny_vit_21m_224')
        pretrained  : load ImageNet-21k pretrained weights (default True)
        freeze      : freeze backbone weights (default False)
    """

    def __init__(
        self,
        model_name: str  = "tiny_vit_21m_224",
        pretrained: bool = True,
        freeze:     bool = False,
    ):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for TinyViT.\n"
                "Install it with:  pip install timm"
            )

        if model_name not in TINYVIT_FEATURE_DIMS:
            raise ValueError(
                f"Unknown TinyViT model '{model_name}'. "
                f"Choose from: {list(TINYVIT_FEATURE_DIMS.keys())}"
            )

        # Load pretrained model, strip the classifier head
        self.backbone = timm.create_model(
            model_name,
            pretrained  = pretrained,
            num_classes = 0,        # removes final FC classifier
            global_pool = "avg",    # avg-pools patch tokens → (B, feature_dim)
        )

        self.out_channels = TINYVIT_FEATURE_DIMS[model_name]

        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, H, W)  — should match the model's expected resolution
                                 e.g. 224×224 for tiny_vit_*_224

        Returns:
            (B, feature_dim)  — already pooled, 1-D feature vector
        """
        return self.backbone(x)    # (B, feature_dim)


# ── Thin embedding head to replace AttentionModule for TinyViT ───────────────

class TinyViTEmbeddingHead(nn.Module):
    """
    Simple FC head that maps TinyViT's pooled features → embedding_dim.

    Used instead of the full AttentionModule (which expects a spatial map).
    The DAN attention is not applicable to 1-D pooled vectors, so we use
    a lightweight two-layer MLP with BN and dropout for regularisation.

    Input  : (B, in_features)   e.g. 576 for tiny_vit_21m
    Output : (B, embedding_dim) e.g. 128
    """

    def __init__(self, in_features: int, embedding_dim: int = 128):
        super().__init__()
        hidden = max(embedding_dim * 2, 256)
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)     # (B, embedding_dim)  NOT yet normalised
