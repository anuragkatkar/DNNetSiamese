"""
Attention Module for DNNet (step 2).

Implements the Dual Attention Network (DAN) from:
  Fu et al., "Dual Attention Network for Scene Segmentation", CVPR 2019.

As used in the paper (Figure 7):
  - Channel Attention Module  (eq. 3–4)
  - Spatial Attention Module  (eq. 5–6)

Both are applied to the feature maps from the Feature Extraction Module.
Their outputs are concatenated with the original feature map along the
channel axis, then passed through GAP + FC to produce the final
1024-dimensional embedding vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionModule(nn.Module):
    """
    Channel attention (Figure 7a / equations 3-4).

    Given A ∈ R^{C×H×W}:
      1. Reshape to R^{C×N}  where N = H*W
      2. Compute pairwise similarity X ∈ R^{C×C}  via softmax(A · Aᵀ)
      3. E = β · (Xᵀ · A) ⊕ A        (element-wise sum with input)
    """

    def __init__(self):
        super().__init__()
        # β is a learnable scalar, initialised at 0 (paper + [47])
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        B, C, H, W = A.size()
        N = H * W

        # Reshape: (B, C, N)
        a_flat = A.view(B, C, N)

        # Pairwise channel similarity: (B, C, C)
        energy = torch.bmm(a_flat, a_flat.permute(0, 2, 1))          # (B, C, C)
        X = F.softmax(energy, dim=-1)                                  # eq. (3)

        # Weighted channel aggregation: (B, C, N) → reshape (B, C, H, W)
        out = torch.bmm(X.permute(0, 2, 1), a_flat)                  # (B, C, N)
        out = out.view(B, C, H, W)

        # Scale and residual (eq. 4)
        E = self.beta * out + A
        return E


class SpatialAttentionModule(nn.Module):
    """
    Spatial attention (Figure 7b / equations 5-6).

    Given A ∈ R^{C×H×W}:
      1. B = Conv(A),  C_feat = Conv(A)   → both ∈ R^{C×N}
      2. S = softmax(Bᵀ · C_feat)        → spatial map ∈ R^{N×N}
      3. D = Conv(A)                      → ∈ R^{C×N}
      4. E = α · (D · Sᵀ) ⊕ A
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_B = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv_C = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv_D = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        # α is a learnable scalar, initialised at 0 (paper + [47])
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        B_sz, C, H, W = A.size()
        N = H * W

        # Project and flatten
        proj_B = self.conv_B(A).view(B_sz, C, N)          # (B, C, N)
        proj_C = self.conv_C(A).view(B_sz, C, N)          # (B, C, N)
        proj_D = self.conv_D(A).view(B_sz, C, N)          # (B, C, N)

        # Spatial similarity: (B, N, N)
        energy = torch.bmm(proj_B.permute(0, 2, 1), proj_C)  # (B, N, N)  eq. (5)
        S = F.softmax(energy, dim=-1)

        # Aggregate spatial features: (B, C, N) → (B, C, H, W)
        out = torch.bmm(proj_D, S.permute(0, 2, 1))           # (B, C, N)  eq. (6)
        out = out.view(B_sz, C, H, W)

        # Scale and residual (eq. 6)
        E = self.alpha * out + A
        return E


class AttentionModule(nn.Module):
    """
    Full Attention Module (Figure 7).

    Applies both channel and spatial attention to the feature map from step 1,
    then concatenates [channel_out, spatial_out, original] along the channel
    axis and passes through GAP + FC to produce the final embedding vector.

    Input  : (B, C, H, W)   – output of FeatureExtractionModule
    Output : (B, embedding_dim)
    """

    def __init__(self, in_channels: int, embedding_dim: int = 1024):
        super().__init__()
        self.channel_att = ChannelAttentionModule()
        self.spatial_att = SpatialAttentionModule(in_channels)

        # After concatenation along channel axis: in_channels × 3
        concat_channels = in_channels * 3

        self.gap = nn.AdaptiveAvgPool2d(1)                   # Global Average Pooling
        self.fc  = nn.Linear(concat_channels, embedding_dim) # final FC → 1024-d

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        # Channel and spatial attention maps
        E_channel = self.channel_att(A)   # (B, C, H, W)
        E_spatial  = self.spatial_att(A)  # (B, C, H, W)

        # Concatenate along channel axis (paper: "connected … according to
        # the channel axes")
        concat = torch.cat([E_channel, E_spatial, A], dim=1)  # (B, 3C, H, W)

        # GAP → flatten → FC
        pooled   = self.gap(concat).squeeze(-1).squeeze(-1)    # (B, 3C)
        embedding = self.fc(pooled)                             # (B, embedding_dim)

        return embedding
