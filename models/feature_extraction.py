"""
Feature Extraction Module for DNNet.

Architecture (Figure 5 of the paper):
  1. Backbone: ResNet-152 (without its final GAP + FC layers)
  2. Extra network: two additional conv blocks to reduce feature-map channels
     Block 1: Conv(3×3, 1024→512) → BN → ReLU
     Block 2: Conv(3×3, 512→256)  → BN → ReLU
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models import ResNet152_Weights, ResNet18_Weights


class ExtraConvBlock(nn.Module):
    """Single extra conv block: Conv → BN → ReLU (Figure 5b)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FeatureExtractionModule(nn.Module):
    """
    Feature Extraction Module (step 1 of DNNet).

    Input  : (B, 3, H, W)
    Output : (B, 256, H', W')   where H', W' ≈ H/32, W/32
    """

    # ResNet-152 outputs 2048 channels before its GAP layer.
    # RESNET_OUT_CHANNELS = 2048
    RESNET_OUT_CHANNELS = 512

    def __init__(self, extra_channels: list = None):
        super().__init__()

        if extra_channels is None:
            extra_channels = [512, 256]   # paper default (Fig. 5b)

        # ── Backbone: ResNet-152 minus the last GAP + FC ───────────────────
        # resnet = tvm.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        resnet = tvm.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Keep everything up to (and including) layer4; drop avgpool and fc
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # ── Extra network: channel-reduction blocks ────────────────────────
        in_ch = self.RESNET_OUT_CHANNELS
        extra_blocks = []
        for out_ch in extra_channels:
            extra_blocks.append(ExtraConvBlock(in_ch, out_ch))
            in_ch = out_ch

        self.extra_net  = nn.Sequential(*extra_blocks)
        self.out_channels = in_ch   # 256 by default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.extra_net(features)
        return features

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
