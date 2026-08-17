import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import IMAGE_SIZE

from .losses import ArcFaceLinear


class Alpha(nn.Module):

    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        self.dim = embedding_dim

        self.conv  = nn.Sequential(
            nn.Conv2d(3, 16, 4, 1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 8, 4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),

            nn.Conv2d(32, 64, 4, 4, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.linear = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.Dropout(0.1),
            nn.Linear(256, self.dim),
        )





    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x


class Siamese(nn.Module):

    def __init__(
        self,
        num_classes:    int,
        embedding_dim: int = 128
    ):
        super().__init__()
        self.custom = Alpha(embedding_dim=embedding_dim)
        self.arcface_head = ArcFaceLinear(embedding_dim, num_classes)

    def forward(
        self,
        anchor: torch.Tensor,
        pair:   torch.Tensor
    ):
        anchor_emb = self.custom(anchor)
        pair_emb   = self.custom(pair)

        anchor_logits = self.arcface_head(anchor_emb)
        pair_logits   = self.arcface_head(pair_emb)

        return anchor_emb, pair_emb, anchor_logits, pair_logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.custom(x)
        embedding = F.normalize(x, p=2, dim=1)
        return embedding






def build_model(num_classes: int, cfg=None) -> Siamese:
    if cfg is None:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from configs import config as cfg

    model = Siamese(
        num_classes    = num_classes,
        embedding_dim  = cfg.EMBEDDING_DIM,
    )
    return model


