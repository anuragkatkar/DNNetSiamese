import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

from .losses import ArcFaceLinear


class Facenet(nn.Module):

    def __init__(self, embedding_dim: int = 128, pretrained='vggface2'):
        super().__init__()
        self.base_model = InceptionResnetV1(pretrained=pretrained).eval()

        for name, param in self.base_model.named_parameters():
            if 'block8' in name or 'mixed_7a' in name or 'last_linear' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
        in_features = self.base_model.logits.in_features
        self.base_model.logits = nn.Identity()
        self.head = nn.Linear(in_features, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        x = self.head(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x
    

class Siamese(nn.Module):

    def __init__(
        self,
        num_classes:    int,
        embedding_dim: int = 128
    ):
        super().__init__()
        self.custom = Facenet(embedding_dim=embedding_dim)
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


