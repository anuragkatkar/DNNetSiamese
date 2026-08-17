import torch
import torch.nn as nn
import torch.nn.functional as F

class CircularMask(nn.Module):
    def __init__(self, image_size=512, diameter=500):
        super().__init__()

        radius = diameter / 2
        center = (image_size - 1) / 2

        y, x = torch.meshgrid(
            torch.arange(image_size),
            torch.arange(image_size),
            indexing="ij"
        )

        distance = torch.sqrt(
            (x - center) ** 2 +
            (y - center) ** 2
        )

        mask = (distance <= radius).float()

        # [1, 1, H, W]
        self.register_buffer(
            "mask",
            mask.unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        return x * self.mask