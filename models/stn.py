import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    def __init__(self):
        super().__init__()

        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Affine parameter regressor
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6),
        )

        # Initialize as identity transform
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor(
                [
                    1, 0, 0,
                    0, 1, 0,
                ],
                dtype=torch.float,
            )
        )

    def forward(self, x):
        """
        x: [B, 3, 512, 512]
        """

        xs = self.localization(x)
        # [B, 32, 4, 4]

        xs = xs.view(xs.size(0), -1)
        # [B, 512]

        theta = self.fc_loc(xs)
        # [B, 6]
        print(theta)

        theta = theta.view(-1, 2, 3)
        # [B, 2, 3]

        grid = F.affine_grid(
            theta,
            x.size(),
            align_corners=False,
        )
        # [B, H, W, 2]

        x = F.grid_sample(
            x,
            grid,
            align_corners=False,
        )
        # [B, 3, 512, 512]

        return x