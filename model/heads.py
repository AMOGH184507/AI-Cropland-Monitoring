import torch
import torch.nn as nn


# ==============================================================================
# BASE PAPER COMPONENT: DUAL MULTI-TASK PREDICTION HEADS
# ==============================================================================
class ParcelExtentHead(nn.Module):
    """
    BASE PAPER COMPONENT:
    Primary head predicting cropland field parcel binary segmentation mask logits.
    Output shape: [B, 1, H, W]
    """
    def __init__(self, in_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


class BoundaryPredictionHead(nn.Module):
    """
    BASE PAPER COMPONENT:
    Auxiliary head predicting cropland field parcel boundary edge logits.
    Output shape: [B, 1, H, W]
    """
    def __init__(self, in_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)
