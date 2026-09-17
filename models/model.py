import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

try:
    from models.baseline_cnn import DoubleConv
    from models.transformer import SpatialTransformer
    from models.boundary_guidance import BoundaryGuidanceModule
except ImportError:
    from baseline_cnn import DoubleConv
    from transformer import SpatialTransformer
    from boundary_guidance import BoundaryGuidanceModule


class CNNEncoder(nn.Module):
    """
    4-Channel Sentinel-2 Multi-Scale CNN Encoder.
    Input:  (B, 4, 256, 256) -> [Red, Green, Blue, NIR]
    Outputs multi-scale feature maps for skip connections and bottleneck processing.
    """
    def __init__(self, in_channels: int = 4, base_features: int = 32):
        super().__init__()
        f = base_features
        self.inc = DoubleConv(in_channels, f)                      # (B, 32, 256, 256)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))    # (B, 64, 128, 128)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4)) # (B, 128, 64, 64)
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8)) # (B, 256, 32, 32)
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 8, f * 16)) # (B, 512, 16, 16)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x_b = self.bottleneck(x4)
        return x1, x2, x3, x4, x_b


class BoundaryDecoder(nn.Module):
    """
    Dedicated Boundary Prediction Decoder Head.
    Generates 1-channel boundary logits from encoder features.
    """
    def __init__(self, base_features: int = 32):
        super().__init__()
        f = base_features
        self.up = nn.Sequential(
            nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2),
            DoubleConv(f * 8, f * 4),
            nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2),
            DoubleConv(f * 2, f * 2)
        )
        self.b_head = nn.Sequential(
            nn.ConvTranspose2d(f * 2, f, kernel_size=4, stride=4),
            DoubleConv(f, f),
            nn.Conv2d(f, 1, kernel_size=1)
        )

    def forward(self, x_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.up(x_b)          # (B, 64, 64, 64)
        boundary_logits = self.b_head(feat) # (B, 1, 256, 256)
        return boundary_logits, feat


class FieldDecoder(nn.Module):
    """
    Field Extent Segmentation Decoder with Skip Connections.
    Decodes boundary-guided features to generate field extent logits.
    """
    def __init__(self, base_features: int = 32):
        super().__init__()
        f = base_features

        self.up1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(f * 16, f * 8)

        self.up2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(f * 4, f * 4)

        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(f * 4, f * 2)

        self.up4 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(f * 2, f)

        self.outc = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x_b: torch.Tensor, x4: torch.Tensor, x3_guided: torch.Tensor, x2: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        x = self.up1(x_b)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3_guided], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        field_logits = self.outc(x)
        return field_logits


class CroplandBoundaryModel(nn.Module):
    """
    Full Architecture: AI-Based Field Boundary Deduction Using Boundary Guidance

    Input:
        x: torch.Tensor of shape (B, 4, 256, 256) -> Sentinel-2 bands [B4, B3, B2, B8]

    Pipeline:
        1. CNN Encoder -> Multi-scale local spatial features
        2. Spatial Transformer -> Global context & long-range field relationships
        3. Boundary Decoder -> Boundary logits & boundary feature maps
        4. Boundary Guidance Module -> Boundary-aware spatial feature gating
        5. Field Decoder -> Final cropland field extent logits

    Returns:
        field_logits: torch.Tensor of shape (B, 1, 256, 256)
        boundary_logits: torch.Tensor of shape (B, 1, 256, 256)
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_features: int = 32,
        num_transformer_layers: int = 2,
        num_heads: int = 4
    ):
        super().__init__()
        f = base_features

        # 1. CNN Encoder
        self.encoder = CNNEncoder(in_channels=in_channels, base_features=f)

        # 2. Spatial Transformer Bottleneck for Global Context
        self.transformer = SpatialTransformer(channels=f * 16, num_layers=num_transformer_layers, num_heads=num_heads)

        # 3. Dedicated Boundary Branch
        self.boundary_decoder = BoundaryDecoder(base_features=f)

        # 4. Boundary Guidance Module
        self.boundary_guidance = BoundaryGuidanceModule(feat_channels=f * 4, boundary_channels=f * 2)

        # 5. Main Field Extent Decoder
        self.field_decoder = FieldDecoder(base_features=f)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Step 1: Extract multi-scale local features
        x1, x2, x3, x4, x_b = self.encoder(x)

        # Step 2: Global Context via Spatial Transformer Bottleneck
        x_b_trans = self.transformer(x_b)

        # Step 3: Predict Boundary Logits & Extract Boundary Features
        boundary_logits, boundary_feat = self.boundary_decoder(x_b_trans)

        # Step 4: Apply Boundary Guidance to Level 3 Features (64x64 resolution)
        x3_guided = self.boundary_guidance(x3, boundary_feat)

        # Step 5: Decode Field Extent Logits using Boundary-Guided Features
        field_logits = self.field_decoder(x_b_trans, x4, x3_guided, x2, x1)

        return field_logits, boundary_logits


if __name__ == "__main__":
    print("Testing CroplandBoundaryModel...")
    dummy_x = torch.randn(2, 4, 256, 256)
    model = CroplandBoundaryModel()

    field_logits, boundary_logits = model(dummy_x)

    print(f"Input Shape          : {dummy_x.shape}")
    print(f"Field Logits Shape   : {field_logits.shape}")
    print(f"Boundary Logits Shape: {boundary_logits.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
