import torch
import torch.nn as nn
from .encoder import HybridEncoder
from .boundary_guidance import BGFEModule
from .decoder import HierarchicalDecoder
from .heads import ParcelExtentHead, BoundaryPredictionHead


class CTHBNet(nn.Module):
    """
    CTHBNet: CNN-Transformer Hybrid Network with Boundary Guidance
    
    BASE PAPER COMPONENTS:
    - CNN-Transformer Hybrid Encoder (Local CNN features + Transformer Bottleneck)
    - Boundary-Guided Feature Enhancement (BGFE) Module
    - Hierarchical Information Fusion Decoder
    - Multi-Task Dual Heads (Extent Mask & Boundary Edge)
    
    OUR SENTINEL-2 ADAPTATIONS:
    - Early Fusion Stem ingesting [B, 5, 6, H, W] (5 bands x 6 timesteps) 
      and flattening into 30 channels [B, 30, H, W].
    """
    def __init__(self, in_bands=5, timesteps=6):
        super().__init__()
        # Multi-stage Hybrid Encoder with Sentinel-2 Early Fusion Stem
        self.encoder = HybridEncoder(in_bands=in_bands, timesteps=timesteps)

        # Boundary-Guided Feature Enhancement (BGFE) Module
        self.bgfe = BGFEModule(in_channels=64, out_channels=32)

        # Hierarchical Information Fusion Decoder
        self.decoder = HierarchicalDecoder(
            encoder_channels=[64, 128, 256, 512],
            bgfe_channels=32
        )

        # Multi-Task Prediction Heads
        self.extent_head = ParcelExtentHead(in_channels=64)
        self.boundary_head = BoundaryPredictionHead(in_channels=64)

    def forward(self, x):
        """
        Forward Pass:
        Input:
            x: Tensor of shape [B, 5, 6, H, W] or [B, 30, H, W]
        Returns:
            extent_logits: Tensor of shape [B, 1, H, W]
            boundary_logits: Tensor of shape [B, 1, H, W]
        """
        # Encoder Stage Feature Maps:
        # c1: [B, 64, H, W]
        # c2: [B, 128, H/2, W/2]
        # c3: [B, 256, H/4, W/4]
        # c4: [B, 512, H/8, W/8]
        c1, c2, c3, c4 = self.encoder(x)

        # Boundary Feature Enhancement:
        # bgfe_feats: [B, 32, H, W]
        bgfe_feats = self.bgfe(c1)

        # Hierarchical Feature Decoding:
        # decoded_feats: [B, 64, H, W]
        decoded_feats = self.decoder(c1, c2, c3, c4, bgfe_feats)

        # Dual Head Predictions:
        extent_logits = self.extent_head(decoded_feats)    # [B, 1, H, W]
        boundary_logits = self.boundary_head(decoded_feats)  # [B, 1, H, W]

        return extent_logits, boundary_logits
