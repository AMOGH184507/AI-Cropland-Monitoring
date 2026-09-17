import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# BASE PAPER COMPONENT: BOUNDARY-GUIDED FEATURE ENHANCEMENT (BGFE) MODULE
# ==============================================================================
class BGFEModule(nn.Module):
    """
    BASE PAPER COMPONENT:
    Boundary-Guided Feature Enhancement (BGFE) Module.
    
    Purpose:
    Extracts and enhances fine boundary/edge representations from high-resolution 
    early encoder features (Stage 1) to sharpen cropland parcel boundaries 
    and prevent adjacent parcel adhesion.
    """
    def __init__(self, in_channels=64, out_channels=32):
        super().__init__()
        # Edge gradient feature extractor
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        # Spatial boundary attention gating
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Projection to output dimension
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x shape: Stage 1 high-res feature map [B, 64, 256, 256]
        edge_feats = self.edge_conv(x)
        attn_map = self.gate_conv(edge_feats)
        
        # Multiply features by boundary attention mask
        enhanced = x * attn_map + edge_feats
        
        # Project output [B, 32, 256, 256]
        out = self.proj(enhanced)
        return out
