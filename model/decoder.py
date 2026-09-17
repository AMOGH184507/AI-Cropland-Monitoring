import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ConvBlock


# ==============================================================================
# BASE PAPER COMPONENT: HIERARCHICAL INFORMATION FUSION DECODER BLOCK
# ==============================================================================
class UpDecoderBlock(nn.Module):
    """
    BASE PAPER COMPONENT:
    Progressive upsampling decoder block fusing encoder skip connections.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


# ==============================================================================
# BASE PAPER COMPONENT: HIERARCHICAL INFORMATION FUSION DECODER
# ==============================================================================
class HierarchicalDecoder(nn.Module):
    """
    BASE PAPER COMPONENT:
    Hierarchical Information Fusion Decoder that progressively reconstructs 
    spatial parcel shapes by fusing bottleneck representations with multi-scale 
    encoder skip connections and BGFE boundary features.
    """
    def __init__(self, encoder_channels=[64, 128, 256, 512], bgfe_channels=32):
        super().__init__()
        c1_ch, c2_ch, c3_ch, c4_ch = encoder_channels

        # Up-Block 3: c4 [512] + c3 [256] -> [256]
        self.up3 = UpDecoderBlock(c4_ch, c3_ch, 256)

        # Up-Block 2: [256] + c2 [128] -> [128]
        self.up2 = UpDecoderBlock(256, c2_ch, 128)

        # Up-Block 1: [128] + c1 [64] + bgfe [32] -> [64]
        self.up1 = UpDecoderBlock(128, c1_ch + bgfe_channels, 64)

    def forward(self, c1, c2, c3, c4, bgfe_feats):
        # c4: [B, 512, 32, 32]
        # c3: [B, 256, 64, 64]
        # c2: [B, 128, 128, 128]
        # c1: [B, 64, 256, 256]
        # bgfe_feats: [B, 32, 256, 256]

        d3 = self.up3(c4, c3)                # [B, 256, 64, 64]
        d2 = self.up2(d3, c2)                # [B, 128, 128, 128]
        
        # Fuse Stage 1 features and BGFE boundary features
        high_res_skip = torch.cat([c1, bgfe_feats], dim=1) # [B, 64 + 32, 256, 256]
        d1 = self.up1(d2, high_res_skip)     # [B, 64, 256, 256]

        return d1
