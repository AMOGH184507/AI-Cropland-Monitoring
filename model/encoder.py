import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# OUR SENTINEL-2 ADAPTATION: EARLY FUSION STEM
# ==============================================================================
class EarlyFusionStem(nn.Module):
    """
    OUR SENTINEL-2 ADAPTATION:
    Reshapes multitemporal Sentinel-2 input tensor [B, 5, 6, H, W] 
    (5 bands: B2, B3, B4, B8, NDVI over 6 monthly timesteps) into [B, 30, H, W] 
    and applies initial 2D convolutions to project 30 channels into model dimension.
    """
    def __init__(self, in_bands=5, timesteps=6, out_channels=64):
        super().__init__()
        self.in_channels = in_bands * timesteps  # 5 * 6 = 30
        
        self.conv1 = nn.Conv2d(
            self.in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Input x can be [B, 5, 6, H, W] or already flattened [B, 30, H, W]
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            x = x.view(B, C * T, H, W)  # [B, 30, H, W]
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


# ==============================================================================
# BASE PAPER COMPONENT: CONVOLUTIONAL RESIDUAL BLOCK
# ==============================================================================
class ConvBlock(nn.Module):
    """
    BASE PAPER COMPONENT:
    Standard Residual Convolutional block for local spatial feature extraction.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


# ==============================================================================
# BASE PAPER COMPONENT / IMPLEMENTATION CHOICE: TRANSFORMER BOTTLENECK BLOCK
# ==============================================================================
class SpatialTransformerBlock(nn.Module):
    """
    BASE PAPER COMPONENT: Transformer Block for modeling global context.
    IMPLEMENTATION CHOICE: Uses PyTorch MultiheadAttention over flattened spatial tokens
    with LayerNorm and FeedForward network for pure PyTorch compatibility.
    """
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        # Flatten spatial dimensions: [B, H*W, C]
        tokens = x.flatten(2).transpose(1, 2)
        
        # Self-Attention with Residual connection
        norm_tokens = self.norm1(tokens)
        attn_out, _ = self.attn(norm_tokens, norm_tokens, norm_tokens)
        tokens = tokens + attn_out
        
        # MLP with Residual connection
        tokens = tokens + self.mlp(self.norm2(tokens))
        
        # Reshape back to feature map [B, C, H, W]
        out = tokens.transpose(1, 2).view(B, C, H, W)
        return out


# ==============================================================================
# BASE PAPER COMPONENT: CNN-TRANSFORMER HYBRID ENCODER
# ==============================================================================
class HybridEncoder(nn.Module):
    """
    BASE PAPER COMPONENT:
    Multi-stage Hybrid Encoder combining local CNN feature maps with 
    global Transformer attention modules.
    
    OUR SENTINEL-2 ADAPTATION:
    Incorporates EarlyFusionStem to ingest 30-channel multitemporal input.
    """
    def __init__(self, in_bands=5, timesteps=6):
        super().__init__()
        # Stem / Stage 1: [B, 64, H, W]
        self.stem = EarlyFusionStem(in_bands=in_bands, timesteps=timesteps, out_channels=64)
        self.stage1_block = ConvBlock(64, 64)

        # Stage 2: [B, 128, H/2, W/2]
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.stage2_block = ConvBlock(128, 128)

        # Stage 3: [B, 256, H/4, W/4] (CNN + Transformer Hybrid)
        self.down3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.stage3_block = ConvBlock(256, 256)
        self.stage3_trans = SpatialTransformerBlock(256, num_heads=8)

        # Stage 4: [B, 512, H/8, W/8] (Transformer Bottleneck)
        self.down4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.stage4_block = ConvBlock(512, 512)
        self.stage4_trans = SpatialTransformerBlock(512, num_heads=8)

    def forward(self, x):
        # Stage 1: [B, 64, 256, 256]
        c1 = self.stem(x)
        c1 = self.stage1_block(c1)

        # Stage 2: [B, 128, 128, 128]
        c2 = F.relu(self.bn2(self.down2(c1)), inplace=True)
        c2 = self.stage2_block(c2)

        # Stage 3: [B, 256, 64, 64]
        c3 = F.relu(self.bn3(self.down3(c2)), inplace=True)
        c3 = self.stage3_block(c3)
        c3 = self.stage3_trans(c3)

        # Stage 4: [B, 512, 32, 32]
        c4 = F.relu(self.bn4(self.down4(c3)), inplace=True)
        c4 = self.stage4_block(c4)
        c4 = self.stage4_trans(c4)

        return c1, c2, c3, c4
