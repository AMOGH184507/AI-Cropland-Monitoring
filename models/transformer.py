import torch
import torch.nn as nn


class SpatialTransformerBlock(nn.Module):
    """
    Spatial Transformer Block using Multi-Head Self-Attention (MHSA)
    to model global context and long-range spatial dependencies across fields.

    CNN         -> extracts local spatial feature maps (high-frequency edges/textures)
    Transformer -> models global contextual relationships (distant boundaries, roads, irregular field structures)
    """
    def __init__(self, channels: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(channels)
        hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, C, H, W)
        Output:  (B, C, H, W)
        """
        b, c, h, w = x.shape

        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)

        # 1. Multi-Head Self-Attention with residual connection
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out

        # 2. Feed-Forward Network with residual connection
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        # Reshape back to (B, C, H, W)
        x_out = x_flat.transpose(1, 2).reshape(b, c, h, w)
        return x_out


class SpatialTransformer(nn.Module):
    """
    Stack of Spatial Transformer Blocks operating on CNN feature maps.
    """
    def __init__(self, channels: int, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([
            SpatialTransformerBlock(channels=channels, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":
    print("Testing SpatialTransformer...")
    # Feature map from CNN encoder bottleneck (B=2, C=256, H=32, W=32)
    dummy_feat = torch.randn(2, 256, 32, 32)
    transformer = SpatialTransformer(channels=256, num_layers=2, num_heads=4)
    out_feat = transformer(dummy_feat)
    print(f"Input Feature Shape : {dummy_feat.shape}")
    print(f"Output Feature Shape: {out_feat.shape}")
    print(f"Parameter Count     : {sum(p.numel() for p in transformer.parameters()):,}")
