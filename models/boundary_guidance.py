import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryGuidanceModule(nn.Module):
    """
    Boundary Guidance Module (BGM) for boundary-aware field segmentation.

    Conceptually:
    CNN Features + Transformer Context -> Boundary Features / Prediction
                                          ↓
                                 [ Boundary Guidance Module ]
                                          ↓
                               Boundary-Aware Refined Features
                                          ↓
                                  Field Extent Decoder

    Uses boundary feature representations to compute spatial attention gating maps,
    sharpening field boundaries and suppressing false positives near crop edges.
    """
    def __init__(self, feat_channels: int, boundary_channels: int = 32):
        super().__init__()
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(boundary_channels, feat_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.gate_conv = nn.Sequential(
            nn.Conv2d(feat_channels + feat_channels // 2, feat_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, main_features: torch.Tensor, boundary_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            main_features: torch.Tensor of shape (B, C_main, H, W)
            boundary_features: torch.Tensor of shape (B, C_bound, H, W)

        Returns:
            refined_features: torch.Tensor of shape (B, C_main, H, W)
        """
        # Align spatial dimensions if needed
        if main_features.shape[2:] != boundary_features.shape[2:]:
            boundary_features = F.interpolate(
                boundary_features, size=main_features.shape[2:], mode="bilinear", align_corners=False
            )

        # Process boundary features
        b_feat = self.boundary_conv(boundary_features)

        # Concatenate main features and boundary features to compute spatial attention gate
        concat_feat = torch.cat([main_features, b_feat], dim=1)
        spatial_gate = self.gate_conv(concat_feat)  # (B, 1, H, W)

        # Apply attention gating: enhance features along boundary-guided regions
        gated_features = main_features * (1.0 + spatial_gate)

        # Final feature refinement
        refined_features = self.fusion_conv(gated_features)
        return refined_features


if __name__ == "__main__":
    print("Testing BoundaryGuidanceModule...")
    main_feat = torch.randn(2, 128, 64, 64)
    bound_feat = torch.randn(2, 32, 64, 64)

    bgm = BoundaryGuidanceModule(feat_channels=128, boundary_channels=32)
    refined = bgm(main_feat, bound_feat)

    print(f"Main Feature Shape    : {main_feat.shape}")
    print(f"Boundary Feature Shape: {bound_feat.shape}")
    print(f"Refined Feature Shape : {refined.shape}")
    print(f"Parameter Count       : {sum(p.numel() for p in bgm.parameters()):,}")
