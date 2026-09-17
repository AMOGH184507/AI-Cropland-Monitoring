import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BaselineUNet(nn.Module):
    """
    Baseline CNN (UNet-style) architecture for cropland segmentation.
    
    Input:  (B, 4, 256, 256) -> [Red, Green, Blue, NIR]
    Output: (B, 1, 256, 256) -> Unnormalized logits for field extent mask
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 1, init_features: int = 32):
        super().__init__()

        features = init_features
        # Encoder
        self.inc = DoubleConv(in_channels, features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features, features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features * 2, features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features * 4, features * 8))

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features * 8, features * 16))

        # Decoder
        self.up1 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(features * 16, features * 8)

        self.up2 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(features * 8, features * 4)

        self.up3 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(features * 4, features * 2)

        self.up4 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(features * 2, features)

        # Output head
        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)         # (B, 32, 256, 256)
        x2 = self.down1(x1)      # (B, 64, 128, 128)
        x3 = self.down2(x2)      # (B, 128, 64, 64)
        x4 = self.down3(x3)      # (B, 256, 32, 32)

        x_b = self.bottleneck(x4) # (B, 512, 16, 16)

        x = self.up1(x_b)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    print("Testing BaselineUNet...")
    dummy_input = torch.randn(2, 4, 256, 256)
    model = BaselineUNet()
    output = model(dummy_input)
    print(f"Input Shape : {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")
