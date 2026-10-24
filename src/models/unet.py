"""
PyTorch 2D U-Net architecture for spatial wildfire spread prediction.
Takes multi-channel remote sensing rasters [B, 12, 64, 64] and predicts
pixel-wise next-day fire propagation probabilities [B, 1, 64, 64].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    """Downscaling with MaxPool2D then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class UpBlock(nn.Module):
    """Upscaling with Bilinear Interpolation then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle spatial padding if odd dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_x > 0 or diff_y > 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class WildfireUNet(nn.Module):
    """
    Multimodal Spatial U-Net for Wildfire Spread Segmentation.
    Input: [Batch, in_channels=12, H=64, W=64]
    Output: [Batch, out_channels=1, H=64, W=64] (Raw Logits)
    """
    def __init__(self, in_channels=12, out_channels=1, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder Path
        self.inc = DoubleConv(in_channels, base_channels)          # 64x64: 12 -> 32
        self.down1 = DownBlock(base_channels, base_channels * 2)  # 32x32: 32 -> 64
        self.down2 = DownBlock(base_channels * 2, base_channels * 4) # 16x16: 64 -> 128
        self.down3 = DownBlock(base_channels * 4, base_channels * 8) # 8x8: 128 -> 256

        # Decoder Path
        self.up1 = UpBlock(base_channels * 8 + base_channels * 4, base_channels * 4) # 256 + 128 = 384 -> 128
        self.up2 = UpBlock(base_channels * 4 + base_channels * 2, base_channels * 2) # 128 + 64  = 192 -> 64
        self.up3 = UpBlock(base_channels * 2 + base_channels, base_channels)         # 64  + 32  = 96  -> 32

        # Output Head
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder with Skip Connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.outc(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = WildfireUNet(in_channels=12, out_channels=1, base_channels=32)
    dummy_input = torch.randn(2, 12, 64, 64)
    out = model(dummy_input)

    print("WildfireUNet Architecture Check:")
    print(f"  Input Tensor Shape:   {dummy_input.shape}")
    print(f"  Output Tensor Shape:  {out.shape}")
    print(f"  Trainable Parameters: {model.count_parameters():,}")
    print("  Output Range Check:   Min={:.3f}, Max={:.3f}".format(out.min().item(), out.max().item()))
    assert out.shape == (2, 1, 64, 64), "Output shape must match input spatial dimensions [B, 1, 64, 64]"
    print("  Status: Verified successfully!")
