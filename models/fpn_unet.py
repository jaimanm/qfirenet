import torch
from torch import nn
import torch.nn.functional as F

from models.unet import DoubleConv, Down, OutConv


class LateralBlock(nn.Module):
    """1x1 conv to unify channel dims across pyramid levels."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SmoothBlock(nn.Module):
    """3x3 conv applied after top-down fusion to remove aliasing."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))


class FPNUNet(nn.Module):
    """
    U-Net encoder with an FPN decoder.

    The encoder is identical to ClassicalUNet, producing feature maps
    x1-x5 at progressively halved resolutions. The FPN decoder builds
    a top-down pyramid with lateral connections, fuses all levels back
    to full resolution, and produces the final segmentation map.

    Pyramid levels (with bilinear=True defaults):
        P5: from x5  (1024//2 = 512 ch) -> 32x32
        P4: from x4  (512 ch)           -> 64x64
        P3: from x3  (256 ch)           -> 128x128
        P2: from x2  (128 ch)           -> 256x256
        P1: from x1  (64 ch)            -> 512x512
    """
    def __init__(self, n_classes, n_channels=13, bilinear=True, config=None):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.bilinear = bilinear

        fpn_channels = config.get('fpn_out_channels', 256) if config else 256
        factor = 2 if bilinear else 1

        # ── Encoder (identical to ClassicalUNet) ──────────────────────────
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        
        # ── Regularization (Bottleneck Dropout) ───────────────────────────
        self.dropout = nn.Dropout2d(p=0.5)

        # ── FPN lateral connections (one per encoder stage) ───────────────
        # Each lateral squashes the encoder's channel dim to fpn_channels
        self.lat5 = LateralBlock(1024 // factor, fpn_channels)
        self.lat4 = LateralBlock(512,            fpn_channels)
        self.lat3 = LateralBlock(256,            fpn_channels)
        self.lat2 = LateralBlock(128,            fpn_channels)
        self.lat1 = LateralBlock(64,             fpn_channels)

        # ── Smoothing convs (applied after top-down addition) ─────────────
        self.smooth5 = SmoothBlock(fpn_channels)
        self.smooth4 = SmoothBlock(fpn_channels)
        self.smooth3 = SmoothBlock(fpn_channels)
        self.smooth2 = SmoothBlock(fpn_channels)
        self.smooth1 = SmoothBlock(fpn_channels)

        # ── Segmentation head ─────────────────────────────────────────────
        # All 5 pyramid levels are upsampled to P1 resolution and summed,
        # then a final 1x1 conv produces per-pixel class scores.
        self.outc = OutConv(fpn_channels, n_classes)

    def _upsample_add(self, top_down, lateral):
        """Upsample top-down feature map and add to lateral connection."""
        return F.interpolate(
            top_down, size=lateral.shape[-2:], mode='bilinear', align_corners=True
        ) + lateral

    def forward(self, x):
        # ── Encoder: same forward pass as ClassicalUNet ───────────────────
        x1 = self.inc(x)       # 64 ch,          512x512
        x2 = self.down1(x1)    # 128 ch,          256x256
        x3 = self.down2(x2)    # 256 ch,          128x128
        x4 = self.down3(x3)    # 512 ch,          64x64
        x5 = self.down4(x4)    # 512 ch (bilin),  32x32
        # x5 = self.dropout(x5)  # Drop out 50% of the deepest features

        # ── Lateral projections ───────────────────────────────────────────
        l5 = self.lat5(x5)
        l4 = self.lat4(x4)
        l3 = self.lat3(x3)
        l2 = self.lat2(x2)
        l1 = self.lat1(x1)

        # ── Top-down pathway ──────────────────────────────────────────────
        p5 = self.smooth5(l5)
        p4 = self.smooth4(self._upsample_add(p5, l4))
        p3 = self.smooth3(self._upsample_add(p4, l3))
        p2 = self.smooth2(self._upsample_add(p3, l2))
        p1 = self.smooth1(self._upsample_add(p2, l1))

        # ── Fuse all pyramid levels at full resolution ────────────────────
        target = p1.shape[-2:]
        fused = (
            p1
            + F.interpolate(p2, size=target, mode='bilinear', align_corners=True)
            + F.interpolate(p3, size=target, mode='bilinear', align_corners=True)
            + F.interpolate(p4, size=target, mode='bilinear', align_corners=True)
            + F.interpolate(p5, size=target, mode='bilinear', align_corners=True)
        )

        return self.outc(fused)