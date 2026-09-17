import torch
from torch import nn
import torch.nn.functional as F

from models.unet import DoubleConv, Down, OutConv
from models.fpn_unet import LateralBlock, SmoothBlock
from models.qbnet_model import QBNetBottleneck


class QBNetFPN(nn.Module):
    """U-Net encoder with a QB-Net quantum bottleneck and an FPN decoder.

    Combines the QB-Net bottleneck (Xu, Zhao & Li 2025) — the known-working
    quantum module used by ``QBNet`` — with FPNUNet's multi-scale top-down
    decoder. The QB-Net circuit processes the deepest (32×32) feature map x5
    per spatial location; the FPN decoder then restores full-resolution detail
    via a top-down pathway with lateral skip connections from the encoder.

    Unlike ``QuantumFPNUNet`` (which squeezes x5 to a single channel descriptor
    and uses the quantum circuit only for SE-net-style channel attention), the
    QB-Net bottleneck is *spatially aware*: it runs the circuit at every one of
    the H×W locations and returns a full ``[B, C, H, W]`` map with a residual
    connection. This is the same bottleneck that produced non-degenerate fire
    detection in the plain-U-Net QBNet runs, dropped into an FPN host instead.

    Architecture summary:
        Encoder   : DoubleConv + Down x4          (identical to ClassicalUNet)
        Bottleneck: Dropout2d → QBNetBottleneck    (RY + U3 + CNOT ladder, residual)
        Decoder   : LateralBlock + SmoothBlock top-down FPN (identical to FPNUNet)
        Head      : OutConv (1×1 conv to n_classes)

    Pyramid levels (bilinear=True defaults):
        P5: from x5  (512 ch)  → 32×32
        P4: from x4  (512 ch)  → 64×64
        P3: from x3  (256 ch)  → 128×128
        P2: from x2  (128 ch)  → 256×256
        P1: from x1  (64 ch)   → 512×512

    Runtime note:
        QBNetBottleneck evaluates the circuit per spatial location, so a forward
        pass runs B·H·W circuits (e.g. 4·32·32 = 4096 at the 32×32 bottleneck).
        Expect the quantum bottleneck to dominate runtime vs the classical FPN.

    Config keys consumed:
        n_qubits         (int)  : number of qubits                 [4]
        n_layers         (int)  : variational layers in circuit     [2]
        fpn_out_channels (int)  : channel width of FPN pyramid      [256]
    """

    def __init__(self, n_classes, n_channels=13, bilinear=True, config=None):
        super().__init__()
        config = config or {}
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.bilinear = bilinear

        n_qubits = config.get('n_qubits', 4)
        n_layers = config.get('n_layers', 2)
        fpn_channels = config.get('fpn_out_channels', 256)
        factor = 2 if bilinear else 1

        # ── Encoder (identical to ClassicalUNet / FPNUNet) ────────────────
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # ── Bottleneck regularization ─────────────────────────────────────
        self.dropout = nn.Dropout2d(p=0.5)

        # ── QB-Net quantum bottleneck ─────────────────────────────────────
        # x5 shape after down4: [B, 512, 32, 32]  (bilinear=True → factor=2 → 512 ch)
        # QBNetBottleneck preserves channels and spatial size (in==out) and adds
        # a residual, so the FPN laterals below see the same shapes as FPNUNet.
        self.qbnet_bottleneck = QBNetBottleneck(
            in_channels=1024 // factor,
            n_qubits=n_qubits,
            n_layers=n_layers,
        )

        # ── FPN lateral connections (one per encoder stage) ───────────────
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
        self.outc = OutConv(fpn_channels, n_classes)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _upsample_add(self, top_down, lateral):
        """Upsample top-down feature map to lateral's spatial size and add."""
        return F.interpolate(
            top_down, size=lateral.shape[-2:], mode='bilinear', align_corners=True
        ) + lateral

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(self, x):
        # ── Encoder ───────────────────────────────────────────────────────
        x1 = self.inc(x)       # [B, 64,  512, 512]
        x2 = self.down1(x1)    # [B, 128, 256, 256]
        x3 = self.down2(x2)    # [B, 256, 128, 128]
        x4 = self.down3(x3)    # [B, 512,  64,  64]
        x5 = self.down4(x4)    # [B, 512,  32,  32]
        x5 = self.dropout(x5)  # bottleneck regularization

        # ── QB-Net quantum bottleneck (spatially aware, residual) ──────────
        x5 = self.qbnet_bottleneck(x5)  # [B, 512, 32, 32]

        # ── FPN lateral projections ───────────────────────────────────────
        l5 = self.lat5(x5)   # [B, fpn_ch, 32,  32 ]
        l4 = self.lat4(x4)   # [B, fpn_ch, 64,  64 ]
        l3 = self.lat3(x3)   # [B, fpn_ch, 128, 128]
        l2 = self.lat2(x2)   # [B, fpn_ch, 256, 256]
        l1 = self.lat1(x1)   # [B, fpn_ch, 512, 512]

        # ── Top-down pathway ──────────────────────────────────────────────
        p5 = self.smooth5(l5)
        p4 = self.smooth4(self._upsample_add(p5, l4))
        p3 = self.smooth3(self._upsample_add(p4, l3))
        p2 = self.smooth2(self._upsample_add(p3, l2))
        p1 = self.smooth1(self._upsample_add(p2, l1))

        # ── Fuse all pyramid levels at full (P1) resolution ───────────────
        target = p1.shape[-2:]
        fused = (
            p1
            + F.interpolate(p2, size=target, mode='bilinear', align_corners=True)
            + F.interpolate(p3, size=target, mode='bilinear', align_corners=True)
            + F.interpolate(p4, size=target, mode='bilinear', align_corners=True)
            + F.interpolate(p5, size=target, mode='bilinear', align_corners=True)
        )

        return self.outc(fused)   # [B, n_classes, 512, 512]
