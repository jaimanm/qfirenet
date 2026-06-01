import torch
from torch import nn
import torch.nn.functional as F
import pennylane as qml

from models.unet import DoubleConv, Down, OutConv
from models.fpn_unet import LateralBlock, SmoothBlock
from models.quantum_unet import preprocess_quantum_input
from circuits import get_circuit


class QuantumFPNUNet(nn.Module):
    """U-Net encoder with a quantum bottleneck and an FPN decoder.

    Combines QuantumUNet's quantum bottleneck with FPNUNet's multi-scale
    decoder. The quantum circuit processes the deepest (32x32) feature map
    x5, where the spatial size is smallest and the semantics are richest.
    The FPN decoder then restores full-resolution spatial detail via a
    top-down pathway with lateral skip connections from the encoder.

    Architecture summary:
        Encoder   : DoubleConv + Down x4  (identical to ClassicalUNet)
        Bottleneck: Dropout2d → GAP → FC → tanh → quantum circuit → FC → sigmoid
                    → channel-attention rescaling of x5  (SE-net style)
        Decoder   : LateralBlock + SmoothBlock top-down FPN (identical to FPNUNet)
        Head      : OutConv (1×1 conv to n_classes)

    Pyramid levels (bilinear=True defaults):
        P5: from x5  (512 ch)  → 32×32
        P4: from x4  (512 ch)  → 64×64
        P3: from x3  (256 ch)  → 128×128
        P2: from x2  (128 ch)  → 256×256
        P1: from x1  (64 ch)   → 512×512

    Config keys consumed:
        circuit          (str)  : circuit name in CIRCUIT_REGISTRY [strongly_entangling]
        n_qubits         (int)  : number of qubits                 [8]
        n_layers         (int)  : variational layers in circuit     [2]
        fpn_out_channels (int)  : channel width of FPN pyramid      [128]
    """

    def __init__(self, n_classes, n_channels=13, bilinear=True, config=None):
        super().__init__()
        config = config or {}
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.bilinear = bilinear

        n_qubits = config.get('n_qubits', 8)
        fpn_channels = config.get('fpn_out_channels', 128)
        factor = 2 if bilinear else 1

        # ── Encoder (identical to ClassicalUNet / FPNUNet) ────────────────
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # ── Bottleneck regularization ─────────────────────────────────────
        self.dropout = nn.Dropout2d(p=0.5)

        # ── Quantum channel-attention bottleneck ──────────────────────────
        # x5 shape after down4: [B, 512, 32, 32]  (bilinear=True → factor=2 → 512 ch)
        #
        # Previous design flattened 512×32×32 = 524 288 features down to
        # 2^n_qubits = 256 values (2048:1 ratio) then tried to reconstruct
        # full spatial maps from n_qubits = 8 outputs — impossible, and the
        # adjoint gradient through AmplitudeEmbedding produced NaN losses.
        #
        # New design: SE-net style channel attention via quantum circuit.
        #   GAP  : [B, 512, 32, 32] → [B, 512]          (spatial squeeze)
        #   FC1  : [B, 512]         → [B, n_qubits]      (reasonable 64:1 ratio)
        #   Tanh : scale to [-1, 1] for AngleEmbedding
        #   QLayer: [B, n_qubits]   → [B, n_qubits]      (quantum transform)
        #   FC2  : [B, n_qubits]    → [B, 512]           (channel excitation)
        #   Sigmoid: gate in (0, 1)
        #   x5 = x5 * gate[:, :, None, None]             (channel-wise rescaling)
        #
        # Spatial structure is fully preserved; the quantum circuit only
        # modulates channel importance — a well-posed learning task.
        bottleneck_ch = 1024 // factor

        self.gap = nn.AdaptiveAvgPool2d(1)          # global average pool
        self.fc1 = nn.Linear(bottleneck_ch, n_qubits)

        qnode, weight_shapes = get_circuit(config)
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        self.fc2   = nn.Linear(n_qubits, bottleneck_ch)
        self.gate  = nn.Sigmoid()

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

    def _quantum_forward(self, x):
        """Run quantum circuit on each sample in the batch individually.

        PennyLane qnodes do not support native batching, so we loop over
        samples and concatenate results.  With the SE-net bottleneck the
        input is only [B, n_qubits] (e.g. 8 values), so this loop is cheap
        compared to the previous 524 288-wide flatten approach.

        Input  x : [B, n_qubits]  — tanh-scaled channel descriptors
        Output   : [B, n_qubits]  — PauliZ expectation values in [-1, 1]
        """
        outputs = []
        for i in range(x.shape[0]):
            out = self.quantum_layer(x[i])
            outputs.append(out.unsqueeze(0))
        return torch.cat(outputs, dim=0)

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(self, x):
        # ── Encoder ───────────────────────────────────────────────────────
        x1 = self.inc(x)       # [B, 64,  512, 512]
        x2 = self.down1(x1)    # [B, 128, 256, 256]
        x3 = self.down2(x2)    # [B, 256, 128, 128]
        x4 = self.down3(x3)    # [B, 512,  64,  64]
        x5 = self.down4(x4)    # [B, 512,  32,  32]
        x5 = self.dropout(x5)  # bottleneck regularization

        # ── Quantum channel-attention bottleneck ───────────────────────────
        # Squeeze: global average pool → channel descriptor
        s = self.gap(x5).squeeze(-1).squeeze(-1)  # [B, 512]
        # Compress to n_qubits; tanh maps to (-1,1) — valid AngleEmbedding range
        s = torch.tanh(self.fc1(s))               # [B, n_qubits]
        s = preprocess_quantum_input(s)            # guard against any NaN/inf
        # Quantum transform
        s = self._quantum_forward(s)              # [B, n_qubits]
        # Excite back to channel dimension and gate in (0, 1)
        s = self.gate(self.fc2(s))                # [B, 512]
        # Apply channel attention: rescale each channel map independently
        x5 = x5 * s.unsqueeze(-1).unsqueeze(-1)  # [B, 512, 32, 32]

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
