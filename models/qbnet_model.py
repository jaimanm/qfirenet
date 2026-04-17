import torch
from torch import nn

from models.unet import DoubleConv, Down, Up, OutConv
from circuits import get_circuit


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_quantum_input(x):
    """Normalize features into [-π, π] for RY angle embedding.

    Uses tanh squashing — appropriate for rotation-angle inputs and avoids
    the NaN/inf issues that arise with amplitude embedding.
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tanh(x) * torch.pi


# ---------------------------------------------------------------------------
# QB-Net bottleneck module
# ---------------------------------------------------------------------------

class QBNetBottleneck(nn.Module):
    """QB-Net quantum bottleneck module — Xu, Zhao & Li (2025), HiQC paper.

    Published in: Quantum Machine Intelligence 7, 97 (2025).
    DOI: https://doi.org/10.1007/s42484-025-00321-0

    Replaces the classical double-conv bottleneck in a U-Net with a hybrid
    quantum-classical module. The design mirrors QuFeXBottleneck but uses the
    QB-Net circuit (U3 gates + open CNOT ladder) in place of the QuFeX circuit
    (split U1/U2 + CZ ring).

    Architecture
    ------------

        [B, C, H, W]
             │
        3×3 Conv  →  C → C//2  (spatial context)
        1×1 Conv  →  C//2 → C//4
        1×1 Conv  →  C//4 → n_qubits   (channel compression)
        BN + ReLU at each step
             │
        [B·H·W, n_qubits]   ←  per-spatial-location processing
             │
        tanh(·)·π            ←  normalise to RY angle range
             │
        QB-Net circuit        ←  RY embedding + U3 layers + CNOT ladder
             │
        Pauli-Z expvals       ←  n_qubits scalars per location
             │
        [B, n_qubits, H, W]  ←  restore spatial layout
             │
        1×1 Conv  →  n_qubits → C//4
        3×3 Conv  →  C//4 → C          (symmetric expansion)
        BN + ReLU at each step
             │
        + residual (input x)  ←  gradient bypass if quantum layer unhelpful

    Key differences from QuFeXBottleneck
    -------------------------------------
    - Uses QB-Net circuit (RY + U3 + CNOT ladder) instead of QuFeX circuit.
    - Single `self.weights` parameter instead of `self.weights_u1` / `self.weights_u2`.
    - Weight count: QB-Net = n_layers × n_qubits × 3
                    QuFeX  = n_layers × n_qubits × 2  (same total when split)

    Parameters
    ----------
    in_channels : int   Number of channels at the bottleneck (e.g. 512).
    n_qubits    : int   Number of qubits. Default 4.
    n_layers    : int   Variational layers inside the QB-Net circuit. Default 2.
    """

    def __init__(self, in_channels: int, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.in_channels = in_channels

        mid = max(in_channels // 4, n_qubits)   # safe intermediate width

        # --- Pre-quantum: staged compression (3×3 → 1×1 → 1×1) ---
        self.pre_conv = nn.Sequential(
            # Step 1 — spatial context: C → C//2
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            # Step 2 — channel squeeze: C//2 → C//4
            nn.Conv2d(in_channels // 2, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            # Step 3 — final squeeze to qubit count: C//4 → n_qubits
            nn.Conv2d(mid, n_qubits, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_qubits),
            nn.ReLU(inplace=True),
        )

        # --- QB-Net quantum layer ---
        config = {
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "circuit": "qbnet_circuit",
        }
        qnode, weight_shapes = get_circuit(config)
        self.qnode = qnode
        # Single unified weight tensor (vs. weights_u1 + weights_u2 in QuFeX)
        self.weights = nn.Parameter(
            torch.randn(*weight_shapes["weights"]) * 0.01
        )

        # --- Post-quantum: symmetric expansion (1×1 → 3×3) ---
        self.post_conv = nn.Sequential(
            # Step 1 — channel lift: n_qubits → C//4
            nn.Conv2d(n_qubits, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            # Step 2 — spatial restore: C//4 → C
            nn.Conv2d(mid, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def _quantum_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress → quantum → restore, returning [B, C, H, W]."""
        B, C, H, W = x.shape

        # 1. Staged channel compression: [B, C, H, W] → [B, n_qubits, H, W]
        x_q = self.pre_conv(x)

        # 2. Flatten spatial dims for per-location quantum processing:
        #    [B, n_qubits, H, W] → [B·H·W, n_qubits]
        x_q = x_q.permute(0, 2, 3, 1).reshape(B * H * W, self.n_qubits)

        # 3. Normalise to angle range [-π, π] for RY embedding
        x_q = preprocess_quantum_input(x_q)

        # 4. Run QB-Net circuit on each spatial location.
        #    Transpose to [n_qubits, B·H·W] so PennyLane indexes qubit-first.
        #    .cpu() required as lightning.qubit runs on CPU.
        result = self.qnode(x_q.cpu().T, self.weights.cpu())
        x_q = torch.stack(result, dim=1).to(x.device).float()

        # 5. Restore spatial layout: [B·H·W, n_qubits] → [B, n_qubits, H, W]
        x_q = x_q.reshape(B, H, W, self.n_qubits).permute(0, 3, 1, 2)

        # 6. Restore channel count and add residual connection
        return self.post_conv(x_q) + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantum_forward(x)


# ---------------------------------------------------------------------------
# QB-Net  (QB-Net bottleneck inside a classical U-Net)
# ---------------------------------------------------------------------------

class QBNet(nn.Module):
    """Hybrid quantum-classical U-Net using QB-Net at the bottleneck.

    Implements the QB-Net architecture from Xu, Zhao & Li (2025). The
    encoder/decoder and skip connections are identical to the classical U-Net;
    only the bottleneck is replaced by a QBNetBottleneck module.

    This is a direct counterpart to QuNet from Jain & Kalev (2025), using the
    QB-Net circuit in place of the QuFeX circuit, enabling a controlled
    comparison between the two quantum ansätze on the same hybrid U-Net host.

    Config keys
    -----------
    n_qubits : int   Number of qubits. Default 4.
    n_layers : int   Variational layers inside QB-Net circuit. Default 2.

    Comparison with QuNet
    ---------------------
    Component       QuNet (Jain & Kalev 2025)   QBNet (Xu et al. 2025)
    --------------- --------------------------- -----------------------
    Encoding        H + RZ (X-basis)            RY
    Trainable gates RX+RZ / RX+RY (split)       U3 (uniform)
    Entanglement    CZ ring (wrap-around)        CNOT ladder (open chain)
    Params (4q,2L)  16                           24
    """

    def __init__(self, n_classes: int, n_channels: int = 13,
                 bilinear: bool = True, config: dict = None):
        super().__init__()
        config = config or {}
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        n_qubits = config.get("n_qubits", 4)
        n_layers = config.get("n_layers", 2)
        factor = 2 if bilinear else 1

        # --- Encoder ---
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # --- QB-Net bottleneck ---
        self.qbnet_bottleneck = QBNetBottleneck(
            in_channels=1024 // factor,
            n_qubits=n_qubits,
            n_layers=n_layers,
        )

        # --- Decoder ---
        self.up1  = Up(1024, 512 // factor, bilinear)
        self.up2  = Up(512,  256 // factor, bilinear)
        self.up3  = Up(256,  128 // factor, bilinear)
        self.up4  = Up(128,  64,            bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # QB-Net bottleneck
        x5 = self.qbnet_bottleneck(x5)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)