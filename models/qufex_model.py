import warnings
import torch
from torch import nn
import pennylane as qml

from models.unet import DoubleConv, Down, Up, OutConv
from circuits import get_circuit
 
 
# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
 
def preprocess_quantum_input(x):
    """Normalize features into [-π, π] for angle-embedding gates (RY/RZ).
 
    Replaces the old amplitude-embedding sanitiser (NaN/inf clamping, zero-row
    fix) with a simple tanh squeeze — appropriate for rotation-angle inputs.
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tanh(x) * torch.pi
 
 
# ---------------------------------------------------------------------------
# QuFeX bottleneck module
# ---------------------------------------------------------------------------
 
class QuFeXBottleneck(nn.Module):
    """QuFeX feature-extraction module placed at the U-Net bottleneck.

    Replaces the old flatten → fc1 → AmplitudeEmbedding → fc2 → unflatten
    chain with the architecture described in Jain & Kalev (2025):

        [B, C, H, W]
             │
        1×1 Convs →  staged compression: C → C//4 → n_qubits channels
        BN + ReLU
             │
        [B·H·W, n_qubits]  ←  reshape: process every spatial location
             │
        tanh(·)·π           ←  normalise to angle range
             │
        QuFeX circuit        ←  U1/U2 angle embedding + trainable θ + CNOT ring
             │
        Pauli-Z expvals      ←  n_qubits scalar outputs per location
             │
        [B, n_qubits, H, W] ←  restore spatial layout
             │
        1×1 Conv  →  restore C channels
        BN + ReLU
             │
        + residual (input x) ← bypass if quantum circuit learns nothing useful

    The 1×1 convolutions avoid the information bottleneck caused by the
    old linear projection to 2^n_qubits values (AmplitudeEmbedding requires
    an input vector of exactly 2^n entries, which forced an expensive
    flattening of the entire spatial feature map into a single vector per
    sample). Here each spatial location is encoded independently, using
    only n_qubits features — one per qubit — collected across channels.

    The staged compression (C → C//4 → n_qubits) preserves more of the
    encoder's learned representation compared to a single aggressive squeeze.
    The residual connection stabilises training by providing a gradient bypass
    if the quantum layer learns something unhelpful early on.
    """

    def __init__(self, in_channels: int, n_qubits: int = 8, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.in_channels = in_channels
        self.fallback_count = 0   # tracks how many times quantum failed at runtime

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
            # Step 3 — final squeeze: C//4 → n_qubits
            nn.Conv2d(mid, n_qubits, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_qubits),
            nn.ReLU(inplace=True),
        )

        # --- Quantum layer ---
        config = {"n_qubits": n_qubits, "n_layers": n_layers, "circuit": "qufex_circuit"}
        qnode, weight_shapes = get_circuit(config)
        self.qnode = qnode
        self.weights_u1 = nn.Parameter(torch.randn(*weight_shapes['weights_u1']) * 0.01)
        self.weights_u2 = nn.Parameter(torch.randn(*weight_shapes['weights_u2']) * 0.01)
 
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

        # 3. Normalise to angle range [-π, π]
        x_q = preprocess_quantum_input(x_q)

        # 4. Run QuFeX circuit on each spatial location
        # .cpu() needed as default.qubit runs on CPU; .detach() removed so
        # gradients flow back through pre_conv (paper trains end-to-end).
        result = self.qnode(x_q.cpu().T, self.weights_u1.cpu(), self.weights_u2.cpu())
        x_q = torch.stack(result, dim=1).to(x.device).float()
 
        # 5. Restore spatial layout: [B·H·W, n_qubits] → [B, n_qubits, H, W]
        x_q = x_q.reshape(B, H, W, self.n_qubits).permute(0, 3, 1, 2)
 
        # 6. Restore channel count and add residual connection
        # Paper: "y = Q(x) + x where Q(x) represents the output of the quantum layer"
        x_out = self.post_conv(x_q) + x
        if torch.isnan(x_out).any() or torch.isinf(x_out).any():
            raise ValueError("Quantum output contains NaN/Inf")
        return x_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run quantum path. If quantum fails for any reason, fall back to
        returning x unchanged so the U-Net decoder still receives a valid
        feature map via the residual connection."""
        try:
            return self._quantum_forward(x)
        except Exception as e:
            self.fallback_count += 1
            warnings.warn(
                f"[QuFeXBottleneck] Quantum failed (fallback #{self.fallback_count}): "
                f"{type(e).__name__}: {e}. Returning classical residual."
            )
            return x


# ---------------------------------------------------------------------------
# Qu-Net  (QuFeX at the bottleneck of a classical U-Net)
# ---------------------------------------------------------------------------
 
class QuNet(nn.Module):
    """Hybrid quantum-classical U-Net using QuFeX at the bottleneck.
 
    Implements Qu-Net from Jain & Kalev (2025). The encoder/decoder and skip
    connections are identical to the classical U-Net; only the bottleneck is
    replaced by a QuFeXBottleneck module.
 
    Config keys
    -----------
    n_qubits : int   Number of qubits (must be even). Default 8.
                     Paper configs: Qu-Net 8(1) = 8 qubits, 1 layer
                                    Qu-Net 4(2) = 4 qubits, 2 layers
    n_layers : int   Trainable rotation layers inside QuFeX. Default 2.
    """
 
    def __init__(self, n_classes: int, n_channels: int = 13,
                 bilinear: bool = True, config: dict = None):
        super().__init__()
        config = config or {}
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
 
        n_qubits = config.get("n_qubits", 8)
        n_layers = config.get("n_layers", 2)
        factor = 2 if bilinear else 1
 
        # --- Encoder ---
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
 
        # --- QuFeX bottleneck ---
        self.qufex_bottleneck = QuFeXBottleneck(
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
 
        # QuFeX bottleneck
        x5 = self.qufex_bottleneck(x5)
 
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)