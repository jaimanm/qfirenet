import torch
from torch import nn
import pennylane as qml

from models.unet import DoubleConv, Down, Up, OutConv
from circuits import get_circuit


def preprocess_quantum_input(x):
    """Clean input tensor for quantum circuit (replace NaN/inf).

    With AngleEmbedding there is no normalization step, so zero-row handling
    is no longer needed — AngleEmbedding simply encodes 0 as a zero rotation,
    which is a valid quantum state.  We still guard against NaN/inf that could
    theoretically arise from upstream linear layers with extreme weights during
    the first few training iterations.
    """
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)


class QuantumUNet(nn.Module):
    """U-Net with a quantum circuit at the bottleneck.

    The quantum circuit is selected via the config's 'circuit' key,
    which must match an entry in CIRCUIT_REGISTRY.
    """
    def __init__(self, n_classes, n_channels=13, bilinear=True, config=None):
        super().__init__()
        config = config or {}
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        n_qubits = config.get('n_qubits', 8)

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Quantum channel-attention bottleneck  (SE-net style)
        # x5: [B, bottleneck_ch, 32, 32] — same fix as QuantumFPNUNet.
        # GAP → FC1 → tanh → quantum → FC2 → sigmoid → channel rescaling.
        # Replaces the previous flatten/expand design that had a 65 000:1
        # compression ratio and produced NaN gradients via AmplitudeEmbedding.
        bottleneck_ch = 1024 // factor

        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(bottleneck_ch, n_qubits)

        qnode, weight_shapes = get_circuit(config)
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        self.fc2  = nn.Linear(n_qubits, bottleneck_ch)
        self.gate = nn.Sigmoid()

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)   # [B, bottleneck_ch, 32, 32]

        # Quantum channel-attention bottleneck
        s = self.gap(x5).squeeze(-1).squeeze(-1)  # [B, bottleneck_ch]
        s = torch.tanh(self.fc1(s))               # [B, n_qubits]  — scaled to (-1,1)
        s = preprocess_quantum_input(s)            # guard NaN/inf

        # Run circuit sample-by-sample (PennyLane doesn't support native batching)
        outputs = []
        for i in range(s.shape[0]):
            outputs.append(self.quantum_layer(s[i]).unsqueeze(0))
        s = torch.cat(outputs, dim=0)             # [B, n_qubits]

        s = self.gate(self.fc2(s))                # [B, bottleneck_ch]  — gate in (0,1)
        x5 = x5 * s.unsqueeze(-1).unsqueeze(-1)  # [B, bottleneck_ch, 32, 32]

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
