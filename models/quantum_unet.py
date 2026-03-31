import torch
from torch import nn
import pennylane as qml

from models.unet import DoubleConv, Down, Up, OutConv
from circuits import get_circuit


def preprocess_quantum_input(x):
    """Clean input tensor for quantum circuit (replace NaN/inf, handle zero rows)."""
    batch = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    is_all_zero = (batch.abs().sum(dim=1) < 1e-6)
    for i, zero_row in enumerate(is_all_zero):
        if zero_row:
            batch[i] = 0
            batch[i, 0] = 1.0
    return batch


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

        # Quantum bottleneck
        quantum_input_shape = 2 ** n_qubits
        H, W = 32, 32  # 512 / (2^4 pooling stages)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((1024 // factor) * H * W, quantum_input_shape)

        circuit_name = config.get('circuit', 'strongly_entangling')
        qnode, weight_shapes = get_circuit(config)
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        self.fc2 = nn.Linear(n_qubits, (1024 // factor) * H * W)
        self.unflatten = nn.Unflatten(1, (1024 // factor, H, W))

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
        x5 = self.down4(x4)

        # Quantum bottleneck
        x_flat = self.flatten(x5)
        x_fc = self.fc1(x_flat)
        x_fc_clean = preprocess_quantum_input(x_fc)

        # Process each sample individually (AmplitudeEmbedding doesn't support batching)
        batch_size = x_fc_clean.shape[0]
        quantum_outputs = []
        for i in range(batch_size):
            single_output = self.quantum_layer(x_fc_clean[i])
            quantum_outputs.append(single_output.unsqueeze(0))
        x_quantum = torch.cat(quantum_outputs, dim=0)

        x_fc2 = self.fc2(x_quantum)
        x5 = self.unflatten(x_fc2)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
