import torch
from torch import nn
import pennylane as qml

from circuits import get_circuit
from models.unet import DoubleConv, Down, OutConv, Up


def preprocess_quantum_input(x):
    """Clean input tensor for quantum circuit (replace NaN/inf, handle zero rows)."""
    batch = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    is_all_zero = batch.abs().sum(dim=1) < 1e-6
    for i, zero_row in enumerate(is_all_zero):
        if zero_row:
            batch[i] = 0
            batch[i, 0] = 1.0
    return batch


class CompactQuantumUNet(nn.Module):
    """Width-scaled U-Net with a compressed quantum bottleneck."""

    def __init__(self, n_classes, n_channels=13, bilinear=True, config=None):
        super().__init__()
        config = config or {}

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channels = config.get('base_channels', 48)

        # Scale the encoder/decoder width the same way as compact_unet.
        width_1 = self.base_channels
        width_2 = self.base_channels * 2
        width_3 = self.base_channels * 4
        width_4 = self.base_channels * 8
        factor = 2 if bilinear else 1
        bottleneck_width = (self.base_channels * 16) // factor

        self.inc = DoubleConv(n_channels, width_1)
        self.down1 = Down(width_1, width_2)
        self.down2 = Down(width_2, width_3)
        self.down3 = Down(width_3, width_4)
        self.down4 = Down(width_4, bottleneck_width)

        n_qubits = config.get('n_qubits', 8)
        pooled_size = config.get('quantum_bottleneck_spatial_size', 4)
        compressed_channels = config.get('quantum_bottleneck_channels', 64)

        # Shrink the bottleneck before flattening so the dense quantum interface stays manageable.
        self.pre_quantum_conv = nn.Conv2d(bottleneck_width, compressed_channels, kernel_size=1)
        self.pre_quantum_pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
        self.flatten = nn.Flatten()

        compressed_features = compressed_channels * pooled_size * pooled_size
        quantum_input_shape = 2 ** n_qubits
        self.fc1 = nn.Linear(compressed_features, quantum_input_shape)

        qnode, weight_shapes = get_circuit(config)
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # Rebuild a decoder-sized bottleneck after the quantum layer.
        self.fc2 = nn.Linear(n_qubits, compressed_features)
        self.unflatten = nn.Unflatten(1, (compressed_channels, pooled_size, pooled_size))
        self.post_quantum_upsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=True)
        self.post_quantum_conv = nn.Conv2d(compressed_channels, bottleneck_width, kernel_size=1)

        self.up1 = Up(width_4 * 2, width_4 // factor, bilinear)
        self.up2 = Up(width_3 * 2, width_3 // factor, bilinear)
        self.up3 = Up(width_2 * 2, width_2 // factor, bilinear)
        self.up4 = Up(width_1 * 2, width_1, bilinear)
        self.outc = OutConv(width_1, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Compress the bottleneck before sending it through the quantum circuit.
        x_compressed = self.pre_quantum_conv(x5)
        x_compressed = self.pre_quantum_pool(x_compressed)
        x_flat = self.flatten(x_compressed)
        x_fc = self.fc1(x_flat)
        x_fc_clean = preprocess_quantum_input(x_fc)

        # Process samples one at a time
        quantum_outputs = []
        for i in range(x_fc_clean.shape[0]):
            single_output = self.quantum_layer(x_fc_clean[i])
            quantum_outputs.append(single_output.unsqueeze(0))
        x_quantum = torch.cat(quantum_outputs, dim=0)

        # Expand the quantum output back into a spatial bottleneck for the decoder.
        x_fc2 = self.fc2(x_quantum)
        x_reconstructed = self.unflatten(x_fc2)
        x_reconstructed = self.post_quantum_upsample(x_reconstructed)
        x5 = self.post_quantum_conv(x_reconstructed)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
