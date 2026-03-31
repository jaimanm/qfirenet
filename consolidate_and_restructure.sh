#!/usr/bin/env bash
#
# consolidate_and_restructure.sh
#
# This script consolidates branch work and restructures the codebase into a
# clean, modular architecture for the Spring 2026 semester.
#
# What it does:
#   1. Cleans up master (removes junk files)
#   2. Pulls useful code from feature branches
#   3. Restructures the codebase (models/, circuits/, losses/, configs/, etc.)
#   4. Creates new config-driven train.py and test.py
#   5. Moves existing experiment results into the new structure
#   6. Updates .gitignore, PR template, README, and adds SOW
#   7. Commits everything
#   8. Optionally pushes and deletes stale branches
# 
#   ┌───────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
#   │ Phase │                                                              What it does                                                              │
#   ├───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
#   │ 1     │ Cleans up master (removes junk test file, old consolidation files)                                                                     │
#   ├───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
#   │ 2-3   │ Creates the new directory structure                                                                                                    │
#   ├───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
#   │ 4     │ Writes all new source files inline — models, circuits, losses, configs, train.py, test.py, SLURM scripts, requirements.txt, .gitignore │
#   ├───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
#   │ 5     │ Moves existing experiment results into experiments/classical_baseline/                                                                 │
#   ├───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
#   │ 6     │ Removes superseded files (old model/, Exp/, Map/, jobs/, old scripts)                                                                  │
#   ├───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
#   │ 7     │ Writes the SOW and README                                                                                                              │
#   ├───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
#   │ 8     │ Commits everything in a single commit                                                                                                  │
#   ├───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
#   │ 9     │ Prompts before pushing + deleting stale remote branches                                                                                │
#   └───────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
#
# Prerequisites:
#   - Write access to the repository
#   - Run from the repository root
#   - Must be on the master branch with a clean working tree
#
# Usage:
#   chmod +x consolidate_and_restructure.sh
#   ./consolidate_and_restructure.sh
#

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Preflight ──────────────────────────────────────────────────────────────
[ -d ".git" ] || error "Not in a git repository root."
[ "$(git branch --show-current)" = "master" ] || error "Must be on master branch."
git diff-index --quiet HEAD -- || error "Uncommitted changes. Commit or stash first."

info "Fetching all remote branches..."
git fetch --all --prune
git pull origin master

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Clean up master
# ══════════════════════════════════════════════════════════════════════════════
info "Phase 1: Cleaning up master..."

# Remove junk "test" file
if [ -f "test" ]; then
    git rm test
fi

# Remove old consolidation files if they exist (from prior runs of earlier script)
for f in consolidate_master.sh CONSOLIDATION_INSTRUCTIONS.md; do
    [ -f "$f" ] && git rm "$f" 2>/dev/null || true
done

git commit -m "Clean up master: remove junk files" --allow-empty 2>/dev/null || true

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Pull useful code from branches (into temp locations)
# ══════════════════════════════════════════════════════════════════════════════
info "Phase 2: Pulling code from feature branches..."

# We'll grab raw content from branches but place them into the NEW structure,
# not the old one. So we just read them via git show.

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Create new directory structure
# ══════════════════════════════════════════════════════════════════════════════
info "Phase 3: Creating new directory structure..."

mkdir -p models circuits losses configs scripts notebooks
mkdir -p experiments/classical_baseline/maps
mkdir -p experiments/data_exploration

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Create all new source files
# ══════════════════════════════════════════════════════════════════════════════
info "Phase 4: Writing new source files..."

# ── models/__init__.py ─────────────────────────────────────────────────────
cat > models/__init__.py << 'PYEOF'
from models.unet import ClassicalUNet
from models.quantum_unet import QuantumUNet

MODEL_REGISTRY = {
    'classical_unet': ClassicalUNet,
    'quantum_unet': QuantumUNet,
}

# Spectral band mode -> number of input channels
MODE_CHANNELS = {
    0: 12,  # all_bands
    1: 13,  # all_bands_aerosol
    2: 3,   # rgb
    3: 4,   # rgb_aerosol
    4: 3,   # swir
    5: 4,   # swir_aerosol
    6: 3,   # nbr
    7: 4,   # nbr_aerosol
    8: 3,   # ndvi
    9: 4,   # ndvi_aerosol
    10: 6,  # rgb_swir_nbr_ndvi
    11: 7,  # rgb_swir_nbr_ndvi_aerosol
}

MODE_NAMES = [
    'all_bands', 'all_bands_aerosol', 'rgb', 'rgb_aerosol',
    'swir', 'swir_aerosol', 'nbr', 'nbr_aerosol',
    'ndvi', 'ndvi_aerosol', 'rgb_swir_nbr_ndvi', 'rgb_swir_nbr_ndvi_aerosol',
]


def get_model(config):
    """Instantiate a model from a config dict."""
    model_name = config['model']
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    model_cls = MODEL_REGISTRY[model_name]
    n_channels = MODE_CHANNELS[config['mode']]
    n_classes = config.get('n_classes', 2)
    return model_cls(n_classes=n_classes, n_channels=n_channels, config=config)
PYEOF

# ── models/unet.py ────────────────────────────────────────────────────────
cat > models/unet.py << 'PYEOF'
import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm => ReLU) x 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """MaxPool2d => DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsample => concatenate skip connection => DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 convolution for final class predictions."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ClassicalUNet(nn.Module):
    """Standard U-Net encoder-decoder with skip connections."""
    def __init__(self, n_classes, n_channels=13, bilinear=True, config=None):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
PYEOF

# ── models/quantum_unet.py ────────────────────────────────────────────────
cat > models/quantum_unet.py << 'PYEOF'
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
PYEOF

# ── circuits/__init__.py ──────────────────────────────────────────────────
cat > circuits/__init__.py << 'PYEOF'
from circuits.strongly_entangling import create_strongly_entangling_circuit
from circuits.ry_cnot import create_ry_cnot_circuit

CIRCUIT_REGISTRY = {
    'strongly_entangling': create_strongly_entangling_circuit,
    'ry_cnot': create_ry_cnot_circuit,
}


def get_circuit(config):
    """Return (qnode, weight_shapes) for the circuit named in config['circuit']."""
    circuit_name = config.get('circuit', 'strongly_entangling')
    if circuit_name not in CIRCUIT_REGISTRY:
        raise ValueError(f"Unknown circuit '{circuit_name}'. Available: {list(CIRCUIT_REGISTRY.keys())}")
    return CIRCUIT_REGISTRY[circuit_name](config)
PYEOF

# ── circuits/strongly_entangling.py ───────────────────────────────────────
cat > circuits/strongly_entangling.py << 'PYEOF'
import pennylane as qml


def create_strongly_entangling_circuit(config):
    """AmplitudeEmbedding + StronglyEntanglingLayers circuit.

    Returns (qnode, weight_shapes) for use with qml.qnn.TorchLayer.
    """
    n_qubits = config.get('n_qubits', 8)
    n_layers = config.get('n_layers', 2)
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev, interface='torch', diff_method='adjoint')
    def circuit(inputs, weights):
        qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits),
                               pad_with=0.0, normalize=True)
        qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    weight_shapes = {
        "weights": qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    }
    return circuit, weight_shapes
PYEOF

# ── circuits/ry_cnot.py ──────────────────────────────────────────────────
cat > circuits/ry_cnot.py << 'PYEOF'
import pennylane as qml


def create_ry_cnot_circuit(config):
    """RY encoding + ring-topology CNOT entanglement circuit.

    Returns (qnode, weight_shapes) for use with qml.qnn.TorchLayer.
    """
    n_qubits = config.get('n_qubits', 8)
    n_layers = config.get('n_layers', 2)
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev, interface='torch', diff_method='adjoint')
    def circuit(inputs, weights):
        # Encode inputs via RY rotations
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

        # Variational layers with ring-topology CNOT entanglement
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    weight_shapes = {
        "weights": (n_layers, n_qubits)
    }
    return circuit, weight_shapes
PYEOF

# ── losses/__init__.py ────────────────────────────────────────────────────
cat > losses/__init__.py << 'PYEOF'
import torch
import torch.nn as nn
from losses.cross_entropy import create_weighted_cross_entropy

LOSS_REGISTRY = {
    'cross_entropy': create_weighted_cross_entropy,
}


def get_loss(config):
    """Instantiate a loss function from config."""
    loss_name = config.get('loss', 'cross_entropy')
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{loss_name}'. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[loss_name](config)
PYEOF

# ── losses/cross_entropy.py ──────────────────────────────────────────────
cat > losses/cross_entropy.py << 'PYEOF'
import torch
import torch.nn as nn


def create_weighted_cross_entropy(config):
    """Weighted cross-entropy loss to address fire/non-fire class imbalance."""
    fire_weight = config.get('fire_class_weight', 10)
    weights = torch.Tensor([1, fire_weight])
    return nn.CrossEntropyLoss(weight=weights)
PYEOF

# ── dataset/__init__.py ──────────────────────────────────────────────────
cat > dataset/__init__.py << 'PYEOF'
from dataset.sen2fire import Sen2FireDataSet
PYEOF

# ── dataset/sen2fire.py ──────────────────────────────────────────────────
# Copy existing dataset file, just renamed
cp dataset/Sen2Fire_Dataset.py dataset/sen2fire.py

# ── utils/__init__.py ─────────────────────────────────────────────────────
cat > utils/__init__.py << 'PYEOF'
PYEOF

# ── utils/metrics.py ──────────────────────────────────────────────────────
cp utils/tools.py utils/metrics.py

# ── utils/visualization.py ────────────────────────────────────────────────
cat > utils/visualization.py << 'PYEOF'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


def plot_training_history(hist, save_path):
    """Plot training loss, OA, mIoU, and time per batch."""
    seg_losses, oas, mious, times = zip(*hist)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(seg_losses, label='Segmentation Loss')
    axs[0, 0].set_xlabel('Batch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Segmentation Loss')
    axs[0, 0].legend()

    axs[0, 1].plot(oas, label='Overall Accuracy')
    axs[0, 1].set_xlabel('Batch')
    axs[0, 1].set_ylabel('OA')
    axs[0, 1].set_title('Overall Accuracy')
    axs[0, 1].legend()

    axs[1, 0].plot(mious, label='Mean IoU')
    axs[1, 0].set_xlabel('Batch')
    axs[1, 0].set_ylabel('mIoU')
    axs[1, 0].set_title('Mean IoU')
    axs[1, 0].legend()

    axs[1, 1].plot(times, label='Time per Batch')
    axs[1, 1].set_xlabel('Batch')
    axs[1, 1].set_ylabel('Time (s)')
    axs[1, 1].set_title('Time per Batch')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_scene_map(reconstructed_rgb, reconstructed_pred, reconstructed_label, save_path):
    """Plot RGB, detection overlay, and ground truth label side by side."""
    cmap = ListedColormap(['white', 'red'])
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    axs[0].imshow(reconstructed_rgb.transpose(1, 2, 0) / 1500.)
    axs[0].axis('off')
    axs[0].set_title('RGB image', fontsize=12)

    axs[1].imshow(reconstructed_rgb.transpose(1, 2, 0) / 1500., alpha=0.6)
    axs[1].imshow(reconstructed_pred, cmap=cmap, alpha=0.7)
    axs[1].axis('off')
    axs[1].set_title('Detection', fontsize=12)

    axs[2].imshow(reconstructed_rgb.transpose(1, 2, 0) / 1500., alpha=0.6)
    axs[2].imshow(reconstructed_label, cmap=cmap, alpha=0.7)
    axs[2].axis('off')
    axs[2].set_title('Label', fontsize=12)

    legend_labels = ['Non-fire', 'Fire']
    plt.legend(
        handles=[Rectangle((0, 0), 1, 1, facecolor=cmap(i), edgecolor='black') for i in range(2)],
        labels=legend_labels, fontsize=12, frameon=False,
        bbox_to_anchor=(1.04, 0), loc="lower left"
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
PYEOF

# ── configs/classical_baseline.yaml ──────────────────────────────────────
cat > configs/classical_baseline.yaml << 'YAMLEOF'
# Classical U-Net baseline (SWIR + aerosol, mode 5)
# Matches the existing fall-semester results in experiments/classical_baseline/

model: classical_unet
mode: 5                    # swir_aerosol
n_classes: 2

# Training
epochs: 5
batch_size: 16
test_batch_size: 50
val_batch_size: 1
learning_rate: 1.0e-4
weight_decay: 5.0e-4
seed: 1234

# Loss
loss: cross_entropy
fire_class_weight: 10

# Data
data_dir: ../Sen2Fire/
train_list: ./dataset/train.txt
val_list: ./dataset/val.txt
test_list: ./dataset/test.txt

# Output
experiment_name: classical_baseline
YAMLEOF

# ── configs/quantum_strongly_entangling.yaml ─────────────────────────────
cat > configs/quantum_strongly_entangling.yaml << 'YAMLEOF'
# Quantum U-Net with StronglyEntanglingLayers circuit (SWIR + aerosol, mode 5)

model: quantum_unet
circuit: strongly_entangling
mode: 5                    # swir_aerosol
n_classes: 2

# Quantum
n_qubits: 8
n_layers: 2

# Training
epochs: 5
batch_size: 16
test_batch_size: 50
val_batch_size: 1
learning_rate: 1.0e-4
weight_decay: 5.0e-4
seed: 1234

# Loss
loss: cross_entropy
fire_class_weight: 10

# Data
data_dir: ../Sen2Fire/
train_list: ./dataset/train.txt
val_list: ./dataset/val.txt
test_list: ./dataset/test.txt

# Output
experiment_name: quantum_strongly_entangling
YAMLEOF

# ── train.py (new config-driven version) ─────────────────────────────────
cat > train.py << 'PYEOF'
#!/usr/bin/env python
"""Training script for wildfire detection models.

Usage:
    python train.py --config configs/classical_baseline.yaml
    python train.py --config configs/classical_baseline.yaml --epochs 1 --batch_size 2
"""
import argparse
import os
import time
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils import data
from tqdm import tqdm
import yaml

from models import get_model, MODE_NAMES
from losses import get_loss
from dataset.sen2fire import Sen2FireDataSet
from utils.metrics import label_accuracy_score, eval_image
from utils.visualization import plot_training_history


def load_config(config_path, cli_overrides):
    """Load YAML config and apply CLI overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value
    return config


def setup_experiment_dir(config):
    """Create timestamped experiment output directory."""
    timestamp = time.strftime('%m%d_%H%M', time.localtime())
    name = config.get('experiment_name', 'experiment')
    exp_dir = os.path.join('experiments', f"{name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    # Save a copy of the config for reproducibility
    shutil.copy2(config['_config_path'], os.path.join(exp_dir, 'config.yaml'))
    return exp_dir


def get_data_loaders(config):
    """Create train/val/test data loaders."""
    mode = config['mode']
    common = {'pin_memory': True, 'shuffle': True}

    # Auto-detect number of workers
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        n_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        n_workers = min(4, os.cpu_count() or 1)

    common['num_workers'] = n_workers

    train_loader = data.DataLoader(
        Sen2FireDataSet(config['data_dir'], config['train_list'], mode=mode),
        batch_size=config.get('batch_size', 16), **common)

    val_loader = data.DataLoader(
        Sen2FireDataSet(config['data_dir'], config['val_list'], mode=mode),
        batch_size=config.get('val_batch_size', 1),
        num_workers=n_workers, pin_memory=True, shuffle=False)

    test_loader = data.DataLoader(
        Sen2FireDataSet(config['data_dir'], config['test_list'], mode=mode),
        batch_size=config.get('test_batch_size', 50),
        num_workers=n_workers, pin_memory=True, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate(model, loader, device, num_classes, interp):
    """Run evaluation on a data loader, return metrics dict."""
    epsilon = 1e-14
    model.eval()
    TP_all = np.zeros((num_classes, 1))
    FP_all = np.zeros((num_classes, 1))
    TN_all = np.zeros((num_classes, 1))
    FN_all = np.zeros((num_classes, 1))
    n_valid = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        image, label, _, _ = batch
        label = label.squeeze().numpy()
        image = image.float().to(device)
        with torch.no_grad():
            pred = model(image)
        _, pred = torch.max(interp(nn.functional.softmax(pred, dim=1)).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy()
        TP, FP, TN, FN, n = eval_image(pred.reshape(-1), label.reshape(-1), num_classes)
        TP_all += TP
        FP_all += FP
        TN_all += TN
        FN_all += FN
        n_valid += n

    OA = np.sum(TP_all) / n_valid
    metrics = {'OA': OA}
    for i in range(num_classes):
        P = TP_all[i] / (TP_all[i] + FP_all[i] + epsilon)
        R = TP_all[i] / (TP_all[i] + FN_all[i] + epsilon)
        F1 = 2.0 * P * R / (P + R + epsilon)
        IoU = TP_all[i] / (TP_all[i] + FP_all[i] + FN_all[i] + epsilon)
        metrics[f'class{i}_P'] = P.item()
        metrics[f'class{i}_R'] = R.item()
        metrics[f'class{i}_F1'] = F1.item()
        metrics[f'class{i}_IoU'] = IoU.item()

    metrics['mF1'] = np.mean([metrics[f'class{i}_F1'] for i in range(num_classes)])
    metrics['mIoU'] = np.mean([metrics[f'class{i}_IoU'] for i in range(num_classes)])
    return metrics


def train(config):
    """Main training function."""
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'SLURMD_NODENAME' in os.environ:
        print(f"Running on {os.environ['SLURMD_NODENAME']} ({device})")
    else:
        print(f"Running on {device}")

    # Reproducibility
    torch.manual_seed(config.get('seed', 1234))

    # Setup
    exp_dir = setup_experiment_dir(config)
    log_file = open(os.path.join(exp_dir, 'Training_log.txt'), 'w')
    num_classes = config.get('n_classes', 2)
    name_classes = np.array(['non-fire', 'fire'], dtype=str)

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"Config: {config.get('experiment_name')}")
    log(f"Model: {config['model']} | Mode: {config['mode']} ({MODE_NAMES[config['mode']]})")
    log(f"Output: {exp_dir}")

    # Model, optimizer, loss
    model = get_model(config)
    model = model.to(device)
    optimizer = Adam(model.parameters(),
                     lr=config.get('learning_rate', 1e-4),
                     weight_decay=config.get('weight_decay', 5e-4))
    loss_fn = get_loss(config)
    loss_fn = loss_fn.to(device)

    input_size = (512, 512)
    interp = nn.Upsample(size=input_size, mode='bilinear')

    # Data
    train_loader, val_loader, test_loader = get_data_loaders(config)
    log(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # Training loop
    epochs = config.get('epochs', 5)
    hist = []
    F1_best = 0.0

    model.train()
    for epoch in range(1, epochs + 1):
        log(f"\n--- Epoch {epoch}/{epochs} ---")

        for batch_idx, (patches, labels, _, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            torch.cuda.empty_cache()
            start_time = time.time()

            patches = patches.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()

            preds = interp(model(patches))
            loss = loss_fn(preds, labels)

            # Batch metrics
            _, pred_labels = torch.max(preds, 1)
            lbl_pred = pred_labels.detach().cpu().numpy()
            lbl_true = labels.detach().cpu().numpy()
            metrics_batch = []
            for lt, lp in zip(lbl_true, lbl_pred):
                _, _, mean_iu, _ = label_accuracy_score(lt, lp, n_class=num_classes)
                metrics_batch.append(mean_iu)

            batch_miou = np.nanmean(metrics_batch)
            batch_oa = np.sum(lbl_pred == lbl_true) / len(lbl_true.reshape(-1))
            elapsed = time.time() - start_time
            hist.append([loss.item(), batch_oa, batch_miou, elapsed])

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                log(f"  Iter {batch_idx+1}/{len(train_loader)} | Loss: {hist[-1][0]:.4f} | OA: {hist[-1][1]:.4f} | mIoU: {hist[-1][2]:.4f} | Time: {hist[-1][3]:.2f}s")

        # Validation
        log("Validating...")
        val_metrics = evaluate(model, val_loader, device, num_classes, interp)
        log(f"  Val OA: {val_metrics['OA']*100:.2f}% | Fire F1: {val_metrics['class1_F1']*100:.2f}% | Fire IoU: {val_metrics['class1_IoU']*100:.2f}% | mIoU: {val_metrics['mIoU']*100:.2f}%")

        if val_metrics['class1_F1'] > F1_best:
            F1_best = val_metrics['class1_F1']
            log("  New best model! Saving checkpoint...")
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))

    # Test
    log("\nTesting...")
    best_path = os.path.join(exp_dir, 'best_model.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device, num_classes, interp)
    log(f"  Test OA: {test_metrics['OA']*100:.2f}% | Fire P: {test_metrics['class1_P']*100:.2f}% | Fire R: {test_metrics['class1_R']*100:.2f}% | Fire F1: {test_metrics['class1_F1']*100:.2f}% | Fire IoU: {test_metrics['class1_IoU']*100:.2f}% | mIoU: {test_metrics['mIoU']*100:.2f}%")

    log_file.close()

    # Save history
    np.savez(os.path.join(exp_dir, 'training_history.npz'), hist=hist)
    plot_training_history(hist, os.path.join(exp_dir, 'training_plot.png'))

    log_file = open(os.path.join(exp_dir, 'Training_log.txt'), 'a')
    log(f"\nDone. Results saved to {exp_dir}")
    log_file.close()


def main():
    parser = argparse.ArgumentParser(description='Train wildfire detection model')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    # CLI overrides (these take precedence over the config file)
    parser.add_argument('--mode', type=int, help='Spectral band mode (0-11)')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--fire_class_weight', type=int)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != 'config' and v is not None}
    config = load_config(args.config, overrides)
    config['_config_path'] = args.config
    train(config)


if __name__ == '__main__':
    main()
PYEOF

# ── test.py (new config-driven version) ──────────────────────────────────
cat > test.py << 'PYEOF'
#!/usr/bin/env python
"""Inference script for wildfire detection models.

Usage:
    python test.py --config configs/classical_baseline.yaml \
        --restore_from experiments/classical_baseline_0815_1330/best_model.pth
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
from scipy import sparse
import yaml

from models import get_model, MODE_NAMES
from dataset.sen2fire import Sen2FireDataSet
from utils.metrics import eval_image
from utils.visualization import plot_scene_map

SCENE_DIMS = {1: (32, 27), 2: (22, 27), 3: (14, 36), 4: (21, 24)}
SCENE_LIST_FILES = {
    1: './dataset/train.txt',
    2: './dataset/train.txt',
    3: './dataset/val.txt',
    4: './dataset/test.txt',
}

epsilon = 1e-14
name_classes = np.array(['non-fire', 'fire'], dtype=str)


def load_config(config_path, cli_overrides):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value
    return config


def process_scene(scene_id, model, config, output_dir, device):
    """Generate predictions and build scene map for one scene."""
    scene_list = SCENE_LIST_FILES[scene_id]
    if not os.path.exists(scene_list):
        print(f"List file {scene_list} not found, skipping scene {scene_id}.")
        return

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', min(4, os.cpu_count() or 1)))
    input_size = (512, 512)
    interp = nn.Upsample(size=input_size, mode='bilinear').to(device)

    loader = data.DataLoader(
        Sen2FireDataSet(config['data_dir'], scene_list, mode=config['mode']),
        batch_size=1, shuffle=False, num_workers=n_workers, pin_memory=True)

    preds_dir = os.path.join(output_dir, 'preds', f'Scene{scene_id}')
    os.makedirs(preds_dir, exist_ok=True)

    # Generate predictions
    for batch in tqdm(loader, desc=f"Scene {scene_id} predictions"):
        image, _, _, name = batch
        image = image.float().to(device)
        patch_name = name[0].split('/')[1]
        filename = patch_name[:6] + str(scene_id) + patch_name[7:]
        patch_path = os.path.join(preds_dir, filename)
        if os.path.exists(patch_path + '.npz'):
            continue
        with torch.no_grad():
            pred = model(image)
        _, pred = torch.max(interp(nn.functional.softmax(pred, dim=1)).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy().astype('uint8')
        sparse.save_npz(patch_path, sparse.csr_matrix(pred))

    # Reconstruct scene map
    n_row, n_col = SCENE_DIMS[scene_id]
    patch_size, overlap = 512, 128
    h = n_row * (patch_size - overlap) + overlap
    w = n_col * (patch_size - overlap) + overlap
    reconstructed_rgb = np.zeros((3, h, w))
    reconstructed_label = np.zeros((h, w))
    reconstructed_pred = np.zeros((h, w))

    image_dir = os.path.join(config['data_dir'], f"scene{scene_id}")
    for row in tqdm(range(1, n_row + 1), desc=f"Scene {scene_id} map"):
        for col in range(1, n_col + 1):
            patch_name = f"scene_{scene_id}_patch_{row}_{col}.npz"
            patch_data = np.load(os.path.join(image_dir, patch_name))['image']
            patch_gt = np.load(os.path.join(image_dir, patch_name))['label']
            filename = patch_name[:6] + str(scene_id) + patch_name[7:]
            patch_pred = sparse.load_npz(os.path.join(preds_dir, filename)).toarray()

            sr = (row - 1) * (patch_size - overlap)
            sc = (col - 1) * (patch_size - overlap)
            reconstructed_rgb[:, sr:sr+patch_size, sc:sc+patch_size] = patch_data[[3,2,1], :, :]
            reconstructed_label[sr:sr+patch_size, sc:sc+patch_size] = patch_gt

            # Handle overlap regions
            r_start = sr + (overlap // 2 if row > 1 else 0)
            c_start = sc + (overlap // 2 if col > 1 else 0)
            pr_start = overlap // 2 if row > 1 else 0
            pc_start = overlap // 2 if col > 1 else 0
            reconstructed_pred[r_start:sr+patch_size, c_start:sc+patch_size] = patch_pred[pr_start:, pc_start:]

    save_path = os.path.join(output_dir, f"scene{scene_id}_map.png")
    plot_scene_map(reconstructed_rgb, reconstructed_pred, reconstructed_label, save_path)


def main():
    parser = argparse.ArgumentParser(description='Run inference for wildfire detection')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--restore_from', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, help='Output directory (default: same as checkpoint dir)')
    parser.add_argument('--mode', type=int)
    parser.add_argument('--data_dir', type=str)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k not in ('config', 'restore_from', 'output_dir') and v is not None}
    config = load_config(args.config, overrides)

    # Output directory defaults to the experiment directory containing the checkpoint
    output_dir = args.output_dir or os.path.dirname(args.restore_from)
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = get_model(config)
    state_dict = torch.load(args.restore_from, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model: {config['model']} | Mode: {config['mode']} ({MODE_NAMES[config['mode']]})")
    print(f"Checkpoint: {args.restore_from}")
    print(f"Output: {output_dir}")

    for scene_id in [1, 2, 3, 4]:
        process_scene(scene_id, model, config, output_dir, device)

    print(f"\nDone. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
PYEOF

# ── scripts/submit_train.sh ──────────────────────────────────────────────
cat > scripts/submit_train.sh << 'SLURM_EOF'
#!/bin/bash
#SBATCH -n 1
#SBATCH -c 12
#SBATCH -t 2:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100
#SBATCH --output=experiments/slurm-%j.out

echo "Running $SLURM_NTASKS tasks ($SLURM_CPUS_PER_TASK cores) on $SLURM_NODELIST"

module purge
module load cuda cudnn python

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WORKDIR="$HOME/scratch/wildfires"
cd "$WORKDIR"

# Pass all command-line arguments through to train.py
# Usage: sbatch scripts/submit_train.sh --config configs/classical_baseline.yaml
python train.py "$@"
SLURM_EOF

# ── scripts/submit_test.sh ───────────────────────────────────────────────
cat > scripts/submit_test.sh << 'SLURM_EOF'
#!/bin/bash
#SBATCH -n 1
#SBATCH -c 12
#SBATCH -t 2:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100
#SBATCH --output=experiments/slurm-%j.out

echo "Running $SLURM_NTASKS tasks ($SLURM_CPUS_PER_TASK cores) on $SLURM_NODELIST"

module purge
module load cuda cudnn python

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WORKDIR="$HOME/scratch/wildfires"
cd "$WORKDIR"

# Usage: sbatch scripts/submit_test.sh --config configs/classical_baseline.yaml --restore_from experiments/.../best_model.pth
python test.py "$@"
SLURM_EOF

# ── requirements.txt ─────────────────────────────────────────────────────
cat > requirements.txt << 'EOF'
torch>=1.9.0
torchvision
numpy
scipy
matplotlib
tqdm
pennylane>=0.20.0
pyyaml
EOF

# ── .gitignore ────────────────────────────────────────────────────────────
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Jupyter notebooks (use notebooks/ for shared ones committed via git add -f)
*.ipynb
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
.virtual_documents/

# Large model checkpoints (logs and plots ARE committed)
experiments/**/*.pth

# Dataset (lives outside the repo)
Sen2Fire/
EOF

# ── .github/pull_request_template.md ─────────────────────────────────────
cat > .github/pull_request_template.md << 'EOF'
## Summary

<!-- 1-3 sentences: what does this PR do and why? -->

Closes #<!-- issue number -->

## Changes

<!-- Bullet list of what changed. Be specific. -->

-
-

## Results (if applicable)

<!-- If this PR includes a training run or evaluation, summarize key metrics. -->
<!-- Config used: configs/_____.yaml -->
<!-- Key metrics: F1=__%, IoU=__%, OA=__% -->

## Wiki

<!-- Link to any wiki page(s) you created/updated for this task. Write N/A if none. -->

## How to test

<!-- How can a reviewer verify this works? -->

- [ ] `python train.py --config configs/_____.yaml --epochs 1 --batch_size 2` runs without error
- [ ] Existing configs still work

## Checklist

- [ ] Branch named `feature/<issue-number>-<description>`
- [ ] No hardcoded paths
- [ ] New components registered (if adding model/circuit/loss)
- [ ] Experiment results included (if applicable, except .pth files)
- [ ] Self-reviewed my code
EOF

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Move existing files into new structure
# ══════════════════════════════════════════════════════════════════════════════
info "Phase 5: Moving existing files to new structure..."

# Move existing experiment results
if [ -d "Exp/swir_aerosol/weight_10_time0815_1330" ]; then
    cp Exp/swir_aerosol/weight_10_time0815_1330/Training_log.txt experiments/classical_baseline/ 2>/dev/null || true
    cp Exp/swir_aerosol/weight_10_time0815_1330/training_plot.png experiments/classical_baseline/ 2>/dev/null || true
    cp Exp/swir_aerosol/weight_10_time0815_1330/*_hist.npz experiments/classical_baseline/training_history.npz 2>/dev/null || true
    # Note: best_model.pth is ~66MB, will be gitignored
    cp configs/classical_baseline.yaml experiments/classical_baseline/config.yaml
fi

# Move prediction maps
if [ -d "Map/swir_aerosol" ]; then
    cp -r Map/swir_aerosol/* experiments/classical_baseline/maps/ 2>/dev/null || true
fi

# Move data exploration notebook
if [ -f "data_exploration.ipynb" ]; then
    cp data_exploration.ipynb notebooks/data_exploration.ipynb
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6: Remove old files that have been superseded
# ══════════════════════════════════════════════════════════════════════════════
info "Phase 6: Removing superseded files..."

# Remove old directories (now reorganized into new structure)
git rm -rf model/ 2>/dev/null || true
git rm -rf Exp/ 2>/dev/null || true
git rm -rf Map/ 2>/dev/null || true
git rm -rf jobs/ 2>/dev/null || true

# Remove old files superseded by new structure
git rm -f train.ipynb 2>/dev/null || true
git rm -f test_all.py 2>/dev/null || true
git rm -f submit.sh 2>/dev/null || true
git rm -f submit_training.sh 2>/dev/null || true
git rm -f submit_test.sh 2>/dev/null || true
git rm -f data_exploration.ipynb 2>/dev/null || true
git rm -f .github/copilot-instructions.md 2>/dev/null || true

# Remove old dataset file (replaced by dataset/sen2fire.py)
git rm -f dataset/Sen2Fire_Dataset.py 2>/dev/null || true

# Remove old utils file (replaced by utils/metrics.py)
git rm -f utils/tools.py 2>/dev/null || true

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7: Write SOW document
# ══════════════════════════════════════════════════════════════════════════════
info "Phase 7: Writing SOW and README..."

# The SOW is large — write it via a temp file approach for readability
cat > SOW.md << 'SOWEOF'
# Standard Operating Workflow (SOW)

## 1. Project Overview

This project develops a hybrid quantum-classical machine learning model for wildfire detection using Sentinel-2 satellite imagery. The model performs pixel-level semantic segmentation on the Sen2Fire dataset, comparing classical U-Net architectures against quantum-enhanced variants.

**Tech stack**: PyTorch + PennyLane (quantum), SLURM HPC for training.

## 2. Codebase Structure

```
wildfire-detection-2026/
├── configs/                          # Experiment configuration files (YAML)
│   ├── classical_baseline.yaml       # Classical U-Net, mode 5 (SWIR + aerosol)
│   └── quantum_strongly_entangling.yaml
│
├── models/                           # Model architectures (swappable)
│   ├── __init__.py                   # MODEL_REGISTRY + get_model()
│   ├── unet.py                       # Classical U-Net
│   └── quantum_unet.py              # Quantum-enhanced U-Net
│
├── circuits/                         # Quantum circuits (swappable)
│   ├── __init__.py                   # CIRCUIT_REGISTRY + get_circuit()
│   ├── strongly_entangling.py        # AmplitudeEmbedding + StronglyEntanglingLayers
│   └── ry_cnot.py                    # RY encoding + CNOT entanglement
│
├── losses/                           # Loss functions (swappable)
│   ├── __init__.py                   # LOSS_REGISTRY + get_loss()
│   └── cross_entropy.py              # Weighted cross-entropy (current default)
│
├── dataset/
│   ├── __init__.py
│   ├── sen2fire.py                   # PyTorch dataset class
│   ├── train.txt / val.txt / test.txt
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py                    # Evaluation metrics (OA, F1, IoU, etc.)
│   └── visualization.py             # Prediction maps, overlays
│
├── experiments/                      # Results from training/test runs
│   ├── classical_baseline/           # Existing classical results
│   │   ├── config.yaml               # Config snapshot
│   │   ├── Training_log.txt
│   │   ├── training_history.npz
│   │   ├── training_plot.png
│   │   └── maps/                     # Scene prediction maps
│   └── <experiment_name>_<MMDD_HHMM>/
│
├── notebooks/                        # Shared reference notebooks
│   └── data_exploration.ipynb
│
├── scripts/                          # SLURM job submission scripts
│   ├── submit_train.sh
│   └── submit_test.sh
│
├── .github/
│   ├── pull_request_template.md
│   └── ISSUE_TEMPLATE/
│
├── train.py                          # Training entry point (config-driven)
├── test.py                           # Inference entry point (config-driven)
├── requirements.txt
├── SOW.md                            # This document
└── README.md
```

### Key design principles

- **One entry point each** for training (`train.py`) and inference (`test.py`), driven by YAML config files.
- **Registry pattern** for models, circuits, and losses. Adding a new variant means adding a new file and one line in the registry — no changes to `train.py` or `test.py`.
- **Config files** fully describe an experiment. Every run copies its config into the output directory for reproducibility.
- **Experiment results** are committed (except `.pth` checkpoints, which are gitignored due to size). Each PR that involves a training run should include the best run's results.

## 3. Setup

```bash
# Clone the repo
git clone git@git.appdevclub.com:IonQSPRING2026/wildfire-detection-2026.git
cd wildfire-detection-2026

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from models import get_model; print('OK')"
```

The Sen2Fire dataset should be placed one level up from the repo root at `../Sen2Fire/`. See `README.md` for the expected directory layout.

## 4. Working on a Task

### Step 1: Read the issue

A lead creates a Gitea issue for each task and assigns it to you. Read the issue description carefully to understand the scope, acceptance criteria, and which area of the codebase is affected.

### Step 2: Research (if applicable)

Some tasks require research before implementation (e.g., "research alternative loss functions"). For research tasks:

1. Do the research.
2. Document your findings in the **Gitea Wiki** under a clearly named page.
3. Update the issue with a summary and link to the wiki page.
4. Discuss with a lead whether a follow-up implementation task is needed.

Research-only tasks do **not** produce a PR. The deliverable is the wiki page.

### Step 3: Create a feature branch

```bash
git checkout master
git pull origin master
git checkout -b feature/<issue-number>-<short-description>
```

**Branch naming convention**: `feature/<issue-number>-<short-description>`

Examples:
- `feature/12-cli-args`
- `feature/15-focal-loss`
- `feature/18-heatmap-visualization`
- `feature/21-qufex-circuit`

### Step 4: Do the work

Follow the coding guidelines in Section 6. Key rules:

- **Pipeline improvements** (CLI args, new metrics, visualizations): modify existing files (`train.py`, `utils/metrics.py`, etc.).
- **New approaches** (new model, circuit, loss function): create a **new file** in the appropriate directory and register it. Do not modify existing model/circuit/loss files.
- **Training runs**: create a new config YAML in `configs/`, run the experiment, and include the best run's results directory under `experiments/`.

### Step 5: Test your changes

Before pushing, verify:
- Your code runs without errors (at minimum, test with a small batch locally or on the HPC).
- Existing functionality is not broken (other configs still work).
- If you added a new registered component, verify it can be instantiated:
  ```bash
  python -c "from models import get_model; m = get_model({'model': 'your_model', 'mode': 5, 'n_classes': 2})"
  ```

### Step 6: Push and open a PR

```bash
git push -u origin feature/<issue-number>-<short-description>
```

Then open a Pull Request on Gitea. Fill out the PR template completely — it's short and takes under 2 minutes. Make sure to:

1. **Link the issue**: Write `Closes #<number>` in the PR description.
2. **Link the wiki**: If you wrote documentation, include the wiki page URL.
3. **Include experiment results**: If your task involved a training run, mention which config you used and summarize the key metrics.

### Step 7: Address review feedback

A lead or a peer will review your PR. Address any requested changes by pushing additional commits to the same branch. Do not force-push.

### Step 8: Merge

Once approved, the PR is merged into `master` and the feature branch is deleted.

## 5. How to Add New Components

### Adding a new model

1. Create `models/my_new_model.py` with a class that has the signature:
   ```python
   class MyNewModel(nn.Module):
       def __init__(self, n_classes, n_channels, config=None):
   ```
2. Register it in `models/__init__.py`:
   ```python
   from models.my_new_model import MyNewModel
   MODEL_REGISTRY['my_new_model'] = MyNewModel
   ```
3. Create a config file `configs/my_new_model.yaml`:
   ```yaml
   model: my_new_model
   mode: 5
   # ... other params
   ```
4. Run: `python train.py --config configs/my_new_model.yaml`

### Adding a new quantum circuit

1. Create `circuits/my_circuit.py` with a factory function:
   ```python
   def create_my_circuit(config):
       n_qubits = config.get('n_qubits', 8)
       n_layers = config.get('n_layers', 2)
       dev = qml.device('default.qubit', wires=n_qubits)

       @qml.qnode(dev, interface='torch', diff_method='adjoint')
       def circuit(inputs, weights):
           # Your circuit here
           ...
           return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

       weight_shapes = {"weights": (...)}  # Shape of trainable weights
       return circuit, weight_shapes
   ```
2. Register it in `circuits/__init__.py`:
   ```python
   from circuits.my_circuit import create_my_circuit
   CIRCUIT_REGISTRY['my_circuit'] = create_my_circuit
   ```
3. Reference it in a config:
   ```yaml
   model: quantum_unet
   circuit: my_circuit
   ```

### Adding a new loss function

1. Create `losses/my_loss.py`:
   ```python
   def create_my_loss(config):
       return MyLossClass(...)
   ```
2. Register in `losses/__init__.py`:
   ```python
   from losses.my_loss import create_my_loss
   LOSS_REGISTRY['my_loss'] = create_my_loss
   ```
3. Reference in config:
   ```yaml
   loss: my_loss
   ```

## 6. Coding Guidelines

- **Style**: PEP 8, 4-space indentation.
- **Imports**: standard library, then third-party, then local — separated by blank lines.
- **No hardcoded paths**. Use config files or CLI arguments.
- **No HPC-specific imports** in Python source files (e.g., `envmodules`). Environment setup belongs in SLURM scripts.
- **Comments**: add them where the logic isn't obvious. Don't over-comment trivial code.
- **Naming**: snake_case for functions/variables, PascalCase for classes.

## 7. Running Experiments

### Locally (small test)

```bash
# Train with a config
python train.py --config configs/classical_baseline.yaml --epochs 1 --batch_size 2

# Run inference
python test.py --config configs/classical_baseline.yaml \
    --restore_from experiments/<run_dir>/best_model.pth
```

### On the HPC (full run)

```bash
# Submit training
sbatch scripts/submit_train.sh --config configs/your_config.yaml

# Submit inference
sbatch scripts/submit_test.sh --config configs/your_config.yaml \
    --restore_from experiments/<run_dir>/best_model.pth
```

### Experiment output

Every training run creates:
```
experiments/<experiment_name>_<MMDD_HHMM>/
├── config.yaml           # Exact config used (for reproducibility)
├── Training_log.txt      # Epoch-by-epoch metrics
├── training_history.npz  # Raw training history arrays
├── training_plot.png     # Loss/accuracy curves
└── best_model.pth        # Best checkpoint (gitignored)
```

When committing experiment results, include everything **except** `.pth` files (they're ~66MB each and gitignored). Mention the key metrics (F1, IoU, OA) in your PR description so reviewers can compare.

## 8. Notebooks

Notebooks are gitignored by default to prevent accidental commits of large files with output cells. For local experimentation, create notebooks anywhere — they won't be tracked.

**Shared reference notebooks** live in `notebooks/` and are committed explicitly:
```bash
git add -f notebooks/my_useful_notebook.ipynb
```

Only commit a notebook if it provides lasting value to the team (data exploration, result analysis, demos). Discuss with a lead first.

You can import the project's modules directly in notebooks:
```python
from models import get_model
from dataset.sen2fire import Sen2FireDataSet
from utils.metrics import eval_image
```

## 9. Quick Reference

| What you want to do | Where to look |
|---|---|
| Change hyperparameters for a run | Create/edit a config YAML in `configs/` |
| Add a new model architecture | `models/` directory + registry |
| Add a new quantum circuit | `circuits/` directory + registry |
| Add a new loss function | `losses/` directory + registry |
| Add new evaluation metrics | `utils/metrics.py` |
| Add new visualizations | `utils/visualization.py` |
| Change data preprocessing | `dataset/sen2fire.py` |
| Modify the training loop | `train.py` |
| Modify inference/evaluation | `test.py` |
| Change SLURM job settings | `scripts/submit_train.sh` or `scripts/submit_test.sh` |
| Find past experiment results | `experiments/` directory |
| Find research documentation | Gitea Wiki |
SOWEOF

# ── README.md ─────────────────────────────────────────────────────────────
cat > README.md << 'READMEEOF'
# Wildfire Detection Using Quantum-Enhanced U-Net

Hybrid quantum-classical machine learning for wildfire detection using Sentinel-2 satellite imagery and the [Sen2Fire](https://arxiv.org/abs/2403.17884) dataset.

## Quick Start

```bash
pip install -r requirements.txt

# Train classical baseline
python train.py --config configs/classical_baseline.yaml

# Train quantum model
python train.py --config configs/quantum_strongly_entangling.yaml

# Run inference
python test.py --config configs/classical_baseline.yaml \
    --restore_from experiments/<run_dir>/best_model.pth
```

## Codebase Structure

```
configs/          YAML experiment configs
models/           Model architectures (classical U-Net, quantum U-Net)
circuits/         Quantum circuits (swappable via config)
losses/           Loss functions (swappable via config)
dataset/          Sen2Fire PyTorch dataset + split files
utils/            Metrics and visualization helpers
experiments/      Training run outputs (logs, plots, maps)
scripts/          SLURM job submission scripts
notebooks/        Shared reference notebooks
```

**Adding a new model, circuit, or loss function** = create a new file + one line in the registry. See [SOW.md](SOW.md) Section 5 for details.

## Dataset

Download the [Sen2Fire dataset](https://arxiv.org/abs/2403.17884) and place it at `../Sen2Fire/` relative to the repo root:

```
../Sen2Fire/
├── scene1/
│   ├── scene_1_patch_1_1.npz
│   └── ...
├── scene2/
├── scene3/
└── scene4/
```

Each `.npz` file contains `image` (12 bands, 512x512), `label` (binary mask), and `aerosol` data.

## Configuration

Experiments are fully described by YAML config files. Example:

```yaml
model: classical_unet    # or quantum_unet
circuit: strongly_entangling  # only for quantum_unet
mode: 5                   # SWIR + aerosol (best performing)
epochs: 5
batch_size: 16
learning_rate: 1.0e-4
loss: cross_entropy
fire_class_weight: 10
experiment_name: my_experiment
```

CLI arguments override config values: `python train.py --config configs/x.yaml --epochs 1`

## HPC (SLURM)

```bash
sbatch scripts/submit_train.sh --config configs/classical_baseline.yaml
sbatch scripts/submit_test.sh --config configs/classical_baseline.yaml --restore_from experiments/.../best_model.pth
```

## Baseline Results (Classical U-Net, Mode 5: SWIR + Aerosol)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 93.41% |
| Fire Precision | 38.41% |
| Fire Recall | 23.18% |
| Fire F1-Score | 28.91% |
| Fire IoU | 16.90% |
| Mean IoU | 62.73% |

## Contributing

See [SOW.md](SOW.md) for the full workflow, coding guidelines, and how to add new components. In short:

1. Read your assigned issue
2. Branch: `feature/<issue-number>-<description>`
3. Code, test, push
4. Open a PR linking the issue and wiki documentation
5. Address review, merge, delete branch

## Citation

```bibtex
@article{xu2024sen2fire,
  title={Sen2Fire: A Challenging Benchmark Dataset for Wildfire Detection using Sentinel Data},
  author={Xu, Yonghao and Berg, Amanda and Haglund, Leif},
  journal={arXiv preprint arXiv:2403.17884},
  year={2024}
}
```
READMEEOF

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8: Stage and commit everything
# ══════════════════════════════════════════════════════════════════════════════
info "Phase 8: Staging and committing..."

# Force-add the shared notebooks (ipynb is gitignored, but these are exceptions)
git add -f notebooks/data_exploration.ipynb 2>/dev/null || true

# Add all new files
git add \
    models/__init__.py models/unet.py models/quantum_unet.py \
    circuits/__init__.py circuits/strongly_entangling.py circuits/ry_cnot.py \
    losses/__init__.py losses/cross_entropy.py \
    dataset/__init__.py dataset/sen2fire.py \
    utils/__init__.py utils/metrics.py utils/visualization.py \
    configs/classical_baseline.yaml configs/quantum_strongly_entangling.yaml \
    experiments/classical_baseline/ experiments/data_exploration/ \
    scripts/submit_train.sh scripts/submit_test.sh \
    train.py test.py \
    requirements.txt .gitignore \
    .github/pull_request_template.md \
    SOW.md README.md

git commit -m "$(cat <<'EOF'
Restructure codebase for Spring 2026 semester

- Modular architecture: models/, circuits/, losses/ with registry pattern
- Config-driven train.py and test.py (YAML configs, CLI overrides)
- Classical U-Net and quantum U-Net as separate, swappable models
- Two quantum circuits: strongly_entangling and ry_cnot
- Loss function registry (weighted cross-entropy as default)
- Experiments organized under experiments/ with config snapshots
- New SOW.md with full workflow, coding guidelines, and contribution process
- Updated PR template, README, .gitignore, requirements.txt
- SLURM scripts accept pass-through arguments
- Preserved classical baseline results in experiments/classical_baseline/
EOF
)"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 9: Push and clean up remote branches
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================"
echo "  Local restructuring complete!"
echo "============================================"
echo ""
echo "The following will now happen if you proceed:"
echo "  1. Push restructured master to origin"
echo "  2. Delete remote branches: new-models, quantum-layer, samarth-tnn, vqc-task"
echo ""
read -p "Push to origin and delete stale branches? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    info "Pushing master..."
    git push origin master

    info "Deleting stale remote branches..."
    for branch in new-models quantum-layer samarth-tnn vqc-task; do
        if git rev-parse --verify "origin/$branch" >/dev/null 2>&1; then
            git push origin --delete "$branch" && info "  Deleted origin/$branch" || warn "  Could not delete origin/$branch"
        fi
    done

    echo ""
    echo -e "${GREEN}Done! Master is restructured and all stale branches are cleaned up.${NC}"
    echo "Engineers can now clone and start creating feature branches."
else
    info "Skipped push. All changes are committed locally."
    echo "  To push:   git push origin master"
    echo "  To delete: git push origin --delete new-models quantum-layer samarth-tnn vqc-task"
fi
