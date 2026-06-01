from models.unet import ClassicalUNet
from models.compact_unet import CompactUNet
from models.quantum_unet import QuantumUNet
from models.compact_quantum_unet import CompactQuantumUNet
from models.qufex_model import QuNet
from models.qbnet_model import QBNet
from models.fpn_unet import FPNUNet

MODEL_REGISTRY = {
    'classical_unet': ClassicalUNet,
    'compact_unet': CompactUNet,
    'quantum_unet': QuantumUNet,
    'compact_quantum_unet': CompactQuantumUNet,
    'qufex_unet': QuNet,
    'qbnet_unet': QBNet,
    'fpn_unet': FPNUNet
}

# Spectral band mode -> number of input channels
MODE_CHANNELS = {
    0: 12, 1: 13, 2: 3, 3: 4, 4: 3, 5: 4,
    6: 3, 7: 4, 8: 3, 9: 4, 10: 6, 11: 7,
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
