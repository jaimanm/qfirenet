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
