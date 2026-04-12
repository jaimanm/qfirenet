import torch
import torch.nn as nn
from losses.cross_entropy import create_weighted_cross_entropy
from losses.focal import create_focal_loss

LOSS_REGISTRY = {
    'cross_entropy': create_weighted_cross_entropy,
    'focal': create_focal_loss,
}


def get_loss(config):
    """Instantiate a loss function from config."""
    loss_name = config.get('loss', 'cross_entropy')
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{loss_name}'. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[loss_name](config)
