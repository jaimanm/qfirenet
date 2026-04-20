import torch
import torch.nn as nn
from losses.cross_entropy import create_weighted_cross_entropy
from losses.focal_loss import create_focal_loss
from losses.dice_loss import create_dice_ce_loss

LOSS_REGISTRY = {
    'cross_entropy': create_weighted_cross_entropy,
    'focal_loss': create_focal_loss,
    'dice_ce': create_dice_ce_loss,
}


def get_loss(config):
    """Instantiate a loss function from config."""
    loss_name = config.get('loss', 'cross_entropy')
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{loss_name}'. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[loss_name](config)
