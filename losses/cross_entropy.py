import torch
import torch.nn as nn


def create_weighted_cross_entropy(config):
    """Weighted cross-entropy loss to address fire/non-fire class imbalance."""
    fire_weight = config.get('fire_class_weight', 10)
    weights = torch.Tensor([1, fire_weight])
    return nn.CrossEntropyLoss(weight=weights)
