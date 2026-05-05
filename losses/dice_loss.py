import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCrossEntropyLoss(nn.Module):
    """Combined Dice and weighted cross-entropy loss for binary segmentation."""

    def __init__(self, fire_weight=10, dice_weight=1.0, ce_weight=1.0, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, float(fire_weight)])
        )

    def forward(self, logits, targets):
        ce_loss = self.cross_entropy(logits, targets)

        fire_probs = F.softmax(logits, dim=1)[:, 1]
        fire_targets = (targets == 1).float()
        intersection = (fire_probs * fire_targets).sum(dim=(1, 2))
        denominator = fire_probs.sum(dim=(1, 2)) + fire_targets.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def create_dice_ce_loss(config):
    return DiceCrossEntropyLoss(
        fire_weight=config.get("fire_class_weight", 10),
        dice_weight=config.get("dice_weight", 1.0),
        ce_weight=config.get("ce_weight", 1.0),
    )
