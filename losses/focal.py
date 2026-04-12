import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for addressing severe class imbalance in fire/non-fire segmentation.

    Down-weights easy (confident correct) predictions so training focuses on
    hard examples like ambiguous fire pixels at smoke boundaries.
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        # inputs: (N, C, H, W), targets: (N, H, W)
        log_p = F.log_softmax(inputs, dim=1)
        # gather log-prob of the correct class for each pixel
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        focal_weight = (1 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        if self.weight is not None:
            # apply per-class weight at each pixel
            class_weight = self.weight.to(inputs.device)
            w = class_weight[targets]
            loss = loss * w

        return loss.mean()


def create_focal_loss(config):
    """Focal loss with optional class weighting for fire/non-fire imbalance."""
    fire_weight = config.get('fire_class_weight', 10)
    gamma = config.get('focal_gamma', 2.0)
    weight = torch.Tensor([1, fire_weight])
    return FocalLoss(gamma=gamma, weight=weight)
