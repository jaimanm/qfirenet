import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for multi-class segmentation with severe class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    The modulating factor (1 - p_t)^gamma down-weights easy, well-classified
    pixels so training focuses on hard examples (e.g. rare fire pixels).

    Args:
        gamma: Focusing exponent. 0 reduces to weighted CE; 2 is the original paper default.
        alpha: 1-D tensor of per-class weights, shape [num_classes]. Mirrors the
               `weight` argument of nn.CrossEntropyLoss.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', alpha.float())
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: logits of shape [B, C, H, W]
            targets: class indices of shape [B, H, W], dtype long
        Returns:
            Scalar mean focal loss.
        """
        # log-probabilities and probabilities: [B, C, H, W]
        log_p = F.log_softmax(inputs, dim=1)
        p = torch.exp(log_p)

        # Gather log p and p at the true class: [B, H, W]
        log_pt = log_p.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        pt = p.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        # Focal modulating factor: [B, H, W]
        focal_weight = (1.0 - pt) ** self.gamma

        # Per-pixel loss: [B, H, W]
        loss = -focal_weight * log_pt

        # Apply per-class alpha weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]   # [B, H, W]
            loss = alpha_t * loss

        return loss.mean()


def create_focal_loss(config: dict) -> FocalLoss:
    """Instantiate FocalLoss from a config dict.

    Reads:
        focal_gamma       (float, default 2.0)   – focusing exponent
        fire_class_weight (int,   default 10)     – weight on fire class (class 1);
                                                    background (class 0) always has weight 1
    """
    gamma = config.get('focal_gamma', 2.0)
    fire_weight = config.get('fire_class_weight', 10)
    alpha = torch.tensor([1.0, float(fire_weight)])
    return FocalLoss(gamma=gamma, alpha=alpha)
