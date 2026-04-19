import numpy as np
import torch

def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, mixed targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    # Also blend the mask directly
    mixed_y = lam * y + (1 - lam) * y_b
    
    return mixed_x, y_a, y_b, mixed_y, lam

def cutmix_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, mixed targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Generate bounding box
    H = x.size()[2]
    W = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    # Create a mixed mask
    mixed_y = y.clone()
    if len(mixed_y.shape) == 3: # [B, H, W]
        mixed_y[:, bby1:bby2, bbx1:bbx2] = y_b[:, bby1:bby2, bbx1:bbx2]
    elif len(mixed_y.shape) == 4: # [B, C, H, W]
        mixed_y[:, :, bby1:bby2, bbx1:bbx2] = y_b[:, :, bby1:bby2, bbx1:bbx2]

    return mixed_x, y_a, y_b, mixed_y, lam


def aerosol_aug(x, prob=0.5, scale_min=1.0, scale_max=6.0, aerosol_channel=3):
    """Randomly scale the aerosol channel to simulate test-domain smoke intensity.

    The train/test domain shift analysis shows Channel 3 (aerosol) is 5.3x
    higher in the test set (mean 1.42 vs 0.27). This augmentation randomly
    scales that channel at training time to expose the model to the full
    intensity range it will encounter at test time.

    Args:
        x: Input tensor of shape [B, C, H, W].
        prob: Per-sample probability of applying the scaling (default 0.5).
        scale_min: Minimum scale factor (default 1.0, i.e. no reduction).
        scale_max: Maximum scale factor (default 6.0, matching domain ratio).
        aerosol_channel: Channel index of the aerosol band (default 3).

    Returns:
        Augmented tensor (cloned, original is not modified).
    """
    if x.size(1) <= aerosol_channel:
        return x
    x = x.clone()
    apply = torch.rand(x.size(0), device=x.device) < prob
    scales = torch.empty(x.size(0), device=x.device).uniform_(scale_min, scale_max)
    scales = torch.where(apply, scales, torch.ones_like(scales))
    x[:, aerosol_channel] *= scales.view(-1, 1, 1)
    return x
