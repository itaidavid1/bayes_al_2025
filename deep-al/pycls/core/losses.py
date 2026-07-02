"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pycls.core.config import cfg

# Supported loss functions
_loss_funs = {
    'cross_entropy': nn.CrossEntropyLoss,
}

def get_loss_fun(reduction="mean"):
    """Retrieves the loss function."""
    assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), \
        'Loss function \'{}\' not supported'.format(cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSS_FUN](reduction=reduction).cuda()


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor


def distillation_loss(preds, soft_targets, mask, temperature=3.0, soft_target_normalization='power'):
    """
    KL divergence loss for distillation from soft targets with temperature scaling.
    
    Temperature scaling (Hinton et al.) softens probability distributions so the model
    can learn from "dark knowledge" — the relative similarities between non-dominant classes.
    Gradients from soft targets scale as 1/T^2, so we multiply by T^2 to keep them
    balanced with the hard-label CE loss.
    
    Args:
        preds: Model predictions (logits), shape (batch_size, num_classes)
        soft_targets: Soft probability targets from C matrix, shape (batch_size, num_classes)
        mask: Boolean mask indicating which samples to use for distillation, shape (batch_size,)
        temperature: Temperature for softening distributions (default: 3.0). Higher = softer.
        soft_target_normalization: Normalization method for soft targets ('power' or 'softmax')
        
    Returns:
        Scalar loss value (mean KL divergence over valid samples, scaled by T^2)
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device)
    
    # Temperature-scaled log-softmax of model predictions
    soft_preds = F.log_softmax(preds / temperature, dim=1)

    # soft_targets_shifted = soft_targets - soft_targets.mean(dim=1, keepdim=True)
    # T_veracity = ((1 -  (soft_targets/ soft_targets.sum(dim=1)[:, None]).max(dim=1)[0])**2)[:, None]
    # scaled_soft_targets = F.softmax(soft_targets_shifted / T_veracity, dim=1)
    #
    # kl = F.kl_div(soft_preds, scaled_soft_targets, reduction='none').sum(dim=1)
    
    # Temperature-scaled soft targets normalization
    if soft_target_normalization == 'softmax':
        # Softmax normalization with temperature
        soft_targets_tempered = F.softmax(soft_targets / 0.1, dim=1)
    else:
        # Power-based temperature scaling (default)
        soft_targets_tempered = torch.pow(soft_targets.clamp(min=1e-8), 1.0 / temperature)
        soft_targets_tempered = soft_targets_tempered / soft_targets_tempered.sum(dim=1, keepdim=True)
    
    kl = F.kl_div(soft_preds, soft_targets_tempered, reduction='none').sum(dim=1)
    
    # Apply mask and compute mean over valid samples
    masked_kl = kl * mask.float()
    loss = masked_kl.sum() / mask.float().sum().clamp(min=1)
    
    # Multiply by T^2 to keep gradients on the same scale as CE loss (Hinton et al.)
    loss = loss * (temperature ** 2)
    
    return loss
