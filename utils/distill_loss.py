import torch.nn.functional as F

def local_distill_loss(pred, target):
    """Normalized patch-level MSE (cosine-direction alignment).
    Args:
        pred:   [B, num_patches, D] — student projected patches
        target: [B, num_patches, D] — frozen teacher patches
    Returns:
        Scalar loss.
    """
    pred_norm   = F.normalize(pred,   p=2, dim=-1)
    target_norm = F.normalize(target, p=2, dim=-1)
    return F.mse_loss(pred_norm, target_norm)