"""Patch-level alignment loss for the joint SALAD + depth branch.

Supports MSE and cosine variants. Normalisation (if any) is handled upstream
inside the alignment MLP, so this class is a pure loss computation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """Computes the patch-level alignment loss between student and teacher tokens.

    Args:
        loss_type: "mse" or "cosine".
    """

    def __init__(self, loss_type: str) -> None:
        super().__init__()
        if loss_type not in ("mse", "cosine"):
            raise ValueError(
                f"Unknown alignment loss type '{loss_type}'. "
                "Choose 'mse' or 'cosine'."
            )
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the alignment loss.

        Args:
            pred:   [B, N, D] student projected patch tokens.
            target: [B, N, D] frozen teacher patch tokens (same D).

        Returns:
            Scalar loss tensor.
        """
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)

        # cosine: 1 - mean cosine similarity over all patches and batch
        return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
