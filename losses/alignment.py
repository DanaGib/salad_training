"""Configurable alignment loss for the joint SALAD + depth branch.

Supports MSE and cosine variants with three normalisation strategies
controlled by config.model.normalization.stage:

    after_mlp  (default): L2-normalise both student output and teacher patches
                          immediately before computing the loss.
    before_mlp           : Normalisation is applied upstream in training_step
                          before the MLP; this class receives already-projected
                          tensors and does NOT normalise them again.
    none                 : Raw magnitudes are passed straight to the loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """Computes the patch-level alignment loss between student and teacher tokens.

    Args:
        loss_type: "mse" or "cosine".
        norm_stage: "after_mlp", "before_mlp", or "none".
                    Only "after_mlp" triggers normalisation inside this class;
                    the other two stages are handled externally.
    """

    def __init__(self, loss_type: str, norm_stage: str) -> None:
        super().__init__()
        if loss_type not in ("mse", "cosine"):
            raise ValueError(
                f"Unknown alignment loss type '{loss_type}'. "
                "Choose 'mse' or 'cosine'."
            )
        if norm_stage not in ("after_mlp", "before_mlp", "none"):
            raise ValueError(
                f"Unknown norm_stage '{norm_stage}'. "
                "Choose 'after_mlp', 'before_mlp', or 'none'."
            )
        self.loss_type = loss_type
        self.norm_stage = norm_stage

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the alignment loss.

        Args:
            pred:   [B, N, D] student projected patch tokens.
            target: [B, N, D] frozen teacher patch tokens (same D).

        Returns:
            Scalar loss tensor.
        """
        if self.norm_stage == "after_mlp":
            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)

        if self.loss_type == "mse":
            return F.mse_loss(pred, target)

        # cosine: 1 - mean cosine similarity over all patches and batch
        return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
