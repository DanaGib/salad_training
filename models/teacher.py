import torch
import torch.nn as nn
from transformers import AutoModelForDepthEstimation


class DepthTeacher(nn.Module):
    """Frozen Depth Anything V2 teacher providing geometric patch embeddings.

    Args:
        model_name: HuggingFace model identifier for Depth Anything V2.
    """

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Base") -> None:
        super().__init__()
        depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)
        # backbone.model is the raw Dinov2Model; its last_hidden_state is
        # [B, num_patches+1, hidden_dim] — the shape needed for distillation
        self.encoder = depth_model.backbone.model
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode: bool = True) -> "DepthTeacher":
        """Keep teacher permanently in eval mode regardless of Lightning calls."""
        return super().train(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from the ViT encoder.

        Args:
            pixel_values: Input images [B, 3, H, W] in FP32.

        Returns:
            Patch token embeddings [B, num_patches, 768].
        """
        with torch.no_grad():
            outputs = self.encoder(pixel_values, output_hidden_states=True)
        # exclude CLS token: [B, num_patches+1, D] -> [B, num_patches, D]
        return outputs.last_hidden_state[:, 1:, :]
