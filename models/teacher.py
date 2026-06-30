import torch
import torch.nn as nn
from transformers import AutoModelForDepthEstimation

# The -hf suffix selects the HuggingFace-formatted repo (has model_type in config.json).
# The plain "Depth-Anything-V2-Base" repo is a raw PyTorch checkpoint without model_type
# and will raise a ValueError when loaded with AutoModelForDepthEstimation.
_DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Base-hf"


class DepthTeacher(nn.Module):
    """Frozen Depth Anything V2 teacher providing geometric patch embeddings.

    Loads the ViT backbone from a Depth Anything V2 HuggingFace model and
    exposes its patch-level hidden states for JEPA-style distillation.

    Args:
        model_name: HuggingFace model ID; must end in -hf (transformers format).
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        super().__init__()
        depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)
        # depth_model.backbone is Dinov2Backbone (embeddings + encoder + layernorm).
        # We store sub-modules individually so forward() is explicit about the data path.
        bb = depth_model.backbone
        self.embeddings = bb.embeddings
        self.vit_encoder = bb.encoder
        self.layernorm = bb.layernorm
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
            emb = self.embeddings(pixel_values)
            enc_out = self.vit_encoder(emb)
            hidden = self.layernorm(enc_out.last_hidden_state)
        # Exclude CLS token: [B, num_patches+1, D] -> [B, num_patches, D]
        return hidden[:, 1:, :]
