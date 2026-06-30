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

    def _encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the ViT encoder and return all hidden states (CLS + patches).

        Args:
            pixel_values: Input images [B, 3, H, W] in FP32.

        Returns:
            hidden: [B, num_patches+1, 768] — index 0 is the CLS token.
        """
        with torch.no_grad():
            emb = self.embeddings(pixel_values)
            enc_out = self.vit_encoder(emb)
            return self.layernorm(enc_out.last_hidden_state)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens for local distillation loss.

        Args:
            pixel_values: Input images [B, 3, H, W] in FP32.

        Returns:
            Patch token embeddings [B, num_patches, 768].
        """
        hidden = self._encode(pixel_values)
        return hidden[:, 1:, :]

    def forward_salad_format(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract depth features in the spatial format expected by SALAD.

        Reshapes patch tokens to a 2D feature map and returns the CLS token
        separately, matching the (feature_map, cls_token) tuple produced by
        the DINOv2 backbone wrapper.

        Args:
            pixel_values: Input images [B, 3, H, W] in FP32.

        Returns:
            feature_map: [B, 768, H/14, W/14]  e.g. [B, 768, 16, 16] at 224px.
            cls_token:   [B, 768]
        """
        hidden = self._encode(pixel_values)
        cls_token = hidden[:, 0, :]          # [B, 768]
        patches = hidden[:, 1:, :]           # [B, num_patches, 768]
        B, N, C = patches.shape
        H = W = int(N ** 0.5)
        feature_map = patches.permute(0, 2, 1).reshape(B, C, H, W)
        return feature_map, cls_token
