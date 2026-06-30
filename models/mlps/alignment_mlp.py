"""Pluggable alignment MLPs for the joint SALAD + depth branch.

All MLP types share the same interface: they receive student patch tokens
[B, N, input_dim] and return projected tokens [B, N, output_dim].

Normalization is controlled per-MLP via the `normalization` parameter:
    "none"   : no L2 normalisation; raw magnitudes flow to the loss
    "before" : F.normalize applied to input patches before the projection
    "after"  : F.normalize applied to projected output before the loss

Add new variants by subclassing BaseAlignmentMLP and registering them in
get_alignment_mlp().
"""
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

_VALID_NORMS = {"none", "before", "after"}


class BaseAlignmentMLP(nn.Module):
    """Abstract base for alignment MLPs.

    Args:
        input_dim: Feature dimension of incoming patch tokens.
        hidden_dim: Internal projection dimension (ignored by some subclasses).
        output_dim: Feature dimension of outgoing patch tokens.
        normalization: When and how to apply L2 norm. One of "none", "before",
            "after".
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        normalization: str = "none",
    ) -> None:
        super().__init__()
        if normalization not in _VALID_NORMS:
            raise ValueError(
                f"normalization must be one of {_VALID_NORMS}, got '{normalization}'."
            )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.normalization = normalization

    def forward(self, x):
        """Args:
            x: [B, N, input_dim] patch tokens.
        Returns:
            [B, N, output_dim] projected tokens.
        """
        raise NotImplementedError


class TokenByTokenMLP(BaseAlignmentMLP):
    """Shared two-layer projection applied independently to every patch token.

    Equivalent to a 1x1 convolution across the patch sequence: weights are
    shared across the N dimension, so the model has no dependence on sequence
    length or patch ordering.

    Architecture: Linear -> ReLU -> Linear
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        normalization: str = "none",
    ) -> None:
        super().__init__(input_dim, hidden_dim, output_dim, normalization)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """Args:
            x: [B, N, input_dim]
        Returns:
            [B, N, output_dim]
        """
        if self.normalization == "before":
            x = F.normalize(x, p=2, dim=-1)
        out = self.net(x)
        if self.normalization == "after":
            out = F.normalize(out, p=2, dim=-1)
        return out


class LinearProjection(BaseAlignmentMLP):
    """Single linear projection applied independently to every patch token.

    Simpler alternative to TokenByTokenMLP: no hidden layer, no activation.
    The hidden_dim argument is accepted for API consistency but is not used.

    Architecture: Linear (input_dim -> output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        normalization: str = "none",
    ) -> None:
        super().__init__(input_dim, hidden_dim, output_dim, normalization)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """Args:
            x: [B, N, input_dim]
        Returns:
            [B, N, output_dim]
        """
        if self.normalization == "before":
            x = F.normalize(x, p=2, dim=-1)
        out = self.fc(x)
        if self.normalization == "after":
            out = F.normalize(out, p=2, dim=-1)
        return out


# Registry: add new MLP type strings and their classes here.
_MLP_REGISTRY = {
    "token_by_token": TokenByTokenMLP,
    "linear": LinearProjection,
}


def get_alignment_mlp(mlp_cfg: DictConfig) -> BaseAlignmentMLP:
    """Factory that builds an alignment MLP from a config node.

    Args:
        mlp_cfg: OmegaConf node with fields: type, input_dim, hidden_dim,
            output_dim, and optionally normalization (default "none").

    Returns:
        Instantiated BaseAlignmentMLP subclass.

    Raises:
        NotImplementedError: If mlp_cfg.type is not in the registry.
    """
    mlp_type = mlp_cfg.type
    if mlp_type not in _MLP_REGISTRY:
        raise NotImplementedError(
            f"MLP type '{mlp_type}' is not registered. "
            f"Available types: {list(_MLP_REGISTRY.keys())}"
        )
    normalization = mlp_cfg.get("normalization", "none")
    return _MLP_REGISTRY[mlp_type](
        input_dim=mlp_cfg.input_dim,
        hidden_dim=mlp_cfg.hidden_dim,
        output_dim=mlp_cfg.output_dim,
        normalization=normalization,
    )
