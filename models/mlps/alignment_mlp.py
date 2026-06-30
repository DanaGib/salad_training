"""Pluggable alignment MLPs for the joint SALAD + depth branch.

All MLP types share the same interface: they receive student patch tokens
[B, N, input_dim] and return projected tokens [B, N, output_dim].
Add new variants by subclassing BaseAlignmentMLP and registering them in
get_alignment_mlp().
"""
import torch.nn as nn
from omegaconf import DictConfig


class BaseAlignmentMLP(nn.Module):
    """Abstract base for alignment MLPs.

    Args:
        input_dim: Feature dimension of incoming patch tokens.
        hidden_dim: Internal projection dimension.
        output_dim: Feature dimension of outgoing patch tokens.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x):
        """Args:
            x: [B, N, input_dim] patch tokens.
        Returns:
            [B, N, output_dim] projected tokens.
        """
        raise NotImplementedError


class TokenByTokenMLP(BaseAlignmentMLP):
    """Shared linear projection applied independently to every patch token.

    Equivalent to a 1x1 convolution across the patch sequence: weights are
    shared across the N dimension, so the model has no dependence on sequence
    length or patch ordering.

    Architecture: Linear -> ReLU -> Linear
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__(input_dim, hidden_dim, output_dim)
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
        return self.net(x)


# Registry: add new MLP type strings and their classes here.
_MLP_REGISTRY = {
    "token_by_token": TokenByTokenMLP,
}


def get_alignment_mlp(mlp_cfg: DictConfig) -> BaseAlignmentMLP:
    """Factory that builds an alignment MLP from a config node.

    Args:
        mlp_cfg: OmegaConf node with fields: type, input_dim, hidden_dim, output_dim.

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
    return _MLP_REGISTRY[mlp_type](
        input_dim=mlp_cfg.input_dim,
        hidden_dim=mlp_cfg.hidden_dim,
        output_dim=mlp_cfg.output_dim,
    )
