"""Utilities for partial checkpoint loading."""
import torch
import torch.nn as nn
from pathlib import Path


def load_aggregator_weights(aggregator: nn.Module, ckpt_path: str) -> None:
    """Load SALAD aggregator weights from a Lightning checkpoint in-place.

    Extracts all keys prefixed with ``aggregator.`` from the checkpoint state
    dict and loads them into *aggregator* with strict matching.  Use this to
    initialise the SALAD aggregator from a pretrained baseline checkpoint
    instead of random initialisation (Trial 4).

    Args:
        aggregator: The aggregator ``nn.Module`` to initialise.
        ckpt_path: Path to a Lightning ``.ckpt`` file that contains
            ``aggregator.*`` keys under ``state_dict``.

    Raises:
        FileNotFoundError: If *ckpt_path* does not exist on disk.
        RuntimeError: If the checkpoint contains no ``aggregator.*`` keys,
            or if the keys do not match the module's parameter names.
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(str(path), map_location="cpu")
    sd = checkpoint.get("state_dict", checkpoint)

    prefix = "aggregator."
    agg_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    if not agg_sd:
        raise RuntimeError(
            f"No 'aggregator.*' keys found in checkpoint: {path}. "
            "Ensure the checkpoint was saved from a VPRModel that uses a SALAD aggregator."
        )

    missing, unexpected = aggregator.load_state_dict(agg_sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading aggregator: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading aggregator: {unexpected}")

    print(f"Loaded aggregator weights from {path} ({len(agg_sd)} tensors)")
