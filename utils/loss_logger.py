"""Loss accumulation and interval console logging for VPR training.

Maintains two independent windows:
- Interval window: resets every `log_interval` steps for console prints.
- Epoch window: resets only at epoch boundaries for W&B epoch averages.

W&B logging is handled externally by Lightning's self.log() calls.
"""
from __future__ import annotations
from typing import Dict


class LossAccumulator:
    """Accumulates loss values and hard-pair counts across training steps.

    Args:
        log_interval: Number of steps between console summary prints.
    """

    def __init__(self, log_interval: int) -> None:
        self.log_interval = log_interval
        self._reset_interval()
        self._reset_epoch()

    def update(self, ms: float, align: float, total: float, hard_pairs: int) -> None:
        """Add one step's values to both interval and epoch windows.

        Args:
            ms: MultiSimilarity loss for this step.
            align: Alignment loss for this step (0.0 for baseline model).
            total: Combined total loss for this step.
            hard_pairs: Number of hard pairs found by the miner this step.
        """
        self._i_ms += ms;    self._e_ms += ms
        self._i_align += align;  self._e_align += align
        self._i_total += total;  self._e_total += total
        self._i_pairs += hard_pairs; self._e_pairs += hard_pairs
        self._i_n += 1;      self._e_n += 1

    def maybe_print_interval(self, step: int, epoch: int) -> None:
        """Print a console summary every log_interval steps; resets interval window.

        Does NOT write to W&B — all W&B logging goes through Lightning self.log().

        Args:
            step: Current global training step.
            epoch: Current epoch number.
        """
        if self._i_n < self.log_interval:
            return
        n = self._i_n
        print(
            f"[Epoch {epoch}, Step {step}] "
            f"MS Loss: {self._i_ms / n:.4f} | "
            f"Align Loss: {self._i_align / n:.4f} | "
            f"Total Loss: {self._i_total / n:.4f} | "
            f"Hard Pairs/step: {self._i_pairs / n:.1f}"
        )
        self._reset_interval()

    def epoch_averages(self) -> Dict[str, float]:
        """Return epoch-level averages as a plain dict for W&B logging.

        Returns:
            Dict with keys: ms, align, total, hard_pairs_avg.
        """
        n = max(self._e_n, 1)
        return {
            "ms": self._e_ms / n,
            "align": self._e_align / n,
            "total": self._e_total / n,
            "hard_pairs_avg": self._e_pairs / n,
        }

    def reset_epoch(self) -> None:
        """Clear both windows at the end of an epoch."""
        self._reset_interval()
        self._reset_epoch()

    def _reset_interval(self) -> None:
        self._i_ms = self._i_align = self._i_total = 0.0
        self._i_pairs = self._i_n = 0

    def _reset_epoch(self) -> None:
        self._e_ms = self._e_align = self._e_total = 0.0
        self._e_pairs = self._e_n = 0
