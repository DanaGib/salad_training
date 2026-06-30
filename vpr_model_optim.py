"""Optimiser and scheduler mixin for VPRModel.

Extracted from vpr_model.py to keep that file under 100 lines.
Supports AdamW and SGD with a LinearLR warmup/decay scheduler.
"""
import torch
from torch.optim import lr_scheduler


class OptimiserMixin:
    """Provides configure_optimizers and the custom optimizer_step for VPRModel."""

    def configure_optimizers(self):
        """Build optimiser and LinearLR scheduler from config.

        Returns:
            Tuple of ([optimizer], [scheduler]) for Lightning.

        Raises:
            ValueError: If cfg.training.optimizer is not 'adamw' or 'sgd'.
        """
        cfg = self.cfg.training
        if cfg.optimizer == "adamw":
            opt = torch.optim.AdamW(
                self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "sgd":
            opt = torch.optim.SGD(
                self.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                momentum=cfg.momentum,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

        scheduler = lr_scheduler.LinearLR(
            opt,
            start_factor=cfg.lr_sched_start_factor,
            end_factor=cfg.lr_sched_end_factor,
            total_iters=cfg.lr_sched_total_iters,
        )
        return [opt], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Step optimizer then immediately step the LR scheduler."""
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()
