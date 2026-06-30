"""Training-step mixin for VPRModel.

Contains the metric loss helper, per-model-type loss dispatch,
interval console logging, and the gradient-norm hook.
"""
import torch
from vpr_model_depth_step import (
    compute_joint_depth_losses,
    compute_predictor_global_losses,
    compute_global_depth_losses,
    compute_global_local_depth_losses,
)


class TrainingMixin:
    """Provides training_step, grad-norm hook, and interval logging."""

    def _vpr_loss(self, descriptors, labels):
        """Compute configured metric loss; return (loss, nb_hard_pairs).

        Args:
            descriptors: L2-normalised global descriptors [B, D].
            labels: Place-id integer labels [B].
        """
        if self.miner is not None:
            mined = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, mined)
            nb_hard = len(set(mined[0].detach().cpu().numpy()))
        else:
            loss = self.loss_fn(descriptors, labels)
            nb_hard = 0
            if isinstance(loss, tuple):
                loss, _ = loss
        return loss, nb_hard

    def _dispatch_depth_losses(self, images, feat_map, labels) -> dict:
        """Route to the correct depth-loss helper based on model type."""
        if self.is_joint:
            return compute_joint_depth_losses(self, images, feat_map)
        if self.is_predictor_global:
            return compute_predictor_global_losses(self, images, feat_map, labels)
        if self.is_global_local:
            return compute_global_local_depth_losses(self, images, feat_map, labels)
        if self.is_global_depth:
            return compute_global_depth_losses(self, images, labels)
        return {}

    def training_step(self, batch, batch_idx):
        """Run one training step; log per-step losses and hard-pair count."""
        places, labels = batch
        BS, N, ch, h, w = places.shape
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)

        backbone_out = self.backbone(images)
        descriptors = self.aggregator(backbone_out)
        loss_vpr, nb_hard = self._vpr_loss(descriptors, labels)

        depth_losses = self._dispatch_depth_losses(images, backbone_out[0], labels)
        _z = torch.tensor(0.0, device=loss_vpr.device)
        loss_local = depth_losses.get("local", _z)
        loss_gdepth = depth_losses.get("global_depth", _z)

        a_l = getattr(self.cfg.loss, "alpha_local", 0.0)
        a_g = getattr(self.cfg.loss, "alpha_global", 0.0)
        loss = loss_vpr + a_l * loss_local + a_g * loss_gdepth

        self.log("train/ms_loss", loss_vpr.item(), on_step=True, on_epoch=False,
                 prog_bar=False, logger=True)
        self.log("train/total_loss", loss.item(), on_step=True, on_epoch=False,
                 prog_bar=True, logger=True)
        self.log("train/hard_pairs", float(nb_hard), on_step=True, on_epoch=False,
                 prog_bar=False, logger=True)
        if "local" in depth_losses:
            self.log("train/local_loss", loss_local.item(), on_step=True,
                     on_epoch=False, prog_bar=False, logger=True)
        if "global_depth" in depth_losses:
            self.log("train/global_depth_loss", loss_gdepth.item(), on_step=True,
                     on_epoch=False, prog_bar=False, logger=True)

        self._loss_acc.update(
            loss_vpr.item(), loss_local.item(), loss.item(), nb_hard, loss_gdepth.item()
        )
        self._loss_acc.maybe_print_interval(self.global_step, self.current_epoch)
        return {"loss": loss}

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log total gradient norm before each optimiser step."""
        norms = [p.grad.detach().norm() for p in self.parameters() if p.grad is not None]
        if norms:
            self.log("train/grad_norm", torch.stack(norms).norm().item(),
                     on_step=True, on_epoch=False, prog_bar=False, logger=True)
