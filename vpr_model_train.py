"""Training-step mixin for VPRModel.

Extracted from vpr_model.py to keep that file under 100 lines.
Contains the metric loss helper, the joint-depth forward pass,
interval console logging, and the gradient-norm hook.
W&B logging is via Lightning self.log() throughout.
"""
import torch


class TrainingMixin:
    """Provides training_step, grad-norm hook, and interval logging for VPRModel."""

    def _vpr_loss(self, descriptors, labels):
        """Compute configured metric loss and return loss + hard-pair count.

        Args:
            descriptors: L2-normalised global descriptors [B, D].
            labels: place-id integer labels [B].

        Returns:
            Tuple of (loss tensor, nb_hard_pairs int).
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

    def training_step(self, batch, batch_idx):
        """Run one training step, log per-step losses and hard-pair count.

        Logs to W&B via Lightning self.log():
            train/ms_loss, train/alignment_loss, train/total_loss,
            train/hard_pairs

        Also accumulates values for interval console prints and epoch
        summaries (consumed in on_train_epoch_end via self._loss_acc).
        """
        places, labels = batch
        BS, N, ch, h, w = places.shape
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)

        backbone_out = self.backbone(images)
        descriptors = self.aggregator(backbone_out)

        loss_vpr, nb_hard = self._vpr_loss(descriptors, labels)
        loss_align = torch.tensor(0.0, device=loss_vpr.device)

        if self.is_joint:
            feat_map = backbone_out[0]
            student = feat_map.flatten(2).permute(0, 2, 1)
            student = self.alignment_mlp(student)
            with torch.cuda.amp.autocast(enabled=False):
                teacher = self.depth_teacher(images.float())
            loss_align = self.alignment_loss(student, teacher.to(student.dtype))

        alpha = self.cfg.loss.alpha if self.is_joint else 0.0
        loss = loss_vpr + alpha * loss_align

        # Per-step W&B metrics via Lightning (axis = global_step).
        self.log("train/ms_loss", loss_vpr.item(), on_step=True, on_epoch=False,
                 prog_bar=False, logger=True)
        self.log("train/total_loss", loss.item(), on_step=True, on_epoch=False,
                 prog_bar=True, logger=True)
        self.log("train/hard_pairs", float(nb_hard), on_step=True, on_epoch=False,
                 prog_bar=False, logger=True)
        if self.is_joint:
            self.log("train/alignment_loss", loss_align.item(), on_step=True,
                     on_epoch=False, prog_bar=False, logger=True)

        # Accumulate for console interval prints and epoch summaries.
        self._loss_acc.update(loss_vpr.item(), loss_align.item(), loss.item(), nb_hard)
        self._loss_acc.maybe_print_interval(self.global_step, self.current_epoch)

        return {"loss": loss}

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log total gradient norm before each optimiser step.

        Detects gradient explosion or vanishing early. Logged per step
        so the W&B step-axis chart shows the full training trajectory.
        """
        norms = [
            p.grad.detach().norm()
            for p in self.parameters()
            if p.grad is not None
        ]
        if norms:
            self.log("train/grad_norm", torch.stack(norms).norm().item(),
                     on_step=True, on_epoch=False, prog_bar=False, logger=True)
