"""Training-step mixin for VPRModel.

Extracted from vpr_model.py to keep that file under 100 lines.
Contains the metric loss helper, the joint-depth forward pass, and the
interval console/W&B logging logic.
"""
import math
import torch
import torch.nn.functional as F


class TrainingMixin:
    """Provides training_step, _vpr_loss, and interval logging for VPRModel."""

    def _vpr_loss(self, descriptors, labels):
        """Compute configured metric loss and track trivial-pair accuracy.

        Args:
            descriptors: L2-normalised global descriptors [B, D].
            labels: place-id integer labels [B].

        Returns:
            Scalar loss tensor.
        """
        if self.miner is not None:
            mined = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, mined)
            nb_mined = len(set(mined[0].detach().cpu().numpy()))
            b_acc = 1.0 - nb_mined / descriptors.shape[0]
        else:
            loss = self.loss_fn(descriptors, labels)
            b_acc = 0.0
            if isinstance(loss, tuple):
                loss, b_acc = loss
        self.batch_acc.append(b_acc)
        self.log("b_acc", sum(self.batch_acc) / len(self.batch_acc), prog_bar=True)
        return loss

    def _log_at_interval(self, loss_vpr, loss_align, loss_total):
        """Accumulate running averages and log every cfg.training.log_interval steps.

        Uses .item() on every update so no computation graph is retained.
        """
        self._accum["ms"] += loss_vpr.item()
        self._accum["align"] += loss_align.item()
        self._accum["total"] += loss_total.item()
        self._accum["n"] += 1

        if self._accum["n"] < self.cfg.training.log_interval:
            return

        n = self._accum["n"]
        avg_ms = self._accum["ms"] / n
        avg_align = self._accum["align"] / n
        avg_total = self._accum["total"] / n

        print(
            f"[Epoch {self.current_epoch}, Iter {self.global_step}] "
            f"Avg MS Loss: {avg_ms:.4f} | "
            f"Avg Alignment Loss: {avg_align:.4f} | "
            f"Avg Total Loss: {avg_total:.4f}"
        )

        # Bug 2 fix: ratio metrics only make sense when the alignment branch is active.
        # For salad_baseline avg_align ~ 0, so the ratio would be a meaningless large number.
        log_payload = {
            "losses/ms_loss": avg_ms,
            "losses/total_loss": avg_total,
            "trainer/global_step": self.global_step,
        }
        if self.is_joint:
            ratio = avg_ms / (avg_align + 1e-8)
            log_payload["losses/alignment_loss"] = avg_align
            log_payload["metrics/raw_ratio"] = ratio
            log_payload["metrics/log_ratio"] = math.log10(ratio + 1e-8)

        # Bug 1 fix: guard against running without WandbLogger (e.g. local debug runs).
        if hasattr(self.trainer.logger, "experiment"):
            self.trainer.logger.experiment.log(log_payload)

        self._accum = {"ms": 0.0, "align": 0.0, "total": 0.0, "n": 0}

    def training_step(self, batch, batch_idx):
        places, labels = batch
        BS, N, ch, h, w = places.shape
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)

        backbone_out = self.backbone(images)
        descriptors = self.aggregator(backbone_out)
        loss_vpr = self._vpr_loss(descriptors, labels)
        loss_align = torch.tensor(0.0, device=loss_vpr.device)

        if self.is_joint:
            feat_map = backbone_out[0]  # [B, C, H, W]
            student = feat_map.flatten(2).permute(0, 2, 1)  # [B, N, C]
            if self.cfg.model.normalization.stage == "before_mlp":
                student = F.normalize(student, p=2, dim=-1)
            student = self.alignment_mlp(student)
            with torch.cuda.amp.autocast(enabled=False):
                teacher = self.depth_teacher(images.float())
            loss_align = self.alignment_loss(student, teacher.to(student.dtype))

        alpha = self.cfg.loss.alpha if self.is_joint else 0.0
        loss = loss_vpr + alpha * loss_align

        self._log_at_interval(loss_vpr, loss_align, loss)
        self.log("loss", loss.item(), prog_bar=True, logger=True)
        return {"loss": loss}
