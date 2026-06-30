"""VPRModel: setup, optimiser, and epoch hooks.

Training-step logic  → vpr_model_train.TrainingMixin
Validation/recall    → vpr_model_val.ValidationMixin

Supported model types (config.model.type):
  salad_baseline    — standard SALAD, no depth branch
  salad_joint_depth — SALAD + frozen DepthTeacher + AlignmentMLP + AlignmentLoss
"""
import torch
import pytorch_lightning as pl
from torch.optim import lr_scheduler
from omegaconf import DictConfig

import utils
from models import helper
from models.mlps import get_alignment_mlp
from losses import AlignmentLoss
from vpr_model_train import TrainingMixin
from vpr_model_val import ValidationMixin


class VPRModel(TrainingMixin, ValidationMixin, pl.LightningModule):
    """Visual Place Recognition model driven by a single OmegaConf config."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.faiss_gpu = cfg.training.faiss_gpu
        self.is_joint = cfg.model.type == "salad_joint_depth"

        self.backbone = helper.get_backbone(
            cfg.model.backbone.arch,
            {
                "num_trainable_blocks": cfg.model.backbone.num_trainable_blocks,
                "return_token": cfg.model.backbone.return_token,
                "norm_layer": cfg.model.backbone.norm_layer,
            },
        )
        self.aggregator = helper.get_aggregator("SALAD", dict(cfg.model.aggregator))

        self.loss_fn = utils.get_loss(cfg.loss.vpr_loss)
        self.miner = utils.get_miner(cfg.loss.miner, cfg.loss.miner_margin)
        self.batch_acc = []
        self.val_outputs = []
        self._accum = {"ms": 0.0, "align": 0.0, "total": 0.0, "n": 0}

        if self.is_joint:
            from models.teacher import DepthTeacher
            self.depth_teacher = DepthTeacher(cfg.model.teacher.name)
            self.alignment_mlp = get_alignment_mlp(cfg.model.mlp)
            self.alignment_loss = AlignmentLoss(
                loss_type=cfg.loss.alignment_loss_type,
                norm_stage=cfg.model.normalization.stage,
            )

    def forward(self, x):
        return self.aggregator(self.backbone(x))

    def on_train_epoch_end(self):
        self.batch_acc = []
        # Bug 3 fix: flush any partial window so values don't bleed into the next epoch.
        self._accum = {"ms": 0.0, "align": 0.0, "total": 0.0, "n": 0}

    def configure_optimizers(self):
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
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()
