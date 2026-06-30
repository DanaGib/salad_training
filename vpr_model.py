"""VPRModel: setup and epoch hooks.

Training-step + grad-norm  → vpr_model_train.TrainingMixin
Validation/recall          → vpr_model_val.ValidationMixin
Optimiser/scheduler        → vpr_model_optim.OptimiserMixin

Supported model types (config.model.type):
  salad_baseline    — standard SALAD, no depth branch
  salad_joint_depth — SALAD + frozen DepthTeacher + AlignmentMLP + AlignmentLoss
"""
import pytorch_lightning as pl
from omegaconf import DictConfig

import utils
from models import helper
from models.mlps import get_alignment_mlp
from losses import AlignmentLoss
from vpr_model_train import TrainingMixin
from vpr_model_val import ValidationMixin
from vpr_model_optim import OptimiserMixin
from utils import LossAccumulator



class VPRModel(TrainingMixin, ValidationMixin, OptimiserMixin, pl.LightningModule):
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
        self.val_outputs = []
        self._loss_acc = LossAccumulator(cfg.training.log_interval)

        if self.is_joint:
            from models.teacher import DepthTeacher
            self.depth_teacher = DepthTeacher(cfg.model.teacher.name)
            self.alignment_mlp = get_alignment_mlp(cfg.model.mlp)
            self.alignment_loss = AlignmentLoss(
                loss_type=cfg.loss.alignment_loss_type,
            )

    def forward(self, x):
        return self.aggregator(self.backbone(x))
    
    def on_fit_start(self) -> None:
        """Bind W&B metric x-axes once the run is initialized."""
        import wandb
        wandb.define_metric("trainer/global_step")
        wandb.define_metric("train/*", step_metric="trainer/global_step")
        wandb.define_metric("train_epoch/*", step_metric="epoch")

    def on_train_epoch_end(self) -> None:
        """Log epoch-level loss averages and alpha-analysis metrics to W&B."""
        avgs = self._loss_acc.epoch_averages()
        alpha = self.cfg.loss.alpha if self.is_joint else 0.0

        self.log("train_epoch/ms_loss", avgs["ms"], on_step=False, on_epoch=True)
        self.log("train_epoch/total_loss", avgs["total"], on_step=False, on_epoch=True)
        self.log("train_epoch/hard_pairs_avg", avgs["hard_pairs_avg"],
                 on_step=False, on_epoch=True)

        if self.is_joint:
            self.log("train_epoch/alignment_loss", avgs["align"],
                     on_step=False, on_epoch=True)
            ratio = avgs["ms"] / (avgs["align"] + 1e-8)
            contrib = 100.0 * alpha * avgs["align"] / (avgs["total"] + 1e-8)
            self.log("train_epoch/loss_scale_ratio", ratio, on_step=False, on_epoch=True)
            self.log("train_epoch/align_contribution_pct", contrib,
                     on_step=False, on_epoch=True)

        sched = self.lr_schedulers()
        if sched is not None:
            self.log("train_epoch/learning_rate", sched.get_last_lr()[0],
                     on_step=False, on_epoch=True)

        self._loss_acc.reset_epoch()
