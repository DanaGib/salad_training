"""VPRModel: Lightning module setup and epoch hooks.

Supported model.type values:
  salad_baseline | salad_joint_depth | salad_predictor_global
  salad_global_depth | salad_global_local_depth
"""
import torch.nn as nn
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

_WITH_TEACHER = {"salad_joint_depth", "salad_predictor_global",
                 "salad_global_depth", "salad_global_local_depth"}
_WITH_MLP = {"salad_joint_depth", "salad_predictor_global", "salad_global_local_depth"}
_WITH_GLOBAL = {"salad_global_depth", "salad_global_local_depth"}


class VPRModel(TrainingMixin, ValidationMixin, OptimiserMixin, pl.LightningModule):
    """Visual Place Recognition model driven by a single OmegaConf config."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.faiss_gpu = cfg.training.faiss_gpu
        mt = cfg.model.type

        self.is_joint = mt == "salad_joint_depth"
        self.is_predictor_global = mt == "salad_predictor_global"
        self.is_global_depth = mt in _WITH_GLOBAL
        self.is_global_local = mt == "salad_global_local_depth"

        bb_cfg = cfg.model.backbone
        self.backbone = helper.get_backbone(bb_cfg.arch, {
            "num_trainable_blocks": bb_cfg.num_trainable_blocks,
            "return_token": bb_cfg.return_token,
            "norm_layer": bb_cfg.norm_layer,
        })
        self.aggregator = helper.get_aggregator("SALAD", dict(cfg.model.aggregator))
        init_ckpt = getattr(cfg.model, "init_aggregator_from", None)
        if init_ckpt:
            from utils.checkpoint import load_aggregator_weights
            load_aggregator_weights(self.aggregator, init_ckpt)
        self.loss_fn = utils.get_loss(cfg.loss.vpr_loss)
        self.miner = utils.get_miner(cfg.loss.miner, cfg.loss.miner_margin)
        self.val_outputs = []
        self._loss_acc = LossAccumulator(cfg.training.log_interval)

        if mt in _WITH_TEACHER:
            from models.teacher import DepthTeacher
            self.depth_teacher = DepthTeacher(cfg.model.teacher.name)
        if mt in _WITH_MLP:
            self.alignment_mlp = get_alignment_mlp(cfg.model.mlp)
            self.alignment_loss = AlignmentLoss(loss_type=cfg.loss.alignment_loss_type)
        if self.is_global_depth and getattr(cfg.loss, "use_linear_proj", False):
            self.depth_proj = nn.Linear(768, 768)

    def forward(self, x):
        return self.aggregator(self.backbone(x))

    def on_fit_start(self) -> None:
        """Bind W&B metric x-axes once the run is initialized."""
        import wandb
        wandb.define_metric("trainer/global_step")
        wandb.define_metric("train/*", step_metric="trainer/global_step")
        wandb.define_metric("train_epoch/*", step_metric="epoch")

    def on_train_epoch_end(self) -> None:
        """Log epoch-level loss averages to W&B."""
        avgs = self._loss_acc.epoch_averages()
        a_l = getattr(self.cfg.loss, "alpha_local", 0.0)
        a_g = getattr(self.cfg.loss, "alpha_global", 0.0)
        uses_local = self.is_joint or self.is_predictor_global or self.is_global_local
        uses_global = self.is_predictor_global or self.is_global_depth or self.is_global_local
        tot = avgs["total"] + 1e-8

        logs = {
            "train_epoch/ms_loss": avgs["ms"],
            "train_epoch/total_loss": avgs["total"],
            "train_epoch/hard_pairs_avg": avgs["hard_pairs_avg"],
        }
        if uses_local:
            logs["train_epoch/local_loss"] = avgs["align"]
            logs["train_epoch/local_contribution_pct"] = 100.0 * a_l * avgs["align"] / tot
        if uses_global:
            logs["train_epoch/global_depth_loss"] = avgs["global_depth"]
            logs["train_epoch/global_depth_contribution_pct"] = 100.0 * a_g * avgs["global_depth"] / tot
        sched = self.lr_schedulers()
        if sched is not None:
            logs["train_epoch/learning_rate"] = sched.get_last_lr()[0]
        for k, v in logs.items():
            self.log(k, v, on_step=False, on_epoch=True)
        self._loss_acc.reset_epoch()
