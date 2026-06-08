"""Training entry point.

Loads config.yaml as the base configuration and applies any key=value
overrides passed on the command line, e.g.:

    python main.py model.type=salad_joint_depth loss.alpha=0.3
    python main.py loss.alignment_loss_type=cosine training.max_epochs=8
"""
import argparse

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from vpr_model import VPRModel


def parse_args():
    """Accept zero or more key=value override strings."""
    parser = argparse.ArgumentParser(description="Train SALAD VPR model")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in key=value format, e.g. model.type=salad_joint_depth",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = OmegaConf.load("config.yaml")
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    datamodule = GSVCitiesDataModule(
        batch_size=cfg.training.batch_size,
        img_per_place=cfg.training.img_per_place,
        min_img_per_place=cfg.training.min_img_per_place,
        shuffle_all=False,
        random_sample_from_each_place=True,
        image_size=tuple(cfg.training.image_size),
        num_workers=cfg.training.num_workers,
        show_data_stats=True,
        val_set_names=list(cfg.training.val_set_names),
    )

    model = VPRModel(cfg)

    logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=cfg.wandb.run_name or None,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        filename=f"{cfg.model.backbone.arch}_epoch{{epoch:02d}}_R1={{pitts30k_val/R1:.4f}}",
        auto_insert_metric_name=False,
        save_weights_only=True,
        monitor="pitts30k_val/R1",
        mode="max",
        save_top_k=3,
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        default_root_dir="./logs/",
        num_nodes=1,
        num_sanity_val_steps=0,
        precision=cfg.training.precision,
        max_epochs=cfg.training.max_epochs,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=20,
        logger=logger,
    )

    trainer.fit(model=model, datamodule=datamodule)
