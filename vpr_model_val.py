"""Validation mixin for VPRModel.

This module contains the Lightning validation hooks extracted from vpr_model.py
so that the core training file stays under 100 lines. No logic is changed here.
VPRModel inherits from this mixin alongside pl.LightningModule.
"""
import torch
import utils


class ValidationMixin:
    """Provides Lightning validation hooks: step, epoch start/end.

    Computes Recall@K (K = 1, 5, 10, 15, 20, 50, 100) via FAISS nearest-
    neighbour search and logs results to the Lightning logger (and W&B when
    WandbLogger is active).
    """

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)
        self.val_outputs[dataloader_idx or 0].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()

    def on_validation_epoch_start(self):
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]

    def on_validation_epoch_end(self):
        """Compute and log Recall@K for every validation dataset.

        Datasets always return descriptors as references first, then queries:
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.val_outputs
        dm = self.trainer.datamodule

        for i, (val_set_name, val_dataset) in enumerate(
            zip(dm.val_set_names, dm.val_datasets)
        ):
            feats = torch.concat(val_step_outputs[i], dim=0)

            # Unified API: datasets expose num_references and ground_truth directly.
            # Older .mat-based datasets fall back to dbStruct / getPositives().
            if hasattr(val_dataset, "num_references"):
                num_references = val_dataset.num_references
                positives = val_dataset.ground_truth
            elif "pitts" in val_set_name:
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif "msls" in val_set_name:
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                raise NotImplementedError(
                    f"Please implement validation_epoch_end for {val_set_name}"
                )

            r_list = feats[:num_references]
            q_list = feats[num_references:]
            recall_dict = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
            )
            del r_list, q_list, feats, num_references, positives

            self.log(f"{val_set_name}/R1", recall_dict[1], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R5", recall_dict[5], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R10", recall_dict[10], prog_bar=False, logger=True)

        print("\n")
        self.val_outputs = []
