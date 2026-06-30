"""Depth-branch loss helpers for VPRModel training.

Each public function maps to one model type and returns a dict of scalar
loss tensors. training_step assembles the total with alpha weights.
"""
import torch
import torch.nn.functional as F


def _reshape_to_spatial(patches: torch.Tensor) -> torch.Tensor:
    """[B, N, C] -> [B, C, sqrt(N), sqrt(N)]."""
    B, N, C = patches.shape
    H = int(N ** 0.5)
    return patches.permute(0, 2, 1).reshape(B, C, H, H)


def _depth_salad_loss(model, feat_map_d, cls_d, labels):
    """Apply optional depth_proj, run shared SALAD, return metric loss."""
    if hasattr(model, "depth_proj"):
        B, C, H, W = feat_map_d.shape
        flat = model.depth_proj(feat_map_d.flatten(2).permute(0, 2, 1))
        feat_map_d = flat.permute(0, 2, 1).reshape(B, C, H, W)
        cls_d = F.normalize(model.depth_proj(cls_d), p=2, dim=-1)
    loss, _ = model._vpr_loss(model.aggregator((feat_map_d, cls_d)), labels)
    return loss


def compute_joint_depth_losses(model, images: torch.Tensor, feat_map: torch.Tensor) -> dict:
    """Local alignment loss for salad_joint_depth."""
    student = model.alignment_mlp(feat_map.flatten(2).permute(0, 2, 1))
    with torch.cuda.amp.autocast(enabled=False):
        teacher = model.depth_teacher(images.float())
    return {"local": model.alignment_loss(student, teacher.to(student.dtype))}


def compute_predictor_global_losses(
    model, images: torch.Tensor, feat_map: torch.Tensor, labels: torch.Tensor
) -> dict:
    """Local + global losses for salad_predictor_global.

    Predictor output is reshaped to spatial and passed through the shared
    SALAD head. Depth teacher CLS token is used alongside the predictor map.
    """
    student = model.alignment_mlp(feat_map.flatten(2).permute(0, 2, 1))
    with torch.cuda.amp.autocast(enabled=False):
        teacher_patches = model.depth_teacher(images.float())
        _, cls_d = model.depth_teacher.forward_salad_format(images.float())
    loss_local = model.alignment_loss(student, teacher_patches.to(student.dtype))
    d_pred = model.aggregator((_reshape_to_spatial(student), cls_d.to(student.dtype)))
    loss_global, _ = model._vpr_loss(d_pred, labels)
    return {"local": loss_local, "global_depth": loss_global}


def compute_global_depth_losses(
    model, images: torch.Tensor, labels: torch.Tensor
) -> dict:
    """Global MS loss on depth features for salad_global_depth."""
    with torch.cuda.amp.autocast(enabled=False):
        feat_map_d, cls_d = model.depth_teacher.forward_salad_format(images.float())
    loss = _depth_salad_loss(model, feat_map_d.to(images.dtype), cls_d.to(images.dtype), labels)
    return {"global_depth": loss}


def compute_global_local_depth_losses(
    model, images: torch.Tensor, feat_map: torch.Tensor, labels: torch.Tensor
) -> dict:
    """Global depth MS loss + local alignment for salad_global_local_depth.

    Depth teacher is called once: spatial output feeds SALAD for the global
    path; flat patch tokens feed the local alignment loss directly.
    """
    with torch.cuda.amp.autocast(enabled=False):
        feat_map_d, cls_d = model.depth_teacher.forward_salad_format(images.float())
    loss_global = _depth_salad_loss(
        model, feat_map_d.to(images.dtype), cls_d.to(images.dtype), labels
    )
    teacher_patches = feat_map_d.to(images.dtype).flatten(2).permute(0, 2, 1)
    student = model.alignment_mlp(feat_map.flatten(2).permute(0, 2, 1))
    loss_local = model.alignment_loss(student, teacher_patches.to(student.dtype))
    return {"local": loss_local, "global_depth": loss_global}
