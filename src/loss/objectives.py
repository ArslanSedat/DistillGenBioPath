from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from config import CLSLossConfig, PatchLossConfig
from loss.functional import (
    cosine_loss,
    resize_patch_grid,
    rkd_angle_loss,
    rkd_distance_loss,
    square_grid)

_CLS_COMPONENT_WEIGHT = {"cosine": "cosine_weight", "rkd": "rkd_weight"}

class CosineCLS(nn.Module):
    """Pointwise cosine loss between the projected student and teacher class tokens."""
    needs_projection = True
    def forward(self, student_tokens, projected, teacher_tokens, student_prefix, teacher_prefix):
        loss = cosine_loss(projected[:, 0].float(), teacher_tokens[:, 0].float())
        return {"loss": loss}

class RKDCLS(nn.Module):
    """Relational knowledge distillation on the class token (distance + angle terms)."""
    needs_projection = False
    def __init__(self, dist_weight: float, angle_weight: float):
        super().__init__()
        self.dist_weight = dist_weight
        self.angle_weight = angle_weight
    def forward(self, student_tokens, projected, teacher_tokens, student_prefix, teacher_prefix):
        student_cls = student_tokens[:, 0].float()
        teacher_cls = teacher_tokens[:, 0].float()
        dist = rkd_distance_loss(student_cls, teacher_cls)
        angle = rkd_angle_loss(student_cls, teacher_cls)
        loss = self.dist_weight * dist + self.angle_weight * angle
        return {"loss": loss, "rkd_dist": dist, "rkd_angle": angle}

class CompositeCLS(nn.Module):
    """Sum several class-token objectives, each scaled by its own weight.
    Each sub-objective's raw (unweighted) loss is exposed as a diagnostic keyed by
    its name, so the relative contribution of the terms can be monitored over training.
    """
    def __init__(self, objectives: dict[str, nn.Module], weights: dict[str, float]):
        super().__init__()
        self.objectives = nn.ModuleDict(objectives)
        self.weights = weights
        self.needs_projection = any(obj.needs_projection for obj in objectives.values())
    def forward(self, student_tokens, projected, teacher_tokens, student_prefix, teacher_prefix):
        out = {}
        total = None
        for name, objective in self.objectives.items():
            result = objective(student_tokens, projected, teacher_tokens, student_prefix, teacher_prefix)
            out[name] = result["loss"]
            for key, value in result.items():
                if key != "loss":
                    out[key] = value
            contribution = self.weights[name] * result["loss"]
            total = contribution if total is None else total + contribution
        out["loss"] = total
        return out

class CosinePatch(nn.Module):
    """Pointwise cosine loss between projected student patch tokens and the teacher
    patch tokens, after resizing the teacher grid to the student grid."""
    needs_projection = True
    def __init__(self, smooth_l1_weight: float):
        super().__init__()
        self.smooth_l1_weight = smooth_l1_weight
    def forward(self, student_tokens, projected, teacher_tokens, student_prefix, teacher_prefix):
        student_patches = projected[:, student_prefix:].float()
        teacher_patches = teacher_tokens[:, teacher_prefix:].float()
        student_side = square_grid(student_patches.shape[1])
        teacher_patches = resize_patch_grid(teacher_patches, student_side, mode="bicubic")
        loss = cosine_loss(student_patches, teacher_patches)
        if self.smooth_l1_weight > 0:
            loss = loss + self.smooth_l1_weight * F.smooth_l1_loss(student_patches, teacher_patches)
        return {"loss": loss}

def _build_cls_component(name: str, cfg: CLSLossConfig) -> nn.Module:
    if name == "cosine":
        return CosineCLS()
    if name == "rkd":
        return RKDCLS(cfg.rkd_dist_weight, cfg.rkd_angle_weight)
    raise ValueError(f"unknown cls_loss.type component: {name!r}")

def build_cls_objective(cfg: CLSLossConfig, student_dim: int, teacher_dim: int) -> nn.Module | None:
    if cfg.type == "none":
        return None
    names = cfg.type.split("+")
    if len(names) == 1:
        return _build_cls_component(names[0], cfg)
    objectives = {name: _build_cls_component(name, cfg) for name in names}
    weights = {name: getattr(cfg, _CLS_COMPONENT_WEIGHT.get(name, ""), 1.0) for name in names}
    return CompositeCLS(objectives, weights)

def build_patch_objective(cfg: PatchLossConfig, student_dim: int, teacher_dim: int) -> nn.Module | None:
    if cfg.type == "none":
        return None
    if cfg.type == "cosine":
        return CosinePatch(cfg.smooth_l1_weight)
    raise ValueError(f"unknown patch_loss.type: {cfg.type}")