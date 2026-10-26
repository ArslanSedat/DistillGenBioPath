from __future__ import annotations
import torch
import torch.nn as nn
from distillpath.config import TrainConfig
from distillpath.losses import build_cls_objective, build_patch_objective, build_projector

class Distiller(nn.Module):
    def __init__(self, cfg: TrainConfig, student_dim: int, teacher_dim: int):
        super().__init__()
        self.cls_weight = cfg.cls_loss.weight
        self.patch_weight = cfg.patch_loss.weight
        self.cls_objective = build_cls_objective(cfg.cls_loss, student_dim, teacher_dim)
        self.patch_objective = build_patch_objective(cfg.patch_loss, student_dim, teacher_dim)
        needs_projection = (self.cls_objective is not None and self.cls_objective.needs_projection) or (self.patch_objective is not None and self.patch_objective.needs_projection)
        self.projector = (build_projector(cfg.projector.variant, student_dim, teacher_dim) if needs_projection else None)

    def forward(self, student_tokens, teacher_tokens, student_prefix, teacher_prefix):
        projected = self.projector(student_tokens) if self.projector is not None else None
        out: dict[str, torch.Tensor] = {}
        total = None
        if self.cls_objective is not None:
            result = self.cls_objective(student_tokens, projected, teacher_tokens, student_prefix, teacher_prefix)
            out["cls_loss"] = result["loss"]
            for key, value in result.items():
                if key != "loss":
                    out[f"cls_{key}"] = value
            total = self.cls_weight * result["loss"]

        if self.patch_objective is not None:
            result = self.patch_objective(student_tokens, projected, teacher_tokens, student_prefix, teacher_prefix)
            out["patch_loss"] = result["loss"]
            for key, value in result.items():
                if key != "loss":
                    out[f"patch_{key}"] = value
            weighted = self.patch_weight * result["loss"]
            total = weighted if total is None else total + weighted

        out["total_loss"] = total
        return out