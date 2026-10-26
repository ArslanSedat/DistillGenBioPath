from __future__ import annotations
import math
import torch
import torch.nn.functional as F

def cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 - mean cosine similarity along the last dimension."""
    return 1.0 - F.cosine_similarity(a, b, dim=-1).mean()

def square_grid(num_patches: int) -> int:
    side = int(math.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(f"expected a square patch grid, got {num_patches} patches")
    return side

def resize_patch_grid(patches: torch.Tensor, target_side: int, mode: str = "bicubic") -> torch.Tensor:
    """Resample patch tokens ``[B, N, C]`` to a ``target_side x target_side`` grid."""
    b, n, c = patches.shape
    source_side = square_grid(n)
    if source_side == target_side:
        return patches
    x = patches.permute(0, 2, 1).reshape(b, c, source_side, source_side)
    x = F.interpolate(x, size=(target_side, target_side), mode=mode, align_corners=False)
    return x.reshape(b, c, target_side * target_side).permute(0, 2, 1).contiguous()

def rkd_distance_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Match the mean-normalized pairwise L2 distance matrices of the two batches."""
    b = student.shape[0]
    num_off_diag = max(b * b - b, 1)
    with torch.no_grad():
        t_d = torch.cdist(teacher, teacher, p=2)
        t_d = t_d / (t_d.sum() / num_off_diag).clamp(min=1e-8)
    s_d = torch.cdist(student, student, p=2)
    s_d = s_d / (s_d.sum() / num_off_diag).clamp(min=1e-8)
    return F.smooth_l1_loss(s_d, t_d)

def rkd_angle_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Match the triplet angle tensors of the two batches."""
    with torch.no_grad():
        t_vd = teacher.unsqueeze(0) - teacher.unsqueeze(1)
        eps = 1e-6 if t_vd.dtype == torch.float16 else 1e-12
        t_norm = F.normalize(t_vd, p=2, dim=-1, eps=eps)
        t_angle = torch.bmm(t_norm, t_norm.transpose(-2, -1))
    s_vd = student.unsqueeze(0) - student.unsqueeze(1)
    eps = 1e-6 if s_vd.dtype == torch.float16 else 1e-12
    s_norm = F.normalize(s_vd, p=2, dim=-1, eps=eps)
    s_angle = torch.bmm(s_norm, s_norm.transpose(-2, -1))
    return F.smooth_l1_loss(s_angle, t_angle)