from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.nn.utils.parametrizations import weight_norm

class DINOHead(nn.Module):
    """MLP -> L2-normalize -> weight-normalized linear, applied over the last dimension.
    Maps the student token sequence into the teacher feature space for the cosine
    losses. The head is discarded after training, so inference uses the student
    backbone alone.
    """
    def __init__(self, in_dim: int, out_dim: int, nlayers: int = 3, hidden_dim: int = 2048, bottleneck_dim: int = 256,):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(nlayers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        return self.last_layer(x)

def build_projector(variant: str, in_dim: int, out_dim: int) -> nn.Module:
    if variant == "dino":
        return DINOHead(in_dim=in_dim, out_dim=out_dim)
    raise ValueError(f"unknown projector variant: {variant}")