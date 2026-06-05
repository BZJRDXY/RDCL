from __future__ import annotations

import torch
from torch import nn


class GaussianRBF(nn.Module):
    def __init__(self, num_centers: int = 32, cutoff: float = 12.0, gamma: float | None = None) -> None:
        super().__init__()
        centers = torch.linspace(0.0, float(cutoff), int(num_centers))
        self.register_buffer("centers", centers)
        self.cutoff = float(cutoff)
        self.gamma = float(gamma) if gamma is not None else 10.0 / max(cutoff, 1e-6)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        d = dist.unsqueeze(-1).clamp_min(0.0)
        return torch.exp(-self.gamma * (d - self.centers) ** 2)
