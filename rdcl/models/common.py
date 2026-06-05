from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        norm: bool = True,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, out_dim)
        layers = []
        d = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            if norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim, hidden_dim, num_layers=2, dropout=dropout, norm=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))
