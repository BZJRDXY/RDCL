from __future__ import annotations

import torch
from torch import nn

from rdcl.models.common import MLP
from rdcl.models.segment_ops import segment_sum


class EGNNLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        mlp_hidden: int,
        dropout: float = 0.0,
        coord_update: bool = True,
        coord_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coord_update = coord_update
        self.coord_scale = coord_scale
        self.edge_mlp = MLP(2 * hidden_dim + edge_dim + 1, hidden_dim, mlp_hidden, num_layers=3, dropout=dropout)
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, mlp_hidden, num_layers=2, dropout=dropout)
        self.coord_mlp = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1))
        nn.init.zeros_(self.coord_mlp[-1].weight)
        nn.init.zeros_(self.coord_mlp[-1].bias)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return h, x
        src, dst = edge_index[0].long(), edge_index[1].long()
        rel = x[src] - x[dst]
        dist2 = (rel * rel).sum(-1, keepdim=True)
        msg = self.edge_mlp(torch.cat([h[src], h[dst], dist2, edge_attr], dim=-1))
        agg = segment_sum(msg, dst, h.shape[0])
        dh = self.node_mlp(torch.cat([h, agg], dim=-1))
        h = self.norm(h + dh)
        if self.coord_update:
            coef = self.coord_mlp(msg).clamp(-10, 10) * self.coord_scale
            dx = segment_sum(rel * coef, dst, x.shape[0])
            x = x + dx
        return h, x


class BipartiteEGNNLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden: int,
        dropout: float = 0.0,
        coord_update: bool = True,
        coord_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.coord_update = coord_update
        self.coord_scale = coord_scale
        self.msg_lr = MLP(2 * hidden_dim + 1, hidden_dim, mlp_hidden, num_layers=3, dropout=dropout)
        self.msg_rl = MLP(2 * hidden_dim + 1, hidden_dim, mlp_hidden, num_layers=3, dropout=dropout)
        self.up_lig = MLP(2 * hidden_dim, hidden_dim, mlp_hidden, num_layers=2, dropout=dropout)
        self.up_res = MLP(2 * hidden_dim, hidden_dim, mlp_hidden, num_layers=2, dropout=dropout)
        self.coord_lig = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1))
        self.coord_res = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1))
        nn.init.zeros_(self.coord_lig[-1].weight); nn.init.zeros_(self.coord_lig[-1].bias)
        nn.init.zeros_(self.coord_res[-1].weight); nn.init.zeros_(self.coord_res[-1].bias)
        self.norm_lig = nn.LayerNorm(hidden_dim)
        self.norm_res = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_lig: torch.Tensor,
        x_lig: torch.Tensor,
        h_res: torch.Tensor,
        x_res: torch.Tensor,
        ar_edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if ar_edge_index.numel() == 0:
            return h_lig, x_lig, h_res, x_res
        lig, res = ar_edge_index[0].long(), ar_edge_index[1].long()
        rel = x_lig[lig] - x_res[res]
        dist2 = (rel * rel).sum(-1, keepdim=True)
        msg_lr = self.msg_lr(torch.cat([h_lig[lig], h_res[res], dist2], dim=-1))
        msg_rl = self.msg_rl(torch.cat([h_res[res], h_lig[lig], dist2], dim=-1))

        agg_res = segment_sum(msg_lr, res, h_res.shape[0])
        agg_lig = segment_sum(msg_rl, lig, h_lig.shape[0])
        h_res = self.norm_res(h_res + self.up_res(torch.cat([h_res, agg_res], dim=-1)))
        h_lig = self.norm_lig(h_lig + self.up_lig(torch.cat([h_lig, agg_lig], dim=-1)))

        if self.coord_update:
            c_res = self.coord_res(msg_lr).clamp(-10, 10) * self.coord_scale
            c_lig = self.coord_lig(msg_rl).clamp(-10, 10) * self.coord_scale
            # Update residue toward ligand and ligand toward residue with opposite relative vector.
            x_res = x_res + segment_sum((-rel) * c_res, res, x_res.shape[0])
            x_lig = x_lig + segment_sum(rel * c_lig, lig, x_lig.shape[0])
        return h_lig, x_lig, h_res, x_res
