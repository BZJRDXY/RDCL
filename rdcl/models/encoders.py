from __future__ import annotations

import torch
from torch import nn

from rdcl.models.common import MLP
from rdcl.models.egnn import BipartiteEGNNLayer, EGNNLayer
from rdcl.models.segment_ops import segment_mean


class AtomAtomEncoder(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, layers: int, mlp_hidden: int, dropout: float, coord_update: bool, coord_scale: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_dim, mlp_hidden, dropout, coord_update, coord_scale)
            for _ in range(layers)
        ])

    def forward(self, h_atom: torch.Tensor, x_atom: torch.Tensor, aa_edge_index: torch.Tensor, aa_edge_attr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            h_atom, x_atom = layer(h_atom, x_atom, aa_edge_index, aa_edge_attr)
        return h_atom, x_atom


class AtomResidueEncoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, mlp_hidden: int, dropout: float, coord_update: bool, coord_scale: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            BipartiteEGNNLayer(hidden_dim, mlp_hidden, dropout, coord_update, coord_scale)
            for _ in range(layers)
        ])

    def forward(self, h_lig: torch.Tensor, x_lig: torch.Tensor, h_res: torch.Tensor, x_res: torch.Tensor, ar_edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            h_lig, x_lig, h_res, x_res = layer(h_lig, x_lig, h_res, x_res, ar_edge_index)
        return h_lig, x_lig, h_res, x_res


class BidirectionalTransmission(nn.Module):
    def __init__(self, hidden_dim: int, mlp_hidden: int, dropout: float) -> None:
        super().__init__()
        self.atom_to_res = MLP(hidden_dim, hidden_dim, mlp_hidden, num_layers=2, dropout=dropout)
        self.res_to_atom = MLP(hidden_dim, hidden_dim, mlp_hidden, num_layers=2, dropout=dropout)
        self.g_atom = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())
        self.g_res = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())
        self.norm_atom = nn.LayerNorm(hidden_dim)
        self.norm_res = nn.LayerNorm(hidden_dim)

    def forward(self, h_pa: torch.Tensor, h_res: torch.Tensor, prot_atom_res_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_res = h_res.shape[0]
        pooled_atom = segment_mean(h_pa, prot_atom_res_index.long(), num_res)
        atom_info_for_res = self.atom_to_res(pooled_atom)
        res_info_for_atom = self.res_to_atom(h_res)[prot_atom_res_index.long()]
        g_a = self.g_atom(torch.cat([h_pa, res_info_for_atom], dim=-1))
        g_r = self.g_res(torch.cat([h_res, atom_info_for_res], dim=-1))
        h_pa_new = self.norm_atom(g_a * h_pa + (1.0 - g_a) * res_info_for_atom)
        h_res_new = self.norm_res(g_r * h_res + (1.0 - g_r) * atom_info_for_res)
        return h_pa_new, h_res_new
