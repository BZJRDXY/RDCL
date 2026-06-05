from __future__ import annotations

import torch
from torch import nn

from rdcl.models.segment_ops import segment_mean, segment_sum


class RationaleMask(nn.Module):
    def __init__(self, temperature: float = 0.3, importance_mode: str = "mean", eps: float = 1e-6) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.importance_mode = importance_mode
        self.eps = eps

    def _aggregate_importance(self, scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if scores.numel() == 0:
            return torch.zeros(num_nodes, dtype=scores.dtype, device=scores.device)
        if self.importance_mode == "sum":
            p = segment_sum(scores, index.long(), num_nodes).clamp(0.0, 1.0)
        else:
            p = segment_mean(scores, index.long(), num_nodes).clamp(0.0, 1.0)
        return p

    def forward(
        self,
        h_lig: torch.Tensor,
        h_pa: torch.Tensor,
        prot_atom_res_index: torch.Tensor,
        ar_edge_index: torch.Tensor,
        site_scores: torch.Tensor,
        training: bool = True,
    ) -> dict:
        n_lig = h_lig.shape[0]
        n_pa = h_pa.shape[0]
        n_res_from_atoms = int(prot_atom_res_index.max().item() + 1) if prot_atom_res_index.numel() > 0 else 0
        n_res_from_edges = int(ar_edge_index[1].max().item() + 1) if ar_edge_index.numel() > 0 else 0
        n_res = max(n_res_from_atoms, n_res_from_edges)
        p_lig = self._aggregate_importance(site_scores, ar_edge_index[0], n_lig)
        p_res = self._aggregate_importance(site_scores, ar_edge_index[1], n_res)
        p_pa = p_res[prot_atom_res_index.long()].clamp(0.0, 1.0)
        p_atom = torch.cat([p_lig, p_pa], dim=0).clamp(self.eps, 1.0 - self.eps)

        h_atom = torch.cat([h_lig, h_pa], dim=0)
        if training:
            u = torch.rand_like(p_atom).clamp(self.eps, 1.0 - self.eps)
            logit = torch.log(p_atom) - torch.log1p(-p_atom) + torch.log(u) - torch.log1p(-u)
            lam = torch.sigmoid(logit / self.temperature)
        else:
            lam = p_atom
        lam = lam.unsqueeze(-1)

        mu = h_atom.mean(dim=0, keepdim=True)
        std = h_atom.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
        noise = mu + torch.randn_like(h_atom) * std if training else mu.expand_as(h_atom)
        h_r = lam * h_atom + (1.0 - lam) * noise
        h_e = (1.0 - lam) * h_atom
        return {"h_atom": h_atom, "h_rationale": h_r, "h_environment": h_e, "p_atom": p_atom, "lambda_atom": lam.squeeze(-1)}
