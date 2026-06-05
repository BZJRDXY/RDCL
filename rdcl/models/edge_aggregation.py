from __future__ import annotations

import torch
from torch import nn

from rdcl.models.common import MLP
from rdcl.models.rbf import GaussianRBF
from rdcl.models.segment_ops import segment_softmax, segment_sum


class DistanceAwareEdgeAggregation(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        prot_atom_dim: int,
        rbf_dim: int,
        rbf_cutoff: float,
        heads: int,
        mlp_hidden: int,
        dropout: float,
        gamma_global_local: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by edge_agg_heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.gamma = float(gamma_global_local)
        self.rbf = GaussianRBF(rbf_dim, rbf_cutoff)
        in_dim = 3 * hidden_dim + rbf_dim + prot_atom_dim
        self.atom_pair_mlp = MLP(in_dim, hidden_dim, mlp_hidden, num_layers=3, dropout=dropout)
        self.attn = nn.Linear(hidden_dim, heads)
        self.edge_post = MLP(hidden_dim, hidden_dim, mlp_hidden, num_layers=2, dropout=dropout)
        self.site_head = MLP(hidden_dim, 1, mlp_hidden, num_layers=2, dropout=dropout, norm=False)

    def forward(
        self,
        h_lig: torch.Tensor,
        lig_pos: torch.Tensor,
        h_res: torch.Tensor,
        h_pa: torch.Tensor,
        pa_pos: torch.Tensor,
        prot_atom_x: torch.Tensor,
        ar_edge_index: torch.Tensor,
        ar_edge_batch: torch.Tensor,
        ar_pair_edge_index: torch.Tensor,
        ar_pair_prot_atom_index: torch.Tensor,
        num_samples: int,
    ) -> dict:
        n_edges = ar_edge_index.shape[1]
        if n_edges == 0:
            z = h_lig.new_zeros((0, self.hidden_dim))
            return {"edge_emb": z, "site_logits": z.new_zeros(0), "site_scores": z.new_zeros(0)}

        if ar_pair_edge_index.numel() == 0:
            edge_emb = h_lig.new_zeros((n_edges, self.hidden_dim))
        else:
            e = ar_pair_edge_index.long()
            pa = ar_pair_prot_atom_index.long()
            lig = ar_edge_index[0, e].long()
            res = ar_edge_index[1, e].long()
            dist = torch.linalg.norm(lig_pos[lig] - pa_pos[pa], dim=-1)
            rbf = self.rbf(dist)
            pair_feat = torch.cat([h_lig[lig], h_res[res], h_pa[pa], rbf, prot_atom_x[pa].float()], dim=-1)
            msg = self.atom_pair_mlp(pair_feat)
            attn_logits = self.attn(msg)
            attn = segment_softmax(attn_logits, e, n_edges)  # [P, H]
            v = msg.view(msg.shape[0], self.heads, self.head_dim)
            weighted = v * attn.unsqueeze(-1)
            edge_emb_h = segment_sum(weighted, e, n_edges)  # [E, H, Hd]
            edge_emb = edge_emb_h.reshape(n_edges, self.hidden_dim)
        edge_emb = self.edge_post(edge_emb)
        site_logits = self.site_head(edge_emb).squeeze(-1)

        global_score = segment_softmax(site_logits, ar_edge_batch.long(), num_samples)
        lig_group = ar_edge_index[0].long()
        local_score = segment_softmax(site_logits, lig_group, int(h_lig.shape[0]))
        site_scores = self.gamma * global_score + (1.0 - self.gamma) * local_score
        return {"edge_emb": edge_emb, "site_logits": site_logits, "site_scores": site_scores}
