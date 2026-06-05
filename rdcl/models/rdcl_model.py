from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from rdcl.models.common import MLP
from rdcl.models.edge_aggregation import DistanceAwareEdgeAggregation
from rdcl.models.encoders import AtomAtomEncoder, AtomResidueEncoder, BidirectionalTransmission
from rdcl.models.rationale_mask import RationaleMask
from rdcl.models.segment_ops import segment_mean
from rdcl.models.set2set import Set2Set


class RDCLBaseModel(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        mcfg = config["model"]
        idim = config["input_dims"]
        d = int(mcfg["hidden_dim"])
        node_h = int(mcfg.get("node_mlp_hidden", 2 * d))
        edge_h = int(mcfg.get("edge_mlp_hidden", 2 * d))
        dropout = float(mcfg.get("dropout", 0.0))

        self.hidden_dim = d
        self.use_atom_res_lig_output = bool(mcfg.get("use_atom_res_lig_output", True))

        self.lig_proj = MLP(int(idim["lig_atom_dim"]), d, node_h, num_layers=2, dropout=dropout)
        self.pa_proj = MLP(int(idim["prot_atom_dim"]), d, node_h, num_layers=2, dropout=dropout)
        self.res_feat_proj = MLP(int(idim["prot_res_dim"]), d, node_h, num_layers=2, dropout=dropout)
        self.esm_proj = MLP(int(idim["esm_dim"]), d, node_h, num_layers=2, dropout=dropout)
        self.res_init = MLP(3 * d, d, node_h, num_layers=2, dropout=dropout)

        self.atom_atom = AtomAtomEncoder(
            d,
            int(idim.get("aa_edge_dim", 10)),
            int(mcfg.get("atom_atom_layers", 3)),
            edge_h,
            dropout,
            bool(mcfg.get("coord_update", True)),
            float(mcfg.get("coord_scale", 0.05)),
        )
        self.atom_residue = AtomResidueEncoder(
            d,
            int(mcfg.get("atom_residue_layers", 3)),
            edge_h,
            dropout,
            bool(mcfg.get("coord_update", True)),
            float(mcfg.get("coord_scale", 0.05)),
        )
        self.transmission = BidirectionalTransmission(d, node_h, dropout)
        self.edge_agg = DistanceAwareEdgeAggregation(
            hidden_dim=d,
            prot_atom_dim=int(idim["prot_atom_dim"]),
            rbf_dim=int(mcfg.get("rbf_dim", 32)),
            rbf_cutoff=float(mcfg.get("rbf_cutoff", 12.0)),
            heads=int(mcfg.get("edge_agg_heads", 8)),
            mlp_hidden=edge_h,
            dropout=dropout,
            gamma_global_local=float(mcfg.get("gamma_global_local", 0.5)),
        )
        self.mask = RationaleMask(
            temperature=float(mcfg.get("gumbel_temperature", 0.3)),
            importance_mode=str(mcfg.get("importance_mode", "mean")),
        )
        self.readout = Set2Set(d, processing_steps=int(mcfg.get("set2set_steps", 4)))
        self.affinity_head = MLP(2 * d, 1, node_h, num_layers=3, dropout=dropout)
        self.environment_head = MLP(2 * d, 1, node_h, num_layers=3, dropout=dropout)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        device = batch["lig_x"].device
        num_samples = int(batch["num_samples"])
        n_atoms = int(batch["num_atoms_combined"])

        h_lig0 = self.lig_proj(batch["lig_x"].float())
        h_pa0 = self.pa_proj(batch["prot_atom_x"].float())
        h_atom0 = torch.zeros((n_atoms, self.hidden_dim), dtype=h_lig0.dtype, device=device)
        x_atom0 = torch.zeros((n_atoms, 3), dtype=batch["lig_pos"].dtype, device=device)
        h_atom0[batch["lig_node_index"].long()] = h_lig0
        h_atom0[batch["prot_atom_node_index"].long()] = h_pa0
        x_atom0[batch["lig_node_index"].long()] = batch["lig_pos"].float()
        x_atom0[batch["prot_atom_node_index"].long()] = batch["prot_atom_pos"].float()

        h_atom, x_atom = self.atom_atom(h_atom0, x_atom0, batch["aa_edge_index"].long(), batch["aa_edge_attr"].float())
        h_lig_aa = h_atom[batch["lig_node_index"].long()]
        h_pa_aa = h_atom[batch["prot_atom_node_index"].long()]
        x_lig_aa = x_atom[batch["lig_node_index"].long()]
        x_pa_aa = x_atom[batch["prot_atom_node_index"].long()]

        pooled_pa_to_res = segment_mean(h_pa_aa, batch["prot_atom_res_index"].long(), batch["res_x"].shape[0])
        h_res0 = self.res_init(torch.cat([
            self.res_feat_proj(batch["res_x"].float()),
            self.esm_proj(batch["esm_x"].float()),
            pooled_pa_to_res,
        ], dim=-1))

        h_lig_ar, x_lig_ar, h_res_ar, x_res_ar = self.atom_residue(
            h_lig_aa,
            x_lig_aa,
            h_res0,
            batch["res_pos"].float(),
            batch["ar_edge_index"].long(),
        )
        h_pa_star, h_res_star = self.transmission(h_pa_aa, h_res_ar, batch["prot_atom_res_index"].long())
        h_lig_star = h_lig_ar if self.use_atom_res_lig_output else h_lig_aa
        x_lig_star = x_lig_ar if self.use_atom_res_lig_output else x_lig_aa

        edge_out = self.edge_agg(
            h_lig=h_lig_star,
            lig_pos=x_lig_star,
            h_res=h_res_star,
            h_pa=h_pa_star,
            pa_pos=x_pa_aa,
            prot_atom_x=batch["prot_atom_x"].float(),
            ar_edge_index=batch["ar_edge_index"].long(),
            ar_edge_batch=batch["ar_edge_batch"].long(),
            ar_pair_edge_index=batch["ar_pair_edge_index"].long(),
            ar_pair_prot_atom_index=batch["ar_pair_prot_atom_index"].long(),
            num_samples=num_samples,
        )

        mask_out = self.mask(
            h_lig=h_lig_star,
            h_pa=h_pa_star,
            prot_atom_res_index=batch["prot_atom_res_index"].long(),
            ar_edge_index=batch["ar_edge_index"].long(),
            site_scores=edge_out["site_scores"],
            training=self.training,
        )
        atom_batch = torch.cat([batch["lig_batch"].long(), batch["prot_atom_batch"].long()], dim=0)
        z_r = self.readout(mask_out["h_rationale"], atom_batch, num_samples)
        z_e = self.readout(mask_out["h_environment"], atom_batch, num_samples)
        affinity_pred = self.affinity_head(z_r).squeeze(-1)
        env_affinity_pred = self.environment_head(z_e).squeeze(-1)

        out = {
            **edge_out,
            **mask_out,
            "z_r": z_r,
            "z_e": z_e,
            "affinity_pred": affinity_pred,
            "env_affinity_pred": env_affinity_pred,
        }
        return out
