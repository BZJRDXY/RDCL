from __future__ import annotations

from typing import Dict, Tuple

import torch


def unique_edge_index(edge_index: torch.Tensor, n_dst: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index.reshape(2, 0)
    keys = edge_index[0].long() * int(n_dst) + edge_index[1].long()
    keys = torch.unique(keys, sorted=True)
    src = keys // int(n_dst)
    dst = keys % int(n_dst)
    return torch.stack([src, dst], dim=0).long()


def label_edges(edge_index: torch.Tensor, positive_pair_index: torch.Tensor, n_res: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros(0, dtype=torch.float32)
    pos_keys = set((positive_pair_index[0].long() * int(n_res) + positive_pair_index[1].long()).tolist())
    keys = (edge_index[0].long() * int(n_res) + edge_index[1].long()).tolist()
    labels = torch.tensor([1.0 if k in pos_keys else 0.0 for k in keys], dtype=torch.float32)
    return labels


def build_ar_edges(
    lig_pos: torch.Tensor,
    res_pos: torch.Tensor,
    positive_pair_index: torch.Tensor,
    cutoff: float = 7.0,
    force_include_positive_pairs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int | float]]:
    n_lig = lig_pos.shape[0]
    n_res = res_pos.shape[0]
    if n_lig == 0 or n_res == 0:
        raise ValueError("empty ligand or residue set")

    dist = torch.cdist(lig_pos.float(), res_pos.float())
    src, dst = torch.nonzero(dist <= float(cutoff), as_tuple=True)
    cutoff_edges = torch.stack([src, dst], dim=0).long()

    pos = positive_pair_index.long()
    valid = (pos[0] >= 0) & (pos[0] < n_lig) & (pos[1] >= 0) & (pos[1] < n_res)
    pos = pos[:, valid]
    n_pos = int(pos.shape[1])

    if force_include_positive_pairs and n_pos > 0:
        ar_edges = torch.cat([cutoff_edges, pos], dim=1)
        ar_edges = unique_edge_index(ar_edges, n_dst=n_res)
    else:
        ar_edges = unique_edge_index(cutoff_edges, n_dst=n_res)

    labels = label_edges(ar_edges, pos, n_res=n_res)

    cutoff_keys = set((cutoff_edges[0] * n_res + cutoff_edges[1]).tolist())
    pos_keys = set((pos[0] * n_res + pos[1]).tolist())
    n_pos_in_cutoff = len(cutoff_keys.intersection(pos_keys))

    stats = {
        "n_cutoff_edges": int(cutoff_edges.shape[1]),
        "n_union_edges": int(ar_edges.shape[1]),
        "n_positive_pairs": int(n_pos),
        "n_positive_in_cutoff": int(n_pos_in_cutoff),
        "n_positive_outside_cutoff": int(n_pos - n_pos_in_cutoff),
        "positive_coverage": float(n_pos_in_cutoff / max(n_pos, 1)),
    }
    return ar_edges, labels, stats


def build_residue_atom_pairs(ar_edge_index: torch.Tensor, prot_res_atom_ptr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return pair lists used by distance-aware aggregation.

    ar_pair_edge_index[p] gives the atom-residue edge id.
    ar_pair_prot_atom_index[p] gives a protein atom inside that edge's residue.
    """
    e_ids = []
    a_ids = []
    res_idx = ar_edge_index[1].long()
    ptr = prot_res_atom_ptr.long()
    for e, r in enumerate(res_idx.tolist()):
        start = int(ptr[r])
        end = int(ptr[r + 1])
        if end <= start:
            continue
        e_ids.extend([e] * (end - start))
        a_ids.extend(range(start, end))
    if not e_ids:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    return torch.tensor(e_ids, dtype=torch.long), torch.tensor(a_ids, dtype=torch.long)


def _make_aa_edge_attr(edge_type: int, n_edges: int, lig_bond_attr: torch.Tensor | None = None) -> torch.Tensor:
    # type one-hot: ligand covalent, protein covalent, induced non-covalent + 7 ligand bond attrs
    out = torch.zeros((n_edges, 10), dtype=torch.float32)
    if n_edges == 0:
        return out
    out[:, edge_type] = 1.0
    if lig_bond_attr is not None and lig_bond_attr.numel() > 0:
        dim = min(7, lig_bond_attr.shape[1])
        out[:, 3 : 3 + dim] = lig_bond_attr[:, :dim].float()
    return out


def build_aa_graph(sample: Dict, ar_edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n_lig = sample["lig_x"].shape[0]
    n_pa = sample["prot_atom_x"].shape[0]
    device = sample["lig_x"].device

    edge_indices = []
    edge_attrs = []

    lig_e = sample["lig_bond_edge_index"].long()
    lig_attr = sample["lig_bond_edge_attr"].float()
    if lig_e.numel() > 0:
        edge_indices.append(lig_e)
        edge_attrs.append(_make_aa_edge_attr(0, lig_e.shape[1], lig_attr))

    prot_e = sample["prot_atom_bond_index"].long()
    if prot_e.numel() > 0:
        edge_indices.append(prot_e + n_lig)
        edge_attrs.append(_make_aa_edge_attr(1, prot_e.shape[1], None))

    # Induced ligand atom - protein atom edges from atom-residue candidate edges.
    pair_e, pair_pa = build_residue_atom_pairs(ar_edge_index, sample["prot_res_atom_ptr"])
    if pair_e.numel() > 0:
        lig_idx = ar_edge_index[0, pair_e]
        pa_idx = pair_pa + n_lig
        nc = torch.stack([lig_idx, pa_idx], dim=0).long()
        nc_rev = torch.stack([pa_idx, lig_idx], dim=0).long()
        nc_all = torch.cat([nc, nc_rev], dim=1)
        edge_indices.append(nc_all)
        edge_attrs.append(_make_aa_edge_attr(2, nc_all.shape[1], None))

    if not edge_indices:
        return torch.zeros((2, 0), dtype=torch.long, device=device), torch.zeros((0, 10), dtype=torch.float32, device=device)

    aa_edge_index = torch.cat(edge_indices, dim=1).long()
    aa_edge_attr = torch.cat(edge_attrs, dim=0).float()
    return aa_edge_index, aa_edge_attr


def build_rdcl_graph(sample: Dict, ar_cutoff: float = 7.0, force_include_positive_pairs: bool = True) -> Dict:
    ar_edge_index, ar_label, stats = build_ar_edges(
        sample["lig_pos"],
        sample["res_pos"],
        sample["positive_pair_index"],
        cutoff=ar_cutoff,
        force_include_positive_pairs=force_include_positive_pairs,
    )
    aa_edge_index, aa_edge_attr = build_aa_graph(sample, ar_edge_index)
    ar_pair_edge_index, ar_pair_prot_atom_index = build_residue_atom_pairs(ar_edge_index, sample["prot_res_atom_ptr"])

    out = dict(sample)
    out.update(
        {
            "ar_edge_index": ar_edge_index,
            "ar_edge_label": ar_label,
            "aa_edge_index": aa_edge_index,
            "aa_edge_attr": aa_edge_attr,
            "ar_pair_edge_index": ar_pair_edge_index,
            "ar_pair_prot_atom_index": ar_pair_prot_atom_index,
            "graph_stats": stats,
        }
    )
    return out
