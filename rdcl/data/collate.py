from __future__ import annotations

from typing import Dict, List

import torch


def _cat(xs, dim=0, dtype=None):
    if not xs:
        return torch.empty(0, dtype=dtype or torch.float32)
    return torch.cat(xs, dim=dim)


def rdcl_collate(samples: List[Dict]) -> Dict:
    if len(samples) == 0:
        raise ValueError("empty batch")

    lig_xs, lig_poss, lig_batches, lig_node_indices = [], [], [], []
    pa_xs, pa_poss, pa_batches, pa_node_indices = [], [], [], []
    res_xs, res_poss, esm_xs, res_batches = [], [], [], []
    aa_edges, aa_attrs = [], []
    ar_edges, ar_labels, ar_batches = [], [], []
    ar_pair_eids, ar_pair_paids = [], []
    prot_atom_res_indices = []
    affinities = []
    sample_ids = []
    graph_stats = []

    lig_off = 0
    pa_off = 0
    res_off = 0
    atom_off = 0
    ar_edge_off = 0

    for b, s in enumerate(samples):
        n_lig = s["lig_x"].shape[0]
        n_pa = s["prot_atom_x"].shape[0]
        n_res = s["res_x"].shape[0]
        n_atom = n_lig + n_pa
        n_ar = s["ar_edge_index"].shape[1]

        lig_xs.append(s["lig_x"])
        lig_poss.append(s["lig_pos"])
        lig_batches.append(torch.full((n_lig,), b, dtype=torch.long))
        lig_node_indices.append(atom_off + torch.arange(n_lig, dtype=torch.long))

        pa_xs.append(s["prot_atom_x"])
        pa_poss.append(s["prot_atom_pos"])
        pa_batches.append(torch.full((n_pa,), b, dtype=torch.long))
        pa_node_indices.append(atom_off + n_lig + torch.arange(n_pa, dtype=torch.long))
        prot_atom_res_indices.append(s["prot_atom_res_index"].long() + res_off)

        res_xs.append(s["res_x"])
        res_poss.append(s["res_pos"])
        esm_xs.append(s["esm_x"])
        res_batches.append(torch.full((n_res,), b, dtype=torch.long))

        aa_edges.append(s["aa_edge_index"].long() + atom_off)
        aa_attrs.append(s["aa_edge_attr"].float())

        ar = s["ar_edge_index"].long().clone()
        ar[0] += lig_off
        ar[1] += res_off
        ar_edges.append(ar)
        ar_labels.append(s["ar_edge_label"].float())
        ar_batches.append(torch.full((n_ar,), b, dtype=torch.long))

        ar_pair_eids.append(s["ar_pair_edge_index"].long() + ar_edge_off)
        ar_pair_paids.append(s["ar_pair_prot_atom_index"].long() + pa_off)

        affinities.append(float(s["affinity"]))
        sample_ids.append(str(s["sample_id"]))
        graph_stats.append(s.get("graph_stats", {}))

        lig_off += n_lig
        pa_off += n_pa
        res_off += n_res
        atom_off += n_atom
        ar_edge_off += n_ar

    batch = {
        "sample_ids": sample_ids,
        "num_samples": len(samples),
        "lig_x": _cat(lig_xs),
        "lig_pos": _cat(lig_poss),
        "lig_batch": _cat(lig_batches, dtype=torch.long),
        "lig_node_index": _cat(lig_node_indices, dtype=torch.long),
        "prot_atom_x": _cat(pa_xs),
        "prot_atom_pos": _cat(pa_poss),
        "prot_atom_batch": _cat(pa_batches, dtype=torch.long),
        "prot_atom_node_index": _cat(pa_node_indices, dtype=torch.long),
        "prot_atom_res_index": _cat(prot_atom_res_indices, dtype=torch.long),
        "res_x": _cat(res_xs),
        "res_pos": _cat(res_poss),
        "esm_x": _cat(esm_xs),
        "res_batch": _cat(res_batches, dtype=torch.long),
        "aa_edge_index": _cat(aa_edges, dim=1, dtype=torch.long),
        "aa_edge_attr": _cat(aa_attrs),
        "ar_edge_index": _cat(ar_edges, dim=1, dtype=torch.long),
        "ar_edge_label": _cat(ar_labels),
        "ar_edge_batch": _cat(ar_batches, dtype=torch.long),
        "ar_pair_edge_index": _cat(ar_pair_eids, dtype=torch.long),
        "ar_pair_prot_atom_index": _cat(ar_pair_paids, dtype=torch.long),
        "affinity": torch.tensor(affinities, dtype=torch.float32),
        "graph_stats": graph_stats,
        "num_atoms_combined": atom_off,
    }
    atom_batch = torch.empty(atom_off, dtype=torch.long)
    for b, s in enumerate(samples):
        # Fill using node index ranges already in mapping lists.
        idx = torch.cat([lig_node_indices[b], pa_node_indices[b]], dim=0)
        atom_batch[idx] = b
    batch["atom_batch"] = atom_batch
    return batch


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out
