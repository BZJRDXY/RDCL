from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from rdcl.utils import as_tensor, safe_torch_load


class PTReadError(RuntimeError):
    pass


def read_mfd_pt_sample(
    cache_dir: str | Path,
    affinity: float,
    sample_id: str | None = None,
    require_interaction: bool = True,
) -> Dict[str, Any]:
    cache_dir = Path(cache_dir)
    sample_id = sample_id or cache_dir.name
    try:
        g = safe_torch_load(cache_dir / "graph.pt")
        pa = safe_torch_load(cache_dir / "protein_atom_static.pt")
        esm = safe_torch_load(cache_dir / "protein_esm_sidecar.pt")
        inter = safe_torch_load(cache_dir / "interaction_sidecar.pt") if (cache_dir / "interaction_sidecar.pt").exists() else None
        if require_interaction and inter is None:
            raise FileNotFoundError(cache_dir / "interaction_sidecar.pt")
    except Exception as exc:
        raise PTReadError(f"Failed to load PT files for {cache_dir}: {exc}") from exc

    try:
        lig_x = as_tensor(g["lig_atom_feat"], torch.float32)
        lig_pos = as_tensor(g["lig_atom_pos_crystal"], torch.float32)
        lig_bond_edge_index = as_tensor(g["lig_bond_edge_index"], torch.long)
        lig_bond_edge_attr = as_tensor(g["lig_bond_edge_attr"], torch.float32)

        res_x = as_tensor(g["prot_res_feat"], torch.float32)
        res_pos = as_tensor(g["prot_res_pos"], torch.float32)
        res_ids = list(g.get("prot_res_ids", []))

        prot_atom_x = as_tensor(pa["prot_atom_feat"], torch.float32)
        prot_atom_pos = as_tensor(pa["prot_atom_pos"], torch.float32)
        prot_atom_res_index = as_tensor(pa["prot_atom_res_index"], torch.long)
        prot_atom_bond_index = as_tensor(pa["prot_atom_bond_index"], torch.long)
        prot_res_atom_count = as_tensor(pa.get("prot_res_atom_count"), torch.long)
        prot_res_atom_ptr = as_tensor(pa.get("prot_res_atom_ptr"), torch.long)

        esm_x = as_tensor(esm["esm_res_emb"], torch.float32)
        if inter is None:
            pos_pair = torch.zeros((2, 0), dtype=torch.long)
        else:
            pos_pair = as_tensor(inter["pair_index"], torch.long)
    except Exception as exc:
        raise PTReadError(f"Missing or incompatible fields for {cache_dir}: {exc}") from exc

    n_lig = lig_x.shape[0]
    n_res = res_x.shape[0]
    n_pa = prot_atom_x.shape[0]

    if lig_pos.shape != (n_lig, 3):
        raise PTReadError(f"bad lig_pos shape for {sample_id}: {tuple(lig_pos.shape)}")
    if res_pos.shape != (n_res, 3):
        raise PTReadError(f"bad res_pos shape for {sample_id}: {tuple(res_pos.shape)}")
    if prot_atom_pos.shape != (n_pa, 3):
        raise PTReadError(f"bad prot_atom_pos shape for {sample_id}: {tuple(prot_atom_pos.shape)}")
    if esm_x.shape[0] != n_res:
        raise PTReadError(f"ESM/residue length mismatch for {sample_id}: {esm_x.shape[0]} vs {n_res}")
    if prot_atom_res_index.numel() != n_pa:
        raise PTReadError(f"prot_atom_res_index length mismatch for {sample_id}")
    if pos_pair.ndim != 2 or pos_pair.shape[0] != 2:
        raise PTReadError(f"bad interaction pair_index for {sample_id}: {tuple(pos_pair.shape)}")

    return {
        "sample_id": sample_id,
        "cache_dir": str(cache_dir),
        "affinity": float(affinity),
        "lig_x": lig_x,
        "lig_pos": lig_pos,
        "lig_bond_edge_index": lig_bond_edge_index,
        "lig_bond_edge_attr": lig_bond_edge_attr,
        "res_x": res_x,
        "res_pos": res_pos,
        "res_ids": res_ids,
        "prot_atom_x": prot_atom_x,
        "prot_atom_pos": prot_atom_pos,
        "prot_atom_res_index": prot_atom_res_index,
        "prot_atom_bond_index": prot_atom_bond_index,
        "prot_res_atom_count": prot_res_atom_count,
        "prot_res_atom_ptr": prot_res_atom_ptr,
        "esm_x": esm_x,
        "positive_pair_index": pos_pair,
    }
