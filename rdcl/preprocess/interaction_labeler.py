from __future__ import annotations

"""Optional site-label generation for strict training-cache construction.

RDCL training uses PLIP-derived ligand-atom/protein-residue positive pairs.
For open-source reproducibility this module provides two modes:

1. plip: best-effort wrapper around the optional `plip` package.  This is the
   recommended mode for strict training cache construction.
2. contact: deterministic geometric fallback for smoke tests only.  It is not a
   replacement for PLIP labels in publication-quality training.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


def _empty_interaction() -> dict:
    return {
        "pair_index": torch.zeros((2, 0), dtype=torch.long),
        "pair_type_mask": torch.zeros((0, 14), dtype=torch.float32),
        "pair_type_bits": torch.zeros((0,), dtype=torch.long),
        "inst_type_id": torch.zeros((0,), dtype=torch.long),
        "label_source": "empty",
    }


def contact_interactions(lig_pos: torch.Tensor, res_pos: torch.Tensor, cutoff: float = 5.0) -> dict:
    dist = torch.cdist(lig_pos.float(), res_pos.float())
    src, dst = torch.nonzero(dist <= float(cutoff), as_tuple=True)
    if src.numel() == 0:
        return _empty_interaction()
    pair_index = torch.stack([src, dst], dim=0).long()
    # one generic contact channel in a 14-dim mask to remain shape-compatible
    mask = torch.zeros((pair_index.shape[1], 14), dtype=torch.float32)
    mask[:, 0] = 1.0
    return {
        "pair_index": pair_index,
        "pair_type_mask": mask,
        "pair_type_bits": torch.ones((pair_index.shape[1],), dtype=torch.long),
        "inst_type_id": torch.zeros((pair_index.shape[1],), dtype=torch.long),
        "label_source": f"contact_{cutoff:.2f}A_fallback",
    }


def plip_interactions_from_complex_pdb(complex_pdb: str | Path, lig_serial_to_idx: Dict[int, int], residue_key_to_idx: Dict[tuple, int]) -> dict:
    """Extract PLIP atom-residue pairs from a complex PDB.

    This function intentionally uses defensive attribute access because PLIP's
    interaction object attributes vary slightly across versions.  It maps ligand
    atom PDB serial numbers to ligand atom indices and protein residue identity
    (chain, resnr, reschain/resname) to RDCL residue indices.
    """
    try:
        from plip.structure.preparation import PDBComplex  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("PLIP is required for --interaction_mode plip. Install plip or use --interaction_mode contact for smoke tests.") from exc

    mol = PDBComplex()
    mol.load_pdb(str(complex_pdb))
    for ligand in mol.ligands:
        try:
            mol.characterize_complex(ligand)
        except Exception:
            continue

    pairs = set()
    type_masks: Dict[tuple, List[int]] = {}
    interaction_sets = getattr(mol, "interaction_sets", {})
    type_lists = [
        "hydrophobic_contacts", "hbonds_ldon", "hbonds_pdon", "pistacking", "pication_laro", "pication_paro",
        "saltbridge_lneg", "saltbridge_pneg", "halogen_bonds", "water_bridges", "metal_complexes",
    ]
    for site in interaction_sets.values():
        for type_id, name in enumerate(type_lists):
            for inter in getattr(site, name, []) or []:
                lig_serials = []
                for attr in ["ligatom", "ligatom_orig_idx", "ligatom_id", "lig_idx", "ligatom1", "ligatom2"]:
                    v = getattr(inter, attr, None)
                    if isinstance(v, int):
                        lig_serials.append(v)
                    elif isinstance(v, (list, tuple)):
                        lig_serials.extend([int(x) for x in v if isinstance(x, int)])
                if not lig_serials:
                    atom = getattr(inter, "ligatom", None)
                    if hasattr(atom, "idx"):
                        lig_serials.append(int(atom.idx))
                resnr = getattr(inter, "resnr", None) or getattr(inter, "prot_resnr", None)
                restype = getattr(inter, "restype", None) or getattr(inter, "prot_restype", None) or getattr(inter, "resname", None)
                reschain = getattr(inter, "reschain", None) or getattr(inter, "prot_chain", None) or getattr(inter, "chain", None)
                if resnr is None or reschain is None:
                    continue
                try:
                    resnr = int(resnr)
                except Exception:
                    continue
                res_candidates = []
                if restype is not None:
                    res_candidates.append((str(reschain).strip() or "_", int(resnr), "_", str(restype).strip().upper()))
                # fall back to chain/resseq only if residue name/icode cannot be matched
                res_candidates.extend([k for k in residue_key_to_idx if k[0] == (str(reschain).strip() or "_") and k[1] == int(resnr)])
                res_idx = None
                for key in res_candidates:
                    if key in residue_key_to_idx:
                        res_idx = residue_key_to_idx[key]
                        break
                if res_idx is None:
                    continue
                for serial in lig_serials:
                    if int(serial) not in lig_serial_to_idx:
                        continue
                    key = (int(lig_serial_to_idx[int(serial)]), int(res_idx))
                    pairs.add(key)
                    type_masks.setdefault(key, [0] * 14)[min(type_id, 13)] = 1
    if not pairs:
        return _empty_interaction()
    pairs_sorted = sorted(pairs)
    pair_index = torch.tensor(pairs_sorted, dtype=torch.long).t().contiguous()
    mask = torch.tensor([type_masks.get(p, [0] * 14) for p in pairs_sorted], dtype=torch.float32)
    bits = torch.zeros((len(pairs_sorted),), dtype=torch.long)
    for i, row in enumerate(mask.long()):
        val = 0
        for b, x in enumerate(row.tolist()[:14]):
            if x:
                val |= (1 << b)
        bits[i] = val
    return {
        "pair_index": pair_index,
        "pair_type_mask": mask,
        "pair_type_bits": bits,
        "inst_type_id": torch.zeros((len(pairs_sorted),), dtype=torch.long),
        "label_source": "plip_best_effort",
    }
