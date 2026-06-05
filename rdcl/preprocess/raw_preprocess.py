from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

import torch

from rdcl.preprocess.complex_parser import split_complex_pdb
from rdcl.preprocess.esm_embedder import ESMEmbedder
from rdcl.preprocess.interaction_labeler import contact_interactions, plip_interactions_from_complex_pdb
from rdcl.preprocess.ligand_featurizer import featurize_ligand_file
from rdcl.preprocess.protein_featurizer import featurize_protein_pdb


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_combined_complex_pdb(protein_path: str | Path, ligand_path: str | Path, out_path: str | Path) -> dict[int, int]:
    """Write a temporary complex PDB for PLIP and return ligand serial -> ligand atom idx.

    For SDF/MOL2/PDB ligands we rely on RDKit to write a PDB block.  RDKit PDB
    atom serials are normally 1..N in atom order, which gives a stable mapping.
    """
    from rdcl.preprocess.ligand_featurizer import load_ligand_mol
    try:
        from rdkit import Chem
    except Exception as exc:  # pragma: no cover
        raise ImportError("RDKit is required to write complex PDB for PLIP") from exc

    protein_lines = []
    for line in Path(protein_path).read_text(errors="ignore").splitlines():
        if line.startswith("ATOM"):
            protein_lines.append(line.rstrip("\n"))
    mol = load_ligand_mol(ligand_path, sanitize=True)
    block = Chem.MolToPDBBlock(mol)
    lig_lines = []
    serial_to_idx = {}
    next_serial = 900000
    for idx, line in enumerate(block.splitlines()):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        serial = next_serial + idx
        # Force HETATM, residue name LIG, chain Z, residue 9999, and unique serial.
        newline = "HETATM" + f"{serial:5d}" + line[11:17] + "LIG Z9999" + line[27:]
        lig_lines.append(newline[:80])
        serial_to_idx[serial] = idx
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(protein_lines + lig_lines + ["END"]) + "\n", encoding="utf-8")
    return serial_to_idx


def _residue_key_to_idx(prot: dict) -> dict[tuple, int]:
    keys = {}
    chains = prot.get("prot_res_chain", [])
    resseq = prot.get("prot_res_resseq", [])
    icodes = prot.get("prot_res_icode", [])
    names = prot.get("prot_res_resname", [])
    if torch.is_tensor(resseq):
        resseq = resseq.detach().cpu().tolist()
    for i, (c, r, ic, n) in enumerate(zip(chains, resseq, icodes, names)):
        keys[(str(c).strip() or "_", int(r), str(ic).strip() or "_", str(n).strip().upper())] = i
    return keys


def build_rdcl_cache_from_files(
    sample_id: str,
    protein_path: str | Path,
    ligand_path: str | Path,
    out_root: str | Path,
    esm_embedder: ESMEmbedder,
    interaction_mode: str = "none",
    contact_cutoff: float = 5.0,
) -> Path:
    """Build an RDCL-compatible cache directory from raw separated files.

    The tensor field names, dimensions, and feature semantics are aligned to the
    MultiFlowDock-style cache used for RDCL training.  `interaction_mode` controls
    whether an interaction_sidecar.pt is generated:
      - none: inference cache only
      - contact: geometric fallback labels for smoke tests
      - plip: PLIP-derived labels; recommended for strict training cache
    """
    out_dir = Path(out_root) / str(sample_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    lig = featurize_ligand_file(ligand_path)
    prot = featurize_protein_pdb(protein_path)
    esm_x = esm_embedder.embed(prot["prot_res_ids"])

    graph = {
        "lig_atom_feat": lig["lig_atom_feat"],
        "lig_atom_pos_crystal": lig["lig_atom_pos_crystal"],
        "lig_bond_edge_index": lig["lig_bond_edge_index"],
        "lig_bond_edge_attr": lig["lig_bond_edge_attr"],
        "prot_res_feat": prot["prot_res_feat"],
        "prot_res_pos": prot["prot_res_pos"],
        "prot_res_ids": prot["prot_res_ids"],
        "prot_res_chain": prot["prot_res_chain"],
        "prot_res_resseq": prot["prot_res_resseq"],
        "prot_res_icode": prot["prot_res_icode"],
        "prot_res_resname": prot["prot_res_resname"],
        "prot_res_resname_raw": prot["prot_res_resname_raw"],
        "prot_res_is_modified": prot["prot_res_is_modified"],
        "prot_res_has_ca": prot["prot_res_has_ca"],
        "prot_res_bb_complete": prot["prot_res_bb_complete"],
        "prot_res_chi1_valid": prot["prot_res_chi1_valid"],
        "feature_schema": "rdcl_multiflowdock_compatible_v1",
    }
    pa = {
        "prot_atom_feat": prot["prot_atom_feat"],
        "prot_atom_pos": prot["prot_atom_pos"],
        "prot_atom_res_index": prot["prot_atom_res_index"],
        "prot_atom_bond_index": prot["prot_atom_bond_index"],
        "prot_res_atom_count": prot["prot_res_atom_count"],
        "prot_res_atom_ptr": prot["prot_res_atom_ptr"],
        "feature_schema": "rdcl_multiflowdock_compatible_v1",
    }
    esm = {"esm_res_emb": esm_x, "prot_res_ids": prot["prot_res_ids"], "esm_dim": int(esm_x.shape[1]), "feature_schema": "esm2_t33_650M_UR50D_layer33_or_compatible"}

    torch.save(graph, out_dir / "graph.pt")
    torch.save(pa, out_dir / "protein_atom_static.pt")
    torch.save(esm, out_dir / "protein_esm_sidecar.pt")

    interaction_mode = str(interaction_mode or "none").lower()
    if interaction_mode not in {"none", "contact", "plip"}:
        raise ValueError("interaction_mode must be one of: none, contact, plip")
    if interaction_mode == "contact":
        inter = contact_interactions(lig["lig_atom_pos_crystal"], prot["prot_res_pos"], cutoff=contact_cutoff)
        torch.save(inter, out_dir / "interaction_sidecar.pt")
    elif interaction_mode == "plip":
        tmp_pdb = out_dir / "_plip_complex.pdb"
        serial_map = _write_combined_complex_pdb(protein_path, ligand_path, tmp_pdb)
        inter = plip_interactions_from_complex_pdb(tmp_pdb, serial_map, _residue_key_to_idx(prot))
        torch.save(inter, out_dir / "interaction_sidecar.pt")

    _save_json({"sample_id": sample_id, "protein_path": str(protein_path), "ligand_path": str(ligand_path), "interaction_mode": interaction_mode, "feature_schema": "rdcl_multiflowdock_compatible_v1"}, out_dir / "meta.json")
    _save_json({"ok": True, "n_lig_atoms": int(lig["lig_atom_feat"].shape[0]), "n_residues": int(prot["prot_res_feat"].shape[0]), "n_protein_atoms": int(prot["prot_atom_feat"].shape[0])}, out_dir / "qc.json")
    return out_dir


def build_inference_cache_from_files(sample_id: str, protein_path: str | Path, ligand_path: str | Path, out_root: str | Path, esm_embedder: ESMEmbedder) -> Path:
    return build_rdcl_cache_from_files(sample_id, protein_path, ligand_path, out_root, esm_embedder, interaction_mode="none")


def build_inference_cache_from_complex(sample_id: str, complex_path: str | Path, ligand_resname: str, ligand_chain: str, ligand_resseq: str | int, out_root: str | Path, esm_embedder: ESMEmbedder, ligand_icode: str | None = None) -> Path:
    tmp_dir = Path(out_root) / str(sample_id) / "_split"
    protein_path, ligand_path = split_complex_pdb(complex_path, ligand_resname, ligand_chain, ligand_resseq, tmp_dir, ligand_icode)
    return build_inference_cache_from_files(sample_id, protein_path, ligand_path, out_root, esm_embedder)


def build_rdcl_cache_from_complex(sample_id: str, complex_path: str | Path, ligand_resname: str, ligand_chain: str, ligand_resseq: str | int, out_root: str | Path, esm_embedder: ESMEmbedder, ligand_icode: str | None = None, interaction_mode: str = "none", contact_cutoff: float = 5.0) -> Path:
    tmp_dir = Path(out_root) / str(sample_id) / "_split"
    protein_path, ligand_path = split_complex_pdb(complex_path, ligand_resname, ligand_chain, ligand_resseq, tmp_dir, ligand_icode)
    return build_rdcl_cache_from_files(sample_id, protein_path, ligand_path, out_root, esm_embedder, interaction_mode=interaction_mode, contact_cutoff=contact_cutoff)
