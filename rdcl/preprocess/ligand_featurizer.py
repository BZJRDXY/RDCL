from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
except Exception:  # pragma: no cover
    Chem = None
    rdMolDescriptors = None

from rdcl.preprocess.feature_schema import LIG_ATOM_DIM, LIG_BOND_DIM

ROTATABLE_BOND_SMARTS = getattr(rdMolDescriptors, "_CalcNumRotatableBonds_smarts", "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]") if rdMolDescriptors is not None else "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]"


@dataclass
class LigandPreprocessInfo:
    path: str | None = None
    n_atoms: int = 0
    n_bonds_directed: int = 0
    feature_schema: str = "multiflowdock_ligand_v1"


def _require_rdkit():
    if Chem is None:
        raise ImportError("RDKit is required for ligand preprocessing. Install rdkit or use predict_from_cache.py.")


def load_ligand_mol(path: str | Path, sanitize: bool = True):
    _require_rdkit()
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".sdf", ".sd"}:
        suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=sanitize)
        mol = suppl[0] if suppl is not None and len(suppl) > 0 else None
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=sanitize)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=sanitize)
    else:
        raise ValueError(f"Unsupported ligand format: {path.suffix}. Supported: .sdf, .mol2, .pdb")
    if mol is None:
        raise ValueError(f"Failed to read ligand: {path}")
    ensure_3d(mol)
    return mol


def ligand_from_pdb_block(block: str, sanitize: bool = True):
    _require_rdkit()
    mol = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=sanitize)
    if mol is None and sanitize:
        mol = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError("Failed to parse ligand PDB block with RDKit")
    ensure_3d(mol)
    return mol


def ensure_3d(mol) -> None:
    if mol is None:
        raise ValueError("Ligand is None")
    if mol.GetNumAtoms() == 0:
        raise ValueError("Ligand has zero atoms")
    if mol.GetNumConformers() == 0:
        raise ValueError("Ligand has no 3D conformer")
    conf = mol.GetConformer()
    xyz = np.asarray([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())], dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("Ligand contains non-finite coordinates")


def _donor_acceptor_flags(mol):
    # Same lightweight SMARTS as MultiFlowDock.  These are cached features, not a
    # full protonation-state model.
    n = mol.GetNumAtoms()
    is_acceptor = np.zeros(n, dtype=np.float32)
    is_donor = np.zeros(n, dtype=np.float32)
    acceptor_smarts = Chem.MolFromSmarts("[!$([#6,F,Cl,Br,I,o,S^0,S^+0,Cl^-,Br^-,I^-]);!$([#7v5,#15v5,#16v4,#16v6]);!$([#7v4,#15v4]);!$([#7v3,#15v3]);!$([#8v3,#16v3]);!$([#8v2,#16v2]);!$([#8v1,#16v1]);!$([#7,#15,#16]);!$([#7,#15,#16]~[#7,#15,#16]);!$([#7,#15,#16]~[#8,#16]);!$([#7,#15,#16]~[#9,#17,#35,#53]);!$([#7,#15,#16]~[#7,#15,#16]~[#7,#15,#16]);!$([#7,#15,#16]~[#7,#15,#16]~[#8,#16]);!$([#7,#15,#16]~[#7,#15,#16]~[#9,#17,#35,#53])]")
    donor_smarts = Chem.MolFromSmarts("[N!H0;+0;!$([N]~[!#6]);!$([N]~[#7,#8,#9,#15,#16,#17,#35,#53]),O!H0,S!H0]")
    if acceptor_smarts is not None:
        for m in mol.GetSubstructMatches(acceptor_smarts):
            is_acceptor[int(m[0])] = 1.0
    if donor_smarts is not None:
        for m in mol.GetSubstructMatches(donor_smarts):
            is_donor[int(m[0])] = 1.0
    return is_donor, is_acceptor


def atom_features(mol) -> np.ndarray:
    _require_rdkit()
    n = mol.GetNumAtoms()
    is_donor, is_acceptor = _donor_acceptor_flags(mol)
    feats = []
    for i, a in enumerate(mol.GetAtoms()):
        Z = int(a.GetAtomicNum())
        z_bins = np.zeros(20, dtype=np.float32)
        z_bins[min(Z, 20) - 1 if 1 <= Z <= 20 else 19] = 1.0
        z_scalar = min(max(Z, 0), 100) / 100.0
        hyb = a.GetHybridization()
        hyb_onehot = np.zeros(4, dtype=np.float32)
        if hyb == Chem.rdchem.HybridizationType.SP:
            hyb_onehot[0] = 1.0
        elif hyb == Chem.rdchem.HybridizationType.SP2:
            hyb_onehot[1] = 1.0
        elif hyb == Chem.rdchem.HybridizationType.SP3:
            hyb_onehot[2] = 1.0
        else:
            hyb_onehot[3] = 1.0
        charge_scalar = max(-5.0, min(5.0, float(a.GetFormalCharge()))) / 5.0
        halogen = 1.0 if Z in (9, 17, 35, 53) else 0.0
        row = np.concatenate([
            z_bins,
            np.asarray([z_scalar, float(a.GetIsAromatic()), float(a.IsInRing()), charge_scalar, is_donor[i], is_acceptor[i], halogen], dtype=np.float32),
            hyb_onehot,
        ])
        feats.append(row)
    out = np.stack(feats, axis=0).astype(np.float32) if feats else np.zeros((0, LIG_ATOM_DIM), dtype=np.float32)
    if out.shape[1] != LIG_ATOM_DIM:
        raise RuntimeError(f"ligand atom feature dim mismatch: {out.shape[1]} != {LIG_ATOM_DIM}")
    return out


def rotatable_bonds(mol) -> list[tuple[int, int]]:
    _require_rdkit()
    patt = Chem.MolFromSmarts(ROTATABLE_BOND_SMARTS)
    matches = mol.GetSubstructMatches(patt) if patt is not None else []
    out, seen = [], set()
    for m in matches:
        a, b = int(m[0]), int(m[1])
        key = tuple(sorted((a, b)))
        if key not in seen:
            seen.add(key)
            out.append((a, b))
    return out


def bond_features(mol) -> tuple[np.ndarray, np.ndarray]:
    _require_rdkit()
    rot = set(tuple(sorted(b)) for b in rotatable_bonds(mol))
    edges, attrs = [], []
    for b in mol.GetBonds():
        i, j = int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())
        bt = b.GetBondType()
        bt_oh = np.zeros(4, dtype=np.float32)
        if bt == Chem.rdchem.BondType.SINGLE:
            bt_oh[0] = 1.0
        elif bt == Chem.rdchem.BondType.DOUBLE:
            bt_oh[1] = 1.0
        elif bt == Chem.rdchem.BondType.TRIPLE:
            bt_oh[2] = 1.0
        elif bt == Chem.rdchem.BondType.AROMATIC:
            bt_oh[3] = 1.0
        a = np.concatenate([bt_oh, np.asarray([float(b.GetIsConjugated()), float(b.IsInRing()), float(tuple(sorted((i, j))) in rot)], dtype=np.float32)])
        edges.append((i, j)); attrs.append(a)
        edges.append((j, i)); attrs.append(a)
    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.stack(attrs, axis=0).astype(np.float32) if attrs else np.zeros((0, LIG_BOND_DIM), dtype=np.float32)
    return edge_index, edge_attr


def get_conformer_coords(mol) -> np.ndarray:
    ensure_3d(mol)
    conf = mol.GetConformer()
    return np.asarray([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())], dtype=np.float32)


def featurize_ligand_mol(mol) -> dict:
    ensure_3d(mol)
    edge_index, edge_attr = bond_features(mol)
    return {
        "lig_atom_feat": torch.from_numpy(atom_features(mol)).float(),
        "lig_atom_pos_crystal": torch.from_numpy(get_conformer_coords(mol)).float(),
        "lig_bond_edge_index": torch.from_numpy(edge_index).long(),
        "lig_bond_edge_attr": torch.from_numpy(edge_attr).float(),
        "ligand_feature_schema": "multiflowdock_ligand_v1",
    }


def featurize_ligand_file(path: str | Path, sanitize: bool = True) -> dict:
    return featurize_ligand_mol(load_ligand_mol(path, sanitize=sanitize))
