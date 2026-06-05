from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from rdcl.preprocess.feature_schema import AA20, AA_LIST, PROT_ATOM_DIM, PROT_RES_DIM, PROT_ATOM_ELEMENT_BUCKETS

AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
HYDROPHOBIC = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "TYR"}
POSITIVE = {"LYS", "ARG", "HIS"}
NEGATIVE = {"ASP", "GLU"}
POLAR = {"SER", "THR", "ASN", "GLN", "CYS", "HIS", "TYR"}
RESNAME_MAP = {
    "MSE": "MET", "SEC": "CYS", "PYL": "LYS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "CYX": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "SEP": "SER", "TPO": "THR",
    "PTR": "TYR", "CSO": "CYS", "PCA": "GLU", "LLP": "LYS",
}
CHI1_FOUR_ATOMS = {
    "ARG": ("N", "CA", "CB", "CG"), "ASN": ("N", "CA", "CB", "CG"), "ASP": ("N", "CA", "CB", "CG"),
    "CYS": ("N", "CA", "CB", "SG"), "GLN": ("N", "CA", "CB", "CG"), "GLU": ("N", "CA", "CB", "CG"),
    "HIS": ("N", "CA", "CB", "CG"), "ILE": ("N", "CA", "CB", "CG1"), "LEU": ("N", "CA", "CB", "CG"),
    "LYS": ("N", "CA", "CB", "CG"), "MET": ("N", "CA", "CB", "CG"), "PHE": ("N", "CA", "CB", "CG"),
    "PRO": ("N", "CA", "CB", "CG"), "SER": ("N", "CA", "CB", "OG"), "THR": ("N", "CA", "CB", "OG1"),
    "TRP": ("N", "CA", "CB", "CG"), "TYR": ("N", "CA", "CB", "CG"), "VAL": ("N", "CA", "CB", "CG1"),
}
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
ELEMENT_TO_IDX = {e: i for i, e in enumerate(PROT_ATOM_ELEMENT_BUCKETS)}
ATOMIC_NUM = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "CL": 17, "BR": 35, "I": 53, "SE": 34, "ZN": 30, "MG": 12, "CA": 20, "NA": 11}
_PROT_ACCEPTOR = {
    ("*", "O"), ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"), ("ASN", "OD1"),
    ("GLN", "OE1"), ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"), ("CYS", "SG"), ("HIS", "ND1"), ("HIS", "NE2"),
}
_PROT_DONOR = {
    ("*", "N"), ("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("ASN", "ND2"),
    ("GLN", "NE2"), ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"), ("CYS", "SG"), ("HIS", "ND1"), ("HIS", "NE2"), ("TRP", "NE1"),
}
_PROT_CHARGE = {("ASP", "OD1"): -0.5, ("ASP", "OD2"): -0.5, ("GLU", "OE1"): -0.5, ("GLU", "OE2"): -0.5, ("LYS", "NZ"): 1.0, ("ARG", "NH1"): 0.5, ("ARG", "NH2"): 0.5}
RES_EXTRA_BONDS = {
    "ALA": [("CA", "CB")],
    "ARG": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")],
    "ASN": [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")],
    "ASP": [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")],
    "CYS": [("CA", "CB"), ("CB", "SG")],
    "GLN": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")],
    "GLU": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")],
    "GLY": [],
    "HIS": [("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("ND1", "CE1"), ("CE1", "NE2"), ("NE2", "CD2"), ("CD2", "CG")],
    "ILE": [("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")],
    "LEU": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
    "LYS": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")],
    "MET": [("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")],
    "PHE": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ")],
    "PRO": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N")],
    "SER": [("CA", "CB"), ("CB", "OG")],
    "THR": [("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")],
    "TRP": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "NE1"), ("NE1", "CE2"), ("CE2", "CD2"), ("CE2", "CZ2"), ("CZ2", "CH2"), ("CH2", "CZ3"), ("CZ3", "CE3"), ("CE3", "CD2")],
    "TYR": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "OH")],
    "VAL": [("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")],
    "UNK": [],
}


def map_resname(resname_raw: str) -> tuple[str, bool]:
    raw = (resname_raw or "").strip().upper()
    mapped = RESNAME_MAP.get(raw, raw)
    if mapped in AA20:
        return mapped, mapped != raw
    return "UNK", raw != "UNK"


def residue_features(resname: str) -> np.ndarray:
    x = np.zeros((PROT_RES_DIM,), dtype=np.float32)
    res = (resname or "UNK").upper()
    x[AA_TO_IDX.get(res, AA_TO_IDX["UNK"])] = 1.0
    x[21] = 1.0 if res in HYDROPHOBIC else 0.0
    x[22] = 1.0 if res in POLAR else 0.0
    x[23] = 1.0 if res in POSITIVE else 0.0
    x[24] = 1.0 if res in NEGATIVE else 0.0
    return x


def _norm_element(e: str) -> str:
    e = (e or "").strip().upper()
    if not e:
        return "X"
    if len(e) == 2:
        return e[0] + e[1]
    return e


def _element_from_pdb(line: str) -> str:
    if len(line) >= 78:
        elem = line[76:78].strip()
        if elem:
            return _norm_element(elem)
    atom = line[12:16].strip()
    a = "".join(c for c in atom if c.isalpha()).upper()
    if len(a) >= 2 and a[:2] in {"CL", "BR", "NA", "CA", "MG", "ZN", "SE"}:
        return a[:2]
    return a[0] if a else "X"


def _build_atom_feat(resname: str, atom_name: str, element: str, is_backbone: bool, is_sidechain: bool):
    elem = _norm_element(element)
    x = np.zeros((PROT_ATOM_DIM,), dtype=np.float32)
    x[ELEMENT_TO_IDX.get(elem, ELEMENT_TO_IDX["OTHER"])] = 1.0
    z = int(ATOMIC_NUM.get(elem, 0))
    donor = 1 if ((resname, atom_name) in _PROT_DONOR or ("*", atom_name) in _PROT_DONOR) else 0
    acceptor = 1 if ((resname, atom_name) in _PROT_ACCEPTOR or ("*", atom_name) in _PROT_ACCEPTOR) else 0
    charge = float(_PROT_CHARGE.get((resname, atom_name), 0.0))
    base = len(PROT_ATOM_ELEMENT_BUCKETS)
    x[base + 0] = float(z) / 30.0
    x[base + 1] = float(is_backbone)
    x[base + 2] = float(is_sidechain)
    x[base + 3] = float(donor)
    x[base + 4] = float(acceptor)
    x[base + 5] = float(charge)
    return x


@dataclass
class AtomRecord:
    atom_name: str
    element: str
    serial: int
    altloc: str
    occupancy: float
    pos: np.ndarray
    res_index: int
    resname: str
    resname_raw: str
    chain: str
    resseq: int
    icode: str


@dataclass
class ResidueRecord:
    chain: str
    resseq: int
    icode: str
    resname: str
    resname_raw: str
    is_modified: bool
    ca: np.ndarray
    centroid: np.ndarray
    has_ca: bool
    bb_complete: bool
    chi1_valid: bool
    atoms: dict[str, tuple[np.ndarray, str, int, float, str]]


def _parse_residues(pdb_path: str | Path) -> list[ResidueRecord]:
    residues: dict[tuple[str, int, str], dict] = {}
    in_model = False
    for line in Path(pdb_path).read_text(errors="ignore").splitlines():
        if line.startswith("MODEL"):
            in_model = True
            continue
        if in_model and line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM") or len(line) < 54:
            continue
        atom = line[12:16].strip()
        altloc = line[16].strip()
        resname_raw = line[17:20].strip().upper()
        chain = (line[21] or " ").strip() or "_"
        try:
            resseq = int(line[22:26])
        except Exception:
            continue
        icode = line[26].strip() or "_"
        try:
            xyz = np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32)
        except Exception:
            continue
        occ = 0.0
        if len(line) >= 60:
            try:
                occ = float(line[54:60])
            except Exception:
                occ = 0.0
        serial = 0
        try:
            serial = int(line[6:11])
        except Exception:
            pass
        element = _element_from_pdb(line)
        key = (chain, resseq, icode)
        residues.setdefault(key, {"atoms": {}, "resname_raw": resname_raw})
        prio = 2 if altloc in ("", "A") else 1
        prev = residues[key]["atoms"].get(atom)
        cur = (prio, occ, xyz, element, serial, altloc)
        if prev is None or cur[0] > prev[0] or (cur[0] == prev[0] and cur[1] > prev[1]):
            residues[key]["atoms"][atom] = cur
    out = []
    for (chain, resseq, icode), v in residues.items():
        atoms = {name: (rec[2], rec[3], rec[4], rec[1], rec[5]) for name, rec in v["atoms"].items()}
        if not atoms:
            continue
        coords = np.stack([a[0] for a in atoms.values()], axis=0)
        centroid = coords.mean(axis=0).astype(np.float32)
        ca = atoms.get("CA", (centroid, "", 0, 0.0, ""))[0].astype(np.float32)
        present = set(atoms)
        resname_raw = str(v.get("resname_raw", "UNK"))
        resname, is_modified = map_resname(resname_raw)
        chi1_atoms = CHI1_FOUR_ATOMS.get(resname)
        chi1_valid = bool(chi1_atoms and all(a in present for a in chi1_atoms))
        out.append(ResidueRecord(chain, int(resseq), str(icode), resname, resname_raw, is_modified, ca, centroid, "CA" in present, all(a in present for a in ["N", "CA", "C", "O"]), chi1_valid, atoms))
    out.sort(key=lambda r: (r.chain, r.resseq, r.icode))
    return out


def _build_protein_bonds(residues: Sequence[ResidueRecord], atom_records: Sequence[AtomRecord]) -> torch.Tensor:
    name_to_idx = {(a.res_index, a.atom_name): i for i, a in enumerate(atom_records)}
    edges = set()
    for ri, r in enumerate(residues):
        # backbone within residue
        for a, b in [("N", "CA"), ("CA", "C"), ("C", "O"), ("C", "OXT")]:
            if (ri, a) in name_to_idx and (ri, b) in name_to_idx:
                u, v = name_to_idx[(ri, a)], name_to_idx[(ri, b)]
                edges.add((u, v)); edges.add((v, u))
        for a, b in RES_EXTRA_BONDS.get(r.resname, []):
            if (ri, a) in name_to_idx and (ri, b) in name_to_idx:
                u, v = name_to_idx[(ri, a)], name_to_idx[(ri, b)]
                edges.add((u, v)); edges.add((v, u))
    for ri in range(len(residues) - 1):
        if residues[ri].chain != residues[ri + 1].chain:
            continue
        if (ri, "C") in name_to_idx and (ri + 1, "N") in name_to_idx:
            u, v = name_to_idx[(ri, "C")], name_to_idx[(ri + 1, "N")]
            d = float(np.linalg.norm(atom_records[u].pos - atom_records[v].pos))
            if d <= 1.9:
                edges.add((u, v)); edges.add((v, u))
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()


def featurize_protein_pdb(path: str | Path) -> dict:
    residues = _parse_residues(path)
    if not residues:
        raise ValueError(f"No protein ATOM residues found in {path}")
    res_ids, res_feats, res_pos = [], [], []
    atom_records: list[AtomRecord] = []
    atom_feats, atom_pos, atom_res_index = [], [], []
    for ri, r in enumerate(residues):
        res_ids.append(f"{r.chain}:{r.resseq}:{r.icode}:{r.resname}")
        res_feats.append(residue_features(r.resname))
        res_pos.append(r.ca)
        for atom_name in sorted(r.atoms.keys()):
            pos, element, serial, occ, altloc = r.atoms[atom_name]
            if _norm_element(element) == "H":
                continue
            is_bb = atom_name in BACKBONE_ATOMS
            is_sc = not is_bb
            atom_records.append(AtomRecord(atom_name, element, int(serial), str(altloc), float(occ), pos.astype(np.float32), ri, r.resname, r.resname_raw, r.chain, r.resseq, r.icode))
            atom_feats.append(_build_atom_feat(r.resname, atom_name, element, is_bb, is_sc))
            atom_pos.append(pos.astype(np.float32))
            atom_res_index.append(ri)
    counts = np.zeros((len(residues),), dtype=np.int64)
    for ri in atom_res_index:
        counts[int(ri)] += 1
    ptr = np.zeros((len(counts) + 1,), dtype=np.int64)
    ptr[1:] = np.cumsum(counts)
    prot_atom_bond_index = _build_protein_bonds(residues, atom_records)
    return {
        "prot_res_feat": torch.from_numpy(np.stack(res_feats, axis=0).astype(np.float32)).float(),
        "prot_res_pos": torch.from_numpy(np.stack(res_pos, axis=0).astype(np.float32)).float(),
        "prot_res_ids": res_ids,
        "prot_res_chain": [r.chain for r in residues],
        "prot_res_resseq": torch.tensor([r.resseq for r in residues], dtype=torch.int32),
        "prot_res_icode": [r.icode for r in residues],
        "prot_res_resname": [r.resname for r in residues],
        "prot_res_resname_raw": [r.resname_raw for r in residues],
        "prot_res_is_modified": torch.tensor([r.is_modified for r in residues], dtype=torch.uint8),
        "prot_res_has_ca": torch.tensor([r.has_ca for r in residues], dtype=torch.uint8),
        "prot_res_bb_complete": torch.tensor([r.bb_complete for r in residues], dtype=torch.uint8),
        "prot_res_chi1_valid": torch.tensor([r.chi1_valid for r in residues], dtype=torch.uint8),
        "prot_atom_feat": torch.tensor(np.stack(atom_feats, axis=0).astype(np.float32), dtype=torch.float32),
        "prot_atom_pos": torch.tensor(np.stack(atom_pos, axis=0).astype(np.float32), dtype=torch.float32),
        "prot_atom_res_index": torch.tensor(atom_res_index, dtype=torch.long),
        "prot_atom_bond_index": prot_atom_bond_index.long(),
        "prot_res_atom_count": torch.from_numpy(counts).long(),
        "prot_res_atom_ptr": torch.from_numpy(ptr).long(),
        "protein_feature_schema": "multiflowdock_protein_v1",
    }
