from __future__ import annotations

"""RDCL/MultiFlowDock-compatible feature schema.

These schemas mirror the cache tensors used by the RDCL training code.  They
are intentionally kept small and deterministic so raw-structure inference and
strict PT cache construction produce tensors with the same field names,
dimensions, ordering, and feature semantics as the training cache.
"""

LIG_ATOM_DIM = 31
LIG_BOND_DIM = 7
PROT_RES_DIM = 25
PROT_ATOM_DIM = 17
ESM_DIM = 1280

LIG_ATOM_SCHEMA = [
    *(f"Z_bucket_{i}" for i in range(1, 21)),
    "atomic_number_clipped_div_100",
    "is_aromatic",
    "is_in_ring",
    "formal_charge_clipped_div_5",
    "is_hbond_donor_rdkit_smarts",
    "is_hbond_acceptor_rdkit_smarts",
    "is_halogen",
    "hybridization_sp",
    "hybridization_sp2",
    "hybridization_sp3",
    "hybridization_other",
]

LIG_BOND_SCHEMA = [
    "bond_single",
    "bond_double",
    "bond_triple",
    "bond_aromatic",
    "is_conjugated",
    "is_in_ring",
    "is_rotatable_rdkit",
]

AA20 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_LIST = AA20 + ["UNK"]
PROT_RES_SCHEMA = [*(f"res_{aa}" for aa in AA_LIST), "is_hydrophobic", "is_polar", "is_positive", "is_negative"]

PROT_ATOM_ELEMENT_BUCKETS = ["C", "N", "O", "S", "P", "F", "CL", "BR", "I", "SE", "OTHER"]
PROT_ATOM_SCHEMA = [
    *(f"elem_{e}" for e in PROT_ATOM_ELEMENT_BUCKETS),
    "atomic_number_div_30",
    "is_backbone",
    "is_sidechain",
    "is_donor_template",
    "is_acceptor_template",
    "formal_charge_template",
]

assert len(LIG_ATOM_SCHEMA) == LIG_ATOM_DIM
assert len(LIG_BOND_SCHEMA) == LIG_BOND_DIM
assert len(PROT_RES_SCHEMA) == PROT_RES_DIM
assert len(PROT_ATOM_SCHEMA) == PROT_ATOM_DIM
