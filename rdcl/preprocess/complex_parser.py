from __future__ import annotations

from pathlib import Path
from typing import Tuple


WATER_NAMES = {"HOH", "WAT", "H2O"}


def split_complex_pdb(
    complex_path: str | Path,
    ligand_resname: str,
    ligand_chain: str,
    ligand_resseq: str | int,
    out_dir: str | Path,
    ligand_icode: str | None = None,
) -> tuple[Path, Path]:
    """Split a PDB complex into protein.pdb and ligand.pdb by ligand residue id.

    This simple splitter is intentionally strict: users must specify the ligand
    residue name, chain, and residue sequence number to avoid accidentally using
    waters, ions, buffers, or cofactors as the ligand.
    """
    complex_path = Path(complex_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lig_resname = str(ligand_resname).strip().upper()
    lig_chain = str(ligand_chain).strip()
    lig_resseq = int(ligand_resseq)
    lig_icode = "" if ligand_icode is None else str(ligand_icode).strip()

    protein_lines = []
    ligand_lines = []
    with complex_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in {"ATOM", "HETATM"}:
                continue
            resname = line[17:20].strip().upper()
            chain = line[21].strip()
            try:
                resseq = int(line[22:26])
            except Exception:
                continue
            icode = line[26].strip()
            is_lig = (resname == lig_resname and chain == lig_chain and resseq == lig_resseq and (not lig_icode or icode == lig_icode))
            if is_lig:
                ligand_lines.append(line)
            else:
                if rec == "ATOM" and resname not in WATER_NAMES:
                    protein_lines.append(line)
    if not ligand_lines:
        raise ValueError(f"Could not find ligand {lig_resname} chain={lig_chain} resseq={lig_resseq} in {complex_path}")
    if not protein_lines:
        raise ValueError(f"No protein ATOM records left after ligand split for {complex_path}")
    protein_path = out_dir / "protein.pdb"
    ligand_path = out_dir / "ligand.pdb"
    protein_path.write_text("".join(protein_lines) + "END\n", encoding="utf-8")
    ligand_path.write_text("".join(ligand_lines) + "END\n", encoding="utf-8")
    return protein_path, ligand_path
