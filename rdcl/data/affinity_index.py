from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class AffinityRecord:
    pdb_id: str
    resolution: Optional[float]
    year: Optional[int]
    affinity: float
    raw_value: str
    ligand_name: str = ""


def _to_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _to_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def load_affinity_index(path: str | Path) -> Dict[str, AffinityRecord]:
    """Parse PDBbind INDEX_general_PL_data.2020-like files.

    Expected useful columns:
    PDB code, resolution, release year, -logKd/Ki, raw Kd/Ki/IC50, reference, ligand name.
    """
    path = Path(path)
    records: Dict[str, AffinityRecord] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            pdb_id = parts[0].lower()
            aff = _to_float(parts[3])
            if aff is None:
                continue
            res = _to_float(parts[1]) if len(parts) > 1 else None
            year = _to_int(parts[2]) if len(parts) > 2 else None
            raw = parts[4] if len(parts) > 4 else ""
            ligand_name = parts[-1] if len(parts) > 0 else ""
            records[pdb_id] = AffinityRecord(
                pdb_id=pdb_id,
                resolution=res,
                year=year,
                affinity=float(aff),
                raw_value=raw,
                ligand_name=ligand_name,
            )
    return records
