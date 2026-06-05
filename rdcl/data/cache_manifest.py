from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from rdcl.data.affinity_index import load_affinity_index
from rdcl.utils import safe_torch_load


REQUIRED_FILES = [
    "graph.pt",
    "protein_atom_static.pt",
    "protein_esm_sidecar.pt",
    "interaction_sidecar.pt",
]


@dataclass
class ManifestRow:
    sample_id: str
    pdb_id: str
    source_name: str
    cache_dir: str
    affinity: float
    raw_affinity: str
    n_positive_pairs: int
    has_graph: int
    has_protein_atom_static: int
    has_esm: int
    has_interaction: int
    usable_for_base: int
    usable_for_affinity_only: int


def scan_cache_roots(
    cache_roots: Sequence[str | Path],
    source_names: Sequence[str],
    affinity_index_path: str | Path,
    drop_zero_pos: bool = True,
    deduplicate_by_pdb_id: bool = True,
    dedup_priority: Sequence[str] | None = None,
) -> List[ManifestRow]:
    if len(cache_roots) != len(source_names):
        raise ValueError("cache_roots and source_names must have the same length")
    affinity = load_affinity_index(affinity_index_path)
    priority = {name: i for i, name in enumerate(dedup_priority or source_names)}

    rows: List[ManifestRow] = []
    for root, source in zip(cache_roots, source_names):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Missing cache root: {root}")
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            sample_id = d.name
            pdb_id = sample_id.lower()
            has = {fn: int((d / fn).exists()) for fn in REQUIRED_FILES}
            has_all = all(has.values())
            if pdb_id not in affinity:
                continue
            n_pos = -1
            if has["interaction_sidecar.pt"]:
                try:
                    inter = safe_torch_load(d / "interaction_sidecar.pt")
                    pair_index = inter.get("pair_index", None)
                    n_pos = int(pair_index.shape[1]) if pair_index is not None and len(pair_index.shape) == 2 else 0
                except Exception:
                    n_pos = -1
            usable_base = int(has_all and n_pos > 0)
            usable_aff = int(has_all)
            if drop_zero_pos and not usable_base:
                continue
            rec = affinity[pdb_id]
            rows.append(
                ManifestRow(
                    sample_id=sample_id,
                    pdb_id=pdb_id,
                    source_name=source,
                    cache_dir=str(d),
                    affinity=rec.affinity,
                    raw_affinity=rec.raw_value,
                    n_positive_pairs=max(n_pos, 0),
                    has_graph=has["graph.pt"],
                    has_protein_atom_static=has["protein_atom_static.pt"],
                    has_esm=has["protein_esm_sidecar.pt"],
                    has_interaction=has["interaction_sidecar.pt"],
                    usable_for_base=usable_base,
                    usable_for_affinity_only=usable_aff,
                )
            )

    if deduplicate_by_pdb_id:
        best: Dict[str, ManifestRow] = {}
        for row in rows:
            old = best.get(row.pdb_id)
            if old is None:
                best[row.pdb_id] = row
                continue
            if priority.get(row.source_name, 9999) < priority.get(old.source_name, 9999):
                best[row.pdb_id] = row
        rows = list(best.values())

    rows.sort(key=lambda r: (r.source_name, r.pdb_id))
    return rows


def write_manifest(rows: Iterable[ManifestRow], out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(asdict(rows[0]).keys()) if rows else [f.name for f in ManifestRow.__dataclass_fields__.values()]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_manifest(path: str | Path) -> List[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["affinity"] = float(r["affinity"])
        r["n_positive_pairs"] = int(r.get("n_positive_pairs", 0))
        for key in ["has_graph", "has_protein_atom_static", "has_esm", "has_interaction", "usable_for_base", "usable_for_affinity_only"]:
            if key in r:
                r[key] = int(r[key])
    return rows
