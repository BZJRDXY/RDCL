from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from rdcl.data.cache_manifest import read_manifest
from rdcl.data.graph_builder import build_rdcl_graph
from rdcl.data.pt_reader import read_mfd_pt_sample


class RDCLDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        ar_cutoff: float = 7.0,
        force_include_positive_pairs: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        rows = read_manifest(manifest_csv)
        rows = [r for r in rows if int(r.get("usable_for_base", 1)) == 1]
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        self.rows: List[dict] = rows
        self.ar_cutoff = float(ar_cutoff)
        self.force_include_positive_pairs = bool(force_include_positive_pairs)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        sample = read_mfd_pt_sample(r["cache_dir"], affinity=float(r["affinity"]), sample_id=r["sample_id"])
        sample = build_rdcl_graph(
            sample,
            ar_cutoff=self.ar_cutoff,
            force_include_positive_pairs=self.force_include_positive_pairs,
        )
        return sample

    def affinities(self) -> list[float]:
        return [float(r["affinity"]) for r in self.rows]
