from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch


def _as_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


class PredictionWriter:
    """Write RDCL prediction tables.

    The writer creates four CSV files:
      - predictions.csv: one line per complex, with predicted affinity.
      - site_edges.csv: atom-residue edge scores and ranks.
      - residue_scores.csv: residue-level aggregated site scores.
      - atom_scores.csv: ligand-atom rationale scores.
    """

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pred_f = (self.out_dir / "predictions.csv").open("w", newline="", encoding="utf-8")
        self.edge_f = (self.out_dir / "site_edges.csv").open("w", newline="", encoding="utf-8")
        self.res_f = (self.out_dir / "residue_scores.csv").open("w", newline="", encoding="utf-8")
        self.atom_f = (self.out_dir / "atom_scores.csv").open("w", newline="", encoding="utf-8")
        self.pred_w = csv.DictWriter(self.pred_f, fieldnames=["sample_id", "pred_affinity", "n_edges", "n_lig_atoms", "n_residues"])
        self.edge_w = csv.DictWriter(
            self.edge_f,
            fieldnames=[
                "sample_id",
                "rank",
                "lig_atom_idx",
                "res_idx",
                "res_id",
                "score",
                "site_logit",
                "is_top_p",
            ],
        )
        self.res_w = csv.DictWriter(
            self.res_f,
            fieldnames=["sample_id", "rank", "res_idx", "res_id", "max_score", "mean_score", "n_edges", "is_top_p_residue"],
        )
        self.atom_w = csv.DictWriter(
            self.atom_f,
            fieldnames=["sample_id", "rank", "lig_atom_idx", "atom_score", "is_top_p_atom"],
        )
        for w in [self.pred_w, self.edge_w, self.res_w, self.atom_w]:
            w.writeheader()

    def close(self) -> None:
        for f in [self.pred_f, self.edge_f, self.res_f, self.atom_f]:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def write_sample(
        self,
        sample_id: str,
        sample: Dict,
        out: Dict[str, torch.Tensor],
        pred_affinity: float,
        top_p: float = 0.15,
        max_edges: int | None = None,
    ) -> None:
        ar = sample["ar_edge_index"].detach().cpu().long()
        scores = out["site_scores"].detach().cpu().float()
        logits = out["site_logits"].detach().cpu().float()
        p_atom = out.get("p_atom", None)
        if p_atom is not None:
            p_atom = p_atom.detach().cpu().float()

        n_edges = int(ar.shape[1])
        n_lig = int(sample["lig_x"].shape[0])
        n_res = int(sample["res_x"].shape[0])
        res_ids = list(sample.get("res_ids", []))
        if len(res_ids) < n_res:
            res_ids = [str(i) for i in range(n_res)]

        self.pred_w.writerow(
            {
                "sample_id": sample_id,
                "pred_affinity": f"{pred_affinity:.6f}",
                "n_edges": n_edges,
                "n_lig_atoms": n_lig,
                "n_residues": n_res,
            }
        )

        if n_edges > 0:
            order = torch.argsort(scores, descending=True)
            n_top = max(1, int(round(float(top_p) * n_edges)))
            top_set = set(order[:n_top].tolist())
            if max_edges is not None:
                order_to_write = order[: int(max_edges)]
            else:
                order_to_write = order
            for rank, e in enumerate(order_to_write.tolist(), start=1):
                lig_i = int(ar[0, e])
                res_i = int(ar[1, e])
                self.edge_w.writerow(
                    {
                        "sample_id": sample_id,
                        "rank": rank,
                        "lig_atom_idx": lig_i,
                        "res_idx": res_i,
                        "res_id": res_ids[res_i] if 0 <= res_i < len(res_ids) else str(res_i),
                        "score": f"{float(scores[e]):.8f}",
                        "site_logit": f"{float(logits[e]):.8f}",
                        "is_top_p": int(e in top_set),
                    }
                )

            # Residue aggregation.
            residue_rows = []
            for r in range(n_res):
                mask = ar[1] == r
                if not bool(mask.any()):
                    continue
                s = scores[mask]
                residue_rows.append((r, float(s.max()), float(s.mean()), int(mask.sum().item())))
            residue_rows.sort(key=lambda x: x[1], reverse=True)
            n_top_res = max(1, int(round(float(top_p) * max(len(residue_rows), 1))))
            for rank, (r, mx, mean, cnt) in enumerate(residue_rows, start=1):
                self.res_w.writerow(
                    {
                        "sample_id": sample_id,
                        "rank": rank,
                        "res_idx": r,
                        "res_id": res_ids[r] if 0 <= r < len(res_ids) else str(r),
                        "max_score": f"{mx:.8f}",
                        "mean_score": f"{mean:.8f}",
                        "n_edges": cnt,
                        "is_top_p_residue": int(rank <= n_top_res),
                    }
                )

        if p_atom is not None and n_lig > 0:
            lig_scores = p_atom[:n_lig]
            atom_order = torch.argsort(lig_scores, descending=True)
            n_top_atom = max(1, int(round(float(top_p) * n_lig)))
            for rank, a in enumerate(atom_order.tolist(), start=1):
                self.atom_w.writerow(
                    {
                        "sample_id": sample_id,
                        "rank": rank,
                        "lig_atom_idx": a,
                        "atom_score": f"{float(lig_scores[a]):.8f}",
                        "is_top_p_atom": int(rank <= n_top_atom),
                    }
                )
