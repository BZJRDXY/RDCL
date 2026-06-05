from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def residue_name_from_res_id(rid: str) -> str:
    """Extract residue name from RDCL/MultiFlowDock residue ids.

    Supported ids:
      - chain:resseq:icode:resname  (training/MultiFlowDock schema)
      - chain:resname:resseq        (legacy lightweight schema)
      - resname
    """
    parts = str(rid).split(":")
    if len(parts) >= 4:
        return parts[3]
    if len(parts) >= 3:
        return parts[1]
    return parts[0]


def sequence_from_res_ids(res_ids: List[str]) -> str:
    seq = []
    for rid in res_ids:
        resname = residue_name_from_res_id(str(rid))
        seq.append(AA3_TO_1.get(resname.upper(), "X"))
    return "".join(seq)


class ESMEmbedder:
    """Optional ESM-2 residue embedder.

    The implementation uses the `esm` package when available. If no model path is
    supplied and `allow_zero_esm=True`, zero embeddings are returned. Zero ESM is
    useful for smoke tests, but it is not recommended for final prediction because
    it differs from the training distribution.
    """

    def __init__(self, model_path: str | None = None, device: str = "cuda", layer: int = 33, allow_zero_esm: bool = False, esm_dim: int = 1280):
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.layer = int(layer)
        self.allow_zero_esm = bool(allow_zero_esm)
        self.esm_dim = int(esm_dim)
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        if model_path:
            try:
                import esm  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ImportError("The `esm` package is required when --esm_model_path is provided. Install fair-esm.") from exc
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(str(model_path))
            self.model.eval().to(self.device)
            self.batch_converter = self.alphabet.get_batch_converter()
        elif not allow_zero_esm:
            raise ValueError("ESM embeddings are required. Provide --esm_model_path or pass --allow_zero_esm for a zero-vector fallback.")

    @torch.no_grad()
    def embed(self, res_ids: List[str]) -> torch.Tensor:
        if self.model is None:
            return torch.zeros((len(res_ids), self.esm_dim), dtype=torch.float32)
        seq = sequence_from_res_ids(res_ids)
        data = [("protein", seq)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        out = self.model(tokens, repr_layers=[self.layer], return_contacts=False)
        reps = out["representations"][self.layer][0, 1 : len(seq) + 1].detach().cpu().float()
        if reps.shape[0] != len(res_ids):
            raise RuntimeError(f"ESM length mismatch: {reps.shape[0]} vs {len(res_ids)}")
        return reps
