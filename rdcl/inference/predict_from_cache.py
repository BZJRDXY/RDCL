from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from rdcl.data.collate import move_batch_to_device, rdcl_collate
from rdcl.data.graph_builder import build_rdcl_graph
from rdcl.data.pt_reader import read_mfd_pt_sample
from rdcl.inference.output_writer import PredictionWriter
from rdcl.models.rdcl_model import RDCLBaseModel
from rdcl.utils import load_config, safe_torch_load


def read_cache_csv(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty input CSV: {path}")
    for i, r in enumerate(rows, start=1):
        if "cache_dir" not in r:
            raise ValueError("input CSV must contain a cache_dir column")
        if not r.get("sample_id"):
            r["sample_id"] = Path(r["cache_dir"]).name
    return rows


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}


def load_model(config_path: str, checkpoint_path: str, device: torch.device, use_checkpoint_config: bool = False) -> tuple[RDCLBaseModel, Dict, Dict]:
    """Load RDCL from any supported checkpoint format.

    Supported checkpoint formats:
      1. full training checkpoint: {model, optimizer, scaler, config, extra, ...}
      2. inference checkpoint: {model, config, extra}
      3. pure state_dict: OrderedDict[str, Tensor]

    For pure state_dict checkpoints, config must be supplied via --config.
    Affinity unstandardization statistics are read from checkpoint['extra'] when
    available, otherwise from config['inference'].
    """
    cfg = load_config(config_path)
    ckpt = safe_torch_load(checkpoint_path)

    if use_checkpoint_config and isinstance(ckpt, dict) and ckpt.get("config") is not None:
        cfg = ckpt["config"]

    model = RDCLBaseModel(cfg).to(device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        extra = ckpt.get("extra", {}) or {}
    elif isinstance(ckpt, dict):
        # Pure state_dict.
        state = ckpt
        extra = {}
    else:
        state = ckpt
        extra = {}

    if isinstance(state, dict):
        state = _strip_module_prefix(state)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg, extra


def infer_one(model: RDCLBaseModel, sample: Dict, device: torch.device, affinity_mean: float, affinity_std: float, standardize: bool) -> tuple[Dict, float]:
    batch = rdcl_collate([sample])
    batch = move_batch_to_device(batch, device)
    with torch.no_grad():
        out = model(batch)
    pred = out["affinity_pred"].detach().float().cpu()
    if standardize:
        pred = pred * float(affinity_std) + float(affinity_mean)
    # Move outputs back to CPU for writing.
    out_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in out.items()}
    return out_cpu, float(pred[0].item())


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RDCL inference from existing RDCL/MultiFlowDock-style PT cache directories.")
    ap.add_argument("--checkpoint", required=True, help="Path to RDCL checkpoint, e.g. checkpoints/rdcl_30m_state_dict.pt or outputs/.../last.pt")
    ap.add_argument("--config", required=True, help="Model config YAML used to build the model.")
    ap.add_argument("--input_csv", required=True, help="CSV with columns sample_id,cache_dir.")
    ap.add_argument("--out_dir", required=True, help="Output directory for prediction CSV files.")
    ap.add_argument("--top_p", type=float, default=0.15, help="Top-p fraction for marking high-confidence site edges/residues/atoms.")
    ap.add_argument("--ar_cutoff", type=float, default=None, help="Atom-residue C-alpha cutoff. Defaults to config data.ar_cutoff.")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    ap.add_argument("--use_checkpoint_config", action="store_true", help="Use config stored in checkpoint instead of --config.")
    ap.add_argument("--affinity_mean", type=float, default=None, help="Affinity mean for unstandardizing predictions. Defaults to checkpoint extra or train_stats.json values when present.")
    ap.add_argument("--affinity_std", type=float, default=None, help="Affinity std for unstandardizing predictions. Defaults to checkpoint extra or train_stats.json values when present.")
    ap.add_argument("--max_edges", type=int, default=None, help="Only write top K edges per sample to site_edges.csv. Default writes all candidate edges.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, cfg, extra = load_model(args.config, args.checkpoint, device, args.use_checkpoint_config)
    standardize = bool(cfg.get("loss", {}).get("affinity_standardize", True))
    aff_mean = args.affinity_mean
    aff_std = args.affinity_std
    infer_cfg = cfg.get("inference", {}) if isinstance(cfg, dict) else {}
    if aff_mean is None:
        aff_mean = extra.get("affinity_mean", infer_cfg.get("affinity_mean", 0.0)) if isinstance(extra, dict) else infer_cfg.get("affinity_mean", 0.0)
    if aff_std is None:
        aff_std = extra.get("affinity_std", infer_cfg.get("affinity_std", 1.0)) if isinstance(extra, dict) else infer_cfg.get("affinity_std", 1.0)
    aff_std = max(float(aff_std), 1e-6)
    ar_cutoff = float(args.ar_cutoff if args.ar_cutoff is not None else cfg.get("data", {}).get("ar_cutoff", 7.0))

    rows = read_cache_csv(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with PredictionWriter(out_dir) as writer:
        for i, r in enumerate(rows, start=1):
            sid = str(r.get("sample_id") or Path(r["cache_dir"]).name)
            # Inference has no site labels. We provide affinity=0 and disable positive edge injection.
            sample = read_mfd_pt_sample(r["cache_dir"], affinity=0.0, sample_id=sid, require_interaction=False)
            sample = build_rdcl_graph(sample, ar_cutoff=ar_cutoff, force_include_positive_pairs=False)
            out, pred_aff = infer_one(model, sample, device, float(aff_mean), float(aff_std), standardize)
            writer.write_sample(sid, sample, out, pred_affinity=pred_aff, top_p=args.top_p, max_edges=args.max_edges)
            if i % 20 == 0:
                print(f"predicted {i}/{len(rows)}")
    print(f"Wrote predictions to {out_dir}")


if __name__ == "__main__":
    main()
