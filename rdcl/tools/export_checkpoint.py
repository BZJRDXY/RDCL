from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Export RDCL checkpoints to a clean inference or pure state_dict format.")
    ap.add_argument("--input", required=True, help="Input full/inference checkpoint or state_dict .pt file.")
    ap.add_argument("--output", required=True, help="Output .pt file.")
    ap.add_argument("--format", choices=["state_dict", "infer"], default="state_dict")
    ap.add_argument("--strip_module_prefix", action="store_true", default=True)
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    ckpt: Any = torch.load(src, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        config = ckpt.get("config", None)
        extra = ckpt.get("extra", {}) or {}
    elif isinstance(ckpt, dict):
        state = ckpt
        config = None
        extra = {}
    else:
        raise TypeError(f"Unsupported checkpoint object type: {type(ckpt)!r}")

    if args.strip_module_prefix:
        state = strip_module_prefix(state)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "state_dict":
        torch.save(state, dst)
    else:
        torch.save({"model": state, "config": config, "extra": extra}, dst)

    print(f"wrote {args.format} checkpoint: {dst}")
    print(f"num_tensors={len(state)}")
    if extra:
        print(f"extra={extra}")


if __name__ == "__main__":
    main()
