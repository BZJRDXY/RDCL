from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rdcl.inference.predict_from_cache import load_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate that a checkpoint can be loaded into RDCLBaseModel with strict=True.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, cfg, extra = load_model(args.config, args.checkpoint, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print("checkpoint load: OK")
    print("checkpoint:", Path(args.checkpoint))
    print("config:", Path(args.config))
    print("device:", device)
    print("num_model_params:", n_params)
    print("extra:", extra)
    if cfg.get("inference"):
        print("inference stats:", cfg.get("inference"))


if __name__ == "__main__":
    main()
