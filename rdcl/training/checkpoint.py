from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: str | Path, model, optimizer, scaler, epoch: int, step: int, config: Dict, extra: Dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "step": step,
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "extra": extra or {},
    }
    torch.save(state, path)


def load_checkpoint(path: str | Path, model, optimizer=None, scaler=None, map_location="cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt
