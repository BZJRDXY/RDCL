from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from rdcl.data.collate import move_batch_to_device, rdcl_collate
from rdcl.data.dataset import RDCLDataset
from rdcl.models.rdcl_model import RDCLBaseModel
from rdcl.training.losses import BaseLoss
from rdcl.utils import count_parameters, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if int(cfg.get("train", {}).get("torch_num_threads", 0)) > 0:
        torch.set_num_threads(int(cfg["train"]["torch_num_threads"]))
    ds = RDCLDataset(args.manifest, cfg["data"]["ar_cutoff"], cfg["data"].get("force_include_positive_pairs", True), max_samples=8)
    loader = DataLoader(ds, batch_size=min(2, len(ds)), shuffle=False, collate_fn=rdcl_collate, num_workers=0)
    batch = next(iter(loader))
    print("sample_ids", batch["sample_ids"])
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape), v.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RDCLBaseModel(cfg).to(device)
    print("params", count_parameters(model))
    aff = torch.tensor(ds.affinities())
    loss_fn = BaseLoss(cfg, float(aff.mean()), float(aff.std().clamp_min(1e-6)))
    batch = move_batch_to_device(batch, device)
    out = model(batch)
    for k, v in out.items():
        if torch.is_tensor(v):
            print("out", k, tuple(v.shape), v.dtype)
    losses = loss_fn(out, batch)
    print({k: float(v.detach().cpu()) for k, v in losses.items()})


if __name__ == "__main__":
    main()
