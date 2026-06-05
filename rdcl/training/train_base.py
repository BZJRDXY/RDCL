from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from rdcl.data.collate import move_batch_to_device, rdcl_collate
from rdcl.data.dataset import RDCLDataset
from rdcl.models.rdcl_model import RDCLBaseModel
from rdcl.training.checkpoint import load_checkpoint, save_checkpoint
from rdcl.training.losses import BaseLoss
from rdcl.training.metrics import MetricAccumulator
from rdcl.utils import count_parameters, ensure_dir, is_rank0, load_config, save_json, set_seed


def init_distributed() -> tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, rank, world, local_rank
    return False, 0, 1, 0


def reduce_scalar(x: torch.Tensor, world: int) -> torch.Tensor:
    if world <= 1:
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= world
    return y


def evaluate(model, loader, device, loss_fn: BaseLoss, top_p: float) -> Dict[str, float]:
    model.eval()
    acc = MetricAccumulator(top_p=top_p)
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            out = model(batch)
            loss_dict = loss_fn(out, batch)
            pred_raw = loss_fn.unstandardize(out["affinity_pred"])
            acc.update(out, batch, pred_raw)
            losses.append(float(loss_dict["loss"].detach().cpu().item()))
    metrics = acc.compute()
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--valid_csv", default="none")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    if int(config.get("train", {}).get("torch_num_threads", 0)) > 0:
        torch.set_num_threads(int(config["train"]["torch_num_threads"]))
    set_seed(int(config.get("seed", 0)))
    ddp, rank, world, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.out_dir)

    dcfg = config["data"]
    train_ds = RDCLDataset(
        args.train_csv,
        ar_cutoff=float(dcfg.get("ar_cutoff", 7.0)),
        force_include_positive_pairs=bool(dcfg.get("force_include_positive_pairs", True)),
        max_samples=dcfg.get("max_train_samples"),
    )
    aff = np.array(train_ds.affinities(), dtype=np.float32)
    aff_mean = float(aff.mean())
    aff_std = float(aff.std() if aff.std() > 1e-6 else 1.0)

    if is_rank0():
        print(f"Train samples: {len(train_ds)}")
        print(f"Affinity mean/std: {aff_mean:.6f} / {aff_std:.6f}")
        save_json({"affinity_mean": aff_mean, "affinity_std": aff_std, "train_samples": len(train_ds)}, out_dir / "train_stats.json")

    sampler = DistributedSampler(train_ds, shuffle=True) if ddp else RandomSampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["train"].get("batch_size", 2)),
        sampler=sampler,
        num_workers=int(dcfg.get("num_workers", 4)),
        collate_fn=rdcl_collate,
        pin_memory=bool(dcfg.get("pin_memory", True)),
        drop_last=False,
        persistent_workers=int(dcfg.get("num_workers", 4)) > 0,
    )

    valid_loader = None
    if args.valid_csv.lower() not in {"none", "null", ""} and is_rank0():
        valid_ds = RDCLDataset(
            args.valid_csv,
            ar_cutoff=float(dcfg.get("ar_cutoff", 7.0)),
            force_include_positive_pairs=bool(dcfg.get("force_include_positive_pairs", True)),
            max_samples=dcfg.get("max_valid_samples"),
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=int(config["train"].get("batch_size", 2)),
            shuffle=False,
            num_workers=int(dcfg.get("num_workers", 4)),
            collate_fn=rdcl_collate,
            pin_memory=bool(dcfg.get("pin_memory", True)),
            drop_last=False,
            persistent_workers=int(dcfg.get("num_workers", 4)) > 0,
        )
        print(f"Valid samples: {len(valid_ds)}")

    model = RDCLBaseModel(config).to(device)
    total, trainable = count_parameters(model)
    if is_rank0():
        print(f"Parameters: total={total:,}, trainable={trainable:,}")
        save_json({"total": total, "trainable": trainable}, out_dir / "param_count.json")

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=bool(config["train"].get("find_unused_parameters", False)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"].get("lr", 1e-4)), weight_decay=float(config["train"].get("weight_decay", 1e-5)))
    use_amp = bool(config["train"].get("amp", True)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = BaseLoss(config, affinity_mean=aff_mean, affinity_std=aff_std)

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scaler, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("step", 0))
        if is_rank0():
            print(f"Resumed from {args.resume}: epoch={start_epoch}, step={global_step}")

    epochs = int(args.epochs if args.epochs is not None else config["train"].get("epochs", 150))
    grad_accum = int(config["train"].get("grad_accum_steps", 1))
    log_every = int(config["train"].get("log_every", 20))
    save_every = int(config["train"].get("save_every", 5))
    eval_every = int(config["train"].get("eval_every", 1))
    clip = float(config["train"].get("grad_clip_norm", 0.0))
    top_p = float(config.get("metrics", {}).get("top_p", 0.15))

    best_valid = float("inf")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs):
        if ddp and isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        model.train()
        running = {"loss": 0.0, "loss_site": 0.0, "loss_affinity": 0.0, "loss_kl": 0.0}
        n_log = 0
        for it, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            is_accum_boundary = ((it + 1) % grad_accum == 0) or ((it + 1) == len(train_loader))
            sync_context = contextlib.nullcontext()
            if ddp and not is_accum_boundary:
                sync_context = model.no_sync()
            with sync_context:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(batch)
                    loss_dict = loss_fn(out, batch)
                    loss = loss_dict["loss"] / grad_accum
                scaler.scale(loss).backward()

            for k in running:
                running[k] += float(loss_dict[k].detach().cpu().item())
            n_log += 1

            if is_accum_boundary:
                if clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if is_rank0() and global_step % log_every == 0:
                    msg = {k: running[k] / max(n_log, 1) for k in running}
                    print(f"epoch={epoch} step={global_step} " + " ".join(f"{k}={v:.6f}" for k, v in msg.items()))
                    running = {k: 0.0 for k in running}
                    n_log = 0

        if ddp:
            dist.barrier()

        if is_rank0() and valid_loader is not None and ((epoch + 1) % eval_every == 0):
            eval_model = model.module if hasattr(model, "module") else model
            metrics = evaluate(eval_model, valid_loader, device, loss_fn, top_p)
            print("VALID epoch={} ".format(epoch) + " ".join(f"{k}={v:.6f}" for k, v in metrics.items()))
            save_json(metrics, out_dir / f"valid_epoch_{epoch:04d}.json")
            if metrics.get("rmse", float("inf")) < best_valid:
                best_valid = metrics["rmse"]
                save_checkpoint(out_dir / "best.pt", model, optimizer, scaler, epoch, global_step, config, {"metrics": metrics, "affinity_mean": aff_mean, "affinity_std": aff_std})

        if is_rank0() and ((epoch + 1) % save_every == 0 or (epoch + 1) == epochs):
            save_checkpoint(out_dir / f"ckpt_epoch_{epoch:04d}.pt", model, optimizer, scaler, epoch, global_step, config, {"affinity_mean": aff_mean, "affinity_std": aff_std})
            save_checkpoint(out_dir / "last.pt", model, optimizer, scaler, epoch, global_step, config, {"affinity_mean": aff_mean, "affinity_std": aff_std})

        if ddp:
            dist.barrier()

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
