from __future__ import annotations

from typing import Dict, List

import torch


def affinity_metrics(pred: torch.Tensor, true: torch.Tensor) -> Dict[str, float]:
    pred = pred.detach().float().cpu()
    true = true.detach().float().cpu()
    diff = pred - true
    rmse = float(torch.sqrt(torch.mean(diff ** 2)).item())
    mae = float(torch.mean(torch.abs(diff)).item())
    if pred.numel() >= 2:
        vx = pred - pred.mean()
        vy = true - true.mean()
        pearson = float((vx * vy).sum().div(torch.sqrt((vx ** 2).sum() * (vy ** 2).sum()).clamp_min(1e-12)).item())
    else:
        pearson = 0.0
    return {"rmse": rmse, "mae": mae, "pearson": pearson}


def binary_auprc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    logits = logits.detach().float().cpu()
    labels = labels.detach().float().cpu()
    if labels.sum() <= 0:
        return 0.0
    order = torch.argsort(logits, descending=True)
    y = labels[order]
    tp = torch.cumsum(y, dim=0)
    fp = torch.cumsum(1 - y, dim=0)
    precision = tp / (tp + fp).clamp_min(1e-12)
    recall = tp / labels.sum().clamp_min(1e-12)
    recall_prev = torch.cat([torch.zeros(1), recall[:-1]])
    auprc = torch.sum((recall - recall_prev) * precision)
    return float(auprc.item())


def recall_at_top_p(logits: torch.Tensor, labels: torch.Tensor, edge_batch: torch.Tensor, num_samples: int, top_p: float = 0.15) -> float:
    vals: List[float] = []
    logits = logits.detach()
    labels = labels.detach()
    edge_batch = edge_batch.detach().long()
    for b in range(num_samples):
        mask = edge_batch == b
        if not bool(mask.any()):
            continue
        y = labels[mask].float()
        if y.sum() <= 0:
            continue
        s = logits[mask]
        k = max(1, int(torch.ceil(torch.tensor(float(top_p) * y.numel())).item()))
        idx = torch.topk(s, k=min(k, y.numel())).indices
        vals.append(float(y[idx].sum().div(y.sum().clamp_min(1.0)).item()))
    return float(sum(vals) / max(len(vals), 1))


class MetricAccumulator:
    def __init__(self, top_p: float = 0.15) -> None:
        self.top_p = top_p
        self.preds = []
        self.trues = []
        self.site_logits = []
        self.site_labels = []
        self.site_batches = []
        self.sample_offset = 0

    def update(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], pred_affinity_raw: torch.Tensor) -> None:
        n = int(batch["num_samples"])
        self.preds.append(pred_affinity_raw.detach().cpu())
        self.trues.append(batch["affinity"].detach().cpu())
        self.site_logits.append(out["site_logits"].detach().cpu())
        self.site_labels.append(batch["ar_edge_label"].detach().cpu())
        self.site_batches.append(batch["ar_edge_batch"].detach().cpu() + self.sample_offset)
        self.sample_offset += n

    def compute(self) -> Dict[str, float]:
        if not self.preds:
            return {}
        pred = torch.cat(self.preds)
        true = torch.cat(self.trues)
        site_logits = torch.cat(self.site_logits)
        site_labels = torch.cat(self.site_labels)
        site_batches = torch.cat(self.site_batches)
        out = affinity_metrics(pred, true)
        out["site_auprc"] = binary_auprc(site_logits, site_labels)
        out["site_recall_top_p"] = recall_at_top_p(site_logits, site_labels, site_batches, self.sample_offset, self.top_p)
        out["site_pos_ratio"] = float(site_labels.float().mean().item())
        return out
