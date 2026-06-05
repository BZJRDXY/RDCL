from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class BaseLoss:
    def __init__(self, config: Dict, affinity_mean: float = 0.0, affinity_std: float = 1.0) -> None:
        lcfg = config["loss"]
        self.lambda_site = float(lcfg.get("lambda_site", 1.0))
        self.lambda_affinity = float(lcfg.get("lambda_affinity", 1.0))
        self.lambda_kl = float(lcfg.get("lambda_kl", 1.0))
        self.pos_weight_mode = str(lcfg.get("site_pos_weight_mode", "none"))
        self.fixed_pos_weight = float(lcfg.get("site_pos_weight", 1.0))
        self.pos_weight_cap = float(lcfg.get("site_pos_weight_cap", 10.0))
        self.standardize = bool(lcfg.get("affinity_standardize", True))
        self.affinity_mean = float(affinity_mean)
        self.affinity_std = max(float(affinity_std), 1e-6)

    def _target_affinity(self, y: torch.Tensor) -> torch.Tensor:
        if self.standardize:
            return (y - self.affinity_mean) / self.affinity_std
        return y

    def _pos_weight(self, labels: torch.Tensor) -> torch.Tensor | None:
        if self.pos_weight_mode == "none":
            return None
        if self.pos_weight_mode == "fixed":
            return labels.new_tensor(self.fixed_pos_weight)
        pos = labels.sum().clamp_min(1.0)
        neg = (labels.numel() - labels.sum()).clamp_min(1.0)
        w = (neg / pos).clamp(max=self.pos_weight_cap)
        return w.detach()

    def _kl_env(self, env_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        if env_pred.numel() < 2:
            return env_pred.sum() * 0.0
        # Non-informative Gaussian reference over the batch.
        q = torch.exp(-0.5 * y_target.detach().pow(2))
        q = q / q.sum().clamp_min(1e-12)
        log_p = F.log_softmax(env_pred, dim=0)
        return (q * (torch.log(q.clamp_min(1e-12)) - log_p)).sum()

    def __call__(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        labels = batch["ar_edge_label"].float()
        pos_weight = self._pos_weight(labels)
        if pos_weight is None:
            loss_site = F.binary_cross_entropy_with_logits(out["site_logits"], labels)
        else:
            loss_site = F.binary_cross_entropy_with_logits(out["site_logits"], labels, pos_weight=pos_weight)

        y = batch["affinity"].float()
        y_t = self._target_affinity(y)
        loss_aff = F.mse_loss(out["affinity_pred"], y_t)
        loss_kl = self._kl_env(out["env_affinity_pred"], y_t)
        total = self.lambda_site * loss_site + self.lambda_affinity * loss_aff + self.lambda_kl * loss_kl
        return {
            "loss": total,
            "loss_site": loss_site.detach(),
            "loss_affinity": loss_aff.detach(),
            "loss_kl": loss_kl.detach(),
        }

    def unstandardize(self, pred: torch.Tensor) -> torch.Tensor:
        if self.standardize:
            return pred * self.affinity_std + self.affinity_mean
        return pred
