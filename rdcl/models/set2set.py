from __future__ import annotations

import torch
from torch import nn

from rdcl.models.segment_ops import segment_softmax, segment_sum


class Set2Set(nn.Module):
    def __init__(self, input_dim: int, processing_steps: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.processing_steps = processing_steps
        self.lstm = nn.LSTMCell(2 * input_dim, input_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor, num_samples: int) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_zeros((num_samples, 2 * self.input_dim))
        q = x.new_zeros((num_samples, self.input_dim))
        h = x.new_zeros((num_samples, self.input_dim))
        c = x.new_zeros((num_samples, self.input_dim))
        r = x.new_zeros((num_samples, self.input_dim))
        q_star = torch.cat([q, r], dim=-1)
        for _ in range(self.processing_steps):
            h, c = self.lstm(q_star, (h, c))
            q = h
            logits = (x * q[batch.long()]).sum(dim=-1)
            a = segment_softmax(logits, batch.long(), num_samples)
            r = segment_sum(a.unsqueeze(-1) * x, batch.long(), num_samples)
            q_star = torch.cat([q, r], dim=-1)
        return q_star
