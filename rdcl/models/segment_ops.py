from __future__ import annotations

import torch


def segment_sum(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    if src.numel() == 0:
        shape = (num_segments,) + tuple(src.shape[1:])
        return torch.zeros(shape, dtype=src.dtype, device=src.device)
    out = torch.zeros((num_segments,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
    out.index_add_(0, index.long(), src)
    return out


def segment_mean(src: torch.Tensor, index: torch.Tensor, num_segments: int, eps: float = 1e-8) -> torch.Tensor:
    out = segment_sum(src, index, num_segments)
    ones = torch.ones((src.shape[0],) + (1,) * (src.ndim - 1), dtype=src.dtype, device=src.device)
    count = segment_sum(ones, index, num_segments).clamp_min(eps)
    return out / count


def segment_softmax(src: torch.Tensor, index: torch.Tensor, num_segments: int, eps: float = 1e-12) -> torch.Tensor:
    """Numerically stable segment softmax.

    AMP-safe: softmax is computed in fp32 to avoid index_add dtype mismatch
    under torch.cuda.amp.autocast, then cast back to the original dtype.
    Supports src shape [N] or [N, H].
    """
    if src.numel() == 0:
        return src

    orig_dtype = src.dtype
    x = src.float()
    index = index.long()

    if x.ndim == 1:
        with torch.no_grad():
            max_buf = torch.full((num_segments,), -torch.inf, dtype=torch.float32, device=x.device)
            max_buf.scatter_reduce_(0, index, x.detach(), reduce="amax", include_self=True)

        exp = torch.exp(x - max_buf[index])
        denom = torch.zeros((num_segments,), dtype=torch.float32, device=x.device)
        denom.index_add_(0, index, exp)
        out = exp / denom[index].clamp_min(eps)
        return out.to(orig_dtype)

    if x.ndim == 2:
        h = x.shape[1]
        with torch.no_grad():
            max_buf = torch.full((num_segments, h), -torch.inf, dtype=torch.float32, device=x.device)
            idx2 = index[:, None].expand(-1, h)
            max_buf.scatter_reduce_(0, idx2, x.detach(), reduce="amax", include_self=True)

        exp = torch.exp(x - max_buf[index])
        denom = torch.zeros((num_segments, h), dtype=torch.float32, device=x.device)
        denom.index_add_(0, index, exp)
        out = exp / denom[index].clamp_min(eps)
        return out.to(orig_dtype)

    raise ValueError(f"segment_softmax supports 1D/2D tensors, got {src.shape}")
