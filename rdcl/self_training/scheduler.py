from __future__ import annotations


def anneal_alpha(epoch: int, t1: int = 150, t2: int = 250, lambda2: float = 1.0) -> float:
    if epoch < t1:
        return 0.0
    if epoch < t2:
        return float(epoch - t1) / float(max(t2 - t1, 1)) * float(lambda2)
    return float(lambda2)
