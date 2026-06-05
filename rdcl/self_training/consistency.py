from __future__ import annotations

import torch


def affinity_consistency_score(z_r: torch.Tensor, z_e: torch.Tensor, affinity: torch.Tensor):
    """Reserved RDCL affinity-consistency interface.

    The intended formulation mixes target rationale representations with other samples'
    environment representations and checks whether the target affinity is preserved.
    """
    raise NotImplementedError("Affinity-consistency self-training is reserved but disabled in the current base-only release.")
