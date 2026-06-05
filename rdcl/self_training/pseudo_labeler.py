from __future__ import annotations

import torch


def generate_pseudo_site_labels(*args, **kwargs):
    """Reserved RDCL self-training interface.

    Future behavior: run the base model on affinity-only samples, retain Top-p% atom-residue
    edges as pseudo site labels, and store them with confidence statistics.
    """
    raise NotImplementedError("Self-training pseudo label generation is reserved but disabled in the current base-only release.")
