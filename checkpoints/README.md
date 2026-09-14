# Checkpoints

The fully trained RDCL checkpoint is released in this directory through Git LFS:

```text
rdcl_30m_state_dict.pt
```

It was trained with the complete RDCL procedure described in the paper. The released file is a pure PyTorch `state_dict` for inference, so it contains model parameters but no optimizer state, epoch, or training history.

Export a pure state_dict from a full training checkpoint:

```bash
python -m rdcl.tools.export_checkpoint \
  --input outputs/rdcl_base/base_30m_alltrain/last.pt \
  --output checkpoints/rdcl_30m_state_dict.pt \
  --format state_dict
```

Validate:

```bash
python -m rdcl.tools.validate_checkpoint \
  --checkpoint checkpoints/rdcl_30m_state_dict.pt \
  --config configs/rdcl_base_30m.yaml
```

Other `.pt`, `.pth`, and `.ckpt` files should remain local unless they are intentionally released.
