# Checkpoints

Place local trained checkpoints here. Pretrained weights are not included in the repository.

Recommended local filename:

```text
rdcl_30m_state_dict.pt
```

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

Do not commit `.pt`, `.pth`, or `.ckpt` files unless you intentionally release weights.
