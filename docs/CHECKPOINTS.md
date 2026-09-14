# Checkpoints

The fully trained RDCL checkpoint is included through Git LFS. It was trained with the complete RDCL procedure described in the paper and is released for inference.

## Recommended local checkpoint

For the released 30M architecture, use:

```text
checkpoints/rdcl_30m_state_dict.pt
```

This is a pure PyTorch `state_dict` file exported after complete RDCL training. It contains only model parameters. It does not contain optimizer state, epoch, step, GradScaler state, or training logs.

Because it is a pure state_dict, inference must also receive the model config:

```bash
python -m rdcl.inference.predict_from_cache \
  --checkpoint checkpoints/rdcl_30m_state_dict.pt \
  --config configs/rdcl_base_30m.yaml \
  --input_csv examples/infer_cache.csv \
  --out_dir outputs/infer_cache
```

## Supported checkpoint formats

RDCL inference supports three formats:

1. **Pure state_dict**

```text
OrderedDict[str, Tensor]
```

Recommended for clean model release. Requires `--config`.

2. **Inference checkpoint**

```python
{
    "model": state_dict,
    "config": config_dict,
    "extra": {"affinity_mean": ..., "affinity_std": ...},
}
```

3. **Full training checkpoint**

```python
{
    "epoch": int,
    "step": int,
    "model": state_dict,
    "optimizer": optimizer_state,
    "scaler": scaler_state,
    "config": config_dict,
    "extra": {...},
}
```

Use the full checkpoint only for resuming training.

## Exporting a clean state_dict

```bash
python -m rdcl.tools.export_checkpoint \
  --input outputs/rdcl_base/base_30m_alltrain/last.pt \
  --output checkpoints/rdcl_30m_state_dict.pt \
  --format state_dict
```

## Validating a checkpoint

```bash
python -m rdcl.tools.validate_checkpoint \
  --checkpoint checkpoints/rdcl_30m_state_dict.pt \
  --config configs/rdcl_base_30m.yaml
```

Expected output includes:

```text
checkpoint load: OK
num_model_params: 34049300
```

## Affinity standardization

The model was trained with standardized affinity labels. If a checkpoint does not contain `extra.affinity_mean` and `extra.affinity_std`, RDCL reads fallback values from:

```yaml
inference:
  affinity_mean: ...
  affinity_std: ...
```

in `configs/rdcl_base_30m.yaml`.

If you train on a different dataset, replace these values with the statistics printed by `train_base.py`.
