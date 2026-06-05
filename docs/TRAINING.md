# RDCL Training Guide

## Build manifest

```bash
python -m rdcl.tools.build_manifest \
  --cache_roots /path/to/CASF /path/to/General_minus_refined /path/to/Refined \
  --source_names casf2016 general refined \
  --affinity_index /path/to/INDEX_general_PL_data.2020 \
  --out_csv outputs/rdcl_base/manifests/all_train.csv \
  --deduplicate_by_pdb_id \
  --dedup_priority casf2016,refined,general \
  --drop_zero_pos
```

## Sanity check labels

```bash
python -m rdcl.tools.sanity_check_labels \
  --manifest outputs/rdcl_base/manifests/all_train.csv \
  --ar_cutoff 7.0 \
  --force_include_positive_pairs \
  --max_samples 1000 \
  --out_json outputs/rdcl_base/label_sanity_1000.json
```

## Single-GPU debug

```bash
CUDA_VISIBLE_DEVICES=0 python -m rdcl.tools.debug_one_batch \
  --config configs/rdcl_base_30m.yaml \
  --manifest outputs/rdcl_base/manifests/all_train.csv
```

## Multi-GPU full training

Use `python -m torch.distributed.run` instead of `torchrun` if the shell resolves `torchrun` outside your conda environment.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m rdcl.training.train_base \
  --config configs/rdcl_base_30m.yaml \
  --train_csv outputs/rdcl_base/manifests/all_train.csv \
  --valid_csv none \
  --out_dir outputs/rdcl_base/base_30m_alltrain \
  --epochs 150
```

Recommended stable settings for A100 training:

```yaml
train:
  batch_size: 4
  grad_accum_steps: 4
  amp: false
  find_unused_parameters: true
  save_every: 1
```

The current code saves:

```text
last.pt
ckpt_epoch_XXXX.pt
best.pt only when a validation CSV is provided
```

## Export a clean release checkpoint

Training checkpoints contain optimizer and scheduler states. For inference or local model sharing, export a pure state_dict:

```bash
python -m rdcl.tools.export_checkpoint \
  --input outputs/rdcl_base/base_30m_alltrain/last.pt \
  --output checkpoints/rdcl_30m_state_dict.pt \
  --format state_dict
```

Validate it:

```bash
python -m rdcl.tools.validate_checkpoint \
  --checkpoint checkpoints/rdcl_30m_state_dict.pt \
  --config configs/rdcl_base_30m.yaml
```
