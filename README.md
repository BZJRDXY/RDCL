# RDCL

RDCL is an open-source PyTorch implementation of **Rationale-based Dual-objective Cooperative Learning** for compound-protein interaction (CPI) prediction.

It supports:

- binding-site prediction through atom-residue interaction rationales;
- binding-affinity regression through rationale-based complex representations;
- multi-GPU DDP base-model training;
- paper-aligned Top-3% Recall and enrichment-factor evaluation;
- inference from existing PT caches;
- direct inference from raw protein-ligand complex poses without requiring users to manually build PT files;
- strict RDCL-compatible cache construction for supervised training or reproducible preprocessing.

> RDCL is **not a docking model**. It does not generate ligand poses. The ligand coordinates must already be placed in the protein coordinate frame.

## Current implementation scope

This repository implements the RDCL **base model** and base training objective:

```text
L_base = L_site + L_affinity + lambda_kl * L_KL
```

The released `checkpoints/rdcl_30m_state_dict.pt` was trained with the complete RDCL procedure described in the paper and is provided for inference. The public training entry point currently covers the base-training stage; full pseudo-label regeneration, affinity-consistency filtering, scPDB auxiliary pretraining, and complete paper-level benchmark scripts are not enabled in this release.

## Repository structure

```text
configs/                 model and training configs
rdcl/data/               PT reading, graph construction, datasets, collate
rdcl/models/             RDCL model modules
rdcl/training/           base training, losses, metrics, checkpointing
rdcl/inference/          cache/raw-structure inference scripts and output writers
rdcl/preprocess/         raw-structure preprocessing and cache construction
rdcl/tools/              manifest, sanity-check, dependency, checkpoint tools
rdcl/self_training/      reserved interfaces for pseudo-label self-training
docs/                    installation, inference, cache-building and format docs
examples/                example CSV formats
checkpoints/             released fully trained checkpoint (Git LFS)
```

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate rdcl
pip install -e .
```

### Existing environment

If PyTorch and the scientific stack are already installed:

```bash
pip install -e .
```

For quick local use without installation:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

Check dependencies:

```bash
python -m rdcl.tools.check_dependencies
python -m rdcl.tools.check_dependencies --strict
```

See [docs/INSTALL.md](docs/INSTALL.md) and [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md).

## Checkpoints

The fully trained RDCL checkpoint used for inference is released through Git LFS:

```text
checkpoints/rdcl_30m_state_dict.pt
```

This file is a pure PyTorch `state_dict` exported after the complete RDCL training procedure. It contains model parameters, but no optimizer state or training history, and must be used together with:

```text
configs/rdcl_base_30m.yaml
```

To export a full training checkpoint into a pure state_dict:

```bash
python -m rdcl.tools.export_checkpoint \
  --input outputs/rdcl_base/base_30m_alltrain/last.pt \
  --output checkpoints/rdcl_30m_state_dict.pt \
  --format state_dict
```

Validate a checkpoint:

```bash
python -m rdcl.tools.validate_checkpoint \
  --checkpoint checkpoints/rdcl_30m_state_dict.pt \
  --config configs/rdcl_base_30m.yaml
```

See [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md).

## Paper evaluation settings

The paper reports atom- and residue-level Recall and EF using a nominal Top-3% evaluation cutoff. For each complex, the integer selection size is `k = ceil(0.03 * N)`, and EF uses the actual selected fraction `k/N`. Per-complex Recall and EF are averaged separately. This evaluation setting is distinct from the Top-15% rationale-selection setting used by inference and model analysis.

The implementation is provided in `rdcl/training/metrics.py`; see [docs/EVALUATION.md](docs/EVALUATION.md) for the exact definitions.

## Training from existing PT caches

Build a manifest:

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

Run multi-GPU full training:

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

A stable starting point for A100 80GB GPUs is:

```yaml
train:
  batch_size: 4
  grad_accum_steps: 4
  amp: false
  find_unused_parameters: true
  save_every: 1
```

See [docs/TRAINING.md](docs/TRAINING.md).

## Inference from existing PT cache

Prepare a CSV:

```csv
sample_id,cache_dir
1a30,/path/to/cache/1a30
```

Run inference:

```bash
python -m rdcl.inference.predict_from_cache \
  --checkpoint checkpoints/rdcl_30m_state_dict.pt \
  --config configs/rdcl_base_30m.yaml \
  --input_csv examples/infer_cache.csv \
  --out_dir outputs/infer_cache \
  --top_p 0.15
```

Outputs:

```text
predictions.csv      predicted affinity per complex
site_edges.csv       atom-residue edge scores
residue_scores.csv   residue-level aggregated site scores
atom_scores.csv      ligand-atom rationale scores
```

## Direct raw-structure inference without manual PT construction

For separated protein and ligand files:

```csv
sample_id,protein_path,ligand_path
case001,protein.pdb,ligand.sdf
```

Run directly:

```bash
python -m rdcl.inference.predict_from_raw \
  --checkpoint checkpoints/rdcl_30m_state_dict.pt \
  --config configs/rdcl_base_30m.yaml \
  --input_csv examples/infer_raw_separated.csv \
  --out_dir outputs/infer_raw \
  --esm_model_path /path/to/esm2_t33_650M_UR50D.pt
```

`predict_from_raw` creates a temporary RDCL-compatible cache internally and removes it after prediction. Add `--cache_dir outputs/infer_raw_cache` if you want to keep the generated PT files.

For a single complex PDB, provide ligand identity explicitly:

```csv
sample_id,complex_path,ligand_resname,ligand_chain,ligand_resseq
case001,complex.pdb,LIG,A,401
```

See [docs/INFERENCE.md](docs/INFERENCE.md). Known limitations are summarized in [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

## PT/cache creation

Build inference caches:

```bash
python -m rdcl.tools.build_infer_cache_from_csv \
  --input_csv examples/infer_raw_separated.csv \
  --out_cache_dir outputs/infer_cache_build \
  --out_csv outputs/infer_cache_build/infer_cache.csv \
  --esm_model_path /path/to/esm2_t33_650M_UR50D.pt
```

Build supervised training caches with PLIP labels:

```bash
python -m rdcl.tools.build_cache_from_raw \
  --input_csv examples/build_cache_raw_separated.csv \
  --out_cache_dir outputs/rdcl_train_cache \
  --out_csv outputs/rdcl_train_cache/cache_manifest.csv \
  --esm_model_path /path/to/esm2_t33_650M_UR50D.pt \
  --interaction_mode plip
```

The raw featurizers are designed to match the training cache feature schemas. See [docs/FEATURE_ALIGNMENT.md](docs/FEATURE_ALIGNMENT.md) and [docs/CACHE_BUILDING.md](docs/CACHE_BUILDING.md).

## Citation

```bibtex
@article{wu2026rdcl,
  title={Rationale-based Dual-objective Cooperative Learning for Compound-Protein Interaction Prediction},
  author={Wu, Zheyu and Ma, Huifang and Deng, Bin and Bing, Rui and Li, Zhixin and Jia, Meihuizi and Zhang, Di},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
