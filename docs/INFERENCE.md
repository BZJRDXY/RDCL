# RDCL Inference Guide

RDCL predicts binding affinity and atom-residue interaction rationales from an
existing protein-ligand complex pose.  It is **not** a docking model: the ligand
must already be placed in the protein coordinate frame.

## 1. Direct raw-structure inference, no manual PT cache step

This is the easiest user-facing path.  RDCL builds a temporary compatible cache
internally, runs the model, writes prediction CSVs, and removes the temporary
cache unless you ask it to keep it.

### 1.1 Protein and ligand are separate files

CSV format:

```csv
sample_id,protein_path,ligand_path
case001,examples/raw/case001/protein.pdb,examples/raw/case001/ligand.sdf
```

The ligand file must contain 3D coordinates in the same coordinate system as the
protein.

Run:

```bash
python -m rdcl.inference.predict_from_raw \
  --checkpoint checkpoints/rdcl_30m_state_dict.pt \
  --config configs/rdcl_base_30m.yaml \
  --input_csv examples/infer_raw_separated.csv \
  --out_dir outputs/infer_raw \
  --esm_model_path /path/to/esm2_t33_650M_UR50D.pt
```

No `--cache_dir` is required.  A temporary cache is created and deleted
automatically.

To keep the generated cache for inspection:

```bash
python -m rdcl.inference.predict_from_raw \
  --checkpoint checkpoints/rdcl_30m_state_dict.pt \
  --config configs/rdcl_base_30m.yaml \
  --input_csv examples/infer_raw_separated.csv \
  --out_dir outputs/infer_raw \
  --cache_dir outputs/infer_raw_cache \
  --esm_model_path /path/to/esm2_t33_650M_UR50D.pt
```

### 1.2 Protein-ligand complex is a single PDB file

CSV format:

```csv
sample_id,complex_path,ligand_resname,ligand_chain,ligand_resseq
case001,examples/raw/case001/complex.pdb,LIG,A,401
```

`ligand_resname`, `ligand_chain`, and `ligand_resseq` are required to avoid
accidentally selecting waters, ions, cofactors, or buffer molecules.

Run:

```bash
python -m rdcl.inference.predict_from_raw \
  --checkpoint checkpoints/rdcl_30m_state_dict.pt \
  --config configs/rdcl_base_30m.yaml \
  --input_csv examples/infer_raw_complex.csv \
  --out_dir outputs/infer_raw_complex \
  --esm_model_path /path/to/esm2_t33_650M_UR50D.pt
```

### 1.3 Zero-ESM smoke tests

For code smoke tests only, you may use zero ESM embeddings:

```bash
python -m rdcl.inference.predict_from_raw \
  --checkpoint checkpoints/rdcl_30m_state_dict.pt \
  --config configs/rdcl_base_30m.yaml \
  --input_csv examples/infer_raw_separated.csv \
  --out_dir outputs/infer_raw_zeroesm \
  --allow_zero_esm
```

Zero ESM embeddings are not recommended for final predictions because the model
was trained with ESM-2 residue embeddings.

## 2. Inference from existing PT cache

This is the most reproducible path when caches already exist.  Prepare a CSV:

```csv
sample_id,cache_dir
1a30,/path/to/cache/1a30
case002,/path/to/cache/case002
```

Each cache directory must contain:

```text
graph.pt
protein_atom_static.pt
protein_esm_sidecar.pt
```

`interaction_sidecar.pt` is not required for inference.

Run:

```bash
python -m rdcl.inference.predict_from_cache \
  --checkpoint checkpoints/rdcl_30m_state_dict.pt \
  --config configs/rdcl_base_30m.yaml \
  --input_csv examples/infer_cache.csv \
  --out_dir outputs/infer_cache \
  --top_p 0.15
```


## 2.1 Checkpoint format

The recommended local checkpoint for this project is:

```text
checkpoints/rdcl_30m_state_dict.pt
```

This is a pure PyTorch `state_dict`, so `--config configs/rdcl_base_30m.yaml` is required.
The inference script also supports full training checkpoints and inference checkpoints that contain a `model` key.

## 3. Outputs

RDCL writes four CSV files:

```text
predictions.csv      one row per complex, predicted affinity on the -logKd/Ki/IC50 scale
site_edges.csv       ranked ligand-atom / protein-residue candidate interaction edges
residue_scores.csv   residue-level aggregation of site edge scores
atom_scores.csv      ligand atom rationale scores
```

Important columns:

- `pred_affinity`: predicted binding affinity in the same negative-log scale used for training.
- `score`: normalized atom-residue interaction score.
- `site_logit`: raw edge logit before normalization.
- `is_top_p`: whether the edge belongs to the Top-p predicted interaction set.
- `res_id`: residue identifier in `chain:resseq:icode:resname` format.

## 4. Feature alignment

Raw inference uses the same RDCL/MultiFlowDock-compatible feature schema as the
training cache.  See [FEATURE_ALIGNMENT.md](FEATURE_ALIGNMENT.md).  This is
important: using a different atom or residue feature order would produce tensors
with matching dimensions but incompatible semantics.

## 5. Limitations

1. RDCL requires a protein-ligand complex pose. It does not generate docking poses.
2. Raw `.complex_path` mode supports PDB input in the current version.
3. Raw ligand input supports SDF, MOL2, and PDB through RDKit.
4. For supervised training labels, use the strict cache builder with PLIP; direct inference does not require PLIP.
