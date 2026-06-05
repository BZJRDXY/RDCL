# Strict PT Cache Construction

RDCL can train from RDCL/MultiFlowDock-compatible `.pt` caches.  This repository
includes a strict cache builder that writes the same core files consumed by the
training and inference code.

## Input CSV formats

### Separated protein and ligand files

```csv
sample_id,protein_path,ligand_path
case001,/path/to/protein.pdb,/path/to/ligand.sdf
```

The ligand file must contain 3D coordinates in the same coordinate system as the
protein.  RDCL is not a docking model and does not place the ligand into a
binding site.

### Single complex PDB

```csv
sample_id,complex_path,ligand_resname,ligand_chain,ligand_resseq
case001,/path/to/complex.pdb,LIG,A,401
```

The ligand identity fields are required.  Automatic ligand detection is avoided
because PDB files may contain waters, ions, cofactors, crystallization agents,
or multiple ligands.

## Build an inference cache

This creates the three files needed for inference:

```text
graph.pt
protein_atom_static.pt
protein_esm_sidecar.pt
```

Command:

```bash
python -m rdcl.tools.build_infer_cache_from_csv \
  --input_csv examples/infer_raw_separated.csv \
  --out_cache_dir outputs/infer_cache_build \
  --out_csv outputs/infer_cache_build/infer_cache.csv \
  --esm_model_path /path/to/esm2_t33_650M_UR50D.pt
```

## Build a supervised training cache with PLIP labels

This creates the same inference files plus `interaction_sidecar.pt`:

```bash
python -m rdcl.tools.build_cache_from_raw \
  --input_csv examples/infer_raw_separated.csv \
  --out_cache_dir outputs/rdcl_train_cache \
  --out_csv outputs/rdcl_train_cache/cache_manifest.csv \
  --esm_model_path /path/to/esm2_t33_650M_UR50D.pt \
  --interaction_mode plip
```

`--interaction_mode plip` requires the optional `plip` package.  It is the
recommended mode for strict supervised training labels.

For smoke tests only, you may use a geometric contact fallback:

```bash
python -m rdcl.tools.build_cache_from_raw \
  --input_csv examples/infer_raw_separated.csv \
  --out_cache_dir outputs/rdcl_train_cache_contact \
  --out_csv outputs/rdcl_train_cache_contact/cache_manifest.csv \
  --allow_zero_esm \
  --interaction_mode contact \
  --contact_cutoff 5.0
```

The contact fallback is not a replacement for PLIP labels in publication-quality
training.

## Output files

Each sample directory contains:

```text
graph.pt
protein_atom_static.pt
protein_esm_sidecar.pt
interaction_sidecar.pt       # only when interaction_mode != none
meta.json
qc.json
```

Core tensor schemas are described in [FEATURE_ALIGNMENT.md](FEATURE_ALIGNMENT.md).

## From cache to training manifest

If your cache directories include affinity labels in an external PDBbind-style
index, build a training manifest with:

```bash
python -m rdcl.tools.build_manifest \
  --cache_roots outputs/rdcl_train_cache \
  --source_names custom \
  --affinity_index /path/to/INDEX_general_PL_data.2020 \
  --out_csv outputs/rdcl_train_cache/train_manifest.csv \
  --drop_zero_pos
```

For custom affinity CSVs, adapt `rdcl/data/affinity_index.py` or create a
manifest with columns `sample_id,cache_dir,affinity,n_positive_pairs`.
