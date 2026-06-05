# Installation

RDCL supports three installation levels. Pick the smallest one that matches your use case.

## 1. Training from existing PT/cache files

This mode is enough for training RDCL from prebuilt `graph.pt`, `protein_atom_static.pt`, `protein_esm_sidecar.pt`, and `interaction_sidecar.pt` files.

```bash
conda create -n rdcl python=3.10 -y
conda activate rdcl
conda install pytorch=2.5 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install -e .
```

On clusters where PyTorch is already installed, you can simply run:

```bash
pip install -e .
```

## 2. Direct raw-structure inference

This mode supports `protein.pdb + ligand.sdf/mol2` or `complex.pdb` inference without manual PT construction.

```bash
pip install -r requirements.txt
pip install -r requirements-preprocess.txt
pip install -e .
```

Raw inference also requires an ESM-2 checkpoint when using real ESM embeddings:

```bash
python -m rdcl.inference.predict_from_raw \
  --checkpoint checkpoints/rdcl_30m_state_dict.pt \
  --config configs/rdcl_base_30m.yaml \
  --input_csv examples/infer_raw_separated.csv \
  --out_dir outputs/infer_raw \
  --esm_model_path /path/to/esm2_t33_650M_UR50D.pt
```

`--allow_zero_esm` is available only for smoke tests. It is not recommended for scientific results because the model was trained with ESM features.

## 3. Strict supervised cache building

This mode builds training caches with PLIP-derived atom-residue labels.

Recommended conda environment:

```bash
conda env create -f environment.yml
conda activate rdcl
pip install -e .
```

Check PLIP and OpenBabel:

```bash
python - <<'PY'
import rdkit, Bio, esm, plip
print('RDKit OK')
print('Biopython OK')
print('ESM OK')
print('PLIP OK')
PY
plip -h
obabel -V
```

## Server-validated environment

The project was checked on a server with:

- Python 3.10.19
- PyTorch 2.5.1 + CUDA 12.1
- NumPy 2.2.6
- pandas 2.3.3
- scikit-learn 1.7.2
- RDKit 2025.09.4
- Biopython 1.86
- fair-esm/esm 2.0.0
- PLIP 3.0.0
- OpenBabel 3.1.0
- 4 × NVIDIA A100 80GB

The listed package ranges in `requirements.txt` and `pyproject.toml` are intentionally broader than this exact server environment.
