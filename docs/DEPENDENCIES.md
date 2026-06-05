# Dependencies

## Dependency tiers

| Use case | Required packages | Notes |
|---|---|---|
| Train from existing PT cache | torch, numpy, pandas, pyyaml, scikit-learn, scipy, tqdm | Does not require RDKit/PLIP at runtime if caches already exist. |
| Predict from existing PT cache | same as training | Uses prebuilt graph and ESM sidecar. |
| Predict directly from raw structures | training deps + RDKit + Biopython + fair-esm | Needs protein/ligand parsing and ESM residue embeddings. |
| Build strict supervised cache | raw deps + PLIP; OpenBabel recommended | PLIP creates atom-residue non-covalent labels. |
| Development | pytest, ruff, build | Optional. |

## Package files

- `requirements.txt`: base training/cache-inference dependencies.
- `requirements-preprocess.txt`: raw-inference and cache-building extras.
- `requirements-dev.txt`: development helpers.
- `environment.yml`: recommended conda environment for full raw inference and strict cache building.
- `pyproject.toml`: package metadata and optional extras.

## Recommended commands

Base training only:

```bash
pip install -r requirements.txt
pip install -e .
```

Raw inference:

```bash
pip install -r requirements.txt
pip install -r requirements-preprocess.txt
pip install -e .
```

Full conda setup:

```bash
conda env create -f environment.yml
conda activate rdcl
pip install -e .
```

## External model files

Direct raw inference requires an ESM-2 checkpoint. The common model used by this project is:

```text
esm2_t33_650M_UR50D.pt
```

The checkpoint is not redistributed in this repository. Pass its path via `--esm_model_path`.

## Command-line tools

PLIP is required only when `--interaction_mode plip` is used for strict supervised cache building. OpenBabel is optional but useful for format conversion and MOL2 handling.

```bash
plip -h
obabel -V
```

If OpenBabel is unavailable, prefer installing it from conda-forge:

```bash
conda install -c conda-forge openbabel
```
