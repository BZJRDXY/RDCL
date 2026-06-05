# Known Limitations

1. RDCL is not a docking or pose-generation model. It requires an existing protein-ligand complex pose.
2. The default released code implements RDCL base training. Full pseudo-label self-training is kept as an extension interface and is not enabled by default.
3. Raw-structure preprocessing is designed to match the RDCL/MultiFlowDock-compatible feature schema, but users should validate feature and atom-order consistency when adapting the pipeline to new structural sources.
4. Direct raw inference requires ESM residue embeddings for publication-quality prediction. `--allow_zero_esm` is only for smoke tests.
5. Strict supervised training cache construction with PLIP depends on PLIP's own ligand detection and interaction rules. Complex PDB inputs should specify ligand identity explicitly.
6. Pretrained weights and ESM checkpoints are not redistributed in this repository.
