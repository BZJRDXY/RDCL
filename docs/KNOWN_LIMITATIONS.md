# Known Limitations

1. RDCL is not a docking or pose-generation model. It requires an existing protein-ligand complex pose.
2. The released checkpoint was trained with the complete RDCL procedure. The public training entry point implements base training; full pseudo-label self-training is kept as an extension interface and is not enabled in this release.
3. Raw-structure preprocessing is designed to match the RDCL/MultiFlowDock-compatible feature schema, but users should validate feature and atom-order consistency when adapting the pipeline to new structural sources.
4. Direct raw inference requires ESM residue embeddings for publication-quality prediction. `--allow_zero_esm` is only for smoke tests.
5. Strict supervised training cache construction with PLIP depends on PLIP's own ligand detection and interaction rules. Complex PDB inputs should specify ligand identity explicitly.
6. The fully trained RDCL checkpoint is distributed through Git LFS. The external ESM-2 checkpoint required for publication-quality raw-structure inference is not redistributed.
