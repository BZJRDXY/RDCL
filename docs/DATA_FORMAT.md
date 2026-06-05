# RDCL Data Format

## Training manifest

`rdcl.tools.build_manifest` creates CSV files with fields such as:

```text
sample_id,pdb_id,source_name,cache_dir,affinity,n_positive_pairs,usable_for_base
```

Base training uses rows where `usable_for_base=1`.

## Cache directory

A supervised training cache directory contains:

```text
graph.pt
protein_atom_static.pt
protein_esm_sidecar.pt
interaction_sidecar.pt
meta.json
qc.json
```

An inference cache directory only requires:

```text
graph.pt
protein_atom_static.pt
protein_esm_sidecar.pt
```

## Required tensors

`graph.pt`:

```text
lig_atom_feat            [N_lig, 31]
lig_atom_pos_crystal     [N_lig, 3]
lig_bond_edge_index      [2, E_lig]
lig_bond_edge_attr       [E_lig, 7]
prot_res_feat            [N_res, 25]
prot_res_pos             [N_res, 3]
prot_res_ids             list[str], format chain:resseq:icode:resname
```

`protein_atom_static.pt`:

```text
prot_atom_feat           [N_atom, 17]
prot_atom_pos            [N_atom, 3]
prot_atom_res_index      [N_atom]
prot_atom_bond_index     [2, E_prot]
prot_res_atom_count      [N_res]
prot_res_atom_ptr        [N_res + 1]
```

`protein_esm_sidecar.pt`:

```text
esm_res_emb              [N_res, 1280]
```

`interaction_sidecar.pt` for supervised training:

```text
pair_index               [2, E_pos]
pair_type_mask           [E_pos, 14]
pair_type_bits           [E_pos]
```

where `pair_index[0]` is the ligand atom index and `pair_index[1]` is the
protein residue index.

Feature schemas are specified in [FEATURE_ALIGNMENT.md](FEATURE_ALIGNMENT.md).

## Candidate atom-residue edges

During training, RDCL constructs:

```text
E_ar = {ligand atom - residue C-alpha distance <= 7 Å} ∪ {PLIP positive pairs}
```

During inference, there are no PLIP labels, so RDCL uses:

```text
E_ar = {ligand atom - residue C-alpha distance <= 7 Å}
```
