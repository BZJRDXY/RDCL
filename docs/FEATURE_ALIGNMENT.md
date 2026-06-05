# Feature Alignment and Cache Schema

This repository treats feature alignment as part of the public API.  The model
was trained on RDCL/MultiFlowDock-style `.pt` caches, so raw-structure inference
and strict cache construction must produce the same tensor names, dimensions,
feature ordering, and feature meanings.

## Required dimensions

| Tensor | Shape | Schema |
|---|---:|---|
| `graph.pt/lig_atom_feat` | `[N_lig, 31]` | `multiflowdock_ligand_v1` |
| `graph.pt/lig_bond_edge_attr` | `[E_lig, 7]` | `multiflowdock_ligand_bond_v1` |
| `graph.pt/prot_res_feat` | `[N_res, 25]` | `multiflowdock_residue_v1` |
| `protein_atom_static.pt/prot_atom_feat` | `[N_atom, 17]` | `multiflowdock_protein_atom_v1` |
| `protein_esm_sidecar.pt/esm_res_emb` | `[N_res, 1280]` | ESM-2 layer 33 or compatible |

## Ligand atom features, 31 dims

The ligand atom features follow the MultiFlowDock cache definition:

1. atomic-number bucket one-hot, 20 dims; elements above 20 share bucket 20;
2. clipped atomic number divided by 100;
3. aromatic flag;
4. ring flag;
5. formal charge clipped to `[-5, 5]` and divided by 5;
6. RDKit SMARTS donor flag;
7. RDKit SMARTS acceptor flag;
8. halogen flag for F/Cl/Br/I;
9. hybridization one-hot: SP, SP2, SP3, other.

## Ligand bond features, 7 dims

1. single bond;
2. double bond;
3. triple bond;
4. aromatic bond;
5. conjugated;
6. ring bond;
7. RDKit rotatable bond flag.

## Protein residue features, 25 dims

1. 21-way one-hot: AA20 + UNK;
2. hydrophobic flag;
3. polar flag;
4. positive flag;
5. negative flag.

Residue IDs use the training-compatible format:

```text
chain:resseq:icode:resname
```

For example:

```text
A:83:_:LEU
```

## Protein atom features, 17 dims

1. element bucket one-hot over `C,N,O,S,P,F,CL,BR,I,SE,OTHER`, 11 dims;
2. atomic number divided by 30;
3. backbone atom flag;
4. side-chain atom flag;
5. template donor flag;
6. template acceptor flag;
7. template charge.

Hydrogen atoms are not included in the protein atom static view by default.

## Site-label sidecar

For supervised training, `interaction_sidecar.pt` stores PLIP-derived positive
atom-residue pairs:

```text
pair_index[0] = ligand atom index
pair_index[1] = protein residue index
```

Training candidate edges are built as:

```text
E_ar = {ligand atom - residue C-alpha distance <= 7 Å} ∪ {PLIP positive pairs}
```

Inference has no PLIP labels, so candidate edges are:

```text
E_ar = {ligand atom - residue C-alpha distance <= 7 Å}
```

## Raw preprocessing guarantee

The open-source raw preprocessors in `rdcl/preprocess/` now implement the same
field names, dimensions, and feature semantics used by the training cache.  They
are therefore suitable for direct inference and for building strict RDCL-style
PT caches.  For publication-grade supervised training, use `--interaction_mode
plip` and verify the generated `interaction_sidecar.pt` statistics.
