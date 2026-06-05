from __future__ import annotations

import argparse
from pathlib import Path

from rdcl.utils import safe_torch_load


def shape(x):
    if hasattr(x, "shape"):
        return tuple(x.shape)
    if isinstance(x, list):
        return f"list[{len(x)}]"
    return type(x).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    root = Path(args.cache_root)
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    print("dirs", len(dirs))
    shown = 0
    for d in dirs:
        if not (d / "graph.pt").exists():
            continue
        g = safe_torch_load(d / "graph.pt")
        print("=", d.name)
        for k in ["lig_atom_feat", "lig_atom_pos_crystal", "lig_bond_edge_index", "lig_bond_edge_attr", "prot_res_feat", "prot_res_pos", "prot_res_ids"]:
            print("graph", k, shape(g.get(k)))
        if (d / "protein_atom_static.pt").exists():
            pa = safe_torch_load(d / "protein_atom_static.pt")
            for k in ["prot_atom_feat", "prot_atom_pos", "prot_atom_res_index", "prot_atom_bond_index", "prot_res_atom_count", "prot_res_atom_ptr"]:
                print("protatom", k, shape(pa.get(k)))
        if (d / "protein_esm_sidecar.pt").exists():
            esm = safe_torch_load(d / "protein_esm_sidecar.pt")
            for k in ["esm_res_emb", "prot_res_ids", "prot_res_ids_sha1"]:
                print("esm", k, shape(esm.get(k)))
        if (d / "interaction_sidecar.pt").exists():
            inter = safe_torch_load(d / "interaction_sidecar.pt")
            for k in ["pair_index", "pair_type_mask", "pair_type_bits", "inst_type_id"]:
                print("interaction", k, shape(inter.get(k)))
        shown += 1
        if shown >= args.n:
            break


if __name__ == "__main__":
    main()
