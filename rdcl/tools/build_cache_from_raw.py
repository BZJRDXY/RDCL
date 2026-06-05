from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rdcl.preprocess.esm_embedder import ESMEmbedder
from rdcl.preprocess.raw_preprocess import build_rdcl_cache_from_complex, build_rdcl_cache_from_files


def main() -> None:
    ap = argparse.ArgumentParser(description="Build strict RDCL/MultiFlowDock-compatible PT cache from raw structures.")
    ap.add_argument("--input_csv", required=True, help="CSV with sample_id and either protein_path,ligand_path or complex_path,ligand_resname,ligand_chain,ligand_resseq.")
    ap.add_argument("--out_cache_dir", required=True)
    ap.add_argument("--out_csv", required=True, help="Output manifest with sample_id,cache_dir. Add affinity later or join with your affinity index for training.")
    ap.add_argument("--esm_model_path", default=None)
    ap.add_argument("--esm_layer", type=int, default=33)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow_zero_esm", action="store_true", help="Only for smoke tests; strict reproduction should provide ESM model.")
    ap.add_argument("--interaction_mode", choices=["none", "plip", "contact"], default="plip", help="Use plip for strict site labels. contact is a smoke-test fallback, not publication-quality PLIP labels.")
    ap.add_argument("--contact_cutoff", type=float, default=5.0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.input_csv, "r", newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"Empty input CSV: {args.input_csv}")
    out_cache_dir = Path(args.out_cache_dir)
    out_cache_dir.mkdir(parents=True, exist_ok=True)
    esm = ESMEmbedder(args.esm_model_path, device=args.device, layer=args.esm_layer, allow_zero_esm=args.allow_zero_esm)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_id", "cache_dir", "status", "message"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for idx, r in enumerate(rows, start=1):
            sid = r.get("sample_id") or f"sample_{idx:05d}"
            try:
                if r.get("protein_path") and r.get("ligand_path"):
                    cdir = build_rdcl_cache_from_files(sid, r["protein_path"], r["ligand_path"], out_cache_dir, esm, interaction_mode=args.interaction_mode, contact_cutoff=args.contact_cutoff)
                elif r.get("complex_path"):
                    cdir = build_rdcl_cache_from_complex(sid, r["complex_path"], r["ligand_resname"], r["ligand_chain"], r["ligand_resseq"], out_cache_dir, esm, r.get("ligand_icode") or None, interaction_mode=args.interaction_mode, contact_cutoff=args.contact_cutoff)
                else:
                    raise ValueError("row must contain protein_path+ligand_path or complex_path+ligand_resname+ligand_chain+ligand_resseq")
                w.writerow({"sample_id": sid, "cache_dir": str(cdir), "status": "ok", "message": ""})
                print(f"[{idx}/{len(rows)}] ok {sid}: {cdir}")
            except Exception as exc:
                w.writerow({"sample_id": sid, "cache_dir": "", "status": "failed", "message": repr(exc)})
                print(f"[{idx}/{len(rows)}] failed {sid}: {exc}")
    print(f"Wrote cache build manifest: {out_csv}")


if __name__ == "__main__":
    main()
