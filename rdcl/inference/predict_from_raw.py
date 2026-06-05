from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

from rdcl.preprocess.esm_embedder import ESMEmbedder
from rdcl.preprocess.raw_preprocess import build_inference_cache_from_complex, build_inference_cache_from_files


def _build_caches(rows, cache_dir: Path, esm: ESMEmbedder) -> Path:
    cache_csv = cache_dir / "infer_cache.csv"
    with cache_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "cache_dir"])
        w.writeheader()
        for idx, r in enumerate(rows, start=1):
            sid = r.get("sample_id") or f"sample_{idx:05d}"
            if r.get("protein_path") and r.get("ligand_path"):
                cdir = build_inference_cache_from_files(sid, r["protein_path"], r["ligand_path"], cache_dir, esm)
            elif r.get("complex_path"):
                for col in ["ligand_resname", "ligand_chain", "ligand_resseq"]:
                    if not r.get(col):
                        raise ValueError(f"Row {idx}: complex_path mode requires {col}")
                cdir = build_inference_cache_from_complex(
                    sid,
                    r["complex_path"],
                    r["ligand_resname"],
                    r["ligand_chain"],
                    r["ligand_resseq"],
                    cache_dir,
                    esm,
                    r.get("ligand_icode") or None,
                )
            else:
                raise ValueError("Each row must contain either protein_path+ligand_path or complex_path+ligand_resname+ligand_chain+ligand_resseq")
            w.writerow({"sample_id": sid, "cache_dir": str(cdir)})
            print(f"[{idx}/{len(rows)}] built in-memory/temporary RDCL cache for {sid}: {cdir}", flush=True)
    return cache_csv


def _run_predict_from_cache(cache_csv: Path, args) -> None:
    from rdcl.inference import predict_from_cache

    argv = [
        "predict_from_cache",
        "--checkpoint", args.checkpoint,
        "--config", args.config,
        "--input_csv", str(cache_csv),
        "--out_dir", args.out_dir,
        "--top_p", str(args.top_p),
        "--device", args.device,
    ]
    if args.max_edges is not None:
        argv += ["--max_edges", str(args.max_edges)]
    if args.ar_cutoff is not None:
        argv += ["--ar_cutoff", str(args.ar_cutoff)]
    old = sys.argv
    try:
        sys.argv = argv
        predict_from_cache.main()
    finally:
        sys.argv = old


def main() -> None:
    ap = argparse.ArgumentParser(description="Direct RDCL inference from raw structures. No manual/offline PT cache construction is required.")
    ap.add_argument("--input_csv", required=True, help="CSV with either protein_path,ligand_path or complex_path,ligand_resname,ligand_chain,ligand_resseq.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default=None, help="Optional cache directory to keep generated RDCL PT files. If omitted, a temporary cache is used and removed after inference.")
    ap.add_argument("--keep_cache", action="store_true", help="Keep generated cache even when --cache_dir is omitted; cache is written under out_dir/_rdcl_tmp_cache.")
    ap.add_argument("--esm_model_path", default=None, help="Path to ESM-2 model. If omitted, use --allow_zero_esm for smoke tests only.")
    ap.add_argument("--esm_layer", type=int, default=33)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow_zero_esm", action="store_true", help="Use zero ESM embeddings when no ESM model is provided. Not recommended for final prediction.")
    ap.add_argument("--top_p", type=float, default=0.15)
    ap.add_argument("--ar_cutoff", type=float, default=None)
    ap.add_argument("--max_edges", type=int, default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.input_csv, "r", newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"Empty CSV: {args.input_csv}")
    esm = ESMEmbedder(args.esm_model_path, device=args.device, layer=args.esm_layer, allow_zero_esm=args.allow_zero_esm)

    if args.cache_dir is not None or args.keep_cache:
        cache_dir = Path(args.cache_dir) if args.cache_dir is not None else Path(args.out_dir) / "_rdcl_tmp_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_csv = _build_caches(rows, cache_dir, esm)
        _run_predict_from_cache(cache_csv, args)
    else:
        with tempfile.TemporaryDirectory(prefix="rdcl_infer_cache_") as tmp:
            cache_csv = _build_caches(rows, Path(tmp), esm)
            _run_predict_from_cache(cache_csv, args)


if __name__ == "__main__":
    main()
