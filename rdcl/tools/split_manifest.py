from __future__ import annotations

import argparse
import hashlib

from rdcl.data.cache_manifest import read_manifest


def write_rows(rows, path):
    import csv
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def score_id(pdb_id: str, seed: int) -> float:
    h = hashlib.sha1((str(seed) + pdb_id).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--valid_csv", required=True)
    ap.add_argument("--valid_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = read_manifest(args.input_csv)
    train, valid = [], []
    for r in rows:
        if score_id(r["pdb_id"], args.seed) < args.valid_ratio:
            valid.append(r)
        else:
            train.append(r)
    write_rows(train, args.train_csv)
    write_rows(valid, args.valid_csv)
    print(f"train={len(train)} valid={len(valid)}")


if __name__ == "__main__":
    main()
