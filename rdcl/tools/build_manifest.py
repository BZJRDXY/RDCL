from __future__ import annotations

import argparse
from collections import Counter

from rdcl.data.cache_manifest import scan_cache_roots, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_roots", nargs="+", required=True)
    ap.add_argument("--source_names", nargs="+", required=True)
    ap.add_argument("--affinity_index", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--deduplicate_by_pdb_id", action="store_true")
    ap.add_argument("--dedup_priority", default=None, help="comma-separated source priority, e.g. casf2016,refined,general")
    ap.add_argument("--drop_zero_pos", action="store_true")
    args = ap.parse_args()

    priority = args.dedup_priority.split(",") if args.dedup_priority else None
    rows = scan_cache_roots(
        args.cache_roots,
        args.source_names,
        args.affinity_index,
        drop_zero_pos=args.drop_zero_pos,
        deduplicate_by_pdb_id=args.deduplicate_by_pdb_id,
        dedup_priority=priority,
    )
    write_manifest(rows, args.out_csv)
    c = Counter(r.source_name for r in rows)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    print("By source:", dict(c))
    print("Zero-positive rows retained:", sum(1 for r in rows if r.n_positive_pairs == 0))


if __name__ == "__main__":
    main()
