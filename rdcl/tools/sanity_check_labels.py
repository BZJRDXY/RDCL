from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdcl.data.cache_manifest import read_manifest
from rdcl.data.graph_builder import build_rdcl_graph
from rdcl.data.pt_reader import read_mfd_pt_sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ar_cutoff", type=float, default=7.0)
    ap.add_argument("--force_include_positive_pairs", action="store_true")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    totals = {
        "n_samples": 0,
        "n_positive_pairs": 0,
        "n_positive_in_cutoff": 0,
        "n_positive_outside_cutoff": 0,
        "n_cutoff_edges": 0,
        "n_union_edges": 0,
        "n_zero_pos": 0,
        "bad": 0,
    }
    worst = []
    for r in rows:
        try:
            s = read_mfd_pt_sample(r["cache_dir"], float(r["affinity"]), r["sample_id"])
            g = build_rdcl_graph(s, args.ar_cutoff, args.force_include_positive_pairs)
            st = g["graph_stats"]
            totals["n_samples"] += 1
            for k in ["n_positive_pairs", "n_positive_in_cutoff", "n_positive_outside_cutoff", "n_cutoff_edges", "n_union_edges"]:
                totals[k] += int(st[k])
            if int(st["n_positive_pairs"]) == 0:
                totals["n_zero_pos"] += 1
            if st["positive_coverage"] < 1.0:
                worst.append((st["positive_coverage"], r["sample_id"], st))
        except Exception as exc:
            totals["bad"] += 1
            if totals["bad"] <= 5:
                print("BAD", r.get("sample_id"), repr(exc))
    n = max(totals["n_samples"], 1)
    report = dict(totals)
    report["positive_coverage_under_cutoff"] = totals["n_positive_in_cutoff"] / max(totals["n_positive_pairs"], 1)
    report["mean_cutoff_edges"] = totals["n_cutoff_edges"] / n
    report["mean_union_edges"] = totals["n_union_edges"] / n
    report["mean_positive_pairs"] = totals["n_positive_pairs"] / n
    report["worst_top20"] = sorted(worst, key=lambda x: x[0])[:20]
    print(json.dumps(report, indent=2))
    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
