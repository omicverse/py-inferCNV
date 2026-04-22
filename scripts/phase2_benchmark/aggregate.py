"""Aggregate per-patient compare_<type>.json files into a top-level summary.

Writes benchmarks/phase2/phase2_py_vs_r_summary.csv with one row per
(cancer, sample, hmm_type). Mirrors pycopykat's benchmarks/full/py_vs_r_summary.csv
shape so downstream visualizations can consume either.

Usage:
    uv run python -m scripts.phase2_benchmark.aggregate
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.phase2_benchmark.dataset_manifest import BENCHMARK_OUT, HMM_TYPES, PATIENTS


def _load_compare(p, hmm_type: str) -> dict | None:
    path = p.out_dir(BENCHMARK_OUT) / f"compare_{hmm_type}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> int:
    rows = []
    for p in PATIENTS:
        for hmm_type in HMM_TYPES:
            rec = _load_compare(p, hmm_type)
            if rec is None:
                rows.append({
                    "cancer": p.cancer, "sample": p.sample, "hmm_type": hmm_type,
                    "status": "missing",
                })
                continue
            j = rec.get("jaccard", {}) or {}
            s = rec.get("subclusters", {}) or {}
            t = rec.get("timings", {}) or {}
            rows.append({
                "cancer": p.cancer,
                "sample": p.sample,
                "hmm_type": hmm_type,
                "status": "ok" if j.get("mean_jaccard") is not None else "partial",
                "n_cells_shared": j.get("n_cells_shared"),
                "n_genes_shared": j.get("n_genes_shared"),
                "mean_jaccard": j.get("mean_jaccard"),
                "mean_jaccard_nonzero_only": j.get("mean_jaccard_nonzero_only"),
                "ari_subcluster": s.get("ari"),
                "py_s": t.get("py_s"),
                "r_s": t.get("r_s"),
                "speedup": t.get("speedup"),
            })
    df = pd.DataFrame(rows)
    out = BENCHMARK_OUT / "phase2_py_vs_r_summary.csv"
    df.to_csv(out, index=False)
    print(f"[aggregate] wrote {out}  (rows={len(df)})")
    # Print a short legible summary
    ok = df[df["status"] == "ok"]
    if not ok.empty:
        print("\n--- mean metrics over completed patients ---")
        for hmm_type in HMM_TYPES:
            sub = ok[ok["hmm_type"] == hmm_type]
            if sub.empty:
                continue
            print(
                f"  {hmm_type:>3}: n={len(sub):2d}  mean_jaccard={sub.mean_jaccard.mean():.3f}  "
                f"nonzero={sub.mean_jaccard_nonzero_only.mean():.3f}  "
                f"ARI={sub.ari_subcluster.mean():.3f}  "
                f"speedup={sub.speedup.mean():.1f}x"
            )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
