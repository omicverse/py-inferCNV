"""Export hg38 gene positions parquet as the 4-column TSV R infercnv expects.

R `infercnv::CreateInfercnvObject(gene_order_file=…)` reads a tab-separated
file with NO header, 4 columns: gene / chromosome / start / end.

Only run once per benchmark; the same gene_order.tsv is shared across all
17 patients.

Usage:
    uv run python -m scripts.phase2_benchmark.export_gene_order \\
        --out benchmarks/phase2/gene_order_hg38.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.phase2_benchmark.dataset_manifest import PYINFERCNV_ROOT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parquet",
        type=Path,
        default=PYINFERCNV_ROOT / "pyinfercnv" / "data" / "hg38_gene_positions.parquet",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=PYINFERCNV_ROOT / "benchmarks" / "phase2" / "gene_order_hg38.tsv",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    df = df[["gene_symbol", "chromosome", "start", "end"]].copy()
    # R infercnv drops chrX/chrY/chrM downstream, but we can pre-drop to match
    # pyinfercnv's default chr_exclude=('chrX','chrY','chrM') and avoid shape
    # mismatches in the gene_order vs counts intersection.
    df = df[~df["chromosome"].isin({"chrX", "chrY", "chrM"})].reset_index(drop=True)
    # Sort by chromosome (natural order 1..22) then start — infercnv does not
    # require sorted input but comparing py vs R intermediates is less surprising
    # when they align on the same gene ordering.
    def _chr_key(c: str) -> tuple[int, str]:
        s = c[3:] if c.startswith("chr") else c
        try:
            return (int(s), "")
        except ValueError:
            return (99, s)
    df = df.sort_values(
        by=["chromosome", "start"], key=lambda col: col.map(_chr_key) if col.name == "chromosome" else col,
        kind="stable",
    ).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", header=False, index=False)
    print(f"[export_gene_order] wrote {len(df)} genes -> {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
