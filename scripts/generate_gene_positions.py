#!/usr/bin/env python3
"""Generate gene_symbol/chromosome/start/end parquet from a GENCODE GTF.

Usage:
    python scripts/generate_gene_positions.py \\
        --gtf gencode.v45.basic.annotation.gtf.gz \\
        --output pyinfercnv/data/hg38_gene_positions.parquet \\
        --genome hg38
"""
from __future__ import annotations

import argparse
import gzip
import re
import sys
from pathlib import Path

import pandas as pd


_GENE_NAME_RE = re.compile(r'gene_name\s+"([^"]+)"')


def parse_gtf(path: Path) -> pd.DataFrame:
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            m = _GENE_NAME_RE.search(parts[8])
            if not m:
                continue
            rows.append(
                {
                    "gene_symbol": m.group(1),
                    "chromosome": parts[0],
                    "start": int(parts[3]),
                    "end": int(parts[4]),
                }
            )
    df = pd.DataFrame(rows, columns=["gene_symbol", "chromosome", "start", "end"])
    df = df.drop_duplicates(subset=["gene_symbol"], keep="first").reset_index(drop=True)
    return df


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gtf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--genome", required=True, help="Label recorded in stderr only")
    args = p.parse_args()

    df = parse_gtf(args.gtf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, engine="pyarrow", compression="snappy")
    sys.stderr.write(f"[generate_gene_positions] {args.genome}: {len(df)} genes -> {args.output}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
