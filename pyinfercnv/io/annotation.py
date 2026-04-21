"""BED-like gene_order_file reader (R infercnv format: symbol\\tchr\\tstart\\tend)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_gene_order_file(path: str | Path) -> pd.DataFrame:
    """Read R infercnv's `gene_order_file` — tab-separated, no header,
    exactly 4 columns: gene_symbol, chromosome, start, end.

    Returns DataFrame with those column names, preserving input row order.
    """
    path = Path(path)
    try:
        df = pd.read_csv(path, sep="\t", header=None, dtype=str, comment="#")
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"gene_order_file is empty: {path}") from e
    if df.empty:
        raise ValueError(f"gene_order_file is empty: {path}")
    if df.shape[1] != 4:
        raise ValueError(
            f"gene_order_file must have exactly 4 columns (symbol, chr, start, end); "
            f"got {df.shape[1]}: {path}"
        )
    df.columns = ["gene_symbol", "chromosome", "start", "end"]
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return df
