"""Load bundled gene-position tables shipped in pyinfercnv/data/."""
from __future__ import annotations

from importlib import resources
from typing import Literal

import pandas as pd


Genome = Literal["hg38", "hg19", "mm10"]
_SUPPORTED: tuple[Genome, ...] = ("hg38", "hg19", "mm10")


def load_gene_positions(genome: Genome = "hg38") -> pd.DataFrame:
    """Return a DataFrame of gene_symbol/chromosome/start/end for the chosen genome.

    Data ships inside the wheel (see pyproject.toml). Drop-in as
    `adata.var[['chromosome','start','end']]` after merging on gene symbol.
    """
    if genome not in _SUPPORTED:
        raise ValueError(f"unknown genome {genome!r}; supported: {_SUPPORTED}")
    path = resources.files("pyinfercnv.data") / f"{genome}_gene_positions.parquet"
    with resources.as_file(path) as p:
        return pd.read_parquet(p)
