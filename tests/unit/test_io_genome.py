"""Tests for pyinfercnv.io.genome.load_gene_positions."""
from __future__ import annotations

import pytest

from pyinfercnv.io.genome import load_gene_positions


def test_load_hg38_default():
    df = load_gene_positions()
    assert list(df.columns) == ["gene_symbol", "chromosome", "start", "end"]
    assert len(df) > 10_000
    assert df["chromosome"].str.startswith("chr").all()


def test_load_hg19():
    df = load_gene_positions(genome="hg19")
    assert len(df) > 10_000


def test_load_mm10():
    df = load_gene_positions(genome="mm10")
    assert len(df) > 10_000


def test_unknown_genome_raises():
    with pytest.raises(ValueError, match="unknown genome"):
        load_gene_positions(genome="bogus")
