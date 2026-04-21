"""Tests for pyinfercnv.io.annotation.read_gene_order_file."""
from __future__ import annotations

from pathlib import Path

import pytest

from pyinfercnv.io.annotation import read_gene_order_file


R_GENE_ORDER = Path(
    "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata/"
    "gencode_downsampled.EXAMPLE_ONLY_DONT_REUSE.txt"
)


def test_read_synthetic(tmp_path):
    fpath = tmp_path / "genes.txt"
    fpath.write_text("GENEA\tchr1\t100\t200\nGENEB\tchr2\t500\t700\nGENEC\tchrX\t10\t20\n")
    df = read_gene_order_file(fpath)
    assert list(df.columns) == ["gene_symbol", "chromosome", "start", "end"]
    assert len(df) == 3
    assert df.iloc[0]["gene_symbol"] == "GENEA"
    assert df.iloc[2]["chromosome"] == "chrX"


@pytest.mark.skipif(not R_GENE_ORDER.exists(), reason="R infercnv fixture not available")
def test_read_r_infercnv_fixture():
    df = read_gene_order_file(R_GENE_ORDER)
    assert len(df) > 1000
    assert df["chromosome"].str.startswith("chr").all()
    assert (df["end"] >= df["start"]).all()


def test_empty_file_raises(tmp_path):
    fpath = tmp_path / "empty.txt"
    fpath.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_gene_order_file(fpath)


def test_malformed_raises(tmp_path):
    fpath = tmp_path / "bad.txt"
    fpath.write_text("only_two_cols\tchr1\n")
    with pytest.raises(ValueError, match="4 columns"):
        read_gene_order_file(fpath)
