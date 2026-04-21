"""Unit tests for scripts/generate_gene_positions.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_gene_positions.py"


def test_parse_gtf_stub(tmp_path):
    gtf = tmp_path / "mini.gtf"
    gtf.write_text(
        '##description: mini test\n'
        'chr1\tHAVANA\tgene\t11869\t14409\t.\t+\t.\tgene_id "ENSG00000223972"; gene_name "DDX11L1"; gene_type "transcribed";\n'
        'chr1\tHAVANA\tgene\t14404\t29570\t.\t-\t.\tgene_id "ENSG00000227232"; gene_name "WASH7P"; gene_type "unprocessed";\n'
        'chrX\tHAVANA\tgene\t100000\t200000\t.\t+\t.\tgene_id "ENSG00000999999"; gene_name "TESTGX"; gene_type "protein_coding";\n'
    )
    out = tmp_path / "out.parquet"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--gtf", str(gtf), "--output", str(out), "--genome", "hg38_test"],
        check=True,
        capture_output=True,
    )
    df = pd.read_parquet(out)
    assert list(df.columns) == ["gene_symbol", "chromosome", "start", "end"]
    assert len(df) == 3
    assert set(df["gene_symbol"]) == {"DDX11L1", "WASH7P", "TESTGX"}
    assert df[df["gene_symbol"] == "DDX11L1"]["chromosome"].iloc[0] == "chr1"


def test_deduplicates_by_gene_symbol(tmp_path):
    gtf = tmp_path / "dup.gtf"
    gtf.write_text(
        'chr1\tHAVANA\tgene\t100\t200\t.\t+\t.\tgene_name "DUP"; gene_type "x";\n'
        'chr1\tHAVANA\tgene\t500\t600\t.\t+\t.\tgene_name "DUP"; gene_type "x";\n'
        'chr2\tHAVANA\tgene\t1000\t2000\t.\t+\t.\tgene_name "UNIQ"; gene_type "x";\n'
    )
    out = tmp_path / "dup.parquet"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--gtf", str(gtf), "--output", str(out), "--genome", "t"],
        check=True,
        capture_output=True,
    )
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert df[df["gene_symbol"] == "DUP"]["start"].iloc[0] == 100
