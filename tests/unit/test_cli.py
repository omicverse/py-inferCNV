"""CLI tests using subprocess invocation."""
from __future__ import annotations

import subprocess
import sys


def test_cli_help():
    r = subprocess.run(
        [sys.executable, "-m", "pyinfercnv.cli", "--help"], capture_output=True, text=True
    )
    assert r.returncode == 0
    # typer flattens single-command apps — check for a known flag instead of command name
    assert "--input" in r.stdout or "run-h5ad" in r.stdout


def test_cli_run_h5ad_smoke(tmp_path, small_synthetic_adata):
    in_path = tmp_path / "in.h5ad"
    out_path = tmp_path / "out.h5ad"
    small_synthetic_adata.write_h5ad(in_path)
    r = subprocess.run(
        [
            sys.executable, "-m", "pyinfercnv.cli", "run-h5ad",
            "--input", str(in_path), "--output", str(out_path),
        ],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"stderr: {r.stderr}"
    from anndata import read_h5ad
    ad = read_h5ad(out_path)
    assert "X_cnv" in ad.obsm
