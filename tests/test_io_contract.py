"""Layout-contract tests (spec section 6.3 + RULES section 1.2 omicverse boundary)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from pyinfercnv import infercnv


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cnv_matrix_dtype_float32(small_synthetic_adata):
    result = infercnv(small_synthetic_adata, inplace=False)
    assert result.cnv_matrix.dtype == np.float32
    assert result.cnv_matrix_fc.dtype == np.float32


def test_no_omicverse_import_in_core():
    """Grep package src for forbidden 'import omicverse' / 'from omicverse'."""
    pkg_dir = REPO_ROOT / "pyinfercnv"
    cmd = [
        "grep", "-r", "-l", "-E", r"^\s*(import|from)\s+omicverse",
        str(pkg_dir),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        leaks = r.stdout.strip().splitlines()
        raise AssertionError(f"omicverse import leaked into core: {leaks}")
    assert r.returncode == 1
