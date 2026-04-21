"""Validate `python -m build` produces py3-none-any wheel with no binary extensions.

Skips if `dist/` is empty — run `uv run python -m build` first.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"


def _wheel_path() -> Path | None:
    if not DIST_DIR.exists():
        return None
    cands = sorted(DIST_DIR.glob("pyinfercnv-*-py3-none-any.whl"))
    return cands[-1] if cands else None


def test_wheel_exists_and_is_pure_python():
    wheel = _wheel_path()
    if wheel is None:
        pytest.skip("dist/*.whl not found; run `uv run python -m build` first")
    assert "py3-none-any" in wheel.name, f"wheel name lacks py3-none-any: {wheel.name}"


def test_wheel_no_binary_extensions():
    wheel = _wheel_path()
    if wheel is None:
        pytest.skip("dist/*.whl not found; run `uv run python -m build` first")
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
    bad = [n for n in names if n.endswith((".so", ".pyd", ".dylib"))]
    assert not bad, f"wheel contains binary extensions: {bad}"


def test_wheel_contains_gene_position_parquets():
    wheel = _wheel_path()
    if wheel is None:
        pytest.skip("dist/*.whl not found; run `uv run python -m build` first")
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
    for genome in ("hg38", "hg19", "mm10"):
        target = f"pyinfercnv/data/{genome}_gene_positions.parquet"
        assert target in names, f"missing bundled gene parquet: {target}"
