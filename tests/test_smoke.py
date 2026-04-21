"""End-to-end sub-second smoke test (no R, no omicverse)."""
from __future__ import annotations

from pyinfercnv import InferCNVConfig, infercnv


def test_subsecond_e2e(small_synthetic_adata):
    cfg = InferCNVConfig(window_length=11, cutoff=0.0, min_cells_per_gene=0)
    infercnv(small_synthetic_adata, config=cfg, inplace=True)
    assert "X_cnv" in small_synthetic_adata.obsm
