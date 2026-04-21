"""End-to-end pipeline smoke tests."""
from __future__ import annotations

import numpy as np

from pyinfercnv import InferCNVConfig, infercnv


def test_end_to_end_synthetic(small_synthetic_adata):
    result = infercnv(small_synthetic_adata, inplace=False)
    assert result is not None
    assert result.cnv_matrix.shape[0] == small_synthetic_adata.n_obs
    assert result.cnv_matrix.dtype == np.float32
    assert result.cnv_matrix_fc.dtype == np.float32
    assert len(result.chr_pos) > 0
    assert "01_extract" in result.profile
    assert result.ref_counts_raw is not None  # G1 P2


def test_inplace_writes_obsm(small_synthetic_adata):
    infercnv(small_synthetic_adata, inplace=True)
    assert "X_cnv" in small_synthetic_adata.obsm
    assert "cnv" in small_synthetic_adata.uns
    assert "chr_pos" in small_synthetic_adata.uns["cnv"]
    assert "profile" in small_synthetic_adata.uns["cnv"]


def test_explicit_reference(small_synthetic_adata):
    """Pass reference_key + reference_cat — pipeline should still complete."""
    small_synthetic_adata.obs["celltype"] = ["normal"] * 50 + ["tumor"] * 50
    result = infercnv(
        small_synthetic_adata,
        reference_key="celltype",
        reference_cat="normal",
        inplace=False,
    )
    assert result is not None
    assert result.cell_meta["is_reference"].sum() == 50


def test_with_outlier_prune_enabled(small_synthetic_adata):
    cfg = InferCNVConfig(prune_outliers=True)
    result = infercnv(small_synthetic_adata, config=cfg, inplace=False)
    assert "13_outlier_prune" in result.profile
