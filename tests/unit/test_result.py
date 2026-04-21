"""Tests for pyinfercnv.result.InferCNVResult — including G1 P1/P2 schema."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from pyinfercnv.result import InferCNVResult


def _make_result(n_cells: int = 50, n_bins: int = 200) -> InferCNVResult:
    chr_pos = {"chr1": 0, "chr2": 100}
    cnv = np.zeros((n_cells, n_bins), dtype=np.float32)
    cnv_fc = np.ones((n_cells, n_bins), dtype=np.float32)
    cell_meta = pd.DataFrame(
        {"is_reference": [True] * (n_cells // 2) + [False] * (n_cells - n_cells // 2)}
    )
    return InferCNVResult(
        chr_pos=chr_pos, cnv_matrix=cnv, cnv_matrix_fc=cnv_fc, cell_meta=cell_meta
    )


def test_construction_with_defaults():
    res = _make_result()
    assert res.n_cells == 50
    assert res.n_bins == 200
    assert list(res.chromosomes) == ["chr1", "chr2"]
    assert res.subclusters is None
    assert res.hmm_states is None
    assert res.ref_counts_raw is None


def test_shape_mismatch_raises():
    chr_pos = {"chr1": 0}
    cnv = np.zeros((50, 100), dtype=np.float32)
    cnv_fc = np.ones((50, 200), dtype=np.float32)
    cell_meta = pd.DataFrame({"is_reference": [True] * 50})
    with pytest.raises(ValueError, match="shape"):
        InferCNVResult(chr_pos=chr_pos, cnv_matrix=cnv, cnv_matrix_fc=cnv_fc, cell_meta=cell_meta)


def test_missing_is_reference_raises():
    chr_pos = {"chr1": 0}
    cnv = np.zeros((50, 100), dtype=np.float32)
    cnv_fc = np.zeros((50, 100), dtype=np.float32)
    cell_meta = pd.DataFrame({"foo": [1] * 50})
    with pytest.raises(ValueError, match="is_reference"):
        InferCNVResult(chr_pos=chr_pos, cnv_matrix=cnv, cnv_matrix_fc=cnv_fc, cell_meta=cell_meta)


def test_write_to_anndata_phase1_roundtrip():
    res = _make_result(n_cells=30, n_bins=100)
    adata = AnnData(X=np.zeros((30, 500), dtype=np.float32))
    res.write_to_anndata(adata, key_added="cnv")
    assert "X_cnv" in adata.obsm
    assert adata.obsm["X_cnv"].shape == (30, 100)
    assert adata.uns["cnv"]["chr_pos"] == {"chr1": 0, "chr2": 100}


def test_write_to_anndata_phase2_schema_present():
    """G1 P1 — when Phase 2 fields are set, they persist into the expected slots."""
    res = _make_result(n_cells=30, n_bins=100)
    res.subclusters = np.arange(30, dtype=np.int32)
    res.hmm_states = np.zeros((30, 100), dtype=np.int8)
    res.hmm_states_i3 = np.zeros((30, 100), dtype=np.int8)

    adata = AnnData(X=np.zeros((30, 500), dtype=np.float32))
    res.write_to_anndata(adata, key_added="cnv")

    assert "cnv_subcluster" in adata.obs
    assert "X_cnv_hmm_states" in adata.obsm
    assert "X_cnv_hmm_states_i3" in adata.obsm
    assert adata.obsm["X_cnv_hmm_states"].shape == (30, 100)


def test_write_to_anndata_phase3_schema_present():
    """G1 P1 — Phase 3 cnv_regions + posterior_p_normal persist into uns."""
    res = _make_result(n_cells=30, n_bins=100)
    res.cnv_regions = pd.DataFrame({
        "cell_group": ["g1", "g2"],
        "chromosome": ["chr1", "chr2"],
        "start": [0, 100],
        "end": [99, 199],
        "state": [1, 5],
    })
    res.posterior_p_normal = pd.DataFrame({"region": ["r1", "r2"], "p": [0.1, 0.9]})

    adata = AnnData(X=np.zeros((30, 500), dtype=np.float32))
    res.write_to_anndata(adata, key_added="cnv")

    assert "cnv_regions" in adata.uns["cnv"]
    assert adata.uns["cnv"]["cnv_regions"].shape == (2, 5)
    assert "posterior_p_normal" in adata.uns["cnv"]


def test_ref_counts_raw_persisted():
    """G1 P2 — raw reference counts captured pre-normalize persist in uns."""
    res = _make_result(n_cells=30, n_bins=100)
    res.ref_counts_raw = np.ones((15, 500), dtype=np.float32)

    adata = AnnData(X=np.zeros((30, 500), dtype=np.float32))
    res.write_to_anndata(adata, key_added="cnv")

    assert "cnv_ref_counts_raw" in adata.uns
    assert adata.uns["cnv_ref_counts_raw"].shape == (15, 500)


def test_profile_persisted():
    res = _make_result(n_cells=30, n_bins=100)
    res.profile = {"filter": {"wallclock_s": 0.01, "rss_mb": 100.0}}
    adata = AnnData(X=np.zeros((30, 500), dtype=np.float32))
    res.write_to_anndata(adata, key_added="cnv")
    assert "profile" in adata.uns["cnv"]
