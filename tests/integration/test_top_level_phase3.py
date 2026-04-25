"""Permanent regression test for Phase 3 wired into top-level ``infercnv()``.

Without this test, a future commit can silently re-disable the Phase 3
hand-off by deleting or reordering the ``run_phase3`` call in
``pyinfercnv/pipeline.py``; the rest of the suite would stay green because
``run_phase3`` is exercised directly elsewhere.

Hspike calibration is monkey-patched (mirrors
``tests/unit/test_pipeline_phase2.py``) so the test stays fast on a
30-cell × 60-gene synthetic AnnData; Phase 3 runs against real
``cnv_matrix_fc`` + HMM state output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse as sp

from pyinfercnv import InferCNVConfig, infercnv


def _make_adata(n_cells: int = 30, n_genes: int = 60, n_ref: int = 10) -> AnnData:
    rng = np.random.default_rng(42)
    X = rng.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes)).astype(np.float32)
    adata = AnnData(X=sp.csr_matrix(X))
    adata.layers["counts"] = sp.csr_matrix(X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    n_chroms = 4
    genes_per_chr = n_genes // n_chroms
    adata.var["chromosome"] = [f"chr{1 + (i // genes_per_chr)}" for i in range(n_genes)]
    adata.var["start"] = np.arange(n_genes) * 1000
    adata.var["end"] = np.arange(n_genes) * 1000 + 500
    adata.obs["celltype"] = ["normal"] * n_ref + ["tumor"] * (n_cells - n_ref)
    return adata


def _make_fake_calibration() -> Any:
    """Mirror of ``tests/unit/test_pipeline_phase2.py::_make_fake_calibration``."""
    from pyinfercnv.hmm.i6 import I6_CNV_LEVELS

    @dataclass
    class FakeCalibration:
        cnv_levels: np.ndarray
        state_mus: np.ndarray
        state_sigmas: np.ndarray
        sd_log_slope: np.ndarray
        sd_log_intercept: np.ndarray

        def sigmas_for_num_cells(self, num_cells: int) -> np.ndarray:
            n = max(1, num_cells)
            return np.exp(self.sd_log_slope * np.log(n) + self.sd_log_intercept)

    sigmas = np.full(6, 0.3, dtype=np.float64)
    return FakeCalibration(
        cnv_levels=I6_CNV_LEVELS.copy(),
        state_mus=I6_CNV_LEVELS.astype(np.float64).copy(),
        state_sigmas=sigmas,
        sd_log_slope=np.full(6, -0.5, dtype=np.float64),
        sd_log_intercept=np.log(sigmas),
    )


def test_top_level_invokes_phase3_when_hmm_and_denoise_on(monkeypatch):
    """``infercnv(HMM=True, denoise=True)`` must populate Phase 3 fields.

    Locks the wire from ``pipeline.py`` Phase 2 → Phase 3. If ``run_phase3``
    is not invoked, ``denoised_matrix`` and ``hmm_proxy_matrix`` stay None
    and this test fails loudly.
    """
    adata = _make_adata()

    import pyinfercnv.pipeline_phase2 as p2mod
    monkeypatch.setattr(
        p2mod, "_calibrate_hmm_emission", lambda *a, **kw: _make_fake_calibration()
    )

    cfg = InferCNVConfig(
        cutoff=0.0,
        min_cells_per_gene=0,
        HMM=True,
        HMM_type="i6",
        BayesMaxPNormal=0.0,
        mask_nonDE_genes=False,
        denoise=True,
        reassignCNVs=False,
        random_state=0,
    )

    result = infercnv(
        adata, config=cfg,
        reference_key="celltype", reference_cat="normal",
        inplace=False,
    )

    assert result is not None
    # P1.1 wire contract: Phase 3 actually ran.
    assert result.denoised_matrix is not None, "denoise=True but denoised_matrix not set"
    assert result.hmm_proxy_matrix is not None, "HMM=True but hmm_proxy_matrix not set"
    # Toggles that were off must NOT populate their fields.
    assert result.bayes_posterior is None, "BayesMaxPNormal=0 but bayes_posterior set"
    assert result.de_mask is None, "mask_nonDE_genes=False but de_mask set"
    # Step 20 contract: proxy matrix shape matches HMM state matrix.
    assert result.hmm_proxy_matrix.shape == result.hmm_states.shape
    assert result.hmm_proxy_matrix.dtype == np.float64
    assert result.denoised_matrix.shape == result.cnv_matrix_fc.shape


def test_top_level_skips_phase3_when_all_off():
    """``infercnv(HMM=False, denoise=False, mask_nonDE_genes=False, BayesMaxPNormal=0)``
    must leave every Phase 3 field None — Phase 3 should be a no-op and
    ``run_phase3`` should not even be invoked (the orchestrator detects this
    via the ``phase3_on`` gate)."""
    adata = _make_adata()
    cfg = InferCNVConfig(
        cutoff=0.0,
        min_cells_per_gene=0,
        HMM=False,
        BayesMaxPNormal=0.0,
        mask_nonDE_genes=False,
        denoise=False,
        random_state=0,
    )

    result = infercnv(
        adata, config=cfg,
        reference_key="celltype", reference_cat="normal",
        inplace=False,
    )

    assert result is not None
    assert result.bayes_posterior is None
    assert result.de_mask is None
    assert result.denoised_matrix is None
    assert result.hmm_proxy_matrix is None


def test_top_level_bayes_requires_hmm_raises():
    """Top-level guard: ``BayesMaxPNormal>0`` without HMM must raise loudly,
    not silently no-op."""
    adata = _make_adata()
    cfg = InferCNVConfig(
        cutoff=0.0,
        min_cells_per_gene=0,
        HMM=False,
        BayesMaxPNormal=0.5,
        reassignCNVs=False,
        random_state=0,
    )
    with pytest.raises(ValueError, match=r"BayesMaxPNormal>0 requires HMM=True"):
        infercnv(
            adata, config=cfg,
            reference_key="celltype", reference_cat="normal",
            inplace=False,
        )
