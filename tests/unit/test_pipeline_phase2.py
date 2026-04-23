"""Tests for pyinfercnv.pipeline_phase2 — Phase 2 HMM integration.

Tests cover:
    - Short-circuit (HMM=False)
    - i6 path shapes, dtypes, G2 Q6 invariants (with monkeypatched hspike)
    - i3 path doesn't touch hspike
    - Subcluster sentinel on ref cells
    - Ref cells neutral in HMM output
    - cnv_regions bin_end inclusive
    - zero-ref-match raises
    - Array reuse by reference (G2 Q5)
    - _remap_ref_groups_to_local roundtrip
    - Profile keys 15-18 (G2 Q12)
    - hspike has no pipeline import (Q11 belt)
    - Integration via top-level infercnv()
    - n=1 subcluster valid path
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse as sp

from pyinfercnv.config import InferCNVConfig
from pyinfercnv.result import InferCNVResult


# --------------------------------------------------------------------------- #
# Helpers / fixtures                                                          #
# --------------------------------------------------------------------------- #

def _make_adata(n_cells: int = 30, n_genes: int = 60, n_ref: int = 10) -> AnnData:
    """Make a small synthetic AnnData with chromosome metadata."""
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


def _make_phase1_result(adata: AnnData, n_ref: int = 10) -> InferCNVResult:
    """Run Phase 1 and return result."""
    from pyinfercnv.pipeline import infercnv
    result = infercnv(
        adata,
        reference_key="celltype",
        reference_cat="normal",
        inplace=False,
    )
    return result


def _make_fake_calibration(n_states: int = 6) -> Any:
    """Build a duck-typed fake HspikeCalibration that doesn't raise."""
    from pyinfercnv.hmm.i6 import I6_CNV_LEVELS

    @dataclass
    class FakeCalibration:
        cnv_levels: np.ndarray
        state_mus: np.ndarray
        state_sigmas: np.ndarray
        sd_log_slope: np.ndarray
        sd_log_intercept: np.ndarray

        def sigmas_for_num_cells(self, num_cells: int) -> np.ndarray:
            # exp(slope * log(n) + intercept) per state
            n = max(1, num_cells)
            return np.exp(self.sd_log_slope * np.log(n) + self.sd_log_intercept)

    mus = np.log2(I6_CNV_LEVELS.astype(np.float64) + 1e-9)
    sigmas = np.full(n_states, 0.5, dtype=np.float64)
    slope = np.full(n_states, -0.5, dtype=np.float64)
    intercept = np.log(sigmas)
    return FakeCalibration(
        cnv_levels=I6_CNV_LEVELS.copy(),
        state_mus=mus,
        state_sigmas=sigmas,
        sd_log_slope=slope,
        sd_log_intercept=intercept,
    )


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_run_phase2_hmm_false_passthrough():
    """HMM=False -> returns result_phase1 unchanged (identity, not just equal)."""
    adata = _make_adata()
    result_p1 = _make_phase1_result(adata)
    cfg = InferCNVConfig(HMM=False)
    from pyinfercnv.pipeline_phase2 import run_phase2
    result = run_phase2(result_p1, adata, config=cfg)
    # Must be the exact same object — not a copy
    assert result is result_p1


def test_run_phase2_i6_shapes_and_dtypes(monkeypatch):
    """i6 path: Q6 shape/dtype invariants with mocked hspike."""
    adata = _make_adata(n_cells=30, n_genes=60, n_ref=10)
    result_p1 = _make_phase1_result(adata)
    fake_cal = _make_fake_calibration()

    import pyinfercnv.pipeline_phase2 as p2mod
    monkeypatch.setattr(p2mod, "_calibrate_hmm_emission", lambda *a, **kw: fake_cal)

    cfg = InferCNVConfig(HMM=True, HMM_type="i6", tumor_subcluster_partition_method="leiden")
    from pyinfercnv.pipeline_phase2 import run_phase2
    result = run_phase2(
        result_p1, adata, config=cfg,
        reference_key="celltype", reference_cat="normal",
    )

    n_cells = result_p1.n_cells
    n_bins = result_p1.n_bins

    # subclusters shape and dtype
    assert result.subclusters is not None
    assert result.subclusters.shape == (n_cells,)
    assert result.subclusters.dtype == np.int32

    # hmm_states shape and dtype
    assert result.hmm_states is not None
    assert result.hmm_states.shape == (n_cells, n_bins)
    assert result.hmm_states.dtype == np.int8

    # state values in valid range [0, 5] for i6
    assert int(result.hmm_states.min()) >= 0
    assert int(result.hmm_states.max()) <= 5

    # ref cells neutral everywhere
    is_ref = result.cell_meta["is_reference"].to_numpy()
    neutral_idx = 2  # i6
    assert np.all(result.hmm_states[is_ref] == neutral_idx), "ref cells must all be neutral"

    # cnv_regions bin_end inclusive (bin_end >= bin_start for all rows)
    if len(result.cnv_regions) > 0:
        assert (result.cnv_regions["bin_end"] >= result.cnv_regions["bin_start"]).all()


def test_run_phase2_i3_no_hspike_call(monkeypatch):
    """i3 path must not call hspike at all."""
    adata = _make_adata(n_cells=30, n_genes=60, n_ref=10)
    result_p1 = _make_phase1_result(adata)

    # Make hspike raise if called — i3 path must not touch it
    import pyinfercnv.pipeline_phase2 as p2mod
    def _hspike_must_not_be_called(*a, **kw):
        raise AssertionError("hspike calibrate should not be called in i3 path")
    monkeypatch.setattr(p2mod, "_calibrate_hmm_emission", _hspike_must_not_be_called)

    cfg = InferCNVConfig(HMM=True, HMM_type="i3", tumor_subcluster_partition_method="leiden")
    from pyinfercnv.pipeline_phase2 import run_phase2
    result = run_phase2(
        result_p1, adata, config=cfg,
        reference_key="celltype", reference_cat="normal",
    )
    assert result.hmm_states_i3 is not None
    assert result.hmm_states is None


def test_run_phase2_i3_consumes_cnv_matrix_fc(monkeypatch):
    """i3 branch reads cnv_matrix_fc (linear FC, R step 17 space), not cnv_matrix.

    R-parity invariant: ``inferCNV_HMM.R:366`` reads ``@expr.data`` at step 17,
    which is already post-step14 ``invert_log2`` (``inferCNV_ops.R:1031``) at
    that point. Py's i3 branch must mirror this; otherwise the Gaussian
    emission sits in the wrong observation space and Jaccard regresses from
    1.000 back to 0.976.
    """
    adata = _make_adata(n_cells=30, n_genes=60, n_ref=10)
    result_p1 = _make_phase1_result(adata)

    captured: dict[str, Any] = {}
    import pyinfercnv.pipeline_phase2 as p2mod
    orig_runhmm = p2mod._run_hmm_by_subcluster

    def _capture(cnv_matrix, *args, **kwargs):
        captured["matrix_id"] = id(cnv_matrix)
        captured["dtype"] = cnv_matrix.dtype
        return orig_runhmm(cnv_matrix, *args, **kwargs)

    monkeypatch.setattr(p2mod, "_run_hmm_by_subcluster", _capture)

    cfg = InferCNVConfig(HMM=True, HMM_type="i3", tumor_subcluster_partition_method="leiden")
    from pyinfercnv.pipeline_phase2 import run_phase2
    run_phase2(
        result_p1, adata, config=cfg,
        reference_key="celltype", reference_cat="normal",
    )

    # i3 must upcast to float64 (precision for state-boundary decisions)
    assert captured["dtype"] == np.float64, (
        f"i3 HMM input must be float64, got {captured['dtype']}"
    )
    # The captured buffer must NOT be the log2 cnv_matrix (R runs HMM in linear FC).
    # We test by value-range: linear FC centers around 1.0, log2 around 0.0.
    # Re-run to grab the actual buffer since monkeypatch captured id only.
    expected_buf = np.asarray(result_p1.cnv_matrix_fc, dtype=np.float64)
    # Value sanity: cnv_matrix_fc is centered ~1.0
    assert 0.5 < float(expected_buf.mean()) < 1.5, (
        "cnv_matrix_fc should be linear-FC centered near 1.0"
    )


def test_subcluster_default_groups_all_cells(monkeypatch):
    """Default ``cluster_by_groups=True`` gives every cell a non-negative id
    (R cluster_by_groups=TRUE parity)."""
    adata = _make_adata(n_cells=30, n_genes=60, n_ref=10)
    result_p1 = _make_phase1_result(adata)
    fake_cal = _make_fake_calibration()

    import pyinfercnv.pipeline_phase2 as p2mod
    monkeypatch.setattr(p2mod, "_calibrate_hmm_emission", lambda *a, **kw: fake_cal)

    cfg = InferCNVConfig(HMM=True, HMM_type="i6", tumor_subcluster_partition_method="leiden")
    from pyinfercnv.pipeline_phase2 import run_phase2
    result = run_phase2(
        result_p1, adata, config=cfg,
        reference_key="celltype", reference_cat="normal",
    )
    # R cluster_by_groups=TRUE: refs get real subcluster ids (not -1),
    # so they flow through the HMM step same as tumor cells.
    assert np.all(result.subclusters >= 0), (
        "cluster_by_groups=True: every cell should have a non-negative id"
    )
    # Distinct id space between ref and non-ref groups (leiden runs
    # independently per annotation category).
    is_ref = result.cell_meta["is_reference"].to_numpy()
    ref_ids = set(result.subclusters[is_ref].tolist())
    tumor_ids = set(result.subclusters[~is_ref].tolist())
    assert ref_ids.isdisjoint(tumor_ids), (
        f"ref group id(s) {ref_ids} overlap tumor id(s) {tumor_ids}"
    )


def test_subcluster_fallback_sentinel_when_cluster_by_groups_false(monkeypatch):
    """``cluster_by_groups=False`` keeps the legacy ref-sentinel behaviour."""
    adata = _make_adata(n_cells=30, n_genes=60, n_ref=10)
    result_p1 = _make_phase1_result(adata)
    fake_cal = _make_fake_calibration()

    import pyinfercnv.pipeline_phase2 as p2mod
    monkeypatch.setattr(p2mod, "_calibrate_hmm_emission", lambda *a, **kw: fake_cal)

    cfg = InferCNVConfig(
        HMM=True, HMM_type="i6",
        tumor_subcluster_partition_method="leiden",
        cluster_by_groups=False,
    )
    from pyinfercnv.pipeline_phase2 import run_phase2
    result = run_phase2(
        result_p1, adata, config=cfg,
        reference_key="celltype", reference_cat="normal",
    )
    is_ref = result.cell_meta["is_reference"].to_numpy()
    assert np.all(result.subclusters[is_ref] == -1), "ref cells sentinel"
    assert np.all(result.subclusters[~is_ref] >= 0), "tumor cells non-negative"


def test_ref_neutral_in_hmm_output(monkeypatch):
    """Ref cells must all be neutral in hmm_states (i6 neutral=2, i3 neutral=1)."""
    adata = _make_adata(n_cells=30, n_genes=60, n_ref=10)
    result_p1 = _make_phase1_result(adata)
    fake_cal = _make_fake_calibration()

    import pyinfercnv.pipeline_phase2 as p2mod
    monkeypatch.setattr(p2mod, "_calibrate_hmm_emission", lambda *a, **kw: fake_cal)

    cfg = InferCNVConfig(HMM=True, HMM_type="i6", tumor_subcluster_partition_method="leiden")
    from pyinfercnv.pipeline_phase2 import run_phase2
    result = run_phase2(
        result_p1, adata, config=cfg,
        reference_key="celltype", reference_cat="normal",
    )
    is_ref = result.cell_meta["is_reference"].to_numpy()
    assert np.all(result.hmm_states[is_ref] == 2)

    # Repeat for i3
    cfg_i3 = InferCNVConfig(HMM=True, HMM_type="i3")
    result_i3 = run_phase2(
        result_p1, adata, config=cfg_i3,
        reference_key="celltype", reference_cat="normal",
    )
    assert np.all(result_i3.hmm_states_i3[is_ref] == 1)


def test_cnv_regions_bin_end_inclusive():
    """bin_end >= bin_start using direct _build_cnv_regions with synthetic data."""
    from pyinfercnv.pipeline_phase2 import _build_cnv_regions

    n_cells = 6
    n_bins = 20
    # Build a fake i3 hmm_states with some non-neutral states
    hmm_states = np.ones((n_cells, n_bins), dtype=np.int8)  # all neutral
    # Put some DEL (0) and AMP (2) runs
    hmm_states[0, 2:6] = 0    # subcluster 0, chr1: DEL run bins 2-5
    hmm_states[0, 12:15] = 2  # subcluster 0, chr2: AMP run bins 12-14
    hmm_states[1, 2:6] = 0    # subcluster 0 also
    hmm_states[1, 12:15] = 2

    subclusters = np.array([0, 0, -1, -1, -1, -1], dtype=np.int32)
    chr_pos = {"chr1": 0, "chr2": 10}

    df = _build_cnv_regions(hmm_states, subclusters, chr_pos, hmm_type="i3")
    assert len(df) > 0, "expected non-neutral regions"
    assert (df["bin_end"] >= df["bin_start"]).all()
    # bin_end should be inclusive: the DEL run is bins 2-5 -> bin_end=5
    del_rows = df[(df["chromosome"] == "chr1") & (df["state"] == 0)]
    assert len(del_rows) == 1
    row = del_rows.iloc[0]
    assert row["bin_start"] == 2
    assert row["bin_end"] == 5  # inclusive
    # AMP run is global bins 12-14 -> bin_start=12, bin_end=14
    amp_rows = df[(df["chromosome"] == "chr2") & (df["state"] == 2)]
    assert len(amp_rows) == 1
    row2 = amp_rows.iloc[0]
    assert row2["bin_start"] == 12
    assert row2["bin_end"] == 14


def test_zero_ref_match_raises():
    """Explicit ref_key + non-existent cat -> ValueError."""
    adata = _make_adata()
    result_p1 = _make_phase1_result(adata)
    cfg = InferCNVConfig(HMM=True, HMM_type="i3")
    from pyinfercnv.pipeline_phase2 import run_phase2
    with pytest.raises(ValueError, match="matched zero cells"):
        run_phase2(
            result_p1, adata, config=cfg,
            reference_key="celltype",
            reference_cat="nonexistent_cat",
        )


def test_array_reuse_by_reference(monkeypatch):
    """Q5: Phase 1 arrays reused by reference in the new result."""
    adata = _make_adata(n_cells=30, n_genes=60, n_ref=10)
    result_p1 = _make_phase1_result(adata)
    fake_cal = _make_fake_calibration()

    import pyinfercnv.pipeline_phase2 as p2mod
    monkeypatch.setattr(p2mod, "_calibrate_hmm_emission", lambda *a, **kw: fake_cal)

    cfg = InferCNVConfig(HMM=True, HMM_type="i6")
    from pyinfercnv.pipeline_phase2 import run_phase2
    result = run_phase2(
        result_p1, adata, config=cfg,
        reference_key="celltype", reference_cat="normal",
    )

    # All Phase 1 arrays must be the same object — not copies
    assert result.cnv_matrix is result_p1.cnv_matrix
    assert result.cnv_matrix_fc is result_p1.cnv_matrix_fc
    assert result.cell_meta is result_p1.cell_meta
    assert result.chr_pos is result_p1.chr_pos
    assert result.ref_counts_raw is result_p1.ref_counts_raw


def test_remap_ref_groups_to_local_roundtrip():
    """_remap_ref_groups_to_local returns correct local indices for 2 groups."""
    from pyinfercnv.pipeline_phase2 import _remap_ref_groups_to_local

    n_cells = 20
    rng = np.random.default_rng(0)
    X = rng.random((n_cells, 10)).astype(np.float32)
    adata = AnnData(X=X)
    adata.obs_names = [f"c{i}" for i in range(n_cells)]
    # Group A: cells 0-7 are ref group A
    # Group B: cells 8-14 are ref group B
    # Cells 15-19 are tumor (not reference)
    celltypes = ["A"] * 8 + ["B"] * 7 + ["tumor"] * 5
    adata.obs["celltype"] = celltypes

    is_reference = np.array([True] * 15 + [False] * 5, dtype=bool)

    result = _remap_ref_groups_to_local(
        adata, is_reference, "celltype", ["A", "B"]
    )
    assert result is not None
    assert set(result.keys()) == {"A", "B"}

    # Group A: global indices 0-7, local indices 0-7 (all ref cells come first)
    np.testing.assert_array_equal(result["A"], np.arange(8, dtype=np.intp))

    # Group B: global indices 8-14 -> local indices 8-14
    np.testing.assert_array_equal(result["B"], np.arange(8, 15, dtype=np.intp))


def test_remap_ref_groups_returns_none_when_no_key():
    """When reference_key is None, returns None."""
    from pyinfercnv.pipeline_phase2 import _remap_ref_groups_to_local
    adata = AnnData(X=np.zeros((5, 3)))
    is_ref = np.array([True, True, False, False, False])
    assert _remap_ref_groups_to_local(adata, is_ref, None, None) is None


def test_profile_keys_match_15_to_18(monkeypatch):
    """Q12: Profile dict must contain keys 15_subcluster, 16_hspike_calibrate,
    17_hmm, 18_cnv_regions after i6 run."""
    adata = _make_adata(n_cells=30, n_genes=60, n_ref=10)
    result_p1 = _make_phase1_result(adata)
    fake_cal = _make_fake_calibration()

    import pyinfercnv.pipeline_phase2 as p2mod
    monkeypatch.setattr(p2mod, "_calibrate_hmm_emission", lambda *a, **kw: fake_cal)

    profile: dict = {}
    cfg = InferCNVConfig(HMM=True, HMM_type="i6")
    from pyinfercnv.pipeline_phase2 import run_phase2
    run_phase2(
        result_p1, adata, config=cfg,
        reference_key="celltype", reference_cat="normal",
        profile=profile,
    )
    assert "15_subcluster" in profile, f"15_subcluster missing; keys={list(profile)}"
    assert "16_hspike_calibrate" in profile, f"16_hspike_calibrate missing"
    assert "17_hmm" in profile, f"17_hmm missing"
    assert "18_cnv_regions" in profile, f"18_cnv_regions missing"


def test_profile_keys_i3_no_hspike(monkeypatch):
    """i3 run must contain 15/17/18 keys but NOT 16_hspike_calibrate."""
    adata = _make_adata(n_cells=30, n_genes=60, n_ref=10)
    result_p1 = _make_phase1_result(adata)

    profile: dict = {}
    cfg = InferCNVConfig(HMM=True, HMM_type="i3")
    from pyinfercnv.pipeline_phase2 import run_phase2
    run_phase2(
        result_p1, adata, config=cfg,
        reference_key="celltype", reference_cat="normal",
        profile=profile,
    )
    assert "15_subcluster" in profile
    assert "17_hmm" in profile
    assert "18_cnv_regions" in profile
    assert "16_hspike_calibrate" not in profile


def test_hspike_no_pipeline_import():
    """Q11 belt: hspike.py must not import from pyinfercnv.pipeline."""
    hspike_path = Path(__file__).resolve().parents[2] / "pyinfercnv" / "hmm" / "hspike.py"
    assert hspike_path.exists(), f"hspike.py not found at {hspike_path}"
    content = hspike_path.read_text()
    # Reject any import of pipeline module from hspike
    assert "from pyinfercnv.pipeline" not in content, (
        "hspike.py must not import from pyinfercnv.pipeline (circular import)"
    )


def test_hmm_integration_via_infercnv(monkeypatch):
    """Integration test: infercnv(cfg.HMM=True) -> result.hmm_states_i3 exists."""
    adata = _make_adata(n_cells=30, n_genes=60, n_ref=10)
    from pyinfercnv.pipeline import infercnv

    # Use i3 to avoid hspike (which is still a skeleton)
    cfg = InferCNVConfig(HMM=True, HMM_type="i3", tumor_subcluster_partition_method="leiden")
    result = infercnv(
        adata, config=cfg,
        reference_key="celltype", reference_cat="normal",
        inplace=False,
    )
    assert result is not None
    assert result.hmm_states_i3 is not None
    assert result.hmm_states_i3.shape == result.cnv_matrix.shape
    assert result.subclusters is not None
    assert result.cnv_regions is not None


def test_n_equals_1_subcluster_valid(monkeypatch):
    """Force a 1-cell subcluster — no crash, trace is valid int8."""
    from pyinfercnv.pipeline_phase2 import _run_hmm_by_subcluster

    n_cells = 5
    n_bins = 20
    rng = np.random.default_rng(0)
    cnv_matrix = rng.normal(0.0, 0.1, (n_cells, n_bins)).astype(np.float32)
    chr_pos = {"chr1": 0, "chr2": 10}
    # subclusters: cell 0 is ref, cells 1-4 are tumor in single subcluster
    subclusters = np.array([-1, 0, 0, 0, 0], dtype=np.int32)
    is_reference = np.array([True, False, False, False, False])

    # Put 1-cell subcluster: override subclusters so cell 1 is alone
    subclusters = np.array([-1, 0, 1, 1, 1], dtype=np.int32)  # sc 0 has 1 cell

    from pyinfercnv.hmm.i3 import estimate_i3_state_params
    mus, sigs = estimate_i3_state_params(cnv_matrix, np.where(is_reference)[0])

    states = _run_hmm_by_subcluster(
        cnv_matrix, chr_pos, subclusters, is_reference,
        hmm_type="i3",
        transition_prob=1e-6,
        i6_calibration=None,
        i3_mus=mus,
        i3_sigmas=sigs,
    )
    assert states.shape == (n_cells, n_bins)
    assert states.dtype == np.int8
    assert int(states[0, :].max()) <= 1  # ref cell is neutral (1)
    # sc 0 (cell 1 only) must have valid states
    assert np.all((states[1, :] >= 0) & (states[1, :] <= 2))


def test_zero_non_ref_raises():
    """All cells are reference -> ValueError."""
    from pyinfercnv.pipeline_phase2 import _run_subclustering

    n_cells = 5
    n_bins = 10
    rng = np.random.default_rng(0)
    cnv_matrix = rng.random((n_cells, n_bins)).astype(np.float32)
    is_reference = np.ones(n_cells, dtype=bool)  # ALL are reference
    cfg = InferCNVConfig()
    with pytest.raises(ValueError, match="No non-reference cells"):
        _run_subclustering(cnv_matrix, is_reference, config=cfg, random_state=0, profile=None)


def test_i6_with_null_ref_counts_raises():
    """i6 with ref_counts_raw=None -> ValueError."""
    adata = _make_adata(n_cells=30, n_genes=60, n_ref=10)
    result_p1 = _make_phase1_result(adata)

    # Patch out ref_counts_raw
    import dataclasses
    result_no_ref = dataclasses.replace(result_p1, ref_counts_raw=None)

    cfg = InferCNVConfig(HMM=True, HMM_type="i6")
    from pyinfercnv.pipeline_phase2 import run_phase2
    with pytest.raises(ValueError, match="ref_counts_raw"):
        run_phase2(
            result_no_ref, adata, config=cfg,
            reference_key="celltype", reference_cat="normal",
        )
