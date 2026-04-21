"""Tests for pyinfercnv.subcluster.leiden."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyinfercnv.subcluster import leiden_subcluster


def _synthetic_cnv_matrix(n_cells: int, n_genes: int = 120, seed: int = 0) -> np.ndarray:
    """Two well-separated blobs in log2 FC space."""
    rng = np.random.default_rng(seed)
    half = n_cells // 2
    block_a = rng.normal(loc=-0.3, scale=0.05, size=(half, n_genes)).astype(np.float32)
    block_b = rng.normal(loc=+0.3, scale=0.05, size=(n_cells - half, n_genes)).astype(np.float32)
    return np.vstack([block_a, block_b])


def test_shape_and_dtype():
    X = _synthetic_cnv_matrix(80)
    labels = leiden_subcluster(X, resolution=1.0, k_nn=15, random_state=0)
    assert labels.shape == (80,)
    assert labels.dtype == np.int32


def test_deterministic_with_fixed_seed():
    X = _synthetic_cnv_matrix(80)
    a = leiden_subcluster(X, resolution=1.0, k_nn=15, random_state=42)
    b = leiden_subcluster(X, resolution=1.0, k_nn=15, random_state=42)
    np.testing.assert_array_equal(a, b)


def _ambiguous_cnv_matrix(n_cells: int = 60, n_genes: int = 5,
                          seed: int = 0) -> np.ndarray:
    """A low-signal matrix where Leiden's starting state genuinely
    matters — no clear cluster structure, so different seeds can
    converge to different local optima. Used to exercise the RNG path
    that the per-call ``random_state`` is supposed to drive. Matches
    codex's probe geometry (60 × 5, ``resolution=0.5``).
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(n_cells, n_genes)).astype(np.float32)


def test_seed_is_reproducible_on_ambiguous_graph():
    """Same seed, same run — must return identical labels even when
    Leiden has genuine choice to make. Regression for the bug codex
    caught where ``random_state`` was silently dropped after the switch
    to ``python-igraph``."""
    X = _ambiguous_cnv_matrix()
    a = leiden_subcluster(X, resolution=0.5, k_nn=10, random_state=123)
    b = leiden_subcluster(X, resolution=0.5, k_nn=10, random_state=123)
    np.testing.assert_array_equal(a, b)


def test_seed_actually_threads_through_to_leiden():
    """At least two different seeds on the ambiguous fixture produce
    DIFFERENT partitions. If this ever starts passing trivially with
    the seed fix deleted, it means igraph's module-global RNG is no
    longer reachable and the ``random_state`` kwarg has been silently
    disconnected again (codex's CRITICAL finding)."""
    X = _ambiguous_cnv_matrix()
    partitions = [
        leiden_subcluster(X, resolution=0.5, k_nn=10, random_state=seed)
        for seed in (1, 2, 3, 4, 5, 6)
    ]
    # Heuristic: at least one pair must differ. Codex saw ARI
    # 0.6906 to 1.0 across six runs on the exact same geometry, so
    # on average many pairs differ; we only need one to prove wire-up.
    some_differ = any(
        not np.array_equal(partitions[i], partitions[j])
        for i in range(len(partitions))
        for j in range(i + 1, len(partitions))
    )
    assert some_differ, (
        "All six seeds produced identical partitions on the ambiguous "
        "60×5 fixture — the random_state kwarg is probably not reaching "
        "igraph's global RNG."
    )


def test_auto_resolution_runs():
    X = _synthetic_cnv_matrix(100)
    labels = leiden_subcluster(X, resolution="auto", k_nn=15, random_state=0)
    assert labels.shape == (100,)
    # auto resolution is tiny ((11.98/100)**(1/1.165) ≈ 0.08) → expect coarse
    # partition; just confirm at least one cluster exists.
    assert labels.min() >= 0


def test_too_few_cells_returns_zeros():
    X = np.random.default_rng(0).normal(size=(2, 10)).astype(np.float32)
    labels = leiden_subcluster(X, resolution=1.0, k_nn=5, random_state=0)
    assert labels.shape == (2,)
    assert np.all(labels == 0)


def test_k_nn_geq_ncells_returns_single_cluster():
    X = _synthetic_cnv_matrix(10)
    labels = leiden_subcluster(X, resolution=1.0, k_nn=15, random_state=0)
    assert labels.shape == (10,)
    assert np.all(labels == 0)


def test_invalid_resolution_string_raises():
    X = _synthetic_cnv_matrix(40)
    with pytest.raises(ValueError, match="resolution"):
        leiden_subcluster(X, resolution="bogus", k_nn=10, random_state=0)


def test_no_omicverse_import_in_subcluster_package():
    """No pyinfercnv/subcluster/*.py may import omicverse."""
    import pyinfercnv.subcluster as mod

    pkg_dir = Path(mod.__file__).resolve().parent
    for py in pkg_dir.glob("*.py"):
        text = py.read_text()
        assert "import omicverse" not in text, f"omicverse import in {py.name}"
        assert "from omicverse" not in text, f"omicverse import in {py.name}"
