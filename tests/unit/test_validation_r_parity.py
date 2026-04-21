"""Tests for pyinfercnv.validation.r_parity."""
from __future__ import annotations

import numpy as np
import pytest

from pyinfercnv.validation.r_parity import (
    bit_exact_assert,
    empirical_jaccard,
    max_abs_diff,
)


def test_max_abs_diff():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.1, 3.0])
    assert max_abs_diff(a, b) == pytest.approx(0.1)


def test_bit_exact_assert_pass():
    a = np.array([1.0])
    b = np.array([1.0 + 1e-12])
    bit_exact_assert(a, b, tol=1e-10)


def test_bit_exact_assert_fail():
    a = np.array([1.0])
    b = np.array([1.0 + 1e-6])
    with pytest.raises(AssertionError, match="max_diff"):
        bit_exact_assert(a, b, tol=1e-10)


def test_empirical_jaccard_sets():
    assert empirical_jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(2 / 4)


def test_empirical_jaccard_empty():
    assert empirical_jaccard([], []) == pytest.approx(1.0)


def test_empirical_ari_optional_skipped_if_no_sklearn():
    """ARI requires sklearn; smoke test if installed."""
    try:
        from pyinfercnv.validation.r_parity import empirical_ari
        labels_a = np.array([0, 0, 1, 1])
        labels_b = np.array([0, 0, 1, 1])
        assert empirical_ari(labels_a, labels_b) == pytest.approx(1.0)
    except ImportError:
        pytest.skip("sklearn not installed")
