"""R-parity validation utilities — used by tests/test_r_parity.py."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def max_abs_diff(py: np.ndarray, r: np.ndarray) -> float:
    return float(np.max(np.abs(py.astype(np.float64) - r.astype(np.float64))))


def bit_exact_assert(py: np.ndarray, r: np.ndarray, *, tol: float = 1e-10) -> None:
    d = max_abs_diff(py, r)
    assert d < tol, f"max_diff={d:.3e} exceeds tol={tol:.3e}"


def approximate_assert(py: np.ndarray, r: np.ndarray, *, tol: float = 1e-6) -> None:
    d = max_abs_diff(py, r)
    assert d < tol, f"max_diff={d:.3e} exceeds tol={tol:.3e}"


def empirical_ari(py_labels: np.ndarray, r_labels: np.ndarray) -> float:
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError as e:
        raise ImportError(
            "empirical_ari requires scikit-learn. Install with: pip install 'pyinfercnv[dev]'"
        ) from e
    return float(adjusted_rand_score(py_labels, r_labels))


def empirical_jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 1.0
    return float(len(sa & sb) / len(union))
