"""Outlier pruning — R `remove_outliers_norm` (step 16).

Tier 4 approximate (empirical floor max_diff < 1e-4) — not bit-exact due to
float32 accumulation in average_bounds reduction over 184 cells.

R `_get_average_bounds` formula (`inferCNV_ops.R:2734`):
    lower_bound = mean over cells of min(cell)
    upper_bound = mean over cells of max(cell)

Note R's `apply(expr_matrix, 2, fn)` iterates over COLUMNS (cells in R's
genes x cells layout). My Python uses (cells x genes) so columns become rows:
    lower_bound = X.min(axis=1).mean()  # per-cell min, then mean
    upper_bound = X.max(axis=1).mean()  # per-cell max, then mean

Asymmetric bounds — NOT a simple symmetric `[-bound, +bound]` clip.
"""
from __future__ import annotations

import numpy as np


def _average_bounds(X: np.ndarray) -> tuple[float, float]:
    """Per-cell min/max averaged over cells (R-parity)."""
    Xc = np.asarray(X, dtype=np.float64)
    lower = float(Xc.min(axis=1).mean())
    upper = float(Xc.max(axis=1).mean())
    return lower, upper


def prune_outliers(
    X: np.ndarray,
    *,
    method: str | None = "average_bound",
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> np.ndarray:
    """Bound-clip outliers per R remove_outliers_norm.

    Explicit `lower_bound`/`upper_bound` override `method`. With `method=None`
    and no explicit bounds, return a copy unchanged.
    """
    if lower_bound is not None or upper_bound is not None:
        lo = -np.inf if lower_bound is None else float(lower_bound)
        hi = np.inf if upper_bound is None else float(upper_bound)
        return np.clip(X, lo, hi).astype(np.float32)
    if method == "average_bound":
        lo, hi = _average_bounds(X)
        return np.clip(X, lo, hi).astype(np.float32)
    if method is None:
        return np.asarray(X, dtype=np.float32).copy()
    raise ValueError(f"unknown outlier method {method!r}")
