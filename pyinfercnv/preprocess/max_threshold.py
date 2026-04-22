"""Apply max-centered threshold clip — R parity with `apply_max_threshold_bounds`
(`inferCNV_ops.R:1830-1870`).

R semantics: clip values to [-threshold, +threshold]. Used at step 9 with
threshold=3.0. If threshold is 'auto', use R's heuristic (max abs across
reference distribution); not yet implemented (raises).
"""
from __future__ import annotations

import numpy as np


def apply_max_centered_threshold(
    X: np.ndarray, *, threshold: float | str | None = 3.0
) -> np.ndarray:
    """Clip X to [-threshold, +threshold]. Pass threshold=None to disable."""
    if threshold is None:
        return np.asarray(X, dtype=np.float64)
    if isinstance(threshold, str):
        if threshold == "auto":
            raise NotImplementedError("threshold='auto' not implemented in Phase 1")
        raise ValueError(f"threshold must be numeric, 'auto', or None; got {threshold!r}")
    t = float(threshold)
    if t <= 0:
        raise ValueError(f"threshold must be > 0, got {t}")
    Xc = np.asarray(X, dtype=np.float64)
    return np.clip(Xc, -t, t)
