"""Reference-expression subtraction (bounded + mean paths).

R source:
    subtract_ref_expr_from_obs   - inferCNV_ops.R:1678-1702
    .get_normal_gene_mean_bounds - inferCNV_ops.R:1708-1735
    .subtract_expr               - inferCNV_ops.R:1742-1786

use_bounds=True (R default with K>1 reference groups):
    Compute per-group per-gene mean. Bounds = [min_k(mean_k), max_k(mean_k)].
    Values within bounds -> 0; above -> value - max; below -> value - min.

use_bounds=False (or K==1):
    Compute single grand-mean across all reference cells; simple subtraction.

R applies subtract on ALL cells (including reference cells themselves) per
inferCNV_ops.R:1742-1786 — reference cells therefore become close-to-zero
after subtraction.

Dtype policy (Phase 1 bit-exact, 2026-04-23): intermediate arithmetic is
float64 throughout. The pipeline downcasts to float32 at result assembly.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _compute_ref_group_means(X: np.ndarray, ref_groups: Mapping[str, Sequence[int]]) -> np.ndarray:
    """Return (K, n_genes) array of per-group per-gene means (float64)."""
    K = len(ref_groups)
    out = np.empty((K, X.shape[1]), dtype=np.float64)
    for k, idx in enumerate(ref_groups.values()):
        idx_arr = np.asarray(list(idx), dtype=np.int64)
        out[k] = X[idx_arr, :].mean(axis=0)
    return out


def subtract_reference(
    X: np.ndarray,
    *,
    ref_groups: Mapping[str, Sequence[int]],
    use_bounds: bool = True,
) -> np.ndarray:
    """Apply per-gene reference subtraction to all cells.

    Parameters
    ----------
    X
        Dense cells x genes array. Float64 preferred for bit-exact parity;
        float32 still accepted but loses the last 1e-10 of precision.
    ref_groups
        Mapping group_label -> list of cell row indices that form that ref group.
    use_bounds
        If True and len(ref_groups)>1, subtract per spec bounds rule. If False,
        subtract grand mean across all ref cells.
    """
    Xc = np.asarray(X, dtype=np.float64)

    if not ref_groups:
        raise ValueError("ref_groups must be non-empty")

    means = _compute_ref_group_means(Xc, ref_groups)

    if use_bounds and means.shape[0] > 1:
        bounds_min = means.min(axis=0)
        bounds_max = means.max(axis=0)
        out = np.zeros_like(Xc)
        above = Xc > bounds_max
        below = Xc < bounds_min
        out[above] = (Xc - bounds_max)[above]
        out[below] = (Xc - bounds_min)[below]
        return out

    grand_mean = means.mean(axis=0)
    return Xc - grand_mean
