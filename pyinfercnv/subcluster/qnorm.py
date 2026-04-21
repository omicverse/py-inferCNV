"""qnorm tumor subclustering — deterministic hierarchical cut.

R parity target: `.single_tumor_subclustering(..., partition_method='qnorm')`
in `R/inferCNV_tumor_subclusters.R:181-268`.

Algorithm (R lines 206-214):
    hc          = hclust(parallelDist(t(expr)), method=hclust_method)    # ward.D2
    heights     = hc$height
    mu          = mean(heights)
    sigma       = sd(heights)                                            # n-1 denom
    cut_height  = qnorm(p=1-p_val, mean=mu, sd=sigma)
    grps        = cutree(hc, h=cut_height)

Deterministic — no RNG path; labels are a pure function of `cnv_matrix`.

scipy vs R linkage note:
    R `hclust(dist, method="ward.D2")` ↔ `scipy.cluster.hierarchy.linkage(X,
    method="ward")` applied to the **raw observation matrix** (rows =
    observations). We pass (n_cells, n_genes) directly — linkage treats rows
    as observations and computes euclidean internally, matching ward.D2.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import norm

__all__ = ["qnorm_subcluster"]


def qnorm_subcluster(
    cnv_matrix: np.ndarray,
    *,
    tumor_subcluster_pval: float = 0.1,
    hclust_method: str = "ward",  # noqa: N803 (R kwarg name preserved)
) -> np.ndarray:
    """Return per-cell int32 subcluster label, shape (n_cells,).

    Parameters
    ----------
    cnv_matrix
        (n_cells, n_genes) CNV log2 fold-change matrix.
    tumor_subcluster_pval
        Tail p-value for the qnorm cut. Matches R `p_val` (default 0.1).
    hclust_method
        Linkage method passed to scipy. "ward" corresponds to R `ward.D2`
        when applied to the raw observation matrix.

    Returns
    -------
    labels : np.ndarray of shape (n_cells,), dtype int32
        1-based cluster labels (matches R `cutree` output convention).
    """
    X = np.ascontiguousarray(cnv_matrix, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"cnv_matrix must be 2D (n_cells, n_genes); got shape {X.shape}")
    n_cells = X.shape[0]

    # R line 189: if (ncol(tumor_expr_data) > 2) — else single cluster.
    if n_cells <= 2:
        return np.ones(n_cells, dtype=np.int32)

    # linkage operates on rows as observations; pass directly.
    Z = linkage(X.astype(np.float64, copy=False), method=hclust_method)
    heights = Z[:, 2]  # scipy linkage: column 2 is the merge distance (= hc$height)

    if heights.size == 0:
        return np.ones(n_cells, dtype=np.int32)

    mu = float(np.mean(heights))
    # R's sd() uses (n-1) denominator — numpy ddof=1 matches.
    sigma = float(np.std(heights, ddof=1)) if heights.size > 1 else 0.0

    if sigma == 0.0:
        # Degenerate: all merge heights equal → R's qnorm returns mu, cutree
        # at h == mu typically gives 1 cluster. Fall back explicitly.
        return np.ones(n_cells, dtype=np.int32)

    cut_height = float(norm.ppf(1.0 - tumor_subcluster_pval, loc=mu, scale=sigma))

    # R cutree(hc, h=cut_height): flat clusters at threshold. scipy fcluster
    # with criterion='distance' is the direct analogue.
    labels = fcluster(Z, t=cut_height, criterion="distance").astype(np.int32)
    return labels
