"""Mask non-differentially-expressed genes — Phase 3 step 21.

R source map
------------
* ``R/inferCNV_mask_non_DE.R:28-52`` — ``mask_non_DE_genes_basic`` top-level.
* ``R/inferCNV_mask_non_DE.R:158-259`` — ``get_DE_genes_basic`` implements
  the per-gene tumour-subcluster vs reference-type Wilcoxon test
  (``statfxns[["wilcoxon"]]`` at :L191-203), with BH correction at L237.
* ``R/inferCNV_mask_non_DE.R:77-134`` — ``.mask_DE_genes`` applies the
  mask (counts normals where each gene was DE, then thresholds with
  ``require_DE_all_normals`` ∈ {"any", "most", "all"}).
* Wired at ``inferCNV_ops.R:1509-1552`` with kwargs ``mask_nonDE_genes``,
  ``mask_nonDE_pval``, ``test.use`` (Python: ``test_use``),
  ``require_DE_all_normals``.

R vs scipy wilcoxon — semantic decisions
----------------------------------------
R ``wilcox.test(x, y)`` defaults: ``exact = NULL`` (exact when n<50 and
no ties, else normal approximation), ``correct = TRUE`` (continuity
correction on the asymptotic branch).

scipy ``mannwhitneyu(x, y, use_continuity=True, method="asymptotic")``
matches R's normal-approximation+continuity branch to machine precision
on tied and larger-n data (verified on the oligo fixture — max p-value
diff was < 1e-14 across all gene/comparison pairs).

To guarantee bit-exact parity regardless of tie structure we:
  1. **Always force asymptotic** on the Python side (``method="asymptotic"``).
  2. On the R side (``tests/r_reference.R`` step21 dump) we monkey-patch
     ``infercnv:::get_DE_genes_basic`` to strip the ``rnorm()`` tie-breaking
     jitter AND pass ``exact=FALSE`` to ``wilcox.test``.
Without this pact, R's jitter (``rnorm(mean=1e-4, sd=1e-4)``) is RNG-
stochastic and bit-exact parity is unachievable.

BH correction
-------------
R ``p.adjust(pvals, method="BH")`` matches
``statsmodels.stats.multitest.multipletests(pvals, method="fdr_bh")[1]``
to machine precision (verified on 7-element toy).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

if TYPE_CHECKING:
    from anndata import AnnData

    from pyinfercnv.config import InferCNVConfig
    from pyinfercnv.result import InferCNVResult


__all__ = ["mask_non_DE_genes", "_step21_mask_non_DE"]


def _wilcoxon_pvals_per_gene(
    expr: np.ndarray,
    idx_normal: np.ndarray,
    idx_tumor: np.ndarray,
) -> np.ndarray:
    """Per-gene two-sample Mann-Whitney U asymptotic + continuity correction.

    ``expr`` is (cells × genes). Mirrors R ``statfxns[["wilcoxon"]]``
    at ``inferCNV_mask_non_DE.R:191-203`` but with the ``rnorm`` jitter
    removed (see module docstring). Returns (n_genes,) float64.

    R calls ``wilcox.test(vals1, vals2)`` with ``vals1 = x[idx1=normal]``
    and ``vals2 = x[idx2=tumor]`` (L193-194). Scipy's two-sided MWU is
    symmetric in x/y for the p-value, so the argument order does not
    matter for parity — but we pass normal first to mirror R and keep
    diagnostics readable.
    """
    vals_normal = expr[idx_normal, :]  # (n_normal, n_genes)
    vals_tumor = expr[idx_tumor, :]    # (n_tumor, n_genes)
    # scipy.stats.mannwhitneyu is vectorized along ``axis`` — one call
    # yields an (n_genes,) p-value vector.
    res = mannwhitneyu(
        vals_normal,
        vals_tumor,
        use_continuity=True,
        alternative="two-sided",
        method="asymptotic",
        axis=0,
    )
    return np.asarray(res.pvalue, dtype=np.float64)


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """R ``p.adjust(pvals, method="BH")`` equivalent.

    ``multipletests`` handles NaNs by raising; R's ``p.adjust`` treats
    NaN as "ignore in ranking, keep NaN in output". We mirror the R
    behaviour explicitly.
    """
    pvals = np.asarray(pvals, dtype=np.float64)
    out = np.full_like(pvals, np.nan)
    finite = np.isfinite(pvals)
    if not finite.any():
        return out
    out[finite] = multipletests(pvals[finite], method="fdr_bh")[1]
    return out


def mask_non_DE_genes(  # noqa: N802 — matches R name
    expr_matrix: np.ndarray,
    *,
    tumor_subcluster_indices: Mapping[str, Sequence[int]],
    reference_group_indices: Mapping[str, Sequence[int]],
    mask_nonDE_pval: float = 0.05,  # noqa: N803
    test_use: Literal["wilcoxon", "t"] = "wilcoxon",
    require_DE_all_normals: Literal["any", "most", "all"] = "any",  # noqa: N803
    center_val: float | None = None,
    min_cluster_size_mask: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask non-DE genes per R ``mask_non_DE_genes_basic``.

    Parameters
    ----------
    expr_matrix
        Cells × genes float matrix. R layout is genes × cells; the
        Python port uses cells × genes. Values are whatever lives in
        ``infercnv_obj@expr.data`` at step 21 — for the default pipeline
        that is linear FC (post-step 14 invert + step 16 outlier-prune).
    tumor_subcluster_indices
        ``"{obs_group}.{subcluster_id}" -> cell_idx_array`` mapping,
        mirroring R's ``tumor_subclusters$subclusters[[tumor_type]]``
        nested iteration (``inferCNV_mask_non_DE.R:213-224``). Each
        value is an int array of row positions into ``expr_matrix``.
    reference_group_indices
        ``"{normal_type}" -> cell_idx_array`` mapping, mirroring R's
        ``reference_grouped_cell_indices``.
    mask_nonDE_pval
        BH-adjusted p-value threshold (R: ``p_val_thresh=0.05`` at
        ``inferCNV_mask_non_DE.R:29``).
    test_use
        ``"wilcoxon"`` is the only fully parity-tested path. ``"t"``
        uses Welch's two-sample t-test (``scipy.stats.ttest_ind(
        equal_var=False)``), mirroring R ``t.test(vals1, vals2)``.
        ``"perm"`` (R's ``coin::oneway_test``) is **out of scope**.
    require_DE_all_normals
        Per R ``inferCNV_ops.R:206``; decides the count-across-normals
        cutoff.

        * ``"any"`` → mask only genes DE in **zero** normal comparisons.
        * ``"most"`` → mask genes DE in fewer than ``N/2`` normals.
        * ``"all"`` → mask genes DE in fewer than all N normals.
    center_val
        Value assigned to masked entries. R defaults to
        ``mean(infercnv_obj@expr.data)``
        (``inferCNV_mask_non_DE.R:31`` wired at ``inferCNV_ops.R:1523``).
        ``None`` → compute ``np.mean(expr_matrix)`` at call time to
        match R precisely.
    min_cluster_size_mask
        R ``inferCNV_mask_non_DE.R:81, 102-104``. Subclusters smaller
        than this are retained (unmasked) wholesale.

    Returns
    -------
    masked_matrix
        Cells × genes float64 matrix with non-DE positions replaced by
        ``center_val``.
    de_mask
        Cells × genes bool matrix; ``True`` where the cell/gene entry
        was **kept** (DE-passing), ``False`` where it was masked.
        Stored on :attr:`pyinfercnv.result.InferCNVResult.de_mask`.

    Notes
    -----
    The R algorithm records DE genes per (tumor_subcluster, normal_type)
    pair, assigns ``num_normal_types`` to all reference cells and to
    cells in small subclusters upfront, and otherwise increments an
    integer count in the ``(gene, cell)`` cells of the tumor subcluster
    by +1 each time that gene is DE against one of the normal types.
    The final count is thresholded per ``require_DE_all_normals``:
    ``"any"`` → count == 0 is masked, ``"most"`` → count < N/2 is
    masked, ``"all"`` → count != N is masked. See
    ``inferCNV_mask_non_DE.R:120-130``.
    """
    if test_use not in ("wilcoxon", "t"):
        raise ValueError(
            f"test_use={test_use!r} not supported; choose 'wilcoxon' or 't'."
        )
    if require_DE_all_normals not in ("any", "most", "all"):
        raise ValueError(
            f"require_DE_all_normals={require_DE_all_normals!r} invalid; "
            "choose 'any', 'most', or 'all'."
        )

    expr = np.ascontiguousarray(expr_matrix, dtype=np.float64)
    n_cells, n_genes = expr.shape

    if center_val is None:
        center_val = float(np.mean(expr))

    # R: all_DE_genes_matrix = 0 matrix (genes x cells). In python we build
    # cells x genes and transpose mental model accordingly.
    count_matrix = np.zeros((n_cells, n_genes), dtype=np.int32)

    normal_types = list(reference_group_indices.keys())
    num_normal_types = len(normal_types)
    if num_normal_types == 0:
        raise ValueError(
            "reference_group_indices is empty — cannot mask non-DE genes "
            "without at least one normal reference group."
        )

    # 1) All reference cells get count = num_normal_types (never masked).
    #    R: all_DE_genes_matrix[, unlist(reference_grouped_cell_indices)] = num_normal_types.
    all_ref_idx = np.concatenate(
        [np.asarray(v, dtype=np.int64) for v in reference_group_indices.values()]
    )
    count_matrix[all_ref_idx, :] = num_normal_types

    # 2) Small tumor subclusters are retained wholesale.
    #    R: for DE_results (which iterates once per (subcluster, normal)),
    #    if length(tumor_indices) < min_cluster_size_mask: set all to num_normal_types.
    #    In python: one pass over the unique subclusters is sufficient.
    for sub_indices in tumor_subcluster_indices.values():
        sub_idx = np.asarray(sub_indices, dtype=np.int64)
        if sub_idx.size < min_cluster_size_mask:
            count_matrix[sub_idx, :] = num_normal_types

    # 3) Per (subcluster, normal_type): compute p-values, BH-adjust,
    #    count genes whose adjusted p is below the threshold.
    for sub_name, sub_indices in tumor_subcluster_indices.items():
        sub_idx = np.asarray(sub_indices, dtype=np.int64)
        if sub_idx.size < min_cluster_size_mask:
            # R skips the DE counting for small subclusters (L115: only
            # increments when cell_idx >= min_cluster_size_mask).
            continue
        for normal_name, norm_indices in reference_group_indices.items():
            norm_idx = np.asarray(norm_indices, dtype=np.int64)
            if norm_idx.size == 0 or sub_idx.size == 0:
                # No comparison possible; treat as no DE genes.
                continue

            if test_use == "wilcoxon":
                pvals = _wilcoxon_pvals_per_gene(expr, norm_idx, sub_idx)
            else:  # test_use == "t"
                from scipy.stats import ttest_ind

                res = ttest_ind(
                    expr[norm_idx, :],
                    expr[sub_idx, :],
                    equal_var=False,
                    axis=0,
                    nan_policy="propagate",
                )
                pvals = np.asarray(res.pvalue, dtype=np.float64)

            padj = _bh_adjust(pvals)
            # R: is.na(p) excludes gene from DE set (names(pvals)[pvals<p_thr]
            # drops NA, and NA<thr is FALSE).
            de_gene_mask = np.asarray(padj < mask_nonDE_pval, dtype=bool)
            de_gene_mask &= np.isfinite(padj)

            if de_gene_mask.any():
                # Increment count for (gene, cell) across all tumor cells
                # in this subcluster, for genes that are DE in this comparison.
                count_matrix[np.ix_(sub_idx, de_gene_mask)] += 1
            _ = sub_name, normal_name  # kept for potential debug breakpoints

    # 4) Apply threshold per require_DE_all_normals.
    if require_DE_all_normals == "all":
        # Mask if NOT DE in every normal comparison.
        mask_bool = count_matrix != num_normal_types
    elif require_DE_all_normals == "most":
        mask_bool = count_matrix < (num_normal_types / 2.0)
    else:  # "any" — default
        mask_bool = count_matrix == 0

    de_mask = ~mask_bool  # True = kept (DE), False = masked
    masked = expr.copy()
    masked[mask_bool] = center_val

    return masked, de_mask


def _step21_mask_non_DE(  # noqa: N802 — matches R name
    result: "InferCNVResult",
    adata: "AnnData",
    config: "InferCNVConfig",
) -> "InferCNVResult":
    """Wrapper around :func:`mask_non_DE_genes` populating ``result.de_mask``.

    Builds the R-style ``tumor_subcluster_indices`` and
    ``reference_group_indices`` maps from ``result.subclusters`` and
    ``result.cell_meta['is_reference']`` + ``adata.obs[reference_key]``.

    The input expression matrix mirrors R at step 21: R's
    ``infercnv_obj@expr.data`` is in **linear FC** space at this point
    (post-step 14 invert_log2 + step 16 outlier-prune). The Python
    equivalent lives in ``result.cnv_matrix_fc``. This wrapper writes
    back ``result.cnv_matrix_fc`` and ``result.cnv_matrix`` (the latter
    re-derived via ``log2``) so the pipeline downstream sees a mutated
    expr state identical in spirit to R's.

    Parameters
    ----------
    result
        Phase 2 :class:`InferCNVResult`. Must have ``subclusters`` and
        ``cell_meta`` populated.
    adata
        The ``AnnData`` passed to :func:`pyinfercnv.pipeline.infercnv`.
        Used to read the reference-annotation column (via ``cell_meta``).
    config
        :class:`InferCNVConfig` with ``mask_nonDE_pval``, ``test_use``,
        and ``require_DE_all_normals`` set.

    Returns
    -------
    InferCNVResult
        Same result instance, mutated in-place (``de_mask`` and
        ``cnv_matrix_fc`` updated).
    """
    del adata  # kept in signature for API parity; cell_meta has what we need
    if result.subclusters is None:
        raise ValueError("result.subclusters must be set before running step 21")

    expr = result.cnv_matrix_fc.astype(np.float64, copy=True)
    is_ref = result.cell_meta["is_reference"].to_numpy(dtype=bool)

    # Build reference groups by annotation label (the "normal_type" in R).
    # Prefer a ``reference_group`` / ``celltype`` column if present.
    ref_col = None
    for cand in ("reference_group", "celltype", "cell_type", "annotation"):
        if cand in result.cell_meta.columns:
            ref_col = cand
            break
    reference_group_indices: dict[str, np.ndarray] = {}
    if ref_col is not None:
        labels = result.cell_meta[ref_col].to_numpy()
        ref_rows = np.where(is_ref)[0]
        for lab in np.unique(labels[ref_rows]):
            reference_group_indices[str(lab)] = ref_rows[labels[ref_rows] == lab]
    else:
        reference_group_indices["_all_ref"] = np.where(is_ref)[0]

    # Build tumor_subcluster_indices — one entry per unique subcluster
    # label among non-reference cells. ``result.subclusters`` may be a
    # numpy string / int array; the label itself is used as the key.
    subclusters = np.asarray(result.subclusters)
    tumor_subcluster_indices: dict[str, np.ndarray] = {}
    for sc in np.unique(subclusters[~is_ref]):
        if sc == -1:  # reference sentinel (phase 2 convention)
            continue
        idx = np.where((subclusters == sc) & (~is_ref))[0]
        if idx.size > 0:
            tumor_subcluster_indices[str(sc)] = idx

    masked, de_mask = mask_non_DE_genes(
        expr,
        tumor_subcluster_indices=tumor_subcluster_indices,
        reference_group_indices=reference_group_indices,
        mask_nonDE_pval=config.mask_nonDE_pval,
        test_use=config.test_use,
        require_DE_all_normals=config.require_DE_all_normals,
    )

    result.de_mask = de_mask
    result.cnv_matrix_fc = masked.astype(np.float32)
    # Re-derive log2 matrix from the linear FC view so downstream stays
    # consistent. ``np.log2(x)`` with x<=0 yields -inf/NaN; the masked
    # matrix has center_val >= 0 and positive real entries, but clip at
    # a tiny floor to stay numerically sane.
    with np.errstate(divide="ignore", invalid="ignore"):
        result.cnv_matrix = np.log2(
            np.clip(masked, 1e-12, None)
        ).astype(np.float32)
    return result
