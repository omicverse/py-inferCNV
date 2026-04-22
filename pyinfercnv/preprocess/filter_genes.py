"""Gene-level filters — R parity with `require_above_min_mean_expr_cutoff`
(`inferCNV_ops.R:2128-2212`).

Returns a boolean mask of length n_genes (True = keep). Both the mean-cutoff
and the min-cells tests are computed on the *full* cell matrix, mirroring
R's stage-1 filter. R's stage-2 `require_above_min_cells_ref` (a ref-only
count test) is not applied under default parameters on the standard
oligodendroglioma fixture — empirically [A] all-cells keeps 8508 genes
bit-exact vs R step02, while a two-stage AND keeps only 7904. See
scripts/triage_phase1/ for the evidence.

G1 patch P5: use `Xc.getnnz(axis=0)` instead of `(Xc > 0).sum(axis=0)` —
sparse-idiomatic and avoids materializing the boolean comparison.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse as sp


def filter_low_expression_genes(
    X,
    *,
    cutoff: float = 1.0,
    min_cells_per_gene: int = 3,
) -> np.ndarray:
    """Boolean mask over genes (cols of X).

    Parameters
    ----------
    X
        Counts matrix, cells x genes, CSR float32 or dense.
    cutoff
        Minimum mean expression across all cells. Genes with mean < cutoff are dropped.
    min_cells_per_gene
        Minimum number of cells with strictly positive expression.
    """
    Xc = X.tocsr() if sp.issparse(X) else X

    if sp.issparse(Xc):
        gene_sum = np.asarray(Xc.sum(axis=0)).ravel()
        gene_nnz = Xc.getnnz(axis=0)
    else:
        gene_sum = np.asarray(Xc).sum(axis=0)
        gene_nnz = (np.asarray(Xc) > 0).sum(axis=0)

    n = Xc.shape[0]
    gene_mean = gene_sum / n if n > 0 else np.zeros_like(gene_sum)

    keep = (gene_mean >= cutoff) & (gene_nnz >= min_cells_per_gene)
    return np.asarray(keep, dtype=bool)
