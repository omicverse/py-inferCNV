"""Random-smoothed-trees tumor subclustering.

R parity target: `R/inferCNV_tumor_subclusters.random_smoothed_trees.R`.

Algorithm (recursive):
    1. Smooth cnv_matrix along the gene axis (runmean, window=101) and
       median-center each cell (R `.center_columns(..., 'median')`).
    2. Fit `hclust` (ward.D2) on the smoothed matrix.
    3. Draw 100 permuted null matrices (shuffle cells independently per gene
       column), repeat smoothing + hclust on each, collect max(height).
    4. ECDF p-value = 1 - F(max(h_obs)). If p <= p_val, cut tree at
       mean(top two heights) and recurse into each child cluster.
    5. Stop at max_recursion_depth or when all children are smaller than
       min_cluster_size_recurse.

P7 patch (subsample_for_tree): if n_cells > subsample_for_tree, warn, draw a
seeded subsample, run the full recursion on the subsample, then assign every
remaining cell to the subsample cluster whose centroid is closest (euclidean).
This bounds the O(n^2) distance computation inside scipy `linkage`.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist

from pyinfercnv.center.center_cells import center_cells
from pyinfercnv.smooth.pyramidinal import smooth_pyramidinal

__all__ = ["random_tree_subcluster"]


# R defaults
_DEFAULT_WINDOW_SIZE = 101
_DEFAULT_MAX_RECURSION_DEPTH = 3
_DEFAULT_MIN_CLUSTER_SIZE_RECURSE = 10
_DEFAULT_N_RAND_ITERS = 100


def _smooth_and_center(X: np.ndarray, window_size: int) -> np.ndarray:
    """Equivalent of R `apply(X, 2, runmean, k=window) |> .center_columns('median')`.

    X is (n_cells, n_genes). Smoothing is along the gene axis within each cell
    (pyinfercnv layout). center_cells median-centres each cell row.
    """
    n_genes = X.shape[1]
    # smooth_pyramidinal requires odd window >= 3. If the matrix is narrow we
    # shrink the window to the largest odd size <= min(window_size, n_genes).
    effective_w = min(window_size, n_genes if n_genes % 2 == 1 else n_genes - 1)
    if effective_w < 3:
        smoothed = np.ascontiguousarray(X, dtype=np.float32)
    else:
        smoothed = smooth_pyramidinal(X, window_length=effective_w)
    return center_cells(smoothed, method="median")


def _linkage_heights(X_sc: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute ward.D2 linkage on the observation matrix.

    Returns (Z, heights) where heights = Z[:, 2] (merge distances).
    """
    Z = linkage(np.ascontiguousarray(X_sc, dtype=np.float64), method=method)
    return Z, Z[:, 2]


def _permute_columns(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """R `permute_col_vals(t(expr))`: shuffle rows (cells) independently per column (gene).

    Input X has shape (n_cells, n_genes). Output same shape with each column
    independently permuted.
    """
    n_cells, n_genes = X.shape
    # argsort of uniform noise per column gives an independent permutation per
    # column vectorised without a Python loop.
    perm_idx = np.argsort(rng.random((n_cells, n_genes), dtype=np.float64), axis=0)
    col_idx = np.broadcast_to(np.arange(n_genes), (n_cells, n_genes))
    return X[perm_idx, col_idx]


def _parameterize_null(
    X: np.ndarray,
    *,
    hclust_method: str,
    window_size: int,
    n_rand_iters: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (Z_obs, heights_obs, max_height_pval) from the random-trees null.

    Fits hclust on the smoothed observed matrix, then draws `n_rand_iters`
    permuted null matrices and records max(height) for each. Empirical ECDF
    gives the one-sided tail p-value.
    """
    X_sc = _smooth_and_center(X, window_size)
    Z_obs, heights_obs = _linkage_heights(X_sc, hclust_method)
    max_obs = float(heights_obs.max()) if heights_obs.size else 0.0

    max_rand_heights = np.empty(n_rand_iters, dtype=np.float64)
    for i in range(n_rand_iters):
        X_perm = _permute_columns(X, rng)
        X_perm_sc = _smooth_and_center(X_perm, window_size)
        _, h_rand = _linkage_heights(X_perm_sc, hclust_method)
        max_rand_heights[i] = float(h_rand.max()) if h_rand.size else 0.0

    if max_obs <= 0.0:
        p_val = 1.0
    else:
        # ECDF at max_obs = fraction of null max heights <= max_obs; p = 1 - ECDF.
        # R `ecdf(x)` is right-continuous with jumps — using `<=` matches.
        ecdf = float((max_rand_heights <= max_obs).sum()) / float(n_rand_iters)
        p_val = 1.0 - ecdf

    return Z_obs, heights_obs, p_val


def _recursive_partition(
    X: np.ndarray,
    cell_indices: np.ndarray,
    labels_out: np.ndarray,
    next_label: list[int],
    *,
    hclust_method: str,
    window_size: int,
    tumor_subcluster_pval: float,
    max_recursion_depth: int,
    min_cluster_size_recurse: int,
    n_rand_iters: int,
    rng: np.random.Generator,
    recursion_depth: int = 1,
) -> None:
    """In-place fill of `labels_out[cell_indices]` using the random-trees recursion.

    Mirrors `.single_tumor_subclustering_recursive_random_smoothed_trees`. Each
    terminal clade is assigned a unique integer label drawn from `next_label`.
    """
    n_cells = X.shape[0]

    if recursion_depth > max_recursion_depth or n_cells < 2:
        lbl = next_label[0]
        next_label[0] += 1
        labels_out[cell_indices] = lbl
        return

    _, heights, p_val_obs = _parameterize_null(
        X,
        hclust_method=hclust_method,
        window_size=window_size,
        n_rand_iters=n_rand_iters,
        rng=rng,
    )

    if p_val_obs > tumor_subcluster_pval or heights.size < 2:
        lbl = next_label[0]
        next_label[0] += 1
        labels_out[cell_indices] = lbl
        return

    # R: cut_height = mean(c(h[length(h)], h[length(h)-1]))
    # scipy heights are already in the order linkage produced (monotone for
    # ward), so the final two entries are the top two merges.
    h_sorted = np.sort(heights)
    cut_height = float((h_sorted[-1] + h_sorted[-2]) / 2.0)

    # Recompute Z on the already-smoothed matrix so `fcluster` sees the same
    # linkage as the one whose heights we cut from.
    X_sc = _smooth_and_center(X, window_size)
    Z_sc, _ = _linkage_heights(X_sc, hclust_method)
    grps = fcluster(Z_sc, t=cut_height, criterion="distance")
    uniq = np.unique(grps)

    # R: if all child groups < min_cluster_size_recurse, don't recurse
    # (assign current clade as single label).
    if all((grps == g).sum() < min_cluster_size_recurse for g in uniq):
        lbl = next_label[0]
        next_label[0] += 1
        labels_out[cell_indices] = lbl
        return

    for g in uniq:
        child_mask = grps == g
        child_size = int(child_mask.sum())
        child_local_idx = np.where(child_mask)[0]
        child_global_idx = cell_indices[child_local_idx]

        if child_size >= min_cluster_size_recurse:
            _recursive_partition(
                X[child_local_idx],
                child_global_idx,
                labels_out,
                next_label,
                hclust_method=hclust_method,
                window_size=window_size,
                tumor_subcluster_pval=tumor_subcluster_pval,
                max_recursion_depth=max_recursion_depth,
                min_cluster_size_recurse=min_cluster_size_recurse,
                n_rand_iters=n_rand_iters,
                rng=rng,
                recursion_depth=recursion_depth + 1,
            )
        else:
            lbl = next_label[0]
            next_label[0] += 1
            labels_out[child_global_idx] = lbl


def random_tree_subcluster(
    cnv_matrix: np.ndarray,
    *,
    subsample_for_tree: int = 500,
    tumor_subcluster_pval: float = 0.1,
    hclust_method: str = "ward",  # noqa: N803 (R kwarg name preserved)
    window_size: int = _DEFAULT_WINDOW_SIZE,
    max_recursion_depth: int = _DEFAULT_MAX_RECURSION_DEPTH,
    min_cluster_size_recurse: int = _DEFAULT_MIN_CLUSTER_SIZE_RECURSE,
    n_rand_iters: int = _DEFAULT_N_RAND_ITERS,
    random_state: int = 0,
) -> np.ndarray:
    """Return per-cell int32 subcluster label, shape (n_cells,).

    G1 patch P7: when ``n_cells > subsample_for_tree`` the tree is fit on a
    seeded random subsample and remaining cells are assigned to the
    nearest-centroid cluster. A ``UserWarning`` is emitted.

    Parameters
    ----------
    cnv_matrix
        (n_cells, n_genes) CNV log2 fold-change matrix.
    subsample_for_tree
        Upper bound on cells used for the recursive tree fit. Default 500.
    tumor_subcluster_pval
        Tail p-value threshold (R `p_val`, default 0.1).
    hclust_method
        scipy linkage method; "ward" is R `ward.D2` parity.
    window_size
        Runmean window for the internal smoothing step (R default 101).
    max_recursion_depth, min_cluster_size_recurse
        R recursion stop conditions.
    n_rand_iters
        Permutation null size (R default 100).
    random_state
        Seed for both subsampling and null permutations.
    """
    X = np.ascontiguousarray(cnv_matrix, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"cnv_matrix must be 2D (n_cells, n_genes); got shape {X.shape}")
    n_cells = X.shape[0]

    if n_cells == 0:
        return np.zeros(0, dtype=np.int32)
    if n_cells < 3:
        return np.ones(n_cells, dtype=np.int32)

    master_rng = np.random.default_rng(random_state)

    # --- P7 subsample path -------------------------------------------------
    if n_cells > subsample_for_tree:
        warnings.warn(
            f"random_tree_subcluster: n_cells={n_cells} exceeds "
            f"subsample_for_tree={subsample_for_tree}; fitting tree on a "
            "seeded random subsample and assigning remaining cells by "
            "nearest-centroid.",
            UserWarning,
            stacklevel=2,
        )
        sub_idx = master_rng.choice(n_cells, size=subsample_for_tree, replace=False)
        sub_idx.sort()  # stable ordering aids reproducibility
        X_sub = X[sub_idx]

        sub_labels = np.full(subsample_for_tree, -1, dtype=np.int32)
        next_label = [1]
        _recursive_partition(
            X_sub,
            np.arange(subsample_for_tree),
            sub_labels,
            next_label,
            hclust_method=hclust_method,
            window_size=window_size,
            tumor_subcluster_pval=tumor_subcluster_pval,
            max_recursion_depth=max_recursion_depth,
            min_cluster_size_recurse=min_cluster_size_recurse,
            n_rand_iters=n_rand_iters,
            rng=master_rng,
        )

        # Centroid-based assignment for the remaining cells (and, for symmetry,
        # re-assign every cell so the labels are a consistent partition on X).
        uniq_labels = np.unique(sub_labels)
        centroids = np.stack(
            [X_sub[sub_labels == lbl].mean(axis=0) for lbl in uniq_labels],
            axis=0,
        ).astype(np.float32)

        dists = cdist(X, centroids, metric="euclidean")
        nearest = np.argmin(dists, axis=1)
        labels = uniq_labels[nearest].astype(np.int32)
        return labels

    # --- Full-data path ----------------------------------------------------
    labels = np.full(n_cells, -1, dtype=np.int32)
    next_label = [1]
    _recursive_partition(
        X,
        np.arange(n_cells),
        labels,
        next_label,
        hclust_method=hclust_method,
        window_size=window_size,
        tumor_subcluster_pval=tumor_subcluster_pval,
        max_recursion_depth=max_recursion_depth,
        min_cluster_size_recurse=min_cluster_size_recurse,
        n_rand_iters=n_rand_iters,
        rng=master_rng,
    )
    return labels
