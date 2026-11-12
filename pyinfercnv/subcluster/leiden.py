"""Leiden-based tumor subclustering — R-parity via igraph C core.

R reference: ``.single_tumor_leiden_subclustering`` +
``.leiden_simple_snn`` in ``R/inferCNV_tumor_subclusters.R:569-740``.

This port targets the R **``leiden_method="simple"``** branch (R lines
725-741): KNN on the raw CNV matrix, undirected adjacency via upper
triangle, then ``igraph::cluster_leiden`` with the ``CPM`` objective at
the auto-resolution formula ``(11.98 / n_cells)^(1/1.165)``. The
``python-igraph`` package wraps the same C core as R's ``igraph``, so
given the same input graph both sides invoke byte-identical Leiden code
and — up to KNN tie-break — produce matching partitions.

R's default ``leiden_method="PCA"`` uses Seurat's ``RunPCA`` +
``FindNeighbors`` pipeline. Matching it bit-exactly requires duplicating
Seurat's numerics (irlba SVD, the Seurat-style Jaccard SNN) which we
deliberately do NOT do. Bit-exact parity on this backend requires the
R test driver to pass ``leiden_method="simple", leiden_function="CPM"``;
``tests/r_reference.R`` does so.

Implementation notes (parity-critical)
--------------------------------------
R ``.leiden_simple_snn``::

    snn <- nn2(t(expr_data), k=k_nn)$nn.idx        # (n_cells, k_nn)
    sparse_adjacency_matrix <- sparseMatrix(
        i = rep(1:n_cells, each=k_nn),
        j = t(snn),
        x = 1, dims = c(n_cells, n_cells), ...
    )
    graph_obj = graph_from_adjacency_matrix(sparse_adjacency_matrix,
                                            mode="undirected")
    partition_obj = cluster_leiden(graph_obj,
                                   resolution_parameter=...,
                                   objective_function="CPM")

The R KNN adjacency matrix ``M[i, snn[i, k]] = 1`` is ASYMMETRIC (cell i
points to its k NN, which are not necessarily mutual). R's
``graph_from_adjacency_matrix(mat, mode="undirected")`` uses the
``"max"`` semantics for current igraph versions: the undirected graph
has edge (i, j) with ``i != j`` iff ``max(M[i, j], M[j, i]) > 0`` — i.e.,
the OR of the two directed-KNN half-edges. Self-loops from the diagonal
are discarded because the undirected graph drops diagonal entries. This
is *strictly more edges* than a pure upper-triangle reading.

We match the R path's **off-diagonal** semantics; the two graphs are
not guaranteed to be vertex-for-vertex identical (see known deviations
below):
  1. ``sklearn.neighbors.NearestNeighbors(algorithm='brute', metric='euclidean')``
     gives the same euclidean KNN as ``RANN::nn2`` up to tie-break.
     Track A2 empirically verified ``edge Jaccard = 1.0000`` against R
     on three 3CA patients.
  2. Undirected edge set = ``{(min(i,j), max(i,j)) : j in KNN(i), i != j}``
     — the OR of both directed-KNN half-edges. No weights.
     **Self-loops omitted**. R ``nn2`` returns self as the first neighbour
     and R's ``graph_from_adjacency_matrix(mode="undirected")`` keeps the
     diagonal as self-loops. Empirically (brute-forced 967 small random
     graphs in the codex review, see
     ``docs/superpowers/reviews/phase2-i3-fix-codex.md`` S1, and the
     Track A3 sweep in ``benchmarks/phase2/.../track_a1/``) those
     self-loops do not change Leiden membership at the CPM resolution
     infercnv uses.
  3. ``igraph.Graph(n=n_cells, edges=edges, directed=False)`` then call
     ``g.community_leiden(objective_function='CPM',
                          resolution=resolution, n_iterations=-1)``.
     Seeded via ``ig.set_random_number_generator(random.Random(random_state))``
     with a finally-block restore to the default PCG32 so the global
     RNG state does not leak into other callers.

Known sources of deviation from R on this path:
  * ``nn2`` uses the ANN C++ library (approximate NN, tie-break via
    insertion order). sklearn's brute-force is exact euclidean and
    breaks ties by index. For well-separated points the two agree; at
    equidistant cells tie-break may differ.
  * Self-loops omitted (see above).
  * Different PRNG backends under the hood: R igraph uses the bundled
    C PCG32; python-igraph with a ``random.Random`` override dispatches
    through Python's Mersenne-Twister. Leiden's internal seed-usage
    should be limited to node shuffling so the impact is bounded, but
    this is the mechanism why two cross-language runs at the same
    ``random_state`` are not bit-identical.

Track A summary (HANDOFF v6 closeout; see
``benchmarks/phase2/Gao2021_Breast/track_a_summary.md``):
  * DCIS1 (n=1101 tumor): py ARI 0.468±0.14 / R self 0.474±0.12
  * TNBC1 (n=796  tumor): py ARI 0.733±0.09 / R self 0.669±0.10
  * TNBC3 (n=195  tumor): py ARI 0.915±0.04 / R self 0.766±0.29
  Py finds equal-or-higher CPM objective on the same graph; no
  systematic gap observed. Per-seed bit-identity across R/py is
  not a contract.

Optional non-default modes
--------------------------
Two kwargs enable downstream-stability modes (default off):
  * ``n_seeds`` — number of seeds to sweep; returns the partition with
    the highest CPM objective (``rbest``). Default ``1``. Each extra
    seed reuses the same KNN graph, so cost is linear in ``n_seeds``
    for Leiden only.
  * ``min_subcluster_size`` — merge clusters smaller than this into
    their nearest non-small cluster via KNN neighbour voting. Default
    ``None`` (off). Non-CPM-optimal by design: trades a small CPM loss
    for downstream HMM stability by suppressing singletons.
"""
from __future__ import annotations

import numpy as np

__all__ = ["leiden_subcluster"]


def _manual_cpm(
    n: int,
    edges_arr: np.ndarray,
    membership: np.ndarray,
    gamma: float,
) -> float:
    """Σ_c [e_c − γ n_c(n_c−1)/2] on undirected, no-self-loop edge set."""
    m = membership
    K = int(m.max()) + 1 if m.size else 0
    if K == 0:
        return 0.0
    same = m[edges_arr[:, 0]] == m[edges_arr[:, 1]]
    e_c = np.bincount(m[edges_arr[:, 0]][same], minlength=K).astype(np.float64)
    n_c = np.bincount(m, minlength=K).astype(np.float64)
    return float((e_c - gamma * n_c * (n_c - 1) / 2.0).sum())


def _collapse_small_clusters(
    labels: np.ndarray,
    knn_idx: np.ndarray,
    min_size: int,
    max_iter: int = 50,
) -> np.ndarray:
    """Merge clusters of size < ``min_size`` into nearest-neighbour
    non-small cluster by KNN vote. Iterates until convergence or
    bounded max_iter (in case cascading merges are needed)."""
    if min_size is None or min_size <= 1:
        return labels
    labels = labels.astype(np.int32, copy=True)
    n = labels.size
    for _ in range(max_iter):
        sizes = np.bincount(labels)
        small = set(np.where(sizes < min_size)[0].tolist())
        if not small:
            break
        if len(small) >= len(sizes):
            # Entire cell set is "small" — fall back to single cluster.
            labels[:] = 0
            break
        changed = False
        # For each cell in a small cluster, vote its KNN neighbours'
        # non-small cluster labels; reassign to majority (ties → lowest id).
        for i in range(n):
            if labels[i] not in small:
                continue
            # Skip self at col 0, consider remaining KNN neighbours.
            nbr_labels = labels[knn_idx[i, 1:]]
            # Filter out neighbours still in small clusters.
            keep = np.array([lbl not in small for lbl in nbr_labels])
            if not keep.any():
                continue  # leave for next pass (cascade)
            votes = nbr_labels[keep]
            # Majority; ties → lowest label.
            uniq, counts = np.unique(votes, return_counts=True)
            best = uniq[np.argmax(counts[::-1])]  # stable argmax, lowest on tie
            # Better: explicit tie break to lowest id.
            top = counts.max()
            candidates = uniq[counts == top]
            best = int(candidates.min())
            if labels[i] != best:
                labels[i] = best
                changed = True
        if not changed:
            # Nobody moved this pass; remaining small-cluster cells have
            # no valid non-small neighbour. Dump them into the single
            # largest cluster for determinism.
            sizes = np.bincount(labels)
            non_small = np.array([s >= min_size for s in sizes])
            if not non_small.any():
                labels[:] = 0
                break
            # Largest non-small cluster
            masked_sizes = np.where(non_small, sizes, -1)
            largest = int(np.argmax(masked_sizes))
            for i in range(n):
                if labels[i] in small:
                    labels[i] = largest
            break
    # Compact label space to 0..K-1
    _, compact = np.unique(labels, return_inverse=True)
    return compact.astype(np.int32)


def leiden_subcluster(
    cnv_matrix: np.ndarray,
    *,
    resolution: float | str = "auto",
    k_nn: int = 20,  # noqa: N803 (R kwarg name preserved)
    random_state: int = 0,
    n_seeds: int = 1,
    min_subcluster_size: int | None = None,
    n_jobs: int = -1,
) -> np.ndarray:
    """Return per-cell int32 subcluster label, shape (n_cells,).

    Parameters
    ----------
    cnv_matrix
        (n_cells, n_genes) log2 fold-change matrix (cells x genes layout).
    resolution
        Leiden CPM resolution. ``"auto"`` applies R formula
        ``(11.98 / n_cells) ** (1 / 1.165)``.
    k_nn
        Number of nearest neighbours for the KNN graph (R ``k_nn``).
    random_state
        Seed passed to ``community_leiden``. The simple-SNN graph is
        deterministic in ``X`` and ``k_nn``; only Leiden's initial state
        depends on the seed.
    n_seeds
        If > 1, run Leiden with seeds ``random_state, random_state+1,
        …, random_state+n_seeds-1`` and return the partition with the
        highest manual CPM objective (``rbest``). KNN is computed once.
        Default 1 (R-parity; single seed).
    min_subcluster_size
        If set, merge any cluster smaller than this size into its
        nearest non-small cluster via KNN majority vote. Default
        ``None`` (off). **Non-CPM-optimal by design** — trades a small
        CPM loss for downstream stability.
    n_jobs
        Passed to ``sklearn.neighbors.NearestNeighbors`` for the
        euclidean brute-force KNN. ``-1`` uses all available cores
        (default); ``1`` disables parallelism for deterministic
        ordering in single-threaded CI. The KNN computation is the
        dominant cost for n_cells ≳ 1k (codex G3 Track C review).
    """
    X = np.ascontiguousarray(cnv_matrix, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"cnv_matrix must be 2D (n_cells, n_genes); got shape {X.shape}")
    n_cells = X.shape[0]

    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1; got {n_seeds}")
    if min_subcluster_size is not None and min_subcluster_size < 1:
        raise ValueError(
            f"min_subcluster_size must be >= 1 or None; got {min_subcluster_size}"
        )

    # R early-return edge cases (inferCNV_tumor_subclusters.R:573-584)
    if n_cells < 3:
        return np.zeros(n_cells, dtype=np.int32)
    if k_nn >= n_cells:
        return np.zeros(n_cells, dtype=np.int32)

    # Codex G3 Track C Q2: exact brute KNN becomes painful at n >~ 8k
    # (roughly 35s at n=10k scaling quadratically from our measured
    # 0.43s at n=1101 on d=9237). We don't auto-switch to approximate
    # KNN because that would silently break parity; we warn instead.
    if n_cells >= 8_000:
        import warnings
        warnings.warn(
            f"leiden_subcluster: exact brute KNN on n_cells={n_cells} may "
            "dominate runtime (O(n²·d)). Downsampling to <8k tumour cells "
            "or implementing an opt-in approximate-KNN flag are the usual "
            "next steps. Tracked in docs/superpowers/reviews/track-c-perf-codex.md.",
            UserWarning,
            stacklevel=2,
        )

    # Resolve auto resolution (R line 588)
    if isinstance(resolution, str):
        if resolution != "auto":
            raise ValueError(f"resolution must be 'auto' or float; got {resolution!r}")
        used_resolution = float((11.98 / n_cells) ** (1.0 / 1.165))
    else:
        used_resolution = float(resolution)

    # Lazy imports keep `import pyinfercnv` cheap
    try:
        import igraph as ig
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "leiden_subcluster requires python-igraph. "
            "Install with `pip install igraph` or `pip install 'pyinfercnv[leiden]'`."
        ) from exc
    from sklearn.neighbors import NearestNeighbors

    # ---- 1. KNN on raw CNV matrix (mirrors R nn2) ----
    # algorithm='brute' + metric='euclidean' for deterministic exact NN.
    # n_jobs parallelises the distance computation (sklearn internally
    # splits rows across workers); ordering is preserved.
    # Codex G3 Track C recommendation: keep brute exact for parity,
    # parallelise the dominant cost.
    nn = NearestNeighbors(
        n_neighbors=k_nn,
        algorithm="brute",
        metric="euclidean",
        n_jobs=n_jobs,
    )
    nn.fit(X)
    idx = nn.kneighbors(X, return_distance=False)

    # ---- 2. Symmetric undirected edge set (mode="max" semantics) ----
    # R `graph_from_adjacency_matrix(mode="undirected")` -> `mode="max"`:
    # undirected edge (i, j), i != j, exists iff j is in KNN(i) or i is in
    # KNN(j). Sort each pair (min, max) and dedupe via np.unique on an
    # int64 key; self-loops dropped (Track A3 showed no effect on Leiden).
    row_idx = np.repeat(np.arange(n_cells, dtype=np.int64), k_nn)
    col_idx = idx.reshape(-1).astype(np.int64)
    non_self = row_idx != col_idx
    a = np.where(row_idx < col_idx, row_idx, col_idx)[non_self]
    b = np.where(row_idx < col_idx, col_idx, row_idx)[non_self]
    edge_keys = np.unique(a * n_cells + b)
    edges_arr = np.empty((edge_keys.size, 2), dtype=np.int64)
    edges_arr[:, 0] = edge_keys // n_cells
    edges_arr[:, 1] = edge_keys % n_cells

    # ---- 3. Build igraph graph (reused across seeds if n_seeds > 1) ----
    g = ig.Graph(n=n_cells, edges=edges_arr.tolist(), directed=False)

    def _run_one(seed: int) -> np.ndarray:
        import random as _random
        ig.set_random_number_generator(_random.Random(seed))
        try:
            # R cluster_leiden defaults: beta = 0.01, n_iterations = 2
            part = g.community_leiden(
                objective_function="CPM",
                resolution=used_resolution,
                beta=0.01,
                n_iterations=2,
                weights=None,
            )
        finally:
            ig.set_random_number_generator(None)
        return np.asarray(part.membership, dtype=np.int32)

    if n_seeds == 1:
        labels = _run_one(random_state)
    else:
        # rbest: sweep seeds, pick highest CPM. Tie-break lowest seed offset.
        best_q = -np.inf
        best_lbl: np.ndarray | None = None
        for k in range(n_seeds):
            lbl = _run_one(random_state + k)
            q = _manual_cpm(n_cells, edges_arr, lbl.astype(np.int64), used_resolution)
            if q > best_q:
                best_q = q
                best_lbl = lbl
        assert best_lbl is not None
        labels = best_lbl

    if min_subcluster_size is not None and min_subcluster_size > 1:
        labels = _collapse_small_clusters(labels, idx, int(min_subcluster_size))

    return labels


def _r_auto_resolution(n_cells: int) -> float:
    """R ``(11.98 / n)^(1 / 1.165)``; R's ``^`` uses ``powl`` on x86-64."""
    return float(np.power(np.longdouble(11.98 / n_cells), np.longdouble(1.0 / 1.165)))


def _r_lower_edges(adj, combine: str):
    """Edges of R ``graph_from_adjacency_matrix(adj, mode = combine)`` for a sparse matrix:
    the lower triangle combined with the transposed upper one (``pmin``/``pmax``), zeros dropped,
    in column-major order; self-loops kept (``diag = TRUE``). Returns (rows, cols, values)."""
    from scipy import sparse

    a = sparse.csr_matrix(adj, dtype=np.float64)
    lo = sparse.tril(a, k=0).tocsc()
    up = sparse.tril(sparse.triu(a, k=0).T, k=0).tocsc()
    m = lo.minimum(up) if combine == "min" else lo.maximum(up)
    m = sparse.csc_matrix(m)
    m.eliminate_zeros()
    m.sort_indices()
    cols = np.repeat(np.arange(m.shape[1]), np.diff(m.indptr))
    return m.indices.astype(np.int64), cols.astype(np.int64), m.data


def _leiden_r(n_cells, rows, cols, weights, resolution, objective, rng):
    """igraph Leiden as R ``cluster_leiden`` (2 iterations, ``beta = 0.01``) on R's generator."""
    import igraph as ig

    from pyinfercnv._rcompat.rng import RIgraphRNG

    g = ig.Graph(n=n_cells, edges=np.column_stack([rows, cols]).tolist(), directed=False)
    ig.set_random_number_generator(RIgraphRNG(rng))
    try:
        part = g.community_leiden(objective_function=objective,
                                  weights=None if weights is None else list(weights),
                                  resolution=resolution, beta=0.01, n_iterations=2)
    finally:
        ig.set_random_number_generator(None)
    return np.asarray(part.membership, dtype=np.int32)


def leiden_simple_r(cnv_matrix: np.ndarray, *, resolution: float | str = "auto", k_nn: int = 20,
                    objective_function: str = "CPM", rng=None) -> np.ndarray:
    """R ``.leiden_simple_snn``: ``nn2`` k-NN (the cell itself included), the adjacency of
    cell -> neighbour, ``graph_from_adjacency_matrix(mode = "undirected")`` (asymmetric, so
    ``"max"``), unweighted Leiden on R's generator ``rng`` (continuing its stream)."""
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors

    from pyinfercnv._rcompat.rng import RRandom

    X = np.asarray(cnv_matrix, dtype=np.float64)
    n = X.shape[0]
    if n < 3 or k_nn >= n:
        return np.zeros(n, dtype=np.int32)
    res = _r_auto_resolution(n) if resolution == "auto" else float(resolution)
    idx = NearestNeighbors(n_neighbors=k_nn, algorithm="brute").fit(X).kneighbors(X, return_distance=False)
    adj = sparse.csr_matrix((np.ones(n * k_nn), idx.ravel(), np.arange(0, n * k_nn + 1, k_nn)), shape=(n, n))
    r, c, _ = _r_lower_edges(adj, "max")
    return _leiden_r(n, r, c, None, res, objective_function, rng if rng is not None else RRandom(1))


def _leiden(g, weights, resolution: float, objective: str, random_state: int) -> np.ndarray:
    """igraph Leiden with R ``cluster_leiden``'s defaults (``beta = 0.01``, 2 iterations)."""
    import random as _random

    import igraph as ig
    ig.set_random_number_generator(_random.Random(random_state))
    try:
        part = g.community_leiden(objective_function=objective, weights=weights,
                                  resolution=resolution, beta=0.01, n_iterations=2)
    finally:
        ig.set_random_number_generator(None)
    return np.asarray(part.membership, dtype=np.int32)


def seurat_snn(embeddings: np.ndarray, k_nn: int = 20, prune: float = 1.0 / 15) -> "sparse.csr_matrix":
    """Seurat ``FindNeighbors`` SNN graph: Jaccard index of the ``k_nn``-nearest-neighbour sets
    (each set includes the cell itself), entries below ``prune`` dropped. Exact euclidean
    neighbours (Seurat uses Annoy, which returns the exact neighbours on data of this size)."""
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors

    n = embeddings.shape[0]
    k = min(k_nn, n)
    idx = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(embeddings).kneighbors(
        embeddings, return_distance=False)
    nn = sparse.csr_matrix((np.ones(n * k), idx.ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))
    shared = (nn @ nn.T).tocsr()
    shared.data = shared.data / (2 * k - shared.data)
    shared.data[shared.data < prune] = 0.0
    shared.eliminate_zeros()
    return shared


def leiden_subcluster_pca(
    cnv_matrix: np.ndarray,
    *,
    resolution: float | str = "auto",
    k_nn: int = 20,  # noqa: N803 (R kwarg name preserved)
    objective_function: str = "CPM",
    random_state: int = 0,
    rng=None,
) -> np.ndarray:
    """R ``.leiden_seurat_preprocess_routine`` (``leiden_method = "PCA"``, R's default).

    On the cells x genes expression matrix: Seurat ``FindVariableFeatures`` (vst, 2,000 genes),
    ``ScaleData``, ``RunPCA(npcs = 10)``, ``FindNeighbors(k.param = k_nn)``; the SNN graph
    (undirected, weighted, ``mode = "min"``) is partitioned with Leiden.

    With ``rng`` (R's generator, :class:`pyinfercnv._rcompat.rng.RRandom`) the random numbers
    follow R: ``RunPCA`` runs ``set.seed(42)`` and irlba draws one normal per feature, then
    Leiden draws through R igraph's generator -- the partition is R's, and ``rng`` is left where
    R's would be. Without ``rng`` a fresh generator seeded with ``random_state`` is used."""
    from pyinfercnv._rcompat.rng import RRandom
    from pyinfercnv._rcompat.seurat import run_pca, scale_data, vst

    X = np.asarray(cnv_matrix, dtype=np.float64)
    n_cells = X.shape[0]
    if n_cells < 3 or k_nn >= n_cells:
        return np.zeros(n_cells, dtype=np.int32)
    rng = rng if rng is not None else RRandom(random_state)
    used_resolution = _r_auto_resolution(n_cells) if resolution == "auto" else float(resolution)
    genes_x_cells = X.T
    try:
        feat = vst(genes_x_cells, n_features=2000)
    except Exception:                       # R: a warning in FindVariableFeatures -> simple path
        return leiden_simple_r(X, resolution=used_resolution, k_nn=k_nn,
                               objective_function=objective_function, rng=rng)
    scaled = scale_data(genes_x_cells[feat])
    var = np.var(scaled, axis=1)
    keep = np.isfinite(var) & (var > 0)        # RunPCA keeps features with variance > 0
    npcs = min(10, n_cells - 1)
    emb = run_pca(scaled[keep], n_pcs=npcs)
    rng.set_seed(42)                           # RunPCA(seed.use = 42)
    rng.skip_normals(int(keep.sum()))          # irlba's start vector rnorm(n_features)
    snn = seurat_snn(emb, k_nn=k_nn)
    r, c, w = _r_lower_edges(snn, "min")
    return _leiden_r(n_cells, r, c, w, used_resolution, objective_function, rng)
