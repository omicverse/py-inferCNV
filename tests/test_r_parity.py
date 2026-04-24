"""Tier-4 R-parity tests against tests/r_out/*.tsv ground-truth fixtures.

Each TSV is the dump of `infercnv_obj@expr.data` (genes x cells) at a specific
R pipeline step (see tests/r_reference.R). We replay the same step in Python
and assert max_abs_diff < 1e-10 (bit-exact-in-practice) for deterministic modules,
or < 1e-6 (tier-4-approximate) where floating-point accumulation order differs.

Skip-when-missing pattern (pycopykat convention): if r_out/<step>.tsv is absent,
pytest.skip with a clear message instead of failing — lets unit tests run on
machines without R installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics

from pyinfercnv.center.center_cells import center_cells
from pyinfercnv.cna.outlier_prune import prune_outliers
from pyinfercnv.preprocess.filter_genes import filter_low_expression_genes
from pyinfercnv.preprocess.log_transform import invert_log2, log2_plus1
from pyinfercnv.preprocess.max_threshold import apply_max_centered_threshold
from pyinfercnv.preprocess.normalize import normalize_by_seq_depth
from pyinfercnv.preprocess.subtract_ref import subtract_reference
from pyinfercnv.smooth.pyramidinal import smooth_pyramidinal
from pyinfercnv.validation.r_parity import bit_exact_assert, max_abs_diff


REPO_ROOT = Path(__file__).resolve().parent.parent
R_OUT_DIR = REPO_ROOT / "tests" / "r_out"

R_FIXTURE = Path(
    "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata/"
    "oligodendroglioma_expression_downsampled.counts.matrix.gz"
)
R_ANNOT = Path(
    "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata/"
    "oligodendroglioma_annotations_downsampled.txt"
)
R_GENE_ORDER = Path(
    "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata/"
    "gencode_downsampled.EXAMPLE_ONLY_DONT_REUSE.txt"
)


def _r_step_available(name: str) -> bool:
    return (R_OUT_DIR / f"{name}.tsv").exists()


def _load_r_step(name: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Load TSV (genes x cells in R layout). Returns (matrix as float64, gene_ids, cell_ids)."""
    df = pd.read_csv(R_OUT_DIR / f"{name}.tsv", sep="\t", index_col=0)
    return df.to_numpy(dtype=np.float64), df.index.tolist(), df.columns.tolist()


@pytest.fixture(scope="module")
def raw_counts_all() -> pd.DataFrame:
    """Original counts as R loads them (genes x cells); still includes genes not in gene_order."""
    if not R_FIXTURE.exists():
        pytest.skip("R infercnv extdata fixture not available")
    return pd.read_csv(R_FIXTURE, sep="\t", index_col=0)


@pytest.fixture(scope="module")
def raw_counts(raw_counts_all, gene_order) -> tuple[np.ndarray, list[str], list[str]]:
    """Counts after R's CreateInfercnvObject filter: intersect(counts, gene_order),
    reorder by gene_order position, AND drop chrX/chrY/chrM (R default chr_exclude).
    Returns (genes x cells) float64."""
    df_all = raw_counts_all
    go = gene_order[~gene_order["chromosome"].isin({"chrX", "chrY", "chrM"})]
    go = go.drop_duplicates("gene_symbol").set_index("gene_symbol")
    common = [g for g in go.index if g in df_all.index]
    df = df_all.loc[common]
    return df.to_numpy(dtype=np.float64), df.index.tolist(), df.columns.tolist()


@pytest.fixture(scope="module")
def annotations() -> pd.DataFrame:
    if not R_ANNOT.exists():
        pytest.skip("R annotations fixture not available")
    df = pd.read_csv(R_ANNOT, sep="\t", header=None, names=["cell_id", "annotation"])
    return df


@pytest.fixture(scope="module")
def gene_order() -> pd.DataFrame:
    if not R_GENE_ORDER.exists():
        pytest.skip("R gene_order fixture not available")
    df = pd.read_csv(R_GENE_ORDER, sep="\t", header=None,
                     names=["gene_symbol", "chromosome", "start", "end"])
    return df


@pytest.fixture(scope="module")
def ref_cell_indices(annotations) -> dict[str, list[int]]:
    """Reference groups: every annotation that doesn't look tumor-like."""
    tumor_pats = ("malignant_", "Tumor_", "tumor_", "Observation", "observation")
    ref_labels = [a for a in annotations["annotation"].unique()
                  if not any(p in a for p in tumor_pats)]
    out = {}
    for label in ref_labels:
        idx = annotations.index[annotations["annotation"] == label].tolist()
        out[label] = idx
    return out


# ============================================================================
# Step 02: filter_genes
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step02_filtered"),
                    reason="r_out/step02_filtered.tsv not generated; run Rscript tests/r_reference.R")
def test_step02_filter_genes_parity(raw_counts):
    """Tier-4 bit-exact: gene set after R's two-stage filter must match."""
    counts_genes_x_cells, gene_ids, cell_ids = raw_counts
    py_counts = counts_genes_x_cells.T  # (cells, genes)
    mask = filter_low_expression_genes(py_counts, cutoff=1.0, min_cells_per_gene=3)
    kept_genes_py = [g for g, k in zip(gene_ids, mask) if k]

    r_data, r_genes, r_cells = _load_r_step("step02_filtered")

    # Tier-3 identity: same gene SET (order can drift on duplicate symbols
    # like CHURC1-FNTB vs FNTB; R's CreateInfercnvObject and pandas dedup
    # may pick different first-occurrence rows).
    assert set(kept_genes_py) == set(r_genes), (
        f"gene set mismatch: py kept {len(kept_genes_py)}, r kept {len(r_genes)}; "
        f"diff py-only={len(set(kept_genes_py)-set(r_genes))} r-only={len(set(r_genes)-set(kept_genes_py))}"
    )

    # Tier-4 bit-exact on the per-gene per-cell values for shared genes.
    py_step02_df = pd.DataFrame(
        py_counts[:, mask].T,
        index=[g for g, k in zip(gene_ids, mask) if k],
        columns=cell_ids,
    ).loc[r_genes]
    bit_exact_assert(py_step02_df.to_numpy(), r_data, tol=1e-10)


# ============================================================================
# Step 03: normalize by seq depth (CPM by median libsize)
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step03_normalized")
                    or not _r_step_available("step02_filtered"),
                    reason="r_out TSVs not available")
def test_step03_normalize_parity():
    """Tier-4 bit-exact: CPM by median libsize."""
    r_step02, _, _ = _load_r_step("step02_filtered")
    r_step03, _, _ = _load_r_step("step03_normalized")

    py_in = r_step02.T  # (cells, genes)
    py_out = normalize_by_seq_depth(py_in)
    py_back = (py_out if not hasattr(py_out, "toarray") else py_out.toarray()).T

    diff = max_abs_diff(py_back.astype(np.float64), r_step03)
    # Phase 1 bit-exact path (2026-04-23): normalize is now float64 through
    # all intermediates; empirical max_diff on oligo is ~5e-11.
    assert diff < 1e-10, f"normalize step03 max_diff={diff:.3e}"


# ============================================================================
# Step 04: log2(x+1)
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step04_logged")
                    or not _r_step_available("step03_normalized"),
                    reason="r_out TSVs not available")
def test_step04_log2_plus1_parity():
    r_step03, _, _ = _load_r_step("step03_normalized")
    r_step04, _, _ = _load_r_step("step04_logged")
    py_out = log2_plus1(r_step03.T).T  # (genes, cells) for compare
    diff = max_abs_diff(py_out.astype(np.float64), r_step04)
    # Phase 1 bit-exact path: float64 log1p; empirical ~5e-14 on oligo.
    assert diff < 1e-10, f"log2_plus1 step04 max_diff={diff:.3e}"


# ============================================================================
# Step 08: subtract_ref (first pass, use_bounds=TRUE)
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step08_subtracted")
                    or not _r_step_available("step04_logged"),
                    reason="r_out TSVs not available")
def test_step08_subtract_ref_parity(raw_counts, annotations, ref_cell_indices):
    """Tier-4 (relaxed): subtract_ref bounded path. Reference groups same as R."""
    r_step04, _, r_cells = _load_r_step("step04_logged")
    r_step08, _, _ = _load_r_step("step08_subtracted")

    # Re-derive ref indices in the SAME cell order R uses (TSV column order)
    annot_by_cell = dict(zip(annotations["cell_id"], annotations["annotation"]))
    tumor_pats = ("malignant_", "Tumor_", "tumor_", "Observation", "observation")
    ref_labels = sorted({annot_by_cell[c] for c in r_cells
                         if not any(p in annot_by_cell.get(c, "") for p in tumor_pats)})
    ref_groups = {}
    for lab in ref_labels:
        ref_groups[lab] = [i for i, c in enumerate(r_cells) if annot_by_cell.get(c) == lab]

    py_in = r_step04.T  # (cells, genes) float64
    py_out = subtract_reference(py_in, ref_groups=ref_groups, use_bounds=True)
    py_back = py_out.T

    diff = max_abs_diff(py_back.astype(np.float64), r_step08)
    # Phase 1 bit-exact path: mean + bounds in float64.
    assert diff < 1e-10, f"subtract_ref step08 max_diff={diff:.3e}"


# ============================================================================
# Step 09: max_centered_threshold = 3
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step09_clipped")
                    or not _r_step_available("step08_subtracted"),
                    reason="r_out TSVs not available")
def test_step09_max_threshold_parity():
    r_step08, _, _ = _load_r_step("step08_subtracted")
    r_step09, _, _ = _load_r_step("step09_clipped")
    py_out = apply_max_centered_threshold(r_step08.T, threshold=3.0).T
    diff = max_abs_diff(py_out.astype(np.float64), r_step09)
    # Phase 1 bit-exact: np.clip in float64 is exact.
    assert diff < 1e-10, f"max_threshold step09 max_diff={diff:.3e}"


# ============================================================================
# Step 11: center_cells (median per cell)
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step11_centered")
                    or not _r_step_available("step10_smoothed"),
                    reason="r_out TSVs not available")
def test_step11_center_cells_parity():
    r_step10, _, _ = _load_r_step("step10_smoothed")
    r_step11, _, _ = _load_r_step("step11_centered")
    # Note: R centers across ALL chromosomes per cell (one median per cell).
    py_out = center_cells(r_step10.T, method="median").T
    diff = max_abs_diff(py_out.astype(np.float64), r_step11)
    # Phase 1 bit-exact: median + subtract in float64.
    assert diff < 1e-10, f"center_cells step11 max_diff={diff:.3e}"


# ============================================================================
# Step 16: prune_outliers (average_bound)
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step14_invert")
                    or not _r_step_available("step12_subtracted2"),
                    reason="r_out TSVs not available")
def test_step14_invert_log2_parity():
    """R run() step 14: invert_log2 on log-space subtracted2 -> linear FC."""
    r_step12, _, _ = _load_r_step("step12_subtracted2")
    r_step14, _, _ = _load_r_step("step14_invert")
    py_out = invert_log2(r_step12.T).T
    diff = max_abs_diff(py_out.astype(np.float64), r_step14)
    # Phase 1 bit-exact: np.exp2 in float64.
    assert diff < 1e-10, f"invert_log2 step14 max_diff={diff:.3e}"


@pytest.mark.skipif(not _r_step_available("step14_invert"),
                    reason="r_out/step14_invert.tsv not generated; run Rscript tests/r_reference.R")
def test_step14_cnv_matrix_spearman_oligo():
    """**Primary py-vs-R parity anchor** (HANDOFF v7 §2.2, Jason 2026-04-23
    review): Spearman ρ on the continuous CNV matrix step14 between py and R
    must be ≥ 0.999 on oligo.

    Why this is primary (not ARI): Leiden bucket IDs are arbitrary
    algorithm-internal labels; the continuous log2-FC matrix is the actual
    scientific output every downstream step consumes. Observed Spearman ρ
    at HEAD is 0.9998 on oligo and 1.0000 on the three 3CA kilocell 10x
    UMI patients (DCIS1 / TNBC1 / TNBC3); full data in
    benchmarks/phase2/Gao2021_Breast/cnv_matrix_spearman.md.

    This test regenerates py step14 from R step12 (same input as
    test_step14_invert_log2_parity) and asserts rank agreement. A drop
    below 0.999 here signals a regression in either ``invert_log2`` or any
    upstream Phase-1 step the step12 fixture depends on.
    """
    from scipy.stats import spearmanr

    r_step12, _, _ = _load_r_step("step12_subtracted2")
    r_step14, _, _ = _load_r_step("step14_invert")
    py_out = invert_log2(r_step12.T.astype(np.float32)).T.astype(np.float64)

    # Global flat Spearman on aligned (genes × cells) matrices
    rho, _ = spearmanr(py_out.ravel(), r_step14.ravel())
    assert rho >= 0.999, (
        f"step14 CNV matrix Spearman ρ={rho:.6f} below floor 0.999 — "
        f"the continuous CNV signal has drifted from R"
    )

    # Per-cell Spearman floor: every cell's gene vector must rank-agree
    # with R's within reasonable tolerance (observed 0.9998 ± 0.0001 on
    # oligo; floor 0.95 is generously permissive and would catch any
    # meaningful per-cell drift).
    n_cells = py_out.shape[1]
    rhos = np.empty(n_cells)
    for j in range(n_cells):
        col_py = py_out[:, j]; col_r = r_step14[:, j]
        if np.all(col_py == col_py[0]) or np.all(col_r == col_r[0]):
            rhos[j] = np.nan
            continue
        rhos[j], _ = spearmanr(col_py, col_r)
    valid = np.isfinite(rhos)
    assert valid.any(), "all cells had constant CNV — test fixture is broken"
    per_cell_min = float(np.nanmin(rhos))
    assert per_cell_min >= 0.95, (
        f"step14 per-cell Spearman ρ min = {per_cell_min:.4f} below floor 0.95"
    )


# ============================================================================
# Step 16: prune_outliers in LINEAR FC space (R run() order: AFTER invert_log2)
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step16_outlier_pruned")
                    or not _r_step_available("step14_invert"),
                    reason="r_out TSVs not available")
def test_step16_outlier_prune_parity():
    """G3 Q5 fix: outlier_prune operates on LINEAR FC (after step 14 invert)."""
    r_step14, _, _ = _load_r_step("step14_invert")
    r_step16, _, _ = _load_r_step("step16_outlier_pruned")
    py_out = prune_outliers(r_step14.T, method="average_bound").T
    diff = max_abs_diff(py_out.astype(np.float64), r_step16)
    # Phase 1 bit-exact: bounds + clip in float64.
    assert diff < 1e-10, f"outlier_prune step16 max_diff={diff:.3e}"


# ============================================================================
# Step 10: smooth_pyramidinal — TIER-4 INTERIOR + TIER-4-APPROX TAIL
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step10_smoothed")
                    or not _r_step_available("step09_clipped"),
                    reason="r_out TSVs not available")
def test_step10_smooth_pyramidinal_per_chromosome(gene_order):  # noqa: F811
    """The trickiest module — interior must bit-match R's triangular kernel,
    tail uses dynamic-denominator R-exact formula."""
    r_step09, r_genes, _ = _load_r_step("step09_clipped")
    r_step10, _, _ = _load_r_step("step10_smoothed")

    # We need to smooth per chromosome — group genes by chr per gene_order.
    chrom_of = dict(zip(gene_order["gene_symbol"], gene_order["chromosome"]))

    # Float64 throughout for bit-exact parity with R's stats::filter.
    py_in_cells_x_genes = r_step09.T.astype(np.float64)
    py_out = py_in_cells_x_genes.copy()

    chroms_in_order = []
    for g in r_genes:
        c = chrom_of.get(g)
        if c is None:
            continue
        if not chroms_in_order or chroms_in_order[-1] != c:
            chroms_in_order.append(c)

    # Build per-chromosome blocks (column ranges).
    col_runs: list[tuple[str, int, int]] = []
    cursor = 0
    while cursor < len(r_genes):
        c = chrom_of.get(r_genes[cursor])
        end = cursor
        while end < len(r_genes) and chrom_of.get(r_genes[end]) == c:
            end += 1
        col_runs.append((c, cursor, end))
        cursor = end

    for chrom, start, end in col_runs:
        block = py_in_cells_x_genes[:, start:end]
        if block.shape[1] >= 2:
            smoothed = smooth_pyramidinal(block, window_length=101)
            py_out[:, start:end] = smoothed

    diff_full = max_abs_diff(py_out.T.astype(np.float64), r_step10)
    # Bit-exact: single centered direct convolution with pre-divided
    # triangular kernel (interior) + R-exact dynamic-denominator tail.
    assert diff_full < 1e-10, f"smooth_pyramidinal step10 max_diff={diff_full:.3e}"


# ============================================================================
# Helpers for Phase 2 tests (build AnnData from R fixture)
# ============================================================================

def _build_adata_from_fixture(raw_counts_df: "pd.DataFrame",
                               annot_df: "pd.DataFrame",
                               gene_order_df: "pd.DataFrame") -> "AnnData":
    """Build a minimal AnnData from the oligodendroglioma downsampled fixture.

    Returns a (cells x genes) AnnData with:
      - .X and .layers["counts"]: raw integer counts (cells x genes)
      - .obs["celltype"]: annotation label per cell
      - .var: gene_symbol index with chromosome/start/end columns
    """
    import anndata as ad
    import scipy.sparse as sp_sparse

    # Align genes to gene_order (R's CreateInfercnvObject convention)
    go = gene_order_df.copy()
    go = go[~go["chromosome"].isin({"chrX", "chrY", "chrM"})]
    go = go.drop_duplicates("gene_symbol").set_index("gene_symbol")
    common = [g for g in go.index if g in raw_counts_df.index]
    counts_sub = raw_counts_df.loc[common]  # genes x cells

    X = counts_sub.T.to_numpy(dtype=np.float32)  # cells x genes
    obs = pd.DataFrame(index=counts_sub.columns)
    obs["celltype"] = obs.index.map(dict(zip(annot_df["cell_id"], annot_df["annotation"])))

    var = go.loc[common].copy()
    var.index.name = "gene_symbol"

    adata = ad.AnnData(
        X=sp_sparse.csr_matrix(X),
        obs=obs,
        var=var,
    )
    adata.layers["counts"] = adata.X.copy()
    return adata


# Reference group names (non-tumor) derived from annotations
_REF_PATTERNS = ("malignant_", "Tumor_", "tumor_", "Observation", "observation")


def _ref_cats_from_adata(adata: "AnnData") -> list[str]:
    all_labels = adata.obs["celltype"].unique().tolist()
    return [a for a in all_labels if not any(p in a for p in _REF_PATTERNS)]


def _get_pipeline_gene_names(adata: "AnnData",
                              cutoff: float = 1.0,
                              min_cells_per_gene: int = 3,
                              chr_exclude: tuple = ("chrX", "chrY", "chrM")) -> list[str]:
    """Reconstruct ordered gene names as the infercnv pipeline sees them.

    Mirrors the pipeline's filter_low_expression_genes -> _build_chromosome_layout
    sequence so the resulting list aligns column-for-column with result.hmm_states.
    """
    import scipy.sparse as _sp
    from pyinfercnv.pipeline import _build_chromosome_layout

    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if not _sp.issparse(X):
        X = _sp.csr_matrix(np.asarray(X, dtype=np.float32))

    keep = filter_low_expression_genes(
        X, cutoff=cutoff, min_cells_per_gene=min_cells_per_gene,
    )
    var_kept = adata.var.iloc[np.where(keep)[0]].copy()

    _, gene_perm = _build_chromosome_layout(var_kept, chr_exclude)
    # gene_perm contains positional indices into var_kept rows
    ordered = [var_kept.index[i] for i in gene_perm]
    return ordered


def _compute_jaccard_floor(py_states: "np.ndarray",
                            r_mat: "np.ndarray",
                            py_cell_idx: list,
                            r_cell_idx: list,
                            py_gene_idx: list,
                            r_gene_idx: list,
                            neutral_py: int,
                            neutral_r: int) -> float:
    """Compute mean per-cell Jaccard of non-neutral bins."""
    py_aligned = py_states[np.ix_(py_cell_idx, py_gene_idx)]
    r_aligned = r_mat[np.ix_(r_gene_idx, r_cell_idx)].T  # r_mat is genes x cells

    jaccards = []
    for i in range(py_aligned.shape[0]):
        py_nn = set(int(x) for x in np.where(py_aligned[i] != neutral_py)[0])
        r_nn = set(int(x) for x in np.where(r_aligned[i] != neutral_r)[0])
        union = py_nn | r_nn
        jaccards.append(1.0 if len(union) == 0 else len(py_nn & r_nn) / len(union))
    return float(np.mean(jaccards))


# ============================================================================
# Phase 2 — Step 15: tumor subclusters ARI floor (OPERATIONAL ANCHOR ONLY)
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step15_subclusters"),
                    reason="r_out/step15_subclusters.tsv not generated; run Rscript tests/r_reference.R")
def test_step15_subclusters_ari_floor(raw_counts_all, annotations, gene_order):
    """Operational floor on Leiden subcluster ARI vs R step15. **Not a parity
    claim.** Retained as a regression anchor that catches Leiden-wiring bugs
    (the kind codex found in 2026-04-xx when ``random_state`` was silently
    dropped) — not as evidence of py-vs-R correctness.

    Why this is operational-only (HANDOFF v7 §2.1, Jason 2026-04-23 review):
    Leiden bucket IDs are arbitrary algorithm-internal labels with no
    cross-language semantic contract. ARI measures label-invariant
    permutation agreement, so a high ARI only means "same cells co-cluster"
    — not "same algorithm behaviour". The load-bearing parity metric is
    Spearman ρ on the continuous CNV matrix; see
    ``test_cnv_matrix_spearman_oligo`` below and
    ``benchmarks/phase2/Gao2021_Breast/cnv_matrix_spearman.md`` for the
    primary evidence.

    Floor 0.85 is the R-parity regression anchor (observed 1.000 on the
    oligodendroglioma fixture; bad values below 0.60 historically meant
    the seed wiring or the igraph C-core call had drifted).
    """
    try:
        import anndata  # noqa: F401
        from pyinfercnv import InferCNVConfig, infercnv  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pyinfercnv or anndata not importable: {exc}")

    try:
        adata = _build_adata_from_fixture(raw_counts_all, annotations, gene_order)
        ref_cats = _ref_cats_from_adata(adata)

        cfg = InferCNVConfig(cutoff=1, HMM=True, HMM_type="i6")
        result = infercnv(
            adata,
            config=cfg,
            reference_key="celltype",
            reference_cat=ref_cats,
            inplace=False,
        )
    except (TypeError, AttributeError, NotImplementedError) as exc:
        pytest.xfail(f"Phase 2 pipeline not yet integrated: {exc}")

    if result is None or result.subclusters is None:
        pytest.xfail("Phase 2 pipeline not yet integrated: result.subclusters is None")

    r_df = pd.read_csv(R_OUT_DIR / "step15_subclusters.tsv", sep="\t")
    r_label_map = dict(zip(r_df["cell_id"], r_df["subcluster"]))

    py_cells = list(adata.obs_names)
    common_cells = [c for c in py_cells if c in r_label_map]
    if len(common_cells) == 0:
        pytest.skip("No overlapping cell ids between Python output and R TSV")

    py_idx = [py_cells.index(c) for c in common_cells]
    py_labels = [str(result.subclusters[i]) for i in py_idx]
    r_labels = [r_label_map[c] for c in common_cells]

    ari = sklearn.metrics.adjusted_rand_score(r_labels, py_labels)
    print(f"  step15 ARI={ari:.3f}")
    # Floor 0.85 = spec §5.2 subcluster ARI target.
    # Observed at HEAD: 1.000 (python-igraph C-core + per-group partitioning,
    # seeded via ig.set_random_number_generator — commits f991ae0, 4b7206d).
    # Known bad values: 0.603 (pre-igraph-swap, scanpy/leidenalg backend).
    assert ari >= 0.85, f"step15 subclusters ARI={ari:.3f} below floor 0.85"


# ============================================================================
# Phase 2 — Step 17: HMM i6 Jaccard floor
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step17_hmm_i6"),
                    reason="r_out/step17_hmm_i6.tsv not generated; run Rscript tests/r_reference.R")
def test_step17_hmm_i6_jaccard_floor(raw_counts_all, annotations, gene_order):
    """Tier-3.5 Jaccard floor for HMM i6 state assignments.

    Mean per-cell Jaccard on non-neutral bins >= 0.60.
    R: states 1-6, neutral=3 (1-based). Python 0-based: neutral=2 (centre of 0-5).
    Genes aligned by reconstructing pipeline filter+chromosome-sort order.
    xfail when Phase 2 pipeline not yet integrated by Agent-E.
    """
    try:
        import anndata  # noqa: F401
        from pyinfercnv import InferCNVConfig, infercnv  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pyinfercnv or anndata not importable: {exc}")

    try:
        adata = _build_adata_from_fixture(raw_counts_all, annotations, gene_order)
        ref_cats = _ref_cats_from_adata(adata)

        cfg = InferCNVConfig(cutoff=1, HMM=True, HMM_type="i6")
        result = infercnv(
            adata,
            config=cfg,
            reference_key="celltype",
            reference_cat=ref_cats,
            inplace=False,
        )
    except (TypeError, AttributeError, NotImplementedError) as exc:
        pytest.xfail(f"Phase 2 pipeline not yet integrated: {exc}")

    if result is None or result.hmm_states is None:
        pytest.xfail("Phase 2 pipeline not yet integrated: result.hmm_states is None")

    # R TSV: genes x cells, states 1-6; convert to 0-based -> subtract 1
    r_df = pd.read_csv(R_OUT_DIR / "step17_hmm_i6.tsv", sep="\t", index_col=0)
    r_mat = r_df.to_numpy(dtype=np.int32) - 1  # genes x cells, now 0-based (0-5)
    r_genes = r_df.index.tolist()
    r_cells = r_df.columns.tolist()

    py_cells = list(adata.obs_names)
    common_cells = [c for c in py_cells if c in set(r_cells)]
    if len(common_cells) == 0:
        pytest.skip("No overlapping cell ids between Python output and R TSV")

    # Reconstruct pipeline gene ordering: filter_genes (all cells) -> chr-sorted.
    try:
        py_gene_names = _get_pipeline_gene_names(adata)
    except Exception as exc:
        pytest.skip(f"Cannot reconstruct pipeline gene order: {exc}")

    r_gene_set = set(r_genes)
    common_genes = [g for g in py_gene_names if g in r_gene_set]
    if len(common_genes) == 0:
        pytest.skip("No overlapping gene ids for HMM i6 Jaccard comparison")

    py_gene_idx = [py_gene_names.index(g) for g in common_genes]
    r_gene_idx = [r_genes.index(g) for g in common_genes]
    r_cell_idx = [r_cells.index(c) for c in common_cells]
    py_cell_idx = [py_cells.index(c) for c in common_cells]

    py_states = np.asarray(result.hmm_states, dtype=np.int32)  # (n_cells, n_bins)

    # i6: R neutral = 3 (1-based) -> 2 (0-based); Python centre = 2 (of 0-5)
    mean_jaccard = _compute_jaccard_floor(
        py_states, r_mat,
        py_cell_idx, r_cell_idx,
        py_gene_idx, r_gene_idx,
        neutral_py=2, neutral_r=2,
    )
    print(f"  step17 HMM i6 mean_jaccard={mean_jaccard:.3f}")
    # Floor 0.96: regression detector, not a bit-exact claim.
    # Observed after threading InferCNVConfig.random_state=42 through Phase 2:
    # 0.979 on this fixture. The R reference script uses set.seed(42); Python's
    # RNG stream is not bit-identical to R's, but using the same public seed
    # avoids the lower hspike calibration basin seen with random_state=0.
    # Cross-process variance measured 0 across 5 fresh uv-run invocations on this
    # fixture with a fixed seed (hspike NB sampling is deterministic under a
    # seeded Generator). 0.019 buffer absorbs future fixture changes / numba
    # fastmath ordering / hspike parameter tweaks without tripping CI.
    # Known bad values: 0.781 (pre-Option-1, ref-only Phase 1 filter), 0.713 (pre-C5 emission fix).
    # Spec §5.2 tier-4 target ~0.95 is exceeded; floor deliberately under target
    # so regression-vs-spec-drift can be distinguished.
    assert mean_jaccard >= 0.96, (
        f"step17 hmm_i6 mean_jaccard={mean_jaccard:.3f} below floor 0.96"
    )


# ============================================================================
# Phase 2 — Step 17: HMM i3 Jaccard floor
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step17_hmm_i3"),
                    reason="r_out/step17_hmm_i3.tsv not generated; run Rscript tests/r_reference.R")
def test_step17_hmm_i3_jaccard_floor(raw_counts_all, annotations, gene_order):
    """Tier-3.5 Jaccard floor for HMM i3 state assignments.

    Mean per-cell Jaccard on non-neutral bins >= 0.70.
    R: states 1-3, neutral=2 (1-based). Python 0-based: neutral=1 (centre of 0-2).
    i3 is deterministic-Z; tighter parity expected vs i6.
    xfail when Phase 2 pipeline not yet integrated by Agent-E.
    """
    try:
        import anndata  # noqa: F401
        from pyinfercnv import InferCNVConfig, infercnv  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pyinfercnv or anndata not importable: {exc}")

    try:
        adata = _build_adata_from_fixture(raw_counts_all, annotations, gene_order)
        ref_cats = _ref_cats_from_adata(adata)

        cfg = InferCNVConfig(cutoff=1, HMM=True, HMM_type="i3")
        result = infercnv(
            adata,
            config=cfg,
            reference_key="celltype",
            reference_cat=ref_cats,
            inplace=False,
        )
    except (TypeError, AttributeError, NotImplementedError) as exc:
        pytest.xfail(f"Phase 2 pipeline not yet integrated: {exc}")

    # i3 path stores states in result.hmm_states_i3 (G1 P1 schema separates i6/i3).
    if result is None or (result.hmm_states_i3 is None and result.hmm_states is None):
        pytest.xfail(
            "Phase 2 pipeline not yet integrated: both hmm_states_i3 and hmm_states are None"
        )

    r_df = pd.read_csv(R_OUT_DIR / "step17_hmm_i3.tsv", sep="\t", index_col=0)
    r_mat = r_df.to_numpy(dtype=np.int32) - 1  # 1-based -> 0-based (0-2); genes x cells
    r_genes = r_df.index.tolist()
    r_cells = r_df.columns.tolist()

    py_cells = list(adata.obs_names)
    common_cells = [c for c in py_cells if c in set(r_cells)]
    if len(common_cells) == 0:
        pytest.skip("No overlapping cell ids between Python output and R TSV")

    # Reconstruct pipeline gene ordering: filter_genes (all cells) -> chr-sorted.
    try:
        py_gene_names = _get_pipeline_gene_names(adata)
    except Exception as exc:
        pytest.skip(f"Cannot reconstruct pipeline gene order: {exc}")

    r_gene_set = set(r_genes)
    common_genes = [g for g in py_gene_names if g in r_gene_set]
    if len(common_genes) == 0:
        pytest.skip("No overlapping gene ids for HMM i3 Jaccard comparison")

    py_gene_idx = [py_gene_names.index(g) for g in common_genes]
    r_gene_idx = [r_genes.index(g) for g in common_genes]
    r_cell_idx = [r_cells.index(c) for c in common_cells]
    py_cell_idx = [py_cells.index(c) for c in common_cells]

    # For i3, use hmm_states_i3 if populated; fall back to hmm_states
    py_states_raw = result.hmm_states_i3 if result.hmm_states_i3 is not None else result.hmm_states
    py_states = np.asarray(py_states_raw, dtype=np.int32)  # (n_cells, n_bins)

    # i3: R neutral = 2 (1-based) -> 1 (0-based); Python centre = 1 (of 0-2)
    mean_jaccard = _compute_jaccard_floor(
        py_states, r_mat,
        py_cell_idx, r_cell_idx,
        py_gene_idx, r_gene_idx,
        neutral_py=1, neutral_r=1,
    )
    print(f"  step17 HMM i3 mean_jaccard={mean_jaccard:.3f}")
    # Floor 0.99: regression detector, not a spec restatement.
    # Observed at HEAD: 1.000 after switching Phase 2's i3 branch to consume
    # cnv_matrix_fc (post-step14 invert_log2 + step16 outlier-prune linear FC).
    # That lifted the metric from 0.976 → 1.0000 on the oligodendroglioma
    # fixture (see HANDOFF_bit_exact.md diagnostic and inferCNV_HMM.R:366
    # which reads @expr.data in linear-FC space at step 17).
    # 0.01 buffer absorbs future igraph C PRNG stream shifts / smooth
    # accumulation drift. Known bad values: 0.976 (log2-space HMM input),
    # 0.750 (pre-Option-1), 0.706 (pre-C5/C6 fixes).
    assert mean_jaccard >= 0.99, (
        f"step17 hmm_i3 mean_jaccard={mean_jaccard:.3f} below floor 0.99"
    )


# ============================================================================
# Phase 3 — skeleton parity gates (pytest.skip until implementations land)
# ============================================================================
#
# These three tests are wired in SKELETON form. They skip cleanly until the
# matching R TSVs exist AND the Python Phase 3 modules are implemented (i.e.
# :attr:`InferCNVResult.bayes_posterior` / ``de_mask`` / ``denoised_matrix``
# are populated by :func:`pyinfercnv.pipeline_phase3.run_phase3`).
#
# Step numbering uses R's `step_count` convention from `inferCNV_ops.R`:
#   step 18 = BayesNet Gibbs (Agent B1)
#   step 19 = filterHighPNormals (post-BayesNet HMM-state override; B1 too)
#   step 21 = mask_non_DE_genes (Agent B2)
#   step 22 = clear_noise_via_ref_mean_sd / clear_noise (Agent B3)
# (step 20 = assign_HMM_states_to_proxy_expr_vals; orchestrator-inline,
#  no separate parity gate.)
#
# Second round acceptance criteria (plan doc §8):
#   * BayesNet — cell_prob vs R |ΔP| < 0.05 (tier 3.5 empirical; RNG-inequiv.)
#   * mask_non_DE — bit-exact (max_diff < 1e-10) on the masked matrix;
#                   upstream p-values within 1e-10 if the R jitter is disabled.
#   * denoise   — bit-exact (max_diff < 1e-10); pure mean/sd/clip arithmetic.
# ============================================================================


def _build_bayesnet_inputs(raw_counts_all, annotations, gene_order):
    """Shared helper for step18 / step19 tests.

    Runs Phase 1 (with HMM=False to preserve ``cpm_matrix_f32``) + explicit
    :func:`run_phase2`, then re-derives an :class:`HspikeCalibration` on the
    same normalized cpm matrix. Returns ``(adata, cfg, result_p2, cal)``.

    This avoids the ``result.cpm_matrix_f32 = None`` cleanup in the top-level
    :func:`infercnv` when ``HMM=True`` (``pipeline.py:313``); the hspike
    calibration is needed by :func:`run_bayesnet_gibbs` but is not persisted
    on the :class:`InferCNVResult` by Phase 2.
    """
    import anndata  # noqa: F401
    from pyinfercnv import InferCNVConfig, infercnv
    from pyinfercnv.hmm.hspike import calibrate_i6_emission
    from pyinfercnv.pipeline_phase2 import run_phase2

    adata = _build_adata_from_fixture(raw_counts_all, annotations, gene_order)
    ref_cats = _ref_cats_from_adata(adata)

    cfg = InferCNVConfig(
        cutoff=1, HMM=True, HMM_type="i6", BayesMaxPNormal=0.5,
        random_state=42,
    )

    cfg_p1 = InferCNVConfig(
        cutoff=cfg.cutoff, HMM=False, HMM_type="i6",
        random_state=cfg.random_state,
    )
    result_p1 = infercnv(
        adata, config=cfg_p1,
        reference_key="celltype", reference_cat=ref_cats,
        inplace=False,
    )
    if result_p1 is None or result_p1.cpm_matrix_f32 is None:
        pytest.xfail("Phase 1 pipeline did not produce cpm_matrix_f32")

    result_p2 = run_phase2(
        result_p1, adata, config=cfg,
        reference_key="celltype", reference_cat=ref_cats,
        random_state=cfg.random_state,
    )
    if result_p2.hmm_states is None or result_p2.cnv_regions is None:
        pytest.xfail("Phase 2 produced no HMM states / cnv_regions")

    is_ref = result_p1.cell_meta["is_reference"].to_numpy(dtype=bool)
    cell_idx_all = np.arange(result_p1.n_cells)
    celltypes = adata.obs["celltype"].to_numpy()
    ref_groups_global = {
        lab: cell_idx_all[(celltypes == lab) & is_ref]
        for lab in np.unique(celltypes[is_ref])
    }
    obs_groups_global = {
        lab: cell_idx_all[(celltypes == lab) & ~is_ref]
        for lab in np.unique(celltypes[~is_ref])
    }
    cal = calibrate_i6_emission(
        result_p1.cpm_matrix_f32, ref_groups_global,
        observation_groups=obs_groups_global,
        config=cfg, random_state=cfg.random_state,
    )
    return adata, cfg, result_p2, cal


def test_step18_bayesnet_parity(raw_counts_all, annotations, gene_order):
    """Phase 3 BayesNet posterior parity gate (Agent B1).

    Contract (see ``pyinfercnv/bayesnet/gibbs.py`` module docstring):
      * R's ``inferCNVBayesNet`` uses JAGS/BUGS Gibbs over the mixture model
        ``gexp ~ Normal(mu[state], sd[state]); epsilon ~ Categorical(theta);
        theta ~ Dirichlet(1..1)`` per CNV region independently.
      * Python ports the same model via numba + PCG-64 RNG. The R / numpy
        RNG streams are not bit-equivalent, so we assert on posterior
        probabilities, not trace reproduction.
      * Floor (per ``docs/superpowers/plans/2026-04-24-phase3-start.md §2.5``):
        median per-region max-|ΔP| ≤ 0.15 (first-round regression anchor;
        stretch 0.05, soft 0.10).

    Alignment:
      * Python ``cnv_regions`` and R's ``cnv_region_name`` (from
        MCMC_inferCNV_obj's ``cell_gene`` slot) are both keyed by
        (subcluster, chromosome, contiguous-state-run). We match by
        chromosome and take in-order positional pairs per chromosome; this
        is correct when step17 Jaccard ≥ 0.96 (already asserted).
    """
    if not _r_step_available("step18_bayes_cnv_prob"):
        pytest.skip(
            "Phase 3 step 18 TSV not yet generated. Run "
            "`Rscript tests/r_reference.R` to produce step18_bayes_cnv_prob.tsv."
        )
    if not _r_step_available("step18_regions_meta"):
        pytest.skip("step18_regions_meta.tsv missing — cannot align regions py↔R")

    try:
        import anndata  # noqa: F401
        from pyinfercnv import InferCNVConfig, infercnv  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pyinfercnv or anndata not importable: {exc}")

    from pyinfercnv.bayesnet import _step18_bayesnet

    adata, cfg, result, cal = _build_bayesnet_inputs(
        raw_counts_all, annotations, gene_order
    )

    # --- Run Py step 18 + 19 ---
    import time as _time
    t0 = _time.time()
    gibbs_result, filtered_states = _step18_bayesnet(
        result, cfg, i6_calibration=cal,
        numBurnin=1000, numSamples=1000, numChains=3,
    )
    elapsed = _time.time() - t0
    print(f"  step18 gibbs elapsed: {elapsed:.1f}s "
          f"(n_regions={len(result.cnv_regions)})")

    py_cnv = np.asarray(gibbs_result["cnv_posterior"], dtype=np.float64)
    n_regions_py = py_cnv.shape[0]

    # --- Load R step18 side ---
    r_cnv_df = pd.read_csv(
        R_OUT_DIR / "step18_bayes_cnv_prob.tsv", sep="\t", index_col=0
    )
    r_cnv = r_cnv_df.to_numpy(dtype=np.float64)  # (n_regions_r, K)
    r_meta_df = pd.read_csv(R_OUT_DIR / "step18_regions_meta.tsv", sep="\t")
    print(f"  R regions: {r_cnv.shape[0]}, Py regions: {n_regions_py}")

    # --- Align regions py ↔ R by cnv_region_name ---
    # R's cnv_region_name pattern (from infercnv's .get_predicted_CNV_regions)
    # is typically "{subcluster}.chr{NN}.{START}-{END}.<state_name>". We try
    # to match on (subcluster, chromosome) pairs; if that's not sufficient,
    # fall back to positional index.
    #
    # For this first-round parity test, we perform a greedy by-chromosome
    # match — group regions by chromosome and match Python regions in-order
    # to R regions in-order. This is correct when Viterbi traces agree
    # step17 Jaccard ≥ 0.96 (already asserted by test_step17_hmm_i6).
    # R `cnv_region_name` format: "chr<N>-region_<idx>". Pull chromosome token.
    r_chrs = r_meta_df["cnv_region_name"].str.extract(r"^(chr[^-\.]+)")[0].tolist()
    r_ncells = r_meta_df["n_cells"].tolist()
    r_ngenes = r_meta_df["n_genes"].tolist()

    # Py cnv_regions: (cell_group, subcluster, chromosome, ...) rows.
    py_chrs = result.cnv_regions["chromosome"].tolist()
    py_subs = result.cnv_regions["subcluster"].astype(int).tolist()
    py_ngenes = [
        int(end - start + 1) for start, end in zip(
            result.cnv_regions["bin_start"].tolist(),
            result.cnv_regions["bin_end"].tolist(),
        )
    ]

    # Count cells per py subcluster via result.subclusters (same length as n_cells)
    py_sub_arr = np.asarray(result.subclusters)
    py_sub_size = {int(s): int((py_sub_arr == s).sum()) for s in np.unique(py_sub_arr)}

    # Group both sides by (n_cells/sub_size, chromosome). Inside each bucket,
    # pair by **minimum gene-count delta** — R and Py should share both the
    # subcluster (via cell count) and the chromosome, and each region's
    # n_genes should be identical or very close (the Viterbi trace decides
    # region boundaries, and step17 Jaccard ≥ 0.96 guarantees close traces).
    from collections import defaultdict
    r_by_key: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    for i, (c, n, g) in enumerate(zip(r_chrs, r_ncells, r_ngenes)):
        r_by_key[(int(n), c)].append((i, int(g)))

    py_by_key: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    for i, (c, s, g) in enumerate(zip(py_chrs, py_subs, py_ngenes)):
        py_by_key[(py_sub_size.get(s, -1), c)].append((i, int(g)))

    # Greedy best-match by gene-count: within each bucket, assign each R
    # region to the closest-unassigned Py region by |Δ n_genes|. Drop any R
    # region whose best Py match has gene-count within max(50, 20% * r_g).
    pairs = []
    unmatched_r = 0
    for k, r_list in r_by_key.items():
        py_list = py_by_key.get(k, [])
        if not py_list:
            unmatched_r += len(r_list)
            continue
        py_avail = list(py_list)
        for r_idx, r_g in r_list:
            if not py_avail:
                unmatched_r += 1
                continue
            best_j = min(range(len(py_avail)),
                         key=lambda j: abs(py_avail[j][1] - r_g))
            p_idx, p_g = py_avail[best_j]
            tol = max(50, int(0.5 * r_g))
            if abs(p_g - r_g) <= tol:
                pairs.append((p_idx, r_idx))
                py_avail.pop(best_j)
            else:
                unmatched_r += 1

    if not pairs:
        pytest.skip("No aligned regions between Py and R (subcluster/chromosome mismatch)")

    diffs_per_state = np.empty((len(pairs), 6), dtype=np.float64)
    for i, (p, r) in enumerate(pairs):
        diffs_per_state[i] = np.abs(py_cnv[p] - r_cnv[r])
    per_region_max = diffs_per_state.max(axis=1)

    max_diff = float(per_region_max.max())
    median_diff = float(np.median(per_region_max))
    q90 = float(np.quantile(per_region_max, 0.90))
    stretch_rate = float(np.mean(per_region_max < 0.05))
    soft_rate = float(np.mean(per_region_max < 0.10))
    print(
        f"  step18 aligned pairs: {len(pairs)} "
        f"(R {r_cnv.shape[0]}, Py {n_regions_py}; {unmatched_r} R regions unpaired)"
    )
    print(
        f"    max|ΔP|={max_diff:.3f}  median={median_diff:.3f}  q90={q90:.3f}"
    )
    print(
        f"    stretch-tier (<0.05) rate: {stretch_rate*100:.1f}%; "
        f"soft-tier (<0.10) rate: {soft_rate*100:.1f}%"
    )

    # Assertion: soft-tier pass rate ≥ 90% of matched pairs. The task
    # brief (plan §2.5) defines stretch 0.05 / soft 0.10 tiers and
    # forbids tuning past them. A handful of outliers can arise from
    # residual alignment ambiguity (R and Py region boundaries differ by
    # a few bins when Viterbi traces shift slightly) plus MCMC noise on
    # borderline regions; the 90% rate filters those while still
    # detecting algorithmic regressions. Observed at HEAD on oligo:
    # soft-tier 95.5%, stretch-tier 88.6% (with the gene-count-aware
    # alignment; the naive in-order pairing reports a misleading 0.82
    # "max" from cross-bin mismatches).
    assert soft_rate >= 0.90, (
        f"step18 soft-tier (<0.10) pass rate {soft_rate*100:.1f}% below 90% "
        f"floor. max={max_diff:.3f} median={median_diff:.3f} q90={q90:.3f}"
    )


def test_step19_filter_high_p_normals(raw_counts_all, annotations, gene_order):
    """Phase 3 step 19 — filterHighPNormals parity gate (Agent B1).

    Verifies that the Python :func:`filter_high_p_normals` override produces
    a post-filter HMM state matrix whose Jaccard vs R's step19_hmm_i6_filtered
    matches the Phase-2 step17 Jaccard within a small delta (BayesNet should
    push some noisy regions to neutral but not destroy the bulk of the
    Viterbi trace).

    Floor: mean per-cell Jaccard ≥ 0.94 (0.02 below the step17 0.96 floor)
    on shared cells/genes. Observed degradation budget of 0.02 accounts for
    RNG-noise disagreements on borderline regions.
    """
    if not _r_step_available("step19_hmm_i6_filtered"):
        pytest.skip(
            "Phase 3 step 19 TSV not yet generated. Run "
            "`Rscript tests/r_reference.R`."
        )
    if not _r_step_available("step18_bayes_cnv_prob"):
        pytest.skip("step18 TSVs missing — cannot drive the filter")

    try:
        import anndata  # noqa: F401
        from pyinfercnv import InferCNVConfig, infercnv  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pyinfercnv or anndata not importable: {exc}")

    from pyinfercnv.bayesnet import _step18_bayesnet

    adata, cfg, result, cal = _build_bayesnet_inputs(
        raw_counts_all, annotations, gene_order
    )

    _, filtered = _step18_bayesnet(
        result, cfg, i6_calibration=cal,
        numBurnin=1000, numSamples=1000, numChains=3,
    )

    r_df = pd.read_csv(R_OUT_DIR / "step19_hmm_i6_filtered.tsv", sep="\t", index_col=0)
    r_mat = r_df.to_numpy(dtype=np.int32) - 1  # 1-based → 0-based
    r_genes = r_df.index.tolist()
    r_cells = r_df.columns.tolist()

    py_cells = list(adata.obs_names)
    common_cells = [c for c in py_cells if c in set(r_cells)]
    if not common_cells:
        pytest.skip("No shared cell ids between Py and R step19")
    try:
        py_gene_names = _get_pipeline_gene_names(adata)
    except Exception as exc:
        pytest.skip(f"Cannot reconstruct pipeline gene order: {exc}")
    common_genes = [g for g in py_gene_names if g in set(r_genes)]
    if not common_genes:
        pytest.skip("No shared gene ids between Py and R step19")

    py_gene_idx = [py_gene_names.index(g) for g in common_genes]
    r_gene_idx = [r_genes.index(g) for g in common_genes]
    r_cell_idx = [r_cells.index(c) for c in common_cells]
    py_cell_idx = [py_cells.index(c) for c in common_cells]

    mean_jaccard = _compute_jaccard_floor(
        filtered.astype(np.int32), r_mat,
        py_cell_idx, r_cell_idx,
        py_gene_idx, r_gene_idx,
        neutral_py=2, neutral_r=2,
    )
    print(f"  step19 post-filter Jaccard={mean_jaccard:.3f}")
    assert mean_jaccard >= 0.94, (
        f"step19 post-filter Jaccard={mean_jaccard:.3f} below floor 0.94"
    )


def test_step21_mask_non_DE_parity():  # noqa: N802 — matches R name
    """Phase 3 step 21 — mask_non_DE genes bit-exact parity (Agent B2).

    Contract (see ``pyinfercnv/mask_de/wilcoxon.py`` module docstring):
    the R fixture dumper monkey-patches ``infercnv:::get_DE_genes_basic``
    to strip the ``rnorm()`` tie-breaking jitter and force
    ``exact = FALSE`` on ``wilcox.test``. With that patch,
    ``scipy.stats.mannwhitneyu(method="asymptotic", use_continuity=True)``
    matches R to machine precision on per-gene p-values, and the final
    mask position set is **bit-exact**.

    We replay step 21 in Python on the R fixture's post-step 16 input,
    using R's own reference/tumor-subcluster partition (from
    ``step15_subclusters.tsv`` + annotations). Assertions:

    1. The masked matrix matches R at ``max_diff < 1e-10`` (bit-exact).
    2. The mask position set is identical (``np.array_equal``).
    """
    if not _r_step_available("step21_mask_nonDE"):
        pytest.skip(
            "Phase 3 step 21 TSV not yet generated. Run "
            "`Rscript tests/r_reference.R` to produce step21_mask_nonDE.tsv."
        )
    # R `phase3_run` default `prune_outliers=FALSE`, so R step 21 runs on
    # **step 14 invert_log2** output (step 16 prune is skipped). Using
    # step16_outlier_pruned as py input would mismatch R by ~5e-1. See
    # denoise agent (B3) who flagged this for step22 first.
    if not _r_step_available("step14_invert"):
        pytest.skip("step14_invert.tsv missing — cannot build py input")
    # Prefer step21's own subcluster dump (same run() invocation → same
    # RNG state); fall back to step15 only if the step21 dump is missing.
    sub_fixture = (
        "step21_subclusters"
        if _r_step_available("step21_subclusters")
        else "step15_subclusters"
    )
    if not _r_step_available(sub_fixture):
        pytest.skip(
            "Neither step21_subclusters.tsv nor step15_subclusters.tsv "
            "present — need an R subcluster partition"
        )

    from pyinfercnv.mask_de import mask_non_DE_genes

    r_step14, r_genes, r_cells = _load_r_step("step14_invert")
    r_step21, r_step21_genes, r_step21_cells = _load_r_step("step21_mask_nonDE")

    assert r_genes == r_step21_genes, "gene order drift between step14 and step21"
    assert r_cells == r_step21_cells, "cell order drift between step14 and step21"

    # Build reference_group_indices from annotations.
    annot_df = pd.read_csv(R_ANNOT, sep="\t", header=None,
                           names=["cell_id", "annotation"])
    annot_by_cell = dict(zip(annot_df["cell_id"], annot_df["annotation"]))
    tumor_pats = ("malignant_", "Tumor_", "tumor_", "Observation", "observation")
    ref_group_indices: dict[str, list[int]] = {}
    for i, c in enumerate(r_cells):
        lab = annot_by_cell.get(c, "")
        if any(p in lab for p in tumor_pats):
            continue
        ref_group_indices.setdefault(lab, []).append(i)
    ref_group_idx_arrays = {
        k: np.asarray(v, dtype=np.int64) for k, v in ref_group_indices.items()
    }

    # Build tumor_subcluster_indices from the chosen subcluster TSV
    # (R's own partition — same run() invocation when possible).
    sub_df = pd.read_csv(R_OUT_DIR / f"{sub_fixture}.tsv", sep="\t")
    sub_by_cell = dict(zip(sub_df["cell_id"], sub_df["subcluster"]))
    tumor_sub_indices: dict[str, list[int]] = {}
    for i, c in enumerate(r_cells):
        if any(p in annot_by_cell.get(c, "") for p in tumor_pats):
            sc = sub_by_cell.get(c)
            if sc is None:
                continue
            tumor_sub_indices.setdefault(sc, []).append(i)
    tumor_sub_idx_arrays = {
        k: np.asarray(v, dtype=np.int64) for k, v in tumor_sub_indices.items()
    }

    # Python input: R step 14 matrix (linear FC, genes × cells) → transpose
    # to cells × genes for the py API.
    expr_py = r_step14.T.astype(np.float64)
    masked_py, de_mask_py = mask_non_DE_genes(
        expr_py,
        tumor_subcluster_indices=tumor_sub_idx_arrays,
        reference_group_indices=ref_group_idx_arrays,
        mask_nonDE_pval=0.05,
        test_use="wilcoxon",
        require_DE_all_normals="any",
    )

    # Compare genes × cells orientation.
    py_genes_x_cells = masked_py.T  # (genes, cells)
    diff = max_abs_diff(py_genes_x_cells, r_step21)
    print(f"  step21 mask_non_DE max_diff={diff:.3e}")
    assert diff < 1e-10, f"step21 mask_non_DE max_diff={diff:.3e}"

    # Mask identity sanity check: where R replaced with center_val, py
    # should also. Inferred indirectly via np.isclose since R doesn't dump
    # the mask boolean separately; tolerates coincidental matches (values
    # that happen to equal center_val on unmasked positions). The max_diff
    # check above is the authoritative bit-exact contract.
    center_val_r = float(np.mean(r_step14))
    r_masked_positions = np.isclose(r_step21, center_val_r, atol=1e-10)
    py_masked_positions = (~de_mask_py).T  # genes × cells
    xor = int((r_masked_positions ^ py_masked_positions).sum())
    total = int(r_masked_positions.size)
    xor_frac = xor / total
    print(
        f"  step21 mask identity: xor={xor}/{total} ({xor_frac:.2e}); "
        f"py-masked={int(py_masked_positions.sum())}, "
        f"r-masked={int(r_masked_positions.sum())}"
    )
    # Soft floor: coincidental-value false positives in the R-side np.isclose
    # inference are unavoidable. Cap at 0.01% — any larger divergence is a
    # real mask bug. Observed on oligo: 72/1,565,472 ≈ 4.6e-5.
    assert xor_frac < 1e-4, (
        f"step21 mask identity divergence xor_frac={xor_frac:.2e} > 1e-4 floor"
    )


def test_step22_noise_reduction_parity():
    """Phase 3 step22 ``clear_noise_via_ref_mean_sd`` parity (Agent B3).

    **Tier-4 bit-exact** (``max_diff < 1e-10``): pure mean / per-cell sd /
    boolean clip — no RNG, no ties. We feed the post-step14 linear-FC
    matrix (R ``step14_invert.tsv``) as the denoise input.

    Why step14 and not step16: R's default ``prune_outliers`` is
    ``FALSE`` (``inferCNV_ops.R:327``), so the full ``infercnv::run()``
    invocation that produced ``step22_denoised.tsv`` SKIPS the step-16
    outlier-prune block entirely. ``infercnv_obj@expr.data`` at step 22
    is therefore the unmodified post-``invert_log2`` (step 14) linear
    FC matrix. Steps 17-20 mutate a *separate* ``hmm.infercnv_obj``;
    steps 18/19/21 are OFF in this parity gate
    (``BayesMaxPNormal=0``, ``mask_nonDE_genes=FALSE``) — so step 14
    → step 22 is the correct pre→post pair.

    R source: ``inferCNV_ops.R:2302-2346`` (``clear_noise_via_ref_mean_sd``),
    wired at ``inferCNV_ops.R:1560-1589``.
    """
    if not _r_step_available("step22_denoised"):
        pytest.skip(
            "step22_denoised.tsv not generated; run Rscript tests/r_reference.R"
        )
    if not _r_step_available("step14_invert"):
        pytest.skip(
            "step14_invert.tsv (denoise input) missing; "
            "run Rscript tests/r_reference.R"
        )

    from pyinfercnv.denoise import denoise_by_ref_mean_sd

    # Load R input (genes x cells) → transpose to cells x bins for Python
    # convention, float64 end-to-end to preserve bit-exactness.
    r_step_in, _, r_cells_in = _load_r_step("step14_invert")
    r_step22, _, r_cells_out = _load_r_step("step22_denoised")

    assert r_cells_in == r_cells_out, (
        "step14 / step22 cell order drift — R should preserve column "
        "order through the pipeline"
    )

    # Resolve reference cell indices against the R TSV cell order.
    # The fixture generator uses the same tumor-pattern heuristic as
    # ``tests/r_reference.R`` lines 33-35.
    import pandas as _pd

    annot_df = _pd.read_csv(R_ANNOT, sep="\t", header=None,
                            names=["cell_id", "annotation"])
    annot_by_cell = dict(zip(annot_df["cell_id"], annot_df["annotation"]))
    tumor_pats = ("malignant_", "Tumor_", "tumor_", "Observation", "observation")
    ref_idx = [
        i for i, c in enumerate(r_cells_in)
        if not any(p in annot_by_cell.get(c, "") for p in tumor_pats)
    ]
    # Oligo fixture has exactly 19 Microglia/Macrophage + 23
    # Oligodendrocytes (non-malignant) = 42 reference cells. Any drift
    # here would mean the substring heuristic mis-tagged a label; pin
    # the count so mis-tagging produces an actionable failure rather
    # than an opaque 1e-5 max_diff.
    assert len(ref_idx) == 42, (
        f"oligo reference cell count drifted: expected 42, got {len(ref_idx)}; "
        "the test's substring-heuristic may be mis-tagging a label"
    )

    # Python layout: cells × bins. Call denoise.
    py_in = r_step_in.T.astype(np.float64, copy=True)  # (cells, genes)
    py_out = denoise_by_ref_mean_sd(
        py_in,
        np.asarray(ref_idx, dtype=np.intp),
        noise_filter=None,
        sd_amplifier=1.5,
        noise_logistic=False,
    )
    # Back to genes × cells to line up with R TSV.
    py_back = py_out.T

    # Sanity guard: denoise must actually flatten *something*; otherwise
    # a silent no-op (e.g. band collapsed to 0) would pass the 1e-10
    # floor trivially by comparing step16 unchanged on both sides.
    assert not np.array_equal(py_in, py_out), (
        "denoise produced a no-op result — band likely collapsed, "
        "refusing to claim bit-exact parity on a trivial identity"
    )

    diff = max_abs_diff(py_back.astype(np.float64), r_step22)
    # Bit-exact: ddof=1 sd + float64 mean + strict-inequality mask. The
    # 1e-10 floor is the standard Phase 1/2 bit-exact bar; no known loose
    # ends (ref-mean arithmetic is order-insensitive and Python/R both
    # evaluate ``mean(apply(vals, 2, sd))`` as mean-over-cells of
    # Bessel-corrected per-cell sd).
    print(f"  step22 denoise max_diff={diff:.3e}")
    assert diff < 1e-10, f"step22 denoise max_diff={diff:.3e}"
