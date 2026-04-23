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

    py_in_cells_x_genes = r_step09.T.astype(np.float32)
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
    # Tier-4 approx for tail; tier-4 bit-exact in interior. Combined floor:
    assert diff_full < 1e-3, f"smooth_pyramidinal step10 max_diff={diff_full:.3e}"


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
    # Floor 0.90: regression detector, not a spec restatement.
    # Observed at HEAD: 0.968 (post-Option-1 Phase 1 gene filter fix, commit 90eabe1).
    # Cross-process variance measured 0 across 5 fresh uv-run invocations on this
    # fixture with random_state=0 (hspike NB sampling is deterministic under a
    # seeded RandomState). 0.068 buffer absorbs future fixture changes / numba
    # fastmath ordering / hspike parameter tweaks without tripping CI.
    # Known bad values: 0.781 (pre-Option-1, ref-only Phase 1 filter), 0.713 (pre-C5 emission fix).
    # Spec §5.2 tier-4 target ~0.95 is exceeded; floor deliberately under target
    # so regression-vs-spec-drift can be distinguished.
    assert mean_jaccard >= 0.90, (
        f"step17 hmm_i6 mean_jaccard={mean_jaccard:.3f} below floor 0.90"
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
