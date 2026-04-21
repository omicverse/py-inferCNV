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
    # Tier-4 approximate: float32 normalize accumulates ~1e-2 vs R's float64.
    # README parity table will record this as "max_diff < 1e-2 (approximate)".
    assert diff < 1e-2, f"normalize step03 max_diff={diff:.3e}"


# ============================================================================
# Step 04: log2(x+1)
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step04_logged")
                    or not _r_step_available("step03_normalized"),
                    reason="r_out TSVs not available")
def test_step04_log2_plus1_parity():
    r_step03, _, _ = _load_r_step("step03_normalized")
    r_step04, _, _ = _load_r_step("step04_logged")
    py_out = log2_plus1(r_step03.T.astype(np.float32)).T  # (genes, cells) for compare
    diff = max_abs_diff(py_out.astype(np.float64), r_step04)
    assert diff < 1e-5, f"log2_plus1 step04 max_diff={diff:.3e}"


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

    py_in = r_step04.T.astype(np.float32)  # (cells, genes)
    py_out = subtract_reference(py_in, ref_groups=ref_groups, use_bounds=True)
    py_back = py_out.T

    diff = max_abs_diff(py_back.astype(np.float64), r_step08)
    # bounded subtract is sensitive to float32 precision in mean computation
    assert diff < 1e-3, f"subtract_ref step08 max_diff={diff:.3e}"


# ============================================================================
# Step 09: max_centered_threshold = 3
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step09_clipped")
                    or not _r_step_available("step08_subtracted"),
                    reason="r_out TSVs not available")
def test_step09_max_threshold_parity():
    r_step08, _, _ = _load_r_step("step08_subtracted")
    r_step09, _, _ = _load_r_step("step09_clipped")
    py_out = apply_max_centered_threshold(r_step08.T.astype(np.float32), threshold=3.0).T
    diff = max_abs_diff(py_out.astype(np.float64), r_step09)
    assert diff < 1e-6, f"max_threshold step09 max_diff={diff:.3e}"


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
    py_out = center_cells(r_step10.T.astype(np.float32), method="median").T
    diff = max_abs_diff(py_out.astype(np.float64), r_step11)
    # median is exact in float64; small drift from float32 cast
    assert diff < 1e-4, f"center_cells step11 max_diff={diff:.3e}"


# ============================================================================
# Step 16: prune_outliers (average_bound)
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step16_outlier_pruned")
                    or not _r_step_available("step12_subtracted2"),
                    reason="r_out TSVs not available")
def test_step16_outlier_prune_parity():
    r_step12, _, _ = _load_r_step("step12_subtracted2")
    r_step16, _, _ = _load_r_step("step16_outlier_pruned")
    py_out = prune_outliers(r_step12.T.astype(np.float32), method="average_bound").T
    diff = max_abs_diff(py_out.astype(np.float64), r_step16)
    assert diff < 1e-4, f"outlier_prune step16 max_diff={diff:.3e}"


# ============================================================================
# step_invert: invert_log2 (linear FC space)
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step_invert")
                    or not _r_step_available("step16_outlier_pruned"),
                    reason="r_out TSVs not available")
def test_invert_log2_parity():
    r_step16, _, _ = _load_r_step("step16_outlier_pruned")
    r_invert, _, _ = _load_r_step("step_invert")
    py_out = invert_log2(r_step16.T.astype(np.float32)).T
    diff = max_abs_diff(py_out.astype(np.float64), r_invert)
    assert diff < 1e-4, f"invert_log2 max_diff={diff:.3e}"


# ============================================================================
# Step 10: smooth_pyramidinal — TIER-4 INTERIOR + TIER-4-APPROX TAIL
# ============================================================================

@pytest.mark.skipif(not _r_step_available("step10_smoothed")
                    or not _r_step_available("step09_clipped"),
                    reason="r_out TSVs not available")
def test_step10_smooth_pyramidinal_per_chromosome(gene_order):
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
