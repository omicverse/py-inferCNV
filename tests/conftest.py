"""Shared pytest fixtures for pyinfercnv."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse as sp


REPO_ROOT = Path(__file__).resolve().parent.parent
R_OUT_DIR = REPO_ROOT / "tests" / "r_out"
R_FIXTURE_COUNTS = Path(
    "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata/"
    "oligodendroglioma_expression_downsampled.counts.matrix.gz"
)
R_FIXTURE_ANNOT = Path(
    "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata/"
    "oligodendroglioma_annotations_downsampled.txt"
)
R_FIXTURE_GENE_ORDER = Path(
    "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata/"
    "gencode_downsampled.EXAMPLE_ONLY_DONT_REUSE.txt"
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=0)


@pytest.fixture
def small_synthetic_counts(rng) -> sp.csr_matrix:
    dense = rng.negative_binomial(n=5, p=0.3, size=(100, 500)).astype(np.float32)
    return sp.csr_matrix(dense)


@pytest.fixture
def small_synthetic_adata(small_synthetic_counts, rng) -> AnnData:
    X = small_synthetic_counts
    adata = AnnData(X=X.copy())
    adata.layers["counts"] = X.copy()
    adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(adata.n_vars)]

    chroms = [f"chr{1 + (i // 25)}" for i in range(adata.n_vars)]
    starts = np.arange(adata.n_vars) * 1000
    ends = starts + 500
    adata.var["chromosome"] = chroms
    adata.var["start"] = starts
    adata.var["end"] = ends
    return adata


@pytest.fixture
def r_out_dir() -> Path:
    return R_OUT_DIR


def r_reference_available(name: str) -> bool:
    return (R_OUT_DIR / f"{name}.tsv").exists()
