"""pyinfercnv — 30-second quickstart on synthetic data.

Self-contained: no fixture download, no R install, no genome parquet.
Builds a 100-cell × 500-gene NB-distributed synthetic AnnData with two
celltypes (reference + tumor), runs the full pipeline (Phase 1 + 2 + 3
with HMM + denoise on), prints the result schema, and saves a heatmap.

Run::

    uv run python examples/quickstart_30s.py

Expected output: ``quickstart_30s_heatmap.png`` in the cwd, plus a
console summary of populated InferCNVResult fields. Total wallclock
~10-30 s on a developer laptop (Phase 2 HMM dominates; Phase 3 is fast).

Use ``examples/tutorial_phase{1,2,3}.ipynb`` for the full walk-through
on the R inferCNV oligodendroglioma fixture.
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from scipy import sparse as sp

from pyinfercnv import InferCNVConfig, infercnv


def make_synth_adata(
    *, n_cells: int = 100, n_genes: int = 500, n_ref: int = 30, seed: int = 0
) -> AnnData:
    """Synthetic NB-distributed AnnData with chromosome metadata.

    Mirrors ``tests/conftest.py::small_synthetic_adata`` plus a celltype
    column so the pipeline has a reference handle.
    """
    rng = np.random.default_rng(seed)
    X = rng.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes)).astype(np.float32)
    adata = AnnData(X=sp.csr_matrix(X))
    adata.layers["counts"] = sp.csr_matrix(X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    n_chroms = 5
    genes_per_chr = n_genes // n_chroms
    adata.var["chromosome"] = [f"chr{1 + (i // genes_per_chr)}" for i in range(n_genes)]
    adata.var["start"] = np.arange(n_genes) * 1000
    adata.var["end"] = np.arange(n_genes) * 1000 + 500
    adata.obs["celltype"] = ["normal"] * n_ref + ["tumor"] * (n_cells - n_ref)
    return adata


def main() -> None:
    adata = make_synth_adata()
    print(f"input: {adata.n_obs} cells × {adata.n_vars} genes "
          f"({(adata.obs['celltype'] == 'normal').sum()} reference, "
          f"{(adata.obs['celltype'] == 'tumor').sum()} tumor)")

    cfg = InferCNVConfig(
        cutoff=0.0,
        min_cells_per_gene=0,
        HMM=True,
        HMM_type="i6",
        BayesMaxPNormal=0.0,    # skip BayesNet for speed; flip to 0.5 to enable
        denoise=True,
        reassignCNVs=False,
        random_state=0,
    )
    t0 = time.perf_counter()
    result = infercnv(
        adata, config=cfg,
        reference_key="celltype", reference_cat="normal",
        inplace=False,
    )
    elapsed = time.perf_counter() - t0
    print(f"infercnv() wallclock: {elapsed:.2f} s")

    assert result is not None
    print("\npopulated InferCNVResult fields:")
    for fname in (
        "cnv_matrix", "cnv_matrix_fc", "subclusters", "hmm_states",
        "cnv_regions", "denoised_matrix", "hmm_proxy_matrix",
        "bayes_posterior", "de_mask",
    ):
        val = getattr(result, fname)
        if val is None:
            print(f"  {fname:20s}: None")
        elif hasattr(val, "dtype"):
            print(f"  {fname:20s}: shape={val.shape}, dtype={val.dtype}")
        elif hasattr(val, "shape"):  # DataFrame
            print(f"  {fname:20s}: shape={val.shape} ({type(val).__name__})")
        else:
            print(f"  {fname:20s}: {type(val).__name__}, len={len(val)}")

    # One-panel heatmap of the denoised CNV matrix.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(
        np.log2(np.maximum(result.denoised_matrix, 1e-3)),
        aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3,
        interpolation="nearest",
    )
    ax.set_title("pyinfercnv quickstart — log2(denoised CNV matrix), synthetic data")
    ax.set_xlabel("genomic bins"); ax.set_ylabel("cells")
    out_path = "quickstart_30s_heatmap.png"
    fig.tight_layout(); fig.savefig(out_path, dpi=120)
    print(f"\nheatmap saved to {out_path}")


if __name__ == "__main__":
    main()
