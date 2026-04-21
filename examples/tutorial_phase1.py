"""Phase 1 end-to-end tutorial for pyinfercnv.

Jupytext percent-format source. Use `examples/_build_notebooks.py` to turn this
into `tutorial_phase1.ipynb` (and optionally `tutorial_phase1.executed.ipynb`).

NOTE: G1 patch P4 — we call `infercnv(adata, inplace=False)` and work with the
returned ``InferCNVResult`` directly. There is no constructor that reconstructs
a result object out of an AnnData; do not reach for one.
"""

# %% [markdown]
# # pyinfercnv — Phase 1 quickstart
#
# End-to-end run on the bundled oligodendroglioma downsampled fixture that
# ships with the upstream R [infercnv](https://github.com/broadinstitute/infercnv)
# package (184 cells × 6,577 genes). The notebook:
#
# 1. Loads counts + cell-type annotations from the R package's `inst/extdata/`.
# 2. Builds an `AnnData` (cells × genes) and attaches gene coordinates from
#    pyinfercnv's bundled hg38 table.
# 3. Runs `pyinfercnv.infercnv(...)` with Microglia + Oligodendrocytes as the
#    reference (normal) cells — the malignant labels are the four `malignant_*`
#    categories in the annotation file.
# 4. Reads the Phase 1 outputs off the returned `InferCNVResult`, prints the
#    `psutil` per-block profile captured by the pipeline, and plots a log2(FC)
#    heatmap with `pyinfercnv.viz.plot_cnv_heatmap`.
#
# This tutorial is matplotlib-only; no `omicverse` import. Phase 2 (tumor
# subclustering + HMM) and Phase 3 (BayesNet filter) are not covered here.

# %% [markdown]
# ## Installation
#
# If `pyinfercnv` is already installed in this environment (it is, under the
# repo's uv venv), the next cell is a no-op.
#
# ```bash
# uv sync               # from the pyinfercnv repo root
# # or, once released:
# pip install pyinfercnv
# ```

# %%
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad

import pyinfercnv
from pyinfercnv import InferCNVConfig, infercnv
from pyinfercnv.io.genome import load_gene_positions
from pyinfercnv.viz import plot_cnv_heatmap

print("pyinfercnv version:", pyinfercnv.__version__)

# %% [markdown]
# ## 1. Load the bundled R fixture
#
# The oligodendroglioma downsampled fixture ships with the R infercnv package
# under `inst/extdata/`. Three files:
#
# | file | contents |
# |---|---|
# | `oligodendroglioma_expression_downsampled.counts.matrix.gz` | genes × cells integer counts |
# | `oligodendroglioma_annotations_downsampled.txt` | `cell_id <TAB> annotation` (no header) |
# | `gencode_downsampled.EXAMPLE_ONLY_DONT_REUSE.txt` | per-gene `symbol, chr, start, end` |
#
# pyinfercnv does not mirror the gencode table (we ship genome-wide hg38 / hg19 /
# mm10 parquets — see §2). Counts + annotations come straight from the R package
# paths.

# %%
FIXTURE_DIR = Path(
    "/media/jason/T7/rerbulid/infercnv/infercnv-master/inst/extdata"
)
COUNTS_PATH = FIXTURE_DIR / "oligodendroglioma_expression_downsampled.counts.matrix.gz"
ANNOT_PATH = FIXTURE_DIR / "oligodendroglioma_annotations_downsampled.txt"

# counts.matrix.gz is genes x cells (first column = gene symbol, header = cell IDs).
counts_gxc = pd.read_csv(COUNTS_PATH, sep="\t", index_col=0)
print("counts shape (genes x cells):", counts_gxc.shape)

# Cell annotations (no header in R fixture).
annot = pd.read_csv(
    ANNOT_PATH, sep="\t", header=None, names=["cell_id", "annotation"]
)
annot = annot.set_index("cell_id")
print("annotation categories:")
print(annot["annotation"].value_counts())

# %% [markdown]
# Build an `AnnData` with cells as rows. Align the annotation table to the
# counts columns. The counts layer is stored under `"counts"` (the default
# `InferCNVConfig.counts_layer`).

# %%
cells = list(counts_gxc.columns)
X = counts_gxc.T.values.astype(np.float32)  # cells x genes

adata = ad.AnnData(
    X=X,
    obs=annot.reindex(cells),
    var=pd.DataFrame(index=counts_gxc.index),
)
adata.obs.index.name = None
adata.var.index.name = None
adata.layers["counts"] = adata.X.copy()
print(adata)

# %% [markdown]
# ## 2. Attach gene coordinates from the bundled hg38 parquet
#
# `pyinfercnv.io.genome.load_gene_positions("hg38")` returns a DataFrame with
# columns `gene_symbol / chromosome / start / end`. The pipeline needs
# `chromosome` (and optionally `start`) on `adata.var`; genes with no
# coordinate or on `chr_exclude` (chrX / chrY / chrM by default) are dropped
# inside the pipeline.
#
# This uses the same join logic as infercnvpy / scanpy — left-join on
# gene symbol. Fixture symbols that are not in the hg38 table (mostly
# aliases / retired symbols) will end up with NaN chromosome and be
# filtered downstream.

# %%
gene_pos = load_gene_positions("hg38")
# Drop duplicated gene_symbol rows in the hg38 table (keep first) so the
# left-join remains 1:1.
gene_pos_unique = gene_pos.drop_duplicates(subset="gene_symbol", keep="first")

n_vars_before = adata.n_vars
var = adata.var.join(
    gene_pos_unique.set_index("gene_symbol")[["chromosome", "start", "end"]],
    how="left",
)
adata.var["chromosome"] = var["chromosome"].values
adata.var["start"] = var["start"].values
adata.var["end"] = var["end"].values
mapped = adata.var["chromosome"].notna().sum()
print(f"genes in adata.var: {n_vars_before}")
print(f"genes with hg38 chromosome match: {mapped} "
      f"({mapped / n_vars_before:.1%})")
print(adata.var.head())

# %% [markdown]
# ## 3. Run `pyinfercnv.infercnv`
#
# The oligodendroglioma fixture has six annotation categories:
#
# * 4× `malignant_*` (tumor cells, one per patient)
# * `Microglia/Macrophage` (normal)
# * `Oligodendrocytes (non-malignant)` (normal)
#
# We pass the two non-malignant categories as the reference via
# `reference_key` + `reference_cat`. Using `inplace=False` returns the
# `InferCNVResult` directly (G1 patch P4: we do not reconstruct a result
# from the AnnData downstream).

# %%
REFERENCE_CATS = ["Microglia/Macrophage", "Oligodendrocytes (non-malignant)"]

cfg = InferCNVConfig(
    window_length=101,
    cutoff=0.1,
    min_cells_per_gene=3,
    prune_outliers=True,
)
cfg.validate()

result = infercnv(
    adata,
    config=cfg,
    reference_key="annotation",
    reference_cat=REFERENCE_CATS,
    inplace=False,
)
assert result is not None, "inplace=False must return an InferCNVResult"
print("InferCNVResult:")
print(f"  n_cells      = {result.n_cells}")
print(f"  n_bins       = {result.n_bins}")
print(f"  chromosomes  = {result.chromosomes}")
print(f"  reference    = {int(result.cell_meta['is_reference'].sum())} cells")
print(f"  cnv_matrix   shape = {result.cnv_matrix.shape}, dtype = {result.cnv_matrix.dtype}")
print(f"  cnv_matrix_fc shape = {result.cnv_matrix_fc.shape}")

# %% [markdown]
# ### Persist the result into AnnData (optional)
#
# `result.write_to_anndata(adata, key_added="cnv")` stores `cnv_matrix` under
# `adata.obsm["X_cnv"]` and the metadata dict under `adata.uns["cnv"]`. This
# is exactly what `inplace=True` would have done — we delay it until after we
# have the result in hand so we can print from the result object *and* from
# the AnnData side-by-side.

# %%
result.write_to_anndata(adata, key_added="cnv")
print("adata.obsm['X_cnv'].shape:", adata.obsm["X_cnv"].shape)
print("adata.obsm['X_cnv'][:3, :5]:")
print(adata.obsm["X_cnv"][:3, :5])

# %% [markdown]
# ## 4. Per-block profile (psutil — G1 patch P11)
#
# The pipeline records wall-clock + RSS deltas for each major block under
# `result.profile` (also written to `adata.uns["cnv"]["profile"]`). This is
# the same dict — we print it from both sides to demonstrate.

# %%
profile = result.profile
assert profile is not None, "Phase 1 pipeline must record a profile dict"

rows = []
for name, entry in profile.items():
    rows.append(
        {
            "block": name,
            "wallclock_s": round(entry["wallclock_s"], 4),
            "rss_delta_mb": (
                round(entry["rss_delta_mb"], 1)
                if entry.get("rss_delta_mb") is not None
                else None
            ),
        }
    )
profile_df = pd.DataFrame(rows)
print(profile_df.to_string(index=False))
print(f"\ntotal wall-clock: {profile_df['wallclock_s'].sum():.3f} s")

# Verify adata round-trip matches
assert adata.uns["cnv"]["profile"] is profile or adata.uns["cnv"]["profile"] == profile

# %% [markdown]
# ## 5. Heatmap — `pyinfercnv.viz.plot_cnv_heatmap`
#
# log2(FC) values are in `result.cnv_matrix`; `plot_cnv_heatmap` uses the
# chromosome strip encoded in `result.chr_pos` to render a compact genome
# track on top of the cell × bin image.

# %%
fig, ax = plt.subplots(figsize=(12, max(4, result.n_cells / 40)))
plot_cnv_heatmap(result, ax=ax, vmin=-0.3, vmax=0.3)
ax.set_title(
    f"pyinfercnv Phase 1 — oligodendroglioma fixture "
    f"({result.n_cells} cells, {result.n_bins} bins across {len(result.chromosomes)} chromosomes)"
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Next steps
#
# * **Phase 2 (in progress)** — tumor subclustering + 6/3-state HMM; outputs
#   will populate `result.subclusters`, `result.hmm_states`,
#   `result.hmm_states_i3`.
# * **Phase 3 (in progress)** — BayesNet posterior filter + region calls
#   (`result.cnv_regions`, `result.posterior_p_normal`).
# * **Parity status** — see the repo's `NAMESPACE_PARITY.md` + `README.md`
#   "Parity status" table. Tier-4 bit-exact R-parity checks live in
#   `tests/test_r_parity.py` and are driven by `tests/r_reference.R`.
# * **Side-by-side R reference** — `examples/r_driver_phase1.R` runs the same
#   pipeline through the R `infercnv::run` driver for a manual sanity check.
