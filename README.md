# pyinfercnv

Pure-Python re-implementation of [inferCNV](https://github.com/broadinstitute/inferCNV) (Broad Institute) — single-cell CNV inference from scRNA-seq, AnnData-native, R-parity-audited.

**Status:** v0.1.0.dev0 — **Phase 1** (preprocess → smooth → invert_log2 → outlier prune) is functionally complete and R-parity-validated. Phase 2 (HMM + tumor subclustering) and Phase 3 (BayesNet MCMC + denoise) are not yet implemented.

## Installation

```bash
pip install pyinfercnv              # core
pip install 'pyinfercnv[viz]'       # + matplotlib for heatmap
pip install 'pyinfercnv[compare]'   # + hmmlearn for parity cross-check (Phase 2+)
pip install 'pyinfercnv[dev]'       # + test/build tooling
```

Python ≥3.10, <3.13. Wheel is `py3-none-any` (pure Python; numba JIT at first call). Bundles hg38/hg19/mm10 gene-position parquets from GENCODE v45 basic / v19 / vM25.

## Quickstart

```python
import scanpy as sc
from pyinfercnv import infercnv
from pyinfercnv.io.genome import load_gene_positions

adata = sc.read_h5ad("my_dataset.h5ad")

# Merge gene positions into adata.var (gene_symbol index assumed)
gene_pos = load_gene_positions(genome="hg38")
adata.var = adata.var.merge(gene_pos.set_index("gene_symbol"),
                            left_index=True, right_index=True, how="left")

infercnv(
    adata,
    reference_key="cell_type",
    reference_cat=["Microglia", "Oligodendrocytes"],
    key_added="cnv",
)
# Results:
#   adata.obsm["X_cnv"]                      — log2(FC) CNV matrix
#   adata.uns["cnv"]["chr_pos"]              — chromosome → bin start index
#   adata.uns["cnv"]["profile"]              — per-block wallclock / RSS
#   adata.uns["cnv_ref_counts_raw"]          — raw ref counts (Phase 2 prep)
```

CLI:

```bash
pyinfercnv run-h5ad --input dataset.h5ad --output out.h5ad \
    --reference-key cell_type --reference-cat Microglia,Oligodendrocytes
```

## Parity status (Phase 1)

R-parity verified against R `infercnv` pipeline's per-step intermediate TSV dumps
from `tests/r_reference.R` on the bundled `oligodendroglioma_expression_downsampled`
fixture (184 cells × 8508 genes after filter).

| R step | Python module | Tier | Measured `max_diff` |
|---|---|---|---|
| filter genes (mean cutoff + min cells) | `preprocess.filter_low_expression_genes` | 4 bit-exact | `< 1e-10` |
| CPM by median libsize | `preprocess.normalize_by_seq_depth` | 4 (relaxed) | `< 1e-2` (float32 cumulative) |
| log2(x+1) | `preprocess.log2_plus1` | 4 approximate | `< 1e-5` |
| subtract_ref (bounded, 1st pass) | `preprocess.subtract_reference` | 4 (relaxed) | `< 1e-3` |
| max_centered_threshold | `preprocess.apply_max_centered_threshold` | 4 approximate | `< 1e-6` |
| smooth (pyramidinal w=101) | `smooth.smooth_pyramidinal` | 4 (relaxed) | `< 1e-3` (interior bit-exact, tail R-exact) |
| per-cell median center | `center.center_cells` | 4 approximate | `< 1e-4` |
| outlier prune (`average_bound`) | `cna.prune_outliers` | 4 approximate | `< 1e-4` |
| invert_log2 | `preprocess.invert_log2` | 4 approximate | `< 1e-4` |

"bit-exact" is used only where the test asserts `max_diff < 1e-10`. Other rows
honestly label their empirical floor.

**Dependencies are not bit-exact on normalize step** due to float32 vs float64 — spec §6.3
mandates CSR float32 I/O for memory efficiency, which accrues ~1e-2 drift over 8508
genes × 184 cells. R uses float64 throughout; the tradeoff is documented and the
relaxed threshold is the empirical floor.

Full crosswalk in [NAMESPACE_PARITY.md](NAMESPACE_PARITY.md).

## Testing

```bash
# Fast tests (no R)
uv run pytest tests/unit tests/test_smoke.py tests/test_io_contract.py tests/test_viz_smoke.py -v

# R-parity (needs Rscript + infercnv installed; generates TSV fixtures first)
Rscript tests/r_reference.R
uv run pytest tests/test_r_parity.py -v

# Wheel validation
uv run python -m build
uv run pytest tests/test_wheel.py -v
```

## Architecture

- `pyinfercnv/` — package core (no `omicverse` import at any level — spec §7)
  - `io/` — AnnData reader, gene-order BED, bundled genome parquets
  - `preprocess/` — filter, normalize, log, subtract_ref, max_threshold
  - `smooth/` — pyramidinal (scipy.ndimage + numba tail)
  - `center/` — per-cell median
  - `cna/` — outlier prune
  - `kernels/` — numba hot kernels (tail smoothing)
  - `validation/` — r-parity metrics (max_diff, ARI, Jaccard)
  - `viz/` — matplotlib heatmap (no omicverse; ov-aligned palette hardcoded)
  - `pipeline.py` — end-to-end orchestration with psutil profile hooks
  - `cli.py` — typer app (`run-h5ad`, `version`)
- `tests/` — unit + R-parity + smoke + io_contract + viz_smoke + wheel
- `scripts/` — codex review tooling, GENCODE GTF → parquet generator
- `docs/superpowers/` — specs, plans, reviews (triple-gate audit trail)

## License

BSD-3-Clause, matching upstream R infercnv.
