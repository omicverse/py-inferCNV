# pyinfercnv

Pure-Python re-implementation of [inferCNV](https://github.com/broadinstitute/inferCNV) (Broad Institute) — single-cell CNV inference from scRNA-seq, AnnData-native, R-parity-audited.

**Status:** v0.2.1 — reproduces R inferCNV 1.26.0 (`run()` with HMM i6/i3, subclusters, denoise and the Bayesian filter), 40–65× faster on one thread. From the same seed (`set.seed(s)` in R, `random_state=s` here) pyinfercnv gives R's numbers: the expression matrices to rounding, and the hidden-spike calibration, the subclusters and the HMM states identically whenever the neighbour graphs are identical — always with `leiden_method="simple"`, and with R's default `"PCA"` whenever Seurat's approximate (Annoy) neighbour search returns the exact neighbours. See *Reproducing R* below and [`report/`](report/).

## Installation

```bash
pip install pyinfercnv              # core
pip install 'pyinfercnv[viz]'       # + matplotlib for heatmap
pip install 'pyinfercnv[compare]'   # + hmmlearn for parity cross-check (Phase 2+)
pip install 'pyinfercnv[dev]'       # + test/build tooling
```

Python ≥3.10, <3.13. Wheel is `py3-none-any` (pure Python; numba JIT at first call). Bundles hg38/hg19/mm10 gene-position parquets from GENCODE v45 basic / v19 / vM25.

## Quickstart

**30 seconds, no fixture download** — runs on synthetic data, prints the populated
`InferCNVResult` schema, saves a heatmap PNG:

```bash
uv run python examples/quickstart_30s.py
# or: python examples/quickstart_30s.py
```

**Real workflow on your own AnnData**:

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

**Phase 2 + 3** — set `HMM=True` (and optionally `BayesMaxPNormal=0.5`,
`mask_nonDE_genes=True`, `denoise=True`, `reassignCNVs=False`); see
`examples/tutorial_phase{1,2,3}.ipynb` for end-to-end notebook walk-throughs
on the bundled R inferCNV oligodendroglioma fixture.

CLI:

```bash
pyinfercnv run-h5ad --input dataset.h5ad --output out.h5ad \
    --reference-key cell_type --reference-cat Microglia,Oligodendrocytes
```

## Reproducing R

pyinfercnv follows the code R inferCNV 1.26.0 runs, step by step:

* **Preprocessing (steps 1–14, 16)** — genes on excluded chromosomes are dropped before the depth
  normalisation, as in `CreateInfercnvObject`; genes are ordered by chromosome, start and end;
  all arithmetic is in double precision. The matrices equal R's to ~1e-14 (the public float32
  outputs to their rounding).
* **Random numbers** — R's Mersenne-Twister and `sample()`, `rnorm` (inversion) and `runif` are
  ported (`pyinfercnv._rcompat`), and drawn in R's order: the hidden-spike simulation (step 3),
  the subclustering (step 15) and the hidden-spike trend (step 17) share one generator started
  by `random_state`, as `set.seed(random_state); infercnv::run(...)` does.
* **Hidden spike (i6 HMM)** — `smooth.spline` with generalised cross-validation is ported from R's
  `sbart.c` (knots, banded solve, Brent search on `spar`); the simulated counts, the per-state
  means and standard deviations and the trend fit equal R's.
* **Subclusters (step 15)** — the genes R leaves out (`z_score_filter`) are left out, and the
  linear fold-change matrix is clustered. `leiden_method="PCA"` (R's default) runs Seurat's
  vst, ScaleData and RunPCA (`set.seed(42)` and irlba's start vector included) and the SNN
  graph; `"simple"` the k-NN graph. Leiden runs on python-igraph with a generator that draws R's
  numbers the way R igraph does, so the same graph gives R's partition. Seurat finds neighbours
  with Annoy, an approximate search; pyinfercnv uses the exact neighbours, so with `"PCA"` a few
  cells can be placed in other subclusters (adjusted Rand index 0.79–0.98 on six 10x tumours).
* **HMM, Bayesian filter, denoise (steps 17–22)** — follow R; CNV regions are built for the
  observation subclusters only, as R's region report.

On the oligodendroglioma example of inferCNV, `infercnv(…, random_state=1)` returns R's
subclusters and HMM states exactly. On six breast tumours from 10x Genomics (532–3,225 cells),
the expression and denoised matrices equal R's to float32 rounding and 98.2–99.5% of the HMM
states agree; pyinfercnv ran 41–64 times faster than R on one thread. Scripts and figure:
[`report/`](report/).

Differences from R's defaults to note: `BayesMaxPNormal` defaults to 0 here (R: 0.5) and
`reassignCNVs=True` is not ported; the HMM states are numbered from 0 (R from 1).

## Testing

```bash
# Fast tests (no R)
uv run pytest tests/unit tests/test_smoke.py tests/test_io_contract.py tests/test_viz_smoke.py -v

# R-parity (needs Rscript + infercnv installed; generates TSV fixtures first)
export PYINFERCNV_R_EXTDATA=$(Rscript -e 'cat(system.file("extdata", package = "infercnv"))')
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
  - `_rcompat/` — R's random numbers, `smooth.spline`, base-R arithmetic and Seurat steps
  - `validation/` — r-parity metrics (max_diff, Spearman ρ, Jaccard; ARI retained as operational helper only)
  - `viz/` — matplotlib heatmap (no omicverse; ov-aligned palette hardcoded)
  - `pipeline.py` — end-to-end orchestration with psutil profile hooks
  - `cli.py` — typer app (`run-h5ad`, `version`)
- `tests/` — unit + R-parity + smoke + io_contract + viz_smoke + wheel
- `scripts/` — codex review tooling, GENCODE GTF → parquet generator
- `docs/superpowers/` — specs, plans, reviews (triple-gate audit trail)

## License

BSD-3-Clause, matching upstream R infercnv.
