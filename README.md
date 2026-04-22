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

## Parity status (Phase 2)

Phase 2 adds tumor subclustering (leiden), HMM state calls (i3 / i6), and
hspike calibration. Measurements on the same oligodendroglioma fixture:

| R step | Python module | Tier | Measurement |
|---|---|---|---|
| `define_signif_tumor_subclusters` (leiden path) | `subcluster.leiden.leiden_subcluster` | 3.5 empirical | ARI `1.000` vs R step15, exceeds spec §5.2 target `0.85` |
| `predict_CNV_via_HMM_wrapper` (i3, deterministic) | `hmm.predict_i3` + `pipeline_phase2.run_phase2` | 3.5 empirical | Jaccard `0.976` vs R step17, exceeds spec §5.2 target `0.95` |
| `predict_CNV_via_HMM_wrapper` (i6 + hspike) | `hmm.predict_i6` + `hmm.hspike.calibrate_i6_emission` | 3.5 empirical | Jaccard `0.968` vs R step17, exceeds spec §5.2 target `0.95` |
| Viterbi.dthmm.adj kernel (diagnostic, R-aligned input) | `kernels.hmm_viterbi_numba` | kernel-isolated parity | Jaccard `0.9999` with R's own step15+step16 feeding the py kernel |

"3.5 empirical" means a floor-based assertion on categorical state agreement
(ARI / per-cell Jaccard), not a continuous `max_diff < 1e-6` claim. CI floors
are `0.85 / 0.90 / 0.90` respectively — deliberately below the spec targets
so that a true regression below the observed numbers can be distinguished
from stochastic drift at spec boundaries. See `tests/test_r_parity.py` for
the per-assert rationale comments, and
`docs/superpowers/findings/2026-04-21-i3-triage-findings.md` §(2) for the
kernel-isolation diagnostic derivation.

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
