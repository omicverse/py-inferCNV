# pyinfercnv

Pure-Python re-implementation of [inferCNV](https://github.com/broadinstitute/inferCNV) (Broad Institute) — single-cell CNV inference from scRNA-seq, AnnData-native, R-parity-audited.

**Status:** v0.2.0.dev2 — **Phase 2** (tumor subclustering + HMM i3/i6 state calls + hspike calibration + CNV regions) complete and R-parity-validated on the smart-seq2 oligodendroglioma fixture (ARI 1.000, Jaccard 0.976/0.968 vs R infercnv). **pyinfercnv is a Python-accelerated re-implementation of R infercnv**; parity numbers come from the R package's own bundled test fixture. Early wallclock evidence on 1–1.4 kilocell 10x UMI inputs: **25–192× py-vs-R speedup** (see `scripts/phase2_benchmark/` — a py-vs-R wallclock + regression-detection suite, not a real-world correctness benchmark). Phase 3 (BayesNet MCMC + denoise) is not yet implemented.

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

| R step | Python module | Tier | Test assertion |
|---|---|---|---|
| filter genes (mean cutoff + min cells) | `preprocess.filter_low_expression_genes` | 4 set-equality | identical gene set (not a numeric max_diff — step02 is a boolean mask, and the parity test asserts the index sets match) |
| CPM by median libsize | `preprocess.normalize_by_seq_depth` | 4 (relaxed) | `< 1e-2` (float32 cumulative) |
| log2(x+1) | `preprocess.log2_plus1` | 4 approximate | `< 1e-5` |
| subtract_ref (bounded, 1st pass) | `preprocess.subtract_reference` | 4 (relaxed) | `< 1e-3` |
| max_centered_threshold | `preprocess.apply_max_centered_threshold` | 4 approximate | `< 1e-6` |
| smooth (pyramidinal w=101) | `smooth.smooth_pyramidinal` | 4 (relaxed) | `< 1e-3` combined (interior and tail measured separately in the test; the asserted bound is the whole-row `diff_full`, not a distinct interior-only bit-exact claim) |
| per-cell median center | `center.center_cells` | 4 approximate | `< 1e-4` |
| outlier prune (`average_bound`) | `cna.prune_outliers` | 4 approximate | `< 1e-4` |
| invert_log2 | `preprocess.invert_log2` | 4 approximate | `< 1e-4` |

"bit-exact" is used only where the test asserts `max_diff < 1e-10`. The
filter_genes row is not numeric at all — it is a set-equality over gene
names (the Phase 1 gene set matches R step02 exactly, verified in
`tests/test_r_parity.py::test_step02_gene_set_matches`). The smooth row
reports the `diff_full` bound asserted in CI; an earlier phrasing that
claimed a separate "interior bit-exact" bound overstated what CI proves.
Other rows honestly label their empirical floor.

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

**Scope of the Phase 2 numbers above.** The fixture is smart-seq2
(oligodendroglioma downsampled, 184 cells) — R infercnv's own built-in
test fixture. The R vs Python comparison uses `cutoff=1`,
`leiden_method="simple"`, and `leiden_function="CPM"` so both sides go
through the `python-igraph.community_leiden` C core — the R default
`leiden_method="PCA"` depends on Seurat SNN + irlba and has no Python
equivalent. For 10x Genomics UMI data the wiki-canonical setting is
`cutoff=0.1`.

**What this project is.** pyinfercnv reproduces (most of) R infercnv's
algorithm in pure Python for runtime speedup. Parity is measured only
against R infercnv's own fixtures where both pipelines are known to be
well-behaved. We do not claim correctness against any external
ground-truth CNV dataset; for that you would want matched WGS or
orthogonal CNV callers (copykat, numbat, inferCNA), and different
assay types (10x UMI vs smart-seq2) would require their own
validation. The `scripts/phase2_benchmark/` harness is a py-vs-R
wallclock measurement + regression suite, **not** a real-world
validation benchmark — cell-type annotations from 3CA upstream are
used as-is on both sides, so any imperfection in those labels is
shared by py and R and cannot be used to adjudicate correctness.

**Track A closeout — Leiden parity-in-distribution.** Phase 2 Leiden
on kilocell 10x UMI inputs produced single-draw ARIs in the 0.4–0.8
range (DCIS1 0.447, TNBC1 0.592, TNBC3 0.840). A diagnostic sweep
(10 seeds × 2 patients × graph variants; see
`benchmarks/phase2/Gao2021_Breast/track_a_summary.md`) showed:

- **KNN edge-set parity**: py `sklearn.NearestNeighbors(brute)` vs R
  `RANN::nn2` yielded Jaccard 1.0000 on all three 3CA patients; scipy
  `cKDTree` and sklearn `kd_tree` also Jaccard 1.0000. KNN is not a
  source of divergence.
- **Leiden self-noise**: R's own `cluster_leiden` across 10 seeds on
  the same graph has ARI 0.47±0.12 (DCIS1), 0.67±0.10 (TNBC1),
  0.77±0.29 (TNBC3) vs its step15 baseline. The DCIS1 0.447 that
  looked like a divergence is a single-draw observation inside R's
  own 1-σ noise floor.
- **py's Leiden in distribution**: ARI 0.47±0.14 (DCIS1, Δ=-0.007),
  0.73±0.09 (TNBC1, Δ=+0.064), 0.92±0.04 (TNBC3, Δ=+0.149) vs same
  baseline. No observed systematic gap.
- **CPM objective**: manual `Q = Σ_c [e_c − γ n_c(n_c−1)/2]`
  computed on R's graph for a DCIS1 seed-0 pair gave py 10744 vs R
  10724. Py finds equal-or-higher Leiden local optima on the same
  graph.

Per-seed identity across R and Python RNGs is not a contract —
R-igraph seeds through R's Mersenne-Twister + C PCG32, python-igraph
seeds through Python's `random.Random` bridge, feeding different
bytes into the same C Leiden core. Distribution-level equivalence on
measured fixtures is the contract.

Two optional stability knobs are available (both default **off**):
- `InferCNVConfig.tumor_subcluster_n_seeds` (`int`, default 1): if
  >1, run Leiden N times with `random_state, random_state+1, …` and
  return the partition with the highest CPM (`rbest`).
- `InferCNVConfig.tumor_subcluster_min_size` (`int | None`, default
  None): merge any cluster smaller than this size into its nearest
  non-small cluster via KNN majority vote. **Non-CPM-optimal** —
  trades a small objective loss for downstream HMM stability.

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
