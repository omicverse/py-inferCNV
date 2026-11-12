# pyinfercnv

Pure-Python re-implementation of [inferCNV](https://github.com/broadinstitute/inferCNV) (Broad Institute) — single-cell CNV inference from scRNA-seq, AnnData-native, R-parity-audited.

**Status:** Phase 1 preprocessing, Phase 2 subclustering/HMM and Phase 3
Bayes filtering are available. The current development changes correct the
i6 hspike calibration, random-tree input and smoothing, and Bayes state/region
exports. These fixes can change subclusters and HMM/Bayes results; see the
migration notes below. The i3/i6 Viterbi decoding kernels are unchanged.

`reassignCNVs=True` is supported. Remaining unsupported branches include
`noise_logistic=True`, `sim_method='splatter'/'simple'`,
`postMcmcMethod='removeCells'` and `threshold='auto'`; they raise explicit
errors. The benchmark tables below describe the earlier 0.2.0 implementation
and its fixtures, rather than validation of every new default.

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
`mask_nonDE_genes=True`, `denoise=True`, `reassignCNVs=True`); see
`examples/tutorial_phase{1,2,3}.ipynb` for end-to-end notebook walk-throughs
on the bundled R inferCNV oligodendroglioma fixture.

CLI:

```bash
pyinfercnv run-h5ad --input dataset.h5ad --output out.h5ad \
    --reference-key cell_type --reference-cat Microglia,Oligodendrocytes
```

## HMM and random-tree correctness changes

The count-to-HMM path now follows these steps:

- Exclude unannotated and excluded chromosomes before gene filtering and
  library-size normalization, matching R object construction. Counts on
  excluded chromosomes no longer affect the normalization denominator.
- Build `random_trees` input from the step-04 log matrix, subtracting reference
  group means computed in linear space (`inv_log=True`). HMM decoding still
  consumes the continuous Phase 1 CNV matrix, not the tree input.
- Smooth tree input with a box running mean and cropped endpoint windows.
  The default gene order is chromosome lexicographic order, then gene start;
  the Phase 1 CNV matrix keeps its natural chromosome order. The full gene
  axis is smoothed for tree building, so its order affects the partition.
- Fit trees on all cells by default. Explicit subsampling remains available
  for users who accept nearest-centroid assignment for the other cells.
- Calibrate i6 using float64 normalized counts and a local R-compatible
  MT/Inversion/Rejection stream. Gene/group sampling order, dropout draw
  consumption, repeated spline inputs, GCV and linear extrapolation are
  covered by synthetic R references. Hspike replay honors `prune_outliers`.
- Apply Bayes removal and optional posterior-argmax reassignment to the final
  matrix and region table together. Exported state and CN values therefore
  describe the same final calls.

No R runtime is needed for normal use. The R-generated numerical references
are checked into `tests/unit/fixtures/`, together with their regeneration
script and provenance. They verify calibration primitives; they do not imply
bitwise equality of arbitrary end-to-end R/Python runs. Random-tree
permutations and Bayes sampling still use their own NumPy streams.

### Configuration and migration

| Setting | Current default | Earlier behavior available explicitly |
|---|---|---|
| `hspike_calibration_backend` | `"r_compatible"` | `"numpy_legacy"`: earlier NumPy sampling and approximate spline |
| `tree_input_source` | `"step04"` | `"cnv_matrix"`: final Phase 1 matrix |
| `tree_smooth_method` | `"runmean"` | `"pyramidinal"`: triangular kernel |
| `tree_gene_order` | `"r_lexicographic"` | `"natural"`: meaningful with step-04 input |
| `subsample_for_tree` | `None` (all cells) | `500`: earlier cell cap |
| `reassignCNVs` | `True` (now implemented) | `False`: remove high-P(normal) regions only |

The tree controls `tumor_subcluster_pval`, `tumor_subcluster_window_size`,
`tumor_subcluster_max_recursion_depth`,
`tumor_subcluster_min_cluster_size_recurse` and
`tumor_subcluster_n_rand_iters` are now forwarded by `InferCNVConfig`.
Direct `run_phase2()` callers using the default random-tree path must supply
`tree_input`; otherwise a `ValueError` explains the missing input. Full-cell
trees require more memory and time than capped trees. Legacy settings select
earlier components; they do not undo normalization, pruning or export fixes
and cannot guarantee reproduction of all historical outputs.

### Bayes result mapping

`result.hmm_states` remains the step-17 matrix.
`result.hmm_states_filtered` stores the post-Bayes step-19 matrix and is
written to `adata.obsm["X_cnv_hmm_states_filtered"]` with the default key.
`cnv_regions` contains retained regions. `posterior_p_normal` contains all
pre-filter regions, including those removed, with:

- `region_id` and `posterior_row`, indexing the unfiltered `bayes_posterior`;
- `state_step17`, final zero-based `state`, and one-based `state_r`;
- final `cn`, `p_normal`, `p_state_0..K-1` and `removed_by_pnormal`.

Bin endpoints are inclusive. Region IDs are local to a run. i6 `cn` uses the
model levels `[0.01, 0.5, 1, 1.5, 2, 3]`; i3 retains its signed-delta convention.
A region reassigned to neutral can remain in the retained table when its
P(normal) is at or below the removal threshold.

Run the portable regression tests from a development installation:

```bash
python -m pytest tests/unit tests/integration tests/test_io_contract.py tests/test_smoke.py
```

Optional cross-checks requiring `hmmlearn` or external R dumps skip when
those inputs are unavailable. Numerical fixture generation is documented in
`tests/unit/fixtures/README.md`.

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
hspike calibration.

### Primary metric — Spearman ρ on the continuous CNV matrix

The step-14 post-Phase-1 `cnv_matrix` is the continuous log2-FC signal that
every downstream step (HMM, region calls, heatmap) consumes. Its rank
correlation between py and R is the most load-bearing parity claim we make.

| Fixture | cells | genes | per-cell ρ mean±std | global flat ρ | Pearson |
|---|---|---|---|---|---|
| oligodendroglioma (smart-seq2, R's own fixture) | 184 | 8508 | **0.9998 ± 0.0001** | **0.9998** | 0.9928 |
| Gao2021_Breast/DCIS1 (10x UMI, 3CA) | 1399 | 9237 | **1.0000 ± 0.0000** | **1.0000** | 0.9980 |
| Gao2021_Breast/TNBC1 (10x UMI, 3CA) | 1022 | 9654 | **1.0000 ± 0.0000** | **1.0000** | 0.9964 |
| Gao2021_Breast/TNBC3 (10x UMI, 3CA) | 520  | 10835 | **1.0000 ± 0.0000** | **1.0000** | 0.9987 |

Raw data + methodology: `benchmarks/phase2/Gao2021_Breast/cnv_matrix_spearman.{json,md}`.
Reproducer: `uv run python scripts/track_a1/cnv_matrix_spearman.py`.

### Secondary metric — HMM state Spearman ρ (post-discretization)

HMM state is a bucketed view of the continuous CNV matrix (i3: 0/1/2 for
deletion/neutral/amplification; i6: 0-5 for finer gradations). Rank
correlation on state sequences tells you whether py and R put the same
bins into the same magnitude bucket.

| Patient | HMM | per-cell ρ mean±std | global flat ρ |
|---|---|---|---|
| DCIS1 | i3 | 0.9202±0.056 | 0.9208 |
| DCIS1 | i6 | 0.8273±0.052 | 0.8178 |
| TNBC1 | i3 | 0.9610±0.023 | 0.9615 |
| TNBC1 | i6 | 0.7524±0.047 | 0.7415 |
| TNBC3 | i3 | 0.9323±0.145 | 0.9578 |
| TNBC3 | i6 | — (R hspike crashed on this fixture, not comparable) | — |

The drop from ρ≈1.0 on the continuous matrix to ρ 0.75-0.96 on HMM states
is **discretization amplification**: when the continuous log2-FC value is
near a state boundary, small floating-point differences between py and R
can flip the assigned bucket for a few bins per cell. The underlying
signal agreement is unchanged.

### Operational metric — oligodendroglioma kernel parity (anchor, not a headline)

| R step | Python module | Measurement |
|---|---|---|
| `predict_CNV_via_HMM_wrapper` (i3) | `hmm.predict_i3` + `pipeline_phase2.run_phase2` | Jaccard 1.000 vs R step17 on oligo |
| `predict_CNV_via_HMM_wrapper` (i6 + hspike) | `hmm.predict_i6` + `hmm.hspike.calibrate_i6_emission` | Jaccard 0.979 vs R step17 on oligo |
| Viterbi.dthmm.adj kernel (R-aligned input) | `kernels.hmm_viterbi_numba` | Jaccard 0.9999 with R's own step15+step16 feeding py kernel |

CI floors are `0.96 / 0.99` on i6/i3 Jaccard — deliberately below observed
values so real regressions can be distinguished from stochastic drift.

### Metrics deliberately dropped

- **Subcluster ARI** is not a parity metric. `subcluster_s1` in py and R
  are **arbitrary algorithm-internal bucket labels** with no cross-language
  semantic mapping; a high ARI only means "same cells land in buckets that
  happened to line up" and a low ARI can mean either real divergence or
  just RNG-driven permutation. The test at
  `tests/test_r_parity.py::test_step15_subclusters_ari_floor` is retained
  as an internal-consistency regression anchor (floor 0.85), not a parity
  claim. The load-bearing py-vs-R claim is the Spearman ρ table above.

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

**Track A closeout — the DCIS1 "divergence" was a label-arbitrariness
artifact.** HANDOFF v5/v6 flagged DCIS1 as having Leiden subcluster
ARI = 0.447, which looked like a systematic py-vs-R gap. Track A
showed two separate things:

1. **The subcluster ID is not a scientific quantity.** Leiden outputs
   arbitrary integer bucket labels (`cluster_0`, `cluster_1`, …); py
   and R have no cross-language contract on which cells get which
   label. ARI measures label-invariant permutation agreement, but
   when Leiden is RNG-dominated (as it is at the low resolution
   infercnv uses on kilocell 10x UMI graphs) both sides' partitions
   legitimately bounce between near-equal CPM optima with different
   labellings. R's own `cluster_leiden` across 10 seeds has ARI
   0.47±0.12 (DCIS1), 0.67±0.10 (TNBC1), 0.77±0.29 (TNBC3) vs its
   own step15 baseline — py is within that self-noise floor in every
   case. ARI is therefore retired as a headline parity metric.
2. **The underlying signal is near-identical.** Spearman ρ on the
   continuous post-Phase-1 CNV matrix is **1.0000 on all three 3CA
   patients and 0.9998 on oligo** (table above). The HMM state
   Spearman on i3 is 0.92-0.96. The Leiden subcluster boundary
   happens to fall inside the RNG-sensitive region of this graph, so
   the categorical subcluster ID fluctuates — but that fluctuation
   is downstream of a nearly bit-exact continuous CNV signal.

Diagnostic sweep details (all under
`benchmarks/phase2/Gao2021_Breast/`):
- **KNN edge-set parity**: py `sklearn.NearestNeighbors(brute)` vs R
  `RANN::nn2` yielded Jaccard 1.0000 on all three 3CA patients; scipy
  `cKDTree` and sklearn `kd_tree` also Jaccard 1.0000. KNN itself is
  deterministic on both sides.
- **CPM objective on same graph**: manual `Q = Σ_c [e_c − γ n_c(n_c−1)/2]`
  computed on R's graph for a DCIS1 seed-0 pair gives py **10744**
  vs R **10724**. Py finds equal-or-higher Leiden local optima.
- See `track_a_summary.md` and `cnv_matrix_spearman.md` for the full
  per-patient tables; diagnostic scripts live under `scripts/track_a1/`.

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
  - `validation/` — r-parity metrics (max_diff, Spearman ρ, Jaccard; ARI retained as operational helper only)
  - `viz/` — matplotlib heatmap (no omicverse; ov-aligned palette hardcoded)
  - `pipeline.py` — end-to-end orchestration with psutil profile hooks
  - `cli.py` — typer app (`run-h5ad`, `version`)
- `tests/` — unit + R-parity + smoke + io_contract + viz_smoke + wheel
- `scripts/` — codex review tooling, GENCODE GTF → parquet generator
- `docs/superpowers/` — specs, plans, reviews (triple-gate audit trail)

## License

BSD-3-Clause, matching upstream R infercnv.
