# Changelog

## 0.2.0 (2026-04-25) — first stable release

First stable cut. Phase 1 + Phase 2 + Phase 3 are all implemented and
R-parity-validated. The R-parity audit covers the R inferCNV package's
own bundled test fixture (oligodendroglioma smart-seq2 downsampled) plus
three 3CA kilocell 10x UMI patients (DCIS1, TNBC1, TNBC3). Bit-exactness
is established at the step level where R writes intermediate fixtures;
end-to-end parity is measured by Spearman ρ on the continuous post-Phase-1
CNV matrix (the quantity every downstream step consumes).

### Phase 1 (R steps 2-14, 16) — bit-exact

| Step | Module | max_diff | Status |
|---|---|---|---|
| step02 filter_genes | `preprocess.filter_genes` | exact | ✓ |
| step03 normalize_by_seq_depth | `preprocess.normalize` | < 1e-10 | bit-exact |
| step04 log2(x+1) | `preprocess.log_transform` | < 1e-10 | bit-exact |
| step08 subtract_ref_mean | `preprocess.subtract_ref` | < 1e-10 | bit-exact |
| step09 max_threshold | `preprocess.max_threshold` | < 1e-10 | bit-exact |
| step10 smooth_pyramidinal | `kernels.smooth_tail` | < 1e-10 | bit-exact (post 9f213cf) |
| step11 center_cells | `preprocess.center_cells` | < 1e-10 | bit-exact |
| step14 invert_log2 | `preprocess.invert_log2` | < 1e-10 | bit-exact |
| step16 prune_outliers | `preprocess.prune_outliers` | < 1e-10 | bit-exact |

All Phase 1 intermediates compute in float64; the public output contract
(`result.cnv_matrix`, `result.cnv_matrix_fc`) downcasts to float32 only
at result assembly in `pipeline.py`. The Phase 1 → Phase 2 handoff
preserves float64 precision via the `cnv_matrix_f64` companion field.

### Phase 2 (R steps 15, 17) — R-parity-validated

| Module | R source | Tier | Status |
|---|---|---|---|
| Leiden subclusters | `inferCNV_tumor_subclusters.R` | operational floor 0.85 | ARI **1.000** on oligo (ARI retired as parity metric — Leiden labels are arbitrary; cluster identity not a parity contract) |
| HMM Viterbi (i6 + i3) | `inferCNV_HMM.R::Viterbi.dthmm.adj` | bit-exact kernel | numba `@njit` port; 0.9999 Jaccard on R-aligned input |
| hspike i6 calibration | `inferCNV_hidden_spike.R::.build_and_add_hspike` | structural rules D.1-D.7 | strict R-parity rewrite (commit `384e983`) |
| step17 HMM i3 Jaccard | — | — | **1.000** (bit-exact post linear-FC switch, commit `8ed9318`) |
| step17 HMM i6 Jaccard | — | — | **0.979** on oligo (HEAD measurement) |
| Spearman ρ (CNV matrix) | — | primary parity metric | **1.0000** on DCIS1/TNBC1/TNBC3 10x UMI; **0.9998** on oligo smart-seq2 |

### Phase 3 (R steps 18-22) — wired into top-level `infercnv()`

| Module | R source | Tier | Status |
|---|---|---|---|
| BayesNet Gibbs (step 18) | `inferCNV_BayesNet.R` (rjags BUGS Mixture Model) | soft-tier (\|ΔP\|<0.10) ≥90% | ✓ implemented; `removeCNV`-only port. `reassignCNVs=True` (R default) raises `NotImplementedError` at orchestrator. |
| filterHighPNormals (step 19) | `inferCNV_BayesNet.R::filterHighPNormals` | post-filter Jaccard ≥0.94 | ✓ implemented (Jaccard 0.979 on oligo) |
| state→CN-ratio proxy (step 20) | `inferCNV_HMM.R:1195-1200` (i6) + `inferCNV_i3HMM.R:409-411` (i3) | bit-exact | ✓ both i6 and i3 `max_diff = 0.000e+00` |
| mask_non_DE (step 21) | `inferCNV_mask_non_DE.R` | bit-exact | ✓ matrix `max_diff = 1.110e-16`; mask boolean xor_frac < 1e-4 (residual-inference soft floor — see commit `f468e2d`) |
| denoise (step 22) | `inferCNV_ops.R:2302-2346` (`clear_noise_via_ref_mean_sd`) | bit-exact | ✓ implemented; `noise_logistic=True` (R sigmoidal mask) raises `NotImplementedError` |
| Top-level `pipeline.infercnv()` ⇒ Phase 3 | — | — | ✓ `cfg.HMM=True` or any of `BayesMaxPNormal>0`/`mask_nonDE_genes`/`denoise` triggers `run_phase3`; permanent regression test in `tests/integration/test_top_level_phase3.py` |

### Result schema additions

`InferCNVResult` gained Phase 3 fields (all default `None`):

- `bayes_posterior` — `(n_regions, K)` float64; `K=6` (i6) or `3` (i3). Persisted from `gibbs_result["cnv_posterior"]`.
- `de_mask` — `(n_cells, n_bins)` bool; True = kept (DE), False = masked to `center_val`.
- `denoised_matrix` — `(n_cells, n_bins)` float32; values inside the ref-mean ± sd band flattened to ref mean. Kept separate from `cnv_matrix_fc` for non-destructive pre/post comparison.
- `hmm_proxy_matrix` — `(n_cells, n_bins)` float64; post-step20 state→CN-ratio remap. Equivalent to R's `hmm.infercnv_obj@expr.data` after step 20.

`InferCNVConfig` gained 9 Phase 3 toggle fields with R-aligned defaults
(`reassignCNVs=True`, `mask_nonDE_genes=False`, `denoise=False`, etc.);
unsupported R-default branches raise `NotImplementedError` with explicit
"set X=False to use the supported branch" messages — no silent
divergence.

### Tests

- pytest: **220 passed** (excluding `tests/test_regression.py`)
  - +5 vs 0.2.0.dev2 baseline 215: orchestrator unit (P0.2), step20 i3 parity (P2.2), 3 top-level integration tests (P1.1)
- Wheel: `pyinfercnv-0.2.0-py3-none-any.whl`; `twine check` PASSED. Pure-Python (no `.so`/`.pyd`/`.dylib`); numba `@njit(cache=True)` for hot kernels only.
- R-parity gate: `tests/test_r_parity.py` 19/19 passed.

### Known limitations / 0.3 backlog

R's non-default branches that the Python port fail-loud-blocks at the
orchestrator boundary (set the listed flag to use the supported branch):

- `BayesNet reassignCNVs=True` (R default; Py uses `removeCNV`-only port at `bayesnet/gibbs.py`)
- `denoise noise_logistic=True` (R sigmoidal mask at `inferCNV_heatmap.R:2783`)
- `BayesNet postMcmcMethod='removeCells'` (R variant at `bayesnet/gibbs.py:181-185`)
- `hspike sim_method='simple'` / `'splatter'` (R alt simulation paths at `hmm/hspike.py:411`)
- `preprocess threshold='auto'` (R threshold-auto branch at `preprocess/max_threshold.py`)
- `denoise apply_median_filtering` (R alt denoise path)
- Phase 3b variational approximation of Gibbs (`bayesnet/vb` — performance optimization; full-quality Gibbs is the primary path)
- i6 hspike μ/σ baseline drift (Jaccard 0.918 → 0.979 between commit `384e983` and HEAD; root cause not isolated, no regression — see `CODEX_HANDOFF.md §0.1-B`)

### Codex collaboration

The Phase 3 wire-fix that landed in this release was developed with two
rounds of plan-stage Codex meta-review (gpt-5.5, xhigh) plus two rounds
of post-implementation review. See
`docs/superpowers/findings/2026-04-25-phase3-wire-fix-summary.md` and
the reviews under `docs/superpowers/reviews/`.

---

## 0.2.0.dev2 (2026-04-24)

### Phase 1 bit-exact — all intermediates float64

All Phase 1 preprocess modules (normalize, subtract_ref, log2_plus1,
max_threshold, center_cells, invert_log2, prune_outliers) now use
float64 for intermediate arithmetic. Public output contract
(`result.cnv_matrix`, `result.cnv_matrix_fc`) still float32 —
downcast happens only at result assembly in `pipeline.py`.

Before / after on oligodendroglioma step-TSV fixtures:

| Step | Before | After |
|---|---|---|
| step03 normalize | `max_diff < 1e-2` (4 relaxed) | **`< 1e-10` (4 bit-exact)** |
| step04 log2_plus1 | `< 1e-5` (4 approximate) | **`< 1e-10` (4 bit-exact)** |
| step08 subtract_ref | `< 1e-3` (4 relaxed) | **`< 1e-10` (4 bit-exact)** |
| step09 max_threshold | `< 1e-6` (4 approximate) | **`< 1e-10` (4 bit-exact)** |
| step11 center_cells | `< 1e-4` (4 approximate) | **`< 1e-10` (4 bit-exact)** |
| step14 invert_log2 | `< 1e-4` (4 approximate) | **`< 1e-10` (4 bit-exact)** |
| step16 prune_outliers | `< 1e-4` (4 approximate) | **`< 1e-10` (4 bit-exact)** |
| step10 smooth | `< 1e-3` (4 relaxed) | `< 1e-3` (scipy `uniform_filter1d` accumulation; untightened — needs further investigation) |

All `tests/test_r_parity.py` floors tightened to `1e-10` on the bit-exact steps.
3 unit-test docstrings rewritten (subtract_ref / max_threshold / center_cells)
to remove float32-dtype assertions: leaf functions return float64, public
float32 contract is enforced at `result.cnv_matrix` assembly in
`pipeline.py` only.

**HMM i3 end-to-end Jaccard on oligodendroglioma: 0.976 (unchanged)**.
The Phase 1 float64 precision is truncated back to float32 at
`result.cnv_matrix` before Phase 2 consumes it, so the bit-exact gain
does not propagate into HMM state calls. Plumbing Phase 2 to consume a
float64 intermediate is the next-session work. See
`docs/superpowers/HANDOFF_bit_exact.md` §8 for the handoff.

### Primary parity metric shift: Spearman ρ on CNV matrix

Subcluster ARI on Leiden output is retired as a parity metric (Jason
v7 review 2026-04-23: "ARI 是彻底不要的，因为这个只是一个算法标签").
Cluster IDs are arbitrary algorithm-internal labels with no
cross-language semantic contract; ARI measures label-invariant
permutation agreement but does not reflect whether py and R produce
the same scientific signal. The primary metric going forward is
**Spearman ρ on the continuous post-Phase-1 CNV matrix** (step 14 —
the log2-FC signal every downstream step consumes):

| Fixture | cells × genes | per-cell ρ | global flat ρ | Pearson |
|---|---|---|---|---|
| oligodendroglioma (smart-seq2) | 184 × 8508 | 0.9998±0.0001 | 0.9998 | 0.9928 |
| Gao2021_Breast/DCIS1 (10x UMI) | 1399 × 9237 | 1.0000±0.0000 | 1.0000 | 0.9980 |
| Gao2021_Breast/TNBC1 (10x UMI) | 1022 × 9654 | 1.0000±0.0000 | 1.0000 | 0.9964 |
| Gao2021_Breast/TNBC3 (10x UMI) | 520 × 10835  | 1.0000±0.0000 | 1.0000 | 0.9987 |

Secondary metric — HMM state Spearman ρ (i3 ordinal, post-discretization):
DCIS1 0.921, TNBC1 0.962, TNBC3 0.958 (global flat). i6 state Spearman
is lower (0.74-0.82) because finer discretization amplifies boundary
drift; i6 on TNBC3 is not comparable because R's hspike crashed on
that fixture.

Raw data: `benchmarks/phase2/Gao2021_Breast/{spearman_benchmark,cnv_matrix_spearman}.md`.
Reproducer: `scripts/track_a1/cnv_matrix_spearman.py` + `scripts/track_a1/spearman_benchmark.py`.

Operational ARI retained at `tests/test_r_parity.py::test_step15_subclusters_ari_floor`
(floor 0.85) as a Leiden-wiring regression anchor. Not a parity claim.

### Track A — py = R Leiden fidelity closeout

Three-patient diagnostic (DCIS1/TNBC1/TNBC3) plus oligo established
that the underlying CNV matrix is near-identical between py and R
(ρ ≥ 0.9998). The HANDOFF v5/v6 "DCIS1 ARI 0.447 divergence" was a
label-arbitrariness + Leiden-RNG artifact at the categorical
subcluster level, not a signal-level disagreement. Full data +
methodology under `benchmarks/phase2/Gao2021_Breast/track_a_summary.md`
and `docs/superpowers/reviews/track-a-closeout-codex.md`.

### Added

- `InferCNVConfig.tumor_subcluster_n_seeds` (default 1) — opt-in
  `rbest` mode that runs Leiden N times and picks the partition with
  the highest CPM. KNN graph reused across seeds so incremental cost
  is Leiden-only.
- `InferCNVConfig.tumor_subcluster_min_size` (default None) — opt-in
  small-cluster merge via KNN majority vote; documented non-CPM-optimal.
- `leiden_subcluster` new `n_jobs` kwarg (default `-1`): parallelises
  the brute-force euclidean KNN over all cores (codex G3 Track C).
- User warning when `leiden_subcluster` is called with `n_cells >= 8000`
  about exact-brute quadratic scaling.
- 12 regression anchor tests (KNN backend equivalence, float32↔float64
  label invariance, label coverage, canonical manual CPM,
  min_subcluster_size correctness, n_seeds rbest dominance + determinism).

### Changed

- `InferCNVConfig.validate()` now errors on `analysis_mode != "subclusters"`;
  `"samples"` and `"cells"` are Phase 3 work and were previously silently
  accepted.
- `InferCNVConfig.random_state` now defaults to `42` and is threaded through
  Phase 2 hspike/HMM execution to match the local R reference workflow's
  `set.seed(42)`.
- `hmm.hspike` now mirrors R dropout fitting more closely by using
  `smooth.spline(log(m), p0)` semantics on positive means and collapsing
  duplicate spline `x` values by mean instead of keeping an arbitrary row.
- README Parity-status (Phase 2) rewritten to cite the three-patient
  Track A distribution data instead of a single DCIS1 run.
- `pipeline_phase2.py` docstring: removed stale "future field in Phase 2"
  language for `tumor_subcluster_partition_method`; now lists the three
  new subcluster knobs.

### Notes

- Local release prep bumps `pyproject.toml` and `pyinfercnv.__version__`
  to `0.2.0.dev2`. Tags `v0.2.0.dev0` and `v0.2.0.dev1` remain immutable.
- Pure-Python wheel invariant preserved: `python -m build` emits
  `py3-none-any.whl` with zero `.so`/`.pyd`/`.dylib`. Twine check passes
  on both wheel and sdist.

## 0.2.0.dev1 (unreleased)

### Performance
- **Leiden edge dedupe vectorized** (G3 codex Q3). Replaced
  `a.tolist()`/`b.tolist()`/`set(zip(...))` edge-set construction in
  `pyinfercnv.subcluster.leiden.leiden_subcluster` with an int64-packed
  `np.unique` pass. For 2000-cell / k_nn=20 graphs this drops the
  edge-build step from a Python-object hotspot to a pure-NumPy C-level
  sort; no behavioural change (same edge set, same Leiden partition;
  all 9 `tests/unit/test_subcluster_leiden.py` pass unchanged).

## 0.2.0.dev0 (unreleased)

### Added
- **Phase 2 — tumor subclustering**: `pyinfercnv.subcluster.leiden` on the
  python-igraph C-core Leiden partitioner with per-annotation-group mode
  (`cluster_by_groups=True`). Honors `random_state`; cross-platform
  deterministic (codex C1 fix, commit `4b7206d`).
- **Phase 2 — HMM state calls**: `pyinfercnv.hmm.i6` (6-state) and
  `pyinfercnv.hmm.i3` (3-state) populate `InferCNVResult.hmm_states` and
  `InferCNVResult.hmm_states_i3`. Numba-JIT Viterbi kernel
  (`pyinfercnv.kernels.hmm_viterbi_numba`) implements R's `Viterbi.dthmm.adj`
  emission semantics (codex C5 fix, commit `e32adcf`).
- **Phase 2 — hspike calibration (i6)**: synthetic-spike NB parameter fit
  for i6 emission means (`pyinfercnv.hmm.hspike.calibrate_i6_emission`).
- **Phase 2 — CNV region calls**: `InferCNVResult.cnv_regions` BED-like
  DataFrame (per cell-group / chromosome, RLE over states).
- **Phase 2 — Phase 1 filter-population fix**: pipeline now filters genes
  on all cells (not ref cells only), closing the Phase 1 upstream-alignment
  gap that was suppressing tier-4 parity (commit `90eabe1`).
- **Phase 2 — py-vs-R wallclock + regression-detection harness**:
  `scripts/phase2_benchmark/` drives R infercnv + pyinfercnv side-by-side
  across the pycopykat-sliced patient corpus (Gao2021_Breast,
  Kim2020_Lung, Lee2020_Colorectal, Obradovic2021_Kidney,
  Qian2020_Ovarian), aggregating Jaccard / ARI / wallclock into a CSV.
  **This is not a real-world validation benchmark** — the 3CA cell-type
  annotations consumed by both pipelines as `ref_group_names` are upstream
  automated labels, not expert-curated ground truth, and consistency
  between py and R does not imply correctness against external CNV truth.
  The harness is useful for (a) per-patient wallclock speedup evidence
  (early observations: 25–192× for 0.5–1.4 kilocell inputs) and (b)
  detecting algorithmic drift between py and R across future releases.
- **Tutorial — `examples/tutorial_phase2.py`** + `examples/r_driver_phase2.R`:
  percent-format source + R reference driver for the oligodendroglioma
  fixture. Rebuild via `uv run python examples/_build_notebooks.py`.

### Parity
- **tier-3.5 empirical on smart-seq2 oligodendroglioma fixture**:
  step 15 subcluster ARI = 1.000; step 17 HMM i6 Jaccard = 0.979;
  step 17 HMM i3 Jaccard = 1.000 (`tests/test_r_parity.py`, floors
  0.85 / 0.96 / 0.99).
- **kernel-isolated parity**: 0.9999 on R-aligned input to
  `pyinfercnv.kernels.hmm_viterbi_numba` (bit-exact-class; not a
  cross-pipeline claim).
- **10x UMI data**: use `cutoff=0.1` (R wiki-canonical for 10x, vs
  1.0 for smart-seq2). Early py-vs-R **consistency** measurements
  (first 3 patients of the regression-detection harness) show
  variable agreement: Gao2021_Breast/DCIS1 i3 Jaccard 0.887 / i6
  Jaccard 0.690; TNBC1 i3 0.907 / i6 0.421; TNBC3 i3 0.937 (i6 R-side
  hspike `rowMeans` crash). The divergence is almost certainly
  algorithmic — codex diagnostic review attributes it to
  `RANN::nn2` (R KD-tree) vs `sklearn.NearestNeighbors(algorithm="brute")`
  tie-breaking on kilocell tumor KNN graphs, which cascades into
  different Leiden partitions on the tumor side while reference-group
  partitions match exactly (ARI 1.000 on both patients). This is
  **consistency, not correctness** — neither py nor R has been
  validated against external ground truth on this data; the oligo
  smart-seq2 fixture above is the only correctness-validated parity
  surface.

### Compatibility
- Breaking — `pyinfercnv.preprocess.filter_low_expression_genes` removed
  the `reference_cell_idx` kwarg (pre-PyPI; no external callers). The
  hspike caller that needed ref-only filtering uses explicit
  `X[ref_idx, :]` slicing before the call instead (commit `4aacc01`).

## 0.1.0.dev0 (unreleased)

### Added
- Phase 1: preprocess (filter, CPM normalize, log2, subtract reference with bounded logic, max-centered threshold)
- Phase 1: smoothing (scipy.ndimage.uniform_filter1d × 2 + numba tail helper) per chromosome
- Phase 1: per-cell median centering
- Phase 1: outlier pruning
- Phase 1: invert log2
- Phase 1: AnnData-native API (`counts_layer="counts"` default)
- Phase 1: typer CLI (`pyinfercnv run-h5ad`)
- Phase 1: matplotlib heatmap visualization (no omicverse dependency)
- Phase 1: gene reference parquets shipped for hg38 / hg19 / mm10
- Phase 1: R-parity test suite (mixed tier-4 bit-exact / tier-4 approximate)
- Phase 1: benchmark scripts reused from pycopykat template

### Parity
- See README.md "Parity status" section.
