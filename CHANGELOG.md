# Changelog

## Unreleased (post-v0.2.0.dev1)

### Track A — py = R Leiden fidelity closeout

Three-patient diagnostic (DCIS1/TNBC1/TNBC3) establishes that py and
R Leiden are statistically equivalent in distribution and that py
finds equal-or-higher CPM objective on the same graph. The HANDOFF
v5/v6 "DCIS1 ARI 0.447 divergence" was a single-draw observation
inside R's own 0.10-0.29 self-noise floor. Full data + methodology
under `benchmarks/phase2/Gao2021_Breast/track_a_summary.md` and
`docs/superpowers/reviews/track-a-closeout-codex.md`.

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
- README Parity-status (Phase 2) rewritten to cite the three-patient
  Track A distribution data instead of a single DCIS1 run.
- `pipeline_phase2.py` docstring: removed stale "future field in Phase 2"
  language for `tumor_subcluster_partition_method`; now lists the three
  new subcluster knobs.

### Notes

- No `twine upload` authorization granted yet. Tags `v0.2.0.dev0` and
  `v0.2.0.dev1` remain immutable. Tag `v0.2.0.dev2` is **not** created
  automatically; Jason bumps when ready.
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
  step 15 subcluster ARI = 1.000; step 17 HMM i6 Jaccard = 0.968;
  step 17 HMM i3 Jaccard = 0.976 (`tests/test_r_parity.py`, floors
  0.85 / 0.90 / 0.90).
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
