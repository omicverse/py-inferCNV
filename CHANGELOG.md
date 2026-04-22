# Changelog

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
- **Phase 2 — 17-patient 3CA cross-cohort benchmark harness**:
  `scripts/phase2_benchmark/` iterates the pycopykat 17-patient manifest
  (Gao2021_Breast, Kim2020_Lung, Lee2020_Colorectal, Obradovic2021_Kidney,
  Qian2020_Ovarian), runs R infercnv + pyinfercnv side-by-side, aggregates
  Jaccard / ARI / wallclock into `benchmarks/phase2/phase2_py_vs_r_summary.csv`.
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
- **3CA 10x UMI cross-cohort** (pending benchmark aggregate): use
  `cutoff=0.1` (wiki-canonical for 10x data, vs 1.0 for smart-seq2).
  Known R upstream issue: `infercnv::run(HMM_type="i6", ...)` crashes in
  hspike `rowMeans` on some 3CA fixtures (independent of pyinfercnv);
  i3 runs successfully and py-vs-R parity is evaluated on i3 only for
  those patients.

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
