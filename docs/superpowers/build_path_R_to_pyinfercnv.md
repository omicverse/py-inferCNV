# Build Path — R inferCNV → pyinfercnv

## 0. Reading order and audience

This document is the architectural / historical companion to
[`algorithm_correspondence_R_vs_Python.md`](./algorithm_correspondence_R_vs_Python.md).
The companion doc answers "which Python function maps to which R function,
per R `step_count`"; this doc answers "why the Python port was structured
this way and how each phase was validated against R". It is written for
two audiences:

1. **Maintainers** picking up the project after the v0.2.0 release —
   they need to know the load-bearing decisions (state encoding,
   observation space, layout convention) before touching any module.
2. **Reviewers and auditors** who want to verify the parity claims in
   `CHANGELOG.md ## 0.2.0` and `NAMESPACE_PARITY.md` from primary
   sources — every commit, file, and parity number cited here has been
   verified against `git show`, the source file, or
   `tests/test_r_parity.py`.

Cross-references inside the package:

- Step-by-step R↔Py mapping → `docs/superpowers/algorithm_correspondence_R_vs_Python.md`
- Module crosswalk + tier table → `NAMESPACE_PARITY.md`
- Per-step bit-exactness numbers → `CHANGELOG.md ## 0.2.0`
- Executable parity proofs → `tests/test_r_parity.py` (19 step-level tests)
- Top-level wire regression → `tests/integration/test_top_level_phase3.py`

This document was rebuilt on 2026-04-26 from those surviving artifacts
plus the R source under `/media/jason/T7/rerbulid/infercnv/infercnv-master/R/`
and the `pyinfercnv` git history. See §10 for caveats on the rebuild.

## 1. Project values and non-negotiables

Three invariants were locked at the start of Phase 1 and survive intact
in v0.2.0. Every later design decision can be traced back to one of
them.

### 1.1 Pure-Python wheel

The published artifact is `pyinfercnv-0.2.0-py3-none-any.whl`
(`dist/pyinfercnv-0.2.0-py3-none-any.whl`). The wheel contains zero
compiled extensions: `unzip -l dist/pyinfercnv-0.2.0-py3-none-any.whl |
grep -E "\.so|\.pyd|\.dylib"` returns empty. Hot kernels live in
`pyinfercnv/kernels/` and use `numba.njit(cache=True)` for JIT
compilation at first call:

- `pyinfercnv/kernels/smooth_center_numba.py` (Phase 1, R `stats::filter`
  parity)
- `pyinfercnv/kernels/smooth_tail_numba.py` (Phase 1, R `.smooth_helper`
  tail)
- `pyinfercnv/kernels/hmm_viterbi_numba.py` (Phase 2, R
  `Viterbi.dthmm.adj`)
- `pyinfercnv/kernels/bayesnet_gibbs_numba.py` (Phase 3, BUGS Mixture
  Model Gibbs)

`pyproject.toml` declares `requires = ["hatchling"]` /
`build-backend = "hatchling.build"` (lines 1-3) and a
`requires-python = ">=3.10,<3.13"` floor (line 11). The `py3-none-any`
classifier is enforced by hatchling because there is no
`tool.hatch.build.targets.wheel.platlib` entry. Source: `README.md`
line 16 ("Wheel is `py3-none-any` (pure Python; numba JIT at first
call)") and `CHANGELOG.md ## 0.2.0` line 73.

### 1.2 R-parity-first

The Python port is not an "infercnv-like" reimplementation; every
module is held to one of three explicit parity tiers, recorded in
`NAMESPACE_PARITY.md` lines 5-9 and `tests/test_r_parity.py`:

- **bit-exact**: `max_diff < 1e-10` against the R intermediate fixture
  written by `tests/r_reference.R`.
- **approximate / empirical**: distributional floor (Spearman ρ,
  Jaccard, soft-tier `|ΔP|<0.10` rate) against R output, justified
  module-by-module.
- **fail-loud guard**: an R-default branch the Python port deliberately
  did not implement raises `NotImplementedError` or `ValueError` at
  the orchestrator boundary, with an explicit "set X=False to use the
  supported branch" message. See §9 for the full inventory.

The R-default-branch enumeration appears in `CHANGELOG.md ## 0.2.0`
"Known limitations" lines 76-89 and is enforced at six guard sites in
the source (counted in §9 below).

### 1.3 AnnData-native

The single I/O contract is `AnnData` in / `AnnData` out (with
`adata.obsm["X_cnv"]`, `adata.uns["cnv"]`, `adata.obs["cnv_subcluster"]`
keys per `pyinfercnv/result.py:184-227`). There is no Seurat, no `.rds`
checkpointing, no S4 object surface, no `phyloseq`/`SingleCellExperiment`
shim. Gene positions ship as parquet (hg38 / hg19 / mm10 in
`pyinfercnv/data/`) so the R `inferCNV.R::CreateInfercnvObject` flow
reduces to one `pyinfercnv.io.h5ad.extract_counts` call plus a
chromosome-layout join (`pyinfercnv/pipeline.py:84-117`).

Source: `README.md` line 5, `NAMESPACE_PARITY.md` lines 76-79
("Skipped (out of scope): seurat_interaction — AnnData-only").

## 2. Repository topology

Top-level layout of `/media/jason/T7/rerbulid/pyinfercnv/` (one-line
annotations; verified by `find pyinfercnv -name "*.py"`):

```
pyinfercnv/
├── __init__.py             public API (infercnv, run_phase3, InferCNVConfig, InferCNVResult, __version__)
├── config.py               InferCNVConfig dataclass (R kwarg parity)
├── result.py               InferCNVResult dataclass (Phase 1/2/3 fields)
├── pipeline.py             top-level infercnv() — Phase 1 + dispatches to Phase 2/3
├── pipeline_phase2.py      run_phase2 — subcluster + HMM
├── pipeline_phase3.py      run_phase3 — BayesNet + step20 + mask + denoise
├── io/                     h5ad reader, gene-position parquet loader
├── preprocess/             step 2-9, 12, 14 leaf functions
├── smooth/, center/, cna/  steps 10, 11, 16 wrappers
├── subcluster/             step 15 backends (leiden, random_trees, qnorm)
├── hmm/                    step 17 (i6, i3, hspike calibration)
├── bayesnet/               step 18 + 19 (Gibbs, filterHighPNormals)
├── mask_de/                step 21 (Wilcoxon mask_non_DE)
├── denoise/                step 22 (clear_noise_via_ref_mean_sd)
├── kernels/                @njit hot loops (4 files; see §1.1)
├── validation/             r_parity.py — max_diff / Jaccard / Spearman helpers
├── viz/                    matplotlib heatmap (no omicverse import)
└── cli.py                  typer CLI

tests/
├── test_r_parity.py                  19 step-level R-parity tests (the load-bearing gate)
├── r_reference.R                     generates tests/r_out/*.tsv via Rscript
├── test_{smoke,io_contract,viz_smoke,wheel}.py
├── unit/                             per-module fast tests (28 files)
└── integration/test_top_level_phase3.py   permanent regression — infercnv → run_phase3 wire

scripts/, examples/, benchmarks/phase2/, docs/superpowers/
```

The `kernels/`, `validation/`, and `pipeline_phase{2,3}.py` separation
is deliberate — orchestration (Phase 2/3) lives at the package root
next to `pipeline.py`, leaf algorithms live in topical subpackages, and
hot loops live in `kernels/` so they can be JIT-compiled independently
of the orchestration code.

## 3. The R↔Py axis: state encodings, observation spaces, layouts

Four cross-cutting conventions were fixed in Phase 1/2 that every
later module relies on. Getting any one of them wrong silently corrupts
parity, so they are documented here once instead of being re-derived
at each call site.

### 3.1 i6 state encoding (1-indexed in R, 0-indexed in Python)

R uses 1-based state labels `{1, 2, 3, 4, 5, 6}` mapping to CN levels
`(0.01, 0.5, 1.0, 1.5, 2.0, 3.0)`; the neutral state is `3`.
`pyinfercnv/hmm/i6.py:7-12` re-encodes this to 0-based int8 `{0..5}`,
neutral=`2`. The mapping is symmetric (`R_state - 1 == Py_state`), so
filtering / region calls / step 20 lookups translate one-for-one.

The same shift applies to the Gibbs sampler (`pyinfercnv/bayesnet/gibbs.py:101-103`),
the filter-high-p-normals helper
(`pyinfercnv/bayesnet/filter_high_p_normals.py:32-34`), and the step-20
state→CN-ratio map (`pyinfercnv/pipeline_phase3.py:96-98`).

### 3.2 i3 state encoding

R uses `{1, 2, 3}` for `(DEL, neutral, AMP)` with neutral=`2`. Python
uses `{0, 1, 2}` with neutral=`1`. See `pyinfercnv/hmm/i3.py:10-13` and
the same neutral-index dict at `pyinfercnv/bayesnet/filter_high_p_normals.py:34`.

### 3.3 The "linear-FC vs log-FC" observation-space switch (Phase 2)

The single most consequential Phase 2 decision was that R's HMM
emission consumes the post-step14 `invert_log2` matrix (linear FC,
centred near 1.0), not the post-step10 log2 matrix the early Python
port was reading. This was discovered during the i3 audit and fixed in
**commit `a8d4526`** (2026-04-23, "Phase 2 i3 R-parity — HMM switches
to linear FC (step14 invert_log2)").

The R source for the constraint:
- `inferCNV_ops.R:1031` runs step 14 `invert_log2` before the HMM block.
- `inferCNV_HMM.R:366` shows the HMM reads `@expr.data`, which by then
  is the post-`invert_log2` linear FC matrix.

The fix at `pyinfercnv/pipeline_phase2.py:325-327` (i6 path) and
`pipeline_phase2.py:339-348` (i3 path) is to feed
`result_phase1.cnv_matrix_fc` (the float64-upcast linear-FC matrix)
into both `estimate_i3_state_params` and `_run_hmm_by_subcluster`.

Measured impact on the oligodendroglioma fixture, recorded verbatim in
the commit body:

| Path | Before | After |
|---|---|---|
| step17 HMM i3 Jaccard | 0.976 | **1.000** (bit-exact to R) |
| step17 HMM i6 Jaccard | 0.968 | unchanged at 0.968 (i6 needed the companion hspike rewrite — see §5.2) |

After the companion hspike rewrite (`3d9ec79`) the i6 Jaccard moved to
`0.979` (current `tests/test_r_parity.py::test_step17_hmm_i6_jaccard_floor`
floor 0.96, observed 0.979 per `CHANGELOG.md ## 0.2.0` line 40).

### 3.4 Matrix layout: R is genes × cells; Python is cells × bins

R `infercnv_obj@expr.data` is a `(n_genes, n_cells)` matrix. Python
operates on `(n_cells, n_bins)` throughout — `result.cnv_matrix`,
`result.cnv_matrix_fc`, `result.hmm_states`, `result.hmm_states_i3`,
`result.de_mask`, `result.denoised_matrix`, `result.hmm_proxy_matrix`
are all cells × bins (see field shapes in `pyinfercnv/result.py:73-138`).
The R-parity fixtures dumped by `tests/r_reference.R` are loaded as R
layout (genes × cells) and transposed before comparison; see
`tests/test_r_parity.py:53-56` (`_load_r_step` returns the TSV as-is)
and the per-test `.T` calls (e.g. `test_step03_normalize_parity` line
158, `test_step10_smooth_pyramidinal_per_chromosome` line 374).

The transpose convention is asymmetric for I/O: `extract_counts`
already returns cells × genes from AnnData (because AnnData's `.X` is
cells × vars), so Python-side data starts in the right layout and
never needs to round-trip through R's orientation except in tests.

### 3.5 cnv_matrix vs cnv_matrix_fc and the float64 handoff

`InferCNVResult` carries two output matrices:

- `cnv_matrix`: log2-FC space, float32 — the "headline" CNV matrix
  (Phase 1's step14 input domain).
- `cnv_matrix_fc`: linear FC space (`2 ** cnv_matrix`), float32 — the
  "what R's `@expr.data` looks like at step 14" matrix.

Both are float32 in the public contract (`pyinfercnv/result.py:88-90`)
because Phase 1's published wheel had a float32 contract from
v0.1.0.dev0. After `ca8bda8` ("Phase 1 bit-exact — float64 intermediates;
r_parity floors tightened to 1e-10", 2026-04-23) all Phase 1 leaf
functions return float64; the public float32 cast happens once at
result assembly (`pyinfercnv/pipeline.py:288-297`). To avoid losing
that precision at the Phase 1 → Phase 2 handoff, a transient
`cnv_matrix_f64` companion field was introduced in the same commit
chain (declared at `pyinfercnv/result.py:97-98`) and consumed at
`pipeline_phase2.py:258-262`. The top-level `infercnv()` nulls
`cnv_matrix_f64` after Phase 2 (`pipeline.py:311-313`) so it never
crosses the public boundary.

### 3.6 Random state plumbing

`InferCNVConfig.random_state` defaults to `42`
(`pyinfercnv/config.py:128-129`), matching `set.seed(42)` in
`tests/r_reference.R`. The seed threads through:

- `pipeline.infercnv` → `run_phase2(..., random_state=cfg.random_state, ...)`
  (`pyinfercnv/pipeline.py:302-305`).
- `_run_subclustering` → `leiden_subcluster(random_state=...)` /
  `random_tree_subcluster(random_state=...)` (`pipeline_phase2.py:467-477`).
- `leiden_subcluster` then wraps `community_leiden` in
  `ig.set_random_number_generator(random.Random(random_state))` with a
  `try/finally` restore — fixed in **commit `7c9eb8c`** ("fix(subcluster/leiden):
  honour random_state (codex C1)") after Codex flagged the C1 critical
  finding that the previous code silently ignored the seed.
- `_calibrate_hmm_emission` → `calibrate_i6_emission(..., random_state=random_state, ...)`
  (`pipeline_phase2.py:312-318`) for the hspike NB simulation.
- `run_bayesnet_gibbs` derives per-region seeds from a
  `np.random.default_rng(int(random_state))` stream
  (`bayesnet/gibbs.py:251-253`).

Cross-language RNG bit-equivalence is **not** a contract — R's
Mersenne-Twister + JAGS L'Ecuyer streams cannot be reproduced from
NumPy PCG-64. The contract is "same public seed → reproducible Python
result" plus "distribution-level parity to R within the declared
tier".

## 4. Phase 1 — preprocess + smoothing + centering (R steps 2-14, 16)

### 4.1 Scope

Phase 1 covers R `inferCNV_ops.R::run` steps 2 through 14 (skipping
gated steps 5, 6, 7) plus step 16. The R `step_count` integer is
incremented at every numbered block in `inferCNV_ops.R:532-1615`; the
Python orchestrator's docstring at `pipeline.py:1-29` enumerates the
exact correspondence:

```
2.  Remove lowly expressed genes    -> filter_low_expression_genes
3.  Normalize by seq depth          -> normalize_by_seq_depth
4.  log2(x+1)                       -> log2_plus1
5.  (skipped; scale_data default False)
6.  (skipped; num_ref_groups not set)
7.  (skipped; tumor subclustering Phase 2)
8.  Subtract ref mean (pre-smooth)  -> subtract_reference (use_bounds default True)
9.  Max-centered threshold          -> apply_max_centered_threshold
10. Smooth per chromosome           -> smooth_pyramidinal per chr
11. Center cells (median)           -> center_cells
12. Subtract ref mean (post-smooth) -> subtract_reference (2nd)
14. invert_log2                     -> invert_log2
16. Prune outliers                  -> prune_outliers (when enabled)
```

Step 13 (remove genes at chromosome ends) is intentionally skipped in
the Python port — see §4.5.

The Phase 1 pipeline body (`pipeline.py:120-297`) is a flat sequence
of these calls with `psutil`-based profile blocks
(`_profile_block`/`_rss_mb`) wrapping each step, so the per-block
wallclock and RSS land in `result.profile`.

### 4.2 The float64 intermediates decision

The original Phase 1 (commit `946f4ad`, "feat(algo): preprocess +
smooth + center + cna + validation modules", 2026-04-19) computed in
float32 throughout. That gave the public `result.cnv_matrix` the
correct dtype but accumulated ~1e-2 drift over 8508 genes × 184 cells
on the normalize step, which propagated into every later step.

The fix landed in **commit `ca8bda8`** ("Phase 1 bit-exact — float64
intermediates; r_parity floors tightened to 1e-10", 2026-04-23). The
commit body records the before/after on the oligodendroglioma fixture
verbatim:

```
step03 normalize    (was 1e-2,  now 4.9e-11)
step04 log2_plus1   (was 1e-5,  now 5.3e-14)
step08 subtract_ref (was 1e-3,  now bit-exact)
step09 max_threshold(was 1e-6,  now 0)
step11 center_cells (was 1e-4,  now bit-exact)
step14 invert_log2  (was 1e-4,  now 5.6e-15)
step16 prune_outliers(was 1e-4, now 4.2e-15)
step02 filter_genes (was and remains bit-exact)
```

Public contract preservation: the leaf functions return float64, and
the cast to float32 happens exactly once, at result assembly in
`pipeline.py:288-297`. Three unit tests had `dtype == np.float32`
assertions; those were rewritten to use `np.issubdtype(x, np.floating)`
in the same commit (CHANGELOG dev2 lines 122-125).

### 4.3 The step10 R-exact rewrite

Step 10 — pyramidinal smoothing — was the last Phase 1 step to reach
bit-exact. The original implementation used
`scipy.ndimage.uniform_filter1d` twice plus a numba tail helper; the
two `uniform_filter1d` calls accumulated in a different order than R's
`stats::filter`, leaving step 10 stuck at `max_diff < 1e-3` after
`ca8bda8` (`CHANGELOG.md ## 0.2.0.dev2` line 121).

The R-exact rewrite landed in **commit `edddb51`** ("Phase 1 bit-exact
收尾 — step10 smooth_pyramidinal R-exact 化", 2026-04-24). Two
substantive changes:

1. New file `pyinfercnv/kernels/smooth_center_numba.py`
   (`smooth_center_interior`): a single centered direct convolution
   with a pre-divided triangular coefficient vector, float64
   accumulation, `prange` over rows. This is bit-exact with R's
   `stats::filter(vals, custom_filter, sides=2)` from
   `src/library/stats/src/filter.c::do_cfilter`.
2. `pyinfercnv/smooth/pyramidinal.py:48-55` was rewritten as a
   two-stage pipeline: `smooth_center_interior(out, X64, window_length)`
   for the interior, then `smooth_tail_overwrite(out, X64,
   window_length)` for the tail. Critically, the tail is computed
   from the **original** input `X64`, not from the post-interior
   `out` matrix — this matches the R algorithm at
   `inferCNV_ops.R:2483-2532` and is documented in the kernel
   docstring at `kernels/smooth_tail_numba.py:18-21`.

The pyramidinal weights are
`[1, 2, ..., tail, tail+1, tail, ..., 2, 1] / (tail² + window_length)`
where `tail = (window_length - 1) // 2` (default `tail = 50` for the
R-default `window_length = 101`). The tail formula at
`smooth_tail_numba.py:60-88` uses R's dynamic-denominator scheme:
`denom = tail² + window_length - r_left*(r_left+1)/2 - r_right*(r_right+1)/2`
where `r_left/r_right` shrink as the position approaches the edge.

Measured impact recorded in `edddb51` commit body:
`step10 max_diff: 1e-3 → 6.44e-15` on oligodendroglioma. Unit-test
floor at `tests/test_r_parity.py:377` tightened from `< 1e-3` to
`< 1e-10`.

### 4.4 R-parity gates (Phase 1)

After `ca8bda8` + `edddb51`, every Phase 1 step has a `< 1e-10` gate
in `tests/test_r_parity.py`. The matching assertions are at lines
139 (step 02 — `bit_exact_assert(tol=1e-10)` over the kept-gene
matrix), 161 (step 03), 177 (step 04), 207 (step 08), 223 (step 09),
240 (step 11), 257 (step 14), 326 (step 16), and 377 (step 10
`diff_full < 1e-10`). The `CHANGELOG.md ## 0.2.0` table lines 15-25
records the observed `max_diff` per step.

Step 14 also has a stochastic-friendly
`test_step14_cnv_matrix_spearman_oligo` (line 262): global flat
Spearman ρ ≥ 0.999 and per-cell minimum ρ ≥ 0.95 against R's step14
output. This is the test that anchors the `ρ = 1.0000` /
`ρ = 0.9998` headline in `README.md` lines 109-114.

### 4.5 Known limitations / 0.3 backlog from Phase 1

The Phase 1 surface that is **not** ported, deferred to v0.3:

- `threshold='auto'` for `apply_max_centered_threshold`. The current
  guard accepts `int | float | str | None` (`config.py:35`) but
  `validate()` only checks the numeric branch (`config.py:140-141`);
  the `'auto'` string branch is documented at
  `pyinfercnv/preprocess/max_threshold.py` but raises rather than
  computing R's auto-percentile. R source: branch in
  `inferCNV_ops.R::apply_max_threshold_bounds`.
- `sim_method='simple'` and `'splatter'` (R alternatives to the
  default `'meanvar'` for hspike NB simulation). Python only
  implements `meanvar`; `pipeline_phase2.py:554` hardcodes
  `sim_method="meanvar"` when calling `calibrate_i6_emission`.
- Step 13 — "remove genes at chromosome ends" — is not ported. The
  Python pipeline goes step 12 → step 14 directly
  (`pipeline.py:253-261`). R's step 13 is gated by an option not
  exposed in `InferCNVConfig`; the omission is benign on the
  oligodendroglioma fixture (Spearman ρ = 0.9998) but could matter
  on small chromosomes with few genes per arm.

See §9 for the full table.

## 5. Phase 2 — subclustering + HMM (R steps 15, 17)

### 5.1 The Leiden subcluster decision

R's `define_signif_tumor_subclusters_via_leiden`
(`inferCNV_tumor_subclusters.R`) calls `igraph::cluster_leiden`, which
itself wraps the C-core Leiden implementation in libigraph. The Python
port chose `python-igraph` for the same reason: both R-igraph and
python-igraph delegate to the same underlying C-core, so the partition
algorithm is byte-identical at the kernel level. The cross-language
divergences are limited to:

1. **PRNG backend**. R-igraph seeds the C PCG32 through R's
   Mersenne-Twister bridge; python-igraph seeds it through Python's
   `random.Random` bridge. The same public seed therefore feeds
   different bytes into the same C kernel, so per-seed identity is
   not a contract. See `subcluster/leiden.py` after commit `7c9eb8c`
   for the seed-restoration `try/finally`.
2. **Constant-Potts vs Modularity**. The Python parameterisation pins
   `leiden_method="simple"` + `leiden_function="CPM"` so both sides
   go through `python-igraph.community_leiden` with the same
   objective — see `README.md` lines 167-171.
3. **KNN backend**. Python uses
   `sklearn.NearestNeighbors(algorithm="brute")` by default; R uses
   `RANN::nn2` (KD-tree). The Track A diagnostic established that
   the KNN edge sets are Jaccard 1.0000 across DCIS1/TNBC1/TNBC3 (R
   parity status §"KNN edge-set parity"), so the brute/KD difference
   does not matter — but the partition can still diverge at the
   ARI level due to RNG cascading on borderline cells.

For these reasons subcluster ARI was retired as a parity metric in
**commit `3241b37`** ("retire ARI as parity metric; promote Spearman
ρ on CNV matrix", 2026-04-23). The replacement metric is Spearman ρ on
the continuous post-Phase-1 CNV matrix (the input every downstream
step actually consumes). The numbers in `README.md` "Phase 2 primary
metric" table (lines 109-114) are the load-bearing claim:
ρ = 1.0000 on three 3CA 10x UMI patients, 0.9998 on the
oligodendroglioma smart-seq2 fixture. ARI is kept as an internal
regression anchor with a deliberately permissive 0.85 floor at
`tests/test_r_parity.py::test_step15_subclusters_ari_floor` (line
545), explicitly labelled "operational anchor only" in the test
docstring (lines 484-496).

### 5.2 The hspike calibration (commit `3d9ec79`)

Phase 2's i6 emission means and per-cell-count sigmas are not
analytical — they come from a synthetic-spike NB simulation that
mirrors R's `.build_and_add_hspike` (`R/inferCNV_hidden_spike.R`).
The first Python implementation (`541cba6`, "feat(hmm/hspike):
implement all private helpers and calibrate_i6_emission") was a
best-effort port that diverged from R on several semantic details.

The strict R-parity rewrite landed in **commit `3d9ec79`** ("Phase 2
i6 hspike — strict R-parity rewrite", 2026-04-23). The commit body
enumerates seven structural rules — D.1 through D.7 — that every
later edit must respect. They are reproduced here verbatim from the
commit body because hspike has no other written specification:

- **D.1** Source matrix for `gene_means` is the post-step2 filter +
  post-step3 normalize **full** expr matrix (obs + ref cells), not
  ref-only raw counts. Plumbed via the new transient
  `InferCNVResult.cpm_matrix_f32` field, captured in `pipeline.py`
  right after `normalize_by_seq_depth`. R source:
  `inferCNV_hidden_spike.R:59`.
- **D.2** mean/variance spline and dropout logistic are fit on
  obs + ref pooled, not ref-only. R source:
  `inferCNV_meanVarSim.R:178-186` (`.get_mean_var_table`).
- **D.3** hspike Phase-1 replay does not re-filter genes (R never
  refilters `.hspike` post-creation).
- **D.4** hspike counts are normalized once at build-time to
  `median(colSums(last_normal_cells_expr))` — the variable-scope leak
  in R's `inferCNV_hidden_spike.R:160` is preserved verbatim.
- **D.5** hspike Phase-1 replay continues through step 14 invert_log2
  and step 16 remove_outliers_norm, so the calibration pool lives in
  linear-FC space (matches R's step-17 HMM input at
  `inferCNV_ops.R:1987`, `2820`).
- **D.6** Per-CN pooling is on observation cells only
  (`inferCNV_HMM.R:49-52`).
- **D.7** Fixed CN state order `[0.01, 0.5, 1, 1.5, 2, 3]`
  (`pyinfercnv/hmm/i6.py:39`); never reorder by empirical mean.

A separate sub-fix in the same commit corrected the sigma-trend lm
fit: R's `get_hspike_cnv_mean_sd_trend_by_num_cells_fit`
(`inferCNV_HMM.R` line 163-172) computes `replicate(nrounds,
sample(expr, size=ncells))` (which yields a `(ncells, nrounds)`
matrix) and then takes `rowMeans` — averaging across **rounds** per
sample-position, not across positions per round. The Python port
previously did `colMeans` and got slopes near `-0.5`; the fix is
`samples.mean(axis=1)`, which yields R-matched slopes near 0. This
matters because the wrong slope over-narrows sigmas on aggregated
subclusters and produces state-0 false positives downstream.

The post-rewrite measurement on oligo (commit body):

```
R cnv:0.5  mu=0.812 sd=0.071   | py mu=0.820 sd=0.073
R cnv:1    mu=0.998 sd=0.078   | py mu=0.997 sd=0.073
R cnv:1.5  mu=1.180 sd=0.120   | py mu=1.156 sd=0.118
R cnv:2    mu=1.314 sd=0.149   | py mu=1.307 sd=0.145
```

with state 0.01 still drifting (R 0.537 vs Py 0.647) — the residual
state-0 drift is the source of the i6-only Jaccard 0.918 → 0.979
fluctuation noted in `CHANGELOG.md ## 0.2.0` line 88.

### 5.3 The Viterbi numba kernel

R's HMM emission is **not** a standard Gaussian log-density. The
`Viterbi.dthmm.adj` function at `inferCNV_HMM.R:1101-1175` applies a
custom emission scoring: it overrides the per-state σ by
`median(state_sigmas)`, then computes
`1 / -scipy.stats.norm.logsf(|z|)` and normalises the row across the
K states. The early Python port used standard Gaussian and silently
mis-scored; the divergence was caught in commit **`7ed3596`**
("feat(hmm/kernel): R Viterbi.dthmm.adj emission parity (C5 fix)",
2026-04-21).

The fix structure (from `pyinfercnv/kernels/hmm_viterbi_numba.py`):

- `compute_log_emit(...)` precomputes a `(T, K)` log-emission matrix.
  `emission='rstyle'` (default) replicates R's median-σ override;
  `emission='gauss_std'` is retained for hmmlearn cross-checks.
- `_viterbi_dp_{numpy,numba}` operate on the precomputed log-emit
  matrix only. Public `viterbi_decode_{numpy,numba}` wrap the two
  steps for backward compatibility.
- The kernel is `@njit(cache=True)` for the DP loop; emission
  precompute uses `scipy.stats.norm.logsf` for stable asymptotics at
  large `|z|`.

Kernel-isolated parity is reported at Jaccard 0.9999 against R's
output when fed R's own step15 + step16 outputs — see
`NAMESPACE_PARITY.md` line 46 and `tests/unit/test_kernels_viterbi_adj.py`
which does the comparison directly. The end-to-end Jaccard residual
(0.979 on i6, 1.000 on i3) is not from the kernel — it's from
hspike (i6) and was zero on i3 once the linear-FC switch landed.

### 5.4 The i3 path's linear-FC switch

The same commit `a8d4526` discussed in §3.3 also tightened the i3
parity floor: `tests/test_r_parity.py::test_step17_hmm_i3_jaccard_floor`
moved from `0.90` to `0.99` (line 728), reflecting the observed
1.000 Jaccard on oligo. The test body's measurement comment at line
723-727 records: "step17 HMM i3 mean_jaccard=1.000 (bit-exact post
linear-FC switch)".

The same change brought i6 from 0.918 to 0.979 (companion-rewrite
effect plus hspike refinement); the tightened i6 floor is `0.96` at
line 635, observed 0.979.

### 5.5 Ref-cell handling

Reference cells in R are excluded from the HMM call — R's
`predict_CNV_via_HMM_on_tumor_subclusters` (`inferCNV_HMM.R:345-408`)
iterates only over tumor subclusters and broadcasts each subcluster's
state trace back to its members. Reference cells inherit the neutral
state for display.

The Python port mirrors this exactly:
`pipeline_phase2._run_hmm_by_subcluster` initialises
`hmm_states = np.full(shape, neutral_idx, dtype=np.int8)` (line 660),
then iterates only over `tumor_subclusters = unique[unique != _SUBCLUSTER_REF_SENTINEL]`
(line 667-669) and writes states only inside that loop. Reference
cells therefore stay at the neutral index everywhere, both for i6
(idx 2) and i3 (idx 1). The constants live at
`pipeline_phase2.py:135-142`.

When `cluster_by_groups=True` (R-default, `config.py:97-98`) reference
cells receive their own subcluster IDs (one per `reference_cat`
category) instead of the `-1` sentinel — see
`pipeline_phase2._run_subclustering` lines 486-517. The HMM step
still skips them via the sentinel filter; the subcluster IDs exist so
the `cnv_regions` BED-like output can group reference cells by
annotation category.

### 5.6 cnv_regions BED-like table

`pipeline_phase2._build_cnv_regions` (`pipeline_phase2.py:721-813`)
run-length-encodes the HMM state matrix into a long-format pandas
DataFrame with columns `[cell_group, subcluster, chromosome,
bin_start, bin_end, state, cn]`. Neutral runs are dropped; non-neutral
runs become one row each, tagged with the CN ratio looked up from
`I6_CNV_LEVELS` (i6) or the i3 `{0: -1.0, 1: 0.0, 2: 1.0}` map.

The output is consumed by Phase 3's BayesNet (each row = one CNV
region for Gibbs) and would be the input to a heatmap viz layer if
viz parity were in scope.

### 5.7 R-parity gates (Phase 2)

| Gate | Test function (line) | Floor / observed |
|---|---|---|
| step15 ARI | `test_step15_subclusters_ari_floor` (483) | floor 0.85 / observed 1.000 — operational only |
| step17 HMM i6 Jaccard | `test_step17_hmm_i6_jaccard_floor` (554) | floor 0.96 / observed 0.979 |
| step17 HMM i3 Jaccard | `test_step17_hmm_i3_jaccard_floor` (646) | floor 0.99 / observed 1.000 |
| step14 CNV Spearman ρ | `test_step14_cnv_matrix_spearman_oligo` (262) | floor 0.999 (global), 0.95 (per-cell min) / observed 0.9998, 0.99+ |

Spearman ρ on the kilocell 10x patients (DCIS1, TNBC1, TNBC3) is
measured by `scripts/track_a1/cnv_matrix_spearman.py` (per
`README.md` line 117) — note that script is referenced in the README
but lives outside the surviving working tree (see §10 caveat).

### 5.8 Known limitations / 0.3 backlog from Phase 2

- **i6 hspike μ/σ baseline drift**. CHANGELOG line 88 records that
  i6 Jaccard on oligo has fluctuated 0.918 ↔ 0.979 between commits
  `3d9ec79` and HEAD without an isolated root cause. No regression
  is observed at HEAD, but the floor (0.96) has only ~2 percentage
  points of headroom.
- `per_chr_hmm_subclusters=TRUE` (R kwarg). Not exposed in
  `InferCNVConfig`; Python always operates per chromosome at the
  block level via `chr_pos`, but the R variant that runs an
  independent subcluster step per chromosome is not ported.
- `analysis_mode='cells'` and `'samples'` (`config.py:75`,
  `config.py:158-164`). `validate()` raises `ValueError` if either
  is set; only `'subclusters'` is implemented. Phase 3 work.
- `use_KS=TRUE` for i3 (R alternative to the deterministic Z-form
  in `determine_mean_delta_via_Z`). Python only implements the Z
  form (`hmm/i3.py:67-122`); KS uses `rnorm` sampling on the R side
  and was not ported.

## 6. Phase 3 — BayesNet + filter + step20 proxy + mask + denoise (R steps 18-22)

### 6.1 Scope and order

R's `inferCNV_ops.R` runs steps 18-22 in a fixed sequence
(`inferCNV_ops.R:1362-1615`):

```
step 18  inferCNVBayesNet (Gibbs over CNV regions)
step 19  filterHighPNormals (overwrite high-P(normal) regions to neutral)
step 20  assign_HMM_states_to_proxy_expr_vals (state → CN-ratio remap)
step 21  mask_non_DE_genes_basic (Wilcoxon vs reference, BH-adjusted)
step 22  clear_noise_via_ref_mean_sd (ref-band flatten)
```

The order is load-bearing: step 19's filtered HMM state matrix is
what step 20 consumes; step 21 and step 22 then operate on the
post-step-20 CN-ratio matrix in turn. The Python orchestrator at
`pyinfercnv/pipeline_phase3.py::run_phase3` (lines 164-301) preserves
this order exactly. See the docstring at lines 16-41 for the per-step
R citation.

### 6.2 BayesNet Gibbs port (`bayesnet/gibbs.py`)

R's BayesNet is `rjags::jags.model` + `rjags::coda.samples` driving
the BUGS Mixture Model at `inst/BUGS_Mixture_Model` (i6) /
`BUGS_Mixture_Model_i3` (i3). The model is a per-region Dirichlet
mixture: `gexp[i,j] ~ dnorm(mu.1[j], tau.1[j])` with the state
mixture chosen by `epsilon[j] ~ dcat(theta[])` and
`theta ~ ddirich(1..1)`.

Why pure NumPy + numba JIT instead of `pyjags` / `pymc`:

- `pyjags` requires JAGS to be installed natively (defeats §1.1
  pure-Python wheel).
- `pymc` is heavy and overkill for this single closed-form mixture.
- Numba `@njit(cache=True, parallel=True)` over regions matches R's
  `nonParallel` / `withParallel` branches for free.

The implementation lives at `pyinfercnv/bayesnet/gibbs.py:74-276`.
Chain dimensions match R's defaults at the orchestrator level
(`pipeline_phase3.py` invokes `_step18_bayesnet` with `numBurnin=1000`,
`numSamples=1000`, `numChains=3` per `bayesnet/__init__.py:41-43`),
though R hardcodes `n.chains = 6` for i6 and `3` for i3
(`inferCNV_BayesNet.R:1098`); the Python default of `3` for both was
chosen to keep the oligo-fixture wall time inside a 60-s budget per
the design rationale at `gibbs.py:38-43`.

The parity contract is **not** posterior trace bit-equivalence (R's
JAGS MT19937 / L'Ecuyer streams cannot be reproduced). It is
distributional, expressed as two tiers in
`tests/test_r_parity.py::test_step18_bayesnet_parity` (line 827):

- **stretch tier**: per-region `max|ΔP| < 0.05` (line 969).
- **soft tier**: per-region `max|ΔP| < 0.10` (line 970).

The asserted floor is `soft_rate >= 0.90` (line 993) — at least 90%
of matched regions must satisfy `max|ΔP| < 0.10`. Observed at HEAD on
oligo: `soft-tier 95.5%, stretch-tier 88.6%` (test docstring lines
989-991).

### 6.3 The reassignCNVs fail-loud guard

R's `inferCNVBayesNet` has a default `reassignCNVs=TRUE`
(`inferCNV_BayesNet.R:1306`). After the `removeCNV` pass overwrites
high-P(normal) regions to neutral, the reassign branch
(`inferCNV_BayesNet.R:491-540`) overwrites *every* non-normal-majority
region to its argmax state. The Python port implements only the
`removeCNV` branch.

Because `InferCNVConfig.reassignCNVs` defaults to `True` for R
alignment (`pyinfercnv/config.py:70`), passing the kwarg through
`run_bayesnet_gibbs` would silently drop it (the function never
forwards it). To prevent that silent divergence, two fail-loud guards
fire at the orchestrator boundary:

1. `pyinfercnv/bayesnet/gibbs.py:186-190` — `run_bayesnet_gibbs`
   itself raises `NotImplementedError` if `reassignCNVs=True`.
2. `pyinfercnv/bayesnet/__init__.py:110-116` — `_step18_bayesnet`
   raises before it even reaches `run_bayesnet_gibbs`, with an
   explicit "Set reassignCNVs=False on InferCNVConfig" message.

Why guard at the orchestrator boundary instead of silently dropping
the kwarg: the user calling `infercnv(config=cfg)` with default
`cfg.reassignCNVs=True` would otherwise believe the R-default branch
ran when in fact only the removeCNV branch did. A fail-loud raise
forces the user to acknowledge the divergence. This pattern is the
template for all six fail-loud guards in §9.

### 6.4 step 20 proxy

Step 20 is a pure state→CN-ratio lookup. R's
`assign_HMM_states_to_proxy_expr_vals` at `inferCNV_HMM.R:1195-1200`
(i6) and `inferCNV_i3HMM.R:409-411` (i3) does sequential in-place
overwrites. The Python equivalent at
`pipeline_phase3._step20_assign_states_to_proxy_expr_vals` (lines
104-161) replaces the sequential overwrite with a vectorised
`state_map[states.astype(np.intp)]` lookup, which is equivalent
because the source state labels are integers and the destination CN
ratios do not collide with any source value.

The lookup tables (lines 96-101) are bit-faithful to R:

- i6: `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]` (R 1-based: state 1→0.0, …,
  state 6→3.0; Python 0-based reads identically).
- i3: `[0.5, 1.0, 1.5]` (R 1-based: state 1→0.5, state 2→1.0,
  state 3→1.5).

Both gates are at `tests/test_r_parity.py::test_step20_state_proxy_parity`
(line 1071, i6) and `test_step20_state_proxy_i3_parity` (line 1138,
i3); both assert `diff < 1e-10` and observe `max_diff = 0.000e+00`
(test print at lines 1134, 1186).

### 6.5 mask_non_DE (`mask_de/wilcoxon.py`)

R's `get_DE_genes_basic` (`inferCNV_mask_non_DE.R:158-259`) computes
a Mann-Whitney U test per (gene, tumor-subcluster vs ref-type) pair,
then BH-adjusts the per-subcluster p-values. The Python port at
`mask_de/wilcoxon.py:60-89` uses
`scipy.stats.mannwhitneyu(method="asymptotic", use_continuity=True)`
and `statsmodels.stats.multitest.multipletests(method="fdr_bh")`.

Two subtleties had to be addressed for bit-exactness:

1. R's default `wilcox.test(exact=NULL)` chooses exact when n<50 and
   no ties, normal-approximation+continuity otherwise. The Python
   side forces asymptotic (`method="asymptotic"`); the R-side
   `tests/r_reference.R` step 21 dump monkey-patches
   `infercnv:::get_DE_genes_basic` to pass `exact=FALSE` and to
   strip the `rnorm(mean=1e-4, sd=1e-4)` tie-breaking jitter that
   R applies. Without this pact, parity is unachievable because the
   jitter is RNG-stochastic — see the docstring at
   `mask_de/wilcoxon.py:16-33`.
2. R's `p.adjust(method="BH")` keeps `NaN` in output but ignores in
   ranking; `multipletests` raises on `NaN`. The Python wrapper at
   `_bh_adjust` (line 92-105) handles this explicitly: filter
   finite-only, fill `NaN` back into the unmasked positions.

The parity gate at `tests/test_r_parity.py::test_step21_mask_non_DE_parity`
(line 1190) makes two assertions:

- **bit-exact matrix**: `max_diff < 1e-10` on the masked CNV matrix
  (line 1286). Observed: `max_diff = 1.110e-16` at HEAD
  (`CHANGELOG.md ## 0.2.0` line 50).
- **soft mask floor**: per-position xor between the Python and R
  boolean masks must be < 1e-4 (line 1307). The xor is non-zero
  because R's `np.isclose` against the post-mask matrix can produce
  coincidental-value false positives (a few unmasked positions
  whose value happens to equal `center_val`). The matrix-level
  bit-exact gate is the load-bearing claim.

The Python port currently requires HMM (uses `result.subclusters`
from Phase 2) — see the fail-loud guard at
`pipeline.py:346-352`. R itself does **not** require HMM for
mask_non_DE (R reads
`observation_grouped_cell_indices`/`reference_grouped_cell_indices`
at `inferCNV_mask_non_DE.R:35`); the Python coupling to subclusters
is a port-side limitation, deferred to v0.3.

### 6.6 denoise (`denoise/ref_mean_sd.py`)

R's `clear_noise_via_ref_mean_sd` (`inferCNV_ops.R:2302-2346`)
algorithm in five steps:

1. Pick reference cells; fall back to all observation cells when no
   reference is set.
2. `mean_ref_vals = mean(vals)` (scalar over the reference matrix).
3. `mean_ref_sd = mean(apply(vals, 2, sd, na.rm=TRUE)) * sd_amplifier`
   — mean of per-cell sd's, scaled.
4. Define band `[mean - mean_sd, mean + mean_sd]`.
5. In-place flatten: `vals[vals > lower & vals < upper] = mean_ref_vals`.

The Python port at `denoise/ref_mean_sd.py:53-...` matches this
exactly. Two pitfalls that bit early prototypes and were fixed:

- `np.std` defaults to `ddof=0`; R `sd` uses `ddof=1` (Bessel
  correction). The Python code passes `ddof=1` explicitly.
- The band uses **strict** `>` / `<` (R `inferCNV_ops.R:2275`,
  `2335`); values exactly on the band boundary are not flattened.

Parity gate at `tests/test_r_parity.py::test_step22_noise_reduction_parity`
(line 1312): `assert diff < 1e-10` on the denoised matrix. Observed
at HEAD: `max_diff = 4.441e-16` (recorded in commit `ddd3c32`'s
Agent-B3 section).

The `noise_logistic=True` branch (R sigmoidal soft mask via
`depress_log_signal_midpt_val` at `inferCNV_heatmap.R:2783`) raises
`NotImplementedError` at two layers: in the leaf function
(`denoise/ref_mean_sd.py:97-99`) and again at the top-level
`pipeline.py:328-333` orchestrator guard.

### 6.7 The orchestrator wire-fix (commit `febc9db`)

Phase 3 was implemented as three module-level deliverables in
`ddd3c32` ("Phase 3 实施 — BayesNet / mask_non_DE / noise_reduction
三模块落地", 2026-04-24) — but at that commit the top-level
`pipeline.infercnv()` did not yet invoke `run_phase3`. Setting
`HMM=True` ran Phase 2 only; setting `denoise=True` did nothing.

The wire-fix landed in **commit `febc9db`** ("Phase 3 wire —
orchestrator + fail-loud guards + R-parity step 20 + post-codex-review
fixes", 2026-04-25). It addressed three "P0/P1/P2" classes of finding
from the Codex meta-review of the Phase 3 plan:

- **P0** (`pipeline_phase3.py:237-252`, `pipeline_phase3.py:267-278`):
  the `bayes_posterior` persistence key was originally `theta_mean`;
  the canonical key returned by
  `bayesnet.gibbs.run_bayesnet_gibbs` is `cnv_posterior` (per
  `gibbs.py:269-276` and the R source `inferCNV_BayesNet.R::cnv_prob`
  at L1137-1141). The fix wraps the read in `try/except KeyError`
  with a diagnostic message so any future contract drift surfaces
  instead of silently leaving `bayes_posterior = None`.
- **P1** (`pipeline.py:299-360`): the `phase3_on` gate
  (`bool(cfg.HMM) | BayesMaxPNormal>0 | mask_nonDE_genes | denoise`)
  was added to `pipeline.infercnv` and three fail-loud guards were
  installed: `denoise + noise_logistic`, HMM=False+Bayes,
  HMM=False+mask. The mask guard is intentionally Python-side-only
  (R does not require HMM for mask) — see the comment at
  `pipeline.py:342-352`.
- **P2** (docs + result.py): `bayes_posterior` shape doc fixed from
  `(K, n_regions)` to `(n_regions, K)`; new `tests/test_r_parity.py::test_step20_state_proxy_i3_parity`
  added (line 1138).

The same commit added the permanent regression test at
`tests/integration/test_top_level_phase3.py` (159 lines). Three
tests there lock the wire:

- `test_top_level_invokes_phase3_when_hmm_and_denoise_on` (line 68)
  — verifies that `denoised_matrix` and `hmm_proxy_matrix` are
  populated when `HMM=True, denoise=True`.
- `test_top_level_skips_phase3_when_all_off` (line 113) — verifies
  the no-op path leaves all four Phase 3 fields `None`.
- `test_top_level_bayes_requires_hmm_raises` (line 142) — verifies
  the fail-loud guard fires when `BayesMaxPNormal=0.5` but
  `HMM=False`.

Without this integration test, a future commit could delete the
`run_phase3` invocation from `pipeline.py` and the rest of the suite
would stay green (`run_phase3` is exercised directly elsewhere).

### 6.8 R-parity gates (Phase 3)

| Gate | Test floor (line) | Observed at HEAD |
|---|---|---|
| step 18 BayesNet | soft-tier `max|ΔP|<0.10` ≥ 90% (line 993) | 95.5% (oligo) |
| step 19 filterHighPNormals | Jaccard ≥ 0.94 (line 1066) | 0.979 (oligo) |
| step 20 i6 proxy | `< 1e-10` (line 1135) | `max_diff = 0.000e+00` |
| step 20 i3 proxy | `< 1e-10` (line 1187) | `max_diff = 0.000e+00` |
| step 21 mask matrix | `< 1e-10` (line 1286) | `max_diff = 1.110e-16` |
| step 21 mask xor | `< 1e-4` (line 1307) | observed below floor |
| step 22 denoise | `< 1e-10` (line 1404) | `max_diff = 4.441e-16` |

Citations: `CHANGELOG.md ## 0.2.0` lines 47-51; line numbers above
are in `tests/test_r_parity.py`.

### 6.9 Known limitations / 0.3 backlog from Phase 3

- `BayesNet reassignCNVs=True` (R default; see §6.3).
- `BayesNet postMcmcMethod='removeCells'` — `gibbs.py:181-185`
  raises `NotImplementedError`.
- `denoise noise_logistic=True` (sigmoidal mask, see §6.6).
- `denoise apply_median_filtering` — R alt denoise path not
  exposed in `InferCNVConfig`.
- `mask_nonDE_genes=True` without HMM — Python-side coupling to
  `result.subclusters` (see §6.5).
- Phase 3b VB approximation — `bayesnet/vb` not implemented;
  full-quality Gibbs is the primary path
  (`NAMESPACE_PARITY.md` line 71).

## 7. Test architecture

### 7.1 `tests/test_r_parity.py` — the parity gate

The single load-bearing R-parity test file. 19 step-level tests; one
per numbered R step where R writes an intermediate fixture. Each
test:

1. Skips with `@pytest.mark.skipif(not _r_step_available("stepNN"))`
   when the matching `tests/r_out/stepNN.tsv` is missing — letting
   the unit-test suite run on machines without R installed.
2. Loads the R intermediate via `_load_r_step` (line 53), which
   returns `(matrix as float64, gene_ids, cell_ids)` in R's genes ×
   cells layout.
3. Runs the equivalent Python step on the same input and transposes
   to compare.
4. Asserts a tier-matched floor: `< 1e-10` for bit-exact,
   `< 1e-6`/`< 1e-3` for approximate steps, Jaccard / Spearman for
   stochastic steps.

Test count by phase (verified by `grep "^def test_"`):

- Phase 1: 9 tests (steps 02, 03, 04, 08, 09, 10, 11, 14, 16) plus
  the step14 Spearman ρ test = 10.
- Phase 2: 3 tests (step15 ARI floor, step17 i6 Jaccard, step17 i3
  Jaccard).
- Phase 3: 6 tests (step18, step19, step20×2, step21, step22).

Total: 19 in `test_r_parity.py`. Plus the integration test (3 in
`test_top_level_phase3.py`) and the orchestrator unit test
(1 in `test_pipeline_phase3_orchestrator.py`).

The Tier label is encoded in the assertion floor: `< 1e-10` is
bit-exact, `< 1e-6` is approximate, Jaccard/Spearman floors are
empirical. See the `_compute_jaccard_floor` helper at line 456 for
the per-cell Jaccard methodology.

### 7.2 `tests/r_reference.R` — the fixture generator

This script invokes R `infercnv::run` (or its internal helpers) and
dumps the post-step matrix at every numbered step into
`tests/r_out/stepNN_*.tsv` as a tab-separated, gene-indexed table.
Phase 3 dumps were added in commit `ddd3c32`; the script source is
not in the present working tree's history (it was edited inside the
filter-repo'd commits) but the produced TSVs in `tests/r_out/` are
gitignored anyway.

Notable R-side patches inside `r_reference.R`:

- The `monkey-patch infercnv:::get_DE_genes_basic` for step 21
  (strips `rnorm` jitter, forces `wilcox.test(exact=FALSE)`) — see
  §6.5.
- Explicit `reassignCNVs=FALSE` at the run() call to align with the
  Python port's removeCNV-only branch — added in `febc9db`'s P0 fix.
- `set.seed(42)` for reproducibility — referenced in
  `config.py:128-129` as the source of the Python `random_state=42`
  default.

### 7.3 `tests/integration/test_top_level_phase3.py`

The permanent regression for the Phase 3 wire (added in `febc9db`).
Three tests, all monkeypatching `_calibrate_hmm_emission` to a
deterministic fake on a 30-cell × 60-gene synthetic AnnData so the
test runs in seconds. See §6.7 for content.

### 7.4 `tests/unit/`

~28 per-module fast tests. Cover every leaf function and one
orchestrator-level test
(`tests/unit/test_pipeline_phase3_orchestrator.py`, added in `febc9db`)
that verifies `run_phase3` correctly persists `cnv_posterior` and
uses the post-step-19 filtered states for step 20.

### 7.5 Pytest count for v0.2.0

`CHANGELOG.md ## 0.2.0` line 71: **220 passed** (excluding
`tests/test_regression.py` which is gitignored / opt-in). Composition
per the same CHANGELOG line: +5 vs the 215 baseline at v0.2.0.dev2:

- +1 P0.2 orchestrator unit (`test_pipeline_phase3_orchestrator.py`).
- +1 P2.2 step20 i3 parity (`test_step20_state_proxy_i3_parity`).
- +3 P1.1 top-level integration (the three tests in
  `test_top_level_phase3.py`).

The R-parity gate runs 19/19 (`CHANGELOG.md ## 0.2.0` line 74).

## 8. Build / release process

### 8.1 Build backend

`pyproject.toml` lines 1-3 declare hatchling as the build backend.
The wheel target is `py3-none-any` because there are no compiled
extensions. Source: `dist/pyinfercnv-0.2.0-py3-none-any.whl` and
`dist/pyinfercnv-0.2.0.tar.gz` (built by `python -m build`).

### 8.2 Pure-Python wheel verification

The wheel invariant is mechanically checked:

```
unzip -l dist/pyinfercnv-0.2.0-py3-none-any.whl | grep -E "\.so|\.pyd|\.dylib"
```

must return empty. This check is part of the release procedure
documented in `CHANGELOG.md ## 0.2.0.dev2` line 213-215 ("Pure-Python
wheel invariant preserved: `python -m build` emits `py3-none-any.whl`
with zero `.so`/`.pyd`/`.dylib`. Twine check passes on both wheel and
sdist.") and gated by `tests/test_wheel.py`.

### 8.3 `twine check`

Run on both wheel and sdist before tagging. Recorded as PASSED in
`CHANGELOG.md ## 0.2.0` line 73 and the v0.2.0 release commit body
(`ff508f3`).

### 8.4 The bump-version + tag flow used for v0.2.0

Verified from the v0.2.0 release commit (`ff508f3`, "0.2.0 release —
first stable cut"):

1. `pyinfercnv/__init__.py::__version__` bumped from `"0.2.0.dev2"`
   to `"0.2.0"`.
2. `pyproject.toml::version` bumped from `"0.2.0.dev2"` to `"0.2.0"`.
3. `CHANGELOG.md ## 0.2.0` written with the per-phase tier table
   (lines 3-97).
4. `README.md` Status line rewritten to reflect Phase 3 status
   (line 5).
5. `chore: sync uv.lock to 0.2.0` (commit `255fbaa`) follows.
6. Tags exist for `v0.2.0`, `v0.2.0-private`, `v0.2.0.dev0`,
   `v0.2.0.dev1`, `v0.2.0.dev2` (verified by `git tag`). The dev tags
   are immutable; the release tag is `v0.2.0`.

### 8.5 Codex collaboration trail

The CLAUDE.md governance contract requires Codex MCP review at three
checkpoints (need-analysis, pre-coding prototype, post-coding review).
This produced a real workflow during Phase 2 and Phase 3 development.
Surviving evidence in commit messages and CHANGELOG:

- `7c9eb8c` — codex C1 critical fix (Leiden seed silent-ignore).
- `7ed3596` — codex C5 critical fix (Viterbi emission divergence).
- `4d5ffde`, `c7658f4`, `e38b722`, `4adaf4f`, `908968e`, `e864a09`
  — explicit "G3" / "G2" / "Track A/C" codex-review-and-adjudication
  commits for Phase 1 and Phase 2 closeout.
- `febc9db` — Phase 3 wire-fix described in commit body as "two
  rounds of plan-stage Codex meta-review (gpt-5.5, xhigh) plus two
  rounds of post-implementation review" (`CHANGELOG.md ## 0.2.0`
  lines 90-96).
- `3d9ec79` — strict R-parity hspike rewrite based on Codex D.1-D.7
  review (commit body refers to
  `docs/superpowers/reviews/track-hspike-rparity-codex.*.md`, which
  is referenced in commit message but absent from the working tree
  post-filter-repo — see §10).

The Codex collaboration is factually present in the build path; this
document does not overclaim it as the sole driver of correctness.
The R-parity gates in `tests/test_r_parity.py` are what actually
enforce correctness; Codex review surfaced the bugs before they
reached the gates.

## 9. Deferred to v0.3 — backlog with R sources

Six fail-loud guards are installed at the orchestrator boundary, plus
several non-default R branches that the pipeline does not reach with
the current `InferCNVConfig` defaults. Effort estimates: S = under 1
session, M = 1-3 sessions, L = >3 sessions including R-parity audit.

| # | R source | Py fail-loud entry | Effort | Notes |
|---|---|---|---|---|
| 1 | `inferCNV_BayesNet.R:491-540` (`reassignCNV` branch) | `bayesnet/__init__.py:110-116` (`_step18_bayesnet`) and `bayesnet/gibbs.py:186-190` (`run_bayesnet_gibbs`) | M | R default `reassignCNVs=TRUE`. Python ports the `removeCNV` branch only. To unblock, port the post-removeCNV reassignment loop (overwrite each non-normal-majority region to its argmax state). |
| 2 | `inferCNV_heatmap.R:2783` (`depress_log_signal_midpt_val` sigmoidal mask) | `pyinfercnv/pipeline.py:328-333` and `denoise/ref_mean_sd.py:97-99` | S | `denoise + noise_logistic=TRUE`. Sigmoidal soft mask path. |
| 3 | `inferCNV_BayesNet.R::removeCells` branch | `bayesnet/gibbs.py:181-185` | M | `postMcmcMethod='removeCells'`. R variant; not the default. |
| 4 | `inferCNV_hidden_spike.R` (`sim_method='simple'` and `'splatter'`) | `pipeline_phase2.py:554` (`sim_method="meanvar"` hardcoded) | M | R alt simulation paths for hspike; only `meanvar` is ported. |
| 5 | `inferCNV_ops.R::apply_max_threshold_bounds` (`threshold='auto'`) | `pyinfercnv/preprocess/max_threshold.py` (auto branch raises) | S | `cfg.max_centered_threshold='auto'`. R auto-percentile path. |
| 6 | `inferCNV_ops.R` `apply_median_filtering` denoise alt | not yet exposed in `InferCNVConfig` | S | R alt denoise path; expose + port. |
| 7 | `inferCNV_i3HMM.R::get_HoneyBADGER_setGexpDev` (`use_KS=TRUE`) | not exposed; `hmm/i3.py:67-122` Z-form only | M | KS variant uses `rnorm` sampling on R side; not bit-reproducible across RNGs. |
| 8 | `inferCNV_ops.R` step 13 (remove genes at chromosome ends) | not present in `pipeline.py` | S | Currently goes step 12 → step 14 directly. |
| 9 | `inferCNV_ops.R` `analysis_mode='cells'` / `'samples'` | `pyinfercnv/config.py:158-164` (`validate()` raises) | L | Phase 3 work; touches the entire region-aggregation contract. |
| 10 | `inferCNV_ops.R` `per_chr_hmm_subclusters=TRUE` | not exposed in `InferCNVConfig` | M | Per-chromosome subclustering variant. |
| 11 | `inferCNV_BayesNet.R` VB approximation of Gibbs | `bayesnet/vb` not implemented; `NAMESPACE_PARITY.md` line 71 | L | Phase 3b. Performance optimisation; full-quality Gibbs is the primary path. |
| 12 | `inferCNV_mask_non_DE.R:35` (mask without HMM) | `pyinfercnv/pipeline.py:346-352` Python-side coupling | M | R does not require HMM; Python `_step21_mask_non_DE` consumes `result.subclusters`. Decouple. |
| 13 | `mask_de` `test_use='perm'` | `mask_de/wilcoxon.py` only supports `wilcoxon` and `t` | M | R uses the `coin` package; out-of-scope for Python without an R-equivalent permutation test. |

The i6 hspike μ/σ baseline drift (Jaccard 0.918 → 0.979 between
`3d9ec79` and HEAD) is **not** a fail-loud item — there is no
regression at HEAD. It is tracked in `CHANGELOG.md ## 0.2.0` line 88
as a known fluctuation source with only ~2 percentage points of
headroom against the 0.96 floor.

## 10. Document provenance

This document was rebuilt on **2026-04-26** from the surviving
artifacts after a `git filter-repo` accident lost the original Phase
1 / 2 / 3 planning history (the `2026-04-24-phase3-start.md`
plan, the per-phase Codex review files, the `CODEX_HANDOFF.md`
chain, the Track A diagnostic notes). The substantive content is
reconstructed from:

- R source under `/media/jason/T7/rerbulid/infercnv/infercnv-master/R/`
- Python source under `/media/jason/T7/rerbulid/pyinfercnv/pyinfercnv/`
- `tests/test_r_parity.py`, `tests/integration/test_top_level_phase3.py`,
  `tests/unit/`
- `CHANGELOG.md ## 0.2.0`, `README.md`, `NAMESPACE_PARITY.md`
- `git log` of the present working-tree branch

### 10.1 Caveat: phantom commit hashes in CHANGELOG

`CHANGELOG.md ## 0.2.0` and a few other surviving docs cite short
SHAs that do not resolve in the current `git log`. Verified remap
(via `git log --oneline` + commit-body content match):

| Phantom SHA | Resolves to | Commit subject |
|---|---|---|
| `8ed9318` | `a8d4526` | Phase 2 i3 R-parity — HMM switches to linear FC (2026-04-23) |
| `384e983` | `3d9ec79` | Phase 2 i6 hspike — strict R-parity rewrite (2026-04-23) |
| `9f213cf` | `ca8bda8` | Phase 1 bit-exact — float64 intermediates (2026-04-23) |
| step10 → bit-exact | `edddb51` | Phase 1 bit-exact 收尾 — step10 R-exact (2026-04-24) — corrects `CHANGELOG.md ## 0.2.0` line 22 mis-attribution to `9f213cf` |
| `4abc95b` | `febc9db` | Phase 3 wire — orchestrator + fail-loud guards (2026-04-25) |
| `90eabe1` | `5ff778d` | fix(pipeline): filter genes on all cells (2026-04-22) |
| `e32adcf` | `7ed3596` | feat(hmm/kernel): R Viterbi.dthmm.adj emission parity (2026-04-21) |
| `4b7206d` | `7c9eb8c` (likely) | fix(subcluster/leiden): honour random_state (2026-04-21) — content match, not bit-verified |
| `4aacc01` | `1f408a8` (likely) | refactor(preprocess): remove reference_cell_idx kwarg (2026-04-22) — content match, not bit-verified |
| `f468e2d` | inside `ddd3c32` (likely) | Phase 3 land commit; soft-floor xor commentary at `test_r_parity.py:1304-1308` — not separately isolatable |

This document uses the verified post-filter-repo SHAs. Hedged
attributions (`7c9eb8c`, `1f408a8`, in-`ddd3c32`) are flagged at
their use sites in §3.6, §6.5, and here.

### 10.2 Caveat: lost cross-references

Several surviving docs reference files that are absent from the
working tree post-filter-repo: `CODEX_HANDOFF.md`,
`docs/superpowers/findings/2026-04-25-phase3-wire-fix-summary.md`,
`docs/superpowers/reviews/` (incl. `track-hspike-rparity-codex.*.md`),
`docs/superpowers/plans/2026-04-24-phase3-start.md`,
`scripts/track_a1/{cnv_matrix_spearman,spearman_benchmark}.py`, and
`benchmarks/phase2/Gao2021_Breast/`. They are cited verbatim where
they appear in surviving CHANGELOG / commit bodies — those quotes are
factual records of the v0.2.0 release docs — but should not be
followed as live links.

### 10.3 How to verify any specific number

Every parity number, max_diff, and Jaccard floor in this document
comes from one of three places:

1. The matching `assert` in `tests/test_r_parity.py` (line numbers
   listed throughout §4.4, §5.7, §6.8).
2. The `CHANGELOG.md ## 0.2.0` table (lines 15-25 for Phase 1, lines
   34-41 for Phase 2, lines 45-52 for Phase 3).
3. The cited commit body (verified by `git show <sha>`).

When in doubt, the test assertion is authoritative — it is what is
actually enforced in CI. Numbers in CHANGELOG and commit bodies are
"observed at the time of writing" and may drift slightly across
later commits; the i6 hspike 0.918 ↔ 0.979 fluctuation in §5.8 is
the canonical example.
