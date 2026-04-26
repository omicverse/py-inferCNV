# Algorithm Correspondence: R `infercnv` vs `pyinfercnv`

Authoritative source-level mapping from the Broad Institute's R `infercnv`
package (`infercnv-master/R/`) to the pure-Python `pyinfercnv` re-implementation
(`pyinfercnv/`). Organised by R `step_count` (the integer counter incremented
by `inferCNV_ops.R::run`, lines 532-1615).

This document is the source-of-record for "which R function does what in
Python", per-step. For project-level R-parity tier numbers see
`CHANGELOG.md ## 0.2.0`; for the high-level module map see
`NAMESPACE_PARITY.md`. For executable parity proofs see
`tests/test_r_parity.py` (one test per numbered step where R writes an
intermediate fixture).

## Conventions

- `Tier` follows `CHANGELOG.md ## 0.2.0`:
  - **bit-exact** — `max_diff < 1e-10` against R intermediate fixture.
  - **approximate** — verified by Jaccard / Spearman / ARI floor; no
    bit-exactness expected (e.g. stochastic samplers, KNN tie-breaking).
  - **fail-loud guard** — Python orchestrator raises
    `NotImplementedError` (or `ValueError`) on R-default branches that
    were not ported. Listed in CHANGELOG "Known limitations".
  - **not ported** — R function exists, Python does not call it; usually
    because the gating R kwarg is left at a non-default value.
  - **out of scope** — R behaviour explicitly excluded from the Python
    port (e.g. JAGS-specific RNG bit-equivalence, viz-only dispatch).
- R citations use `inferCNV_ops.R:NNNN` etc., pointing at the actual file
  and line range in `/media/jason/T7/rerbulid/infercnv/infercnv-master/R/`.
- Python references point at `pyinfercnv/...` paths and named functions
  in the installed package.
- Steps that R *increments the counter for* but *gates with an `if`* (so
  the function body executes only under non-default kwargs, or only when
  HMM is enabled) are marked accordingly. The Python orchestrator
  preserves the R numbering of these gated steps even when the body is a
  no-op under default settings.
- "Phase 1" = R steps 2-14 (+16); "Phase 2" = R steps 15, 17, plus the
  hspike calibration sub-step that R folds into step 3; "Phase 3" =
  R steps 18-22.

## Phase / orchestrator entry points

| Phase | R entry | Python entry |
|---|---|---|
| Phase 1 + Phase 2 + Phase 3 | `infercnv::run` (`inferCNV_ops.R:242-1652`) | `pyinfercnv.pipeline.infercnv` (`pyinfercnv/pipeline.py:120-359`) |
| Phase 2 (subcluster + HMM) | inline in `run` (`inferCNV_ops.R:1066-1351`) | `pyinfercnv.pipeline_phase2.run_phase2` (`pyinfercnv/pipeline_phase2.py:153-401`) |
| Phase 3 (BayesNet / mask / denoise) | inline in `run` (`inferCNV_ops.R:1362-1615`) | `pyinfercnv.pipeline_phase3.run_phase3` (`pyinfercnv/pipeline_phase3.py:164-301`) |

The top-level Python `infercnv()` invokes Phase 2 internally when
`cfg.HMM=True` (`pipeline.py:300-306`) and Phase 3 when any of
`cfg.HMM=True`, `BayesMaxPNormal>0`, `mask_nonDE_genes=True`, or
`denoise=True` (`pipeline.py:316-354`). This composition mirrors R
which runs everything in a single `run()` call.

---

## Step 1 — Incoming data (`inferCNV_ops.R:538-543`)

**R function**: implicit — the orchestrator simply logs `"STEP 1: incoming
data"` and (when `save_rds=TRUE`) snapshots the freshly-constructed
`infercnv_obj` to disk. The actual data load happens upstream in
`infercnv::CreateInfercnvObject` (`inferCNV.R`) before `run()` is called.

**Py entry point**: `pyinfercnv.io.h5ad.extract_counts` (called from
`pyinfercnv/pipeline.py:139`) — pulls the counts layer from `AnnData`
and returns CSR float32. The `_build_chromosome_layout` helper at
`pyinfercnv/pipeline.py:84-117` joins gene positions from `adata.var`
and reproduces R's per-chromosome ordering used by `chr_pos`.

**Tier**: not a numerical step — purely I/O. There is no R-parity test
since R writes only the wrapped object, not a tabular dump.

**Behavior**: Materialises the cells × genes counts matrix and
constructs the per-step bookkeeping (`reload_info`, `step_count = 0`).
Python additionally captures `ref_counts_raw` here (G1 patch P2 at
`pipeline.py:149-158`), the dense float32 reference-only counts kept
for Phase 2 hspike calibration before normalize destroys raw counts.

**R signature**: n/a (constructor lives outside `run`).

**Py signature**: `extract_counts(adata: AnnData, *, counts_layer: str | None = "counts") -> scipy.sparse.csr_matrix`.

**Key divergences**:
- Python keeps a side-band `ref_counts_raw` array for hspike replay; R
  re-reads `infercnv_obj@count.data` later because the raw counts are
  preserved on the S4 slot. Python's CPM normalize at step 3 overwrites
  the only counts handle, so the snapshot is required.

**Tests**: none (no R fixture).

---

## Step 2 — Removing lowly expressed genes (`inferCNV_ops.R:551-569`)

**R functions**:
- `require_above_min_mean_expr_cutoff(infercnv_obj, cutoff)`
  (`inferCNV_ops.R:2128-2152`) — drops genes whose mean expression
  across the *full* matrix is below `cutoff`.
- `require_above_min_cells_ref(infercnv_obj, min_cells_per_gene=…)`
  (`inferCNV_ops.R:2182-2210`) — additional filter requiring each gene
  to be observed in at least `min_cells_per_gene` reference cells.

**Py entry point**: `pyinfercnv.preprocess.filter_low_expression_genes`
(`pyinfercnv/preprocess/filter_genes.py:21-51`). Called from
`pyinfercnv/pipeline.py:168-176`.

**Tier**: bit-exact (`max_diff` exact — same set of kept genes; see
`tests/test_r_parity.py::test_step02_filter_genes_parity` line 116).

**Behavior**: Computes per-gene mean over all cells and per-gene
non-zero count, applies `(mean >= cutoff) & (nnz >= min_cells_per_gene)`
mask. Both predicates run on the full cell population (Python
explicitly mirrors R's `require_above_min_mean_expr_cutoff` semantics
in the file docstring at `filter_genes.py:6-13`).

**R signature**: `require_above_min_mean_expr_cutoff(infercnv_obj, min_mean_expr_cutoff)` and `require_above_min_cells_ref(infercnv_obj, min_cells_per_gene)`.

**Py signature**: `filter_low_expression_genes(X, *, cutoff: float = 1.0, min_cells_per_gene: int = 3) -> np.ndarray` (returns boolean mask).

**Key divergences**:
- Python collapses R's two-stage AND filter into a single per-gene
  predicate computed on all cells. On the standard oligodendroglioma
  fixture (default cutoff=1.0, min_cells=3) the two-stage R filter and
  the all-cells one-stage filter return the *same* 8508-gene survivor
  set; the Python comment block at `filter_genes.py:7-10` cites the
  triage script that established this.
- Earlier (pre-`5ff778d`) Python used `reference_cell_idx=ref_idx_all`
  which collapsed to ref-only and dropped ~1559 extra genes; the breaking
  change to remove that kwarg is in `CHANGELOG.md ## 0.2.0.dev0
  Compatibility`.

**Tests**: `tests/test_r_parity.py::test_step02_filter_genes_parity`
(line 116) — checks Python's surviving gene set is bit-identical to
R's step02 fixture.

---

## Step 3 — Normalize by sequencing depth (`inferCNV_ops.R:579-600`)

**R function**: `normalize_counts_by_seq_depth(infercnv_obj, normalize_factor=NA)`
(`inferCNV_ops.R:3064-3080`), delegating to
`.normalize_data_matrix_by_seq_depth(counts.matrix, normalize_factor=NA)`
(`inferCNV_ops.R:3082-3110`). When `normalize_factor` is `NA`, R uses
`median(colSums)` of the post-filter matrix.

**Py entry point**: `pyinfercnv.preprocess.normalize_by_seq_depth`
(`pyinfercnv/preprocess/normalize.py:19-47`). Called from
`pipeline.py:180`.

**Tier**: bit-exact (`max_diff < 1e-10`; `CHANGELOG.md ## 0.2.0` Phase 1
table; `tests/test_r_parity.py::test_step03_normalize_parity` line 149).

**Behavior**: Each cell row is rescaled by
`normalize_factor / row_sum`; cells with zero library size are left at
zero (no division by zero). When `normalize_factor` is omitted, the
median of the row-sums is used. Sparse-input path returns a sparse
float64 result; dense input returns dense float64. The downcast to
float32 for the public `result.cnv_matrix` is performed only at result
assembly in `pipeline.py:288-291` (Phase 1 bit-exact path landed in
`0.2.0.dev2`, see `CHANGELOG.md`).

**R signature**: `normalize_counts_by_seq_depth(infercnv_obj, normalize_factor=NA)`.

**Py signature**: `normalize_by_seq_depth(X, *, normalize_factor: float | None = None)`.

**Key divergences**:
- R computes `colSums` on its genes × cells layout; Python computes
  `sum(axis=1)` on cells × genes — same numbers, transposed convention.
- All intermediate arithmetic in float64; bit-exact tightening was
  achieved in `0.2.0.dev2` (`CHANGELOG.md`). Earlier float32 code was
  `< 1e-2`.

### Step 3.1 (R-only sub-step) — `.build_and_add_hspike` (`inferCNV_ops.R:586-595`)

When `HMM=TRUE && HMM_type=='i6'`, R additionally invokes
`.build_and_add_hspike(infercnv_obj, sim_method, aggregate_normals)`
(`inferCNV_hidden_spike.R:3-167`) immediately after CPM normalize. This
constructs a synthetic CNV-spiked matrix used later by step 17 i6 to
fit per-state emission distributions.

**Py entry point**: `pyinfercnv.hmm.hspike.calibrate_i6_emission`
(`pyinfercnv/hmm/hspike.py:298`). The Python pipeline does not run
hspike inline at step 3; it is deferred until Phase 2 needs it
(`pipeline_phase2.py:312-323`, profile key `16_hspike_calibrate`).
The deferred placement is supported by Phase 1 capturing
`cpm_matrix_f32` (`result.py:50-58`) — a snapshot of `infercnv_obj@expr.data`
at exactly the moment R calls `.build_and_add_hspike`.

**Tier**: structural rules D.1–D.7 per CHANGELOG `## 0.2.0` Phase 2 row.
Strict R-parity rewrite landed in commit `3d9ec79`.

**Sim method support**:
- `sim_method='meanvar'` (R default) — Python implements via
  `_fit_meanvar_spline` (`hspike.py:790`) +
  `_simulate_meanvar_counts` (`hspike.py:855`).
- `sim_method='simple'` (R alt) — **fail-loud guard**;
  see `CHANGELOG.md ## 0.2.0 Known limitations`.
- `sim_method='splatter'` (R alt; uses Splatter scrape) —
  **fail-loud guard**.

The Python orchestrator pins to `sim_method='meanvar'` at
`pipeline_phase2.py:553-557`; alternative sim methods raise
`NotImplementedError` from `calibrate_i6_emission`.

**Tests**: hspike calibration is exercised end-to-end via the i6
parity floor `tests/test_r_parity.py::test_step17_hmm_i6_jaccard_floor`
(line 554). There is no direct hspike fixture test because R does not
write a tractable on-disk fixture for the spike-only object.

---

## Step 4 — log2(x+1) transform (`inferCNV_ops.R:610-639`)

**R function**: `log2xplus1(infercnv_obj)` defined within `inferCNV_ops.R`
(see helper at `inferCNV_ops.R:2756-2826` for `log2xplus1`,
`invert_log2xplus1`, `invert_log2`).

**Py entry point**: `pyinfercnv.preprocess.log_transform.log2_plus1`
(`pyinfercnv/preprocess/log_transform.py:13-19`). Called from
`pipeline.py:195`.

**Tier**: bit-exact (`max_diff < 1e-10`;
`tests/test_r_parity.py::test_step04_log2_plus1_parity` line 171).

**Behavior**: Element-wise `log2(x + 1)`. Sparse-preserving via
`np.log1p(data) / np.log(2.0)` on the `.data` array of a CSR copy.
Dense path uses the same formula on the dense array.

**R signature**: `log2xplus1(infercnv_obj)`.

**Py signature**: `log2_plus1(X)` (sparse or dense).

**Key divergences**: none algorithmic. Float64 throughout.

**Tests**: `test_step04_log2_plus1_parity`.

---

## Step 5 — Scale data (`inferCNV_ops.R:646-679`)

**R function**: `scale_infercnv_expr(infercnv_obj)` (`inferCNV_ops.R:3174-3191`),
gated on `scale_data=TRUE`.

**Py entry point**: not ported. The Python `InferCNVConfig.scale_data`
field defaults to `False` (`config.py:38`). The Python pipeline does
not call any scale routine; if a future config exposes
`scale_data=True`, it should fail-loud or call a not-yet-written
`pyinfercnv.preprocess.scale_infercnv_expr`.

**Tier**: not ported (default-off path).

**Behavior (R)**: Applies a Z-score across the full expression matrix
(`scale(t(expr.data))`).

**R signature**: `scale_infercnv_expr(infercnv_obj)`.

**Py signature**: n/a.

**Key divergences**: Python silently honours its own default-off path;
no fail-loud guard is wired because the Python config field exists
purely for API-surface parity (`config.py:38`).

**Tests**: none.

---

## Step 6 — Split reference data into groups (`inferCNV_ops.R:689-707`)

**R function**: `split_references(infercnv_obj, num_groups=num_ref_groups, hclust_method=hclust_method)`
(`inferCNV_ops.R:1917-1967`). Gated on `!is.null(num_ref_groups)`.

**Py entry point**: not ported. `InferCNVConfig.num_ref_groups` defaults
to `None` (`config.py:31`). No Python equivalent is wired into the
pipeline.

**Tier**: not ported (default-off path).

**Behavior (R)**: Hierarchically clusters reference cells into
`num_ref_groups` sub-references and stores the resulting
`reference_grouped_cell_indices` for later use in steps 8/12.

**Key divergences**: Python's `subtract_reference` consumes
`reference_groups` directly from the user-supplied
`reference_key`/`reference_cat` annotation (`pipeline.py:215-223`),
treating each annotation category as its own ref group. This matches
R behaviour when `num_ref_groups=NULL` and explicit
`ref_group_names` is supplied.

**Tests**: none.

---

## Step 7 — Tumor subclusters via random_trees (`inferCNV_ops.R:715-756`)

**R function**: `define_signif_tumor_subclusters_via_random_smooothed_trees(infercnv_obj, p_val, hclust_method, cluster_by_groups)`
(`inferCNV_tumor_subclusters.random_smoothed_trees.R`). Gated on
`analysis_mode == 'subclusters' & tumor_subcluster_partition_method == 'random_trees'`.

**Py entry point**: `pyinfercnv.subcluster.random_trees.random_tree_subcluster`
(`pyinfercnv/subcluster/random_trees.py:208`). The Python orchestrator
runs subclustering only at step 15 (`pipeline_phase2._run_subclustering`,
`pipeline_phase2.py:409-529`); when the user opts into
`tumor_subcluster_partition_method='random_trees'` the pre-step (step 7)
position is collapsed into the step 15 dispatch
(`pipeline_phase2.py:475-477`).

**Tier**: not ported as a *standalone step 7* — the random_trees backend
runs only at step 15.

**Behavior (R)**: Recursive hierarchical partitioning of the *log-space
smoothed and centered* matrix, using a permutation-based null
distribution on tree heights to set a significance cutoff
(`inferCNV_tumor_subclusters.R:181-303`).

**Key divergences**:
- R places this step *before* smoothing/centering when
  `random_trees`-only is selected; Python always runs subclustering
  after smoothing (step 15 position).

**Tests**: none for the step-7 position. The step-15 position is
exercised by `tests/test_r_parity.py::test_step15_subclusters_ari_floor`
(line 483) which uses the default `leiden` backend.

---

## Step 8 — Subtract average reference (pre-smoothing) (`inferCNV_ops.R:767-794`)

**R function**: `subtract_ref_expr_from_obs(infercnv_obj, inv_log=FALSE, use_bounds=ref_subtract_use_mean_bounds)`
(`inferCNV_ops.R:1678-1702`), delegating to:
- `.get_normal_gene_mean_bounds(expr.data, ref_groups, inv_log)`
  (`inferCNV_ops.R:1708-1740`)
- `.subtract_expr(expr_matrix, ref_grp_gene_means, use_bounds)`
  (`inferCNV_ops.R:1742-1786`)

**Py entry point**: `pyinfercnv.preprocess.subtract_reference`
(`pyinfercnv/preprocess/subtract_ref.py:39-77`). Called from
`pipeline.py:227`.

**Tier**: bit-exact (`max_diff < 1e-10`;
`tests/test_r_parity.py::test_step08_subtract_ref_parity` line 187).

**Behavior**: Compute per-group per-gene mean across reference cells.
When `use_bounds=True` and `K>1`, define
`bounds_min = min_k(mean_k), bounds_max = max_k(mean_k)` per gene;
values within `[bounds_min, bounds_max]` are zeroed, values above
`bounds_max` shift down by `bounds_max`, values below `bounds_min`
shift up by `bounds_min`. When `K==1` or `use_bounds=False`, subtract
the grand mean across all reference cells.

R applies subtract on *all* cells (including reference cells
themselves) per `inferCNV_ops.R:1742-1786`; reference cells become
close-to-zero post-subtraction, mirrored exactly in Python (the
docstring at `subtract_ref.py:14-17` notes this).

**R signature**: `subtract_ref_expr_from_obs(infercnv_obj, inv_log=FALSE, use_bounds=TRUE)`.

**Py signature**: `subtract_reference(X, *, ref_groups: Mapping[str, Sequence[int]], use_bounds: bool = True)`.

**Key divergences**:
- R's `inv_log` kwarg (mean computed in `2^x - 1` space then
  re-logged) is **not exposed** in the Python signature — Phase 1
  always passes `inv_log=FALSE` (R's call site at `inferCNV_ops.R:771`
  also uses `FALSE`), so Python's omission is faithful to the actual
  pipeline call.
- Python's `ref_groups` is keyed by string label; R uses positional
  integer indices keyed by `infercnv_obj@reference_grouped_cell_indices`.

**Tests**: `test_step08_subtract_ref_parity` (line 187).

---

## Step 9 — Apply max-centered expression threshold (`inferCNV_ops.R:801-844`)

**R function**: `apply_max_threshold_bounds(infercnv_obj, threshold)`
(`inferCNV_ops.R:2970-2998`). Gated on `!is.na(max_centered_threshold)`.
When `max_centered_threshold == "auto"`, R derives the threshold from
`mean(abs(get_average_bounds(infercnv_obj)))` (`inferCNV_ops.R:812-815`,
`get_average_bounds` at `inferCNV_ops.R:2723-2737`).

**Py entry point**: `pyinfercnv.preprocess.max_threshold.apply_max_centered_threshold`
(`pyinfercnv/preprocess/max_threshold.py:13-27`). Called from
`pipeline.py:232`.

**Tier**: bit-exact for numeric threshold (`max_diff < 1e-10`;
`tests/test_r_parity.py::test_step09_max_threshold_parity` line 217).
**Fail-loud guard** for `threshold='auto'` (`max_threshold.py:21`
raises `NotImplementedError`).

**Behavior**: Symmetric clip to `[-threshold, +threshold]`. Default
threshold is 3.0 (`config.py:35`).

**R signature**: `apply_max_threshold_bounds(infercnv_obj, threshold)`.

**Py signature**: `apply_max_centered_threshold(X: np.ndarray, *, threshold: float | str | None = 3.0) -> np.ndarray`.

**Key divergences**:
- `threshold='auto'` raises `NotImplementedError` in Python; R supports
  it. Listed in `CHANGELOG.md ## 0.2.0 Known limitations` as the
  `preprocess threshold='auto'` deferred branch.

**Tests**: `test_step09_max_threshold_parity`.

---

## Step 10 — Smoothing per chromosome (`inferCNV_ops.R:855-895`)

**R function**: dispatched on `smooth_method`:
- `smooth_by_chromosome(infercnv_obj, window_length, smooth_ends=TRUE)`
  (`inferCNV_ops.R:2406-2438`) — used when `smooth_method=='pyramidinal'`
  (R default). Calls `.smooth_window`/`.smooth_helper`/`.smooth_center_helper`
  (`inferCNV_ops.R:2440-2660`).
- `smooth_by_chromosome_runmeans(infercnv_obj, window_length)`
  (`inferCNV_ops.R:2679-2721`) — `smooth_method=='runmeans'`.
- `smooth_by_chromosome_coordinates(infercnv_obj, window_length)`
  (`inferCNV_ops.R:2534-2563`) — `smooth_method=='coordinates'`.

**Py entry point**: `pyinfercnv.smooth.pyramidinal.smooth_pyramidinal`
(`pyinfercnv/smooth/pyramidinal.py:26-55`). Called per chromosome
from `pipeline.py:236-246`. Hot path delegated to numba kernels
`pyinfercnv.kernels.smooth_center_numba.smooth_center_interior`
(`kernels/smooth_center_numba.py:30`) and
`pyinfercnv.kernels.smooth_tail_numba.smooth_tail_overwrite`
(`kernels/smooth_tail_numba.py:30`).

**Tier**: bit-exact `pyramidinal` (`max_diff < 1e-10` post commit
`ca8bda8`; `CHANGELOG.md ## 0.2.0`). The `0.2.0.dev2` table notes the
pre-fix scipy `uniform_filter1d` path was `< 1e-3`; the bit-exact path
uses an explicit triangular convolution instead.

**Other smooth methods**:
- `runmeans` — **not ported**. Python config validator allows
  `smooth_method='runmeans'` (`config.py:138-139`) but no module
  implements it; pipeline always calls `smooth_pyramidinal`.
- `coordinates` — **not ported** (same as runmeans).

**Behavior**: Triangular kernel of length `window_length` (default 101,
`config.py:27`). Interior positions use a single centered direct
convolution with pre-divided coefficients
`[1,2,...,tail,tail+1,tail,...,2,1] / (tail^2 + window_length)`
matching R's `stats::filter(vals, custom_filter, sides=2)`. Tail
positions use R's dynamic-denominator tail formula (kernel docstring
at `pyinfercnv/kernels/smooth_tail_numba.py`). Smoothing happens
*independently per chromosome* — `pipeline.py:239-246` iterates
chromosome blocks and calls the smoother on each `[start, end)` slice.

**R signature**: `smooth_by_chromosome(infercnv_obj, window_length, smooth_ends=TRUE)`.

**Py signature**: `smooth_pyramidinal(X: np.ndarray, *, window_length: int = 101) -> np.ndarray`.

**Key divergences**:
- R's `smooth_method='runmeans'` and `'coordinates'` not ported.
- Per-chromosome iteration is in the orchestrator, not the kernel
  (the kernel operates on a single 2-D slice).

**Tests**: `tests/test_r_parity.py::test_step10_smooth_pyramidinal_per_chromosome`
(line 336).

---

## Step 11 — Re-center cells across chromosomes after smoothing (`inferCNV_ops.R:907-937`)

**R function**: `center_cell_expr_across_chromosome(infercnv_obj, method="median")`
(`inferCNV_ops.R:2074-2092`), delegating to
`.center_columns(expr_data, method)` (`inferCNV_ops.R:2094-2109`).

**Py entry point**: `pyinfercnv.center.center_cells.center_cells`
(`pyinfercnv/center/center_cells.py:9-17`). Called from `pipeline.py:250`.

**Tier**: bit-exact (`max_diff < 1e-10`;
`tests/test_r_parity.py::test_step11_center_cells_parity` line 233).

**Behavior**: Per-cell median (or mean) subtracted from each row.
R uses `method='median'` by default at the Phase 1 step 11 call site
(`inferCNV_ops.R:911`). Python honours both `'median'` and `'mean'`.

**R signature**: `center_cell_expr_across_chromosome(infercnv_obj, method="mean")` (default), called at step 11 with `method="median"`.

**Py signature**: `center_cells(X: np.ndarray, *, method: str = "median") -> np.ndarray`.

**Key divergences**: R default kwarg is `"mean"` but step 11 overrides
to `"median"`; Python defaults to `"median"` because that is the only
value used in the actual pipeline call.

**Tests**: `test_step11_center_cells_parity`.

---

## Step 12 — Subtract average reference (post-smoothing) (`inferCNV_ops.R:948-975`)

**R function**: same as step 8 — `subtract_ref_expr_from_obs(infercnv_obj, inv_log=FALSE, use_bounds=ref_subtract_use_mean_bounds)`.

**Py entry point**: same — `pyinfercnv.preprocess.subtract_reference`,
called a second time at `pipeline.py:255`.

**Tier**: bit-exact (covered by the same `subtract_reference` parity
test at step 8; the post-smooth re-application uses identical code).

**Behavior**: Identical to step 8 but applied to the smoothed matrix.
R re-runs the same function; Python likewise calls
`subtract_reference` again with the same `ref_groups`/`use_bounds`.

**Key divergences**: none.

**Tests**: covered by `test_step08_subtract_ref_parity` semantically;
no separate step 12 fixture since it is a re-application of the same
function.

---

## Step 13 — Remove genes at chromosome ends (`inferCNV_ops.R:984-1015`)

**R function**: `remove_genes_at_ends_of_chromosomes(infercnv_obj, window_length)`
(`inferCNV_ops.R:3000-3062`). Gated on
`remove_genes_at_chr_ends == TRUE && smooth_method != 'coordinates'`.
R's default for `remove_genes_at_chr_ends` is `FALSE` (verified by
absence of an explicit override in the standard fixture run).

**Py entry point**: not ported. The Python pipeline does not include
this step; `InferCNVConfig` has no `remove_genes_at_chr_ends` field.

**Tier**: not ported (default-off path).

**Behavior (R)**: Drops a `tail = floor(window_length / 2)` band of
genes from each chromosome edge so the smoothed values do not contain
ramp-up artifacts.

**Key divergences**:
- Python keeps end-of-chromosome smoothed values (the tail-formula
  smoother at step 10 already overwrites those positions with
  R-faithful dynamic-denominator means).

**Tests**: none.

---

## Step 14 — Invert log2 (log2(FC) → FC) (`inferCNV_ops.R:1026-1055`)

**R function**: `invert_log2(infercnv_obj)` (`inferCNV_ops.R:2756-2826`
helpers). Applies `2^x` element-wise, moving the matrix from
log2-fold-change space into linear FC space.

**Py entry point**: `pyinfercnv.preprocess.log_transform.invert_log2`
(`pyinfercnv/preprocess/log_transform.py:31-35`). Called from
`pipeline.py:260`.

**Tier**: bit-exact (`max_diff < 1e-10`;
`tests/test_r_parity.py::test_step14_invert_log2_parity` line 250).

**Behavior**: `2 ** X`. Sparse input is densified first because the
result is dense (all entries become `>= 1` after exponentiating).

**R signature**: `invert_log2(infercnv_obj)`.

**Py signature**: `invert_log2(X)` returning float64 dense.

**Key divergences**: none algorithmic.

**Tests**: `test_step14_invert_log2_parity`. Additionally
`test_step14_cnv_matrix_spearman_oligo` (line 262) measures Spearman
ρ on the post-step14 matrix against R, the *primary parity metric*
(`CHANGELOG.md ## 0.2.0.dev2 Primary parity metric shift`).

---

## Step 15 — Tumor subclusters (Leiden / qnorm / random_trees) (`inferCNV_ops.R:1066-1150`)

**R function**: `define_signif_tumor_subclusters(infercnv_obj, p_val, k_nn, leiden_resolution, leiden_method, leiden_function, hclust_method, cluster_by_groups, partition_method, per_chr_hmm_subclusters, z_score_filter, …)`
(`inferCNV_tumor_subclusters.R:2-178`). Gated on
`analysis_mode == 'subclusters' & tumor_subcluster_partition_method != 'random_trees'`.
Internally dispatches to `.single_tumor_leiden_subclustering`
(`inferCNV_tumor_subclusters.R:569-644`) for Leiden.

**Py entry points** (orchestrator dispatch at
`pipeline_phase2._run_subclustering`, `pipeline_phase2.py:409-529`):
- `pyinfercnv.subcluster.leiden.leiden_subcluster`
  (`pyinfercnv/subcluster/leiden.py:194`) — default
  (`tumor_subcluster_partition_method='leiden'`).
- `pyinfercnv.subcluster.random_trees.random_tree_subcluster`
  (`pyinfercnv/subcluster/random_trees.py:208`).
- `pyinfercnv.subcluster.qnorm.qnorm_subcluster`
  (`pyinfercnv/subcluster/qnorm.py:31`).

**Tier**: approximate. ARI = 1.000 on oligodendroglioma (operational
floor 0.85; CHANGELOG `## 0.2.0` Phase 2 row, with the caveat that
ARI is retired as a *parity* metric). Spearman ρ on the post-Phase-1
CNV matrix is the primary metric: 1.0000 on DCIS1/TNBC1/TNBC3 10x UMI,
0.9998 on the oligo smart-seq2 fixture.

**Behavior**: Per `cluster_by_groups`:
- True (R parity, default in Python `config.py:98`): iterate over
  unique annotation labels in `reference_key`, run the partition
  backend independently within each group, assign globally unique
  subcluster ids. Reference cells receive real subcluster ids
  (no sentinel) so the HMM step processes them, matching R.
- False: pool all non-reference cells, run a single partition;
  reference cells get the sentinel `-1`.

For Leiden, the Python implementation builds a brute-force exact
KNN graph (sklearn `NearestNeighbors(algorithm="brute")`), then
runs igraph's C-core Leiden partitioner with CPM as the objective
(`leiden.py` end-to-end). `n_seeds > 1` (`config.py:104`) runs Leiden
N times and returns the highest-CPM partition (`rbest`, `0.2.0.dev2`).

**R signature**: `define_signif_tumor_subclusters(infercnv_obj, p_val=0.05, k_nn=20, leiden_resolution=0.05, …)`.

**Py signature**: `leiden_subcluster(X: np.ndarray, *, random_state: int = 42, k_nn: int = 20, resolution: float = 0.05, n_seeds: int = 1, min_subcluster_size: int | None = None, n_jobs: int = -1) -> np.ndarray[int32]`.

**Key divergences**:
- KNN backend: R uses `RANN::nn2` (KD-tree); Python uses
  `sklearn.NearestNeighbors(algorithm="brute")`. Tie-breaking on
  identical distances differs, which on kilocell tumor matrices
  produces different Leiden partitions even with bit-identical CNV
  matrices upstream. Documented at `CHANGELOG.md ## 0.2.0.dev0
  Parity` (10x UMI section).
- ARI is retired as a parity metric (CHANGELOG `## 0.2.0.dev2`).
- `min_subcluster_size` is a Python-only post-Leiden cleanup
  (`leiden.py:128 _collapse_small_clusters`); R has no equivalent.

**Tests**:
- `tests/test_r_parity.py::test_step15_subclusters_ari_floor` (line 483)
  with floor 0.85.

---

## Step 16 — Remove outliers (visualization) (`inferCNV_ops.R:1190-1228`)

**R function**: `remove_outliers_norm(infercnv_obj, out_method, lower_bound, upper_bound)`
(`inferCNV_ops.R:1969-1996`), delegating to
`.remove_outliers_norm(data, out_method, lower_bound, upper_bound)`
(`inferCNV_ops.R:1998-2072`). Helper
`get_average_bounds`/`.get_average_bounds`
(`inferCNV_ops.R:2723-2737`) computes
`lower = mean over cells of min(cell)` and
`upper = mean over cells of max(cell)`.

**Py entry point**: `pyinfercnv.cna.outlier_prune.prune_outliers`
(`pyinfercnv/cna/outlier_prune.py:30-51`). Called from
`pipeline.py:268-276` only when `cfg.prune_outliers=True`
(default `False`, `config.py:41`).

**Tier**: bit-exact (`max_diff < 1e-10` post `0.2.0.dev2`;
`tests/test_r_parity.py::test_step16_outlier_prune_parity` line 319).

**Behavior**: Asymmetric clip to
`[lower_bound, upper_bound]`. With `method='average_bound'` (default),
the bounds are derived from per-cell min/max means as above.

**R signature**: `remove_outliers_norm(infercnv_obj, out_method="average_bound", lower_bound=NA, upper_bound=NA)`.

**Py signature**: `prune_outliers(X: np.ndarray, *, method: str | None = "average_bound", lower_bound: float | None = None, upper_bound: float | None = None) -> np.ndarray`.

**Key divergences**:
- R applies in *log-space* (the matrix has not yet been inverted).
  Python applies in *linear-FC* space because step 14 invert is
  performed *before* step 16 in `pipeline.py:258-276`. The Python
  comment at `pipeline.py:263-267` explicitly notes the order
  matches R's `remove_outliers_norm` call site (which fires after
  invert in R as well — i.e., the invariant `cnv_fc == invert_log2(smoothed)`
  is preserved up to the outlier clip on the linear side).

**Tests**: `test_step16_outlier_prune_parity`.

---

## Step 17 — HMM-based CNV prediction (`inferCNV_ops.R:1235-1351`)

**R functions** (`inferCNV_HMM.R` for i6, `inferCNV_i3HMM.R` for i3):

i6 path:
- `predict_CNV_via_HMM_on_tumor_subclusters(infercnv_obj, t)`
  (`inferCNV_HMM.R:345-408`) — `analysis_mode='subclusters'`.
- `predict_CNV_via_HMM_on_tumor_subclusters_per_chr(infercnv_obj, subclusters_per_chr, t)`
  (`inferCNV_HMM.R:412-507`) — `per_chr_hmm_subclusters=TRUE`.
- `predict_CNV_via_HMM_on_indiv_cells(infercnv_obj, cnv_mean_sd, t)`
  (`inferCNV_HMM.R:284-343`) — `analysis_mode='cells'`.
- `predict_CNV_via_HMM_on_whole_tumor_samples(infercnv_obj, cluster_by_groups, t)`
  (`inferCNV_HMM.R:509-584`) — `analysis_mode='samples'`.
- `Viterbi.dthmm.adj` (`inferCNV_HMM.R:1101-1189`) — adjusted Viterbi
  decoder used by all i6 paths.
- `.get_HMM(cnv_mean_sd, t)` (`inferCNV_HMM.R:230-265`) — transition
  + initial distribution builder. Diagonal `1 - (K-1)*t`, off-diagonal
  `t`. Initial distribution `c(t, t, 1-5t, t, t, t)` (line 242).

i3 path (`inferCNV_i3HMM.R`):
- `i3HMM_predict_CNV_via_HMM_on_tumor_subclusters(infercnv_obj, i3_p_val, t, use_KS)`
  (search anchor `inferCNV_i3HMM.R:283`).
- `i3HMM_predict_CNV_via_HMM_on_indiv_cells(infercnv_obj, i3_p_val, t, use_KS)`
  (anchor at `inferCNV_i3HMM.R:206`).
- `i3HMM_predict_CNV_via_HMM_on_whole_tumor_samples(infercnv_obj, cluster_by_groups, i3_p_val, t, use_KS)`
  (anchor at `inferCNV_i3HMM.R:365`).
- `.i3HMM_get_sd_trend_by_num_cells_fit` (`inferCNV_i3HMM.R:17-80`).
- `.i3HMM_get_HMM` (`inferCNV_i3HMM.R:99-156`) — transition matrix
  builder; uses `1 - 5*t` on the diagonal even for K=3 (verbatim
  copy of the i6 form, mirrored exactly in Python).
- `determine_mean_delta_via_Z(sigma, p)` (`inferCNV_i3HMM.R:435-445`)
  — `|qnorm(p, mean=0, sd=sigma)| = sigma * |z_p|`.

**Py entry points**:
- Orchestration: `pyinfercnv.pipeline_phase2._run_hmm_by_subcluster`
  (`pipeline_phase2.py:606-718`) — iterates over subclusters and
  chromosomes, computes per-subcluster `rowMeans`, calls per-cell
  HMM, then broadcasts the trace back to all members.
- i6 decoder: `pyinfercnv.hmm.i6.predict_i6` (`hmm/i6.py:71-147`).
- i3 decoder: `pyinfercnv.hmm.i3.predict_i3` (`hmm/i3.py:124-184`)
  with parameter estimation in
  `pyinfercnv.hmm.i3.estimate_i3_state_params` (`hmm/i3.py:67-121`).
- Numba kernel: `pyinfercnv.kernels.hmm_viterbi_numba.viterbi_decode_numba`
  (`kernels/hmm_viterbi_numba.py:230`) — JIT-compiled Viterbi DP
  matching R's adjusted decoder.

**Tier**:
- Viterbi kernel: `0.9999` Jaccard on R-aligned input (CHANGELOG
  `## 0.2.0` Phase 2 — kernel-isolated parity).
- i3 end-to-end: Jaccard `1.000` (bit-exact post linear-FC switch,
  commit `a8d4526`; CHANGELOG `## 0.2.0`).
- i6 end-to-end: Jaccard `0.979` on oligo (HEAD measurement; floor
  0.96 in tests; CHANGELOG `## 0.2.0`). i6 hspike μ/σ baseline drift
  Jaccard `0.918 → 0.979` between commit `3d9ec79` and HEAD,
  documented as deferred root-cause investigation in CHANGELOG
  `## 0.2.0 Known limitations`.

**Behavior** (Python orchestration):
1. For each subcluster, compute `rowMeans(cnv_matrix[members, chr_slice])`.
2. For i6: pass per-state `mus` from
   `HspikeCalibration.state_mus` and per-state `sigmas` from
   `HspikeCalibration.sigmas_for_num_cells(n_members)` —
   matching R's `.get_state_emission_params`
   (`inferCNV_HMM.R:586-629`). Decode via
   `viterbi_decode_numba(seg, log_delta, log_trans, mus, sigmas)`.
3. For i3: pass per-state `(mus, sigmas)` from
   `estimate_i3_state_params(cnv_matrix_linear_fc, ref_idx, i3_p_val)`
   — Z-based variant (line 110-120 of `hmm/i3.py`). Decode via the
   same `viterbi_decode_numba` kernel.
4. Broadcast the decoded `(1, n_bins_chr)` trace to all member rows
   (`pipeline_phase2.py:695, 714`), mirroring R's
   `hmm.data[chr_gene_idx,tumor_subcluster_cells_idx] <<- hmm_trace`
   at `inferCNV_HMM.R:398`.
5. Reference cells retain `neutral_idx` everywhere
   (`pipeline_phase2.py:660`); R does not HMM-call reference cells.

**Observation-space contract**:
- i6 reads `result_phase1.cnv_matrix_fc` (post-step14 invert_log2 +
  step16 outlier-prune linear FC), per `pipeline_phase2.py:325-327`.
- i3 reads the same `cnv_matrix_fc` and re-estimates mu/sigma in
  that space (`pipeline_phase2.py:344-352`), matching R step 17
  which consumes `@expr.data` after step 14
  (`inferCNV_HMM.R:366` for i3; `inferCNV_HMM.R:383` for i6).

**R signature**: `predict_CNV_via_HMM_on_tumor_subclusters(infercnv_obj, t=1e-6)`.

**Py signature**: `predict_i6(cnv_matrix, chr_pos, *, transition_prob=1e-6, state_mus=None, state_sigmas=None) -> np.ndarray[int8]` and `predict_i3(cnv_matrix, chr_pos, *, transition_prob=1e-6, state_mus=None, state_sigmas=None) -> np.ndarray[int8]`.

**Key divergences**:
- State indexing: R uses 1-based labels `{1..6}` (i6) / `{1..3}` (i3);
  Python uses 0-based int8 `{0..5}` / `{0..2}`. Conversion is local
  to the kernel; `filter_high_p_normals.py:32-34` documents the
  mapping for downstream consumers.
- `analysis_mode='cells'` and `analysis_mode='samples'` are
  config-fail-loud at `config.py:158-164`
  (`InferCNVConfig.validate()` raises). Phase 3 work.
- `per_chr_hmm_subclusters=TRUE` is not implemented (Python always
  uses pooled HMM per `_run_hmm_by_subcluster`).
- Default state_mus / state_sigmas in `hmm/i6.py:39-46`
  (`I6_DEFAULT_MUS = log2([0.01, 0.5, 1.0, 1.5, 2.0, 3.0])`,
  `I6_DEFAULT_SIGMA = 0.5`) are documented as a *proxy*, NOT
  bit-exact; pipeline always supplies fitted values from
  `HspikeCalibration`.

**Tests**:
- `tests/test_r_parity.py::test_step17_hmm_i6_jaccard_floor` (line 554),
  floor 0.96.
- `tests/test_r_parity.py::test_step17_hmm_i3_jaccard_floor` (line 646),
  floor 0.99.

### Step 17b — Convert HMM state matrix to BED-like CNV regions

**R function**: `get_predicted_CNV_regions(infercnv_obj, by)`
(`inferCNV_HMM.R:706-788`) and
`generate_cnv_region_reports(infercnv_obj, output_filename_prefix, out_dir, ignore_neutral_state, by)`
(`inferCNV_HMM.R:790-889`). Called inline at `inferCNV_ops.R:1313-1317`.

**Py entry point**: `pyinfercnv.pipeline_phase2._build_cnv_regions`
(`pipeline_phase2.py:721-813`).

**Tier**: structural — produces a long-format DataFrame with
columns `[cell_group, subcluster, chromosome, bin_start, bin_end,
state, cn]`. Run-length-encodes non-neutral state segments per
`(subcluster, chromosome)`. Used downstream by step 18 (BayesNet)
and viz. `bin_end` is **inclusive** (mirrors R's BED-style
`HMM.txt` output convention).

---

## Step 18 — BayesNet Gibbs posterior (`inferCNV_ops.R:1362-1394`)

**R function**: `inferCNVBayesNet(infercnv_obj, HMM_states, file_dir, no_plot, postMcmcMethod="removeCNV", out_dir, resume_file_token, quietly, CORES, plotingProbs, diagnostics, HMM_type, k_obs_groups, cluster_by_groups, reassignCNVs, useRaster)`
(`inferCNV_BayesNet.R:1237-1392`). Internal helpers:
- `run_gibb_sampling(gene_exp, …)` (`inferCNV_BayesNet.R:1054-1108`)
  — actual JAGS BUGS run.
- `cnv_prob(combined_samples)` (`inferCNV_BayesNet.R:1137-1142`) —
  posterior summarizer producing `cnv_posterior[r, k]`.
- `cell_prob(combined_samples, obj)` (`inferCNV_BayesNet.R:1144-1151`).

JAGS BUGS model definitions live in `inst/BUGS_Mixture_Model` (i6)
and `inst/BUGS_Mixture_Model_i3` (i3) inside the R package
(`infercnv-master/inst/`).

Gated on `HMM == TRUE && BayesMaxPNormal > 0 && length(unique(apply(hmm.infercnv_obj@expr.data,2,unique))) != 1`
(`inferCNV_ops.R:1364`).

**Py entry points**:
- Orchestrator wrapper: `pyinfercnv.bayesnet._step18_bayesnet`
  (`bayesnet/__init__.py:34-167`).
- Sampler: `pyinfercnv.bayesnet.gibbs.run_bayesnet_gibbs`
  (`bayesnet/gibbs.py:74`). Numba kernel:
  `pyinfercnv.kernels.bayesnet_gibbs_numba.gibbs_sample_regions`
  (`kernels/bayesnet_gibbs_numba.py:115`) and
  `pack_regions` (`kernels/bayesnet_gibbs_numba.py:87`).

**Tier**: soft-tier (`|ΔP| < 0.10` ≥ 90% of regions; CHANGELOG
`## 0.2.0` Phase 3 row). Implements the `removeCNV`-only branch.

**Fail-loud guards**:
- `postMcmcMethod='removeCells'` → `NotImplementedError`
  (`bayesnet/gibbs.py:181-185`).
- `reassignCNVs=True` (R default) → `NotImplementedError`
  (`bayesnet/__init__.py:110-116` orchestrator boundary;
  `bayesnet/gibbs.py:186-190` sampler boundary). Listed in
  `CHANGELOG.md ## 0.2.0 Known limitations`.

**Behavior**: For each region (one row of `cnv_regions`), assemble
the linear-FC sub-matrix `gexp[gene_idx, cell_idx]` (note the
transpose to genes × cells inside the kernel), run a Gibbs sampler
with the i6/i3 BUGS model:
- Likelihood: `gexp[g, c] ~ Normal(mu[z[c]], sigma[z[c]])`.
- Prior: `z[c] ~ Categorical(theta)`, `theta ~ Dirichlet(alpha)`.
The sampler returns posterior means of `theta[k]` per region
(`cnv_posterior[r, k]`) plus per-cell state-occupancy counts
(`cell_posterior_counts`).

**R signature**: `inferCNVBayesNet(file_dir, infercnv_obj, HMM_states, …)`.

**Py signature**: `run_bayesnet_gibbs(hmm_state_matrix, cnv_matrix, cnv_regions, *, state_mus, state_sigmas, subclusters, HMM_type="i6", BayesMaxPNormal=0.5, numBurnin=1000, numSamples=1000, numChains=3, postProbsToRemoveLimit=0.5, reassignCNVs=False, CORES=1, quietly=True, diagnostics=False, postMcmcMethod="removeCNV", random_state=42) -> dict`.

**Key divergences**:
- R uses JAGS (rjags) → MT19937 RNG; Python uses NumPy PCG-64 with
  per-region derived seeds. Bit-equivalence not achievable;
  parity target is posterior probability (out of scope for bit-exact).
- R `numBurnin` ≈ `n.adapt + update` ≈ 700 vs Py default 1000;
  `numChains` R 6 vs Py 3 (runtime budget); see `gibbs.py:131-139`.
- `reassignCNVs=True` (R default) raises in Python; only the
  `removeCNV`-only branch is ported.
- The orchestrator at `bayesnet/__init__.py:110` reads
  `config.reassignCNVs` (Python config default `True` to mirror R)
  and **raises immediately** unless the user explicitly sets it to
  False.
- `BayesMaxPNormal` is consumed at step 19, not step 18 (Python
  passes through as kwarg for signature parity, see
  `gibbs.py:126-130`).

**Tests**: `tests/test_r_parity.py::test_step18_bayesnet_parity`
(line 827).

---

## Step 19 — Filter high P(normal) CNVs (`inferCNV_ops.R:1405-1454`)

**R function**: `filterHighPNormals(MCMC_inferCNV_obj, HMM_states, BayesMaxPNormal, useRaster)`
(`inferCNV_BayesNet.R:1394` start; `removeCNV` body at
`inferCNV_BayesNet.R:562-630`). Same gating as step 18.

**Py entry point**: `pyinfercnv.bayesnet.filter_high_p_normals.filter_high_p_normals`
(`bayesnet/filter_high_p_normals.py:37-101`). Wrapped by
`_step18_bayesnet` (`bayesnet/__init__.py:160-165`) so steps 18+19
fire back-to-back.

**Tier**: post-filter Jaccard `≥ 0.94`; on oligo `0.979` (CHANGELOG
`## 0.2.0` Phase 3 row).

**Behavior**: For each region with
`cnv_posterior[r, neutral] > BayesMaxPNormal`, overwrite all
`(gene, cell)` positions of that region in the HMM state matrix
back to the neutral state. Returns a *new* int8 state matrix; input
is not mutated.

**R signature**: `filterHighPNormals(MCMC_inferCNV_obj, HMM_states, BayesMaxPNormal=0.5, useRaster=TRUE)`.

**Py signature**: `filter_high_p_normals(hmm_state_matrix: np.ndarray, bayesnet_result: dict, *, HMM_type: str = "i6", BayesMaxPNormal: float = 0.5) -> np.ndarray`.

**Key divergences**:
- Implements only the `removeCNV` branch (R's
  `postMcmcMethod='removeCNV'` default). The `reassignCNV` branch
  (`inferCNV_BayesNet.R:491-540`) is not ported — see step 18 fail-loud
  guard.
- Neutral state index hard-coded per HMM type:
  `_NEUTRAL_IDX = {"i6": 2, "i3": 1}` (`filter_high_p_normals.py:34`),
  matching the 0-based Python convention.

**Tests**: `tests/test_r_parity.py::test_step19_filter_high_p_normals`
(line 999).

---

## Step 20 — Convert HMM states to representative expression values (`inferCNV_ops.R:1463-1500`)

**R functions**:
- `assign_HMM_states_to_proxy_expr_vals(infercnv_obj)`
  (`inferCNV_HMM.R:1191-1202`) — i6. Sequential in-place overwrites:
  ```
  expr.data[expr.data == 1] <- 0
  expr.data[expr.data == 2] <- 0.5
  expr.data[expr.data == 3] <- 1
  expr.data[expr.data == 4] <- 1.5
  expr.data[expr.data == 5] <- 2
  expr.data[expr.data == 6] <- 3
  ```
- `i3HMM_assign_HMM_states_to_proxy_expr_vals(infercnv_obj)`
  (`inferCNV_i3HMM.R:405-413`) — i3. Sequential in-place overwrites:
  ```
  expr.data[expr.data == 1] <- 0.5
  expr.data[expr.data == 2] <- 1
  expr.data[expr.data == 3] <- 1.5
  ```

R-level gating: `if (HMM)` and `skip_hmm < 4`
(`inferCNV_ops.R:1464-1465`). Note: step 20 fires whenever HMM=TRUE
regardless of BayesMaxPNormal — when BayesMaxPNormal=0 it operates
on the unfiltered step-17 state matrix.

**Py entry point**: `pyinfercnv.pipeline_phase3._step20_assign_states_to_proxy_expr_vals`
(`pipeline_phase3.py:104-161`). Lookup tables:
`_I6_STATE_TO_CN = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]` (line 96)
and `_I3_STATE_TO_CN = [0.5, 1.0, 1.5]` (line 99).

**Tier**: bit-exact (`max_diff = 0.000e+00` for both i6 and i3;
CHANGELOG `## 0.2.0` Phase 3 row).

**Behavior**: Vectorised fancy-index lookup
`proxy = state_map[states.astype(np.intp)]`. R uses sequential
in-place overwrites which are equivalent because the original
values are integer state labels with no overlap (the substitution
order is irrelevant). Output dtype float64.

When BayesNet ran (step 18+19), the post-filter `filtered_states`
array is used; otherwise the raw Phase 2 HMM state matrix
(`pipeline_phase3.py:130-143`).

**R signature**: `assign_HMM_states_to_proxy_expr_vals(infercnv_obj)` and `i3HMM_assign_HMM_states_to_proxy_expr_vals(infercnv_obj)`.

**Py signature**: `_step20_assign_states_to_proxy_expr_vals(result: InferCNVResult, config: InferCNVConfig, *, filtered_states: np.ndarray | None = None) -> np.ndarray`.

**Key divergences**:
- Python uses 0-based state indices throughout; lookup table indices
  shift by -1 versus R's 1-based labels. Comments at
  `pipeline_phase3.py:77-93` cite the R lines explicitly.
- Output is a separate `result.hmm_proxy_matrix` field
  (`result.py:131-138`) instead of overwriting in place; R mutates
  `hmm.infercnv_obj@expr.data`. Python's separation preserves the
  non-destructive pre/post comparison.

**Tests**:
- `tests/test_r_parity.py::test_step20_state_proxy_parity` (line 1071,
  i6).
- `tests/test_r_parity.py::test_step20_state_proxy_i3_parity`
  (line 1138, i3).

---

## Step 21 — Mask non-DE genes (`inferCNV_ops.R:1509-1552`)

**R function**: `mask_non_DE_genes_basic(infercnv_obj, p_val_thresh, test.use, center_val, require_DE_all_normals)`
(`inferCNV_mask_non_DE.R:28-52`). Internals:
- `get_DE_genes_basic(…)` (`inferCNV_mask_non_DE.R:158-259`) —
  per-gene Wilcoxon test with BH adjustment at L237.
  `statfxns[["wilcoxon"]]` at L191-203.
- `.mask_DE_genes(…)` (`inferCNV_mask_non_DE.R:77-134`) — applies
  the mask, counts normals where each gene was DE, thresholds with
  `require_DE_all_normals` ∈ {"any","most","all"}.

Gated on `mask_nonDE_genes` (R kwarg, default FALSE).

**Py entry point**: `pyinfercnv.mask_de.wilcoxon._step21_mask_non_DE`
(`mask_de/wilcoxon.py:283`). Underlying public function:
`mask_non_DE_genes` (`mask_de/wilcoxon.py:108`). Helpers:
- `_wilcoxon_pvals_per_gene` (line 60) — vectorised
  `scipy.stats.mannwhitneyu(..., use_continuity=True, method="asymptotic", axis=0)`.
- `_bh_adjust` (line 92) — BH FDR via
  `statsmodels.stats.multitest.multipletests(method="fdr_bh")`.

**Tier**: bit-exact (`max_diff = 1.110e-16`; mask boolean `xor_frac < 1e-4`,
soft floor due to residual-inference jitter; CHANGELOG `## 0.2.0`
Phase 3 row, commit `ddd3c32`).

**R vs scipy parity pact** (mask_de/wilcoxon.py:16-33):
- Force `method="asymptotic"` on the Python side and pass
  `exact=FALSE` on the R side via a monkey-patch in
  `tests/r_reference.R` step21 dump.
- R additionally adds `rnorm(mean=1e-4, sd=1e-4)` jitter for
  tie-breaking; the test patch strips this. Without the patch
  bit-exact parity is unachievable because R RNG is not
  bit-equivalent to NumPy PCG-64.

**Behavior**: For each tumour subcluster × reference-group pair, run
`mannwhitneyu(normal_vals, tumor_vals, two-sided)` per gene, apply
BH adjustment, count "DE against this normal type" per gene, then
threshold by `require_DE_all_normals`:
- `"any"` (default): keep gene if DE against ≥1 normal type.
- `"most"`: keep gene if DE against `> n_normals/2`.
- `"all"`: keep gene if DE against all normal types.

Non-DE positions are flattened to `center_val` (R uses
`mean(infercnv_obj@expr.data)`, `inferCNV_ops.R:1523`).

**R signature**: `mask_non_DE_genes_basic(infercnv_obj, p_val_thresh=0.05, test.use="wilcoxon", center_val, require_DE_all_normals="any")`.

**Py signature**: `mask_non_DE_genes(expr_matrix, *, tumor_subcluster_indices, reference_group_indices, mask_nonDE_pval=0.05, test_use="wilcoxon", require_DE_all_normals="any", center_val=None, min_cluster_size_mask=5) -> tuple[np.ndarray, np.ndarray]`.

**Key divergences**:
- `test.use="perm"` (R coin package) is **out of scope**; Python
  honours only `"wilcoxon"` (default) and `"t"` (lazy).
- `min_cluster_size_mask=5` is a Python-only safety floor (R has no
  equivalent); subclusters smaller than this are skipped.
- `mask_nonDE_genes=True` requires `HMM=True` in Python because
  step 21 consumes `result.subclusters` from Phase 2 — fail-loud
  guard at `pipeline.py:346-352`. R itself does not require HMM
  for mask_non_DE (`inferCNV_mask_non_DE.R:35` reads
  `observation_grouped_cell_indices` / `reference_grouped_cell_indices`
  directly).

**Tests**: `tests/test_r_parity.py::test_step21_mask_non_DE_parity`
(line 1190).

---

## Step 22 — Denoising (`inferCNV_ops.R:1559-1615`)

**R functions** (dispatch on `noise_filter`):
- `clear_noise_via_ref_mean_sd(infercnv_obj, sd_amplifier=1.5, noise_logistic=FALSE)`
  (`inferCNV_ops.R:2302-2346`) — `noise_filter` is `NA` (default).
  Algorithm (L2304-2335):
  1. Pick reference cells; fall back to all observation cells if no
     reference is set (L2308-2311).
  2. `mean_ref_vals = mean(vals)` (scalar).
  3. `mean_ref_sd = mean(apply(vals, 2, sd, na.rm=TRUE)) * sd_amplifier`.
  4. `upper = mean_ref_vals + mean_ref_sd`,
     `lower = mean_ref_vals - mean_ref_sd`.
  5. `expr[expr > lower & expr < upper] = mean_ref_vals` (in-place
     flatten).
- `clear_noise(infercnv_obj, threshold, noise_logistic=FALSE)`
  (`inferCNV_ops.R:2232-2268`) — `noise_filter > 0`. Band =
  `[mean_ref - threshold, mean_ref + threshold]`. The mean falls
  back to the *full matrix* mean when no reference is set (L2246) —
  *different* from the sd-path fallback.
- `apply_median_filtering` (`noise_reduction.R:43-90`) — alternate
  denoise path, **fail-loud guard** in Python.

Gated on `denoise` (R kwarg default FALSE).

**Py entry point**: `pyinfercnv.denoise.ref_mean_sd._step22_denoise`
(`denoise/ref_mean_sd.py:218-294`). Public function:
`denoise_by_ref_mean_sd` (line 53).

**Tier**: bit-exact (`max_diff < 1e-10`; CHANGELOG `## 0.2.0` Phase 3
row; `tests/test_r_parity.py::test_step22_noise_reduction_parity` line 1312).

**Fail-loud guards**:
- `noise_logistic=True` → `NotImplementedError`
  (`denoise/ref_mean_sd.py:128-133`). Pipeline-level guard at
  `pipeline.py:328-333` raises before run_phase3 is called when
  `denoise=True && noise_logistic=True`.
- `apply_median_filtering` → not exposed as a config field; the
  feature is listed in CHANGELOG `## 0.2.0 Known limitations`.

**Behavior**: Pure mean / sd / boolean clip; no RNG, no ties. Strict
inequalities (`>` and `<`) per R `inferCNV_ops.R:2275, 2335`; values
exactly on the band boundary are *not* flattened. R's `sd` uses
`ddof=1`; Python passes `ddof=1` explicitly (NumPy default `ddof=0`
would break parity).

R's `apply(vals, 2, sd)` is per-cell sd (R orients `expr.data` as
genes × cells, so `MARGIN=2` picks columns = cells). Python's
`cnv_matrix` is cells × genes, so the per-cell axis is `axis=1`
(`denoise/ref_mean_sd.py:120-123`).

Python writes the result to `result.denoised_matrix` (a separate
field, `result.py:122-127`); R mutates in place. Separation
preserves the pre/post non-destructive comparison.

**R signature**: `clear_noise_via_ref_mean_sd(infercnv_obj, sd_amplifier=1.5, noise_logistic=FALSE)` and `clear_noise(infercnv_obj, threshold, noise_logistic=FALSE)`.

**Py signature**: `denoise_by_ref_mean_sd(cnv_matrix, ref_indices, *, noise_filter: float | None = None, sd_amplifier: float = 1.5, noise_logistic: bool = False) -> np.ndarray`.

**Key divergences**:
- `noise_logistic=True` not ported (sigmoidal mask
  `depress_log_signal_midpt_val` at `inferCNV_heatmap.R:2783`).
- `apply_median_filtering` not ported.
- Output goes to `result.denoised_matrix`, not in-place mutation.

**Tests**: `tests/test_r_parity.py::test_step22_noise_reduction_parity`
(line 1312).

---

## Cross-cutting topics

### State indexing convention

| HMM type | R labels | Python labels | Neutral idx (R / Py) |
|---|---|---|---|
| i6 | `{1, 2, 3, 4, 5, 6}` | `{0, 1, 2, 3, 4, 5}` | 3 / 2 |
| i3 | `{1, 2, 3}` | `{0, 1, 2}` | 2 / 1 |

CN-ratio mapping (i6): R `{1,…,6}` ↔ Py `{0,…,5}` ↔
CN `{0.0, 0.5, 1.0, 1.5, 2.0, 3.0}`
(`pipeline_phase3.py:96-98`, mirrored in
`hmm/i6.py:39 I6_CNV_LEVELS`).

CN-ratio mapping (i3): R `{1,2,3}` ↔ Py `{0,1,2}` ↔
CN `{0.5, 1.0, 1.5}` (`pipeline_phase3.py:99-101`).

### Observation-space invariants by step

| Step | Space (entering step) | Space (leaving step) |
|---|---|---|
| 2 (filter) | raw counts | raw counts (subset cols) |
| 3 (normalize) | raw counts | CPM |
| 4 (log2+1) | CPM | log2(CPM+1) |
| 8 (subtract ref 1) | log2(CPM+1) | log2(FC) |
| 9 (max threshold) | log2(FC) | log2(FC) clipped |
| 10 (smooth) | log2(FC) | smoothed log2(FC) |
| 11 (center) | smoothed log2(FC) | per-cell-median-centered |
| 12 (subtract ref 2) | centered | centered + ref-zeroed |
| 14 (invert log2) | centered log2(FC) | linear FC |
| 16 (outliers) | linear FC | linear FC clipped |
| 17 (HMM) | linear FC | int8 state matrix |
| 18 (BayesNet) | linear FC + states | posterior |
| 19 (filter) | states + posterior | filtered states |
| 20 (proxy) | filtered states | CN ratio |
| 21 (mask) | linear FC | masked linear FC |
| 22 (denoise) | linear FC (or masked) | denoised linear FC |

### Fields populated on `InferCNVResult` per phase

(See `pyinfercnv/result.py:88-156` for the full schema.)

| Field | Phase | Populated by |
|---|---|---|
| `chr_pos` | 1 | `pipeline.infercnv` |
| `cnv_matrix` (float32) | 1 | `pipeline.infercnv` (cast at end of Phase 1) |
| `cnv_matrix_fc` (float32) | 1 | `pipeline.infercnv` (post-step 14, post-step 16) |
| `cell_meta` (DataFrame) | 1 | `pipeline.infercnv` |
| `ref_counts_raw` | 1 (G1 P2) | `pipeline.infercnv` (pre-normalize snapshot) |
| `cpm_matrix_f32` (transient) | 1 | `pipeline.infercnv` (post-normalize, pre-log2 snapshot) |
| `cnv_matrix_f64` (transient) | 1 | `pipeline.infercnv` (full-precision Phase 1 output) |
| `subclusters` (int32) | 2 | `pipeline_phase2._run_subclustering` (step 15) |
| `hmm_states` (int8) | 2 | `pipeline_phase2._run_hmm_by_subcluster` (step 17 i6) |
| `hmm_states_i3` (int8) | 2 | `pipeline_phase2._run_hmm_by_subcluster` (step 17 i3) |
| `cnv_regions` (DataFrame) | 2 | `pipeline_phase2._build_cnv_regions` (step 17b) |
| `hspike_calibration` | 2 | `pipeline_phase2._calibrate_hmm_emission` (i6) |
| `i3_state_mus` / `i3_state_sigmas` | 2 | `pipeline_phase2.run_phase2` (i3 path) |
| `bayes_posterior` (n_regions, K) | 3 | `pipeline_phase3.run_phase3` step 18 |
| `hmm_proxy_matrix` (float64) | 3 | `pipeline_phase3._step20_assign_states_to_proxy_expr_vals` (step 20) |
| `de_mask` (bool) | 3 | `pipeline_phase3` step 21 |
| `denoised_matrix` (float32) | 3 | `pipeline_phase3` step 22 |
| `profile` (dict) | 1/2/3 | `_profile_block` per major operation |

### R-default branches that fail-loud in Python

(See `CHANGELOG.md ## 0.2.0 Known limitations` for the
authoritative list.)

| R kwarg / branch | Python guard location | Status |
|---|---|---|
| `BayesNet reassignCNVs=TRUE` | `bayesnet/__init__.py:110-116` (orchestrator) and `bayesnet/gibbs.py:186-190` | `NotImplementedError` |
| `denoise noise_logistic=TRUE` | `pipeline.py:328-333` (orchestrator) and `denoise/ref_mean_sd.py:128-133` | `NotImplementedError` |
| `BayesNet postMcmcMethod='removeCells'` | `bayesnet/gibbs.py:181-185` | `NotImplementedError` |
| `hspike sim_method='simple' / 'splatter'` | `hmm/hspike.py` (search `sim_method`) | `NotImplementedError` |
| `preprocess threshold='auto'` | `preprocess/max_threshold.py:21` | `NotImplementedError` |
| `denoise apply_median_filtering` | not exposed as config field | `NotImplementedError` (no entry point) |
| `analysis_mode in {"samples","cells"}` | `config.py:158-164` (`InferCNVConfig.validate`) | `ValueError` |
| `mask_nonDE_genes=True && HMM=False` | `pipeline.py:346-352` | `ValueError` (Python-side; R allows it) |
| `BayesMaxPNormal>0 && HMM=False` | `pipeline.py:336-341` | `ValueError` |

### Default-config skipped or no-op steps

R steps where the body is *gated* and does not execute under default
`InferCNVConfig`:

- **Step 5** (`scale_data`): R default FALSE. Python does not implement
  `scale_infercnv_expr` at all.
- **Step 6** (`num_ref_groups`): R default NULL. Python's
  `subtract_reference` consumes annotation-derived ref groups
  directly; step 6 never fires.
- **Step 7** (`tumor_subcluster_partition_method='random_trees'`):
  R default 'leiden' (which routes to step 15 instead). Python
  ignores the step-7 position; the random_trees backend is dispatched
  at step 15.
- **Step 13** (`remove_genes_at_chr_ends`): R default FALSE.
  Python does not implement this routine.
- **Step 16** (`prune_outliers`): R default FALSE. Python's
  `prune_outliers` runs only when `cfg.prune_outliers=True`.
- **Step 21** (`mask_nonDE_genes`): default FALSE in both.
- **Step 22** (`denoise`): default FALSE in both.

When a gate is OFF, the orchestrator still increments `step_count`
and the corresponding `if (up_to_step == step_count)` early-return
is honoured. Python preserves this numbering implicitly because the
top-level `infercnv()` calls each module unconditionally only for
the always-on steps (2, 3, 4, 8, 9, 10, 11, 12, 14) and gates the
rest on config flags.

### Phase 2 → Phase 3 emission handoff

The hspike calibration produced at Phase 2 step 17 (i6) and the
i3 per-state `(mus, sigmas)` are persisted onto the result so
Phase 3 step 18 BayesNet can reuse them without re-running the
calibration:

- `result.hspike_calibration: HspikeCalibration | None`
  (`result.py:152`) — i6 only.
- `result.i3_state_mus: np.ndarray | None`,
  `result.i3_state_sigmas: np.ndarray | None`
  (`result.py:153-154`) — i3 only.

`pipeline_phase3.run_phase3` auto-sources these fields when
`BayesMaxPNormal > 0`:
- i6: `pipeline_phase3.py:235-249` reads `result.hspike_calibration`
  and forwards into `_step18_bayesnet(..., i6_calibration=…)`.
- i3: `pipeline_phase3.py:250-261` reads
  `result.i3_state_mus`/`result.i3_state_sigmas` and forwards into
  `_step18_bayesnet(..., i3_mus=…, i3_sigmas=…)`.

Either field being `None` when its HMM_type-corresponding branch
fires raises `ValueError` with a clear "did Phase 2 run?" message.

### Sim-only R modules (out-of-scope by design)

| R file | R purpose | Py mapping |
|---|---|---|
| `inferCNV_meanVarSim.R` | mean-var trend simulation for hspike | inlined into `pyinfercnv.hmm.hspike._fit_meanvar_spline` (`hspike.py:790`) and `_simulate_meanvar_counts` (`hspike.py:855`) |
| `inferCNV_simple_sim.R` | "simple" hspike simulation alt | not ported (fail-loud) |
| `SplatterScrape.R` | Splatter-style alt simulation | not ported (fail-loud) |
| `inferCNV_heatmap.R` | viz / heatmap rendering | partially mirrored in `pyinfercnv.viz` (matplotlib heatmap, no omicverse dep, see CHANGELOG `## 0.1.0.dev0`) |
| `seurat_interaction.R` | Seurat object I/O | out of scope (Python uses AnnData) |
| `data.R` | dataset descriptors | n/a |

### Subcluster backends overview

| Backend | R source | Py source | Notes |
|---|---|---|---|
| Leiden (default) | `inferCNV_tumor_subclusters.R:569-644` | `pyinfercnv/subcluster/leiden.py:194` | KNN tie-break differences vs R `RANN::nn2` |
| random_trees | `inferCNV_tumor_subclusters.random_smoothed_trees.R` (file) | `pyinfercnv/subcluster/random_trees.py:208` | Recursive permutation-based partition |
| qnorm | `.get_tree_height_via_ecdf` (`inferCNV_tumor_subclusters.R:561-568`) | `pyinfercnv/subcluster/qnorm.py:31` | Hierarchical + qnorm cut, deterministic |

### Smoothing kernels overview

| Kernel | R | Py module | Tier |
|---|---|---|---|
| `pyramidinal` (R default) | `smooth_by_chromosome` (`inferCNV_ops.R:2406`), `.smooth_helper`, `.smooth_center_helper` (`inferCNV_ops.R:2483-2660`) | `pyinfercnv/smooth/pyramidinal.py` + `kernels/smooth_center_numba.py` + `kernels/smooth_tail_numba.py` | bit-exact |
| `runmeans` | `smooth_by_chromosome_runmeans` (`inferCNV_ops.R:2679-2721`) | not ported | n/a |
| `coordinates` | `smooth_by_chromosome_coordinates` (`inferCNV_ops.R:2534-2563`) | not ported | n/a |

### HMM kernels overview

| Component | R | Py |
|---|---|---|
| Viterbi decoder | `Viterbi.dthmm.adj` (`inferCNV_HMM.R:1101-1189`) | `kernels/hmm_viterbi_numba.py:230 viterbi_decode_numba` |
| Forward-Backward | (R does not expose a public FB) | `kernels/hmm_viterbi_numba.py:261 forward_backward_numpy` (used internally for diagnostics) |
| i6 transition / initial | `.get_HMM(cnv_mean_sd, t)` (`inferCNV_HMM.R:230-265`) | `hmm/i6.py:50-68 _build_transition_matrix` / `_build_initial_distribution` |
| i3 transition / initial | `.i3HMM_get_HMM(cnv_mean_sd, t)` (`inferCNV_i3HMM.R:99-156`) | `hmm/i3.py:44-64 _build_transition_matrix` / `_build_initial_distribution` |
| i6 emission Gaussian | inline in `Viterbi.dthmm.adj` | `kernels/hmm_viterbi_numba.py:60 _rstyle_log_emit` |
| i3 emission Gaussian | inline in `i3HMM_predict_*` | same kernel |
| i6 hspike calibration | `.build_and_add_hspike` + `get_spike_dists` + `.get_state_emission_params` + `get_hspike_cnv_mean_sd_trend_by_num_cells_fit` (`inferCNV_hidden_spike.R`, `inferCNV_HMM.R:154-228, 586-629`) | `hmm/hspike.py:298 calibrate_i6_emission` (frozen `HspikeCalibration` at line 192) |
| i3 mu/sigma estimation | `.i3HMM_get_sd_trend_by_num_cells_fit` + `determine_mean_delta_via_Z` (`inferCNV_i3HMM.R:17-80, 435-445`) | `hmm/i3.py:67 estimate_i3_state_params` |

### BayesNet kernel composition

`pyinfercnv.bayesnet.gibbs.run_bayesnet_gibbs` calls the JIT-compiled
`pyinfercnv.kernels.bayesnet_gibbs_numba.gibbs_sample_regions`
(`kernels/bayesnet_gibbs_numba.py:115`) after packing the
per-region submatrices via
`pack_regions(gexps)` (line 87).

Variational backend `pyinfercnv.bayesnet.vb` is listed as a 0.3
backlog item in `CHANGELOG.md ## 0.2.0 Known limitations`
(performance optimization; full-quality Gibbs is the primary path).

---

## Document provenance

This document was rebuilt on 2026-04-26 by a general-purpose agent
after the original `docs/superpowers/algorithm_correspondence_R_vs_Python.md`
was lost in a `git filter-repo` accident together with `CODEX_HANDOFF.md`.
The rebuild was performed by reading the R source under
`infercnv-master/R/` and the Python source under `pyinfercnv/`
side-by-side; every R source citation was verified by opening the
referenced file at the cited line range.

For any specific tier claim, **verify against
`tests/test_r_parity.py`** — that file is the executable parity
contract. The CHANGELOG entry `## 0.2.0` summarises the
test-by-test results at release time but `tests/test_r_parity.py`
is canonical.

For the high-level R↔Py module map (less granular than this
document, no per-step breakdown), see
`NAMESPACE_PARITY.md`.
