"""Top-level infercnv() — Phase 1 orchestration of R steps 1-14 + 16.

R-parity step mapping:
    1.  Incoming data                   -> extract_counts + adata.var join
    2.  Remove lowly expressed genes    -> preprocess.filter_low_expression_genes
        (chr_exclude applied here too: drop chrX/Y/M genes)
    3.  Normalize by seq depth          -> preprocess.normalize_by_seq_depth
    4.  log2(x+1)                       -> preprocess.log2_plus1
    5.  (skipped; scale_data default False)
    6.  (skipped; num_ref_groups not set)
    7.  (skipped; tumor subclustering Phase 2)
    8.  Subtract ref mean (pre-smooth)  -> preprocess.subtract_reference
        (use_bounds per config; default TRUE)
    9.  Max-centered threshold          -> preprocess.apply_max_centered_threshold
    10. Smooth per chromosome           -> smooth.smooth_pyramidinal per chr
    11. Center cells (median)           -> center.center_cells
    12. Subtract ref mean (post-smooth) -> preprocess.subtract_reference (2nd)
    14. invert_log2                     -> preprocess.invert_log2
    16. Prune outliers                  -> cna.prune_outliers (when enabled)

G1 patches applied:
    P2  -- ref_counts_raw captured BEFORE normalize for Phase 2 hspike
           calibration; threaded into InferCNVResult.
    P6  -- TODO: per-chromosome chunked densify. Current implementation
           densifies once at step 8 (acceptable for <50k cells; spec
           noted as Phase 1.5 optimisation).
    P11 -- psutil profile hooks per major block, written to
           InferCNVResult.profile dict.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import sparse as sp

from pyinfercnv.center import center_cells
from pyinfercnv.cna import prune_outliers
from pyinfercnv.config import InferCNVConfig
from pyinfercnv.io.h5ad import extract_counts
from pyinfercnv.preprocess import (
    apply_max_centered_threshold,
    filter_low_expression_genes,
    invert_log2,
    log2_plus1,
    normalize_by_seq_depth,
    subtract_reference,
)
from pyinfercnv.result import InferCNVResult
from pyinfercnv.smooth import smooth_pyramidinal

if TYPE_CHECKING:
    from anndata import AnnData


_PROFILE_LOG = logging.getLogger("pyinfercnv.profile")


def _rss_mb() -> float | None:
    """Return current process RSS in MB via psutil, or None if not installed."""
    try:
        import psutil
        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        return None


def _profile_block(profile: dict[str, dict[str, Any]], name: str, t0: float, rss_before: float | None) -> None:
    elapsed = time.perf_counter() - t0
    rss_after = _rss_mb()
    profile[name] = {
        "wallclock_s": float(elapsed),
        "rss_mb_before": rss_before,
        "rss_mb_after": rss_after,
        "rss_delta_mb": (rss_after - rss_before) if (rss_after is not None and rss_before is not None) else None,
    }
    _PROFILE_LOG.info("block=%s wallclock=%.3fs rss_after=%s", name, elapsed, rss_after)


def _build_chromosome_layout(
    var_df: pd.DataFrame, exclude_chromosomes: Sequence[str]
) -> tuple[dict[str, int], np.ndarray]:
    """Return (chr_pos, gene_perm) — chromosome -> start col idx + permutation
    that sorts genes by chromosome (natural order) then start position."""
    if "chromosome" not in var_df.columns:
        raise ValueError(
            "adata.var must have 'chromosome' column. Use io.genome.load_gene_positions "
            "and merge into adata.var first."
        )
    mask = var_df["chromosome"].notna() & ~var_df["chromosome"].isin(exclude_chromosomes)
    kept = var_df[mask].copy()

    def _chr_key(c: str) -> tuple:
        s = c.replace("chr", "")
        try:
            return (0, int(s))
        except ValueError:
            return (1, s)

    sorted_chroms = sorted(kept["chromosome"].unique(), key=_chr_key)
    out_idx: list = []
    chr_pos: dict[str, int] = {}
    running = 0
    for c in sorted_chroms:
        sub = kept[kept["chromosome"] == c]
        if "start" in sub.columns:
            sub = sub.sort_values("start")
        chr_pos[c] = running
        out_idx.extend(sub.index.tolist())
        running += len(sub)

    gene_perm = var_df.index.get_indexer(out_idx)
    return chr_pos, gene_perm


def infercnv(
    adata: AnnData,
    *,
    config: InferCNVConfig | None = None,
    reference_key: str | None = None,
    reference_cat: str | Sequence[str] | None = None,
    exclude_chromosomes: Sequence[str] | None = None,
    key_added: str = "cnv",
    inplace: bool = True,
) -> InferCNVResult | None:
    """Phase 1 end-to-end pipeline. Returns InferCNVResult or writes into adata."""
    cfg = config or InferCNVConfig()
    cfg.validate()
    excl = tuple(exclude_chromosomes) if exclude_chromosomes is not None else cfg.chr_exclude
    profile: dict[str, dict[str, Any]] = {}

    # Step 1 — extract counts (CSR float32)
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    X = extract_counts(adata, counts_layer=cfg.counts_layer)
    _profile_block(profile, "01_extract", t0, rss0)

    # Identify reference cells
    if reference_key is None or reference_cat is None:
        ref_idx_all = np.arange(adata.n_obs)
    else:
        cats = [reference_cat] if isinstance(reference_cat, str) else list(reference_cat)
        ref_idx_all = np.where(adata.obs[reference_key].isin(cats).to_numpy())[0]

    # G1 P2 — capture ref_counts_raw BEFORE normalize/log
    t0 = time.perf_counter(); rss = _rss_mb()
    if len(ref_idx_all) > 0:
        ref_counts_raw = (
            X[ref_idx_all, :].toarray().astype(np.float32) if sp.issparse(X)
            else np.asarray(X[ref_idx_all, :], dtype=np.float32)
        )
    else:
        ref_counts_raw = None
    _profile_block(profile, "02_capture_ref_raw", t0, rss)

    # Step 2 — filter genes (mean cutoff + min cells)
    # Option 1 (2026-04-22): population is all cells, mirroring R's
    # require_above_min_mean_expr_cutoff which runs on the full matrix.
    # Passing reference_cell_idx=ref_idx_all here collapsed the filter to
    # ref-only and dropped ~1559 genes (8508 → 6949) that R keeps. R's
    # secondary require_above_min_cells_ref stage is not applied under
    # default parameters on the oligodendroglioma fixture — verified by
    # tmp_phase1_gap_probe.py [A] mode matching R step02 bit-exact.
    t0 = time.perf_counter(); rss = _rss_mb()
    keep = filter_low_expression_genes(
        X, cutoff=cfg.cutoff, min_cells_per_gene=cfg.min_cells_per_gene,
    )
    X = X[:, keep] if sp.issparse(X) else X[:, keep]
    var_kept = adata.var.iloc[np.where(keep)[0]].copy()
    if ref_counts_raw is not None:
        ref_counts_raw = ref_counts_raw[:, keep]
    _profile_block(profile, "03_filter_genes", t0, rss)

    # Step 3 — normalize by seq depth
    t0 = time.perf_counter(); rss = _rss_mb()
    X = normalize_by_seq_depth(X)
    _profile_block(profile, "04_normalize", t0, rss)

    # R-parity hspike source snapshot: post-filter + post-normalize +
    # pre-log2. Mirrors infercnv_obj@expr.data at the moment R calls
    # .build_and_add_hspike (inferCNV_ops.R:586-595). Dense float32 copy
    # (hspike is always built on a small synthetic matrix, so this is a
    # one-off ~MB on oligo). Nulled after Phase 2.
    if sp.issparse(X):
        cpm_matrix_f32: np.ndarray | None = X.toarray().astype(np.float32, copy=False)
    else:
        cpm_matrix_f32 = np.ascontiguousarray(X, dtype=np.float32)

    # Step 4 — log2(x+1)
    t0 = time.perf_counter(); rss = _rss_mb()
    X = log2_plus1(X)
    _profile_block(profile, "05_log2", t0, rss)

    # Build chromosome layout (also drops chr_exclude genes)
    chr_pos, gene_perm_local = _build_chromosome_layout(var_kept, excl)
    # Per-bin genomic coordinates aligned to the plotted column order (same
    # permutation applied to X below). Enables arm-aware (p/q) plotting.
    _pos_cols = [c for c in ("chromosome", "start", "end") if c in var_kept.columns]
    bin_meta = (
        var_kept.iloc[gene_perm_local][_pos_cols].reset_index(drop=True)
        if "chromosome" in _pos_cols
        else None
    )
    if sp.issparse(X):
        X = X.tocsc()[:, gene_perm_local].tocsr()
    else:
        X = X[:, gene_perm_local]
    if ref_counts_raw is not None:
        ref_counts_raw = ref_counts_raw[:, gene_perm_local]

    # Densify (Phase 1: full matrix; Phase 1.5 will move to per-chrom chunks).
    # Phase 1 bit-exact path (2026-04-23): float64 through all intermediate
    # steps; final cast to float32 happens only at result.cnv_matrix
    # assembly (end of function) for the public downstream contract.
    t0 = time.perf_counter(); rss = _rss_mb()
    X_dense = X.toarray().astype(np.float64) if sp.issparse(X) else np.ascontiguousarray(X, dtype=np.float64)
    _profile_block(profile, "06_densify", t0, rss)

    # Reference groups
    if reference_key is None or reference_cat is None or len(ref_idx_all) == 0:
        ref_groups: dict[str, list[int]] = {"proxyNormal": list(range(X_dense.shape[0]))}
    else:
        cats = [reference_cat] if isinstance(reference_cat, str) else list(reference_cat)
        ref_groups = {
            str(c): np.where(adata.obs[reference_key].to_numpy() == c)[0].tolist()
            for c in cats
        }

    # Step 8 — subtract ref (1st pass, bounded if K>1)
    t0 = time.perf_counter(); rss = _rss_mb()
    X_dense = subtract_reference(X_dense, ref_groups=ref_groups, use_bounds=cfg.ref_subtract_use_mean_bounds)
    _profile_block(profile, "07_subtract_ref_1", t0, rss)

    # Step 9 — max-centered threshold clip
    t0 = time.perf_counter(); rss = _rss_mb()
    X_dense = apply_max_centered_threshold(X_dense, threshold=cfg.max_centered_threshold)
    _profile_block(profile, "08_max_threshold", t0, rss)

    # Step 10 — smooth per chromosome
    t0 = time.perf_counter(); rss = _rss_mb()
    smoothed = np.empty_like(X_dense)
    chroms = list(chr_pos.keys())
    for i, c in enumerate(chroms):
        start = chr_pos[c]
        end = chr_pos[chroms[i + 1]] if i + 1 < len(chroms) else X_dense.shape[1]
        if end - start >= 2:
            smoothed[:, start:end] = smooth_pyramidinal(X_dense[:, start:end], window_length=cfg.window_length)
        else:
            smoothed[:, start:end] = X_dense[:, start:end]
    _profile_block(profile, "09_smooth", t0, rss)

    # Step 11 — center cells (median)
    t0 = time.perf_counter(); rss = _rss_mb()
    smoothed = center_cells(smoothed, method="median")
    _profile_block(profile, "10_center", t0, rss)

    # Step 12 — subtract ref (2nd pass)
    t0 = time.perf_counter(); rss = _rss_mb()
    smoothed = subtract_reference(smoothed, ref_groups=ref_groups, use_bounds=cfg.ref_subtract_use_mean_bounds)
    _profile_block(profile, "11_subtract_ref_2", t0, rss)

    # Step 14 — invert log2 -> linear FC (R run() order: step 14 BEFORE step 16)
    t0 = time.perf_counter(); rss = _rss_mb()
    cnv_fc = invert_log2(smoothed)
    _profile_block(profile, "12_invert_log2", t0, rss)

    # Step 16 — prune outliers in LINEAR FC space (codex G3 Q5 fix).
    # R's remove_outliers_norm is called AFTER invert_log2, so bounds are
    # computed in linear FC space. We apply only to cnv_fc; the log-space
    # `smoothed` is left untouched so the invariant cnv_fc == invert_log2(smoothed)
    # holds up to the outlier clip applied on the linear side.
    if cfg.prune_outliers:
        t0 = time.perf_counter(); rss = _rss_mb()
        cnv_fc = prune_outliers(
            cnv_fc,
            method=cfg.outlier_method_bound,
            lower_bound=cfg.outlier_lower_bound,
            upper_bound=cfg.outlier_upper_bound,
        )
        _profile_block(profile, "13_outlier_prune", t0, rss)

    # Build result
    is_ref = np.zeros(adata.n_obs, dtype=bool)
    is_ref[ref_idx_all] = True
    cell_meta = pd.DataFrame({"is_reference": is_ref}, index=adata.obs_names)

    # Keep `smoothed` (float64) as companion so Phase 2 can consume it
    # without the float32 precision drop. Keep `cpm_matrix_f32` as the
    # R-parity hspike input (post-filter, post-normalize, pre-log2 full
    # matrix; consumed by calibrate_i6_emission at step 16). Both nulled
    # after Phase 2 or, when HMM is off, dropped before returning.
    result = InferCNVResult(
        chr_pos=chr_pos,
        bin_meta=bin_meta,
        cnv_matrix=smoothed.astype(np.float32),
        cnv_matrix_fc=cnv_fc.astype(np.float32),
        cell_meta=cell_meta,
        ref_counts_raw=ref_counts_raw,
        cnv_matrix_f64=smoothed,
        cpm_matrix_f32=cpm_matrix_f32,
        profile=profile,
    )

    # Phase 2 — HMM subclustering + state calls (only when cfg.HMM=True)
    if cfg.HMM:
        from pyinfercnv.pipeline_phase2 import run_phase2
        result = run_phase2(
            result, adata, config=cfg,
            reference_key=reference_key, reference_cat=reference_cat,
            random_state=cfg.random_state, profile=profile,
        )

    # Drop the transient companion fields once Phase 2 has consumed them.
    # When HMM is off we preserve them so a downstream caller can still
    # run ``run_phase2`` piecewise without rebuilding Phase 1.
    if cfg.HMM:
        result.cnv_matrix_f64 = None
        result.cpm_matrix_f32 = None

    # Phase 3 — BayesNet / mask_non_DE / denoise + R-faithful step 20 proxy.
    # Triggered when HMM is on (R runs step 20 whenever ``HMM=TRUE``,
    # ``inferCNV_ops.R:1463-1499``) OR any Phase 3 toggle is active.
    phase3_on = (
        bool(cfg.HMM)
        or cfg.BayesMaxPNormal > 0.0
        or bool(getattr(cfg, "mask_nonDE_genes", False))
        or bool(cfg.denoise)
    )
    if phase3_on:
        # Fail-loud guards for branches the new top-level integration newly
        # exposes (previously unreachable because ``infercnv()`` did not call
        # ``run_phase3`` at all).
        if cfg.denoise and getattr(cfg, "noise_logistic", False):
            raise NotImplementedError(
                "infercnv: denoise=True with noise_logistic=True is not implemented "
                "(R inferCNV_heatmap.R:2783, py ref_mean_sd.py:128). Set "
                "noise_logistic=False or implement the sigmoidal mask branch."
            )
        # BayesNet requires HMM (consumes hmm_states + hspike_calibration / i3
        # emission params, all populated only by Phase 2 when ``cfg.HMM=True``).
        if not cfg.HMM and cfg.BayesMaxPNormal > 0.0:
            raise ValueError(
                "infercnv: BayesMaxPNormal>0 requires HMM=True (BayesNet "
                "consumes hmm_states + hspike_calibration / i3 emission params "
                "from Phase 2)."
            )
        # mask_nonDE_genes is a Python-only HMM dependency: step 21 consumes
        # ``result.subclusters`` from Phase 2. R itself does NOT require HMM
        # for mask_non_DE (``inferCNV_mask_non_DE.R:35`` reads
        # observation_grouped_cell_indices / reference_grouped_cell_indices).
        if not cfg.HMM and bool(getattr(cfg, "mask_nonDE_genes", False)):
            raise ValueError(
                "infercnv: mask_nonDE_genes=True currently requires HMM=True in "
                "Python because step 21 consumes result.subclusters from Phase 2 "
                "(see pyinfercnv/mask_de/wilcoxon.py). R itself does not "
                "require HMM for mask_non_DE — this is a Python-side limitation."
            )
        from pyinfercnv.pipeline_phase3 import run_phase3
        result = run_phase3(result, config=cfg)

    if inplace:
        result.write_to_anndata(adata, key_added=key_added)
        return None
    return result
