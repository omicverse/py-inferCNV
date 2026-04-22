"""InferCNVConfig — dataclass mirroring R `infercnv::run()` kwargs.

Preserves R's camelCase/dotted names verbatim (# noqa: N803/N815) so end users
can translate R scripts 1-to-1. Phase 1 uses a subset; Phase 2/3 extend this
dataclass in place without renaming existing fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SmoothMethod = Literal["pyramidinal", "runmeans", "coordinates"]
HMMType = Literal["i6", "i3"]
AnalysisMode = Literal["subclusters", "samples", "cells"]


@dataclass
class InferCNVConfig:  # noqa: N801
    """Configuration for infercnv() pipeline. R run() kwargs preserved verbatim."""

    # --- gene filtering (step 2) ---
    cutoff: float = 1.0
    min_cells_per_gene: int = 3

    # --- smoothing (step 10) ---
    window_length: int = 101
    smooth_method: SmoothMethod = "pyramidinal"

    # --- reference handling (steps 8, 12) ---
    num_ref_groups: int | None = None
    ref_subtract_use_mean_bounds: bool = True

    # --- clip (step 9) ---
    max_centered_threshold: float | str | None = 3.0

    # --- z-score scale (step 5, optional) ---
    scale_data: bool = False

    # --- outlier prune (step 16) ---
    prune_outliers: bool = False
    outlier_method_bound: str = "average_bound"
    outlier_lower_bound: float | None = None
    outlier_upper_bound: float | None = None

    # --- I/O contract ---
    counts_layer: str | None = "counts"
    chr_exclude: tuple[str, ...] = ("chrX", "chrY", "chrM")

    # --- Phase 2+ placeholders (declared now so API stable across phases) ---
    HMM: bool = False  # noqa: N815
    HMM_type: HMMType = "i6"  # noqa: N815
    HMM_transition_prob: float = 1e-6  # noqa: N815
    HMM_i3_pval: float = 0.05  # noqa: N815
    BayesMaxPNormal: float = 0.5  # noqa: N815
    analysis_mode: AnalysisMode = "subclusters"

    # --- Phase 2 subcluster partition ---
    tumor_subcluster_partition_method: str = "leiden"  # noqa: N815
    # R cluster_by_groups=TRUE: run leiden independently per observation/
    # reference group (each annotation category is its own leiden input
    # and its own set of subcluster IDs). False => leiden on all non-ref
    # cells pooled, and ref cells get a single sentinel label. Default
    # matches R behaviour.
    cluster_by_groups: bool = True

    # --- Phase 2 Leiden stability knobs (Track A closeout, default off) ---
    # n_seeds > 1 runs Leiden N times with seeds (random_state,
    # random_state+1, …) and returns the highest-CPM partition (``rbest``).
    # Default 1 = R-parity behaviour.
    tumor_subcluster_n_seeds: int = 1  # noqa: N815
    # min_subcluster_size: if set, merge clusters with < N cells into
    # their nearest non-small cluster via KNN majority vote. Default
    # None = off; R has no such cleanup. Non-CPM-optimal by design.
    tumor_subcluster_min_size: int | None = None  # noqa: N815

    # --- misc ---
    lfc_clip: float = 3.0
    denoise: bool = False
    noise_filter: float | None = None
    sd_amplifier: float = 1.5
    debug: bool = False
    num_threads: int = 4

    def validate(self) -> None:
        if self.cutoff < 0:
            raise ValueError(f"cutoff must be >= 0, got {self.cutoff}")
        if self.min_cells_per_gene < 0:
            raise ValueError(f"min_cells_per_gene must be >= 0, got {self.min_cells_per_gene}")
        if self.window_length < 3 or self.window_length % 2 == 0:
            raise ValueError(f"window_length must be odd and >= 3, got {self.window_length}")
        if self.smooth_method not in ("pyramidinal", "runmeans", "coordinates"):
            raise ValueError(f"smooth_method invalid: {self.smooth_method}")
        if isinstance(self.max_centered_threshold, (int, float)) and self.max_centered_threshold <= 0:
            raise ValueError("max_centered_threshold must be > 0 or 'auto' or None")
        if not 0.0 <= self.BayesMaxPNormal <= 1.0:
            raise ValueError(f"BayesMaxPNormal must be in [0,1], got {self.BayesMaxPNormal}")
        # G1 patch P8: strictest bound for both i6 (K=6) and i3 (K=3).
        # Off-diagonal mass in `.get_HMM` is (K-1)*t; requiring (K-1)*t < 1
        # for K=6 yields t < 1/5 = 0.2.
        if self.HMM_transition_prob <= 0.0 or self.HMM_transition_prob >= 0.2:
            raise ValueError(
                "HMM_transition_prob must be in (0, 1/(K-1)=0.2), "
                f"got {self.HMM_transition_prob}"
            )
        if self.tumor_subcluster_partition_method not in ("leiden", "random_trees", "qnorm"):
            raise ValueError(
                f"tumor_subcluster_partition_method must be one of "
                f"('leiden', 'random_trees', 'qnorm'), "
                f"got {self.tumor_subcluster_partition_method!r}"
            )
        if self.analysis_mode != "subclusters":
            # "samples" and "cells" modes are Phase 3 work; fail loudly
            # rather than silently accept and quietly run subclusters.
            raise ValueError(
                f"analysis_mode={self.analysis_mode!r} not yet supported; "
                "only 'subclusters' is implemented in Phase 2."
            )
        if self.tumor_subcluster_n_seeds < 1:
            raise ValueError(
                f"tumor_subcluster_n_seeds must be >= 1, got {self.tumor_subcluster_n_seeds}"
            )
        if (
            self.tumor_subcluster_min_size is not None
            and self.tumor_subcluster_min_size < 1
        ):
            raise ValueError(
                "tumor_subcluster_min_size must be >= 1 or None, "
                f"got {self.tumor_subcluster_min_size}"
            )
