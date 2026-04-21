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
