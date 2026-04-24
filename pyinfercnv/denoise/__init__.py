"""Phase 3 — reference-mean±sd denoising of the CNV matrix.

R source: ``R/inferCNV_ops.R:2302-2346`` (``clear_noise_via_ref_mean_sd``)
and ``R/inferCNV_ops.R:2232-2264`` (``clear_noise`` with an explicit
``noise_filter`` threshold instead of ref-mean-sd). Wired at
``inferCNV_ops.R:1560-1615`` (step 22, guarded by ``denoise=TRUE``).

**Note**: ``R/noise_reduction.R`` contains ``apply_median_filtering``,
a *different* denoise step (not ported in this skeleton; see plan §5 for
disposition).

Agent B3 implementation landed: :func:`denoise_by_ref_mean_sd` is the
pure-Python bit-exact port; :func:`_step22_denoise` is the orchestrator
entry point that wraps it using :class:`InferCNVResult` / :class:`InferCNVConfig`.
See ``docs/superpowers/plans/2026-04-24-phase3-start.md`` §4.
"""
from __future__ import annotations

from pyinfercnv.denoise.ref_mean_sd import _step22_denoise, denoise_by_ref_mean_sd

__all__ = ["denoise_by_ref_mean_sd", "_step22_denoise"]
