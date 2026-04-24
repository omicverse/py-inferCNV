"""Phase 3 — mask non-DE genes between tumour subclusters and references.

R source: ``R/inferCNV_mask_non_DE.R``. Wired at
``inferCNV_ops.R:1509-1552`` (step 21, guarded by ``mask_nonDE_genes``).
"""
from __future__ import annotations

from pyinfercnv.mask_de.wilcoxon import _step21_mask_non_DE, mask_non_DE_genes

__all__ = ["mask_non_DE_genes", "_step21_mask_non_DE"]
