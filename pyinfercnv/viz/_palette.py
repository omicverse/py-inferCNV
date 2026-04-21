"""Colour palette aligned with omicverse defaults — hardcoded so pyinfercnv core
never imports omicverse (standalone constraint, spec section 7).
"""
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap


OV_RDBU_COLORS = [
    "#2166AC",
    "#67A9CF",
    "#D1E5F0",
    "#F7F7F7",
    "#FDDBC7",
    "#EF8A62",
    "#B2182B",
]
OV_RDBU_CMAP = LinearSegmentedColormap.from_list("ov_rdbu", OV_RDBU_COLORS, N=256)

CHROM_STRIP_COLORS = ["#333333", "#888888"]
