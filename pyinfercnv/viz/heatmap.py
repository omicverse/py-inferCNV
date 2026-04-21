"""matplotlib-only CNV heatmaps. NO omicverse import (spec section 7)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyinfercnv.viz._palette import CHROM_STRIP_COLORS, OV_RDBU_CMAP

if TYPE_CHECKING:
    from pyinfercnv.result import InferCNVResult


def plot_cnv_heatmap(
    result: "InferCNVResult",
    *,
    ax: Axes | None = None,
    output: str | Path | None = None,
    vmin: float = -0.5,
    vmax: float = 0.5,
    cmap=OV_RDBU_CMAP,
    chrom_strip: bool = True,
) -> Axes:
    """Render the log2(FC) CNV matrix as a heatmap. Cells on y, bins on x."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, max(4, result.n_cells / 50)))
    else:
        fig = ax.figure

    im = ax.imshow(
        result.cnv_matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    fig.colorbar(im, ax=ax, label="log2(FC)", fraction=0.02, pad=0.01)
    ax.set_xlabel("Bins (sorted by chromosome, start)")
    ax.set_ylabel("Cells")

    if chrom_strip:
        chr_starts = list(result.chr_pos.values())
        chr_names = list(result.chr_pos.keys())
        for i, (c, start) in enumerate(zip(chr_names, chr_starts)):
            end = chr_starts[i + 1] if i + 1 < len(chr_starts) else result.n_bins
            color = CHROM_STRIP_COLORS[i % 2]
            ax.axvspan(start, end, ymin=1.00, ymax=1.02, color=color, clip_on=False, alpha=0.8)

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    return ax


def plot_cnv_heatmap_compare(py: "InferCNVResult", r: "InferCNVResult", *, output=None) -> Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plot_cnv_heatmap(py, ax=ax1)
    ax1.set_title("pyinfercnv")
    plot_cnv_heatmap(r, ax=ax2)
    ax2.set_title("R infercnv")
    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    return fig


def plot_cnv_delta(py: "InferCNVResult", r: "InferCNVResult", *, ax=None, output=None) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    delta = py.cnv_matrix - r.cnv_matrix
    im = ax.imshow(delta, aspect="auto", cmap=OV_RDBU_CMAP, vmin=-0.1, vmax=0.1, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, label="py - R (log2FC)")
    ax.set_title("Parity delta")
    if output:
        ax.figure.savefig(output, dpi=150, bbox_inches="tight")
    return ax
