"""pyinfercnv CLI entrypoint — typer app."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from anndata import read_h5ad

from pyinfercnv.config import InferCNVConfig
from pyinfercnv.pipeline import infercnv


app = typer.Typer(no_args_is_help=True, help="pyinfercnv — R-parity Python port of inferCNV.")


@app.command("run-h5ad")
def run_h5ad(
    input: Path = typer.Option(..., help="Path to input .h5ad"),
    output: Path = typer.Option(..., help="Path to write annotated .h5ad"),
    reference_key: Optional[str] = typer.Option(None, help="adata.obs column with cell-type labels."),
    reference_cat: Optional[str] = typer.Option(None, help="Comma-separated categories treated as normal."),
    counts_layer: str = typer.Option("counts", help="Layer name for raw counts."),
    window_length: int = typer.Option(101),
    cutoff: float = typer.Option(1.0),
    exclude_chromosomes: str = typer.Option("chrX,chrY,chrM"),
) -> None:
    """Run pyinfercnv Phase 1 pipeline on an h5ad and write annotated copy."""
    adata = read_h5ad(input)
    cfg = InferCNVConfig(counts_layer=counts_layer, window_length=window_length, cutoff=cutoff)
    ref_cat = reference_cat.split(",") if reference_cat else None
    excl = tuple(exclude_chromosomes.split(","))
    infercnv(
        adata,
        config=cfg,
        reference_key=reference_key,
        reference_cat=ref_cat,
        exclude_chromosomes=excl,
    )
    adata.write_h5ad(output)
    typer.echo(f"[pyinfercnv] wrote {output}")


@app.command("version")
def version() -> None:
    """Print pyinfercnv version."""
    from pyinfercnv import __version__
    typer.echo(__version__)


if __name__ == "__main__":
    app()
