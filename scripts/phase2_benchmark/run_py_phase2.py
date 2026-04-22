"""Run pyinfercnv Phase 2 on a single patient + HMM_type, dump parity TSVs.

Inputs:
    counts.tsv (gene x cell, pycopykat-sliced)
    annotations_phase2.txt (2-col, cell_id \\t annotation)
    gene_order_hg38.tsv (gene / chr / start / end, no header)

Outputs (under <out-dir>/):
    step17_hmm_<type>.tsv         genes x cells integer state matrix
    step15_subclusters_<type>.tsv cell_id \\t subcluster
    py_<type>_summary.json         timing + shapes + profile

Resumable: if py_<type>_summary.json exists, skip (delete to re-run).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pyinfercnv import InferCNVConfig, infercnv
from pyinfercnv.pipeline import _build_chromosome_layout
from pyinfercnv.preprocess import filter_low_expression_genes


def _pipeline_gene_order(
    adata: ad.AnnData, cutoff: float, min_cells_per_gene: int,
    chr_exclude: tuple[str, ...],
) -> list[str]:
    """Reconstruct the gene labels aligned with result.hmm_states columns.

    Mirrors tests/test_r_parity.py::_get_pipeline_gene_names — runs the same
    filter + chromosome layout the pipeline applies internally.
    """
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(np.asarray(X, dtype=np.float32))
    keep = filter_low_expression_genes(
        X, cutoff=cutoff, min_cells_per_gene=min_cells_per_gene,
    )
    var_kept = adata.var.iloc[np.where(keep)[0]].copy()
    _, gene_perm = _build_chromosome_layout(var_kept, chr_exclude)
    return [var_kept.index[i] for i in gene_perm]


def _load_adata(
    counts_path: Path, annotations_path: Path, gene_order_path: Path,
) -> ad.AnnData:
    # counts.tsv is gene x cell with header row = cell ids, col0 = "gene".
    counts = pd.read_csv(counts_path, sep="\t", index_col=0)
    counts.index.name = "gene"

    ann = pd.read_csv(
        annotations_path, sep="\t", header=None, names=["cell_id", "annotation"],
    )
    cell_ids = ann["cell_id"].tolist()

    missing = set(cell_ids) - set(counts.columns)
    if missing:
        raise ValueError(
            f"{len(missing)} annotated cells missing from counts.tsv, e.g. "
            f"{sorted(missing)[:3]}"
        )
    counts = counts[cell_ids]  # align cell order + drop unlabeled

    var = pd.DataFrame(index=counts.index)
    gene_order = pd.read_csv(
        gene_order_path, sep="\t", header=None,
        names=["gene_symbol", "chromosome", "start", "end"],
    )
    gene_order = gene_order.drop_duplicates(subset=["gene_symbol"], keep="first")
    gene_order = gene_order.set_index("gene_symbol")
    var = var.join(gene_order, how="left")

    # Drop genes without chromosomal coordinates (needed by pyinfercnv)
    keep = var["chromosome"].notna()
    var = var.loc[keep].copy()
    counts = counts.loc[var.index]

    X = counts.T.values.astype(np.float32)  # cells x genes
    obs = pd.DataFrame({"annotation": ann["annotation"].values}, index=cell_ids)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata


def run_one(
    counts_path: Path, annotations_path: Path, gene_order_path: Path,
    out_dir: Path, hmm_type: str, cutoff: float,
) -> dict:
    t_load = time.perf_counter()
    adata = _load_adata(counts_path, annotations_path, gene_order_path)
    load_s = time.perf_counter() - t_load

    all_labels = adata.obs["annotation"].unique().tolist()
    ref_labels = [x for x in all_labels if not x.startswith("malignant_")]

    # Config mirrors tests/r_reference.R exactly — this is the only setting
    # combination with validated oligodendroglioma tier-3.5 parity (i3=0.976,
    # i6=0.968). Deviations (analysis_mode="subclusters", PCA-leiden, etc.)
    # produce R-Py subcluster divergence on real-world data.
    cfg = InferCNVConfig(
        cutoff=cutoff,
        HMM=True,
        HMM_type=hmm_type,
        BayesMaxPNormal=0.0,
        tumor_subcluster_partition_method="leiden",
        cluster_by_groups=True,
        num_threads=4,
        counts_layer=None,  # use adata.X as counts
    )
    # Keep a counts layer so the pipeline's extract_counts (counts_layer=None
    # path) reads adata.X directly as integer-valued floats.
    adata.layers["counts"] = adata.X.copy()

    t0 = time.perf_counter()
    result = infercnv(
        adata, config=cfg, reference_key="annotation", reference_cat=ref_labels,
        inplace=False,
    )
    elapsed = time.perf_counter() - t0

    # --- step17 HMM state matrix (genes x cells) ---
    state_matrix = result.hmm_states_i3 if hmm_type == "i3" else result.hmm_states
    if state_matrix is None:
        raise RuntimeError(
            f"pyinfercnv result has no hmm_states for HMM_type={hmm_type}"
        )
    # state_matrix expected shape: (n_cells, n_genes) per pyinfercnv convention.
    # R dumps genes x cells — transpose to match.
    arr = np.asarray(state_matrix)
    if arr.ndim != 2:
        raise RuntimeError(f"hmm_states ndim {arr.ndim} != 2")
    if arr.shape[0] == adata.n_obs:
        arr = arr.T  # -> genes x cells
    gene_idx = result.cnv_matrix.var_names if hasattr(result.cnv_matrix, "var_names") else adata.var_names
    # Use adata.var_names (post-filter pipeline uses internal order; for parity
    # we rely on pyinfercnv returning state_matrix aligned with the pipeline's
    # final gene axis — stored in result via chr_pos layout).
    n_genes_out, n_cells_out = arr.shape
    if n_cells_out != adata.n_obs:
        raise RuntimeError(
            f"state_matrix cell dim {n_cells_out} != adata.n_obs {adata.n_obs}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / f"step17_hmm_{hmm_type}.tsv"
    gene_labels = _pipeline_gene_order(
        adata, cutoff=cfg.cutoff,
        min_cells_per_gene=cfg.min_cells_per_gene,
        chr_exclude=cfg.chr_exclude,
    )
    if len(gene_labels) != n_genes_out:
        raise RuntimeError(
            f"pipeline_gene_order len {len(gene_labels)} != n_genes_out {n_genes_out}"
        )
    cell_labels = list(adata.obs_names)
    pd.DataFrame(arr, index=gene_labels, columns=cell_labels).to_csv(
        state_path, sep="\t", index_label=""
    )

    # --- step15 subclusters (cell_id, subcluster) ---
    subc = result.subclusters
    if subc is None:
        raise RuntimeError("pyinfercnv result has no subclusters")
    # result.subclusters is documented as np.ndarray shape (n_cells,) int32.
    arr_s = np.asarray(subc)
    if arr_s.shape[0] != adata.n_obs:
        raise RuntimeError(
            f"subclusters length {arr_s.shape[0]} != n_cells {adata.n_obs}"
        )
    sub_df = pd.DataFrame(
        {"cell_id": list(adata.obs_names), "subcluster": arr_s.astype(str)}
    )
    group_path = out_dir / f"step15_subclusters_{hmm_type}.tsv"
    sub_df.to_csv(group_path, sep="\t", index=False)

    summary = {
        "hmm_type": hmm_type,
        "n_cells": int(adata.n_obs),
        "n_genes_in": int(adata.n_vars),
        "n_genes_out": int(n_genes_out),
        "load_s": load_s,
        "infercnv_s": elapsed,
        "ref_groups": ref_labels,
        "cutoff": cutoff,
        "state_path": str(state_path),
        "group_path": str(group_path),
    }
    (out_dir / f"py_{hmm_type}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--gene-order", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hmm-type", choices=["i6", "i3"], required=True)
    ap.add_argument("--cutoff", type=float, default=0.1)
    args = ap.parse_args()

    skip_marker = args.out_dir / f"py_{args.hmm_type}_summary.json"
    if skip_marker.exists():
        print(f"[run_py_phase2] skip (already done): {skip_marker}")
        return 0

    summary = run_one(
        args.counts, args.annotations, args.gene_order, args.out_dir,
        args.hmm_type, args.cutoff,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
