"""pyinfercnv on one task, mirroring 02_run_infercnv.R.

Usage: python 03_run_pyinfercnv.py <task_dir> <gene_order_file> <out_dir> [seed] [cutoff]
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

import pyinfercnv
from pyinfercnv.config import InferCNVConfig
from pyinfercnv.pipeline import infercnv


def main():
    td, gof, od = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    cutoff = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
    if (td / "cutoff.txt").exists():
        cutoff = float((td / "cutoff.txt").read_text().split()[0])
    if (td / "gene_order.tsv").exists():
        gof = str(td / "gene_order.tsv")
    refs = ((td / "ref_groups.txt").read_text().splitlines() if (td / "ref_groups.txt").exists()
            else ["normal"])
    od.mkdir(parents=True, exist_ok=True)
    m = sp.csr_matrix(scipy.io.mmread(td / "counts.mtx").T, dtype=np.float64)       # cells x genes
    genes = (td / "genes.txt").read_text().split("\n")[:-1]
    cells = (td / "cells.txt").read_text().split()
    ann = pd.read_csv(td / "annotations.txt", sep="\t", header=None, index_col=0)[1]
    go = pd.read_csv(gof, sep="\t", header=None, index_col=0, names=["gene", "chromosome", "start", "end"])
    go.index.name = None
    adata = ad.AnnData(m)
    adata.layers["counts"] = m
    adata.var_names, adata.obs_names = genes, cells
    adata.var = go.reindex(adata.var_names)
    adata.obs["annotation"] = ann.reindex(cells).to_numpy()
    cfg = InferCNVConfig(cutoff=cutoff, denoise=True, HMM=True, BayesMaxPNormal=0.0, cluster_by_groups=True,
                         num_threads=1, random_state=seed)
    t0 = time.perf_counter()
    r = infercnv(adata, config=cfg, reference_key="annotation", reference_cat=refs, inplace=False)
    el = time.perf_counter() - t0
    g = np.asarray(r.gene_names)
    for tag, mat, off in [("14", r.cnv_matrix_fc, 0), ("17", r.hmm_states, 1), ("20", r.hmm_proxy_matrix, 0),
                          ("22", r.denoised_matrix, 0)]:
        (np.asarray(mat, dtype=np.float64) + off).tofile(od / f"{tag}.bin")   # cells x genes rows = R column-major
        (od / f"{tag}_genes.txt").write_text("\n".join(g) + "\n")
        (od / f"{tag}_cells.txt").write_text("\n".join(cells) + "\n")
    pd.DataFrame({"cell": cells, "subcluster": r.subclusters}).to_csv(od / "subclusters.tsv", sep="\t", index=False)
    (od / "result.json").write_text(json.dumps({"engine": "pyinfercnv", "version": pyinfercnv.__version__,
        "seed": seed, "cutoff": cutoff, "dim": [len(g), len(cells)], "seconds": el}))
    print(td.name, f"{len(g)} genes x {len(cells)} cells, {el:.1f} s", flush=True)


if __name__ == "__main__":
    main()
