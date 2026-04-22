"""Per-patient py-vs-R parity metrics for Phase 2 outputs.

Compares:
    step17_hmm_<type>.tsv   py vs R: mean per-cell Jaccard of non-neutral bins
    step15_subclusters_<type>.tsv  py vs R: ARI over the tumor-cell partition

Writes a small JSON summary into <out-dir>/compare_<hmm_type>.json.

Alignment:
    genes — intersection by gene symbol (both files have gene labels as index)
    cells — intersection by cell id (R drops some cells during filter, py drops
            some during annotation prep; intersection is the common support).

Neutral state:
    i6 — state 3 (R coding: 1=loss, 2=copy, 3=neutral, 4=gain, 5=amp, 6=high-amp)
          pyinfercnv follows same coding per pipeline_phase2 docstring.
    i3 — state 2 (1=loss, 2=neutral, 3=gain).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


# State coding convention (ground truth from observation + pipeline_phase2.py):
#   R output: 1-indexed. i3 = {1=loss, 2=neutral, 3=gain}. i6 = {1..6, 3=neutral}.
#   Python output: 0-indexed. i3 = {0=loss, 1=neutral, 2=gain}. i6 = {0..5, 2=neutral}.
# Callers pass side="py" or side="r"; auto-detection on min(mat) is unsafe
# because a patient whose HMM never visits state 0 (e.g. no deep-loss bin on
# i6) has py_min=1 and would be misclassified as 1-indexed. Hit on
# Gao2021_Breast/DCIS1 i6 — Jaccard collapsed from ~0.86 (true) to 0.27
# (misaligned neutral). Fix: explicit side lookup, no min() inference.
_R_NEUTRAL = {"i6": 3, "i3": 2}
_PY_NEUTRAL = {"i6": 2, "i3": 1}


def _neutral_for(side: str, hmm_type: str) -> int:
    """Return neutral-state integer for `side` ∈ {"py", "r"} and hmm_type."""
    if side == "py":
        return _PY_NEUTRAL[hmm_type]
    if side == "r":
        return _R_NEUTRAL[hmm_type]
    raise ValueError(f"side must be 'py' or 'r', got {side!r}")


def _load_tsv_matrix(path: Path) -> pd.DataFrame:
    """Load genes × cells TSV (R write.table format: col.names=NA)."""
    return pd.read_csv(path, sep="\t", index_col=0)


def compare_jaccard(py_tsv: Path, r_tsv: Path, hmm_type: str) -> dict:
    py = _load_tsv_matrix(py_tsv)
    r = _load_tsv_matrix(r_tsv)

    shared_genes = sorted(set(py.index) & set(r.index))
    shared_cells = sorted(set(py.columns) & set(r.columns))
    if not shared_genes or not shared_cells:
        return {
            "n_genes_shared": len(shared_genes),
            "n_cells_shared": len(shared_cells),
            "mean_jaccard": None,
            "error": "empty intersection",
        }

    py_m = py.loc[shared_genes, shared_cells].to_numpy(dtype=np.int8).T  # cells × genes
    r_m = r.loc[shared_genes, shared_cells].to_numpy(dtype=np.int8).T

    py_neutral = _neutral_for("py", hmm_type)
    r_neutral = _neutral_for("r", hmm_type)
    jaccards = []
    for i in range(py_m.shape[0]):
        py_nn = set(int(x) for x in np.where(py_m[i] != py_neutral)[0])
        r_nn = set(int(x) for x in np.where(r_m[i] != r_neutral)[0])
        union = py_nn | r_nn
        jaccards.append(1.0 if not union else len(py_nn & r_nn) / len(union))

    # Cells whose py AND R both have no non-neutral bins -> auto-1.0 (trivially
    # agree), which can inflate the metric when a patient is near-neutral.
    nonzero_mask = [
        bool((py_m[i] != py_neutral).any() or (r_m[i] != r_neutral).any())
        for i in range(py_m.shape[0])
    ]
    n_nonzero = int(sum(nonzero_mask))
    mean_j = float(np.mean(jaccards))
    mean_j_nonzero = float(
        np.mean([jaccards[i] for i, m in enumerate(nonzero_mask) if m])
    ) if n_nonzero else None

    return {
        "n_genes_py": int(py.shape[0]),
        "n_genes_r": int(r.shape[0]),
        "n_genes_shared": len(shared_genes),
        "n_cells_py": int(py.shape[1]),
        "n_cells_r": int(r.shape[1]),
        "n_cells_shared": len(shared_cells),
        "n_cells_nonzero": n_nonzero,
        "mean_jaccard": mean_j,
        "mean_jaccard_nonzero_only": mean_j_nonzero,
    }


def compare_subclusters(py_tsv: Path, r_tsv: Path) -> dict:
    py = pd.read_csv(py_tsv, sep="\t")
    r = pd.read_csv(r_tsv, sep="\t")
    # Both have columns cell_id, subcluster.
    shared = (
        py.set_index("cell_id")
          .join(r.set_index("cell_id"), how="inner", lsuffix="_py", rsuffix="_r")
          .reset_index()
    )
    if shared.empty:
        return {"n_cells_shared": 0, "ari": None}
    return {
        "n_cells_py": int(len(py)),
        "n_cells_r": int(len(r)),
        "n_cells_shared": int(len(shared)),
        "ari": float(adjusted_rand_score(
            shared["subcluster_py"].astype(str), shared["subcluster_r"].astype(str)
        )),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--py-state", type=Path, required=True)
    ap.add_argument("--r-state", type=Path, required=True)
    ap.add_argument("--py-subclusters", type=Path, required=True)
    ap.add_argument("--r-subclusters", type=Path, required=True)
    ap.add_argument("--hmm-type", choices=["i6", "i3"], required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--py-summary", type=Path, required=False, default=None,
                    help="Optional path to py_<type>_summary.json for timing.")
    ap.add_argument("--r-summary", type=Path, required=False, default=None,
                    help="Optional path to r_<type>_summary.txt for timing.")
    args = ap.parse_args()

    j = compare_jaccard(args.py_state, args.r_state, args.hmm_type)
    s = compare_subclusters(args.py_subclusters, args.r_subclusters)

    timings = {}
    if args.py_summary and args.py_summary.exists():
        py_sum = json.loads(args.py_summary.read_text())
        timings["py_s"] = py_sum.get("infercnv_s")
        timings["py_load_s"] = py_sum.get("load_s")
    if args.r_summary and args.r_summary.exists():
        for line in args.r_summary.read_text().splitlines():
            if line.startswith("elapsed_s"):
                timings["r_s"] = float(line.split("=")[1].strip())

    if timings.get("py_s") and timings.get("r_s"):
        timings["speedup"] = timings["r_s"] / timings["py_s"]

    out = {
        "hmm_type": args.hmm_type,
        "jaccard": j,
        "subclusters": s,
        "timings": timings,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"compare_{args.hmm_type}.json").write_text(
        json.dumps(out, indent=2)
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
