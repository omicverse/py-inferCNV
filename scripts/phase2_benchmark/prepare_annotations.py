"""Transform pycopykat's cells.csv into a 2-col annotations file for R infercnv.

Policy (see dataset_manifest.py docstring):
    Malignant cells   -> "malignant_<patient>"
    Non-empty non-Malignant -> keep cell_type verbatim as ref group label
    Empty cell_type   -> dropped (cannot assign reliably)

Writes:
    <patient_out_dir>/annotations_phase2.txt   2-col, no header, tab-separated
    <patient_out_dir>/cell_id_order.txt         cell_id list after filtering
    <patient_out_dir>/prep_summary.json         n_cells_obs / n_cells_ref / ref_groups
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def prepare(cells_csv: Path, sample: str, out_dir: Path) -> dict:
    df = pd.read_csv(cells_csv)
    if "cell_name" not in df.columns or "cell_type" not in df.columns:
        raise ValueError(f"{cells_csv}: expected 'cell_name' and 'cell_type' columns")

    # Drop empty cell_type (pandas reads empty as NaN)
    df = df.dropna(subset=["cell_type"]).copy()
    df = df[df["cell_type"].str.strip() != ""].copy()

    def _annotate(ct: str) -> str:
        return f"malignant_{sample}" if ct == "Malignant" else ct

    df["annotation"] = df["cell_type"].map(_annotate)

    tumor_label = f"malignant_{sample}"
    ref_labels = sorted(set(df["annotation"]) - {tumor_label})

    # R infercnv Phase 2 HMM requires ≥2 ref cells per group; drop singleton
    # ref groups (rare but TNBC1 has T_cell=1, Endothelial=1). Collapse them
    # into a single catch-all "Other_normal" if any exist — infercnv permits
    # multi-group refs but singletons break subcluster leiden.
    ref_counts = df[df.annotation.isin(ref_labels)].annotation.value_counts()
    singletons = ref_counts[ref_counts < 2].index.tolist()
    if singletons:
        df.loc[df.annotation.isin(singletons), "annotation"] = "Other_normal"
        ref_labels = sorted(
            (set(ref_labels) - set(singletons)) | ({"Other_normal"} if singletons else set())
        )

    if not ref_labels:
        raise ValueError(
            f"{cells_csv}: sample {sample} has no reference cells after filtering"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    ann_path = out_dir / "annotations_phase2.txt"
    df[["cell_name", "annotation"]].to_csv(
        ann_path, sep="\t", header=False, index=False
    )

    order_path = out_dir / "cell_id_order.txt"
    df["cell_name"].to_csv(order_path, header=False, index=False)

    summary = {
        "sample": sample,
        "n_cells_total": int(len(df)),
        "n_cells_obs": int((df.annotation == tumor_label).sum()),
        "n_cells_ref": int(df.annotation.isin(ref_labels).sum()),
        "tumor_label": tumor_label,
        "ref_groups": ref_labels,
        "ref_group_counts": {
            lbl: int((df.annotation == lbl).sum()) for lbl in ref_labels
        },
        "singletons_collapsed": singletons,
    }
    (out_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells-csv", type=Path, required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    summary = prepare(args.cells_csv, args.sample, args.out_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
