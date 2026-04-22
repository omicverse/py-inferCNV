"""Orchestrator: run R infercnv + pyinfercnv on the 17-patient manifest.

Serial execution (infercnv internally uses num_threads=4). Resumable — skips
per-patient work whose summary markers already exist. Writes a top-level
run_all.log in the benchmark output root.

Usage:
    uv run python -m scripts.phase2_benchmark.run_all            # all patients
    uv run python -m scripts.phase2_benchmark.run_all --only Qian2020_Ovarian/11
    uv run python -m scripts.phase2_benchmark.run_all --subset smoke  # TNBC3 only

The smoke subset runs the single smallest patient (TNBC3, 532 cells) to
measure real wallclock before committing to the full 17.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.phase2_benchmark.dataset_manifest import (
    BENCHMARK_OUT, HMM_TYPES, PATIENTS, PYINFERCNV_ROOT, Patient, lookup,
)


SMOKE: tuple[Patient, ...] = (lookup("Gao2021_Breast", "TNBC3"),)  # smallest
# Representative 5-patient subset: one per cancer, picking the smallest/fastest.
SUBSET_5: tuple[Patient, ...] = (
    lookup("Gao2021_Breast", "TNBC3"),      # 532
    lookup("Lee2020_Colorectal", "SMC21"),  # 2084
    lookup("Kim2020_Lung", "P0034"),        # 2704
    lookup("Obradovic2021_Kidney", "Patient2"),  # 2932
    lookup("Qian2020_Ovarian", "13"),       # 2600
)


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


def _run(cmd: list[str], log_path: Path) -> int:
    _log("$ " + " ".join(str(c) for c in cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as lf:
        lf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  $ {' '.join(cmd)} ===\n")
        lf.flush()
        return subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)


def ensure_gene_order(benchmark_root: Path) -> Path:
    path = benchmark_root / "gene_order_hg38.tsv"
    if path.exists():
        return path
    _log(f"Generating {path}")
    rc = subprocess.call([
        sys.executable, "-m", "scripts.phase2_benchmark.export_gene_order",
        "--out", str(path),
    ])
    if rc != 0:
        raise RuntimeError("export_gene_order failed")
    return path


def prepare_patient(p: Patient, out_dir: Path) -> None:
    summary = out_dir / "prep_summary.json"
    if summary.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_csv = p.pycopykat_dir / "cells.csv"
    if not cells_csv.exists():
        raise FileNotFoundError(f"missing pycopykat input: {cells_csv}")
    rc = subprocess.call([
        sys.executable, "-m", "scripts.phase2_benchmark.prepare_annotations",
        "--cells-csv", str(cells_csv),
        "--sample", p.sample,
        "--out-dir", str(out_dir),
    ])
    if rc != 0:
        raise RuntimeError(f"prepare_annotations failed for {p.cancer}/{p.sample}")


def run_patient(p: Patient, gene_order: Path, benchmark_root: Path) -> None:
    out_dir = p.out_dir(benchmark_root)
    prepare_patient(p, out_dir)
    counts_tsv = p.pycopykat_dir / "counts.tsv"
    annotations = out_dir / "annotations_phase2.txt"
    log_path = out_dir / "run.log"

    for hmm_type in HMM_TYPES:
        r_out = out_dir / "r_out"
        py_out = out_dir / "py_out"

        # --- R driver -----------------------------------------------------
        r_summary = r_out / f"r_{hmm_type}_summary.txt"
        if not r_summary.exists():
            _log(f"[{p.cancer}/{p.sample}] R infercnv HMM={hmm_type} start")
            rc = _run([
                "Rscript", "scripts/phase2_benchmark/run_r_phase2.R",
                "--counts", str(counts_tsv),
                "--annotations", str(annotations),
                "--gene-order", str(gene_order),
                "--out-dir", str(r_out),
                "--hmm-type", hmm_type,
                "--num-threads", "4",
            ], log_path)
            if rc != 0:
                _log(f"[{p.cancer}/{p.sample}] R HMM={hmm_type} FAILED rc={rc}")
                continue
        else:
            _log(f"[{p.cancer}/{p.sample}] R HMM={hmm_type} already done")

        # --- Python driver ------------------------------------------------
        py_summary = py_out / f"py_{hmm_type}_summary.json"
        if not py_summary.exists():
            _log(f"[{p.cancer}/{p.sample}] pyinfercnv HMM={hmm_type} start")
            rc = _run([
                sys.executable, "-m", "scripts.phase2_benchmark.run_py_phase2",
                "--counts", str(counts_tsv),
                "--annotations", str(annotations),
                "--gene-order", str(gene_order),
                "--out-dir", str(py_out),
                "--hmm-type", hmm_type,
            ], log_path)
            if rc != 0:
                _log(f"[{p.cancer}/{p.sample}] py HMM={hmm_type} FAILED rc={rc}")
                continue
        else:
            _log(f"[{p.cancer}/{p.sample}] py HMM={hmm_type} already done")

        # --- Compare ------------------------------------------------------
        compare_out = out_dir / f"compare_{hmm_type}.json"
        if not compare_out.exists():
            rc = _run([
                sys.executable, "-m", "scripts.phase2_benchmark.compare_py_vs_r",
                "--py-state", str(py_out / f"step17_hmm_{hmm_type}.tsv"),
                "--r-state", str(r_out / f"step17_hmm_{hmm_type}.tsv"),
                "--py-subclusters", str(py_out / f"step15_subclusters_{hmm_type}.tsv"),
                "--r-subclusters", str(r_out / f"step15_subclusters_{hmm_type}.tsv"),
                "--hmm-type", hmm_type,
                "--out-dir", str(out_dir),
                "--py-summary", str(py_summary),
                "--r-summary", str(r_summary),
            ], log_path)
            if rc != 0:
                _log(f"[{p.cancer}/{p.sample}] compare HMM={hmm_type} FAILED rc={rc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", default=[],
                    help="Scope to <Cancer>/<Patient> (repeatable).")
    ap.add_argument("--subset", choices=["smoke", "5", "all"], default="all")
    args = ap.parse_args()

    os.chdir(PYINFERCNV_ROOT)
    BENCHMARK_OUT.mkdir(parents=True, exist_ok=True)
    gene_order = ensure_gene_order(BENCHMARK_OUT)

    if args.only:
        scope = []
        for spec in args.only:
            c, s = spec.split("/")
            scope.append(lookup(c, s))
    elif args.subset == "smoke":
        scope = list(SMOKE)
    elif args.subset == "5":
        scope = list(SUBSET_5)
    else:
        scope = list(PATIENTS)

    _log(f"scope: {len(scope)} patients")
    t0 = time.time()
    for i, p in enumerate(scope, 1):
        _log(f"=== [{i}/{len(scope)}] {p.cancer}/{p.sample} ===")
        try:
            run_patient(p, gene_order, BENCHMARK_OUT)
        except Exception as e:
            _log(f"[{p.cancer}/{p.sample}] EXCEPTION: {e}")
    _log(f"total wallclock: {time.time() - t0:.1f}s over {len(scope)} patients")
    return 0


if __name__ == "__main__":
    sys.exit(main())
