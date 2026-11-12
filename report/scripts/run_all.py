"""Benchmark driver: every task x repeat (seed 1; repeated for timing) for inferCNV (R) and pyinfercnv, each run pinned to its own
CPU core (taskset), so both are single-threaded and timed under the same load.

Usage: python run_all.py <tasks_dir> <out_dir> <rscript> <python> [n_parallel] [repeats, e.g. 1,2] [engines]
"""
from __future__ import annotations

import os
import pathlib
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
tasks_dir, out_dir, rscript, python = map(str, sys.argv[1:5])
n_par = int(sys.argv[5]) if len(sys.argv) > 5 else 8
seeds = [int(s) for s in (sys.argv[6] if len(sys.argv) > 6 else "1,2").split(",")]
engines = (sys.argv[7] if len(sys.argv) > 7 else "infercnv,pyinfercnv").split(",")
def idle_cores(n):
    """The n allowed cores with the least load over one second (from /proc/stat)."""
    def snap():
        out = {}
        for line in open("/proc/stat"):
            if line.startswith("cpu") and line[3].isdigit():
                f = line.split()
                v = list(map(int, f[1:]))
                out[int(f[0][3:])] = (sum(v), v[3] + v[4])
        return out
    a = snap()
    time.sleep(1)
    b = snap()
    allowed = sorted(os.sched_getaffinity(0))
    busy = {c: 1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in allowed}
    return sorted(allowed, key=lambda c: (busy[c], c))[:n]


cores = queue.Queue()
for c in idle_cores(n_par):
    cores.put(c)
env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           OMP_WAIT_POLICY="PASSIVE", PYTHONWARNINGS="ignore")

jobs = []
GENE_ORDER = str(pathlib.Path(tasks_dir).parent / "gene_order_gencode_v24.tsv")
tasks = sorted(p for p in pathlib.Path(tasks_dir).iterdir() if (p / "counts.mtx").exists())
for t in sorted(tasks, key=lambda p: -p.joinpath("counts.mtx").stat().st_size):   # largest first
    for s in seeds:
        if "infercnv" in engines:
            jobs.append(("infercnv", [rscript, str(HERE / "02_run_infercnv.R")], t, s))
        if "pyinfercnv" in engines:
            jobs.append(("pyinfercnv", [python, str(HERE / "03_run_pyinfercnv.py")], t, s))


def run(job):
    engine, cmd, t, s = job
    od = pathlib.Path(out_dir) / engine / t.name / f"run{s}"
    if (od / "result.json").exists():
        return
    core = cores.get()
    try:
        # every run uses set.seed(1) / random_state=1; the repeat only times it again
        r = subprocess.run(["taskset", "-c", str(core), *cmd, str(t), GENE_ORDER, str(od), "1"], env=env,
                           capture_output=True, text=True)
        print(engine, r.stdout.strip() or r.stderr.strip()[-400:], flush=True)
    finally:
        cores.put(core)


with ThreadPoolExecutor(n_par) as ex:
    list(ex.map(run, jobs))
