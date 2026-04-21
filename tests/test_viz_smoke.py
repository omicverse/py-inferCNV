"""viz smoke + omicverse-boundary enforcement."""
from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_viz_does_not_import_omicverse():
    viz_dir = REPO_ROOT / "pyinfercnv" / "viz"
    cmd = ["grep", "-r", "-l", "-E", r"^\s*(import|from)\s+omicverse", str(viz_dir)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 1, f"omicverse imported in viz: {r.stdout}"


def test_viz_imports_succeed():
    import matplotlib
    matplotlib.use("Agg")
    from pyinfercnv.viz import plot_cnv_delta, plot_cnv_heatmap, plot_cnv_heatmap_compare  # noqa: F401
