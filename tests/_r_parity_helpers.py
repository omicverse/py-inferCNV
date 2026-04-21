"""Helpers for R-parity tests — loaders and assertion utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_r_matrix(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.to_numpy(dtype=np.float64)


def assert_bit_exact(py: np.ndarray, r: np.ndarray, *, tol: float = 1e-10) -> None:
    assert py.shape == r.shape, f"shape mismatch: py={py.shape} r={r.shape}"
    diff = np.max(np.abs(py.astype(np.float64) - r.astype(np.float64)))
    assert diff < tol, f"max_diff={diff:.3e} exceeds tol={tol:.3e}"


def assert_approx(py: np.ndarray, r: np.ndarray, *, tol: float = 1e-6) -> None:
    assert py.shape == r.shape, f"shape mismatch: py={py.shape} r={r.shape}"
    diff = np.max(np.abs(py.astype(np.float64) - r.astype(np.float64)))
    assert diff < tol, f"max_diff={diff:.3e} exceeds tol={tol:.3e}"
