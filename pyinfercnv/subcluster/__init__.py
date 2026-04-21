"""Tumor subcluster discovery — R parity with inferCNV_tumor_subclusters.R
and inferCNV_tumor_subclusters.random_smoothed_trees.R.

Three backends:
    leiden_subcluster      — graph-based (KNN + Leiden), default in modern R infercnv
    random_tree_subcluster — recursive hierarchical + permutation p-value
                             (R random_smoothed_trees, P7-patched with subsample cap)
    qnorm_subcluster       — hierarchical + qnorm cut on tree heights, deterministic
"""
from __future__ import annotations

from pyinfercnv.subcluster.leiden import leiden_subcluster
from pyinfercnv.subcluster.qnorm import qnorm_subcluster
from pyinfercnv.subcluster.random_trees import random_tree_subcluster

__all__ = [
    "leiden_subcluster",
    "qnorm_subcluster",
    "random_tree_subcluster",
]
