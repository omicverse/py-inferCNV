"""Regression test for the Phase 3 orchestrator persistence contract.

Catches the silent persistence bug fixed by P0.1 (`gibbs_result["theta_mean"]`
→ `gibbs_result["cnv_posterior"]`) and the step-20 invariant that
``hmm_proxy_matrix`` is computed from ``filtered_states`` rather than raw
``hmm_states`` when BayesNet ran.

Monkey-patch target rationale: ``run_phase3`` does
``from pyinfercnv.bayesnet import _step18_bayesnet`` LOCALLY inside the
``if bayes_on:`` block (see ``pipeline_phase3.py``), so the call site reads
the attribute on the source module ``pyinfercnv.bayesnet`` at call time. The
patch target is therefore ``pyinfercnv.bayesnet._step18_bayesnet`` — patching
``pyinfercnv.pipeline_phase3._step18_bayesnet`` would silently no-op and let
the heavy real Gibbs sampler run.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from pyinfercnv.config import InferCNVConfig
from pyinfercnv.pipeline_phase3 import run_phase3
from pyinfercnv.result import InferCNVResult


def _minimal_phase2_result(
    *,
    n_cells: int = 5,
    n_bins: int = 8,
    raw_state_idx: int = 5,  # i6 idx 5 → CN ratio 3.0 (high-amp)
) -> InferCNVResult:
    """Build a Phase 2 fixture with non-neutral raw HMM states.

    The raw / filtered divergence is the test's whole point — see
    ``test_run_phase3_persists_cnv_posterior_and_uses_filtered_states``.
    """
    return InferCNVResult(
        chr_pos={"chr1": 0},
        cnv_matrix=np.zeros((n_cells, n_bins), dtype=np.float32),
        cnv_matrix_fc=np.ones((n_cells, n_bins), dtype=np.float32),
        cell_meta=pd.DataFrame(
            {"is_reference": [False] * n_cells},
            index=[f"cell{i}" for i in range(n_cells)],
        ),
        subclusters=np.zeros(n_cells, dtype=np.int32),
        hmm_states=np.full((n_cells, n_bins), raw_state_idx, dtype=np.int8),
        cnv_regions=pd.DataFrame(
            {
                "subcluster": np.array([0], dtype=np.int32),
                "bin_start": [0],
                "bin_end": [n_bins - 1],
            }
        ),
        hspike_calibration=SimpleNamespace(
            state_mus=np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float64),
            state_sigmas=np.full(6, 0.3, dtype=np.float64),
        ),
    )


def test_run_phase3_persists_cnv_posterior_and_uses_filtered_states(monkeypatch):
    # Raw hmm_states all idx 5 (CN 3.0 high-amp); filtered_states all idx 2
    # (neutral, CN 1.0). If the orchestrator mistakenly remaps from the raw
    # matrix the proxy becomes 3.0, not 1.0 — that is what makes this test
    # actually load-bearing for the "step 20 consumes filtered_states" contract.
    result = _minimal_phase2_result(raw_state_idx=5)
    cfg = InferCNVConfig(BayesMaxPNormal=0.5, HMM_type="i6", reassignCNVs=False)

    cnv_posterior = np.zeros((1, 6), dtype=np.float64)
    filtered_states = np.full(result.hmm_states.shape, fill_value=2, dtype=np.int8)

    # Sanity: fake matches the real gibbs.py:269-276 / filter_high_p_normals.py contract.
    assert cnv_posterior.dtype == np.float64
    assert filtered_states.dtype == np.int8
    assert filtered_states.shape == result.hmm_states.shape
    assert int(result.hmm_states.min()) == 5  # raw really is non-neutral
    assert int(filtered_states.min()) == 2     # filtered really is neutral

    def fake_step18(result_arg, config_arg, *, i6_calibration):
        assert result_arg is result
        assert config_arg is cfg
        assert i6_calibration is result.hspike_calibration
        return {"cnv_posterior": cnv_posterior}, filtered_states

    monkeypatch.setattr("pyinfercnv.bayesnet._step18_bayesnet", fake_step18)

    out = run_phase3(result, config=cfg)

    # P0.1 contract: bayes_posterior persisted under the canonical key,
    # shape (n_regions, K) per gibbs.py:269.
    assert out.bayes_posterior is not None
    assert out.bayes_posterior.shape == (1, 6)
    assert out.bayes_posterior.dtype == np.float64

    # Step 20 contract: proxy is all 1.0 (filtered all neutral) — would be
    # 3.0 if the orchestrator used the raw hmm_states.
    assert out.hmm_proxy_matrix is not None
    assert out.hmm_proxy_matrix.shape == result.hmm_states.shape
    assert out.hmm_proxy_matrix.dtype == np.float64
    np.testing.assert_allclose(out.hmm_proxy_matrix, 1.0)
