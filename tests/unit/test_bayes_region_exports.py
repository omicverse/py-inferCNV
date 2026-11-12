"""Check the scientific export contract after removal and reassignment."""
import numpy as np
import pandas as pd
import pytest

from pyinfercnv.bayesnet.filter_high_p_normals import filter_high_p_normals, summarize_bayes_regions


@pytest.mark.parametrize("reassign", [False, True])
def test_region_cn_state_and_posterior_rows_stay_aligned(reassign):
    regions = pd.DataFrame({"subcluster": [0, 0, 0], "chromosome": ["chr1"] * 3,
                            "bin_start": [0, 2, 4], "bin_end": [1, 3, 5],
                            "state": [5, 1, 3], "cn": [3., .5, 1.5]})
    post = np.array([[.01, .01, .01, .01, .95, .01],
                     [.01, .01, .95, .01, .01, .01],
                     [.01, .01, .50, .46, .01, .01]])
    kept, all_regions = summarize_bayes_regions(regions, post, reassign=reassign)
    # P(normal)==0.5 is retained; only the middle region is removed.
    assert kept.posterior_row.tolist() == [0, 2]
    assert kept.region_id.tolist() == ["region_000001", "region_000003"]
    assert all_regions.removed_by_pnormal.tolist() == [False, True, False]
    assert kept.state.tolist() == ([4, 2] if reassign else [5, 3])
    assert kept.cn.tolist() == ([2., 1.] if reassign else [3., 1.5])
    np.testing.assert_array_equal(kept.state_r, kept.state + 1)
    np.testing.assert_array_equal(kept[[f"p_state_{i}" for i in range(6)]], post[[0, 2]])
    raw_matrix = np.array([[5, 5, 1, 1, 3, 3]], dtype=np.int8)
    filtered = filter_high_p_normals(raw_matrix, {
        "cnv_posterior": post, "region_cell_idx": [np.array([0])] * 3,
        "region_gene_idx": [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])],
    }, reassign=reassign)
    for r in all_regions.itertuples():
        assert np.all(filtered[0, r.bin_start:r.bin_end + 1] == r.state)
    assert regions.state.tolist() == [5, 1, 3]  # caller's pre-Bayes evidence is intact


def test_export_rejects_missing_posterior_rows():
    with pytest.raises(ValueError, match="matching rows"):
        summarize_bayes_regions(pd.DataFrame({"state": [1]}), np.empty((0, 6)))


def test_empty_region_export_retains_schema():
    kept, full = summarize_bayes_regions(pd.DataFrame({"state": pd.Series(dtype=int)}), np.empty((0, 6)))
    assert len(kept) == len(full) == 0
    assert {"region_id", "posterior_row", "state", "state_r", "cn", "p_normal"} <= set(full)


def test_i3_export_preserves_signed_delta_convention():
    kept, full = summarize_bayes_regions(pd.DataFrame({"state": [0]}), np.array([[.1, .2, .7]]),
                                         HMM_type="i3", reassign=True)
    assert kept.state.tolist() == [2]
    assert kept.cn.tolist() == [1.]
