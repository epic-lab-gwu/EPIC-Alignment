import numpy as np

from epa.core.time_alignment import (
    compute_psr,
    interpolate_quat_linear,
    interpolate_quat_slerp,
    matching_time_indices,
)


def test_matching_time_indices_respects_offset() -> None:
    stamps_ref = np.array([0.0, 1.0, 2.0])
    stamps_est = np.array([0.1, 1.1, 2.1])

    ids_ref, ids_est = matching_time_indices(
        stamps_ref,
        stamps_est,
        max_diff=1e-6,
        offset_2=-0.1,
    )

    assert ids_ref == [0, 1, 2]
    assert ids_est == [0, 1, 2]


def test_compute_psr_returns_nan_for_short_signal() -> None:
    corr = np.array([1.0, 3.0, 1.0])
    psr = compute_psr(corr, peak_idx=1, guard_bins=1)
    assert np.isnan(psr)


def test_interpolate_quat_linear_outputs_unit_quaternions() -> None:
    t_src = np.array([0.0, 1.0])
    q_src = np.array([[0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 2.0, 0.0]])
    q_query = interpolate_quat_linear(t_src, q_src, np.array([0.5]))

    norms = np.linalg.norm(q_query, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms))


def test_interpolate_quat_slerp_clamps_out_of_range_queries() -> None:
    t_src = np.array([0.0, 1.0])
    q_src = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
    q_query = interpolate_quat_slerp(t_src, q_src, np.array([-1.0, 2.0]))

    np.testing.assert_allclose(q_query[0], q_src[0])
    np.testing.assert_allclose(q_query[1], q_src[1])
