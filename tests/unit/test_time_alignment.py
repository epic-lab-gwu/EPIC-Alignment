from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.spatial.transform import Rotation as R

from epa.core.time_alignment import (
    compute_psr,
    get_angular_velocity_norm,
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


def test_matching_time_indices_enforces_one_to_one_usage() -> None:
    stamps_ref = np.array([0.000, 0.004, 0.008, 0.012])
    stamps_est = np.array([0.006])

    ids_ref, ids_est = matching_time_indices(
        stamps_ref,
        stamps_est,
        max_diff=0.01,
        offset_2=0.0,
    )

    assert len(ids_ref) == 1
    assert ids_est == [0]


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


def _build_pose_with_duplicate_timestamps(n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n, dtype=float) * 0.01
    yaw = 0.6 * np.sin(np.linspace(0.0, 22.0, n)) + 0.05 * np.linspace(0.0, 1.0, n)
    q = R.from_euler("z", yaw).as_quat()

    dup_idx = np.arange(20, n, 25)
    t_dup = np.insert(t, dup_idx, t[dup_idx])
    q_dup = np.insert(q, dup_idx, q[dup_idx], axis=0)
    return t_dup, q_dup


def test_get_angular_velocity_norm_filters_duplicate_timestamps() -> None:
    t_dup, q_dup = _build_pose_with_duplicate_timestamps()
    t_mid, omega = get_angular_velocity_norm(t_dup, q_dup)

    assert t_mid.size > 10
    assert omega.size == t_mid.size
    assert np.all(np.isfinite(t_mid))
    assert np.all(np.isfinite(omega))


def test_step1_correlation_stays_near_zero_with_duplicate_timestamps() -> None:
    t_ref, q_ref = _build_pose_with_duplicate_timestamps()
    t_est, q_est = _build_pose_with_duplicate_timestamps()

    t_ref_mid, om_ref = get_angular_velocity_norm(t_ref, q_ref)
    t_est_mid, om_est = get_angular_velocity_norm(t_est, q_est)

    dt_resample = 0.001
    t_min = max(t_ref_mid[0], t_est_mid[0])
    t_max = min(t_ref_mid[-1], t_est_mid[-1])
    t_uniform = np.arange(t_min, t_max, dt_resample)

    sig_ref = interp1d(t_ref_mid, om_ref, kind="linear")(t_uniform)
    sig_est = interp1d(t_est_mid, om_est, kind="linear")(t_uniform)
    sig_ref -= np.mean(sig_ref)
    sig_est -= np.mean(sig_est)

    corr = correlate(sig_est, sig_ref, mode="full")
    assert np.all(np.isfinite(corr))

    lags = np.arange(-len(sig_ref) + 1, len(sig_est))
    offsets = lags.astype(float) * dt_resample
    mask = np.abs(offsets) <= 2.0
    peak_idx = np.flatnonzero(mask)[int(np.argmax(corr[mask]))]
    offset = float(offsets[peak_idx])

    ids_ref, _ = matching_time_indices(t_ref, t_est, max_diff=0.02, offset_2=-offset)
    ratio = float(len(ids_ref)) / float(min(len(t_ref), len(t_est)))

    assert abs(offset) < 0.01
    assert ratio > 0.95
