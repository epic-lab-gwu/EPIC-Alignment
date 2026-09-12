from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.spatial.transform import Rotation as R

from epa.core.time_alignment import (
    compute_psr,
    get_angular_velocity_norm,
    interpolate_linear_extrapolate,
    interpolate_quat_linear,
    interpolate_quat_slerp,
    matching_time_indices,
)
from epa.core.time_sync import (
    _correlate_full,
    _correlate_zncc,
    _next_fast_len_235,
    _run_time_alignment,
)


def _legacy_matching_time_indices(stamps_1, stamps_2, max_diff=0.01, offset_2=0.0):
    s1 = np.asarray(stamps_1, dtype=float).reshape(-1)
    s2 = np.asarray(stamps_2, dtype=float).reshape(-1) + float(offset_2)
    idx_1 = []
    idx_2 = []
    j = 0
    for i, timestamp in enumerate(s1):
        while j < s2.size and s2[j] < timestamp - float(max_diff):
            j += 1
        if j >= s2.size:
            break
        best_j = -1
        best_diff = np.inf
        for candidate in (j, j + 1):
            if candidate >= s2.size:
                continue
            difference = abs(s2[candidate] - timestamp)
            if difference <= float(max_diff) and difference < best_diff:
                best_j = int(candidate)
                best_diff = float(difference)
        if best_j >= 0:
            idx_1.append(int(i))
            idx_2.append(best_j)
            j = best_j + 1
    return idx_1, idx_2


def test_vectorized_linear_interpolation_matches_scipy() -> None:
    rng = np.random.default_rng(20260902)
    t_src = np.cumsum(rng.uniform(0.01, 0.2, size=200))
    values = rng.normal(size=(t_src.size, 4))
    t_query = np.linspace(t_src[0] - 0.5, t_src[-1] + 0.5, 500)

    expected = interp1d(t_src, values, axis=0, fill_value="extrapolate")(t_query)
    actual = interpolate_linear_extrapolate(t_src, values, t_query)

    # scipy 1.11 and 1.18 use algebraically equivalent linear-interpolation
    # expressions with machine-precision rounding differences.
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=2e-15)


def test_vectorized_linear_interpolation_preserves_duplicate_fallback() -> None:
    t_src = np.array([0.0, 1.0, 1.0, 2.0])
    values = np.array([[0.0], [1.0], [1.5], [4.0]])
    t_query = np.array([-0.5, 0.5, 1.0, 1.5, 2.5])

    expected = interp1d(t_src, values, axis=0, fill_value="extrapolate")(t_query)
    actual = interpolate_linear_extrapolate(t_src, values, t_query)

    np.testing.assert_array_equal(actual, expected)


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


def test_matching_time_indices_randomized_equivalence_to_legacy() -> None:
    rng = np.random.default_rng(20260905)
    for _ in range(300):
        n_ref = int(rng.integers(1, 300))
        n_est = int(rng.integers(1, 300))
        stamps_ref = np.cumsum(rng.uniform(0.001, 0.2, size=n_ref))
        stamps_est = np.cumsum(rng.uniform(0.001, 0.2, size=n_est))
        if rng.random() < 0.2 and n_est > 2:
            duplicate = int(rng.integers(1, n_est))
            stamps_est[duplicate] = stamps_est[duplicate - 1]
        max_diff = float(rng.uniform(0.001, 0.1))
        offset = float(rng.uniform(-0.2, 0.2))

        expected = _legacy_matching_time_indices(
            stamps_ref,
            stamps_est,
            max_diff=max_diff,
            offset_2=offset,
        )
        actual = matching_time_indices(
            stamps_ref,
            stamps_est,
            max_diff=max_diff,
            offset_2=offset,
        )

        assert actual == expected


def test_compute_psr_returns_nan_for_short_signal() -> None:
    corr = np.array([1.0, 3.0, 1.0])
    psr = compute_psr(corr, peak_idx=1, guard_bins=1)
    assert np.isnan(psr)


def test_compute_psr_ignores_unsupported_lags_and_guards_entire_peak() -> None:
    corr = np.array([np.nan, -0.2, 0.1, 0.0, 0.2, 0.8, 1.0, 0.9, -0.1, 0.15, -0.05, np.nan])
    sidelobes = np.array([-0.2, 0.1, 0.0, 0.2, -0.1, 0.15, -0.05])

    actual = compute_psr(corr, peak_idx=6, guard_bins=1)

    assert actual == pytest.approx((1.0 - sidelobes.mean()) / sidelobes.std())


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


def _quat_from_omega_z(t: np.ndarray, omega: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0])
    yaw = np.cumsum(np.maximum(omega, 0.01) * dt)
    return R.from_euler("z", yaw).as_quat()


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


def test_step1_false_large_offset_prefers_zero_when_near_zero_is_sane() -> None:
    t = np.arange(0.0, 30.0, 0.05)
    rng = np.random.default_rng(0)
    kernel = np.ones(21) / 21
    omega_gt = np.convolve(np.abs(rng.normal(size=t.size)), kernel, mode="same") + 0.1
    omega_est = np.convolve(np.abs(rng.normal(size=t.size)), kernel, mode="same") + 0.1

    out = _run_time_alignment(
        t_gt=t,
        quat_gt=_quat_from_omega_z(t, omega_gt),
        t_est=t,
        quat_est=_quat_from_omega_z(t, omega_est),
        dt_resample=0.01,
        offset_search_window_s=0.0,
        offset_min_match_ratio=0.3,
        evo_match_max_diff_s=0.02,
    )

    assert abs(float(out["calculated_offset"])) < 1e-9
    assert float(out["time_metrics"]["offset_match_ratio_gate"]) >= 0.99


@pytest.mark.parametrize("true_offset_s", [-5.0, 5.0])
def test_step1_true_large_offset_is_not_forced_to_zero(true_offset_s: float) -> None:
    t_gt = np.arange(0.0, 60.0, 0.05)
    rng = np.random.default_rng(20260906)
    omega = 0.2 + np.convolve(rng.uniform(0.0, 2.0, size=t_gt.size), np.ones(5) / 5, mode="same")
    quat = _quat_from_omega_z(t_gt, omega)

    out = _run_time_alignment(
        t_gt=t_gt,
        quat_gt=quat,
        t_est=t_gt + true_offset_s,
        quat_est=quat,
        dt_resample=0.01,
        offset_search_window_s=0.0,
        offset_min_match_ratio=0.3,
        evo_match_max_diff_s=0.02,
    )

    assert float(out["calculated_offset"]) == pytest.approx(true_offset_s, abs=0.01)
    assert float(out["time_metrics"]["offset_match_ratio_gate"]) >= 0.3
    assert out["time_metrics"]["offset_confidence_rejected"] == 0.0
    assert out["time_metrics"]["xcorr_peak_normalized"] > 0.99
    assert out["time_metrics"]["xcorr_psr"] >= out["time_metrics"]["xcorr_min_psr"]
    assert out["time_metrics"]["omega_rmse_after"] < 0.1 * out["time_metrics"]["omega_rmse_before"]


@pytest.mark.parametrize("omega_z", [0.0, 0.8])
def test_step1_constant_motion_has_no_identifiable_time_offset(omega_z: float) -> None:
    t = np.arange(0.0, 20.0, 0.05)
    quat = R.from_euler("z", omega_z * t).as_quat()

    out = _run_time_alignment(
        t_gt=t,
        quat_gt=quat,
        t_est=t + 2.0,
        quat_est=quat,
        dt_resample=0.01,
        offset_search_window_s=0.0,
        offset_min_match_ratio=0.3,
        evo_match_max_diff_s=0.02,
    )

    assert float(out["calculated_offset"]) == 0.0
    assert out["time_metrics"]["offset_confidence_rejected"] == 1.0
    assert np.isnan(out["time_metrics"]["xcorr_candidate_normalized"])


def test_step1_unrelated_motion_rejects_low_normalized_correlation() -> None:
    t = np.arange(0.0, 60.0, 0.05)
    rng = np.random.default_rng(20260907)
    omega_gt = rng.uniform(0.2, 2.0, size=t.size)
    omega_est = rng.uniform(0.2, 2.0, size=t.size)

    out = _run_time_alignment(
        t_gt=t,
        quat_gt=_quat_from_omega_z(t, omega_gt),
        t_est=t,
        quat_est=_quat_from_omega_z(t, omega_est),
        dt_resample=0.01,
        offset_search_window_s=0.0,
        offset_min_match_ratio=0.3,
        evo_match_max_diff_s=0.02,
    )

    assert float(out["calculated_offset"]) == 0.0
    assert out["time_metrics"]["offset_confidence_rejected"] == 1.0
    assert (
        out["time_metrics"]["xcorr_candidate_normalized"]
        < out["time_metrics"]["xcorr_min_normalized"]
    )


def test_step1_periodic_motion_rejects_high_correlation_with_ambiguous_psr() -> None:
    t = np.arange(0.0, 60.0, 0.05)
    omega = 1.0 + 0.5 * np.sin(np.pi * t)
    quat = _quat_from_omega_z(t, omega)

    out = _run_time_alignment(
        t_gt=t,
        quat_gt=quat,
        t_est=t + 0.6,
        quat_est=quat,
        dt_resample=0.01,
        offset_search_window_s=0.0,
        offset_min_match_ratio=0.3,
        evo_match_max_diff_s=0.02,
    )

    assert float(out["calculated_offset"]) == 0.0
    assert out["time_metrics"]["offset_confidence_rejected"] == 1.0
    assert out["time_metrics"]["xcorr_candidate_normalized"] > 0.99
    assert out["time_metrics"]["xcorr_candidate_psr"] < out["time_metrics"]["xcorr_min_psr"]


def test_step1_initialization_reset_does_not_create_nonzero_time_offset() -> None:
    t = np.arange(0.0, 100.0, 0.05)
    rng = np.random.default_rng(20260908)
    omega = 0.1 + np.convolve(rng.uniform(0.0, 0.2, size=t.size), np.ones(21) / 21, mode="same")
    quat_gt = _quat_from_omega_z(t, omega)
    # Mirror the MH02 failure: a 104-degree frame jump over 0.05 s,
    # then a tracking gap and another frame change. Timestamps stay synchronized.
    reset_yaw = np.where(t >= 0.2, np.deg2rad(104.0), 0.0)
    reset_yaw += np.where(t >= 3.15, np.deg2rad(-110.0), 0.0)
    quat_est = (R.from_euler("z", reset_yaw) * R.from_quat(quat_gt)).as_quat()
    keep = (t <= 0.2) | (t >= 3.15)

    out = _run_time_alignment(
        t_gt=t,
        quat_gt=quat_gt,
        t_est=t[keep],
        quat_est=quat_est[keep],
        dt_resample=0.01,
        offset_search_window_s=0.0,
        offset_min_match_ratio=0.3,
        evo_match_max_diff_s=0.02,
    )

    assert float(out["calculated_offset"]) == 0.0
    assert out["time_metrics"]["offset_confidence_rejected"] == 1.0


def test_step1_disable_time_offset_calibration_forces_zero() -> None:
    t_gt = np.arange(0.0, 30.0, 0.05)
    yaw = 0.3 * np.sin(0.7 * t_gt) + 0.02 * t_gt * t_gt
    quat = R.from_euler("z", yaw).as_quat()

    out = _run_time_alignment(
        t_gt=t_gt,
        quat_gt=quat,
        t_est=t_gt + 5.0,
        quat_est=quat,
        dt_resample=0.01,
        offset_search_window_s=0.0,
        offset_min_match_ratio=0.3,
        evo_match_max_diff_s=0.02,
        disable_time_offset_calibration=True,
    )

    assert float(out["calculated_offset"]) == 0.0
    assert float(out["time_metrics"]["time_offset_calibration_disabled"]) == 1.0


@pytest.mark.parametrize("n_est,n_gt", [(7, 5), (65, 43), (1000, 800), (2048, 2048)])
def test_numpy_fft_full_correlation_matches_scipy(n_est: int, n_gt: int) -> None:
    rng = np.random.default_rng(20260902 + n_est + n_gt)
    sig_est = rng.normal(size=n_est)
    sig_gt = rng.normal(size=n_gt)

    expected = correlate(sig_est, sig_gt, mode="full")
    actual = _correlate_full(sig_est, sig_gt)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert int(np.argmax(actual)) == int(np.argmax(expected))


@pytest.mark.parametrize("n_est,n_gt", [(7, 5), (43, 65), (1000, 800)])
def test_zncc_matches_independent_overlap_pearson_correlation(n_est: int, n_gt: int) -> None:
    rng = np.random.default_rng(20260906 + n_est + n_gt)
    sig_est = 3.0 + 2.0 * rng.normal(size=n_est)
    sig_gt = -2.0 + 0.5 * rng.normal(size=n_gt)
    expected_corr = []
    expected_overlap = []
    for lag in range(-n_gt + 1, n_est):
        lo = max(0, lag)
        hi = min(n_est, n_gt + lag)
        x = sig_est[lo:hi]
        y = sig_gt[lo - lag : hi - lag]
        expected_overlap.append(x.size)
        if x.size < 2:
            expected_corr.append(np.nan)
        else:
            expected_corr.append(np.corrcoef(x, y)[0, 1])

    actual_corr, actual_overlap = _correlate_zncc(sig_est, sig_gt)

    np.testing.assert_array_equal(actual_overlap, expected_overlap)
    np.testing.assert_allclose(actual_corr, expected_corr, atol=1e-9, rtol=1e-9, equal_nan=True)
    assert np.all(np.abs(actual_corr[np.isfinite(actual_corr)]) <= 1.0)


@pytest.mark.parametrize("constant_side", ["lhs", "rhs", "both"])
def test_zncc_marks_zero_variance_overlaps_invalid(constant_side: str) -> None:
    lhs = np.arange(8, dtype=float)
    rhs = np.arange(5, dtype=float)
    if constant_side in ("lhs", "both"):
        lhs[:] = 7.0
    if constant_side in ("rhs", "both"):
        rhs[:] = -3.0

    corr, overlap = _correlate_zncc(lhs, rhs)

    assert np.all(np.isnan(corr))
    assert overlap.max() == min(lhs.size, rhs.size)


def test_next_fast_len_uses_smallest_235_smooth_size() -> None:
    assert _next_fast_len_235(1) == 1
    assert _next_fast_len_235(7) == 8
    assert _next_fast_len_235(36_099) == 36_450
    assert _next_fast_len_235(187_199) == 187_500
