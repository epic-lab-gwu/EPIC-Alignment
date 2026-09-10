from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from epa.core.evaluation import (
    apply_input_coverage_to_success,
    build_rpe_pairs,
    compute_ape,
    compute_rpe,
    compute_success_regions,
    compute_valid_segment_metrics,
    estimate_knee_threshold,
    normalize_pose_relation,
    rpe_pairs_by_path,
    rpe_pairs_by_time,
    resolve_drift_thresholds,
    resolve_success_threshold,
)


def test_complete_reference_sr_penalizes_truncated_but_locally_perfect_run() -> None:
    success = {
        "success_rate_distance": 1.0,
        "success_rate_time": 1.0,
        "raw_success_rate_distance": 1.0,
        "raw_success_rate_time": 1.0,
        "valid_distance_m": 10.0,
        "total_distance_m": 10.0,
        "valid_time_s": 20.0,
        "total_time_s": 20.0,
        "raw_valid_distance_m": 10.0,
        "raw_valid_time_s": 20.0,
        "sr_reliability_status": "ok",
    }
    coverage = {
        "coverage_status": "failed",
        "coverage_hard_reasons": ["path_coverage_critical"],
        "coverage_soft_reasons": [],
        "reference_path_length_m": 100.0,
        "reference_duration_s": 80.0,
        "path_coverage_ratio": 0.1,
        "temporal_coverage_ratio": 0.25,
    }

    apply_input_coverage_to_success(success, coverage)

    assert success["local_success_rate_distance"] == 1.0
    assert success["local_success_rate_time"] == 1.0
    assert success["complete_success_rate_distance"] == 0.1
    assert success["complete_success_rate_time"] == 0.25
    assert success["success_rate_distance"] == 0.1
    assert success["success_rate_time"] == 0.25
    assert success["success_rate_scope"] == "complete_reference"
    assert success["sr_reliability_status"] == "failed"


def test_complete_reference_sr_matches_local_sr_at_full_coverage() -> None:
    success = {
        "success_rate_distance": 0.8,
        "success_rate_time": 0.75,
        "raw_success_rate_distance": 0.9,
        "raw_success_rate_time": 0.85,
        "valid_distance_m": 8.0,
        "total_distance_m": 10.0,
        "valid_time_s": 15.0,
        "total_time_s": 20.0,
        "raw_valid_distance_m": 9.0,
        "raw_valid_time_s": 17.0,
        "sr_reliability_status": "ok",
    }
    coverage = {
        "coverage_status": "ok",
        "coverage_hard_reasons": [],
        "coverage_soft_reasons": [],
        "reference_path_length_m": 10.0,
        "reference_duration_s": 20.0,
        "path_coverage_ratio": 1.0,
        "temporal_coverage_ratio": 1.0,
    }

    apply_input_coverage_to_success(success, coverage)

    assert success["success_rate_distance"] == success["local_success_rate_distance"] == 0.8
    assert success["success_rate_time"] == success["local_success_rate_time"] == 0.75


def _identity_traj(n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros((n, 3), dtype=float)
    pos[:, 0] = np.arange(n, dtype=float)
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (n, 1))
    return pos, quat


def _legacy_rpe_pairs_by_path(poses, delta, tol=0.0):
    positions = np.array([pose[:3, 3] for pose in poses])
    distances = np.zeros(positions.shape[0], dtype=float)
    if positions.shape[0] > 1:
        distances[1:] = np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    pairs = []
    for start in range(distances.size - 1):
        offset = start + 1
        distances_from_start = distances[offset:] - distances[start]
        candidate = int(np.argmin(np.abs(distances_from_start - delta)))
        if np.abs(distances_from_start[candidate] - delta) <= tol:
            pairs.append((start, candidate + offset))
    return pairs


def _legacy_rpe_pairs_by_time(timestamps, delta, tol=0.0):
    stamps = np.asarray(timestamps, dtype=float).reshape(-1)

    def nearest_after(start_index: int, target: float) -> int | None:
        insertion = int(np.searchsorted(stamps, target, side="left"))
        candidates = [
            index
            for index in (insertion - 1, insertion, insertion + 1)
            if start_index < index < stamps.size
        ]
        if not candidates:
            return None
        best = min(candidates, key=lambda index: abs(float(stamps[index]) - float(target)))
        if abs(float(stamps[best]) - float(target)) > tol:
            return None
        return best

    pairs = []
    for start in range(stamps.size - 1):
        end = nearest_after(start, float(stamps[start]) + float(delta))
        if end is not None:
            pairs.append((start, end))
    return pairs


def test_compute_ape_zero_for_identical_trajectories() -> None:
    pos, quat = _identity_traj()
    metrics = compute_ape(pos, quat, pos, quat)

    assert metrics["translation_part"]["rmse"] == 0.0
    assert metrics["rotation_angle_deg"]["rmse"] == 0.0


def test_compute_rpe_zero_for_identical_trajectories() -> None:
    pos, quat = _identity_traj()
    metrics = compute_rpe(pos, quat, pos, quat, delta=1, delta_unit="f")

    assert metrics["pair_count"] == 5
    assert metrics["translation_part"]["rmse"] == 0.0
    assert metrics["rotation_angle_deg"]["rmse"] == 0.0


def test_build_rpe_pairs_rejects_non_integer_frame_delta() -> None:
    pos, quat = _identity_traj()
    poses = np.repeat(np.eye(4)[None, :, :], pos.shape[0], axis=0)

    with pytest.raises(ValueError):
        build_rpe_pairs(poses, delta=1.5, delta_unit="f")


@pytest.mark.parametrize("delta,tol", [(0.0, 0.0), (0.4, 0.05), (1.5, 0.2), (20.0, 0.5)])
def test_rpe_pairs_by_path_vectorization_matches_legacy_scalar_oracle(
    delta: float, tol: float
) -> None:
    rng = np.random.default_rng(20260907)
    steps = rng.normal(size=(180, 3))
    steps[20:25] = 0.0
    positions = np.cumsum(steps * rng.uniform(0.0, 0.08, size=(180, 1)), axis=0)
    poses = np.repeat(np.eye(4, dtype=float)[None, :, :], positions.shape[0], axis=0)
    poses[:, :3, 3] = positions

    expected = _legacy_rpe_pairs_by_path(poses, delta, tol=tol)
    actual = rpe_pairs_by_path(poses, delta, tol=tol, all_pairs=True)

    assert actual == expected


@pytest.mark.parametrize("delta,tol", [(0.01, 0.0), (0.2, 0.01), (1.0, 0.1), (50.0, 1.0)])
def test_rpe_pairs_by_time_vectorization_matches_legacy_scalar_oracle(
    delta: float, tol: float
) -> None:
    rng = np.random.default_rng(20260909)
    timestamps = np.cumsum(rng.uniform(0.005, 0.08, size=500))

    expected = _legacy_rpe_pairs_by_time(timestamps, delta, tol=tol)
    actual = rpe_pairs_by_time(timestamps, delta, tol=tol, all_pairs=True)

    assert actual == expected


def test_compute_rpe_seconds_unit_uses_time_pairs() -> None:
    t = np.arange(0.0, 4.1, 0.5)
    pos_ref = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    pos_est = np.column_stack([1.1 * t, np.zeros_like(t), np.zeros_like(t)])
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))

    metrics = compute_rpe(
        pos_ref,
        quat,
        pos_est,
        quat,
        delta=1.0,
        delta_unit="s",
        rel_delta_tol=0.01,
        timestamps=t,
        include_raw=True,
    )

    assert metrics["pair_count"] == 4
    np.testing.assert_allclose(metrics["_pair_ids"], np.array([[0, 2], [2, 4], [4, 6], [6, 8]]))
    np.testing.assert_allclose(metrics["_error_arrays"]["translation_part"], np.full(4, 0.1))
    np.testing.assert_allclose(metrics["translation_part"]["rmse"], 0.1)


def test_compute_rpe_seconds_unit_handles_nonuniform_timestamps() -> None:
    t = np.array([0.0, 0.4, 1.0, 1.7, 2.0, 3.05], dtype=float)
    pos_ref = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    pos_est = np.column_stack([1.1 * t, np.zeros_like(t), np.zeros_like(t)])
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))

    metrics = compute_rpe(
        pos_ref,
        quat,
        pos_est,
        quat,
        delta=1.0,
        delta_unit="s",
        rel_delta_tol=0.06,
        timestamps=t,
        all_pairs=True,
        include_raw=True,
    )

    np.testing.assert_array_equal(metrics["_pair_ids"], np.array([[0, 2], [2, 4], [4, 5]]))
    np.testing.assert_allclose(metrics["_error_arrays"]["translation_part"], np.array([0.1, 0.1, 0.105]))


@pytest.mark.parametrize(
    "timestamps",
    [
        np.array([0.0, 1.0, 1.0, 2.0], dtype=float),
        np.array([0.0, 1.0, 0.5, 2.0], dtype=float),
    ],
)
def test_compute_rpe_seconds_unit_rejects_duplicate_or_nonmonotonic_timestamps(timestamps: np.ndarray) -> None:
    pos = np.column_stack([timestamps, np.zeros_like(timestamps), np.zeros_like(timestamps)])
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (timestamps.size, 1))

    with pytest.raises(ValueError, match="strictly increasing"):
        compute_rpe(pos, quat, pos, quat, delta=1.0, delta_unit="s", timestamps=timestamps)


def test_compute_rpe_seconds_unit_requires_timestamps() -> None:
    pos, quat = _identity_traj()

    with pytest.raises(ValueError, match="requires timestamps"):
        compute_rpe(pos, quat, pos, quat, delta=1.0, delta_unit="s")


def test_compute_rpe_seconds_unit_checks_timestamp_length() -> None:
    pos, quat = _identity_traj()

    with pytest.raises(ValueError, match="same length"):
        compute_rpe(pos, quat, pos, quat, delta=1.0, delta_unit="s", timestamps=np.array([0.0, 1.0]))


def test_compute_metrics_can_include_raw_arrays() -> None:
    pos, quat = _identity_traj()
    ape = compute_ape(pos, quat, pos, quat, include_raw=True)
    rpe = compute_rpe(pos, quat, pos, quat, delta=1, include_raw=True)

    assert "_error_arrays" in ape
    assert ape["_error_arrays"]["translation_part"].shape[0] == pos.shape[0]
    assert "_error_arrays" in rpe
    assert rpe["_error_arrays"]["translation_part"].shape[0] == rpe["pair_count"]


def test_rotation_rpe_has_known_ground_truth_for_constant_yaw_rate() -> None:
    t = np.arange(6, dtype=float)
    pos = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    quat_ref = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))
    quat_est = R.from_euler("z", 10.0 * t, degrees=True).as_quat()

    metrics = compute_rpe(
        pos,
        quat_ref,
        pos,
        quat_est,
        delta=1.0,
        delta_unit="s",
        timestamps=t,
        all_pairs=True,
        include_raw=True,
    )

    np.testing.assert_allclose(metrics["_error_arrays"]["rotation_angle_deg"], np.full(5, 10.0), atol=1e-12)
    np.testing.assert_allclose(metrics["rotation_angle_deg"]["rmse"], 10.0, atol=1e-12)


def test_short_trajectory_has_no_long_distance_or_time_rpe_pairs() -> None:
    t = np.array([0.0, 0.5], dtype=float)
    pos = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))

    rpe_dist = compute_rpe(pos, quat, pos, quat, delta=8.0, delta_unit="m", all_pairs=True)
    rpe_time = compute_rpe(pos, quat, pos, quat, delta=1.0, delta_unit="s", timestamps=t, all_pairs=True)

    assert rpe_dist["pair_count"] == 0
    assert rpe_time["pair_count"] == 0


def test_compute_success_regions_uses_valid_distance_segments() -> None:
    t = np.arange(6, dtype=float)
    pos = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    errors = np.array([0.0, 1.0, 12.0, 14.0, 1.0, 0.5], dtype=float)

    regions = compute_success_regions(t, pos, errors, threshold_m=10.0)

    np.testing.assert_array_equal(
        regions["valid_sample_mask"],
        np.array([True, True, False, False, True, True]),
    )
    np.testing.assert_array_equal(
        regions["valid_segment_mask"],
        np.array([True, False, False, False, True]),
    )
    assert regions["fail_segment_count"] == 1
    assert regions["fail_segments"][0]["start_index"] == 2
    assert regions["fail_segments"][0]["end_index"] == 3
    np.testing.assert_allclose(regions["success_rate_distance"], 0.4)
    np.testing.assert_allclose(regions["success_rate_time"], 0.4)


def test_compute_success_regions_keeps_multiple_fail_and_recover_blocks_separate() -> None:
    t = np.arange(9, dtype=float)
    pos = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    errors = np.array([0.0, 0.2, 12.0, 13.0, 0.3, 0.4, 15.0, 0.5, 0.6], dtype=float)

    regions = compute_success_regions(t, pos, errors, threshold_m=10.0)

    assert regions["fail_segment_count"] == 2
    assert [(s["start_index"], s["end_index"]) for s in regions["fail_segments"]] == [(2, 3), (6, 6)]
    np.testing.assert_allclose(regions["success_rate_distance"], 3.0 / 8.0)
    np.testing.assert_array_equal(
        regions["valid_segment_mask"],
        np.array([True, False, False, False, True, False, False, True]),
    )


def test_compute_success_regions_treats_nan_and_inf_errors_as_failed_samples() -> None:
    t = np.arange(5, dtype=float)
    pos = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    errors = np.array([0.0, np.nan, 0.2, np.inf, 0.1], dtype=float)

    regions = compute_success_regions(t, pos, errors, threshold_m=10.0)

    np.testing.assert_array_equal(regions["valid_sample_mask"], np.array([True, False, True, False, True]))
    np.testing.assert_allclose(regions["success_rate_distance"], 0.0)
    assert regions["fail_segment_count"] == 2


def test_estimate_knee_threshold_finds_error_distribution_knee() -> None:
    errors = np.array([1.0, 1.2, 1.3, 1.4, 1.5, 8.0, 30.0, 40.0], dtype=float)

    threshold = estimate_knee_threshold(errors, min_threshold_m=0.5, max_threshold_m=50.0)

    assert 7.0 <= threshold <= 9.0


def test_estimate_knee_threshold_trims_extreme_tail() -> None:
    errors = np.array([10.0, 12.0, 15.0, 18.0, 20.0, 21.0, 22.0, 30.0, 250.0], dtype=float)

    threshold = estimate_knee_threshold(errors, min_threshold_m=1.0, max_threshold_m=100.0, trim_percentile=95.0)

    assert threshold < 40.0


def test_estimate_knee_threshold_handles_flat_all_good_distribution() -> None:
    errors = np.full(20, 0.2, dtype=float)

    threshold = estimate_knee_threshold(errors, min_threshold_m=1.0, max_threshold_m=30.0)

    assert threshold == 1.0


def test_estimate_knee_threshold_clips_all_bad_distribution_to_max() -> None:
    errors = np.array([100.0, 110.0, 120.0, 130.0], dtype=float)

    threshold = estimate_knee_threshold(errors, min_threshold_m=1.0, max_threshold_m=30.0)

    assert threshold == 30.0


def test_resolve_success_threshold_returns_nan_for_no_finite_errors() -> None:
    threshold, info = resolve_success_threshold(
        np.array([np.nan, np.inf], dtype=float),
        mode="adaptive_knee",
        min_threshold_m=1.0,
        max_threshold_m=30.0,
    )

    assert np.isnan(threshold)
    assert np.isnan(info["threshold_m"])


def test_resolve_success_threshold_supports_fixed_and_adaptive_modes() -> None:
    errors = np.array([1.0, 1.2, 1.3, 8.0, 30.0], dtype=float)

    fixed, fixed_info = resolve_success_threshold(errors, mode="fixed", fixed_threshold_m=12.0)
    adaptive, adaptive_info = resolve_success_threshold(
        errors,
        mode="adaptive_knee",
        min_threshold_m=2.0,
        max_threshold_m=20.0,
    )

    assert fixed == 12.0
    assert fixed_info["mode"] == "fixed"
    assert 2.0 <= adaptive <= 20.0
    assert adaptive_info["mode"] == "adaptive_knee"


def test_compute_valid_segment_metrics_filters_ape_and_rpe_pairs() -> None:
    t = np.arange(6, dtype=float)
    pos_ref = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    pos_est = pos_ref.copy()
    pos_est[3:, 1] = 12.0
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))

    ape = compute_ape(pos_ref, quat, pos_est, quat, include_raw=True)
    rpe = compute_rpe(pos_ref, quat, pos_est, quat, delta=1, delta_unit="f", include_raw=True)
    rpe_time = compute_rpe(
        pos_ref,
        quat,
        pos_est,
        quat,
        delta=1.0,
        delta_unit="s",
        timestamps=t,
        all_pairs=True,
        include_raw=True,
    )

    metrics = compute_valid_segment_metrics(
        timestamps=t,
        pos_ref=pos_ref,
        ape_block=ape,
        rpe_block=rpe,
        rpe_time_1s_block=rpe_time,
        threshold_m=10.0,
        drift_threshold_mode="fixed",
        drift_rpe_1s_m=2.0,
        drift_ape_slope_mps=1.0,
        drift_ape_jump_m=5.0,
        include_raw=True,
    )

    np.testing.assert_allclose(metrics["success"]["success_rate_distance"], 0.8)
    np.testing.assert_allclose(metrics["success"]["raw_success_rate_distance"], 0.8)
    np.testing.assert_allclose(metrics["success"]["local_success_rate_distance"], 0.8)
    assert metrics["success"]["sr_reliability_status"] == "ok"
    assert metrics["ape"]["translation_part"]["rmse"] == np.sqrt(72.0)
    assert metrics["rpe"]["pair_count"] == 4
    assert metrics["rpe_time_1s"]["pair_count"] == 4


def test_resolve_drift_thresholds_adapts_to_case_scale() -> None:
    slow_t = np.arange(20, dtype=float)
    slow_pos = np.column_stack([0.05 * slow_t, np.zeros_like(slow_t), np.zeros_like(slow_t)])
    fast_t = np.arange(20, dtype=float)
    fast_pos = np.column_stack([5.0 * fast_t, np.zeros_like(fast_t), np.zeros_like(fast_t)])
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (slow_t.size, 1))
    slow_rpe = compute_rpe(slow_pos, quat, slow_pos, quat, delta=1.0, delta_unit="s", timestamps=slow_t, all_pairs=True)
    fast_rpe = compute_rpe(fast_pos, quat, fast_pos, quat, delta=1.0, delta_unit="s", timestamps=fast_t, all_pairs=True)

    slow = resolve_drift_thresholds(slow_t, slow_pos, slow_rpe)
    fast = resolve_drift_thresholds(fast_t, fast_pos, fast_rpe)

    assert slow["mode"] == "adaptive"
    assert fast["mode"] == "adaptive"
    assert slow["rpe_1s_m"] < fast["rpe_1s_m"]
    assert slow["ape_jump_m"] < fast["ape_jump_m"]


def test_compute_valid_segment_metrics_ignores_legacy_adaptive_ape_threshold() -> None:
    t = np.arange(6, dtype=float)
    pos_ref = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    pos_est = pos_ref.copy()
    pos_est[3:, 1] = 3.0
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))

    ape = compute_ape(pos_ref, quat, pos_est, quat, include_raw=True)
    rpe = compute_rpe(pos_ref, quat, pos_est, quat, delta=1, delta_unit="f", include_raw=True)
    rpe_time = compute_rpe(pos_ref, quat, pos_est, quat, delta=1.0, delta_unit="s", timestamps=t, all_pairs=True, include_raw=True)

    metrics = compute_valid_segment_metrics(
        timestamps=t,
        pos_ref=pos_ref,
        ape_block=ape,
        rpe_block=rpe,
        rpe_time_1s_block=rpe_time,
        threshold_m=1.0,
        global_gate_m=100.0,
        drift_threshold_mode="adaptive",
        include_raw=True,
    )

    assert metrics["success"]["policy"] == "rpe_1s_motion_relative"
    np.testing.assert_allclose(metrics["success"]["success_rate_distance"], 0.8)


def test_compute_valid_segment_metrics_drops_isolated_valid_samples() -> None:
    t = np.arange(3, dtype=float)
    pos_ref = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    pos_est = pos_ref.copy()
    pos_est[[0, 2], 1] = 12.0
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))

    ape = compute_ape(pos_ref, quat, pos_est, quat, include_raw=True)
    rpe = compute_rpe(pos_ref, quat, pos_est, quat, delta=1, delta_unit="f", include_raw=True)
    rpe_time = compute_rpe(
        pos_ref,
        quat,
        pos_est,
        quat,
        delta=1.0,
        delta_unit="s",
        timestamps=t,
        all_pairs=True,
        include_raw=True,
    )

    metrics = compute_valid_segment_metrics(
        timestamps=t,
        pos_ref=pos_ref,
        ape_block=ape,
        rpe_block=rpe,
        rpe_time_1s_block=rpe_time,
        threshold_m=10.0,
        include_raw=True,
    )

    assert metrics["ape"]["_error_arrays"]["translation_part"].size == 0
    assert metrics["rpe"]["pair_count"] == 0
    np.testing.assert_allclose(metrics["success"]["success_rate_distance"], 0.0)


def test_compute_valid_segment_metrics_allows_stable_bias_above_ape_threshold() -> None:
    t = np.arange(6, dtype=float)
    pos_ref = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    pos_est = pos_ref.copy()
    pos_est[:, 1] = 6.0
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))

    ape = compute_ape(pos_ref, quat, pos_est, quat, include_raw=True)
    rpe = compute_rpe(pos_ref, quat, pos_est, quat, delta=1, delta_unit="f", include_raw=True)
    rpe_time = compute_rpe(
        pos_ref,
        quat,
        pos_est,
        quat,
        delta=1.0,
        delta_unit="s",
        timestamps=t,
        all_pairs=True,
        include_raw=True,
    )

    metrics = compute_valid_segment_metrics(
        timestamps=t,
        pos_ref=pos_ref,
        ape_block=ape,
        rpe_block=rpe,
        rpe_time_1s_block=rpe_time,
        threshold_m=5.0,
        global_gate_m=30.0,
        include_raw=True,
    )

    np.testing.assert_allclose(metrics["success"]["success_rate_distance"], 1.0)
    assert metrics["success"]["global_gate_failed"] is False
    np.testing.assert_allclose(metrics["ape"]["translation_part"]["rmse"], 6.0)
    np.testing.assert_allclose(metrics["rpe_time_1s"]["translation_part"]["rmse"], 0.0)


def test_compute_valid_segment_metrics_invalidates_full_bad_time_rpe_interval() -> None:
    t = np.arange(21, dtype=float) * 0.1
    pos_ref = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    pos_est = pos_ref.copy()
    pos_est[10:, 1] = 10.0
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))

    ape = compute_ape(pos_ref, quat, pos_est, quat, include_raw=True)
    rpe = compute_rpe(pos_ref, quat, pos_est, quat, delta=1, delta_unit="f", include_raw=True)
    rpe_time = compute_rpe(
        pos_ref,
        quat,
        pos_est,
        quat,
        delta=1.0,
        delta_unit="s",
        timestamps=t,
        all_pairs=True,
        pairs_from_reference=True,
        include_raw=True,
    )

    metrics = compute_valid_segment_metrics(
        timestamps=t,
        pos_ref=pos_ref,
        ape_block=ape,
        rpe_block=rpe,
        rpe_time_1s_block=rpe_time,
        threshold_m=100.0,
        global_gate_m=1000.0,
        drift_rpe_1s_m=2.0,
        drift_ape_slope_mps=1000.0,
        drift_ape_jump_m=1000.0,
        include_raw=True,
        include_masks=True,
    )

    np.testing.assert_allclose(metrics["success"]["success_rate_distance"], 0.05)
    assert metrics["rpe_time_1s"]["pair_count"] == 0


def test_compute_valid_segment_metrics_preserves_recovered_segment_between_failures() -> None:
    t = np.arange(9, dtype=float)
    pos_ref = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    pos_est = pos_ref.copy()
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))

    ape = compute_ape(pos_ref, quat, pos_est, quat, include_raw=True)
    rpe = compute_rpe(pos_ref, quat, pos_est, quat, delta=1, delta_unit="f", include_raw=True)
    rpe_time = compute_rpe(
        pos_ref,
        quat,
        pos_est,
        quat,
        delta=1.0,
        delta_unit="s",
        timestamps=t,
        all_pairs=True,
        include_raw=True,
    )
    rpe_time["_error_arrays"]["translation_part"][1] = 12.0
    rpe_time["_error_arrays"]["translation_part"][5] = 12.0

    metrics = compute_valid_segment_metrics(
        timestamps=t,
        pos_ref=pos_ref,
        ape_block=ape,
        rpe_block=rpe,
        rpe_time_1s_block=rpe_time,
        threshold_m=10.0,
        drift_rpe_1s_m=2.0,
        drift_ape_slope_mps=100.0,
        drift_ape_jump_m=100.0,
        include_raw=True,
        include_masks=True,
    )

    np.testing.assert_array_equal(
        metrics["success"]["valid_segment_mask"],
        np.array([True, False, True, True, True, False, True, True]),
    )
    np.testing.assert_allclose(metrics["success"]["success_rate_distance"], 6.0 / 8.0)
    np.testing.assert_array_equal(metrics["rpe"]["_pair_ids"], np.array([[0, 1], [2, 3], [3, 4], [4, 5], [6, 7], [7, 8]]))


def test_compute_valid_segment_metrics_global_gate_keeps_local_success_rate() -> None:
    t = np.arange(10, dtype=float)
    pos_ref = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    pos_est = pos_ref.copy()
    pos_est[:, 1] = 50.0
    pos_est[5:, 1] = 70.0
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))

    ape = compute_ape(pos_ref, quat, pos_est, quat, include_raw=True)
    rpe = compute_rpe(pos_ref, quat, pos_est, quat, delta=1, delta_unit="f", include_raw=True)
    rpe_time = compute_rpe(
        pos_ref,
        quat,
        pos_est,
        quat,
        delta=1.0,
        delta_unit="s",
        timestamps=t,
        all_pairs=True,
        include_raw=True,
    )

    metrics = compute_valid_segment_metrics(
        timestamps=t,
        pos_ref=pos_ref,
        ape_block=ape,
        rpe_block=rpe,
        rpe_time_1s_block=rpe_time,
        threshold_m=100.0,
        global_gate_m=30.0,
        global_gate_percentile=5.0,
        include_raw=True,
    )

    assert metrics["success"]["case_status"] == "valid_segment"
    assert metrics["success"]["global_gate_failed"] is False
    np.testing.assert_allclose(metrics["success"]["success_rate_distance"], 8.0 / 9.0)
    assert metrics["ape"]["_error_arrays"]["translation_part"].size == 10


def test_rpe_point_distance_ratio_keeps_pair_aligned_raw_arrays() -> None:
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (pos.shape[0], 1))

    rpe = compute_rpe(
        pos,
        quat,
        pos,
        quat,
        delta=1,
        delta_unit="f",
        include_raw=True,
    )
    ratio = rpe["_error_arrays"]["point_distance_error_ratio"]

    assert ratio.shape[0] == rpe["pair_count"]
    assert np.isnan(ratio[0])
    assert ratio[1] == 0.0


def test_normalize_pose_relation_accepts_common_aliases() -> None:
    assert normalize_pose_relation("ape", "trans_part") == "translation_part"
    assert normalize_pose_relation("rpe", "angle_deg") == "rotation_angle_deg"
    assert normalize_pose_relation("rpe", "point_distance_error_ratio") == "point_distance_error_ratio"
