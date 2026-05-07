from __future__ import annotations

import numpy as np
import pytest

from epa.core.evaluation import (
    build_rpe_pairs,
    compute_ape_evo_style,
    compute_rpe_evo_style,
    compute_success_regions,
    compute_valid_segment_metrics,
    estimate_knee_threshold,
    normalize_pose_relation,
    resolve_success_threshold,
)


def _identity_traj(n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros((n, 3), dtype=float)
    pos[:, 0] = np.arange(n, dtype=float)
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (n, 1))
    return pos, quat


def test_compute_ape_zero_for_identical_trajectories() -> None:
    pos, quat = _identity_traj()
    metrics = compute_ape_evo_style(pos, quat, pos, quat)

    assert metrics["translation_part"]["rmse"] == 0.0
    assert metrics["rotation_angle_deg"]["rmse"] == 0.0


def test_compute_rpe_zero_for_identical_trajectories() -> None:
    pos, quat = _identity_traj()
    metrics = compute_rpe_evo_style(pos, quat, pos, quat, delta=1, delta_unit="f")

    assert metrics["pair_count"] == 5
    assert metrics["translation_part"]["rmse"] == 0.0
    assert metrics["rotation_angle_deg"]["rmse"] == 0.0


def test_build_rpe_pairs_rejects_non_integer_frame_delta() -> None:
    pos, quat = _identity_traj()
    poses = np.repeat(np.eye(4)[None, :, :], pos.shape[0], axis=0)

    with pytest.raises(ValueError):
        build_rpe_pairs(poses, delta=1.5, delta_unit="f")


def test_compute_rpe_seconds_unit_uses_time_pairs() -> None:
    t = np.arange(0.0, 4.1, 0.5)
    pos_ref = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    pos_est = np.column_stack([1.1 * t, np.zeros_like(t), np.zeros_like(t)])
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))

    metrics = compute_rpe_evo_style(
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


def test_compute_rpe_seconds_unit_requires_timestamps() -> None:
    pos, quat = _identity_traj()

    with pytest.raises(ValueError, match="requires timestamps"):
        compute_rpe_evo_style(pos, quat, pos, quat, delta=1.0, delta_unit="s")


def test_compute_rpe_seconds_unit_checks_timestamp_length() -> None:
    pos, quat = _identity_traj()

    with pytest.raises(ValueError, match="same length"):
        compute_rpe_evo_style(pos, quat, pos, quat, delta=1.0, delta_unit="s", timestamps=np.array([0.0, 1.0]))


def test_compute_metrics_can_include_raw_arrays() -> None:
    pos, quat = _identity_traj()
    ape = compute_ape_evo_style(pos, quat, pos, quat, include_raw=True)
    rpe = compute_rpe_evo_style(pos, quat, pos, quat, delta=1, include_raw=True)

    assert "_error_arrays" in ape
    assert ape["_error_arrays"]["translation_part"].shape[0] == pos.shape[0]
    assert "_error_arrays" in rpe
    assert rpe["_error_arrays"]["translation_part"].shape[0] == rpe["pair_count"]


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


def test_estimate_knee_threshold_finds_error_distribution_knee() -> None:
    errors = np.array([1.0, 1.2, 1.3, 1.4, 1.5, 8.0, 30.0, 40.0], dtype=float)

    threshold = estimate_knee_threshold(errors, min_threshold_m=0.5, max_threshold_m=50.0)

    assert 7.0 <= threshold <= 9.0


def test_estimate_knee_threshold_trims_extreme_tail() -> None:
    errors = np.array([10.0, 12.0, 15.0, 18.0, 20.0, 21.0, 22.0, 30.0, 250.0], dtype=float)

    threshold = estimate_knee_threshold(errors, min_threshold_m=1.0, max_threshold_m=100.0, trim_percentile=95.0)

    assert threshold < 40.0


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

    ape = compute_ape_evo_style(pos_ref, quat, pos_est, quat, include_raw=True)
    rpe = compute_rpe_evo_style(pos_ref, quat, pos_est, quat, delta=1, delta_unit="f", include_raw=True)
    rpe_time = compute_rpe_evo_style(
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
        drift_rpe_1s_m=2.0,
        drift_ape_slope_mps=1.0,
        drift_ape_jump_m=5.0,
        include_raw=True,
    )

    np.testing.assert_allclose(metrics["success"]["success_rate_distance"], 0.4)
    assert metrics["ape"]["translation_part"]["rmse"] == 0.0
    assert metrics["rpe"]["pair_count"] == 2
    assert metrics["rpe_time_1s"]["pair_count"] == 2


def test_compute_valid_segment_metrics_drops_isolated_valid_samples() -> None:
    t = np.arange(3, dtype=float)
    pos_ref = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    pos_est = pos_ref.copy()
    pos_est[[0, 2], 1] = 12.0
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))

    ape = compute_ape_evo_style(pos_ref, quat, pos_est, quat, include_raw=True)
    rpe = compute_rpe_evo_style(pos_ref, quat, pos_est, quat, delta=1, delta_unit="f", include_raw=True)
    rpe_time = compute_rpe_evo_style(
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


def test_compute_valid_segment_metrics_does_not_fail_stable_bias_as_drift() -> None:
    t = np.arange(6, dtype=float)
    pos_ref = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    pos_est = pos_ref.copy()
    pos_est[:, 1] = 6.0
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))

    ape = compute_ape_evo_style(pos_ref, quat, pos_est, quat, include_raw=True)
    rpe = compute_rpe_evo_style(pos_ref, quat, pos_est, quat, delta=1, delta_unit="f", include_raw=True)
    rpe_time = compute_rpe_evo_style(
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
    np.testing.assert_allclose(metrics["ape"]["translation_part"]["rmse"], 6.0)
    np.testing.assert_allclose(metrics["rpe_time_1s"]["translation_part"]["rmse"], 0.0)


def test_compute_valid_segment_metrics_global_gate_marks_failed_case() -> None:
    t = np.arange(5, dtype=float)
    pos_ref = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    pos_est = pos_ref.copy()
    pos_est[:, 1] = 50.0
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))

    ape = compute_ape_evo_style(pos_ref, quat, pos_est, quat, include_raw=True)
    rpe = compute_rpe_evo_style(pos_ref, quat, pos_est, quat, delta=1, delta_unit="f", include_raw=True)
    rpe_time = compute_rpe_evo_style(
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

    assert metrics["success"]["case_status"] == "globally_failed"
    assert metrics["success"]["success_rate_distance"] == 0.0
    assert metrics["ape"]["_error_arrays"]["translation_part"].size == 0


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

    rpe = compute_rpe_evo_style(
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


def test_normalize_pose_relation_accepts_evo_aliases() -> None:
    assert normalize_pose_relation("ape", "trans_part") == "translation_part"
    assert normalize_pose_relation("rpe", "angle_deg") == "rotation_angle_deg"
    assert normalize_pose_relation("rpe", "point_distance_error_ratio") == "point_distance_error_ratio"
