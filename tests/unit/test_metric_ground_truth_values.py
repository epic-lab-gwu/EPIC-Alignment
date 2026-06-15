from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from epa.core.evaluation import (
    compute_ape,
    compute_rpe,
    compute_valid_segment_metrics,
)
from epa.ov_eval_compat import _drift_rate_percent


def _quat_identity(n: int) -> np.ndarray:
    return np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (int(n), 1))


def _line_traj(t: np.ndarray, *, scale: float = 1.0, bias_y: float = 0.0) -> np.ndarray:
    return np.column_stack([float(scale) * t, np.full_like(t, float(bias_y)), np.zeros_like(t)])


def test_ground_truth_identity_metrics_are_zero() -> None:
    t = np.arange(6, dtype=float)
    pos = _line_traj(t)
    quat = _quat_identity(t.size)

    ape = compute_ape(pos, quat, pos, quat, include_raw=True)
    rpe_time = compute_rpe(
        pos,
        quat,
        pos,
        quat,
        delta=1.0,
        delta_unit="s",
        timestamps=t,
        all_pairs=True,
        pairs_from_reference=True,
        include_raw=True,
    )
    valid = compute_valid_segment_metrics(
        timestamps=t,
        pos_ref=pos,
        ape_block=ape,
        rpe_block=rpe_time,
        rpe_time_1s_block=rpe_time,
        threshold_m=10.0,
        include_raw=True,
    )

    assert ape["translation_part"]["rmse"] == 0.0
    assert rpe_time["translation_part"]["rmse"] == 0.0
    assert valid["success"]["success_rate_distance"] == 1.0


def test_ground_truth_scale_drift_metrics_match_closed_form_values() -> None:
    t = np.arange(6, dtype=float)
    pos_ref = _line_traj(t)
    pos_est = _line_traj(t, scale=1.1)
    quat = _quat_identity(t.size)

    ape = compute_ape(pos_ref, quat, pos_est, quat, include_raw=True)
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
    rpe_dist_ref = compute_rpe(
        pos_ref,
        quat,
        pos_est,
        quat,
        delta=1.0,
        delta_unit="m",
        rel_delta_tol=0.01,
        all_pairs=True,
        pairs_from_reference=True,
        include_raw=True,
    )
    rpe_dist_est = compute_rpe(
        pos_ref,
        quat,
        pos_est,
        quat,
        delta=1.0,
        delta_unit="m",
        rel_delta_tol=0.01,
        all_pairs=True,
        pairs_from_reference=False,
        include_raw=True,
    )

    expected_ape_rmse = np.sqrt(np.mean((0.1 * t) ** 2))
    np.testing.assert_allclose(ape["translation_part"]["rmse"], expected_ape_rmse)
    np.testing.assert_allclose(rpe_time["translation_part"]["rmse"], 0.1)
    np.testing.assert_allclose(rpe_dist_ref["translation_part"]["rmse"], 0.1)
    np.testing.assert_allclose(_drift_rate_percent(rpe_dist_ref["_error_arrays"]["translation_part"], 1.0), 10.0)
    assert rpe_dist_est["pair_count"] == 0


def test_ground_truth_pure_rotation_metrics_match_closed_form_values() -> None:
    t = np.arange(6, dtype=float)
    pos = np.zeros((t.size, 3), dtype=float)
    quat_ref = _quat_identity(t.size)
    quat_est = R.from_euler("z", 10.0 * t, degrees=True).as_quat()

    ape = compute_ape(pos, quat_ref, pos, quat_est, include_raw=True)
    rpe_time = compute_rpe(
        pos,
        quat_ref,
        pos,
        quat_est,
        delta=1.0,
        delta_unit="s",
        timestamps=t,
        all_pairs=True,
        pairs_from_reference=True,
        include_raw=True,
    )

    expected_rot_ape_rmse = np.sqrt(np.mean((10.0 * t) ** 2))
    np.testing.assert_allclose(ape["rotation_angle_deg"]["rmse"], expected_rot_ape_rmse)
    np.testing.assert_allclose(rpe_time["rotation_angle_deg"]["rmse"], 10.0)
    np.testing.assert_allclose(rpe_time["translation_part"]["rmse"], 0.0)


def test_ground_truth_stable_bias_is_not_local_drift_when_global_gate_allows_it() -> None:
    t = np.arange(6, dtype=float)
    pos_ref = _line_traj(t)
    pos_est = _line_traj(t, bias_y=6.0)
    quat = _quat_identity(t.size)

    ape = compute_ape(pos_ref, quat, pos_est, quat, include_raw=True)
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
    valid = compute_valid_segment_metrics(
        timestamps=t,
        pos_ref=pos_ref,
        ape_block=ape,
        rpe_block=rpe_time,
        rpe_time_1s_block=rpe_time,
        threshold_m=10.0,
        global_gate_m=30.0,
        include_raw=True,
    )
    globally_rejected = compute_valid_segment_metrics(
        timestamps=t,
        pos_ref=pos_ref,
        ape_block=ape,
        rpe_block=rpe_time,
        rpe_time_1s_block=rpe_time,
        threshold_m=10.0,
        global_gate_m=5.0,
        include_raw=True,
    )

    np.testing.assert_allclose(ape["translation_part"]["rmse"], 6.0)
    np.testing.assert_allclose(rpe_time["translation_part"]["rmse"], 0.0)
    np.testing.assert_allclose(valid["success"]["success_rate_distance"], 1.0)
    assert globally_rejected["success"]["case_status"] == "globally_unstable"
    assert globally_rejected["success"]["global_gate_failed"] is True
    np.testing.assert_allclose(globally_rejected["success"]["success_rate_distance"], 1.0)
    np.testing.assert_allclose(globally_rejected["ape"]["translation_part"]["rmse"], 6.0)


def test_ground_truth_local_jump_invalidates_drift_valid_metrics() -> None:
    t = np.arange(21, dtype=float) * 0.1
    pos_ref = _line_traj(t)
    pos_est = pos_ref.copy()
    pos_est[10:, 1] = 10.0
    quat = _quat_identity(t.size)

    ape = compute_ape(pos_ref, quat, pos_est, quat, include_raw=True)
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
    valid = compute_valid_segment_metrics(
        timestamps=t,
        pos_ref=pos_ref,
        ape_block=ape,
        rpe_block=rpe_time,
        rpe_time_1s_block=rpe_time,
        threshold_m=100.0,
        global_gate_m=1000.0,
        include_raw=True,
    )

    np.testing.assert_allclose(ape["translation_part"]["rmse"], np.sqrt(11.0 * 100.0 / 21.0))
    np.testing.assert_allclose(valid["success"]["success_rate_distance"], 0.0)
    assert valid["success"]["valid_sample_count"] == 0
    assert valid["rpe_time_1s"]["pair_count"] == 0


def test_ground_truth_multiple_failures_preserve_recovered_segments() -> None:
    t = np.arange(9, dtype=float)
    pos = _line_traj(t)
    quat = _quat_identity(t.size)

    ape = compute_ape(pos, quat, pos, quat, include_raw=True)
    rpe_frame = compute_rpe(pos, quat, pos, quat, delta=1, delta_unit="f", include_raw=True)
    rpe_time = compute_rpe(
        pos,
        quat,
        pos,
        quat,
        delta=1.0,
        delta_unit="s",
        timestamps=t,
        all_pairs=True,
        include_raw=True,
    )
    rpe_time["_error_arrays"]["translation_part"][1] = 12.0
    rpe_time["_error_arrays"]["translation_part"][5] = 12.0

    valid = compute_valid_segment_metrics(
        timestamps=t,
        pos_ref=pos,
        ape_block=ape,
        rpe_block=rpe_frame,
        rpe_time_1s_block=rpe_time,
        threshold_m=10.0,
        drift_rpe_1s_m=2.0,
        drift_ape_slope_mps=100.0,
        drift_ape_jump_m=100.0,
        include_raw=True,
    )

    np.testing.assert_allclose(valid["success"]["success_rate_distance"], 0.25)
    assert valid["rpe"]["pair_count"] == 2
    np.testing.assert_allclose(valid["ape"]["translation_part"]["rmse"], 0.0)
