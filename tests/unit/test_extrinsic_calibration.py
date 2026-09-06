import numpy as np
from scipy.spatial.transform import Rotation as R

from epa.core.calibration import (
    extrinsic_rotation_transition_mask,
    solve_extrinsic_rotation,
    solve_extrinsic_rotation_multibaseline,
    solve_extrinsic_translation,
)
from epa.core.trajectory_alignment import _solve_extrinsic_and_world_alignment


def _integrate_relative_rotations(rotations: R) -> R:
    absolute = [R.identity()]
    for delta in rotations:
        absolute.append(absolute[-1] * delta)
    return R.concatenate(absolute)


def _calibration_case_with_two_bad_initial_transitions():
    axes = np.asarray(
        [[1.0, 0.2, 0.1], [0.1, 1.0, 0.2], [0.2, 0.1, 1.0], [1.0, -0.3, 0.4]],
        dtype=float,
    )
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    angles = np.radians(np.linspace(4.0, 18.0, 39))
    relative_ref = R.from_rotvec(axes[np.arange(angles.size) % axes.shape[0]] * angles[:, None])
    rotation_ref = _integrate_relative_rotations(relative_ref)

    extrinsic_rotation = R.from_euler("xyz", [12.0, -8.0, 20.0], degrees=True)
    rotation_est = list(
        R.from_matrix(rotation_ref.as_matrix() @ extrinsic_rotation.as_matrix())
    )
    # Reproduce the two inconsistent initial transitions in the MH_01 ORB-SLAM3 output.
    rotation_est[1] = rotation_est[2] * R.from_rotvec(
        np.radians(31.0) * np.asarray([0.0, 1.0, 0.0])
    )
    rotation_est[0] = rotation_est[1] * R.from_rotvec(
        np.radians(110.0) * np.asarray([-1.0, 0.0, 0.0])
    )
    rotation_est = R.concatenate(rotation_est)

    u = np.linspace(0.0, 1.0, len(rotation_ref))
    position_ref = np.column_stack(
        [5.0 * u + np.sin(4.0 * u), np.sin(7.0 * u), np.cos(3.0 * u)]
    )
    extrinsic_translation = np.asarray([0.15, -0.08, 0.05])
    position_est = position_ref + np.einsum(
        "nij,j->ni", rotation_ref.as_matrix(), extrinsic_translation
    )
    position_est[:2] += np.asarray([[8.0, -5.0, 3.0], [-6.0, 7.0, -4.0]])
    return (
        position_ref,
        rotation_ref.as_quat(),
        position_est,
        rotation_est.as_quat(),
        extrinsic_rotation,
        extrinsic_translation,
    )


def test_rotation_consistency_uses_absolute_gate_for_small_rotations() -> None:
    ref_angles_deg = np.asarray([0.2, 0.4, 10.0, 10.0, 8.0])
    est_angles_deg = np.asarray([0.35, 0.8, 11.5, 14.0, 8.1])
    axis = np.asarray([0.0, 0.0, 1.0])
    q_ref = _integrate_relative_rotations(
        R.from_rotvec(np.radians(ref_angles_deg)[:, None] * axis)
    ).as_quat()
    q_est = _integrate_relative_rotations(
        R.from_rotvec(np.radians(est_angles_deg)[:, None] * axis)
    ).as_quat()

    mask, _, _, info = extrinsic_rotation_transition_mask(
        q_ref,
        q_est,
        angle_abs_threshold_deg=1.0,
        angle_rel_threshold=0.2,
        relative_angle_floor_deg=1.0,
    )

    # The first two rotations are below the relative-angle floor. Their large
    # relative differences are intentionally ignored because the absolute errors are small.
    assert np.array_equal(mask, np.asarray([True, True, True, False, True]))
    assert int(info["rejected_angle_count"]) == 1


def test_rotation_mask_is_reused_by_translation_calibration() -> None:
    position_ref, q_ref, position_est, q_est, expected_rotation, expected_translation = (
        _calibration_case_with_two_bad_initial_transitions()
    )

    fitted_rotation, transition_mask, info = solve_extrinsic_rotation(
        q_ref, q_est, return_info=True
    )
    fitted_translation = solve_extrinsic_translation(
        position_ref,
        q_ref,
        position_est,
        q_est,
        fitted_rotation,
        transition_mask=transition_mask,
    )
    unfiltered_translation = solve_extrinsic_translation(
        position_ref,
        q_ref,
        position_est,
        q_est,
        fitted_rotation,
    )

    assert np.array_equal(np.flatnonzero(~transition_mask), np.asarray([0, 1]))
    assert int(info["rejected_angle_count"]) == 2
    assert np.degrees(
        (R.from_matrix(fitted_rotation) * expected_rotation.inv()).magnitude()
    ) < 1e-10
    np.testing.assert_allclose(fitted_translation, expected_translation, atol=1e-10)
    assert np.linalg.norm(unfiltered_translation - expected_translation) > 1.0


def test_full_alignment_passes_rotation_mask_to_translation_solver() -> None:
    position_ref, q_ref, position_est, q_est, expected_rotation, expected_translation = (
        _calibration_case_with_two_bad_initial_transitions()
    )

    result = _solve_extrinsic_and_world_alignment(
        pr_sync=position_est,
        qr_sync=q_est,
        pos_gt_solve=position_ref,
        quat_gt_solve=q_ref,
        pr_solve=position_est,
        qr_solve=q_est,
        global_align_mode="se3",
    )

    assert np.degrees(
        (R.from_matrix(result["R_calc"]) * expected_rotation.inv()).magnitude()
    ) < 1e-10
    np.testing.assert_allclose(result["t_calc"], expected_translation, atol=1e-10)
    choice = result["step3_choice"]
    assert int(choice["extrinsic_rotation_reset_boundary_count"]) == 2
    assert int(choice["extrinsic_rotation_accepted_count"]) > 37
    assert int(choice["step2_translation_constraint_count"]) == int(
        choice["extrinsic_rotation_accepted_count"]
    )


def test_full_alignment_can_disable_extrinsic_calibration() -> None:
    position_ref, q_ref, position_est, q_est, _, _ = (
        _calibration_case_with_two_bad_initial_transitions()
    )

    result = _solve_extrinsic_and_world_alignment(
        pr_sync=position_est,
        qr_sync=q_est,
        pos_gt_solve=position_ref,
        quat_gt_solve=q_ref,
        pr_solve=position_est,
        qr_solve=q_est,
        global_align_mode="se3",
        disable_extrinsic_calibration=True,
    )

    np.testing.assert_allclose(result["R_calc"], np.eye(3), atol=0.0)
    np.testing.assert_allclose(result["t_calc"], np.zeros(3), atol=0.0)
    choice = result["step3_choice"]
    assert float(choice["extrinsic_calibration_disabled"]) == 1.0
    assert int(choice["orientation_candidate_count"]) == 1
    assert int(choice["step2_translation_constraint_count"]) == 0


def test_multibaseline_pairs_stay_inside_continuous_time_segments() -> None:
    axes = np.eye(3)[np.arange(79) % 3]
    rotation_ref = _integrate_relative_rotations(
        R.from_rotvec(axes * np.radians(2.0))
    )
    extrinsic = R.from_euler("xyz", [8.0, -5.0, 12.0], degrees=True)
    rotation_est = R.from_matrix(rotation_ref.as_matrix() @ extrinsic.as_matrix())
    timestamps = np.arange(len(rotation_ref), dtype=float) * 0.05
    timestamps[40:] += 5.0

    fitted, pair_indices, inliers, info = solve_extrinsic_rotation_multibaseline(
        rotation_ref.as_quat(),
        rotation_est.as_quat(),
        timestamps_s=timestamps,
        source_timestamps_s=timestamps,
        constrain_unobservable=False,  # Isolate segmentation from the identity prior.
    )

    assert np.any(pair_indices[:, 1] - pair_indices[:, 0] > 1)
    assert not np.any((pair_indices[:, 0] < 40) & (pair_indices[:, 1] >= 40))
    assert np.max(timestamps[pair_indices[:, 1]] - timestamps[pair_indices[:, 0]]) <= 2.0
    assert int(info["segment_count"]) == 2
    assert int(info["accepted_count"]) == int(np.count_nonzero(inliers))
    assert np.degrees((R.from_matrix(fitted) * extrinsic.inv()).magnitude()) < 1e-10


def test_dense_interpolation_does_not_reconnect_an_original_source_gap() -> None:
    timestamps = np.arange(201, dtype=float) * 0.05
    source_timestamps = timestamps[(timestamps <= 2.0) | (timestamps >= 7.0)]
    axes = np.eye(3)[np.arange(timestamps.size - 1) % 3]
    rotation_ref = _integrate_relative_rotations(
        R.from_rotvec(axes * np.radians(1.0))
    )
    extrinsic = R.from_euler("xyz", [4.0, -3.0, 6.0], degrees=True)
    rotation_est = R.from_matrix(rotation_ref.as_matrix() @ extrinsic.as_matrix())

    fitted, pair_indices, _, info = solve_extrinsic_rotation_multibaseline(
        rotation_ref.as_quat(),
        rotation_est.as_quat(),
        timestamps_s=timestamps,
        source_timestamps_s=source_timestamps,
        constrain_unobservable=False,  # Test raw recovery, not weak-axis suppression.
    )

    assert int(info["invalid_interpolated_sample_count"]) > 0
    assert int(info["segment_count"]) == 2
    assert not np.any(
        (timestamps[pair_indices[:, 0]] <= 2.0)
        & (timestamps[pair_indices[:, 1]] >= 7.0)
    )
    assert np.degrees((R.from_matrix(fitted) * extrinsic.inv()).magnitude()) < 1e-10


def test_invalid_orientation_sample_becomes_a_segment_boundary() -> None:
    timestamps = np.arange(81, dtype=float) * 0.05
    axes = np.eye(3)[np.arange(timestamps.size - 1) % 3]
    rotation_ref = _integrate_relative_rotations(
        R.from_rotvec(axes * np.radians(2.0))
    )
    extrinsic = R.from_euler("xyz", [5.0, -2.0, 9.0], degrees=True)
    q_est = R.from_matrix(rotation_ref.as_matrix() @ extrinsic.as_matrix()).as_quat()
    q_est[40] = np.nan

    fitted, pair_indices, _, info = solve_extrinsic_rotation_multibaseline(
        rotation_ref.as_quat(),
        q_est,
        timestamps_s=timestamps,
        source_timestamps_s=timestamps,
        constrain_unobservable=False,
    )

    assert int(info["invalid_rotation_sample_count"]) == 1
    assert not np.any(pair_indices == 40)
    assert np.degrees((R.from_matrix(fitted) * extrinsic.inv()).magnitude()) < 1e-10


def test_minimum_rotation_tracks_measured_orientation_noise() -> None:
    count = 80
    axes = np.eye(3)[np.arange(count) % 3]
    ref_angles_deg = np.full(count, 3.0)
    angle_noise_deg = 0.2 * np.sin(np.arange(count, dtype=float) * 1.7)
    rotation_ref = _integrate_relative_rotations(
        R.from_rotvec(axes * np.radians(ref_angles_deg)[:, None])
    )
    rotation_est = _integrate_relative_rotations(
        R.from_rotvec(axes * np.radians(ref_angles_deg + angle_noise_deg)[:, None])
    )
    timestamps = np.arange(len(rotation_ref), dtype=float) * 0.05

    _, _, _, info = solve_extrinsic_rotation_multibaseline(
        rotation_ref.as_quat(),
        rotation_est.as_quat(),
        timestamps_s=timestamps,
        source_timestamps_s=timestamps,
    )

    assert float(info["orientation_noise_deg"]) > 0.05
    assert float(info["min_rotation_deg"]) > 0.25


def test_equal_angle_wrong_axis_motion_is_capped_and_rejected_by_full_residual() -> None:
    pose_count = 201
    axes = np.eye(3)[np.arange(pose_count - 2) % 3]
    common = list(
        _integrate_relative_rotations(R.from_rotvec(axes * np.radians(2.0)))
    )
    rotation_ref = R.concatenate(
        common
        + [common[-1] * R.from_rotvec(np.radians(120.0) * np.asarray([1.0, 0.0, 0.0]))]
    )
    rotation_est = R.concatenate(
        common
        + [common[-1] * R.from_rotvec(np.radians(120.0) * np.asarray([0.0, 1.0, 0.0]))]
    )
    timestamps = np.arange(pose_count, dtype=float) * 0.05

    fitted, pair_indices, inliers, info = solve_extrinsic_rotation_multibaseline(
        rotation_ref.as_quat(),
        rotation_est.as_quat(),
        timestamps_s=timestamps,
        source_timestamps_s=timestamps,
    )

    bad_endpoint_pairs = pair_indices[:, 1] == pose_count - 1
    assert np.count_nonzero(bad_endpoint_pairs) > 0
    assert not np.any(inliers[bad_endpoint_pairs])
    assert int(info["rejected_residual_count"]) == int(np.count_nonzero(bad_endpoint_pairs))
    assert float(info["influence_cap_deg"]) == 15.0
    assert np.degrees(R.from_matrix(fitted).magnitude()) < 1e-10
