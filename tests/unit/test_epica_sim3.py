from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from epa.core.evaluation import compute_ape
from epa.core.sim3 import solve_epa_sim3, solve_epa_sim3_v1, solve_epa_sim3_v2, solve_epica_sim3_variant, solve_orientation_consistent_sim3
from epa.metric_cli_common import align_for_eval_with_info


def test_se3_replaces_se3r_and_original_name_preserves_position_only_fit() -> None:
    n = 80
    t = np.linspace(0.0, 8.0, n)
    pos_est = np.column_stack([t, np.sin(0.7 * t), 0.2 * np.cos(0.3 * t)])
    quat_est = R.from_euler(
        "zyx", np.column_stack([0.08 * t, 0.03 * np.sin(t), 0.02 * t])
    ).as_quat()
    position_rotation = R.from_euler("z", 35.0, degrees=True)
    orientation_rotation = R.from_euler("z", 10.0, degrees=True)
    translation = np.array([1.0, -0.4, 0.2], dtype=float)
    pos_ref = (position_rotation.as_matrix() @ pos_est.T).T + translation
    quat_ref = (orientation_rotation * R.from_quat(quat_est)).as_quat()

    pos_original, quat_original, original_info = align_for_eval_with_info(
        pos_ref, quat_ref, pos_est, quat_est, mode="se3-original"
    )
    pos_se3, quat_se3, info = align_for_eval_with_info(
        pos_ref, quat_ref, pos_est, quat_est, mode="se3"
    )
    pos_se3r, quat_se3r, alias_info = align_for_eval_with_info(
        pos_ref, quat_ref, pos_est, quat_est, mode="se3r"
    )
    pos_typo, quat_typo, typo_info = align_for_eval_with_info(
        pos_ref, quat_ref, pos_est, quat_est, mode="se3-orginal"
    )
    original = compute_ape(pos_ref, quat_ref, pos_original, quat_original)
    se3 = compute_ape(pos_ref, quat_ref, pos_se3, quat_se3)
    se3r = compute_ape(pos_ref, quat_ref, pos_se3r, quat_se3r)

    assert original["translation_part"]["rmse"] < 1e-9
    assert original["rotation_angle_deg"]["rmse"] > 20.0
    assert original_info["se3_original_solver"] == "position_only_umeyama"
    assert se3["rotation_angle_deg"]["rmse"] < 1e-9
    assert se3["translation_part"]["rmse"] > 0.1
    assert info["se3_solver"] == "orientation_chordal_mean_then_translation_mean"
    np.testing.assert_allclose(pos_se3r, pos_se3, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(quat_se3r, quat_se3, rtol=1e-12, atol=1e-12)
    assert se3r["rotation_angle_deg"]["rmse"] < 1e-9
    assert alias_info["se3r_solver"] == "orientation_chordal_mean_then_translation_mean"
    np.testing.assert_allclose(pos_typo, pos_original, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(quat_typo, quat_original, rtol=1e-12, atol=1e-12)
    assert typo_info["align_mode"] == "se3-original"


def test_orientation_consistent_sim3_recovers_pose_aware_similarity() -> None:
    n = 80
    t = np.linspace(0.0, 6.0, n)
    pos_ref = np.column_stack([np.cos(t), np.sin(0.7 * t), 0.2 * t])
    quat_ref = R.from_euler("zyx", np.column_stack([0.2 * t, 0.04 * np.sin(t), 0.03 * t])).as_quat()

    scale_true = 2.4
    r_fit_true = R.from_euler("zyx", [32.0, -8.0, 11.0], degrees=True)
    t_fit_true = np.array([0.4, -0.7, 0.2], dtype=float)
    pos_est = (r_fit_true.inv().as_matrix() @ ((pos_ref - t_fit_true) / scale_true).T).T
    quat_est = (r_fit_true.inv() * R.from_quat(quat_ref)).as_quat()

    scale, r_fit, t_fit, info = solve_orientation_consistent_sim3(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est,
    )

    np.testing.assert_allclose(scale, scale_true, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(r_fit, r_fit_true.as_matrix(), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(t_fit, t_fit_true, rtol=1e-9, atol=1e-9)
    assert float(info["sim3_orientation_rmse_deg"]) < 1e-9
    assert float(info["sim3_position_rmse_m"]) < 1e-9


def test_epica_sim3_differs_from_position_only_baseline_when_pose_is_inconsistent() -> None:
    n = 80
    t = np.linspace(0.0, 8.0, n)
    pos_ref = np.column_stack([np.cos(t), 0.4 * t, np.sin(0.5 * t)])
    quat_ref = R.from_euler("z", 0.15 * t).as_quat()

    r_position = R.from_euler("z", 90.0, degrees=True)
    scale_true = 1.7
    t_position = np.array([1.0, -0.5, 0.2], dtype=float)
    pos_est = (r_position.inv().as_matrix() @ ((pos_ref - t_position) / scale_true).T).T

    quat_est = quat_ref.copy()

    pos_baseline, quat_baseline, baseline_info = align_for_eval_with_info(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est,
        mode="ov_sim3",
    )
    pos_epica, quat_epica, epica_info = align_for_eval_with_info(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est,
        mode="epica_sim3",
    )

    baseline_ape = compute_ape(pos_ref, quat_ref, pos_baseline, quat_baseline)
    epica_ape = compute_ape(pos_ref, quat_ref, pos_epica, quat_epica)

    assert baseline_ape["translation_part"]["rmse"] < 1e-9
    assert baseline_ape["rotation_angle_deg"]["rmse"] > 80.0
    assert epica_ape["rotation_angle_deg"]["rmse"] < 1e-9
    assert epica_ape["translation_part"]["rmse"] > 0.1
    assert baseline_info["align_mode"] == "ov_sim3"
    assert epica_info["align_mode"] == "epica_sim3"
    assert epica_info["sim3_solver"] == "epica_orientation_consistent"


def test_public_sim3_handles_body_frame_extrinsic_rotation() -> None:
    n = 160
    t = np.linspace(0.0, 12.0, n)
    pos_ref = np.column_stack(
        [
            0.6 * t + 0.4 * np.sin(0.7 * t),
            1.2 * np.sin(0.35 * t),
            0.3 * np.cos(0.5 * t),
        ]
    )
    quat_ref = R.from_euler(
        "zyx",
        np.column_stack(
            [
                0.4 * t,
                0.25 * np.sin(0.6 * t),
                0.18 * np.cos(0.4 * t),
            ]
        ),
    ).as_quat()

    scale_true = 1.8
    r_world = R.from_euler("zyx", [25.0, -7.0, 4.0], degrees=True)
    t_world = np.array([0.5, -0.8, 0.3], dtype=float)
    r_body = R.from_euler("xyz", [12.0, -18.0, 32.0], degrees=True)
    pos_est = (r_world.inv().as_matrix() @ ((pos_ref - t_world) / scale_true).T).T
    quat_est_imu = (r_world.inv() * R.from_quat(quat_ref)).as_quat()
    quat_est_cam = (R.from_quat(quat_est_imu) * r_body).as_quat()

    pos_new, quat_new, info = align_for_eval_with_info(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est_cam,
        mode="sim3",
        t_ref=t,
    )
    ape = compute_ape(pos_ref, quat_ref, pos_new, quat_new)

    assert info["align_mode"] == "sim3"
    assert info["sim3_extrinsic_rotation_correction_used"] is True
    assert ape["translation_part"]["rmse"] < 1e-9
    assert ape["rotation_angle_deg"]["rmse"] < 1e-9


def test_epa_posyaw_aligns_xy_yaw_without_roll_pitch_or_scale() -> None:
    n = 80
    t = np.linspace(0.0, 10.0, n)
    pos_ref = np.column_stack([t, 0.5 * np.sin(t), 0.2 * np.cos(0.3 * t)])
    quat_ref = R.from_euler("zyx", np.column_stack([0.1 * t, 0.02 * np.sin(t), -0.03 * np.cos(t)])).as_quat()

    yaw_fit = R.from_euler("z", 35.0, degrees=True)
    t_fit = np.array([2.0, -1.0, 0.4], dtype=float)
    pos_est = (yaw_fit.inv().as_matrix() @ (pos_ref - t_fit).T).T
    quat_est = (yaw_fit.inv() * R.from_quat(quat_ref)).as_quat()

    pos_new, quat_new, info = align_for_eval_with_info(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est,
        mode="posyaw",
    )
    ape = compute_ape(pos_ref, quat_ref, pos_new, quat_new)

    assert info["align_mode"] == "posyaw"
    assert info["align_scale"] == 1.0
    assert info["posyaw_solver"] == "epa_yaw_only_umeyama"
    assert abs(float(info["posyaw_yaw_deg"]) - 35.0) < 1e-9
    assert ape["translation_part"]["rmse"] < 1e-9
    assert ape["rotation_angle_deg"]["rmse"] < 1e-9


def test_epa_posyaw_alignment_indices_ignore_divergent_tail() -> None:
    n = 100
    t = np.linspace(0.0, 10.0, n)
    pos_ref = np.column_stack([t, 0.4 * np.sin(t), 0.1 * np.cos(t)])
    quat_ref = R.from_euler("z", 0.05 * t).as_quat()

    yaw_fit = R.from_euler("z", 20.0, degrees=True)
    t_fit = np.array([2.0, -0.7, 0.3], dtype=float)
    pos_est = (yaw_fit.inv().as_matrix() @ (pos_ref - t_fit).T).T
    quat_est = (yaw_fit.inv() * R.from_quat(quat_ref)).as_quat()
    pos_est[60:] += np.array([500.0, -300.0, 0.0], dtype=float)

    pos_full, quat_full, full_info = align_for_eval_with_info(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est,
        mode="posyaw",
    )
    pos_indexed, quat_indexed, indexed_info = align_for_eval_with_info(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est,
        mode="posyaw",
        align_indices=np.arange(60),
    )

    full_prefix_ape = compute_ape(pos_ref[:60], quat_ref[:60], pos_full[:60], quat_full[:60])
    indexed_prefix_ape = compute_ape(pos_ref[:60], quat_ref[:60], pos_indexed[:60], quat_indexed[:60])

    assert full_prefix_ape["translation_part"]["rmse"] > 100.0
    assert indexed_prefix_ape["translation_part"]["rmse"] < 1e-9
    assert full_info["align_pair_count"] == n
    assert indexed_info["align_pair_count"] == 60
    assert indexed_info["align_index_last"] == 59


def test_epa_posyaw_does_not_absorb_roll_pitch_mismatch() -> None:
    n = 60
    t = np.linspace(0.0, 6.0, n)
    pos_ref = np.column_stack([t, np.sin(t), 0.1 * t])
    quat_ref = R.from_euler("z", 0.1 * t).as_quat()

    full_rotation = R.from_euler("zyx", [25.0, 12.0, -8.0], degrees=True)
    pos_est = (full_rotation.inv().as_matrix() @ pos_ref.T).T
    quat_est = (full_rotation.inv() * R.from_quat(quat_ref)).as_quat()

    _, quat_posyaw, info = align_for_eval_with_info(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est,
        mode="posyaw",
    )
    _, quat_se3, _ = align_for_eval_with_info(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est,
        mode="se3",
    )
    posyaw_ape = compute_ape(pos_ref, quat_ref, pos_ref, quat_posyaw)
    se3_ape = compute_ape(pos_ref, quat_ref, pos_ref, quat_se3)

    assert info["posyaw_solver"] == "epa_yaw_only_umeyama"
    assert posyaw_ape["rotation_angle_deg"]["rmse"] > 5.0
    assert se3_ape["rotation_angle_deg"]["rmse"] < 1e-9


def test_epica_sim3_variants_recover_clean_similarity() -> None:
    n = 60
    t = np.linspace(0.0, 5.0, n)
    pos_ref = np.column_stack([0.5 * t, np.sin(t), np.cos(0.3 * t)])
    quat_ref = R.from_euler("zyx", np.column_stack([0.1 * t, 0.03 * t, 0.02 * np.sin(t)])).as_quat()

    scale_true = 1.8
    r_fit_true = R.from_euler("xyz", [12.0, -15.0, 22.0], degrees=True)
    t_fit_true = np.array([-0.2, 0.3, 1.1], dtype=float)
    pos_est = (r_fit_true.inv().as_matrix() @ ((pos_ref - t_fit_true) / scale_true).T).T
    quat_est = (r_fit_true.inv() * R.from_quat(quat_ref)).as_quat()

    for method in ["epica_sim3", "epica_sim3_stable", "epica_sim3_joint", "epica_sim3_trimmed"]:
        scale, r_fit, t_fit, info = solve_epica_sim3_variant(
            pos_ref=pos_ref,
            quat_ref=quat_ref,
            pos_est=pos_est,
            quat_est=quat_est,
            method=method,
        )
        np.testing.assert_allclose(scale, scale_true, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(r_fit, r_fit_true.as_matrix(), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(t_fit, t_fit_true, rtol=1e-8, atol=1e-8)
        assert float(info["sim3_position_rmse_m"]) < 1e-8


def test_epica_sim3_stable_uses_prefix_when_tail_scale_drifts() -> None:
    n = 120
    t = np.linspace(0.0, 8.0, n)
    pos_ref = np.column_stack([t, np.sin(t), 0.2 * np.cos(0.5 * t)])
    quat_ref = R.from_euler("z", 0.05 * t).as_quat()
    pos_est = pos_ref.copy()
    quat_est = quat_ref.copy()
    drift_start = 55
    pos_est[drift_start:] = -18.0 * pos_ref[drift_start:] + np.array([8.0, -3.0, 0.0])

    try:
        solve_epica_sim3_variant(
            pos_ref=pos_ref,
            quat_ref=quat_ref,
            pos_est=pos_est,
            quat_est=quat_est,
            method="epica_sim3",
        )
    except ValueError as exc:
        assert "non-positive scale" in str(exc)
    else:
        raise AssertionError("full-trajectory EPICA should reject the negative global scale")

    scale, r_fit, t_fit, info = solve_epica_sim3_variant(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est,
        method="epica_sim3_stable",
    )

    assert info["sim3_solver"] == "epica_sim3_stable"
    assert info["sim3_anchor_status"] == "ok"
    assert info["sim3_reliable"] is True
    assert int(info["sim3_anchor_start_index"]) == 0
    assert int(info["sim3_anchor_end_index"]) <= drift_start
    np.testing.assert_allclose(scale, 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(r_fit, np.eye(3), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(t_fit, np.zeros(3), rtol=1e-6, atol=1e-6)


def test_epica_joint_grid_trades_position_for_orientation() -> None:
    n = 80
    t = np.linspace(0.0, 8.0, n)
    pos_ref = np.column_stack([np.cos(t), 0.4 * t, np.sin(0.5 * t)])
    quat_ref = R.from_euler("z", 0.15 * t).as_quat()

    r_position = R.from_euler("z", 90.0, degrees=True)
    scale_true = 1.7
    t_position = np.array([1.0, -0.5, 0.2], dtype=float)
    pos_est = (r_position.inv().as_matrix() @ ((pos_ref - t_position) / scale_true).T).T
    quat_est = quat_ref.copy()

    pos_joint, quat_joint, info = align_for_eval_with_info(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est,
        mode="epica_sim3_joint",
    )
    joint_ape = compute_ape(pos_ref, quat_ref, pos_joint, quat_joint)

    assert info["sim3_solver"] == "epica_joint_grid"
    assert joint_ape["translation_part"]["rmse"] < 0.5
    assert joint_ape["rotation_angle_deg"]["rmse"] <= 90.0 + 1e-9


def test_main_sim3_uses_pose_consistent_solve_when_available(monkeypatch) -> None:
    pos_ref = np.zeros((4, 3), dtype=float)
    quat_ref = R.identity(4).as_quat()
    pos_est = np.column_stack([np.arange(4.0), np.zeros((4, 2))])
    quat_est = quat_ref.copy()

    def fake_pose_consistent(*_args, **_kwargs):
        return 2.0, np.eye(3), np.array([1.0, 0.0, 0.0]), {"sim3_solver": "pose_consistent"}

    monkeypatch.setattr("epa.core.sim3.solve_orientation_consistent_sim3", fake_pose_consistent)
    scale, r_fit, t_fit, info = solve_epa_sim3(pos_ref, quat_ref, pos_est, quat_est)

    np.testing.assert_allclose(scale, 2.0)
    np.testing.assert_allclose(r_fit, np.eye(3))
    np.testing.assert_allclose(t_fit, np.array([1.0, 0.0, 0.0]))
    assert info["sim3_solver"] == "sim3"
    assert info["sim3_pose_consistent_solve_used"] is True
    assert info["sim3_robust_fallback_used"] is False

    pos_new, _, info = align_for_eval_with_info(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est,
        mode="sim3",
    )

    np.testing.assert_allclose(pos_new, 2.0 * pos_est + np.array([1.0, 0.0, 0.0]))
    assert info["align_mode"] == "sim3"
    assert info["sim3_solver"] == "sim3"
    assert info["sim3_robust_fallback_used"] is False


def test_main_sim3_uses_stable_anchor_when_pose_consistent_fails(monkeypatch) -> None:
    pos_ref = np.zeros((4, 3), dtype=float)
    quat_ref = R.identity(4).as_quat()
    pos_est = np.column_stack([np.arange(4.0), np.zeros((4, 2))])
    quat_est = quat_ref.copy()

    def fake_pose_consistent(*_args, **_kwargs):
        raise ValueError("primary failed")

    def fake_stable(*_args, **_kwargs):
        return 3.0, np.eye(3), np.array([0.0, 1.0, 0.0]), {"sim3_solver": "epica_sim3_stable", "sim3_reliable": True}

    monkeypatch.setattr("epa.core.sim3.solve_orientation_consistent_sim3", fake_pose_consistent)
    monkeypatch.setattr("epa.core.sim3.solve_stable_epica_sim3", fake_stable)
    scale, r_fit, t_fit, info = solve_epa_sim3(pos_ref, quat_ref, pos_est, quat_est)

    np.testing.assert_allclose(scale, 3.0)
    np.testing.assert_allclose(r_fit, np.eye(3))
    np.testing.assert_allclose(t_fit, np.array([0.0, 1.0, 0.0]))
    assert info["sim3_solver"] == "sim3"
    assert info["sim3_pose_consistent_solve_used"] is False
    assert info["sim3_pose_consistent_failure"] == "primary failed"
    assert info["sim3_stable_anchor_used"] is True
    assert info["sim3_robust_fallback_used"] is False

    monkeypatch.setattr("epa.metric_cli_common.solve_epa_sim3", lambda **_kwargs: (scale, r_fit, t_fit, info))
    pos_new, _, info = align_for_eval_with_info(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est,
        mode="sim3",
    )

    np.testing.assert_allclose(pos_new, 3.0 * pos_est + np.array([0.0, 1.0, 0.0]))
    assert info["sim3_solver"] == "sim3"
    assert info["sim3_stable_anchor_used"] is True
    assert info["sim3_robust_fallback_used"] is False


def test_main_sim3_uses_stable_anchor_when_pose_consistent_scale_is_severe(monkeypatch) -> None:
    pos_ref = np.zeros((4, 3), dtype=float)
    quat_ref = R.identity(4).as_quat()
    pos_est = np.column_stack([np.arange(4.0), np.zeros((4, 2))])
    quat_est = quat_ref.copy()

    def fake_pose_consistent(*_args, **_kwargs):
        return 0.01, np.eye(3), np.zeros(3), {"sim3_solver": "pose_consistent", "sim3_reliable": True}

    def fake_stable(*_args, **_kwargs):
        return 0.3, np.eye(3), np.array([0.0, 1.0, 0.0]), {"sim3_solver": "epica_sim3_stable", "sim3_reliable": True}

    monkeypatch.setattr("epa.core.sim3.solve_orientation_consistent_sim3", fake_pose_consistent)
    monkeypatch.setattr("epa.core.sim3.solve_stable_epica_sim3", fake_stable)
    scale, r_fit, t_fit, info = solve_epa_sim3(pos_ref, quat_ref, pos_est, quat_est)

    np.testing.assert_allclose(scale, 0.3)
    np.testing.assert_allclose(r_fit, np.eye(3))
    np.testing.assert_allclose(t_fit, np.array([0.0, 1.0, 0.0]))
    assert info["sim3_solver"] == "sim3"
    assert info["sim3_pose_consistent_solve_used"] is False
    assert "severe-scale" in info["sim3_pose_consistent_failure"]
    assert info["sim3_stable_anchor_used"] is True
    assert info["sim3_robust_fallback_used"] is False


def test_main_sim3_uses_robust_fallback_when_stable_anchor_fails(monkeypatch) -> None:
    pos_ref = np.zeros((4, 3), dtype=float)
    quat_ref = R.identity(4).as_quat()
    pos_est = np.column_stack([np.arange(4.0), np.zeros((4, 2))])
    quat_est = quat_ref.copy()

    def fake_pose_consistent(*_args, **_kwargs):
        raise ValueError("primary failed")

    def fake_stable(*_args, **_kwargs):
        raise ValueError("stable failed")

    def fake_epa_v2(*_args, **_kwargs):
        return 1.0, np.eye(3), np.zeros(3), {"sim3_solver": "epa_sim3_v2", "sim3_reliable": True}

    monkeypatch.setattr("epa.core.sim3.solve_orientation_consistent_sim3", fake_pose_consistent)
    monkeypatch.setattr("epa.core.sim3.solve_stable_epica_sim3", fake_stable)
    monkeypatch.setattr("epa.core.sim3.solve_epa_sim3_v2", fake_epa_v2)
    scale, r_fit, t_fit, info = solve_epa_sim3(pos_ref, quat_ref, pos_est, quat_est)

    np.testing.assert_allclose(scale, 1.0)
    np.testing.assert_allclose(r_fit, np.eye(3))
    np.testing.assert_allclose(t_fit, np.zeros(3))
    assert info["sim3_solver"] == "sim3"
    assert info["sim3_pose_consistent_solve_used"] is False
    assert info["sim3_stable_anchor_used"] is False
    assert info["sim3_robust_fallback_used"] is True
    assert info["sim3_robust_fallback_reason"] == "primary failed; stable failed"


def test_main_sim3_uses_robust_fallback_when_stable_anchor_scale_is_severe(monkeypatch) -> None:
    pos_ref = np.zeros((4, 3), dtype=float)
    quat_ref = R.identity(4).as_quat()
    pos_est = np.column_stack([np.arange(4.0), np.zeros((4, 2))])
    quat_est = quat_ref.copy()

    def fake_pose_consistent(*_args, **_kwargs):
        raise ValueError("primary failed")

    def fake_stable(*_args, **_kwargs):
        return 0.01, np.eye(3), np.zeros(3), {"sim3_solver": "epica_sim3_stable", "sim3_reliable": True}

    def fake_epa_v2(*_args, **_kwargs):
        return 1.0, np.eye(3), np.zeros(3), {"sim3_solver": "epa_sim3_v2", "sim3_reliable": True}

    monkeypatch.setattr("epa.core.sim3.solve_orientation_consistent_sim3", fake_pose_consistent)
    monkeypatch.setattr("epa.core.sim3.solve_stable_epica_sim3", fake_stable)
    monkeypatch.setattr("epa.core.sim3.solve_epa_sim3_v2", fake_epa_v2)
    scale, r_fit, t_fit, info = solve_epa_sim3(pos_ref, quat_ref, pos_est, quat_est)

    np.testing.assert_allclose(scale, 1.0)
    np.testing.assert_allclose(r_fit, np.eye(3))
    np.testing.assert_allclose(t_fit, np.zeros(3))
    assert info["sim3_solver"] == "sim3"
    assert info["sim3_stable_anchor_used"] is False
    assert info["sim3_robust_fallback_used"] is True
    assert "severe-scale" in info["sim3_robust_fallback_reason"]


def test_main_sim3_keeps_pose_consistent_when_only_scale_guard_fails_and_stable_is_bad(monkeypatch) -> None:
    pos_ref = np.zeros((4, 3), dtype=float)
    quat_ref = R.identity(4).as_quat()
    pos_est = np.column_stack([np.arange(4.0), np.zeros((4, 2))])
    quat_est = quat_ref.copy()

    def fake_pose_consistent(*_args, **_kwargs):
        return 0.01, np.eye(3), np.array([1.0, 0.0, 0.0]), {"sim3_solver": "pose_consistent", "sim3_reliable": True}

    def fake_stable(*_args, **_kwargs):
        return 0.02, np.eye(3), np.array([0.0, 1.0, 0.0]), {"sim3_solver": "epica_sim3_stable", "sim3_reliable": False}

    def fake_epa_v2(*_args, **_kwargs):
        return 1.0, np.eye(3), np.zeros(3), {"sim3_solver": "epa_sim3_v2", "sim3_reliable": True}

    monkeypatch.setattr("epa.core.sim3.solve_orientation_consistent_sim3", fake_pose_consistent)
    monkeypatch.setattr("epa.core.sim3.solve_stable_epica_sim3", fake_stable)
    monkeypatch.setattr("epa.core.sim3.solve_epa_sim3_v2", fake_epa_v2)
    scale, r_fit, t_fit, info = solve_epa_sim3(pos_ref, quat_ref, pos_est, quat_est)

    np.testing.assert_allclose(scale, 0.01)
    np.testing.assert_allclose(r_fit, np.eye(3))
    np.testing.assert_allclose(t_fit, np.array([1.0, 0.0, 0.0]))
    assert info["sim3_solver"] == "sim3"
    assert info["sim3_pose_consistent_solve_used"] is True
    assert "severe-scale" in info["sim3_pose_consistent_scale_guard_warning"]
    assert info["sim3_stable_anchor_used"] is False
    assert info["sim3_robust_fallback_used"] is False


def test_anchor_sim3_exposes_terminal_drift() -> None:
    n = 120
    t = np.linspace(0.0, 12.0, n)
    pos_ref = np.column_stack([t, np.sin(0.4 * t), 0.1 * np.cos(t)])
    quat_ref = R.from_euler("z", 0.05 * t).as_quat()

    scale_true = 2.0
    r_fit_true = R.from_euler("z", 20.0, degrees=True)
    t_fit_true = np.array([0.5, -1.0, 0.2], dtype=float)
    pos_est = (r_fit_true.inv().as_matrix() @ ((pos_ref - t_fit_true) / scale_true).T).T
    quat_est = (r_fit_true.inv() * R.from_quat(quat_ref)).as_quat()

    drift = np.zeros_like(pos_est)
    drift[40:, 1] = np.linspace(0.0, 8.0, n - 40)
    pos_est = pos_est + drift

    pos_ov, quat_ov, _ = align_for_eval_with_info(pos_ref, quat_ref, pos_est, quat_est, mode="ov_sim3")
    pos_anchor, quat_anchor, anchor_info = align_for_eval_with_info(
        pos_ref,
        quat_ref,
        pos_est,
        quat_est,
        mode="epica_anchor_sim3",
    )

    ov_errors = np.linalg.norm(pos_ov - pos_ref, axis=1)
    anchor_errors = np.linalg.norm(pos_anchor - pos_ref, axis=1)

    assert anchor_info["sim3_solver"] == "epica_anchor_sim3"
    assert anchor_errors[:20].mean() < ov_errors[:20].mean()
    assert anchor_errors[-20:].mean() > ov_errors[-20:].mean()


def test_epa_sim3_v1_selects_reliable_window_without_using_drift_as_reward() -> None:
    n = 180
    t = np.linspace(0.0, 18.0, n)
    pos_ref = np.column_stack([0.6 * t, np.sin(0.4 * t), 0.4 * np.cos(0.2 * t)])
    quat_ref = R.from_euler("zyx", np.column_stack([0.05 * t, 0.02 * np.sin(t), 0.01 * t])).as_quat()

    scale_true = 1.9
    r_fit_true = R.from_euler("zyx", [25.0, -7.0, 4.0], degrees=True)
    t_fit_true = np.array([0.5, -0.8, 0.3], dtype=float)
    pos_est = (r_fit_true.inv().as_matrix() @ ((pos_ref - t_fit_true) / scale_true).T).T
    quat_est = (r_fit_true.inv() * R.from_quat(quat_ref)).as_quat()

    pos_est_drift = pos_est.copy()
    pos_est_drift[90:, 1] += np.linspace(0.0, 9.0, n - 90)

    scale, r_fit, t_fit, info = solve_epa_sim3_v1(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est_drift,
        quat_est=quat_est,
        min_window_samples=30,
    )
    pos_epa, _, epa_info = align_for_eval_with_info(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est_drift,
        quat_est=quat_est,
        mode="epa_sim3_v1",
    )
    pos_ov, _, _ = align_for_eval_with_info(pos_ref, quat_ref, pos_est_drift, quat_est, mode="ov_sim3")

    epa_err = np.linalg.norm(pos_epa - pos_ref, axis=1)
    ov_err = np.linalg.norm(pos_ov - pos_ref, axis=1)

    assert info["sim3_solver"] == "epa_sim3_v1"
    assert epa_info["sim3_anchor_status"] == "ok"
    assert epa_info["sim3_reliable"] is True
    np.testing.assert_allclose(scale, scale_true, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(r_fit, r_fit_true.as_matrix(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(t_fit, t_fit_true, rtol=1e-5, atol=1e-5)
    assert epa_err[:60].mean() < ov_err[:60].mean()
    assert epa_err[-40:].mean() > ov_err[-40:].mean()


def test_epa_sim3_v2_requires_anchor_consensus() -> None:
    n = 220
    t = np.linspace(0.0, 22.0, n)
    pos_ref = np.column_stack([0.5 * t, np.sin(0.3 * t), 0.5 * np.cos(0.2 * t)])
    quat_ref = R.from_euler("zyx", np.column_stack([0.04 * t, 0.02 * np.sin(t), 0.01 * t])).as_quat()

    scale_true = 1.7
    r_fit_true = R.from_euler("zyx", [18.0, -5.0, 3.0], degrees=True)
    t_fit_true = np.array([0.3, -0.4, 0.2], dtype=float)
    pos_est = (r_fit_true.inv().as_matrix() @ ((pos_ref - t_fit_true) / scale_true).T).T
    quat_est = (r_fit_true.inv() * R.from_quat(quat_ref)).as_quat()
    pos_est[160:, 1] += np.linspace(0.0, 7.0, n - 160)

    scale, r_fit, t_fit, info = solve_epa_sim3_v2(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est,
        min_window_samples=30,
    )

    assert info["sim3_solver"] == "epa_sim3_v2"
    assert info["sim3_confidence"] in {"high", "medium"}
    assert info["sim3_global_support_window_count"] >= 2
    assert info["sim3_global_support_distance_ratio"] > 0.5
    assert 1.0 < scale < 2.0
    pos_aligned = scale * (r_fit @ pos_est.T).T + t_fit
    healthy_prefix_rmse = np.sqrt(np.mean(np.sum((pos_aligned[:150] - pos_ref[:150]) ** 2, axis=1)))
    drift_suffix_rmse = np.sqrt(np.mean(np.sum((pos_aligned[170:] - pos_ref[170:]) ** 2, axis=1)))
    assert healthy_prefix_rmse < 1.5
    assert drift_suffix_rmse > healthy_prefix_rmse


def test_epa_sim3_v2_declares_no_reliable_anchor_for_degenerate_motion() -> None:
    n = 50
    t = np.linspace(0.0, 1.0, n)
    pos_ref = np.zeros((n, 3), dtype=float)
    pos_est = np.zeros((n, 3), dtype=float)
    quat_ref = R.from_euler("z", 0.0 * t).as_quat()
    quat_est = quat_ref.copy()

    scale, r_fit, t_fit, info = solve_epa_sim3_v2(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est,
        min_window_samples=10,
    )

    assert scale == 1.0
    np.testing.assert_allclose(r_fit, np.eye(3))
    np.testing.assert_allclose(t_fit, np.zeros(3))
    assert info["sim3_solver"] == "epa_sim3_v2"
    assert info["sim3_anchor_status"] == "no_reliable_anchor"
    assert info["sim3_confidence"] == "no_anchor"
    assert info["sim3_reliable"] is False
