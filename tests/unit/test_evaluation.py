from __future__ import annotations

import numpy as np
import pytest

from epa.core.evaluation import (
    build_rpe_pairs,
    compute_ape_evo_style,
    compute_rpe_evo_style,
    normalize_pose_relation,
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


def test_compute_metrics_can_include_raw_arrays() -> None:
    pos, quat = _identity_traj()
    ape = compute_ape_evo_style(pos, quat, pos, quat, include_raw=True)
    rpe = compute_rpe_evo_style(pos, quat, pos, quat, delta=1, include_raw=True)

    assert "_error_arrays" in ape
    assert ape["_error_arrays"]["translation_part"].shape[0] == pos.shape[0]
    assert "_error_arrays" in rpe
    assert rpe["_error_arrays"]["translation_part"].shape[0] == rpe["pair_count"]


def test_normalize_pose_relation_accepts_evo_aliases() -> None:
    assert normalize_pose_relation("ape", "trans_part") == "translation_part"
    assert normalize_pose_relation("rpe", "angle_deg") == "rotation_angle_deg"
    assert normalize_pose_relation("rpe", "point_distance_error_ratio") == "point_distance_error_ratio"
