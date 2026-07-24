import numpy as np
from scipy.spatial.transform import Rotation as R

from epa.core.calibration import solve_world_alignment
from epa.core.diagnostics import _compute_rigid_alignability


def test_rigid_alignability_positive_case() -> None:
    n = 300
    t = np.linspace(0.0, 30.0, n)
    x = np.linspace(0.0, 12.0, n)
    y = 0.8 * np.sin(np.linspace(0.0, 5.0, n))
    z = 0.3 * np.cos(np.linspace(0.0, 2.5, n))
    pos_ref = np.column_stack([x, y, z])

    R_true = R.from_euler("zyx", [20.0, -8.0, 5.0], degrees=True).as_matrix()
    t_true = np.array([0.7, -0.3, 0.2], dtype=float)
    pos_step2 = (R_true.T @ (pos_ref - t_true).T).T
    pos_step2 += 0.002 * np.random.default_rng(3).normal(size=pos_step2.shape)

    Rw, tw = solve_world_alignment(pos_step2, pos_ref)
    pos_step3 = (Rw @ pos_step2.T).T + tw

    out = _compute_rigid_alignability(
        t_ref=t,
        pos_ref=pos_ref,
        pos_step2=pos_step2,
        pos_step3=pos_step3,
        segment_duration_s=8.0,
        overlap_ratio=0.5,
        min_samples=40,
        max_path_ratio=3.0,
        max_bbox_ratio=3.0,
        max_global_local_ratio=6.0,
        max_sim3_gain_ratio=0.3,
    )

    assert out["rigid_alignability_code"] == 1.0
    assert out["_rigid_alignability_label"] == "rigidly_alignable"
    assert out["rigid_check_fail_count"] == 0.0
    assert out["scale_mismatch_severe_code"] == 0.0
    assert "scale_mismatch_severe" not in out["_rigid_alignability_reasons"]


def test_rigid_alignability_rejects_large_scale_mismatch() -> None:
    n = 250
    t = np.linspace(0.0, 25.0, n)
    x = np.linspace(0.0, 15.0, n)
    y = 0.6 * np.sin(np.linspace(0.0, 4.0, n))
    z = 0.4 * np.cos(np.linspace(0.0, 3.0, n))
    pos_ref = np.column_stack([x, y, z])

    # Large scale mismatch cannot be explained by single SE3.
    pos_step2 = 50.0 * pos_ref + np.array([200.0, -100.0, 40.0], dtype=float)
    Rw, tw = solve_world_alignment(pos_step2, pos_ref)
    pos_step3 = (Rw @ pos_step2.T).T + tw

    out = _compute_rigid_alignability(
        t_ref=t,
        pos_ref=pos_ref,
        pos_step2=pos_step2,
        pos_step3=pos_step3,
        segment_duration_s=8.0,
        overlap_ratio=0.5,
        min_samples=40,
        max_path_ratio=3.0,
        max_bbox_ratio=3.0,
        max_global_local_ratio=6.0,
        max_sim3_gain_ratio=0.3,
    )

    assert out["rigid_alignability_code"] == 0.0
    assert out["_rigid_alignability_label"] == "not_rigidly_alignable"
    assert out["rigid_check_fail_count"] >= 1.0
    assert out["scale_mismatch_severe_code"] == 1.0
    assert "scale_mismatch_severe" in out["_rigid_alignability_reasons"]
