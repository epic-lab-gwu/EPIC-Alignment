from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from epa.core.sim3 import solve_orientation_consistent_sim3
from epa.core.sim3_utils import _solve_scale_translation_fixed_rotation
from epa.metric_cli_common import align_for_eval_with_info


def _case(n=150):
    t = np.linspace(0, 9, n)
    pos_est = np.column_stack((t, np.sin(t), np.cos(0.6 * t)))
    quat_est = R.from_euler("zyx", np.column_stack((t, 0.3 * np.sin(t), 0.1 * t))).as_quat()
    rotation = R.from_euler("zyx", [35, -12, 8], degrees=True)
    scale = 2.4
    translation = np.array([0.4, -0.7, 0.2])
    pos_ref = scale * rotation.apply(pos_est) + translation
    quat_ref = (rotation * R.from_quat(quat_est)).as_quat()
    return pos_ref, quat_ref, pos_est, quat_est, scale, rotation, translation


@pytest.mark.parametrize("rotation_jump,position_jump", [(False, False), (True, False), (True, True)])
def test_sim3_shares_rotation_mask_without_hiding_evaluation_errors(rotation_jump, position_jump):
    pr, qr, pe, qe, scale_true, rotation_true, translation_true = _case()
    if rotation_jump:
        qe[:30] = (R.from_euler("x", 120, degrees=True) * R.from_quat(qe[:30])).as_quat()
    if position_jump:
        pe[:30] += np.array([1000, -700, 400])
    qe[::2] *= -1  # Quaternion signs and genuine shared fast motion are harmless.

    scale, rotation, translation, info = solve_orientation_consistent_sim3(pr, qr, pe, qe)
    np.testing.assert_allclose(scale, scale_true, atol=1e-10)
    np.testing.assert_allclose(rotation, rotation_true.as_matrix(), atol=1e-10)
    np.testing.assert_allclose(translation, translation_true, atol=1e-10)
    assert info["sim3_rotation_rejected_count"] == (30 if rotation_jump else 0)
    assert info["sim3_position_rejected_count"] == 0
    assert info["sim3_position_input_count"] == (120 if rotation_jump else 150)
    assert info["sim3_position_rotation_excluded_count"] == (30 if rotation_jump else 0)
    assert info["sim3_position_total_rejected_count"] == (30 if rotation_jump else 0)
    assert info["sim3_pair_count"] == len(pr)
    assert info["sim3_rotation_threshold_deg"] >= 1
    assert info["sim3_position_mad_filter_enabled"] is False

    aligned, orientations, _ = align_for_eval_with_info(pr, qr, pe, qe, mode="sim3")
    assert aligned.shape == pr.shape
    assert orientations.shape == qr.shape
    if position_jump:
        assert not np.allclose(aligned[:30], pr[:30])
        assert info["sim3_position_rmse_m"] > 100
    if rotation_jump:
        assert info["sim3_orientation_rmse_deg"] > 40


def test_sim3_rotation_outliers_cannot_enter_scale_translation_fit(monkeypatch):
    pr, qr, pe, qe, scale_true, rotation_true, translation_true = _case()
    qe[:30] = (R.from_euler("x", 120, degrees=True) * R.from_quat(qe[:30])).as_quat()
    # These positions must be excluded from the default scale/translation fit.
    pe[:30] += [0.1, -0.2, 0.3]
    inputs = []

    def checked_fit(pos_ref, pos_est, rotation):
        inputs.append(len(pos_ref))
        np.testing.assert_array_equal(pos_ref, pr[30:])
        np.testing.assert_array_equal(pos_est, pe[30:])
        return _solve_scale_translation_fixed_rotation(pos_ref, pos_est, rotation)

    monkeypatch.setattr("epa.core.sim3._solve_scale_translation_fixed_rotation", checked_fit)
    scale, rotation, translation, info = solve_orientation_consistent_sim3(pr, qr, pe, qe)
    assert inputs == [120]
    np.testing.assert_allclose(scale, scale_true, atol=1e-12)
    np.testing.assert_allclose(rotation, rotation_true.as_matrix(), atol=1e-12)
    np.testing.assert_allclose(translation, translation_true, atol=1e-12)
    assert info["sim3_position_rejected_count"] == 0
    assert info["sim3_position_total_rejected_count"] == 30
    assert info["sim3_position_mad_filter_enabled"] is False


def test_sim3_rejects_negative_scale_without_position_trimming():
    pr, qr, pe, qe, _, rotation_true, _ = _case()
    pe[-30:] *= -100
    with pytest.raises(ValueError, match="non-positive scale"):
        _solve_scale_translation_fixed_rotation(pr, pe, rotation_true.as_matrix())
    with pytest.raises(ValueError, match="non-positive scale"):
        solve_orientation_consistent_sim3(pr, qr, pe, qe)


def test_sim3_preserves_small_position_noise_and_ordinary_fit():
    pr, qr, pe, qe, _, rotation_true, _ = _case()
    pr += np.random.default_rng(5).uniform(-0.0001, 0.0001, pr.shape)
    expected_scale, expected_translation = _solve_scale_translation_fixed_rotation(pr, pe, rotation_true.as_matrix())
    scale, rotation, translation, info = solve_orientation_consistent_sim3(pr, qr, pe, qe)
    np.testing.assert_allclose(scale, expected_scale, atol=1e-12)
    np.testing.assert_allclose(translation, expected_translation, atol=1e-12)
    np.testing.assert_allclose(rotation, rotation_true.as_matrix(), atol=1e-12)
    assert info["sim3_rotation_rejected_count"] == 0
    assert info["sim3_position_rejected_count"] == 0


def test_sim3_two_pair_scale_translation_helper_keeps_original_fit():
    pr, qr, pe, qe, scale_true, rotation_true, translation_true = _case(n=2)
    scale, translation = _solve_scale_translation_fixed_rotation(pr, pe, rotation_true.as_matrix())
    np.testing.assert_allclose(scale, scale_true, atol=1e-12)
    np.testing.assert_allclose(translation, translation_true, atol=1e-12)
    with pytest.raises(ValueError, match="at least 3 paired samples"):
        solve_orientation_consistent_sim3(pr, qr, pe, qe)


def test_sim3_rejects_degenerate_position_consensus():
    pr, qr, pe, qe, *_ = _case()
    pr[:] = [1, 2, 3]
    pe[:] = [4, 5, 6]
    with pytest.raises(ValueError, match="Degenerate source trajectory"):
        solve_orientation_consistent_sim3(pr, qr, pe, qe)


def test_default_sim3_keeps_position_outliers_with_valid_orientations():
    pr, qr, pe, qe, _, rotation_true, _ = _case()
    pe[-30:] += [1000, -700, 400]
    expected_scale, expected_translation = _solve_scale_translation_fixed_rotation(pr, pe, rotation_true.as_matrix())
    scale, rotation, translation, info = solve_orientation_consistent_sim3(pr, qr, pe, qe)
    np.testing.assert_allclose(scale, expected_scale, atol=1e-12)
    np.testing.assert_allclose(translation, expected_translation, atol=1e-12)
    np.testing.assert_allclose(rotation, rotation_true.as_matrix(), atol=1e-12)
    assert info["sim3_position_mad_filter_enabled"] is False
    assert info["sim3_position_inlier_count"] == len(pr)
    assert info["sim3_position_rejected_count"] == 0
