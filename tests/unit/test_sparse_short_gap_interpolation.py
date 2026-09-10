import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from epa.ov_align import associate_est_gt


def associate(est, gt, enabled=True, offset=0):
    est, gt = np.asarray(est, float), np.asarray(gt, float)
    p_est = np.column_stack((est, est * 0, est * 0))
    p_gt = np.column_stack((gt, gt * 0, gt * 0))
    q_est = Rotation.from_euler('z', est).as_quat()
    q_gt = Rotation.from_euler('z', gt).as_quat()
    return associate_est_gt(est, p_est, q_est, gt, p_gt, q_gt, .02, offset, enabled)


def test_gt_position_and_rotation_interpolated_at_estimate_times():
    est = np.array([.035, .135, .235])
    out = associate(est, np.arange(0, .4, .1))
    np.testing.assert_array_equal(out[0], est)
    np.testing.assert_array_equal(out[3], est)
    np.testing.assert_allclose(out[4][:, 0], est)
    np.testing.assert_allclose(Rotation.from_quat(out[5]).as_rotvec()[:, 2], est)
    np.testing.assert_array_equal(out[1][:, 0], est)
    np.testing.assert_array_equal(out[2], Rotation.from_euler('z', est).as_quat())
    np.testing.assert_array_equal(out[6], np.arange(3))


def test_dense_gt_does_not_add_estimate_poses():
    est = np.arange(0, 1.01, .1)
    out = associate(est, np.arange(0, 1.01, .01))
    np.testing.assert_array_equal(out[0], est)
    np.testing.assert_array_equal(out[6], np.arange(est.size))


def test_disabled_keeps_legacy_timestamp_gate():
    with pytest.raises(ValueError, match='Unable to associate'):
        associate([.035, .135, .235], np.arange(0, .4, .1), False)


def test_long_gt_gap_and_no_extrapolation():
    out = associate([-1, .035, .135, 5, 10.035, 11], [0, .1, .2, 10, 10.1, 10.2])
    np.testing.assert_allclose(out[0], [.035, .135, 10.035])
    np.testing.assert_array_equal(out[6], [1, 2, 4])


def test_sparse_exact_matches_preserved():
    out = associate([0, 10, 20], [0, 5, 10, 15, 20])
    np.testing.assert_array_equal(out[0], [0, 10, 20])


def test_offset_applied_to_estimate_timeline():
    est = np.array([.035, .135, .235])
    out = associate(est, np.arange(1, 1.4, .1), offset=1)
    np.testing.assert_array_equal(out[0], est + 1)
    np.testing.assert_array_equal(out[1][:, 0], est)
    np.testing.assert_allclose(out[4][:, 0], est + 1)


def test_nearby_gt_is_interpolated_instead_of_snapping_estimate_time():
    est = np.array([.01, .11, .21])
    out = associate(est, [0, .1, .2, .3])
    np.testing.assert_array_equal(out[0], est)
    np.testing.assert_allclose(out[4][:, 0], est)


def test_low_rate_gt_does_not_bridge_long_intervals():
    with pytest.raises(ValueError, match='Unable to associate'):
        associate([.025, 1.025, 2.025], [0, 1, 2, 3])
