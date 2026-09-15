import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from epa.core.evaluation import compute_ape, compute_rpe, compute_valid_segment_metrics


def evaluate(t, ref, est, qr, qe, **kwargs):
    ape = compute_ape(ref, qr, est, qe, include_raw=True)
    rpe = compute_rpe(ref, qr, est, qe, delta=1, delta_unit='s',
                      timestamps=t, all_pairs=True, include_raw=True)
    return compute_valid_segment_metrics(
        timestamps=t, pos_ref=ref, quat_ref=qr, pos_est=est, quat_est=qe,
        ape_block=ape, rpe_block=rpe, rpe_time_1s_block=rpe,
        include_masks=True, **kwargs)


def identity(n):
    return np.tile([0., 0., 0., 1.], (n, 1))


@pytest.mark.parametrize('motion,error,passing', [
    (1., 2.99, True), (1., 3., False), (1., 3.01, False),
    (.09, .299, True), (.09, .301, False),
    (.1, .301, False), (0., .299, True), (0., .3, False),
])
def test_translation_limits(motion, error, passing):
    ref = np.array([[0., 0., 0.], [motion, 0., 0.]])
    est = ref + [[0., 0., 0.], [0., error, 0.]]
    out = evaluate(np.array([0., 1.]), ref, est, identity(2), identity(2))
    assert out['success']['success_rate_time'] == float(passing)


@pytest.mark.parametrize('motion,error,passing', [
    (10., 29.9, True), (10., 30.1, False),
    (.9, 2.99, True), (.9, 3.01, False), (0., 2.99, True), (0., 3.01, False),
])
def test_rotation_limits(motion, error, passing):
    ref = np.zeros((2, 3))
    qr = R.from_euler('z', [0., motion], degrees=True).as_quat()
    qe = R.from_euler('z', [0., motion + error], degrees=True).as_quat()
    out = evaluate(np.array([0., 1.]), ref, ref, qr, qe)
    assert out['success']['success_rate_time'] == float(passing)


def test_missing_one_second_pairs_use_sequential_rpe():
    t = np.array([0., 2., 4.])
    ref = np.column_stack([t, t * 0, t * 0])
    out = evaluate(t, ref, ref, identity(3), identity(3))
    assert out['success']['success_rate_distance'] == 1
    assert out['success']['unscored_segment_count'] == 0
    assert out['success']['sequential_fallback_pair_count'] == 2
    assert out['success']['one_second_pair_count'] == 0
    assert out['rpe_time_1s']['pair_count'] == 0
    assert out['ape']['translation_part']['rmse'] == 0


@pytest.mark.parametrize('reference_duration,expected_realign', [(4., False), (8., False), (10., True)])
def test_world_alignment_uses_complete_sr_and_preserves_full_metrics(reference_duration, expected_realign):
    t = np.arange(5, dtype=float)
    ref = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.], [0., 2., 0.]])
    world = R.from_euler('z', 35, degrees=True)
    est = world.apply(ref) + [50., -20., 2.]
    qr, qe = identity(5), np.tile(world.as_quat(), (5, 1))
    full = compute_ape(ref, qr, est, qe, include_raw=True)
    out = evaluate(t, ref, est, qr, qe, input_coverage={
        'reference_path_length_m': reference_duration,
        'reference_duration_s': reference_duration, 'coverage_status': 'ok'})
    assert out['success']['valid_only_world_alignment']['applied'] == expected_realign
    assert out['success']['success_rate_distance'] == 4 / reference_duration
    assert full['translation_part']['rmse'] > 40
    if expected_realign:
        assert out['ape']['translation_part']['rmse'] < 1e-12
        assert out['ape']['rotation_angle_deg']['rmse'] < 1e-12
    else:
        assert out['ape']['translation_part']['rmse'] > 40
    assert out['rpe']['translation_part']['rmse'] < 1e-12


def test_low_sr_alignment_excludes_failed_poses_and_resolves_collinear_rotation():
    t = np.arange(6, dtype=float)
    ref = np.column_stack([t, t * 0, t * 0])
    world = R.from_euler('x', 30, degrees=True)
    est = world.apply(ref) + [20., 10., 0.]
    est[2:, 1] += np.arange(1, 5) * 20
    qr, qe = identity(6), np.tile(world.as_quat(), (6, 1))
    out = evaluate(t, ref, est, qr, qe)
    assert out['success']['success_rate_distance'] == .2
    assert out['success']['valid_only_world_alignment']['sample_count'] == 2
    assert out['ape']['translation_part']['rmse'] < 1e-12
    assert out['ape']['rotation_angle_deg']['rmse'] < 1e-12
    assert out['rpe']['pair_count'] == 1


def test_empty_sr_has_no_valid_metrics():
    out = compute_valid_segment_metrics(
        timestamps=[], pos_ref=[], ape_block={}, rpe_block={}, rpe_time_1s_block={})
    assert np.isnan(out['success']['success_rate_time'])
    assert out['rpe']['pair_count'] == 0


def test_nonfinite_rotation_rpe_fails_pair():
    t = np.array([0., 1.])
    pos = np.column_stack([t, t * 0, t * 0])
    quat = identity(2)
    ape = compute_ape(pos, quat, pos, quat, include_raw=True)
    rpe = compute_rpe(pos, quat, pos, quat, delta=1, delta_unit='s',
                      timestamps=t, all_pairs=True, include_raw=True)
    rpe['_error_arrays']['rotation_angle_deg'][0] = np.nan
    out = compute_valid_segment_metrics(timestamps=t, pos_ref=pos, ape_block=ape,
                                        rpe_block=rpe, rpe_time_1s_block=rpe)
    assert out['success']['success_rate_distance'] == 0


@pytest.mark.parametrize('duration,motion,error,passing', [
    (2., 2., 5.99, True), (2., 2., 6., False),
    (10., 10., 29.9, True), (10., 10., 30.1, False),
    (5., .09, .299, True), (5., .09, .301, False),
])
def test_sequential_fallback_uses_motion_not_error_rate(duration, motion, error, passing):
    ref = np.array([[0., 0., 0.], [motion, 0., 0.]])
    est = ref + [[0., 0., 0.], [0., error, 0.]]
    out = evaluate(np.array([0., duration]), ref, est, identity(2), identity(2))
    assert out['success']['success_rate_time'] == float(passing)
    assert out['success']['sequential_fallback_pair_count'] == 1


@pytest.mark.parametrize('motion,error,passing', [(10., 29.9, True), (10., 30.1, False), (.5, 2.99, True), (.5, 3.01, False)])
def test_sequential_fallback_rotation_limits(motion, error, passing):
    ref = np.zeros((2, 3))
    qr = R.from_euler('z', [0., motion], degrees=True).as_quat()
    qe = R.from_euler('z', [0., motion + error], degrees=True).as_quat()
    out = evaluate(np.array([0., 3.]), ref, ref, qr, qe)
    assert out['success']['success_rate_time'] == float(passing)


def test_fallback_does_not_rescue_failed_one_second_pairs():
    t = np.array([0., 1., 3., 4.])
    ref = np.column_stack([t, t * 0, t * 0])
    est = ref.copy()
    est[1:, 1] = 11.
    out = evaluate(t, ref, est, identity(4), identity(4))
    success = out['success']
    np.testing.assert_array_equal(success['valid_segment_mask'], [False, True, True])
    np.testing.assert_array_equal(success['sequential_fallback_pair_ids'], [[1, 2]])
    assert success['success_rate_distance'] == .75
    assert success['scored_pair_count'] == 3
    assert success['passing_pair_count'] == 2


def test_refit_reuses_step3_rotation_first_solver_when_positions_disagree():
    from epa.core.trajectory_alignment import _solve_extrinsic_and_world_alignment

    t = np.arange(5, dtype=float)
    ref = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.], [0., 2., 0.]])
    # Relative translation error passes SR, but position-only fitting would
    # rotate by -20 degrees despite the orientation pairs agreeing exactly.
    est = R.from_euler('z', 20, degrees=True).apply(ref) + [5., -3., 0.]
    quat = identity(5)
    out = evaluate(t, ref, est, quat, quat, input_coverage={
        'reference_path_length_m': 10., 'reference_duration_s': 10.,
        'coverage_status': 'ok'})
    expected = _solve_extrinsic_and_world_alignment(
        pr_sync=est, qr_sync=quat, pos_gt_solve=ref, quat_gt_solve=quat,
        pr_solve=est, qr_solve=quat, global_align_mode='se3r',
        disable_extrinsic_calibration=True)
    alignment = out['success']['valid_only_world_alignment']
    assert alignment['solver'] == '_solve_extrinsic_and_world_alignment'
    assert alignment['method'].startswith('rotation_first_')
    np.testing.assert_allclose(alignment['rotation'], expected['Rw_calc'], atol=1e-12)
    np.testing.assert_allclose(alignment['translation_m'], expected['tw_calc'], atol=1e-12)
    np.testing.assert_allclose(alignment['rotation'], np.eye(3), atol=1e-12)
    assert out['ape']['rotation_angle_deg']['rmse'] < 1e-12
    assert out['ape']['translation_part']['rmse'] > .1
    assert out['success']['success_rate_distance'] == .4
