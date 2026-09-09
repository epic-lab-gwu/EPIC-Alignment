"""Regression coverage for time-bounded Sim3 extrinsic calibration pairs."""
import numpy as np
from scipy.spatial.transform import Rotation as R
import epa.metric_cli_common as alignment
from epa.compat.ov_eval.evaluate import _evaluate_pair


def test_sim3_evaluation_calibrates_on_matched_timestamps(tmp_path, monkeypatch):
    t = np.linspace(1000.0, 1012.0, 160)
    t[80:] += 5.0
    phase = np.linspace(0.0, 12.0, t.size)
    pos = np.column_stack([0.6 * phase, np.sin(0.35 * phase), 0.3 * np.cos(0.5 * phase)])
    quat = R.from_euler('zyx', np.column_stack([0.4 * phase, 0.8 * np.sin(0.6 * phase), 0.7 * np.cos(0.4 * phase)])).as_quat()
    world = R.from_euler('zyx', [25.0, -7.0, 4.0], degrees=True)
    body = R.from_euler('xyz', [150.0, -18.0, 32.0], degrees=True)
    scale = 1.8
    est_pos = world.inv().apply((pos - [0.5, -0.8, 0.3]) / scale)
    est_quat = (world.inv() * R.from_quat(quat) * body).as_quat()
    selected = slice(3, -4)
    gt, est = tmp_path / 'gt.txt', tmp_path / 'est.txt'
    np.savetxt(gt, np.column_stack([t, pos, quat]))
    np.savetxt(est, np.column_stack([t[selected] + 0.001, est_pos[selected], est_quat[selected]]))
    original = alignment.solve_extrinsic_rotation_multibaseline
    checked = []

    def check_pairs(q_ref, q_est, **kwargs):
        timestamps = kwargs.get('timestamps_s')
        np.testing.assert_allclose(timestamps, t[selected])
        result = original(q_ref, q_est, **kwargs)
        pairs = result[1]
        durations = timestamps[pairs[:, 1]] - timestamps[pairs[:, 0]]
        assert np.all((durations > 0.0) & (durations <= 2.0))
        checked.append(len(pairs))
        return result

    monkeypatch.setattr(alignment, 'solve_extrinsic_rotation_multibaseline', check_pairs)
    result = _evaluate_pair(gt, est, 'sim3', 0.02, epa_no_fallback=True, epa_disable_time_offset_calibration=True)
    assert checked and checked[0] > 0
    assert result['eval_alignment']['sim3_extrinsic_rotation_correction_used']
    np.testing.assert_allclose(result['eval_alignment']['align_scale'], scale, atol=1e-8)
    assert result['ate3_ori']['rmse'] < 1e-7
    assert result['ate3_pos']['rmse'] < 1e-7
