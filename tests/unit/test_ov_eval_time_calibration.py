from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from epa.compat.ov_eval import evaluate


def _write_trajectory(path: Path, times: np.ndarray, offset: float = 0.0, scale: float = 1.0):
    positions = np.column_stack(
        [np.sin(0.31 * times), np.cos(0.17 * times), 0.05 * times]
    )
    quaternions = Rotation.from_euler(
        "xyz",
        np.column_stack(
            [0.2 * np.sin(times), 0.1 * np.cos(0.7 * times), 0.08 * times]
        ),
    ).as_quat()
    np.savetxt(path, np.column_stack([times + offset, positions / scale, quaternions]))


@pytest.mark.parametrize("mode", ["se3", "se3-original", "posyaw", "sim3"])
def test_comparison_applies_step1_once_and_exposes_confidence(tmp_path, monkeypatch, mode):
    times = np.arange(301, dtype=float) * 0.1
    gt_path, est_path = tmp_path / "gt.tum", tmp_path / "est.tum"
    _write_trajectory(gt_path, times)
    _write_trajectory(est_path, times, offset=2.3, scale=2.5 if mode == "sim3" else 1.0)
    metrics = {"offset_est_s": 2.3, "xcorr_peak_normalized": 0.91, "xcorr_psr": 12.0}
    calls = []

    def time_alignment(**kwargs):
        calls.append(kwargs)
        return {"calculated_offset": 2.3, "time_metrics": metrics}

    monkeypatch.setattr(evaluate, "_run_time_alignment", time_alignment)
    result = evaluate._evaluate_pair(
        gt_path, est_path, mode, 0.02,
        epa_dt_resample=0.01,
        epa_offset_min_match_ratio=0.4,
        epa_disable_extrinsic_calibration=True,
        epa_no_fallback=True,
    )

    assert len(calls) == 1
    assert calls[0]["dt_resample"] == 0.01
    assert calls[0]["offset_min_match_ratio"] == 0.4
    np.testing.assert_allclose(calls[0]["t_est"], times + 2.3)
    assert result["calculated_offset"] == 2.3
    assert result["time_metrics"] == metrics
    assert result["matched"] == len(times)
    np.testing.assert_allclose(result["gt_t"], times)
    assert result["ate3_pos"]["rmse"] < 1e-6
    assert result["ate3_ori"]["rmse"] < 1e-6
    assert result["eval_alignment"]["extrinsic_calibration_disabled"] is True
    if mode == "sim3":
        assert result["eval_alignment"]["align_scale"] == pytest.approx(2.5)
        assert result["eval_alignment"]["sim3_extrinsic_rotation_correction_used"] is False


@pytest.mark.parametrize("disable_all", [False, True])
@pytest.mark.parametrize("mode", ["se3", "se3-original", "posyaw", "sim3"])
def test_disabled_time_calibration_skips_solver_for_short_trajectories(
    tmp_path, monkeypatch, disable_all, mode,
):
    # Valid for spatial alignment, but too short for Step1's 100 resampled points.
    times = np.arange(6, dtype=float) * 0.05
    gt_path, est_path = tmp_path / "gt.tum", tmp_path / "est.tum"
    _write_trajectory(gt_path, times)
    _write_trajectory(est_path, times, scale=2.5 if mode == "sim3" else 1.0)

    def unexpected_time_alignment(**kwargs):
        pytest.fail("Disabled time calibration must skip Step1")

    def unexpected_extrinsic_solve(*args, **kwargs):
        pytest.fail("Sim3 must not calibrate extrinsics when disabled")

    monkeypatch.setattr(evaluate, "_run_time_alignment", unexpected_time_alignment)
    monkeypatch.setattr("epa.metric_cli_common.solve_extrinsic_rotation", unexpected_extrinsic_solve)
    result = evaluate._evaluate_pair(
        gt_path, est_path, mode, 0.02,
        epa_disable_calibration=disable_all,
        epa_disable_time_offset_calibration=not disable_all,
        epa_disable_extrinsic_calibration=not disable_all,
    )

    assert result["calculated_offset"] == 0.0
    assert result["time_metrics"]["offset_est_s"] == 0.0
    assert result["time_metrics"]["evo_t_offset_used_s"] == 0.0
    assert result["time_metrics"]["time_offset_calibration_disabled"] == 1.0
    assert result["eval_alignment"]["time_offset_calibration_disabled"] is True
    assert result["eval_alignment"]["extrinsic_calibration_disabled"] is True
    assert result["ate3_pos"]["rmse"] < 1e-6


def test_sim3_resampled_fallback_receives_shifted_timestamps(tmp_path, monkeypatch):
    t_gt = np.arange(0.0, 30.01, 1.0)
    t_est = np.arange(0.025, 30.0, 0.05)
    gt_path, est_path = tmp_path / "gt.tum", tmp_path / "est.tum"
    _write_trajectory(gt_path, t_gt)
    _write_trajectory(est_path, t_est, offset=2.0, scale=2.5)
    monkeypatch.setattr(
        evaluate, "_run_time_alignment",
        lambda **kwargs: {"calculated_offset": 2.0, "time_metrics": {"offset_est_s": 2.0}},
    )
    result = evaluate._evaluate_pair(
        gt_path, est_path, "sim3", 0.02,
        epa_disable_extrinsic_calibration=True,
    )

    assert result["eval_alignment"]["resampled_fallback_used"] is True
    assert result["eval_alignment"]["dense_timeline_used"] is True
    assert result["calculated_offset"] == 2.0
    assert result["gt_t"][0] == 1.0
    assert result["gt_t"][-1] == 29.0
    assert result["ate3_pos"]["rmse"] < 0.01
