import numpy as np

from epa.core.diagnostics import (
    _build_failure_diagnosis,
    _build_user_alert,
    _compute_alignment_quality,
    _compute_input_coverage_diagnostics,
)
from epa.core.pipeline_views import _compute_orientation_diagnostics


def _line_trajectory(duration_s: float = 100.0, rate_hz: float = 10.0, scale: float = 1.0):
    t = np.arange(0.0, duration_s + 1e-9, 1.0 / rate_hz)
    pos = np.column_stack([scale * t, np.zeros_like(t), np.zeros_like(t)])
    return t, pos


def test_input_coverage_detects_truncation_and_short_overlap() -> None:
    t_ref, pos_ref = _line_trajectory()
    truncated = _compute_input_coverage_diagnostics(
        t_ref=t_ref,
        pos_ref=pos_ref,
        t_est=np.arange(0.0, 60.0, 0.1),
        offset_est_s=0.0,
    )
    short = _compute_input_coverage_diagnostics(
        t_ref=t_ref,
        pos_ref=pos_ref,
        t_est=np.arange(40.0, 60.0, 0.1),
        offset_est_s=0.0,
    )

    assert truncated["coverage_status"] == "warning"
    assert 0.59 < truncated["temporal_coverage_ratio"] < 0.61
    assert short["coverage_status"] == "failed"
    assert short["path_coverage_ratio"] < 0.25


def test_input_coverage_detects_internal_reference_gap() -> None:
    t_full, pos_full = _line_trajectory()
    keep = (t_full < 35.0) | (t_full > 65.0)
    result = _compute_input_coverage_diagnostics(
        t_ref=t_full[keep],
        pos_ref=pos_full[keep],
        t_est=t_full,
        offset_est_s=0.0,
    )

    assert result["coverage_status"] == "failed"
    assert result["reference_internal_gap_ratio"] > 0.29
    assert "reference_internal_gap_critical" in result["coverage_hard_reasons"]


def test_estimate_gap_with_stationary_reference_is_exempt() -> None:
    t, pos = _line_trajectory()
    pos[:, 0] = np.where(t < 30, t, np.where(t <= 70, 30, t - 40))
    keep = (t <= 35) | (t >= 65)
    result = _compute_input_coverage_diagnostics(
        t_ref=t, pos_ref=pos, t_est=t[keep] + 7, offset_est_s=7,
    )
    assert result["coverage_status"] == "ok"
    assert result["estimate_raw_internal_gap_ratio"] > 0.25
    assert result["estimate_internal_gap_ratio"] == 0
    assert result["estimate_gap_motion_details"][0]["stationary_exempt"]


def test_estimate_gap_with_motion_still_fails() -> None:
    t, pos = _line_trajectory()
    result = _compute_input_coverage_diagnostics(
        t_ref=t, pos_ref=pos, t_est=t[(t <= 35) | (t >= 65)], offset_est_s=0,
    )
    assert "estimate_internal_gap_critical" in result["coverage_hard_reasons"]


def test_estimate_gap_with_return_motion_is_not_stationary() -> None:
    t, pos = _line_trajectory()
    pos[:, 0] = np.maximum(0, 15 - np.abs(t - 50))
    result = _compute_input_coverage_diagnostics(
        t_ref=t, pos_ref=pos, t_est=t[(t <= 35) | (t >= 65)], offset_est_s=0,
    )
    assert "estimate_internal_gap_critical" in result["coverage_hard_reasons"]
    assert result["estimate_gap_motion_details"][0]["reference_position_extent_m"] == 15


def test_estimate_gap_without_reference_support_is_not_exempt() -> None:
    t, pos = _line_trajectory()
    pos[:] = 0
    keep = (t <= 35) | (t >= 65)
    result = _compute_input_coverage_diagnostics(
        t_ref=t[keep], pos_ref=pos[keep], t_est=t[keep], offset_est_s=0,
    )
    assert "estimate_internal_gap_critical" in result["coverage_hard_reasons"]


def test_stationary_gap_tolerates_small_position_noise() -> None:
    t, pos = _line_trajectory()
    pos[:, 0] = 0.01 * np.sin(t * 10)
    result = _compute_input_coverage_diagnostics(
        t_ref=t, pos_ref=pos, t_est=t[(t <= 35) | (t >= 65)], offset_est_s=0,
    )
    assert result["estimate_internal_gap_ratio"] == 0


def test_alignment_quality_is_scale_adaptive() -> None:
    labels = []
    normalized = []
    for scale in (1.0, 100.0):
        t, pos_ref = _line_trajectory(duration_s=20.0, scale=scale)
        error = 0.02 * (20.0 * scale)
        pos_est = pos_ref + np.array([0.0, error, 0.0])
        out = _compute_alignment_quality(
            t_ref=t,
            pos_ref=pos_ref,
            pos_step3=pos_est,
            raw_rmse_m=4.0 * error,
            step3_rmse_m=error,
            segment_duration_s=5.0,
            overlap_ratio=0.5,
            min_samples=20,
            good_rmse_m=0.5,
            partial_rmse_m=8.0,
            good_seg_cv=0.4,
            good_heading_p90_deg=60.0,
            partial_min_improve_pct=20.0,
            threshold_mode="adaptive",
        )
        labels.append(out["_quality_label"])
        normalized.append(out["step3_rmse_m"] / out["reference_characteristic_scale_m"])

    assert labels == ["partial_align", "partial_align"]
    assert np.isclose(normalized[0], normalized[1])


def test_time_warning_is_not_suppressed_by_good_final_alignment() -> None:
    out = _build_user_alert(
        time_metrics={
            "xcorr_peak_normalized": 0.60,
            "xcorr_psr": 3.0,
            "evo_matches_ratio_equivalent": 0.95,
            "omega_rmse_improve_pct": 90.0,
            "offset_est_s": 0.123,
            "reference_median_dt_s": 0.01,
            "estimate_median_dt_s": 0.01,
        },
        traj_metrics={
            "ate_rmse_step3_m": 0.02,
            "ate_rmse_improve_raw_to_step3_pct": 95.0,
        },
        quality_label="good_align",
        rigid_label="rigidly_alignable",
        rigid_reasons="",
        alignment_quality={"good_rmse_threshold_m": 0.1, "critical_rmse_threshold_m": 1.0},
    )

    assert out["_alert_level"] == "warning"
    assert "Weak time-alignment confidence" in out["_alert_reasons"]


def test_orientation_diagnosis_is_independent_of_translation_sr() -> None:
    pose_metrics = {
        "ape": {"step3": {"rotation_angle_deg": {"rmse": 50.0}}},
        "rpe": {"step3": {"rotation_angle_deg": {"rmse": 5.0}}},
        "rpe_time_1s": {"step3": {"rotation_angle_deg": {"rmse": 5.0}}},
        "valid_segment": {"step3": {"success": {"success_rate_distance": 0.4}}},
    }
    out = _compute_orientation_diagnostics(pose_metrics)

    assert out["orientation_status"] == "critical"
    assert out["orientation_unstable"] is True
    assert out["orientation_inconsistent_with_translation_sr"] is False


def test_failure_diagnosis_keeps_axes_separate() -> None:
    result = _build_failure_diagnosis(
        user_alert_level="warning",
        input_coverage={
            "coverage_status": "warning",
            "coverage_hard_reasons": [],
            "coverage_soft_reasons": ["temporal_coverage_low"],
        },
        alignment_quality={"step3_rmse_m": 0.1, "critical_rmse_threshold_m": 1.0},
        quality_label="good_align",
        rigid_label="rigidly_alignable",
        rigid_reasons="",
        case_diagnostics={"diagnosis_tags": ["time_alignment_weak"]},
        orientation={"orientation_status": "ok"},
        sr_reliability={"sr_reliability_status": "warning"},
    )

    assert result["failure_diagnosis_level"] == "warning"
    assert result["trajectory_quality"] == "good_align"
    assert result["trajectory_diagnosis_level"] == "warning"
    assert result["trajectory_diagnosis_soft_reasons"] == [
        "temporal_coverage_low",
    ]
    assert result["calibration_confidence"] == "low"
    assert result["calibration_confidence_reasons"] == ["time_alignment_weak"]
    assert result["coverage_reliability"] == "warning"


def test_time_confidence_warning_does_not_downgrade_trajectory_quality() -> None:
    result = _build_failure_diagnosis(
        user_alert_level="warning",
        input_coverage={
            "coverage_status": "ok",
            "coverage_hard_reasons": [],
            "coverage_soft_reasons": [],
        },
        alignment_quality={"step3_rmse_m": 0.02, "critical_rmse_threshold_m": 1.0},
        quality_label="good_align",
        rigid_label="rigidly_alignable",
        rigid_reasons="",
        case_diagnostics={"diagnosis_tags": ["time_alignment_weak"]},
        orientation={"orientation_status": "ok"},
        sr_reliability={"sr_reliability_status": "ok"},
    )

    assert result["failure_diagnosis_level"] == "warning"
    assert result["trajectory_diagnosis_level"] == "ok"
    assert result["trajectory_diagnosis_hard_reasons"] == []
    assert result["trajectory_diagnosis_soft_reasons"] == []
    assert result["calibration_confidence"] == "low"
    assert result["calibration_confidence_reasons"] == ["time_alignment_weak"]
