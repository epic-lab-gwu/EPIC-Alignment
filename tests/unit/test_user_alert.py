import numpy as np

from epa.core.diagnostics import _diagnosis_tags_from_metrics
from epa.core.pipeline_modular import _build_user_alert


def test_user_alert_highlights_severe_scale_mismatch() -> None:
    out = _build_user_alert(
        time_metrics={
            "xcorr_peak_normalized": 0.95,
            "xcorr_psr": 12.0,
            "evo_matches_ratio_equivalent": 0.9,
            "omega_rmse_improve_pct": 20.0,
        },
        traj_metrics={
            "ate_rmse_step3_m": 500.0,
            "ate_rmse_improve_raw_to_step3_pct": 20.0,
        },
        quality_label="partial_align",
        rigid_label="not_rigidly_alignable",
        rigid_reasons="scale_mismatch_severe,sim3_gain",
    )

    assert out["_alert_level"] == "critical"
    assert "orders of magnitude in scale" in out["_alert_message"]
    assert "Severe scale mismatch" in out["_alert_reasons"]


def test_user_alert_ok_case_has_no_scale_warning() -> None:
    out = _build_user_alert(
        time_metrics={
            "xcorr_peak_normalized": 0.95,
            "xcorr_psr": 12.0,
            "evo_matches_ratio_equivalent": 0.95,
            "omega_rmse_improve_pct": 20.0,
        },
        traj_metrics={
            "ate_rmse_step3_m": 0.2,
            "ate_rmse_improve_raw_to_step3_pct": 60.0,
        },
        quality_label="good_align",
        rigid_label="rigidly_alignable",
        rigid_reasons="",
    )

    assert out["_alert_level"] == "ok"
    assert "Severe scale mismatch" not in out["_alert_reasons"]
    assert np.isclose(float(out["alert_count"]), 0.0)


def test_case_diagnosis_tags_classify_alignment_failure_modes() -> None:
    out = _diagnosis_tags_from_metrics(
        success={
            "success_rate_distance": 0.0,
            "case_status": "globally_unstable",
            "global_gate_failed": True,
            "global_gate_value_m": 120.0,
            "global_gate_m": 30.0,
        },
        time_metrics={"xcorr_peak_normalized": 0.4, "xcorr_psr": 3.0},
        traj_metrics={},
        step3_selection={"step3_stable_segment_used": 0.0, "step3_stable_solve_ratio": 0.05},
        alignment_quality={"_quality_label": "poor_align"},
        rigid_alignability={
            "_rigid_alignability_label": "not_rigidly_alignable",
            "_rigid_alignability_reasons": "global_local_ratio,scale_mismatch_severe",
            "segment_global_local_rmse_ratio": 8.0,
            "sim3_scale": 0.01,
        },
        orientation={"orientation_unstable": "False", "orientation_warning": ""},
    )

    assert out["diagnosis_primary"] == "time_alignment_weak"
    assert "step3_no_reliable_stable_segment" in out["diagnosis_tags"]
    assert "global_gate_too_large" in out["diagnosis_tags"]
    assert "trajectory_jump" in out["diagnosis_tags"]
    assert "scale_or_unit_suspect" in out["diagnosis_tags"]
    assert "gt_mapping_suspect" in out["diagnosis_tags"]
    assert "orientation_unstable" not in out["diagnosis_tags"]


def test_case_diagnosis_does_not_flag_stable_segment_for_valid_full_solve() -> None:
    out = _diagnosis_tags_from_metrics(
        success={
            "success_rate_distance": 1.0,
            "case_status": "valid_segment",
            "global_gate_failed": False,
            "global_gate_value_m": 0.1,
            "global_gate_m": 30.0,
        },
        time_metrics={"xcorr_peak_normalized": 0.98, "xcorr_psr": 12.0},
        traj_metrics={},
        step3_selection={"step3_stable_segment_used": 0.0, "step3_stable_solve_ratio": 1.0},
        alignment_quality={"_quality_label": "good_align"},
        rigid_alignability={
            "_rigid_alignability_label": "rigidly_alignable",
            "_rigid_alignability_reasons": "",
            "segment_global_local_rmse_ratio": 2.0,
            "sim3_scale": 1.0,
        },
        orientation={"orientation_unstable": False, "orientation_warning": ""},
    )

    assert out["diagnosis_tags"] == []
    assert out["diagnosis_primary"] == ""
