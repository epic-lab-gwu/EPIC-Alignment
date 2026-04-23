import numpy as np

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
