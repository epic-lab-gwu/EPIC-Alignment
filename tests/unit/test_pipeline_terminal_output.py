from __future__ import annotations

import numpy as np

from epa.core.pipeline_modular import _print_sim3_ov_eval_terminal_metrics


def test_sim3_terminal_output_includes_latex_table(capsys) -> None:
    stats = {"rmse": 1.2, "mean": 1.0, "min": 0.1, "max": 2.0, "std": 0.2, "median": 0.9}
    pose_metrics = {
        "ape": {"step3": {"rotation_angle_deg": stats, "translation_part": stats, "pair_count": 5}},
        "rpe": {"step3": {"rotation_angle_deg": stats, "translation_part": stats, "pair_count": 4}},
        "rpe_time_1s": {"step3": {"rotation_angle_deg": stats, "translation_part": stats, "pair_count": 3}},
        "valid_segment": {
            "step3": {
                "success": {
                    "success_rate_distance": 0.88,
                    "success_rate_time": 0.77,
                    "valid_distance_m": 8.8,
                    "total_distance_m": 10.0,
                    "threshold": {"threshold_m": 5.0, "mode": "fixed"},
                    "sr_reliability_status": "ok",
                }
            }
        },
        "eval_alignment": {
            "step3": {
                "align_pair_count": 5,
                "sim3_scale": 1.01,
                "sim3_reliable": True,
                "sim3_solver": "epa_sim3_v2",
            }
        },
        "rpe_config": {"delta": 1, "delta_unit": "f"},
    }

    _print_sim3_ov_eval_terminal_metrics(
        pose_metrics=pose_metrics,
        ape_pose_relation="translation_part",
        rpe_pose_relation="translation_part",
        eval_source="epa_eval_align",
        t_ref=np.array([0.0, 1.0]),
        pos_ref=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        mode_label="sim3",
    )

    out = capsys.readouterr().out
    assert "Absolute Trajectory Error" in out
    assert "Relative Pose Error" in out
    assert "SINGLE-RUN LATEX TABLE" in out
    assert r"\begin{tabular}{lrrrrrl}" in out
    assert r"sim3 & 1.200 & 1.200 & 1.200 & 1.200 & 88.00\% & true \\" in out
