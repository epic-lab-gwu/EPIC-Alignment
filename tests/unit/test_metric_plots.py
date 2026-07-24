import json
import re
from pathlib import Path

import numpy as np

from epa.viz.interactive_html import write_interactive_run_html
from epa.viz.metric_plots import generate_ape_stage_raw_plot, generate_metric_plots, generate_time_rpe_metric_plots


def _stats(values):
    values = np.asarray(values, dtype=float)
    sq = values**2
    return {
        "rmse": float(np.sqrt(np.mean(sq))),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "sse": float(np.sum(sq)),
    }


def _stage_block(values):
    arr = np.asarray(values, dtype=float)
    return {
        "translation_part": _stats(arr),
        "rotation_angle_deg": _stats(arr),
        "_error_arrays": {
            "translation_part": arr,
            "rotation_angle_deg": arr,
        },
        "_x_axis": {
            "index": np.arange(arr.size, dtype=float),
            "seconds_from_start": np.arange(arr.size, dtype=float) * 0.1,
            "distances_from_start": np.arange(arr.size, dtype=float) * 0.2,
        },
    }


def _extract_interactive_payload(html: str) -> dict:
    match = re.search(r'<script id="epaInteractiveData" type="application/json">(.*?)</script>', html, re.S)
    assert match is not None
    return json.loads(match.group(1))


def test_generate_metric_plots(tmp_path: Path) -> None:
    payload = {
        "pose_metrics": {
            "ape": {
                "raw": _stage_block([1.0, 2.0, 3.0]),
                "step2": _stage_block([0.9, 1.8, 2.7]),
                "step3": _stage_block([0.5, 1.0, 1.5]),
            },
            "rpe": {
                "raw": _stage_block([0.4, 0.5, 0.6]),
                "step2": _stage_block([0.3, 0.4, 0.5]),
                "step3": _stage_block([0.2, 0.25, 0.3]),
            },
        }
    }

    produced = generate_metric_plots(
        metrics_payload=payload,
        out_dir=tmp_path,
        ape_relation="translation_part",
        rpe_relation="translation_part",
        x_dimension="seconds",
    )

    names = {p.name for p in produced}
    assert len(produced) >= 10
    assert "ape_translation_part_hist_p95.png" in names
    assert "ape_translation_part_hist_p99.png" in names
    assert "rpe_translation_part_hist_p95.png" in names
    assert "ape_translation_part_series_p95.png" in names
    assert "rpe_translation_part_series_p95.png" in names
    assert "ape_translation_part_raw_p95.png" not in names
    assert "rpe_translation_part_raw_p95.png" not in names
    assert "ape_translation_part_stats_core.png" in names
    assert "rpe_translation_part_stats_core.png" in names
    assert "ape_translation_part_box_p95.png" in names
    assert "rpe_translation_part_violin_p95.png" in names
    assert all(p.suffix == ".png" for p in produced)


def test_generate_metric_plots_defaults_to_step3(monkeypatch, tmp_path: Path) -> None:
    payload = {
        "pose_metrics": {
            "ape": {
                "raw": _stage_block([100.0, 120.0, 140.0]),
                "step2": _stage_block([40.0, 50.0, 60.0]),
                "step3": _stage_block([0.5, 1.0, 1.5]),
            },
            "rpe": {
                "raw": _stage_block([100.0, 120.0, 140.0]),
                "step2": _stage_block([40.0, 50.0, 60.0]),
                "step3": _stage_block([0.2, 0.25, 0.3]),
            },
        }
    }
    captured = []

    def fake_plot_raw(traces, *args, **kwargs):
        captured.append([label for label, _, _ in traces])
        Path(kwargs["out_path"]).write_bytes(b"png")

    monkeypatch.setattr("epa.viz.metric_plots._plot_raw", fake_plot_raw)

    generate_metric_plots(
        metrics_payload=payload,
        out_dir=tmp_path,
        ape_relation="translation_part",
        rpe_relation="translation_part",
        x_dimension="seconds",
    )

    assert captured
    assert all(labels == ["step3"] for labels in captured)


def test_generate_metric_plots_debug_can_include_all_stages(tmp_path: Path) -> None:
    payload = {
        "pose_metrics": {
            "ape": {
                "raw": _stage_block([1.0, 2.0, 3.0]),
                "step2": _stage_block([0.9, 1.8, 2.7]),
                "step3": _stage_block([0.5, 1.0, 1.5]),
            },
            "rpe": {
                "raw": _stage_block([0.4, 0.5, 0.6]),
                "step2": _stage_block([0.3, 0.4, 0.5]),
                "step3": _stage_block([0.2, 0.25, 0.3]),
            },
        }
    }

    produced = generate_metric_plots(
        metrics_payload=payload,
        out_dir=tmp_path,
        ape_relation="translation_part",
        rpe_relation="translation_part",
        x_dimension="seconds",
        stages=None,
        file_prefix="debug_",
    )

    names = {p.name for p in produced}
    assert "debug_ape_translation_part_raw.png" in names
    assert "debug_rpe_translation_part_stats_core.png" in names


def test_generate_time_rpe_metric_plots(tmp_path: Path) -> None:
    payload = {
        "pose_metrics": {
            "ape": {
                "raw": _stage_block([1.0, 2.0, 3.0]),
                "step2": _stage_block([0.9, 1.8, 2.7]),
                "step3": _stage_block([0.5, 1.0, 1.5]),
            },
            "rpe": {
                "raw": _stage_block([0.4, 0.5, 0.6]),
                "step2": _stage_block([0.3, 0.4, 0.5]),
                "step3": _stage_block([0.2, 0.25, 0.3]),
            },
            "rpe_time_1s": {
                "raw": _stage_block([0.7, 0.8, 0.9]),
                "step2": _stage_block([0.5, 0.6, 0.7]),
                "step3": _stage_block([0.2, 0.3, 0.4]),
            },
        }
    }
    produced = generate_time_rpe_metric_plots(
        metrics_payload=payload,
        out_dir=tmp_path,
        relation="translation_part",
        x_dimension="seconds",
    )

    names = {p.name for p in produced}
    assert "rpe_time_1s_translation_part_series.png" in names
    assert "rpe_time_1s_translation_part_raw.png" not in names
    assert "rpe_time_1s_translation_part_box.png" in names
    assert "rpe_time_1s_translation_part_stats.png" in names
    assert "rpe_time_1s_translation_part_hist.png" not in names
    assert "rpe_time_1s_translation_part_map.png" not in names
    assert "rpe_time_1s_translation_part_violin.png" not in names


def test_generate_ape_stage_raw_plot(tmp_path: Path) -> None:
    payload = {
        "pose_metrics": {
            "ape": {
                "raw": _stage_block([1.0, 2.0, 3.0]),
                "step2": _stage_block([0.9, 1.8, 2.7]),
                "step3": _stage_block([0.5, 1.0, 1.5]),
            },
            "rpe": {
                "raw": _stage_block([0.4, 0.5, 0.6]),
                "step2": _stage_block([0.3, 0.4, 0.5]),
                "step3": _stage_block([0.2, 0.25, 0.3]),
            },
        }
    }

    out = generate_ape_stage_raw_plot(
        metrics_payload=payload,
        out_dir=tmp_path,
        ape_relation="translation_part",
        stage="step3",
        x_dimension="seconds",
        file_name="ape_translation_part_step3_series.png",
    )
    assert out is not None
    assert out.exists()
    assert out.name == "ape_translation_part_step3_series.png"


def test_write_interactive_run_html_contains_plotly_payload(tmp_path: Path) -> None:
    payload = {
        "pose_metrics": {
            "ape": {"raw": _stage_block([1.0, 2.0]), "step2": _stage_block([0.8, 1.6]), "step3": _stage_block([0.2, 0.4])},
            "rpe": {"raw": _stage_block([0.5, 0.6]), "step2": _stage_block([0.4, 0.5]), "step3": _stage_block([0.1, 0.2])},
            "rpe_time_1s": {"raw": _stage_block([0.7, 0.8]), "step2": _stage_block([0.5, 0.6]), "step3": _stage_block([0.2, 0.3])},
        },
        "metadata": {"plot": {"ape_relation": "translation_part", "rpe_relation": "translation_part", "x_dimension": "seconds"}},
        "step3_selection": {"step3_alignment_mode": "standard", "step3_inlier_count": 2, "step3_rejected_count": 0},
    }
    pos_gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    out = write_interactive_run_html(
        tmp_path / "interactive_report.html",
        title="demo",
        metrics_payload=payload,
        pos_gt=pos_gt,
        pr_sync=pos_gt + 1.0,
        pr_corrected=pos_gt + 0.5,
        pr_final=pos_gt + 0.1,
        trajectory_views={
            "ov_sim3": {"label": "OV Sim3", "pos": pos_gt + 0.2, "meta": {"align_scale": 1.0}},
            "sim3": {
                "label": "Sim3",
                "pos": pos_gt + 0.4,
                "meta": {"align_scale": 1.0, "solver": "epa_sim3_v2", "anchor_samples": 2, "anchor_status": "ok", "confidence": "high"},
            },
        },
        trajectory_view_metrics={
            "step3": {"sr_distance": 0.5, "ape_trans_rmse_m": 1.2, "case_status": "step3_status"},
            "ov_sim3": {"sr_distance": 0.75, "ape_trans_rmse_m": 0.8, "case_status": "ov_status"},
            "sim3": {
                "sr_distance": 0.35,
                "ape_trans_rmse_m": 2.4,
                "case_status": "epa_status",
                "sim3_anchor_status": "ok",
                "sim3_confidence": "high",
                "sim3_consensus_count": 3,
                "sim3_candidate_reliable_count": 5,
                "sim3_reliable": True,
            },
        },
        time_alignment={
            "t_uniform": np.array([0.0, 1.0]),
            "sig_gt": np.array([1.0, 2.0]),
            "sig_est": np.array([1.1, 1.9]),
            "corr": np.array([0.2, 0.9]),
            "lags": np.array([-1, 0]),
            "dt_resample": 0.01,
            "calculated_offset": 0.0,
        },
    )
    text = out.read_text(encoding="utf-8")
    assert "renderPlot('trajectory3d'" in text
    assert "trajectoryProgress" in text
    assert "trajectoryPlay" in text
    assert "startPlayback" in text
    assert "trajectoryMode" in text
    assert "trajectoryViews" in text
    assert "trajectoryViewMetrics" in text
    assert "currentMetricSummary" in text
    assert "trajectoryTraces(pct)" in text
    assert "currentTrajectoryCamera" in text
    assert "trajectoryLayout(camera)" in text
    assert "uirevision:'trajectory-camera'" in text
    assert "Metrics Summary" in text
    assert "Linear Velocity" in text
    assert "Time Alignment Signals" not in text
    assert "Time Alignment Correlation" not in text
    assert "Metrics details" in text
    assert "metricSummary" in text
    assert "metricDetails" in text
    assert "renderMetrics" in text
    assert "setupDashboard" in text
    assert "resetLayout" in text
    assert "data-close-panel" in text
    assert "scrollZoom:true" in text
    assert "dragmode:'orbit'" in text
    assert "scatter3d" in text
    assert "scattergl" in text
    assert "epaInteractiveData" in text
    embedded = _extract_interactive_payload(text)
    assert "NaN" not in json.dumps(embedded)
    assert all([trace["stage"] for trace in metric["traces"]] == ["step3"] for metric in embedded["metrics"])
    assert "raw" not in embedded["trajectory"]
    assert "step2" not in embedded["trajectory"]
    assert "ov_sim3" in embedded["trajectory"]
    assert "sim3" in embedded["trajectory"]
    assert [embedded["trajectoryViews"][key]["label"] for key in ["step3", "ov_sim3", "sim3"]] == [
        "se3",
        "sim3",
        "sim3",
    ]
    assert embedded["trajectoryViewMetrics"]["ov_sim3"]["case_status"] == "ov_status"
    assert embedded["trajectoryViewMetrics"]["sim3"]["sim3_confidence"] == "high"
    assert embedded["speed"]["title"] == "Linear velocity"
    assert [trace["label"] for trace in embedded["speed"]["traces"]] == [
        "ground truth",
        "se3",
        "sim3",
    ]


def test_write_interactive_run_html_debug_keeps_metric_stages(tmp_path: Path) -> None:
    payload = {
        "pose_metrics": {
            "ape": {
                "raw": _stage_block([1.0, 2.0]),
                "step2": _stage_block([0.8, 1.6]),
                "step3": _stage_block([0.2, 0.4]),
            },
            "rpe": {
                "raw": _stage_block([0.5, 0.6]),
                "step2": _stage_block([0.4, 0.5]),
                "step3": _stage_block([0.1, 0.2]),
            },
            "rpe_time_1s": {
                "raw": _stage_block([0.7, 0.8]),
                "step2": _stage_block([0.5, 0.6]),
                "step3": _stage_block([0.2, 0.3]),
            },
        },
        "metadata": {
            "debug": True,
            "plot": {
                "ape_relation": "translation_part",
                "rpe_relation": "translation_part",
                "x_dimension": "seconds",
            },
        },
        "step3_selection": {"step3_alignment_mode": "standard", "step3_inlier_count": 2, "step3_rejected_count": 0},
    }
    pos_gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    out = write_interactive_run_html(
        tmp_path / "interactive_report.html",
        title="demo",
        metrics_payload=payload,
        pos_gt=pos_gt,
        pr_sync=pos_gt + 1.0,
        pr_corrected=pos_gt + 0.5,
        pr_final=pos_gt + 0.1,
    )

    embedded = _extract_interactive_payload(out.read_text(encoding="utf-8"))
    text = out.read_text(encoding="utf-8")
    assert "Time Alignment Signals" in text
    assert "Time Alignment Correlation" in text
    assert "raw" in embedded["trajectory"]
    assert "step2" in embedded["trajectory"]
    assert [trace["label"] for trace in embedded["speed"]["traces"]] == ["ground truth", "se3", "raw", "step2"]
    assert all(
        [trace["stage"] for trace in metric["traces"]] == ["raw", "step2", "step3"]
        for metric in embedded["metrics"]
    )


def test_write_interactive_run_html_defaults_to_10k_points(tmp_path: Path) -> None:
    n = 12050
    pos_gt = np.column_stack([np.arange(n, dtype=float), np.zeros(n), np.zeros(n)])
    payload = {
        "pose_metrics": {
            "ape": {"step3": _stage_block(np.linspace(0.0, 1.0, n))},
            "rpe": {"step3": _stage_block(np.linspace(0.0, 0.5, n))},
            "rpe_time_1s": {"step3": _stage_block(np.linspace(0.0, 0.25, n))},
        },
        "metadata": {
            "plot": {
                "ape_relation": "translation_part",
                "rpe_relation": "translation_part",
                "x_dimension": "seconds",
            },
        },
    }

    out = write_interactive_run_html(
        tmp_path / "interactive_report.html",
        title="demo",
        metrics_payload=payload,
        pos_gt=pos_gt,
        pr_sync=pos_gt,
        pr_corrected=pos_gt,
        pr_final=pos_gt,
        timestamps_s=np.arange(n, dtype=float) * 0.05,
    )

    embedded = _extract_interactive_payload(out.read_text(encoding="utf-8"))
    assert len(embedded["trajectory"]["step3"]["x"]) == 10000
    assert len(embedded["metrics"][0]["traces"][0]["x"]) == 10000
    assert len(embedded["speed"]["traces"][0]["x"]) == 10000
