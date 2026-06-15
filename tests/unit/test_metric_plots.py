from pathlib import Path

import numpy as np

from epa.viz.metric_plots import generate_ape_stage_raw_plot, generate_metric_plots, generate_time_rpe_metric_plots
from epa.viz.interactive_html import write_interactive_run_html


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
    assert "ape_translation_part_raw_p95.png" in names
    assert "rpe_translation_part_raw_p95.png" in names
    assert "ape_translation_part_stats_core.png" in names
    assert "rpe_translation_part_stats_core.png" in names
    assert "ape_translation_part_box_p95.png" in names
    assert "rpe_translation_part_violin_p95.png" in names
    assert all(p.suffix == ".png" for p in produced)


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
    assert "rpe_time_1s_translation_part_raw.png" in names
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
        file_name="ape_translation_part_se3_raw.png",
    )
    assert out is not None
    assert out.exists()
    assert out.name == "ape_translation_part_se3_raw.png"


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
    assert "Plotly.newPlot('trajectory3d'" in text
    assert "trajectoryProgress" in text
    assert "trajectoryTraces(pct)" in text
    assert "Metrics Summary" in text
    assert "Metrics details" in text
    assert "metricSummary" in text
    assert "metricDetails" in text
    assert "renderMetrics" in text
    assert "scrollZoom:false" in text
    assert "dragmode:'pan'" in text
    assert "scatter3d" in text
    assert "scattergl" in text
    assert "epaInteractiveData" in text
    assert "NaN" not in text
