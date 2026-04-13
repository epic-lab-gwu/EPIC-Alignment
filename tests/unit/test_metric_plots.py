from pathlib import Path

import numpy as np

from vicon_ws.viz.metric_plots import generate_metric_plots


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

    assert len(produced) >= 11
    assert (tmp_path / "plots.md").exists()
